#!/usr/bin/env python3
"""
实验 05：FlashAttention 在线归约（Online Softmax / Tiled Attention）与解码阶段 Split-KV 合并算法验证
对应计划：第 5 周（对齐 zero-to-sglang Part 3 第 1 章注意力后端）
运行方式：先复制整个实验目录，再在副本目录运行 python3 lab05_online_softmax_and_split_kv.py（见 README.md）。
"""

from __future__ import annotations

import math
import random


def naive_attention_1d(q: list[float], keys: list[list[float]], values: list[list[float]], scale: float) -> list[float]:
    """标准三遍式 Attention：先算全量 QK^T 落盘，再求全局最大值，最后算指数和与加权 V。"""
    scores = [sum(q[d] * k[d] for d in range(len(q))) * scale for k in keys]
    max_s = max(scores)
    exps = [math.exp(s - max_s) for s in scores]
    denom = sum(exps)
    head_dim = len(values[0])
    return [sum((exps[i] / denom) * values[i][d] for i in range(len(keys))) for d in range(head_dim)]


def tiled_online_softmax_attention(
    q: list[float],
    keys: list[list[float]],
    values: list[list[float]],
    scale: float,
    tile_size: int = 4,
) -> tuple[list[float], float]:
    """
    FlashAttention 分块（Tiling）与 Online Softmax 单遍递推：
    每次仅加载 tile_size 个 KV 到片上 SRAM，边扫描边用缩放因子 exp(m_old - m_new) 修正历史分子与分母。
    同时返回对数求和指数 LSE = m + log(l)，供 Split-KV 跨线程块归约使用。
    """
    head_dim = len(q)
    m = -float("inf")
    l = 0.0
    acc_o = [0.0] * head_dim

    for start in range(0, len(keys), tile_size):
        k_tile = keys[start : start + tile_size]
        v_tile = values[start : start + tile_size]
        scores = [sum(q[d] * k[d] for d in range(head_dim)) * scale for k in k_tile]
        m_tile = max(scores)
        m_new = max(m, m_tile)

        correction = math.exp(m - m_new) if m != -float("inf") else 0.0
        p_tile = [math.exp(s - m_new) for s in scores]
        l_new = l * correction + sum(p_tile)

        for d in range(head_dim):
            tile_wv = sum(p_tile[i] * v_tile[i][d] for i in range(len(k_tile)))
            acc_o[d] = acc_o[d] * correction + tile_wv

        m = m_new
        l = l_new

    out = [val / l for val in acc_o]
    lse = m + math.log(l)
    return out, lse


def split_kv_decode_attention(
    q: list[float],
    keys: list[list[float]],
    values: list[list[float]],
    scale: float,
    num_splits: int = 4,
) -> list[float]:
    """
    Decode 阶段 Split-KV（FlashDecoding）：
    当 Query 长度为 1 而 KV 序列很长时，将 KV 序列切分为 num_splits 份交给不同 SM 并行计算局部 (O_i, LSE_i)，
    最后用 LSE 权重 w_i = exp(LSE_i - LSE_global) 无损合并。
    """
    chunk_size = (len(keys) + num_splits - 1) // num_splits
    partial_outputs: list[list[float]] = []
    partial_lses: list[float] = []

    for s in range(num_splits):
        k_chunk = keys[s * chunk_size : (s + 1) * chunk_size]
        v_chunk = values[s * chunk_size : (s + 1) * chunk_size]
        if not k_chunk:
            continue
        o_chunk, lse_chunk = tiled_online_softmax_attention(q, k_chunk, v_chunk, scale, tile_size=4)
        partial_outputs.append(o_chunk)
        partial_lses.append(lse_chunk)

    max_lse = max(partial_lses)
    weights = [math.exp(lse - max_lse) for lse in partial_lses]
    sum_weights = sum(weights)
    head_dim = len(q)
    return [
        sum((weights[i] / sum_weights) * partial_outputs[i][d] for i in range(len(partial_outputs)))
        for d in range(head_dim)
    ]


def run_experiment() -> None:
    random.seed(42)
    seq_len = 64
    head_dim = 16
    scale = head_dim ** -0.5

    q = [random.uniform(-1.0, 1.0) for _ in range(head_dim)]
    keys = [[random.uniform(-1.0, 1.0) for _ in range(head_dim)] for _ in range(seq_len)]
    values = [[random.uniform(-1.0, 1.0) for _ in range(head_dim)] for _ in range(seq_len)]

    out_naive = naive_attention_1d(q, keys, values, scale)
    out_tiled, _ = tiled_online_softmax_attention(q, keys, values, scale, tile_size=8)
    out_split_kv = split_kv_decode_attention(q, keys, values, scale, num_splits=4)

    diff_tiled = max(abs(a - b) for a, b in zip(out_naive, out_tiled))
    diff_split = max(abs(a - b) for a, b in zip(out_naive, out_split_kv))

    print("=== [实验 05-A] Online Softmax 分块注意力与 Decode Split-KV 归约精度验证 ===")
    print(f"  序列长度: {seq_len}, Head Dim: {head_dim}, 分块大小: 8, Split-KV 切分数: 4")
    print(f"  Tiled Online Softmax vs 标准 Attention 最大绝对误差: {diff_tiled:.2e}")
    print(f"  Split-KV (4 分片 LSE 归约) vs 标准 Attention 最大绝对误差: {diff_split:.2e}")
    assert diff_tiled < 1e-12
    assert diff_split < 1e-12


def verify_moe_align_block_size() -> None:
    """
    对齐 mini-sglang/python/minisgl/moe/fused.py 与 kernel/triton/fused_moe.py：
    复现 4 个 token、每个选 Top-3 专家（共 12 个路由项）、分配给 4 个专家、block_size = 4 时的
    moe_align_block_size 桶重排与padding 假下标（padding_token = 12）对齐填充，
    并核验 offs_token // top_k 找回原始token下标及掩码 offs_token < num_valid_tokens 的矩阵乘等价性。
    """
    num_tokens = 4
    top_k = 3
    num_experts = 4
    block_size = 4
    hidden_dim = 4
    out_dim = 3

    # 对齐 minisgl/moe/fused.py 文档字符串示例（转为 0-based 专家编号 0..3）
    topk_ids = [
        [1, 2, 3],
        [0, 1, 3],
        [0, 2, 3],
        [0, 1, 2],
    ]
    topk_weights = [
        [0.50, 0.30, 0.20],
        [0.60, 0.25, 0.15],
        [0.40, 0.35, 0.25],
        [0.55, 0.25, 0.20],
    ]

    flat_topk_ids = [e for row in topk_ids for e in row]
    flat_topk_weights = [w for row in topk_weights for w in row]
    num_valid_tokens = len(flat_topk_ids)
    padding_token_id = num_valid_tokens
    assert num_valid_tokens == 12 and padding_token_id == 12

    # 按专家分桶并用padding 假下标 padding_token_id=12 填充到 block_size=4 的整数倍
    sorted_token_ids: list[int] = []
    expert_ids: list[int] = []
    for expert_id in range(num_experts):
        bucket = [flat_idx for flat_idx, e in enumerate(flat_topk_ids) if e == expert_id]
        padded_len = ((len(bucket) + block_size - 1) // block_size) * block_size
        bucket_padded = bucket + [padding_token_id] * (padded_len - len(bucket))
        sorted_token_ids.extend(bucket_padded)
        for _ in range(padded_len // block_size):
            expert_ids.append(expert_id)

    assert len(sorted_token_ids) == 16
    assert len(expert_ids) == 4
    assert sorted_token_ids == [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12]
    assert expert_ids == [0, 1, 2, 3]

    # 核验 offs_token // top_k 能从展平路由下标精确还原原始token行号 (0..3)
    recovered_orig_tokens = [
        offs_token // top_k if offs_token < num_valid_tokens else -1
        for offs_token in sorted_token_ids
    ]
    assert recovered_orig_tokens == [1, 2, 3, -1, 0, 1, 3, -1, 0, 2, 3, -1, 0, 1, 2, -1]

    # 构造输入特征 A [num_tokens, hidden_dim] 与专家权重 B [num_experts, hidden_dim, out_dim]
    random.seed(105)
    mat_a = [[random.uniform(-1.0, 1.0) for _ in range(hidden_dim)] for _ in range(num_tokens)]
    mat_b = [
        [[random.uniform(-1.0, 1.0) for _ in range(out_dim)] for _ in range(hidden_dim)]
        for _ in range(num_experts)
    ]

    # 1. 朴素逐 token路由计算基准
    ref_out = [[0.0] * out_dim for _ in range(num_tokens)]
    for tok_idx in range(num_tokens):
        for k_idx in range(top_k):
            exp_id = topk_ids[tok_idx][k_idx]
            weight = topk_weights[tok_idx][k_idx]
            for n in range(out_dim):
                dot_val = sum(mat_a[tok_idx][d] * mat_b[exp_id][d][n] for d in range(hidden_dim))
                ref_out[tok_idx][n] += weight * dot_val

    # 2. 模拟 Triton fused_moe_kernel + moe_sum_reduce_kernel
    # 输出缓存 C 形状为 [num_valid_tokens, out_dim]（即 [num_tokens * top_k, out_dim]）
    c_cache = [[0.0] * out_dim for _ in range(num_valid_tokens)]
    for pid_m, exp_id in enumerate(expert_ids):
        offs_token_block = sorted_token_ids[pid_m * block_size : (pid_m + 1) * block_size]
        for row_in_block, offs_token in enumerate(offs_token_block):
            token_mask = offs_token < num_valid_tokens
            # 对齐 fused_moe.py: a_ptrs 使用 offs_token // top_k 索引原始token特征，并用 token_mask 屏蔽padding 假下标 12
            a_row = mat_a[offs_token // top_k] if token_mask else [0.0] * hidden_dim
            moe_weight = flat_topk_weights[offs_token] if token_mask else 0.0
            for n in range(out_dim):
                acc = sum(a_row[d] * mat_b[exp_id][d][n] for d in range(hidden_dim))
                acc *= moe_weight
                if token_mask:
                    c_cache[offs_token][n] = acc

    # 对齐 moe_sum_reduce_kernel: 沿 top_k 维度归约每个 token的专家输出
    fused_out = [[0.0] * out_dim for _ in range(num_tokens)]
    for tok_idx in range(num_tokens):
        for n in range(out_dim):
            fused_out[tok_idx][n] = sum(c_cache[tok_idx * top_k + k_idx][n] for k_idx in range(top_k))

    max_moe_diff = max(
        abs(ref_out[t][n] - fused_out[t][n])
        for t in range(num_tokens)
        for n in range(out_dim)
    )
    assert max_moe_diff < 1e-12

    print("\n=== [实验 05-B] Fused MoE 块对齐填充 (moe_align_block_size) 与掩码矩阵乘核验 ===")
    print(f"  展平路由项数 num_valid_tokens={num_valid_tokens}, padding 假下标编号 padding_token={padding_token_id}, block_size={block_size}")
    print(f"  填充重排后 sorted_token_ids (len={len(sorted_token_ids)}): {sorted_token_ids}")
    print(f"  块级专家映射 expert_ids     (len={len(expert_ids)}): {expert_ids}")
    print(f"  offs_token // top_k 还原原始token行号: {recovered_orig_tokens}")
    print(f"  Fused MoE 块矩阵乘 vs 朴素逐 token路由最大绝对误差: {max_moe_diff:.2e}")


if __name__ == "__main__":
    run_experiment()
    verify_moe_align_block_size()
    print("\n[PASS] 实验 05 全部断言核验通过。")
