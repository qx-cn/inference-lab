#!/usr/bin/env python3
"""
实验 07：DeepSeek MLA 矩阵吸收（Matrix Absorption）数学等价性证明与有限状态机Jump-Forward（快进解码）
对应计划：第 5 周 & 第 6 周（对齐 sglang forward_mla.py 与 outlines_jump_forward.py）
运行方式：先复制整个实验目录，再在副本目录运行 python3 lab07_mla_absorption_and_constrained_decoding.py（见 README.md）。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random


def verify_mla_matrix_absorption() -> None:
    """
    实验 07-A：对齐 sglang/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
    验证未吸收路径（解压历史所有位置的 K, V）与矩阵吸收路径（将 W^UK 吸收进查询侧 w_kc、
    将 W^UV 吸收进输出侧 w_vc，直接与压缩潜向量 c_t^KV 做 MQA 点积）的最大绝对误差 < 1e-12，
    并核算 DeepSeek-V3 解码单步（S=8192）投影浮点运算量缩减 8192 倍的账本。
    """
    random.seed(202607)
    seq_len = 16
    num_heads = 4
    kv_lora_rank = 6
    qk_nope_head_dim = 4
    qk_rope_head_dim = 2
    v_head_dim = 4
    hidden_size = 8
    scale = (qk_nope_head_dim + qk_rope_head_dim) ** -0.5

    # 当前解码步查询向量：q_nope [num_heads, d_nope], q_pe [num_heads, d_rope]
    q_nope = [[random.uniform(-0.8, 0.8) for _ in range(qk_nope_head_dim)] for _ in range(num_heads)]
    q_pe = [[random.uniform(-0.8, 0.8) for _ in range(qk_rope_head_dim)] for _ in range(num_heads)]

    # 历史 KV Cache：仅存压缩潜向量 c_kv [seq_len, kv_lora_rank] 与共享旋转位置键 k_pe [seq_len, d_rope]
    c_kv_cache = [[random.uniform(-0.8, 0.8) for _ in range(kv_lora_rank)] for _ in range(seq_len)]
    k_pe_cache = [[random.uniform(-0.8, 0.8) for _ in range(qk_rope_head_dim)] for _ in range(seq_len)]

    # 每个注意力头的上投影矩阵 W_UK [num_heads, kv_lora_rank, d_nope], W_UV [num_heads, kv_lora_rank, d_v]
    # 以及输出投影矩阵 W_O [num_heads, d_v, hidden_size]
    w_uk = [
        [[random.uniform(-0.5, 0.5) for _ in range(qk_nope_head_dim)] for _ in range(kv_lora_rank)]
        for _ in range(num_heads)
    ]
    w_uv = [
        [[random.uniform(-0.5, 0.5) for _ in range(v_head_dim)] for _ in range(kv_lora_rank)]
        for _ in range(num_heads)
    ]
    w_o = [
        [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] for _ in range(v_head_dim)]
        for _ in range(num_heads)
    ]

    # 1. 未吸收路径（forward_normal）：遍历历史所有位置 t 解压出每个头的 K_nope 与 V
    out_unabsorbed = [0.0] * hidden_size
    for h in range(num_heads):
        scores_h: list[float] = []
        v_decompressed_h: list[list[float]] = []
        for t in range(seq_len):
            # k_{t,h}^C = c_t^{KV} @ W_h^{UK}
            k_nope_th = [
                sum(c_kv_cache[t][r] * w_uk[h][r][d] for r in range(kv_lora_rank))
                for d in range(qk_nope_head_dim)
            ]
            # v_{t,h}^C = c_t^{KV} @ W_h^{UV}
            v_th = [
                sum(c_kv_cache[t][r] * w_uv[h][r][dv] for r in range(kv_lora_rank))
                for dv in range(v_head_dim)
            ]
            v_decompressed_h.append(v_th)
            dot_nope = sum(q_nope[h][d] * k_nope_th[d] for d in range(qk_nope_head_dim))
            dot_pe = sum(q_pe[h][dr] * k_pe_cache[t][dr] for dr in range(qk_rope_head_dim))
            scores_h.append((dot_nope + dot_pe) * scale)

        max_s = max(scores_h)
        exps = [math.exp(s - max_s) for s in scores_h]
        denom = sum(exps)
        probs = [e / denom for e in exps]

        u_h = [
            sum(probs[t] * v_decompressed_h[t][dv] for t in range(seq_len))
            for dv in range(v_head_dim)
        ]
        for dim_out in range(hidden_size):
            out_unabsorbed[dim_out] += sum(u_h[dv] * w_o[h][dv][dim_out] for dv in range(v_head_dim))

    # 2. 矩阵吸收路径（forward_absorb）：
    # 将 W_UK 吸收进查询侧 w_kc: q_absorb_h = q_nope_h @ (W_h^{UK})^T -> 维度变为 [kv_lora_rank]
    # 将 W_UV 吸收进输出侧 w_vc: 在潜空间完成加权求和 u_latent_h 后再乘以 W_h^{UV}
    out_absorbed = [0.0] * hidden_size
    for h in range(num_heads):
        q_absorb_h = [
            sum(q_nope[h][d] * w_uk[h][r][d] for d in range(qk_nope_head_dim))
            for r in range(kv_lora_rank)
        ]
        scores_abs_h: list[float] = []
        for t in range(seq_len):
            dot_latent = sum(q_absorb_h[r] * c_kv_cache[t][r] for r in range(kv_lora_rank))
            dot_pe = sum(q_pe[h][dr] * k_pe_cache[t][dr] for dr in range(qk_rope_head_dim))
            scores_abs_h.append((dot_latent + dot_pe) * scale)

        max_s = max(scores_abs_h)
        exps = [math.exp(s - max_s) for s in scores_abs_h]
        denom = sum(exps)
        probs = [e / denom for e in exps]

        # 在压缩潜向量空间直接做 MQA 加权归约
        u_latent_h = [
            sum(probs[t] * c_kv_cache[t][r] for t in range(seq_len))
            for r in range(kv_lora_rank)
        ]
        # 输出侧乘以 w_vc (即 W_h^{UV}) 还原到 v_head_dim
        u_h_abs = [
            sum(u_latent_h[r] * w_uv[h][r][dv] for r in range(kv_lora_rank))
            for dv in range(v_head_dim)
        ]
        for dim_out in range(hidden_size):
            out_absorbed[dim_out] += sum(u_h_abs[dv] * w_o[h][dv][dim_out] for dv in range(v_head_dim))

    max_abs_err = max(abs(a - b) for a, b in zip(out_unabsorbed, out_absorbed))
    assert max_abs_err < 1e-12

    # 3. 核算 DeepSeek-V3 解码单步（B=1, S=8192）投影浮点运算量（FLOPs）缩减账本
    ds_heads = 128
    ds_kv_lora_rank = 512
    ds_nope_dim = 128
    ds_v_dim = 128
    ds_seq_len = 8192

    flops_unabsorbed = 2 * ds_seq_len * ds_heads * ds_kv_lora_rank * (ds_nope_dim + ds_v_dim)
    flops_absorbed = 2 * 1 * ds_heads * ds_kv_lora_rank * (ds_nope_dim + ds_v_dim)
    reduction_ratio = flops_unabsorbed // flops_absorbed
    assert reduction_ratio == 8192

    print("=== [实验 07-A] DeepSeek MLA 矩阵吸收 (w_kc / w_vc) 数学等价性与算力账本核验 ===")
    print(f"  未吸收路径 vs 矩阵吸收路径最大绝对误差: {max_abs_err:.2e} (< 1e-12)")
    print(
        f"  DeepSeek-V3 解码单步 (S={ds_seq_len}) 历史 KV 解压投影算力: "
        f"{flops_unabsorbed / 1e9:.2f} GFLOPs"
    )
    print(
        f"  DeepSeek-V3 解码单步 (S={ds_seq_len}) 矩阵吸收后投影算力  : "
        f"{flops_absorbed / 1e6:.2f} MFLOPs (缩减倍数: {reduction_ratio}x)"
    )


@dataclass(frozen=True)
class ConstrainedStepLog:
    action: str
    state_from: int
    state_to: int
    emitted_tokens: list[str]


def verify_fsm_constrained_jump_forward() -> None:
    """
    实验 07-B：对齐 sglang/python/sglang/srt/constrained/outlines_jump_forward.py
    构造生成 '{"id":25,"ok":true}' 的确定性有限状态机（DFA），验证：
    1. 在出度为 1 的状态链上直接调用 try_jump_forward 免模型前向拼接token；
    2. 在分支状态通过 fill_vocab_mask 屏蔽非法高分 token；
    3. 将原本需要 12 步的模型自回归解码压缩至仅需 2 次模型前向传播。
    """
    vocab: list[str] = [
        "{",
        "\"",
        "id",
        ":",
        "2",
        "5",
        "9",
        "0",
        ",",
        "ok",
        ":true}",
        "true",
        "abc",
        "}",
    ]
    token_to_id = {tok: idx for idx, tok in enumerate(vocab)}

    # 定义 DFA 状态转移表: state -> {token_str: next_state}
    # 目标合法结构: {"id":<两位数字>,"ok":true}，共由 12 个 token组成
    dfa_transitions: dict[int, dict[str, int]] = {
        0: {"{": 1},
        1: {"\"": 2},
        2: {"id": 3},
        3: {"\"": 4},
        4: {":": 5},
        5: {"2": 6, "9": 6},        # 分支状态 1：首位数字可取 '2' 或 '9'
        6: {"5": 7, "0": 7},        # 分支状态 2：次位数字可取 '5' 或 '0'
        7: {",": 8},
        8: {"\"": 9},
        9: {"ok": 10},
        10: {"\"": 11},
        11: {":true}": 12},
    }
    final_state = 12

    def try_jump_forward(cur_state: int) -> tuple[list[str], int]:
        """当状态出度严格为 1 时，无需调用 GPU 模型前向，直接沿确定性边跳跃拼接。"""
        jumped_tokens: list[str] = []
        while cur_state in dfa_transitions and len(dfa_transitions[cur_state]) == 1:
            only_tok, next_state = next(iter(dfa_transitions[cur_state].items()))
            jumped_tokens.append(only_tok)
            cur_state = next_state
        return jumped_tokens, cur_state

    def fill_vocab_mask(cur_state: int) -> list[bool]:
        """生成当前 DFA 状态下的合法词表布尔掩码（True 表示允许，False 表示屏蔽为 -inf）。"""
        allowed_tokens = set(dfa_transitions.get(cur_state, {}).keys())
        return [tok in allowed_tokens for tok in vocab]

    # 模拟模型在分支状态 5 和状态 6 的原始 Logits：
    # 故意让非法 token（如 "abc"、"true"、"}"）拥有极高分数，验证掩码能否强制拦截非法 token
    raw_logits_by_state: dict[int, dict[str, float]] = {
        5: {"abc": 15.0, "true": 12.0, "2": 4.5, "9": 1.2, "}": 9.0},
        6: {"}": 18.0, "abc": 11.0, "5": 6.8, "0": 2.1, "true": 8.5},
    }

    state = 0
    generated_tokens: list[str] = []
    step_logs: list[ConstrainedStepLog] = []
    model_forward_count = 0

    while state != final_state:
        # 优先尝试Jump-Forward（Jump-Forward）
        jumped, next_state = try_jump_forward(state)
        if jumped:
            generated_tokens.extend(jumped)
            step_logs.append(
                ConstrainedStepLog(
                    action="Jump-Forward (0 次模型前向)",
                    state_from=state,
                    state_to=next_state,
                    emitted_tokens=jumped,
                )
            )
            state = next_state
            continue

        # 遇到出度 > 1 的分支状态：执行模型前向 + fill_vocab_mask 掩码采样
        model_forward_count += 1
        vocab_mask = fill_vocab_mask(state)
        raw_scores = raw_logits_by_state[state]
        masked_logits = [
            raw_scores.get(tok, 0.0) if vocab_mask[idx] else -float("inf")
            for idx, tok in enumerate(vocab)
        ]
        best_token_id = max(range(len(vocab)), key=lambda i: masked_logits[i])
        chosen_tok = vocab[best_token_id]
        assert token_to_id[chosen_tok] == best_token_id

        next_state = dfa_transitions[state][chosen_tok]
        generated_tokens.append(chosen_tok)
        step_logs.append(
            ConstrainedStepLog(
                action=f"Model-Forward #{model_forward_count} (掩码拦截非法高分 token)",
                state_from=state,
                state_to=next_state,
                emitted_tokens=[chosen_tok],
            )
        )
        state = next_state

    output_json = "".join(generated_tokens)
    traditional_steps = len(generated_tokens)
    assert output_json == '{"id":25,"ok":true}'
    assert traditional_steps == 12
    assert model_forward_count == 2

    print("\n=== [实验 07-B] 有限状态机词表掩码 (fill_vocab_mask) 与Jump-Forward 快进解码 (Jump-Forward) ===")
    for log in step_logs:
        joined_piece = "".join(log.emitted_tokens)
        print(
            f"  状态 {log.state_from:2d} -> {log.state_to:2d} | {log.action:34s} | "
            f"产出 {len(log.emitted_tokens):2d} 个 token: {log.emitted_tokens} ('{joined_piece}')"
        )
    print(
        f"  最终生成 JSON: {output_json} | 总 token 数: {traditional_steps} | "
        f"实际模型前向次数: {model_forward_count} (节省 {traditional_steps - model_forward_count} 次前向，压缩率 {(1 - model_forward_count / traditional_steps) * 100:.1f}%)"
    )


if __name__ == "__main__":
    verify_mla_matrix_absorption()
    verify_fsm_constrained_jump_forward()
    print("\n[PASS] 实验 07 全部断言核验通过。")

# ==============================================================================
# 思考与动手改代码挑战题（完成实验后请亲手验证）
# ==============================================================================
#
# 【挑战题 1：为什么旋转位置编码（RoPE）分支不能像 q_nope 一样被静态矩阵吸收？】
#   - 原理提示：在矩阵吸收路径中，无位置编码分支满足线性结合律：
#     q_h^N @ (c_t^{KV} @ W_h^{UK})^T = (q_h^N @ (W_h^{UK})^T) @ (c_t^{KV})^T，
#     中间不依赖历史位置下标 t，因此可以提前把 W_h^{UK} 吸收到当前查询向量上。
#     但如果对潜向量先上投影再施加位置相关的正交旋转矩阵 R(t)，由于查询位置 pos_q 与
#     历史键位置 t 的相对旋转矩阵 R(pos_q - t) 随每个历史位置 t 动态变化，
#     矩阵链 q_h @ R(pos_q - t) @ (W_h^{UKR})^T 中间夹着与 t 强相关的 R(pos_q - t)，
#     无法提前与独立于 t 的权重矩阵 W_h^{UKR} 静态合并。这正是 DeepSeek MLA 采用解耦 RoPE
#     （单独保留 64 维共享 k_t^R 缓存而不做潜空间吸收）的根本数学原因。
#   - 动手实验：尝试在 verify_mla_matrix_absorption() 中给 k_nope_th 乘以与位置 t 相关的缩放/旋转因子，
#     观察 out_unabsorbed 与静态吸收路径 out_absorbed 的误差如何瞬间突破 1e-12。
#
# 【挑战题 2：字节级 Jump-Forward 直接拼接字符串为什么可能触发重分词（Retokenization）边界偏移？】
#   - 原理提示：见 sglang/python/sglang/srt/constrained/outlines_jump_forward.py。
#     确定性有限状态机在字节级（Byte-level）沿单出度路径跳跃产出字符串（如 ',"ok":true}'）后，
#     如果直接按单字符 token塞回上下文，可能与 BPE 分词器的贪心合并规则不一致（例如前序已生成的数字 '5'
#     与跳跃拼接的 ',' 在某些词表里可能存在合并 token '5,'，或者 '":true}' 被切分为不同的 Token ID 组合）。
#     因此在真实引擎中，跳跃产出的字符串必须拼回已解码文本重新调用 tokenizer.encode()，并回退/对齐 KV Cache。
#   - 动手实验：在 verify_fsm_constrained_jump_forward() 的 vocab 中新增合并 token '":'，
#     对比“逐状态单边追加 Token”与“将跳跃字符串整体按最长前缀贪心重分词”后的 Token ID 序列差异。
