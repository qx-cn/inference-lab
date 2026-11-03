#!/usr/bin/env python3
"""
实验 01：现代大模型前向形状、KV Cache 显存账本与 Roofline 算术强度定量分析
对应计划：第 0 周 & 第 1 周
运行方式：先复制整个实验目录，再在副本目录运行 python3 lab01_roofline_and_kvcache_calc.py（见 README.md）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    num_layers: int
    hidden_size: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    param_billions: float
    mla_kv_lora_rank: int = 0
    mla_qk_rope_head_dim: int = 0


@dataclass(frozen=True)
class GpuSpec:
    name: str
    vram_gb: float
    fp16_tflops: float
    hbm_bw_gb_s: float

    @property
    def ridge_point_ops_per_byte(self) -> float:
        return (self.fp16_tflops * 1e12) / (self.hbm_bw_gb_s * 1e9)


def kv_cache_bytes_per_token(model: ModelSpec, dtype_bytes: int = 2) -> int:
    """计算单个 token（Token）在全模型所有层的键值缓存（KV Cache）字节数。"""
    if model.mla_kv_lora_rank > 0:
        per_layer_elements = model.mla_kv_lora_rank + model.mla_qk_rope_head_dim
        return model.num_layers * per_layer_elements * dtype_bytes
    return 2 * model.num_layers * model.num_kv_heads * model.head_dim * dtype_bytes


def verify_qwen3_shapes() -> None:
    """核验第 0 周必答题 1 & 2：GQA 权重维度与张量形状流转。"""
    total_tokens = 512
    num_seqs = 4
    hidden_size = 4096
    num_q_heads = 32
    num_kv_heads = 4
    head_dim = 128
    intermediate_size = 12288
    vocab_size = 151936

    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    qkv_out_dim = q_size + 2 * kv_size
    assert q_size == 4096
    assert kv_size == 512
    assert qkv_out_dim == 5120

    shapes = {
        "input_ids": (total_tokens,),
        "embed_tokens_out": (total_tokens, hidden_size),
        "qkv_proj_out": (total_tokens, qkv_out_dim),
        "q_after_split_and_rope": (total_tokens, num_q_heads, head_dim),
        "kv_after_split_and_rope": (total_tokens, num_kv_heads, head_dim),
        "attn_out": (total_tokens, num_q_heads, head_dim),
        "o_proj_out": (total_tokens, hidden_size),
        "gate_up_proj_out": (total_tokens, intermediate_size * 2),
        "act_fn_silu_and_mul_out": (total_tokens, intermediate_size),
        "down_proj_out": (total_tokens, hidden_size),
        "lm_head_prefill_sliced_in": (num_seqs, hidden_size),
        "lm_head_logits_out": (num_seqs, vocab_size),
    }

    print("=== [实验 01-A] 第 0 周：Qwen3 GQA 投影维度与前向张量形状核验 ===")
    print(f"q_proj 权重维度: ({q_size}, {hidden_size}) | k_proj / v_proj 权重维度: ({kv_size}, {hidden_size})")
    print(f"合并 QKVParallelLinear 输出特征维: {qkv_out_dim}")
    for step_name, shape in shapes.items():
        print(f"  {step_name:28s} -> {shape}")
    print()


def verify_kvcache_and_vram_budget() -> None:
    """核验第 1 周必答题 3 & 动手产出：单卡显存容量账本与并发上限。"""
    qwen3_0_6b = ModelSpec(
        name="Qwen3-0.6B (GQA)",
        num_layers=28,
        hidden_size=1024,
        num_q_heads=16,
        num_kv_heads=8,
        head_dim=128,
        intermediate_size=3072,
        vocab_size=151936,
        param_billions=0.6,
    )
    llama3_8b = ModelSpec(
        name="Llama-3-8B (GQA)",
        num_layers=32,
        hidden_size=4096,
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
        intermediate_size=14336,
        vocab_size=128256,
        param_billions=8.0,
    )
    deepseek_v3_mla = ModelSpec(
        name="DeepSeek-V3 (MLA 61L)",
        num_layers=61,
        hidden_size=7168,
        num_q_heads=128,
        num_kv_heads=1,
        head_dim=128,
        intermediate_size=18432,
        vocab_size=129280,
        param_billions=671.0,
        mla_kv_lora_rank=512,
        mla_qk_rope_head_dim=64,
    )

    llama_per_token = kv_cache_bytes_per_token(llama3_8b, dtype_bytes=2)
    assert llama_per_token == 131072, f"Expected 131072 bytes (128 KB), got {llama_per_token}"

    batch_32_seq_4096_gb = (32 * 4096 * llama_per_token) / (1024 ** 3)
    assert abs(batch_32_seq_4096_gb - 16.0) < 1e-6

    print("=== [实验 01-B] 第 1 周：各架构单 Token KV Cache 字节数与 24GB 显存容量账本 ===")
    vram_total_bytes = 24 * (1024 ** 3)
    gpu_util = 0.90
    activation_reserve_bytes = int(1.2 * (1024 ** 3))

    for model in (qwen3_0_6b, llama3_8b, deepseek_v3_mla):
        per_tok_bytes = kv_cache_bytes_per_token(model, dtype_bytes=2)
        weight_bytes = int(model.param_billions * 2 * (1024 ** 3))
        if weight_bytes < vram_total_bytes * gpu_util:
            avail_kv_bytes = int(vram_total_bytes * gpu_util) - weight_bytes - activation_reserve_bytes
            max_tokens = max(0, avail_kv_bytes // per_tok_bytes)
            max_4k_seqs = max_tokens // 4096
            cap_str = f"可缓存 {max_tokens:,d} Tokens (约 {max_4k_seqs} 条 4K 序列)"
        else:
            cap_str = "单卡 24GB 无法装下全量 FP16 权重，需多卡并行或量化"
        print(
            f"  {model.name:22s}: {per_tok_bytes / 1024:6.2f} KB/token | "
            f"32x4K 并发占 {32 * 4096 * per_tok_bytes / (1024**3):5.2f} GB | {cap_str}"
        )
    print()


def verify_roofline_arithmetic_intensity() -> None:
    """核验第 1 周必答题 2：线性层在不同总 Token 数 N 下的算术强度与 Roofline 拐点。"""
    rtx4090 = GpuSpec(name="RTX 4090", vram_gb=24.0, fp16_tflops=330.0, hbm_bw_gb_s=1008.0)
    h200 = GpuSpec(name="NVIDIA H200", vram_gb=141.0, fp16_tflops=989.0, hbm_bw_gb_s=4800.0)

    k_dim = 4096
    n_dim = 4096
    dtype_bytes = 2

    print("=== [实验 01-C] 第 1 周：Roofline 算术强度推导 (Linear 4096x4096, FP16) ===")
    print(
        f"  RTX 4090 算力拐点: {rtx4090.ridge_point_ops_per_byte:.1f} FLOPs/Byte | "
        f"H200 算力拐点: {h200.ridge_point_ops_per_byte:.1f} FLOPs/Byte"
    )
    for num_tokens in (1, 8, 32, 128, 256, 1024):
        flops = 2 * num_tokens * k_dim * n_dim
        bytes_accessed = dtype_bytes * (k_dim * n_dim + num_tokens * k_dim + num_tokens * n_dim)
        ai = flops / bytes_accessed
        bound_h200 = "Compute-bound (compute-bound（计算受限）)" if ai >= h200.ridge_point_ops_per_byte else "Memory-bound (带宽受限)"
        if num_tokens == 1:
            assert 0.99 < ai < 1.0
        print(f"  Batch Tokens = {num_tokens:4d} -> 算术强度 = {ai:6.1f} FLOPs/Byte -> [H200: {bound_h200}]")
    print()


@dataclass
class IncrementalDecodeState:
    """对齐 mini-sglang/python/minisgl/tokenizer/detokenize.py 的三指针增量反分词状态。"""

    decoded_ids: list[int]
    decoded_str: str = ""
    read_offset: int = 0
    surr_offset: int = 0
    sent_offset: int = 0


def verify_strided_qkv_view_and_utf8_detokenizer() -> None:
    """
    核验第 0 周 & 第 1 周底层内存步幅与字节级流式反分词机制：
    1. 模拟 QKVParallelLinear 合并输出 [N, 5120] 切片出 q(4096)、k(512)、v(512) 视图后，
       验证为什么 k.stride(0) 等于 5120 而非 512，并核验 store_kvcache_kernel 的物理偏移公式；
    2. 模拟字节级 BPE 将 3 字节中文字符 '智' (0xE6 0x99 0xBA) 拆成两个不完整字节 token时，
       对比逐 token直接解码产生替换符 '\\ufffd' 与三指针增量反分词缓冲的无损输出。
    """
    print("=== [实验 01-D] 合并 QKV 视图跨步访存 (Strided View) 与三指针 UTF-8 增量反分词核验 ===")

    # 1. 模拟合并 QKVParallelLinear 底层一维连续显存 [N, 5120]
    num_tokens = 4
    q_dim = 4096
    kv_dim = 512
    qkv_dim = q_dim + 2 * kv_dim
    assert qkv_dim == 5120

    # 为每个物理位置写入唯一标识值: token_idx * 10000 + col_idx
    flat_qkv_storage = [
        tok_idx * 10000 + col_idx
        for tok_idx in range(num_tokens)
        for col_idx in range(qkv_dim)
    ]

    # 切片视图 k 位于每行偏移 [4096 : 4608]，逻辑形状为 [num_tokens, kv_dim]，但底层行步幅仍为 5120
    k_base_offset = q_dim
    key_stride = qkv_dim
    assert key_stride == 5120 and key_stride != kv_dim

    # 模拟 nano-vllm store_kvcache_kernel: key_offsets = idx * key_stride + tl.arange(0, D)
    slot_mapping = [2, 0, 3, 1]
    k_cache_storage = [-1] * (num_tokens * kv_dim)
    for idx in range(num_tokens):
        slot = slot_mapping[idx]
        for d in range(kv_dim):
            src_phys_offset = k_base_offset + idx * key_stride + d
            dst_cache_offset = slot * kv_dim + d
            k_cache_storage[dst_cache_offset] = flat_qkv_storage[src_phys_offset]

    # 核验第 idx=1 个 token的 K 向量首尾元素是否正确落入物理槽位 slot=0
    assert k_cache_storage[0 * kv_dim + 0] == 1 * 10000 + 4096
    assert k_cache_storage[0 * kv_dim + (kv_dim - 1)] == 1 * 10000 + 4607
    # 若错误地将 stride(0) 当成连续张量的 512，则 idx=1 会读到偏移 4096 + 512 = 4608 (即token 0 的 V 向量！)
    wrong_offset_idx1 = k_base_offset + 1 * kv_dim + 0
    assert flat_qkv_storage[wrong_offset_idx1] == 0 * 10000 + 4608
    print(
        f"  [核验 1] QKV 合并张量 [N={num_tokens}, 5120] 切片后 k.stride(0)={key_stride} (而非 {kv_dim}): "
        f"idx=1 真实 K 首元素偏移={k_base_offset + key_stride}，误用紧凑步幅会误读偏移={wrong_offset_idx1} (token 0 的 V 起点)"
    )

    # 2. 模拟字节级 BPE 词表与“智” ('\xe6\x99\xba') 跨token拆分流式解码
    token_byte_table: dict[int, bytes] = {
        101: b"AI",
        102: b"\xe6\x99",  # '智' 的前 2 个字节（不完整 UTF-8 序列）
        103: b"\xba",      # '智' 的最后 1 个字节
        104: b"!",
    }
    stream_token_ids = [101, 102, 103, 104]

    # (a) 朴素逐 token独立解码：遇到不完整 UTF-8 字节流直接产出替换符 '\ufffd' ('')
    naive_chunks = [token_byte_table[tid].decode("utf-8", errors="replace") for tid in stream_token_ids]
    naive_joined = "".join(naive_chunks)
    assert "\ufffd" in naive_joined

    # (b) 对齐 mini-sglang DetokenizeManager 的三指针增量反分词
    state = IncrementalDecodeState(decoded_ids=[])
    incremental_chunks: list[str] = []

    def decode_token_slice(ids: list[int]) -> str:
        raw = b"".join(token_byte_table[t] for t in ids)
        return raw.decode("utf-8", errors="replace")

    for tid in stream_token_ids:
        state.decoded_ids.append(tid)
        read_str = decode_token_slice(state.decoded_ids[state.surr_offset :])
        surr_str = decode_token_slice(state.decoded_ids[state.surr_offset : state.read_offset])
        new_text = read_str[len(surr_str) :]

        if len(new_text) > 0 and not new_text.endswith("\ufffd"):
            output_str = state.decoded_str + new_text
            state.decoded_str = output_str
            state.surr_offset = state.read_offset
            state.read_offset = len(state.decoded_ids)
        else:
            # 末尾存在半个多字节字符产生的替换符 '\ufffd'，暂不推进 read_offset，等待后续字节 token拼齐
            output_str = state.decoded_str

        incremental_out = output_str[state.sent_offset :]
        state.sent_offset = len(output_str)
        incremental_chunks.append(incremental_out)

    final_text = "".join(incremental_chunks)
    assert "\ufffd" not in final_text
    assert incremental_chunks == ["AI", "", "智", "!"]
    assert final_text == "AI智!"
    print(
        f"  [核验 2] 字节级 BPE 流式反分词: 朴素逐 token解码={naive_chunks} (含替换符 \\ufffd) | "
        f"三指针增量缓冲输出={incremental_chunks} -> 拼接结果='{final_text}'"
    )


if __name__ == "__main__":
    verify_qwen3_shapes()
    verify_kvcache_and_vram_budget()
    verify_roofline_arithmetic_intensity()
    verify_strided_qkv_view_and_utf8_detokenizer()
    print("\n[PASS] 实验 01 全部断言核验通过。")
