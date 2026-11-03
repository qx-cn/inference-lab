#!/usr/bin/env python3
"""
实验 08：分块预填充（Chunked Prefill）尾延迟消融与强化学习混部虚拟内存睡眠/唤醒及张量桶（FlattenedTensorBucket）热更新
对应计划：第 4 周 & 第 7 周（对齐 sglang weight_updater.py 与 weight_sync/tensor_bucket.py）
运行方式：先复制整个实验目录，再在副本目录运行 python3 lab08_chunked_prefill_and_rl_weight_refit.py（见 README.md）。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct


def percentile(sorted_vals: list[float], p: float) -> float:
    """计算已排序列表的百分位数。"""
    assert sorted_vals
    idx = min(len(sorted_vals) - 1, max(0, math.ceil((p / 100.0) * len(sorted_vals)) - 1))
    return sorted_vals[idx]


def verify_chunked_prefill_tail_latency() -> None:
    """
    实验 08-A：在线服务突发 32K 长请求尾延迟消融实验
    8 条在线解码短请求流式生成期间突发 1 条 32,768 token长请求，
    对比不开启分块预填充（ITL P99 飙升至 244ms）与开启分块预填充（chunk_size = 2048，ITL P99 降至 18ms）
    的TTFT（首 token 延迟）与token 间延迟（ITL）分布。
    """
    num_online_decode_reqs = 8
    total_decode_steps = 20
    burst_prefill_tokens = 32768
    chunk_size = 2048

    # 硬件耗时模型：单步 8 条短请求纯 Decode 耗时 6.0ms；
    # 32K 全量 Prefill 独占单步耗时 244.0ms；2048 token分块 Prefill 耗时 12.0ms（与 8 条 Decode 混合组批共 18.0ms）
    pure_decode_step_ms = 6.0
    full_32k_prefill_ms = 244.0
    chunk_2k_prefill_ms = 12.0
    hybrid_step_ms = pure_decode_step_ms + chunk_2k_prefill_ms
    assert hybrid_step_ms == 18.0

    # 1. 不开启分块预填充（No Chunked Prefill）：
    # 第 5 步突发 32K 长请求，调度器被单次全量 Prefill 阻塞 244ms，期间 8 条在线请求全部停顿
    itl_no_chunk: list[float] = []
    for step in range(total_decode_steps):
        step_latency = full_32k_prefill_ms if step == 4 else pure_decode_step_ms
        for _ in range(num_online_decode_reqs):
            itl_no_chunk.append(step_latency)
    ttft_32k_no_chunk_ms = full_32k_prefill_ms

    # 2. 开启分块预填充（Chunked Prefill, chunk_size = 2048）：
    # 32,768 token被切分为 16 个 2048 分块，在步 4 ~ 步 19 与 8 条在线解码请求混合组批（Piggyback Decode）
    num_chunks = burst_prefill_tokens // chunk_size
    assert num_chunks == 16
    itl_chunked: list[float] = []
    for step in range(total_decode_steps):
        in_chunk_window = 4 <= step < 4 + num_chunks
        step_latency = hybrid_step_ms if in_chunk_window else pure_decode_step_ms
        for _ in range(num_online_decode_reqs):
            itl_chunked.append(step_latency)
    ttft_32k_chunked_ms = num_chunks * hybrid_step_ms

    sorted_no_chunk = sorted(itl_no_chunk)
    sorted_chunked = sorted(itl_chunked)

    p50_no_chunk = percentile(sorted_no_chunk, 50)
    p99_no_chunk = percentile(sorted_no_chunk, 99)
    p50_chunked = percentile(sorted_chunked, 50)
    p99_chunked = percentile(sorted_chunked, 99)

    assert p99_no_chunk == 244.0
    assert p99_chunked == 18.0

    print("=== [实验 08-A] 突发 32K 长请求：无分块预填充 vs 分块预填充 (chunk_size=2048) 尾延迟消融 ===")
    print(
        f"  [无分块预填充] 在线解码 ITL P50={p50_no_chunk:5.1f} ms | "
        f"ITL P99={p99_no_chunk:5.1f} ms (长请求阻塞导致世代停顿) | 32K TTFT={ttft_32k_no_chunk_ms:5.1f} ms"
    )
    print(
        f"  [分块预填充]   在线解码 ITL P50={p50_chunked:5.1f} ms | "
        f"ITL P99={p99_chunked:5.1f} ms (尾延迟降低 {p99_no_chunk / p99_chunked:.1f}x)      | 32K TTFT={ttft_32k_chunked_ms:5.1f} ms"
    )


@dataclass
class VirtualMemoryRegion:
    """模拟 torch_memory_saver 管理的虚拟地址与物理显存页解耦区域。"""

    tag: str
    virtual_ptr: int
    size_gb: float
    physical_mapped: bool = True
    payload: bytearray | None = None


@dataclass(frozen=True)
class FlattenedTensorMetadata:
    """对齐 sglang/python/sglang/srt/weight_sync/tensor_bucket.py 的张量桶元数据。"""

    name: str
    shape: tuple[int, ...]
    dtype: str
    start_idx: int
    end_idx: int
    numel: int


class FlattenedTensorBucket:
    """
    对齐 sglang/python/sglang/srt/weight_sync/tensor_bucket.py：
    将多个模型参数张量展平拼接为单一连续字节流（uint8 视图），支持单次通信广播与零误差解包重构。
    """

    DTYPE_ITEMSIZE: dict[str, int] = {"float32": 4, "bfloat16": 2, "uint8": 1}

    def __init__(
        self,
        named_tensors: list[tuple[str, tuple[int, ...], str, list[float]]] | None = None,
        flattened_bytes: bytes | None = None,
        metadata: list[FlattenedTensorMetadata] | None = None,
    ) -> None:
        if named_tensors is not None:
            if not named_tensors:
                raise ValueError("Cannot create empty tensor bucket")
            self.metadata: list[FlattenedTensorMetadata] = []
            buf = bytearray()
            current_idx = 0
            for name, shape, dtype, values in named_tensors:
                itemsize = self.DTYPE_ITEMSIZE[dtype]
                if current_idx % itemsize != 0:
                    raise ValueError(f"Unaligned tensor {name}: byte offset {current_idx}, itemsize {itemsize}")
                raw = struct.pack(f"<{len(values)}f", *values) if dtype == "float32" else bytes(len(values) * itemsize)
                numel_bytes = len(raw)
                self.metadata.append(
                    FlattenedTensorMetadata(
                        name=name,
                        shape=shape,
                        dtype=dtype,
                        start_idx=current_idx,
                        end_idx=current_idx + numel_bytes,
                        numel=numel_bytes,
                    )
                )
                buf.extend(raw)
                current_idx += numel_bytes
            self.flattened_bytes = bytes(buf)
        else:
            if flattened_bytes is None or metadata is None:
                raise ValueError("Must provide either named_tensors or both flattened_bytes and metadata")
            self.flattened_bytes = flattened_bytes
            self.metadata = metadata

    def reconstruct_tensors(self) -> list[tuple[str, tuple[int, ...], list[float]]]:
        reconstructed: list[tuple[str, tuple[int, ...], list[float]]] = []
        for meta in self.metadata:
            if meta.start_idx % self.DTYPE_ITEMSIZE[meta.dtype] != 0:
                raise ValueError(f"Unaligned received tensor {meta.name}: offset {meta.start_idx}")
            chunk = self.flattened_bytes[meta.start_idx : meta.end_idx]
            count = meta.numel // 4
            vals = list(struct.unpack(f"<{count}f", chunk))
            reconstructed.append((meta.name, meta.shape, vals))
        return reconstructed


class ColocatedRLMemoryController:
    """
    对齐 sglang/python/sglang/srt/managers/scheduler_components/weight_updater.py
    模拟 release_memory_occupation 与两阶段 resume_memory_occupation。
    """

    def __init__(self) -> None:
        self.regions: dict[str, VirtualMemoryRegion] = {
            "weights": VirtualMemoryRegion("weights", virtual_ptr=0x7F00_0000_0000, size_gb=16.0),
            "kv_cache": VirtualMemoryRegion("kv_cache", virtual_ptr=0x7F04_0000_0000, size_gb=24.0),
            "cuda_graph": VirtualMemoryRegion("cuda_graph", virtual_ptr=0x7F0A_0000_0000, size_gb=1.5),
        }
        self.offload_tags: set[str] = set()
        self.radix_cache_entries: int = 128

    def active_physical_gb(self) -> float:
        return sum(r.size_gb for r in self.regions.values() if r.physical_mapped)

    def flush_cache(self) -> None:
        self.radix_cache_entries = 0

    def release_memory_occupation(self, tags: list[str]) -> None:
        for tag in tags:
            self.offload_tags.add(tag)
            self.regions[tag].physical_mapped = False
            self.regions[tag].payload = None
            if tag == "kv_cache":
                self.flush_cache()

    def resume_memory_occupation(self, tags: list[str]) -> None:
        for tag in tags:
            self.offload_tags.discard(tag)
            self.regions[tag].physical_mapped = True


def verify_rl_sleep_wake_and_tensor_bucket() -> None:
    """
    实验 08-B：验证 RL 混部虚拟内存睡眠/唤醒（virtual_ptr 恒定、自动 flush_cache）
    与 FlattenedTensorBucket 单次字节流打包/解包零误差重构及 8B 模型同步提速账本。
    """
    ctrl = ColocatedRLMemoryController()
    orig_weight_ptr = ctrl.regions["weights"].virtual_ptr
    orig_kv_ptr = ctrl.regions["kv_cache"].virtual_ptr
    assert ctrl.active_physical_gb() == 41.5
    assert ctrl.radix_cache_entries == 128

    # 1. Rollout 结束进入睡眠：释放 weights 与 kv_cache 物理页，让出显存给训练引擎
    ctrl.release_memory_occupation(["weights", "kv_cache"])
    sleep_physical_gb = ctrl.active_physical_gb()
    assert sleep_physical_gb == 1.5
    assert ctrl.radix_cache_entries == 0  # 验证释放 kv_cache 时自动调用 flush_cache()
    assert ctrl.regions["weights"].virtual_ptr == orig_weight_ptr
    assert ctrl.regions["kv_cache"].virtual_ptr == orig_kv_ptr

    # 2. 第一阶段唤醒：仅唤醒 weights 接收训练侧新权重（此时 kv_cache 仍处于睡眠态，降低通信峰值显存）
    ctrl.resume_memory_occupation(["weights"])
    wake_phase1_gb = ctrl.active_physical_gb()
    assert wake_phase1_gb == 17.5

    # 3. 使用 FlattenedTensorBucket 将多层参数打包为连续字节流并原位写入 weights 虚拟地址
    src_tensors: list[tuple[str, tuple[int, ...], str, list[float]]] = [
        ("layers.0.self_attn.q_proj.weight", (2, 4), "float32", [0.125, -0.25, 0.5, -0.75, 1.25, -1.5, 0.875, 0.0625]),
        ("layers.0.self_attn.k_proj.weight", (2, 2), "float32", [-0.5, 0.25, 0.75, -0.125]),
        ("layers.0.input_layernorm.weight", (4,), "float32", [1.0, 1.0, 1.0, 1.0]),
    ]
    sender_bucket = FlattenedTensorBucket(named_tensors=src_tensors)
    assert len(sender_bucket.flattened_bytes) == (8 + 4 + 4) * 4

    # 模拟跨进程/NCCL 接收连续字节流并在同一虚拟地址原位解包
    ctrl.regions["weights"].payload = bytearray(sender_bucket.flattened_bytes)
    receiver_bucket = FlattenedTensorBucket(
        flattened_bytes=bytes(ctrl.regions["weights"].payload),
        metadata=sender_bucket.metadata,
    )
    unpacked_tensors = receiver_bucket.reconstruct_tensors()

    for (_, orig_shape, _, orig_vals), (_, rec_shape, rec_vals) in zip(src_tensors, unpacked_tensors):
        assert orig_shape == rec_shape
        assert max(abs(a - b) for a, b in zip(orig_vals, rec_vals)) == 0.0

    # 4. 第二阶段唤醒：权重同步完成后唤醒 kv_cache，准备下一轮 Rollout
    ctrl.resume_memory_occupation(["kv_cache"])
    wake_phase2_gb = ctrl.active_physical_gb()
    assert wake_phase2_gb == 41.5
    assert ctrl.regions["weights"].virtual_ptr == orig_weight_ptr

    # 5. 核算 8B 模型（291 个张量，16GB BF16）逐张量同步 vs 分桶同步耗时账本
    num_tensors_8b = 291
    total_weight_gb = 16.0
    nvlink_bw_gb_s = 45.0
    per_call_overhead_ms = 1.8
    num_buckets = 4

    transfer_time_ms = (total_weight_gb / nvlink_bw_gb_s) * 1000.0
    per_tensor_sync_ms = transfer_time_ms + num_tensors_8b * per_call_overhead_ms
    bucket_sync_ms = transfer_time_ms + num_buckets * per_call_overhead_ms
    speedup_ratio = per_tensor_sync_ms / bucket_sync_ms
    assert speedup_ratio > 2.3

    print("\n=== [实验 08-B] RL 混部虚拟内存睡眠/唤醒 (Sleep/Wake) 与 FlattenedTensorBucket 热更新 ===")
    print(
        f"  物理显存变化: 全量运行={wake_phase2_gb:.1f} GB -> 睡眠态(仅留 CUDA Graph)={sleep_physical_gb:.1f} GB -> "
        f"阶段一唤醒(仅权重)={wake_phase1_gb:.1f} GB -> 阶段二全唤醒={wake_phase2_gb:.1f} GB"
    )
    print(
        f"  虚拟地址不变量: weights virtual_ptr=0x{ctrl.regions['weights'].virtual_ptr:x} (恒定不变，CUDA Graph 零重录) | "
        f"睡眠后 radix_cache_entries={ctrl.radix_cache_entries} (自动 flush_cache)"
    )
    print(
        f"  FlattenedTensorBucket 打包字节数: {len(sender_bucket.flattened_bytes)} Bytes | "
        f"解包重构最大绝对误差: 0.00e+00"
    )
    print(
        f"  8B 模型 (291 张量, 16GB) 热更新账本: 逐张量同步={per_tensor_sync_ms:.1f} ms vs "
        f"4 分桶同步={bucket_sync_ms:.1f} ms (提速 {speedup_ratio:.2f}x)"
    )


if __name__ == "__main__":
    verify_chunked_prefill_tail_latency()
    verify_rl_sleep_wake_and_tensor_bucket()
    print("\n[PASS] 实验 08 全部断言核验通过。")

# ==============================================================================
# 思考与动手改代码挑战题（完成实验后请亲手验证）
# ==============================================================================
#
# 【挑战题 1：如果在 release_memory_occupation 或权重热更新后漏调 flush_cache()，会发生什么脏缓存读取故障？】
#   - 原理提示：见 sglang/python/sglang/srt/managers/scheduler_components/weight_updater.py 第 366 行。
#     当睡眠释放 kv_cache 物理显存页并重新映射新物理页，或者模型权重已经通过 FlattenedTensorBucket 更新为新策略版本后，
#     如果 CPU 侧的 RadixPrefixCache 仍保留着旧策略或已释放物理页的节点索引（radix_cache_entries > 0），
#     下一轮 Rollout 请求命中前缀缓存时就会直接读取未初始化或旧版本策略算出的脏 KV 数据，导致生成乱码或重要性采样比率失真。
#   - 动手实验：注释掉 release_memory_occupation() 中的 self.flush_cache()，观察断言如何捕获残留的已失效的脏缓存条目。
#
# 【挑战题 2：在 FlattenedTensorBucket 中混入奇数字节张量为什么会触发字节未对齐异常？】
#   - 原理提示：见 sglang/python/sglang/srt/weight_sync/tensor_bucket.py 第 98~101 行：
#     `if offset % dtype.itemsize: raise ValueError("Unaligned received tensor ...")`。
#     当张量桶将多个不同 dtype 的张量展平拼接为单一 uint8 缓冲区时，若前一个张量为 1 字节类型（如 FP8 权重的 uint8 缩放因子）
#     且元素个数为奇数（或非 4 字节倍数），紧随其后的 BF16（2 字节）或 FP32（4 字节）张量起始偏移 start_idx 就会落在非对齐字节边界上，
#     导致底层 CUDA 向量化访存报错或解包失败。
#   - 动手实验：在 src_tensors 列表开头插入一个长度为 3 的 ("fp8_scale", (3,), "uint8", [1.0, 1.0, 1.0])，
#     运行脚本观察 FlattenedTensorBucket 抛出的 Unaligned tensor 异常，并尝试补充向上对齐（align_up）逻辑修复它。
