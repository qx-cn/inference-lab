#!/usr/bin/env python3
"""
实验 04：零开销重叠调度（Overlap Scheduling + FutureMap）与张量并行（TP）分块矩阵乘验证
对应计划：第 4 周（对齐 mini-sglang/python/minisgl/scheduler/scheduler.py 与 layers/linear.py）
运行方式：先复制整个实验目录，再在副本目录运行 python3 lab04_overlap_scheduling_and_tp.py（见 README.md）。
"""


from __future__ import annotations

import math


def simulate_serial_vs_overlap_timeline(num_steps: int = 8) -> None:
    """对比串行调度（Normal Loop）与重叠调度（Overlap Loop）的 GPU 空泡（Bubble）占比。"""
    cpu_pre_ms = 1.5
    gpu_fwd_ms = 4.0
    cpu_post_ms = 1.0

    # 1. 串行调度：每步总耗时 = CPU Pre + GPU Forward + CPU Post
    serial_total_ms = num_steps * (cpu_pre_ms + gpu_fwd_ms + cpu_post_ms)
    serial_gpu_busy_ms = num_steps * gpu_fwd_ms
    serial_bubble_ratio = 1.0 - (serial_gpu_busy_ms / serial_total_ms)

    # 2. 重叠调度：第 N 步的 GPU Forward 与“第 N-1 步的 CPU Post + 第 N+1 步的 CPU Pre”并行
    overlap_step_ms = max(gpu_fwd_ms, cpu_pre_ms + cpu_post_ms)
    overlap_total_ms = cpu_pre_ms + num_steps * overlap_step_ms + cpu_post_ms
    overlap_bubble_ratio = 1.0 - (serial_gpu_busy_ms / overlap_total_ms)

    print("=== [实验 04-A] 串行调度 vs 零开销重叠调度时间线仿真 ===")
    print(f"  单步耗时设定: CPU Pre={cpu_pre_ms}ms, GPU Forward+Sample={gpu_fwd_ms}ms, CPU Post={cpu_post_ms}ms")
    print(f"  [串行 Normal Loop]  {num_steps} 步总耗时: {serial_total_ms:5.1f} ms | GPU GPU Bubble 占比: {serial_bubble_ratio * 100:4.1f}%")
    print(f"  [重叠 Overlap Loop] {num_steps} 步总耗时: {overlap_total_ms:5.1f} ms | GPU GPU Bubble 占比: {overlap_bubble_ratio * 100:4.1f}%")
    assert overlap_total_ms < serial_total_ms


def verify_future_map_mechanism() -> None:
    """验证 SGLang FutureMap 负索引占位在未完成第 N 步采样时如何提前组第 N+1 步 Batch。"""
    future_map = [0] * 16
    req_slot = 3

    # 第 N 步 CPU 调度时：采样尚未发生，分配负索引 -req_slot 作为占位符
    step_n_plus_1_input_id = -req_slot
    assert step_n_plus_1_input_id < 0

    # 第 N 步 GPU 采样流完成时：将采样出的真实 Token ID 8888 写入 future_map[req_slot]
    sampled_token_id = 8888
    future_map[req_slot] = sampled_token_id

    # 第 N+1 步 GPU 前向流启动时：将负索引解引用为 future_map[-input_id]
    resolved_input_id = future_map[-step_n_plus_1_input_id] if step_n_plus_1_input_id < 0 else step_n_plus_1_input_id
    assert resolved_input_id == 8888
    print("\n=== [实验 04-B] FutureMap 负索引占位与 GPU 流内异步解引用核验 ===")
    print(
        f"  CPU 预设输入占位符: {step_n_plus_1_input_id} -> "
        f"GPU 第 N 步采样写回 future_map[{req_slot}]={sampled_token_id} -> "
        f"GPU 第 N+1 步解引用得到真实 Token ID={resolved_input_id}"
    )


def matmul_2d(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    m, k = len(a), len(a[0])
    k2, n = len(b), len(b[0])
    assert k == k2
    return [[sum(a[i][p] * b[p][j] for p in range(k)) for j in range(n)] for i in range(m)]


def swiglu_elem(gate: list[list[float]], up: list[list[float]]) -> list[list[float]]:
    """逐元素激活函数：因为按元素独立计算，所以在按列切分后无需跨卡通信即可直接激活。"""
    m, n = len(gate), len(gate[0])
    out = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            g = gate[i][j]
            silu_g = g / (1.0 + math.exp(-g))
            out[i][j] = silu_g * up[i][j]
    return out


def verify_tp_column_row_parallel() -> None:
    """验证张量并行（TP=2）中 ColumnParallelLinear + 逐元素激活 + RowParallelLinear 只需 1 次 All-Reduce。"""
    x = [[0.5, -1.0, 1.5, 0.2], [-0.3, 0.8, -0.5, 1.2]]
    w_gate = [
        [0.1, 0.2, -0.1, 0.4],
        [0.3, -0.2, 0.5, 0.1],
        [-0.4, 0.1, 0.2, -0.3],
        [0.2, 0.3, -0.2, 0.5],
    ]
    w_up = [
        [0.2, -0.1, 0.3, 0.2],
        [-0.2, 0.4, 0.1, -0.3],
        [0.5, 0.2, -0.4, 0.1],
        [0.1, -0.3, 0.2, 0.4],
    ]
    w_down = [
        [0.3, -0.1, 0.2, 0.4],
        [-0.2, 0.5, 0.1, -0.2],
        [0.4, 0.2, -0.3, 0.1],
        [0.1, -0.4, 0.5, 0.2],
    ]

    # 1. 单卡全量计算基准：Y = SwiGLU(X @ W_gate, X @ W_up) @ W_down
    full_hidden = swiglu_elem(matmul_2d(x, w_gate), matmul_2d(x, w_up))
    y_single_gpu = matmul_2d(full_hidden, w_down)

    # 2. 双卡 TP=2：第一层按列切分 W_gate 和 W_up（中间零通信独立算 SwiGLU），第二层按行切分 W_down 后做 1 次求和 All-Reduce
    w_gate_rank0 = [row[:2] for row in w_gate]
    w_gate_rank1 = [row[2:] for row in w_gate]
    w_up_rank0 = [row[:2] for row in w_up]
    w_up_rank1 = [row[2:] for row in w_up]

    h_rank0 = swiglu_elem(matmul_2d(x, w_gate_rank0), matmul_2d(x, w_up_rank0))
    h_rank1 = swiglu_elem(matmul_2d(x, w_gate_rank1), matmul_2d(x, w_up_rank1))

    w_down_rank0 = w_down[:2]
    w_down_rank1 = w_down[2:]
    partial_y0 = matmul_2d(h_rank0, w_down_rank0)
    partial_y1 = matmul_2d(h_rank1, w_down_rank1)

    # 唯一一次 All-Reduce (Sum)
    y_tp2 = [
        [partial_y0[i][j] + partial_y1[i][j] for j in range(len(partial_y0[0]))]
        for i in range(len(partial_y0))
    ]

    max_diff = max(
        abs(y_single_gpu[i][j] - y_tp2[i][j])
        for i in range(len(y_single_gpu))
        for j in range(len(y_single_gpu[0]))
    )
    assert max_diff < 1e-12
    print("\n=== [实验 04-C] 张量并行 ColumnParallel + SwiGLU + RowParallel 数学等价性核验 ===")
    print(f"  单卡全量输出与双卡 TP=2 (仅 1 次 All-Reduce) 最大绝对误差: {max_diff:.2e}")
    print("\n[PASS] 实验 04 全部断言核验通过。")


if __name__ == "__main__":
    simulate_serial_vs_overlap_timeline()
    verify_future_map_mechanism()
    verify_tp_column_row_parallel()
