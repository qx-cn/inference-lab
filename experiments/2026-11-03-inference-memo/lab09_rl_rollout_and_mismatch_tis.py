#!/usr/bin/env python3
"""
实验 09：强化学习过采样部分采样（Partial Rollout）、训推不一致（MoE 路由翻转与 MIS/Veto 修正）及 Perfetto Trace 自动诊断
对应计划：第 7 周 & 第 8 周（对齐 slime/code-walk-through/readme.md 与 slime/mismatch/blog-cn.md）
运行方式：先复制整个实验目录，再在副本目录运行 python3 lab09_rl_rollout_and_mismatch_tis.py（见 README.md）。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct


@dataclass
class RolloutGroup:
    prompt_id: int
    completion_times_s: list[float]
    rewards: list[float]
    total_tokens_per_req: int = 512

    @property
    def group_finish_time_s(self) -> float:
        return max(self.completion_times_s)

    @property
    def reward_std(self) -> float:
        mean_r = sum(self.rewards) / len(self.rewards)
        variance = sum((r - mean_r) ** 2 for r in self.rewards) / len(self.rewards)
        return math.sqrt(variance)


def verify_partial_rollout_and_multiturn_loss_mask() -> None:
    """
    实验 09-A：对齐 Awesome-ML-SYS-Tutorial/rlhf/slime/code-walk-through/readme.md
    1. 对比同步等待 180s 长尾请求与“过采样 + 动态过滤（剔除零方差组）+ 达标Partial Rollout（中断未完成长尾请求并回池留到下一轮接着生成）”的墙钟耗时；
    2. 核验多轮工具调用中 assistant（loss_mask=1）与 tool（loss_mask=0）的损失掩码生成规则。
    """
    rollout_batch_size = 4
    over_sampling_batch_size = 6

    # 构造 6 个候选 Prompt Group（每组 G=4 条采样轨迹）：
    # Group 1 奖励全为 1.0（reward_std == 0，零梯度无效组，会被 dynamic_filter 剔除）
    # Group 5 包含一条 180.0s 的超长尾请求（Straggler）
    candidate_groups = [
        RolloutGroup(prompt_id=101, completion_times_s=[18.0, 22.0, 25.0, 28.0], rewards=[1.0, 0.0, 1.0, 0.0]),
        RolloutGroup(prompt_id=102, completion_times_s=[15.0, 19.0, 20.0, 21.0], rewards=[1.0, 1.0, 1.0, 1.0]),
        RolloutGroup(prompt_id=103, completion_times_s=[20.0, 26.0, 31.0, 34.0], rewards=[0.0, 1.0, 0.0, 0.0]),
        RolloutGroup(prompt_id=104, completion_times_s=[24.0, 29.0, 35.0, 38.0], rewards=[1.0, 1.0, 0.0, 1.0]),
        RolloutGroup(prompt_id=105, completion_times_s=[28.0, 33.0, 39.0, 42.0], rewards=[0.0, 0.0, 1.0, 0.0]),
        RolloutGroup(prompt_id=106, completion_times_s=[30.0, 36.0, 45.0, 180.0], rewards=[1.0, 0.0, 1.0, 0.0]),
    ]
    assert len(candidate_groups) == over_sampling_batch_size

    # 1. 传统同步等待模式：必须等待批次内最慢的 180.0s 长尾请求完成，且未剔除零方差组
    sync_wall_time_s = max(g.group_finish_time_s for g in candidate_groups)
    assert sync_wall_time_s == 180.0

    # 2. slime 过采样 + 动态过滤 (check_reward_nonzero_std) + 达标Partial Rollout 回池 (abort & recycle to data_buffer)
    # 按完成时间顺序流式收集，剔除 reward_std == 0 的 Group 102，集齐 4 个有效组时（42.0s）立即触发 abort
    completed_by_time = sorted(candidate_groups, key=lambda g: g.group_finish_time_s)
    accepted_groups: list[RolloutGroup] = []
    filtered_zero_std_groups: list[int] = []
    aborted_recycled_groups: list[tuple[int, int]] = []
    partial_rollout_wall_time_s = 0.0

    for grp in completed_by_time:
        if len(accepted_groups) < rollout_batch_size:
            partial_rollout_wall_time_s = grp.group_finish_time_s
            if grp.reward_std > 0.0:
                accepted_groups.append(grp)
            else:
                filtered_zero_std_groups.append(grp.prompt_id)
        else:
            # 达到 target_data_size = rollout_batch_size，立即向 SGLang 发送 /abort_request 回收未完成样本组写入 data_buffer
            saved_tokens = int(
                sum(min(1.0, partial_rollout_wall_time_s / t) * grp.total_tokens_per_req for t in grp.completion_times_s)
            )
            aborted_recycled_groups.append((grp.prompt_id, saved_tokens))

    assert len(accepted_groups) == rollout_batch_size
    assert filtered_zero_std_groups == [102]
    assert partial_rollout_wall_time_s == 42.0
    assert len(aborted_recycled_groups) == 1 and aborted_recycled_groups[0][0] == 106

    # 3. 核验多轮工具调用（Search-R1 / Agent Loop）的 loss_mask 生成规则
    # 仅模型自身生成的 assistant token计入策略梯度损失（loss_mask=1），环境/工具返回与用户提示词严格置 0
    multiturn_messages: list[tuple[str, list[int]]] = [
        ("system", [1, 2, 3]),
        ("user", [10, 11, 12, 13]),
        ("assistant", [20, 21, 22, 23, 24]),      # 第 1 轮思考与发起工具调用 -> loss_mask = 1
        ("tool", [30, 31, 32, 33, 34, 35]),       # 工具检索返回结果（环境观测） -> loss_mask = 0
        ("assistant", [40, 41, 42, 43]),          # 第 2 轮基于工具结果给出最终回答 -> loss_mask = 1
    ]

    full_tokens: list[int] = []
    full_loss_mask: list[int] = []
    first_assistant_idx: int | None = None

    for role, toks in multiturn_messages:
        if role == "assistant" and first_assistant_idx is None:
            first_assistant_idx = len(full_tokens)
        full_tokens.extend(toks)
        mask_val = 1 if role == "assistant" else 0
        full_loss_mask.extend([mask_val] * len(toks))

    assert first_assistant_idx is not None
    response_length = len(full_tokens) - first_assistant_idx
    response_loss_mask = full_loss_mask[-response_length:]
    assert len(response_loss_mask) == response_length == 15
    assert response_loss_mask == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    assert sum(response_loss_mask) == 9

    print("=== [实验 09-A] 强化学习过采样 + 部分采样 (Partial Rollout) 与多轮工具 loss_mask 核验 ===")
    print(
        f"  同步等待长尾耗时: {sync_wall_time_s:.1f} s vs 过采样+动态过滤+达标中断耗时: "
        f"{partial_rollout_wall_time_s:.1f} s (提速 {sync_wall_time_s / partial_rollout_wall_time_s:.2f}x)"
    )
    print(
        f"  动态过滤剔除零方差组: {filtered_zero_std_groups} | "
        f"Partial Rollout 回池 (data_buffer) 样本组及已缓存 token 数: {aborted_recycled_groups}"
    )
    print(
        f"  多轮工具轨迹 response_length={response_length}, 有效训练 token 数={sum(response_loss_mask)}, "
        f"response_loss_mask={response_loss_mask}"
    )


def fp32_round(val: float) -> float:
    """使用 struct.pack('<f', ...) 强制截断为 IEEE-754 单精度浮点数（FP32）。"""
    return struct.unpack("<f", struct.pack("<f", val))[0]


def fp32_sequential_sum(vals: list[float]) -> float:
    """模拟训练侧标准顺序累加归约。"""
    acc = fp32_round(0.0)
    for v in vals:
        acc = fp32_round(acc + fp32_round(v))
    return acc


def fp32_split_k_sum(vals: list[float], split_k: int = 2) -> float:
    """模拟推理侧解码阶段 Split-K 分块并行归约。"""
    chunk_size = len(vals) // split_k
    partial_sums: list[float] = []
    for s in range(split_k):
        chunk = vals[s * chunk_size : (s + 1) * chunk_size]
        acc = fp32_round(0.0)
        for v in chunk:
            acc = fp32_round(acc + fp32_round(v))
        partial_sums.append(acc)
    total = fp32_round(0.0)
    for p in partial_sums:
        total = fp32_round(total + p)
    return total


def k3_kl_divergence(p_train: float, q_infer: float) -> float:
    """计算 K3 KL 散度估计器: k3(x) = r - 1 - log(r), 其中 r = p_train / q_infer。"""
    r = p_train / q_infer
    return r - 1.0 - math.log(r)


def verify_train_infer_mismatch_and_mis_veto() -> None:
    """
    实验 09-B：对齐 Awesome-ML-SYS-Tutorial/rlhf/slime/mismatch/blog-cn.md
    1. 使用 struct.pack('<f', ...) 复现训练侧顺序累加与推理侧 Split-K 分块归约的浮点非结合律差异；
    2. 展示该微小差异如何在 MoE 门控临界处触发专家路由翻转，使 K3 KL 散度放大 > 10^9 倍；
    3. 验证掩码重要性采样（MIS）+ 否决阈值（Veto）+ 批次自归一化对异常序列梯度爆炸的精准拦截。
    """
    # 1. 复现 FP32 顺序归约 vs Split-K 归约的微小数值差异
    # 先加大数 1.0 再累加 7 个 5e-8：顺序累加时每次 1.0 + 5e-8 在 FP32 下被舍入吞噬 (ULP=1.19e-7)，
    # 而 Split-K=2 时后半部分 4 个 5e-8 先聚合成 2.0e-7 超过半个 ULP，成功进位！
    summands = [1.0, 5e-8, 5e-8, 5e-8, 5e-8, 5e-8, 5e-8, 5e-8]
    gate_logit_train = fp32_sequential_sum(summands)
    gate_logit_infer = fp32_split_k_sum(summands, split_k=2)
    fp32_diff = abs(gate_logit_train - gate_logit_infer)
    assert gate_logit_train != gate_logit_infer
    assert 0.0 < fp32_diff < 1e-6

    # 2. MoE 门控临界翻转：假设专家 0 的门控得分为常数 1.0000000596 (即 1.0 + 0.5 * ULP)
    # 专家 1 的门控得分即为上面算出的 gate_logit（训练侧为 1.0000000，推理侧为 1.0000001192）
    expert0_score = 1.0000000596
    selected_expert_train = 1 if gate_logit_train > expert0_score else 0
    selected_expert_infer = 1 if gate_logit_infer > expert0_score else 0
    assert selected_expert_train == 0 and selected_expert_infer == 1

    # 未翻转时（Dense 模型或同专家分支）：两端的预测概率仅有 FP32 舍入级差异
    p_train_dense = 0.2500000
    q_infer_dense = float(fp32_round(0.2500000 + fp32_diff))
    k3_dense = k3_kl_divergence(p_train_dense, q_infer_dense)

    # 发生专家路由翻转后：专家 0 与专家 1 给出的目标token概率截然不同（如 0.42 vs 1.2e-7），导致 K3 KL 暴增 > 10^9 倍
    p_train_moe_flipped = 0.42
    q_infer_moe_flipped = 1.2e-7
    k3_moe_flipped = k3_kl_divergence(p_train_moe_flipped, q_infer_moe_flipped)
    amplification = k3_moe_flipped / k3_dense
    assert amplification > 1e9

    # 3. 验证 MIS (Masked Importance Sampling) + Veto (否决机制) + 批次自归一化 (--tis-batch-normalize)
    # 构造一个含 4 条序列的训练批次，其中序列 2 出现了 MoE 路由翻转产生的极端低概率token (q_infer = 1.2e-7 < veto_threshold = 1e-6)，
    # 序列 3 几何重要性比率偏离信任域 [0.99, 1.001]
    veto_threshold = 1e-6
    tis_clip_min, tis_clip_max = 0.5, 2.0
    geo_mis_min, geo_mis_max = 0.99, 1.001

    batch_sequences: list[tuple[str, list[tuple[float, float]]]] = [
        ("seq_0 (正常对齐)", [(0.30, 0.3001), (0.45, 0.4499), (0.20, 0.2000)]),
        ("seq_1 (轻微偏移)", [(0.25, 0.2498), (0.50, 0.4997), (0.35, 0.3499)]),
        ("seq_2 (MoE路由翻转极小概率)", [(0.30, 0.3000), (0.42, 1.2e-7), (0.25, 0.2500)]),
        ("seq_3 (中度分布漂移)", [(0.40, 0.3600), (0.35, 0.3100), (0.30, 0.2700)]),
    ]

    raw_max_ratios: list[float] = []
    surviving_seq_weights: list[float] = []
    seq_status_notes: list[str] = []

    for seq_name, token_probs in batch_sequences:
        ratios = [p_tr / q_inf for p_tr, q_inf in token_probs]
        raw_max_ratios.append(max(ratios))

        # (a) Veto 否决机制：若序列中存在任何旧策略概率 q_infer < 1e-6 的token，整条序列权重置 0
        if any(q_inf < veto_threshold for _, q_inf in token_probs):
            surviving_seq_weights.append(0.0)
            seq_status_notes.append(f"{seq_name}: 触发 Veto (min q_infer < {veto_threshold:.0e}) -> 序列掩码置 0")
            continue

        # (b) Geometric MIS：计算序列几何平均重要性比率 g = (prod r_t)^(1/T)
        geo_ratio = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        if not (geo_mis_min <= geo_ratio <= geo_mis_max):
            surviving_seq_weights.append(0.0)
            seq_status_notes.append(
                f"{seq_name}: 触发 Geometric MIS (geo_ratio={geo_ratio:.5f} 越界) -> 序列掩码置 0"
            )
            continue

        # (c) Token-level TIS 裁剪并求均值作为有效权重
        clipped_ratios = [min(tis_clip_max, max(tis_clip_min, r)) for r in ratios]
        seq_w = sum(clipped_ratios) / len(clipped_ratios)
        surviving_seq_weights.append(seq_w)
        seq_status_notes.append(f"{seq_name}: 通过校验 (geo_ratio={geo_ratio:.5f}) -> 原始有效权重={seq_w:.5f}")

    # (d) 批次自归一化 (--tis-batch-normalize)：使有效非零样本的平均重要性权重严格等于 1.0
    valid_weights = [w for w in surviving_seq_weights if w > 0.0]
    mean_valid_w = sum(valid_weights) / len(valid_weights)
    normalized_weights = [w / mean_valid_w if w > 0.0 else 0.0 for w in surviving_seq_weights]

    assert raw_max_ratios[2] > 3e6
    assert normalized_weights[2] == 0.0 and normalized_weights[3] == 0.0
    assert abs((sum(w for w in normalized_weights if w > 0.0) / len(valid_weights)) - 1.0) < 1e-12

    print("\n=== [实验 09-B] 训推不一致复现、MoE 专家路由翻转与 MIS/Veto 修正核验 ===")
    print(
        f"  FP32 非结合律复现: 训练顺序归约={gate_logit_train:.8f} vs 推理 Split-K 归约={gate_logit_infer:.8f} "
        f"(差值={fp32_diff:.2e})"
    )
    print(
        f"  MoE 门控临界翻转: 训练选专家 {selected_expert_train} vs 推理选专家 {selected_expert_infer} -> "
        f"Dense K3 KL={k3_dense:.2e}, MoE 翻转后 K3 KL={k3_moe_flipped:.2e} (放大 {amplification:.2e} 倍)"
    )
    for note, norm_w in zip(seq_status_notes, normalized_weights):
        print(f"  {note} | 自归一化后最终权重={norm_w:.4f}")


def build_perfetto_trace_events(sync_item_stall: bool) -> dict[str, object]:
    """
    生成符合 Chrome Trace Event 规范（可直接拖入 https://ui.perfetto.dev 查看）的时间线数据。
    pid=1 表示 CPU Scheduler 主线程，pid=2 表示 GPU Compute Stream。
    时间戳单位为微秒（us）。
    """
    events: list[dict[str, object]] = [
        {"name": "process_name", "ph": "M", "pid": 1, "tid": 1, "args": {"name": "CPU Scheduler Process"}},
        {"name": "thread_name", "ph": "M", "pid": 1, "tid": 1, "args": {"name": "Scheduler Main Thread"}},
        {"name": "process_name", "ph": "M", "pid": 2, "tid": 1, "args": {"name": "GPU Device 0"}},
        {"name": "thread_name", "ph": "M", "pid": 2, "tid": 1, "args": {"name": "GPU Compute Stream"}},
    ]

    num_steps = 4
    cpu_pre_us = 1200
    gpu_fwd_us = 4000
    sync_copy_us = 900
    cpu_post_us = 800

    if sync_item_stall:
        # 反例（Bad Trace）：每步在 CPU 上调用 next_token_ids.item() 触发 cudaStreamSynchronize，
        # 导致 CPU 必须等 GPU 跑完前向与同步拷贝后才能做后处理和下一步 Pre-process，GPU 流出现大段空泡
        cur_us = 0
        for step in range(num_steps):
            events.append(
                {"name": f"CPU_Prepare_Step_{step}", "cat": "cpu", "ph": "X", "pid": 1, "tid": 1, "ts": cur_us, "dur": cpu_pre_us}
            )
            gpu_start_us = cur_us + cpu_pre_us
            events.append(
                {"name": f"GPU_Forward_Sample_Step_{step}", "cat": "gpu", "ph": "X", "pid": 2, "tid": 1, "ts": gpu_start_us, "dur": gpu_fwd_us}
            )
            sync_start_us = gpu_start_us
            events.append(
                {
                    "name": f"cudaStreamSynchronize_tensor.item()_Step_{step}",
                    "cat": "sync_stall",
                    "ph": "X",
                    "pid": 1,
                    "tid": 1,
                    "ts": sync_start_us,
                    "dur": gpu_fwd_us + sync_copy_us,
                }
            )
            post_start_us = sync_start_us + gpu_fwd_us + sync_copy_us
            events.append(
                {"name": f"CPU_PostProcess_Step_{step}", "cat": "cpu", "ph": "X", "pid": 1, "tid": 1, "ts": post_start_us, "dur": cpu_post_us}
            )
            cur_us = post_start_us + cpu_post_us
    else:
        # 正例（Good Overlap Trace）：使用 FutureMap 负索引占位，CPU 提前发射下一步 Kernel，
        # GPU Compute Stream 连续满载无缝拼接
        cpu_cur_us = 0
        gpu_cur_us = cpu_pre_us
        for step in range(num_steps):
            events.append(
                {"name": f"CPU_Prepare_Step_{step}", "cat": "cpu", "ph": "X", "pid": 1, "tid": 1, "ts": cpu_cur_us, "dur": cpu_pre_us}
            )
            gpu_start_us = max(gpu_cur_us, cpu_cur_us + cpu_pre_us)
            events.append(
                {"name": f"GPU_Forward_Sample_Step_{step}", "cat": "gpu", "ph": "X", "pid": 2, "tid": 1, "ts": gpu_start_us, "dur": gpu_fwd_us}
            )
            post_start_us = cpu_cur_us + cpu_pre_us
            events.append(
                {"name": f"CPU_Async_PostProcess_Step_{step}", "cat": "cpu", "ph": "X", "pid": 1, "tid": 1, "ts": post_start_us, "dur": cpu_post_us}
            )
            cpu_cur_us = post_start_us + cpu_post_us
            gpu_cur_us = gpu_start_us + gpu_fwd_us

    return {"traceEvents": events, "displayTimeUnit": "ms"}


def diagnose_trace_file(trace_path: Path) -> tuple[float, float, float, int]:
    """读取并解析 Perfetto Trace JSON 文件，计算 GPU 流总窗口、有效计算时长、GPU Bubble 占比与同步阻塞次数。"""
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    events = payload["traceEvents"]

    gpu_events = [e for e in events if e.get("ph") == "X" and e.get("pid") == 2]
    sync_stalls = [
        e for e in events if e.get("ph") == "X" and "tensor.item()" in str(e.get("name", ""))
    ]

    all_complete = [e for e in events if e.get("ph") == "X"]
    timeline_start_us = min(int(e["ts"]) for e in all_complete)
    timeline_end_us = max(int(e["ts"]) + int(e["dur"]) for e in all_complete)
    total_window_ms = (timeline_end_us - timeline_start_us) / 1000.0

    gpu_busy_ms = sum(int(e["dur"]) for e in gpu_events) / 1000.0
    gpu_bubble_ms = total_window_ms - gpu_busy_ms
    bubble_ratio = gpu_bubble_ms / total_window_ms
    return total_window_ms, gpu_busy_ms, bubble_ratio, len(sync_stalls)


def verify_perfetto_trace_generation_and_diagnosis() -> None:
    """
    实验 09-C：生成 trace_bad_sync.json 与 trace_good_overlap.json 到 experiments/2026-11-03-inference-memo/evidence/，
    并自动读取解析这两份 JSON，诊断出因同步调用 tensor.item() 造成的 GPU Stream 空泡时长与GPU Bubble 占比对比。
    """
    evidence_dir = Path(__file__).resolve().parent / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    bad_trace_path = evidence_dir / "trace_bad_sync.json"
    good_trace_path = evidence_dir / "trace_good_overlap.json"

    bad_trace_path.write_text(
        json.dumps(build_perfetto_trace_events(sync_item_stall=True), indent=2),
        encoding="utf-8",
    )
    good_trace_path.write_text(
        json.dumps(build_perfetto_trace_events(sync_item_stall=False), indent=2),
        encoding="utf-8",
    )

    bad_total_ms, bad_busy_ms, bad_bubble_ratio, bad_sync_count = diagnose_trace_file(bad_trace_path)
    good_total_ms, good_busy_ms, good_bubble_ratio, good_sync_count = diagnose_trace_file(good_trace_path)

    assert bad_sync_count == 4 and good_sync_count == 0
    assert bad_busy_ms == good_busy_ms == 16.0
    assert bad_bubble_ratio > 0.40
    assert good_bubble_ratio < 0.08

    print("\n=== [实验 09-C] 5 合 1 排障靶场与 Perfetto Trace JSON 自动生成及空泡诊断 ===")
    print(f"  已生成可交互 Trace 文件 1: {bad_trace_path}")
    print(f"  已生成可交互 Trace 文件 2: {good_trace_path}")
    print(
        f"  [诊断 trace_bad_sync.json]     总耗时={bad_total_ms:5.2f} ms | GPU 有效计算={bad_busy_ms:5.2f} ms | "
        f"GPU GPU Bubble 占比={bad_bubble_ratio * 100:4.1f}% | 检出 tensor.item() 同步阻塞={bad_sync_count} 次"
    )
    print(
        f"  [诊断 trace_good_overlap.json] 总耗时={good_total_ms:5.2f} ms | GPU 有效计算={good_busy_ms:5.2f} ms | "
        f"GPU GPU Bubble 占比={good_bubble_ratio * 100:4.1f}% | 检出 tensor.item() 同步阻塞={good_sync_count} 次"
    )

    # 5 合 1 线上高频故障特征自动归因靶场
    troubleshooting_cases: list[tuple[str, str, str]] = [
        ("GPU_Stream_Bubble>40% + cudaStreamSynchronize", "Python 线程误调 .item()/.cpu() 阻塞重叠调度", "改用 FutureMap 负索引占位与异步拷贝"),
        ("evictable_size==0 + protected_size 持续单调增长", "请求异常退出漏调 lock_handle(unlock=True) 引发基数树泄漏", "在 abort/finish 路径统一释放树节点引用计数"),
        ("ITL_P99>200ms 且伴随单步 32K Prefill 尖刺", "未开启分块预填充导致长请求独占阻塞在线 Decode", "启用 Chunked Prefill (chunk_size=2048)"),
        ("MoE 训推 K3_KL>1e2 且单步 grad_norm 暴增", "Split-K 非结合律引发门控临界处专家路由翻转", "开启 Veto (1e-6) + Geometric MIS + 批次自归一化"),
        ("Decode 阶段 SM 活跃度 <15% 但显存带宽未打满", "长序列单查询未启用 Split-KV 导致网格并行度不足", "启用 FlashDecoding Split-KV 并用 LSE 归约"),
    ]
    assert len(troubleshooting_cases) == 5
    for idx, (symptom, root_cause, fix) in enumerate(troubleshooting_cases, start=1):
        print(f"  [靶场归因 #{idx}] 现象: {symptom} -> 根因: {root_cause} -> 处方: {fix}")


if __name__ == "__main__":
    verify_partial_rollout_and_multiturn_loss_mask()
    verify_train_infer_mismatch_and_mis_veto()
    verify_perfetto_trace_generation_and_diagnosis()
    print("\n[PASS] 实验 09 全部断言核验通过。")
