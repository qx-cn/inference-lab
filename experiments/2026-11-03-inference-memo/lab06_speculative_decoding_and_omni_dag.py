#!/usr/bin/env python3
"""
实验 06：无损投机解码（Speculative Decoding）拒绝采样分布等价性证明与 sglang-omni 多阶段流式延迟仿真
对应计划：第 6 周 & 第 8 周（对齐 zero-to-sglang Part 2 第 10 章与 Awesome-ML-SYS-Tutorial/sglang/sglang-omni/）
运行方式：先复制整个实验目录，再在副本目录运行 python3 lab06_speculative_decoding_and_omni_dag.py（见 README.md）。
"""

from __future__ import annotations

import random


def sample_categorical(probs: list[float]) -> int:
    r = random.random()
    cdf = 0.0
    for idx, p in enumerate(probs):
        cdf += p
        if r <= cdf:
            return idx
    return len(probs) - 1


def speculative_rejection_step(target_p: list[float], draft_q: list[float]) -> tuple[int, bool]:
    """
    执行单步无损投机采样：
    1. 从草稿模型分布 q(x) 采样候选 token x；
    2. 以 min(1, p(x) / q(x)) 的概率接受 x；
    3. 若拒绝，则从修正后的残差分布 p'(x) = norm(max(0, p(x) - q(x))) 中重新采样。
    """
    x = sample_categorical(draft_q)
    accept_prob = min(1.0, target_p[x] / draft_q[x])
    if random.random() < accept_prob:
        return x, True

    residual = [max(0.0, p_i - q_i) for p_i, q_i in zip(target_p, draft_q)]
    norm_sum = sum(residual)
    residual_probs = [r / norm_sum for r in residual]
    return sample_categorical(residual_probs), False


def verify_lossless_speculative_sampling(num_trials: int = 200_000) -> None:
    """通过蒙特卡洛实验验证投机采样输出分布与目标大模型分布 p(x) 严格一致。"""
    random.seed(2026)
    # 目标大模型真实分布 p 与草稿小模型有偏分布 q
    target_p = [0.45, 0.25, 0.15, 0.10, 0.05]
    draft_q = [0.20, 0.40, 0.20, 0.10, 0.10]

    counts = [0] * len(target_p)
    accepted_count = 0
    for _ in range(num_trials):
        tok, accepted = speculative_rejection_step(target_p, draft_q)
        counts[tok] += 1
        if accepted:
            accepted_count += 1

    empirical_p = [c / num_trials for c in counts]
    theoretical_accept_rate = sum(min(p_i, q_i) for p_i, q_i in zip(target_p, draft_q))
    empirical_accept_rate = accepted_count / num_trials
    tv_distance = 0.5 * sum(abs(ep - tp) for ep, tp in zip(empirical_p, target_p))

    print("=== [实验 06-A] 无损投机解码（Speculative Decoding）分布一致性蒙特卡洛核验 ===")
    print(f"  目标分布 p(x): {[round(x, 4) for x in target_p]}")
    print(f"  草稿分布 q(x): {[round(x, 4) for x in draft_q]} (存在显著偏差)")
    print(f"  投机输出分布 : {[round(x, 4) for x in empirical_p]} ({num_trials:,d} 次采样)")
    print(f"  理论接受率 sum(min(p,q)) = {theoretical_accept_rate:.4f} | 实测接受率 = {empirical_accept_rate:.4f}")
    print(f"  与目标分布总变差距离 (TVD) = {tv_distance:.4f} (< 0.005 即证明无损等价)")
    assert tv_distance < 0.005
    assert abs(empirical_accept_rate - theoretical_accept_rate) < 0.005


def simulate_sglang_omni_ttfa() -> None:
    """
    验证第 8 周 sglang-omni 必答题 1 & 2：
    1. 对比 Talker 与 MTP 跨 Stage 拆分通信 vs 封装进同一 FeedbackARModelRunner 进程内融合的步间延迟；
    2. 对比 Vocoder 不同流式窗口大小（window_frames）对首包音频延迟（TTFA）的影响。
    """
    thinker_ttft_ms = 35.0
    talker_backbone_ms = 1.8
    mtp_7heads_ms = 1.2
    cross_stage_zmq_relay_ms = 2.5

    # 若将 Talker 与 MTP 拆为两个独立 Stage，每帧音频 Codec 都要额外付 2 次跨进程 Relay 通信开销
    step_disaggregated_ms = talker_backbone_ms + mtp_7heads_ms + 2 * cross_stage_zmq_relay_ms
    # sglang-omni 将 Talker + MTP 融合在同一个 FeedbackARModelRunner Stage 内并捕获 CUDA Graph
    step_fused_runner_ms = (talker_backbone_ms + mtp_7heads_ms) * 0.75

    print("\n=== [实验 06-B] sglang-omni 多阶段流式语音生成首包延迟 (TTFA) 仿真 ===")
    print(f"  Talker+MTP 跨进程拆分单帧耗时: {step_disaggregated_ms:.2f} ms vs FeedbackARModelRunner 进程内融合耗时: {step_fused_runner_ms:.2f} ms")
    assert step_fused_runner_ms < step_disaggregated_ms * 0.4

    vocoder_conv_ms = 4.0
    for window_frames in (4, 8, 24):
        ttfa_ms = thinker_ttft_ms + window_frames * step_fused_runner_ms + vocoder_conv_ms
        note = (
            "首包极快，但需维护卷积历史上下文缓存避免块边界爆音"
            if window_frames == 4
            else ("吞吐与音质平衡点" if window_frames == 8 else "窗口过大拖慢首包交互感")
        )
        print(f"  Vocoder 流式窗口 = {window_frames:2d} 帧 -> 首包音频延迟 (TTFA) = {ttfa_ms:5.1f} ms ({note})")
    print("\n[PASS] 实验 06 全部断言核验通过。")


if __name__ == "__main__":
    verify_lossless_speculative_sampling()
    simulate_sglang_omni_ttfa()
