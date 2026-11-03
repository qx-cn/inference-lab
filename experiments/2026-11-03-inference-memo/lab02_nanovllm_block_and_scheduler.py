#!/usr/bin/env python3
"""
实验 02：nano-vllm 分页块管理器（BlockManager）、连续批处理抢占调度器（Scheduler）与物理槽位映射（slot_mapping）
对应计划：第 2 周（精确对齐 nano-vllm/nanovllm/engine/ 下的 sequence.py, block_manager.py, scheduler.py, model_runner.py）
运行方式：先复制整个实验目录，再在副本目录运行 python3 lab02_nanovllm_block_and_scheduler.py（见 README.md）。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
import hashlib


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """对齐 nano-vllm/nanovllm/engine/sequence.py 的序列状态载体。"""

    block_size: int = 4
    _counter: int = 0

    def __init__(self, token_ids: list[int], max_tokens: int = 4):
        self.seq_id = Sequence._counter
        Sequence._counter += 1
        self.status = SequenceStatus.WAITING
        self.token_ids = list(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table: list[int] = []
        self.max_tokens = max_tokens

    def __len__(self) -> int:
        return self.num_tokens

    @property
    def num_completion_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    @property
    def num_blocks(self) -> int:
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i: int) -> list[int]:
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1


@dataclass
class Block:
    block_id: int
    ref_count: int = 0
    hash_val: int = -1
    token_ids: list[int] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.token_ids is None:
            self.token_ids = []

    def update(self, hash_val: int, token_ids: list[int]) -> None:
        self.hash_val = hash_val
        self.token_ids = list(token_ids)

    def reset(self) -> None:
        self.ref_count = 1
        self.hash_val = -1
        self.token_ids = []


class BlockManager:
    """对齐 nano-vllm/nanovllm/engine/block_manager.py 的链式哈希块管理器。"""

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1) -> int:
        h = hashlib.blake2b(digest_size=8)
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little", signed=False))
        for tok in token_ids:
            h.update(tok.to_bytes(4, "little", signed=True))
        return int.from_bytes(h.digest(), "little", signed=False)

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash_val != -1 and self.hash_to_block_id.get(block.hash_val) == block_id:
            del self.hash_to_block_id[block.hash_val]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int) -> None:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int) -> None:
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for _ in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence) -> None:
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= int(len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence) -> None:
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence) -> None:
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end:
            return
        h = self.blocks[seq.block_table[start - 1]].hash_val if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id


def compute_prefill_slot_mapping(seq: Sequence, block_size: int) -> list[int]:
    """对齐 nano-vllm/nanovllm/engine/model_runner.py 第 151~161 行的预填充槽位映射。"""
    start = seq.num_cached_tokens
    end = start + seq.num_scheduled_tokens
    start_block = start // block_size
    end_block = (end + block_size - 1) // block_size
    slot_mapping: list[int] = []
    for i in range(start_block, end_block):
        slot_start = seq.block_table[i] * block_size
        if i == start_block:
            slot_start += start % block_size
        if i != end_block - 1:
            slot_end = seq.block_table[i] * block_size + block_size
        else:
            slot_end = seq.block_table[i] * block_size + end - i * block_size
        slot_mapping.extend(range(slot_start, slot_end))
    return slot_mapping


def compute_decode_slot(seq: Sequence, block_size: int) -> int:
    """对齐 nano-vllm/nanovllm/engine/model_runner.py 第 181 行的解码单步槽位映射。"""
    return seq.block_table[-1] * block_size + seq.last_block_num_tokens - 1


def run_experiment() -> None:
    print("=== [实验 02] nano-vllm 链式哈希前缀复用、slot_mapping 与显存抢占实验 ===")
    Sequence.block_size = 4
    bm = BlockManager(num_blocks=6, block_size=4)

    # 1. 验证必答题 2：相同块内容但前序块不同，链式哈希值必然不同
    h_prefix_a = bm.compute_hash([10, 11, 12, 13], prefix=-1)
    h_prefix_b = bm.compute_hash([99, 11, 12, 13], prefix=-1)
    h_block1_after_a = bm.compute_hash([20, 21, 22, 23], prefix=h_prefix_a)
    h_block1_after_b = bm.compute_hash([20, 21, 22, 23], prefix=h_prefix_b)
    assert h_block1_after_a != h_block1_after_b
    print("[核验 1] 链式哈希前序隔离性通过：前序块不同则后续同内容块哈希值不同。")

    # 2. 请求 0 预填充（9 个 Token -> 占 3 个块：2 个满块 + 1 个未满块）
    seq0 = Sequence([10, 11, 12, 13, 20, 21, 22, 23, 30], max_tokens=3)
    cached0 = bm.can_allocate(seq0)
    assert cached0 == 0
    bm.allocate(seq0, cached0)
    seq0.num_scheduled_tokens = seq0.num_tokens - seq0.num_cached_tokens
    slots0 = compute_prefill_slot_mapping(seq0, block_size=4)
    bm.hash_blocks(seq0)
    seq0.num_cached_tokens += seq0.num_scheduled_tokens
    seq0.num_scheduled_tokens = 0
    seq0.status = SequenceStatus.RUNNING
    print(f"[核验 2] seq0 预填充完成: block_table={seq0.block_table}, slot_mapping={slots0}")
    assert slots0 == [0, 1, 2, 3, 4, 5, 6, 7, 8]

    # 3. 请求 1 到达，共享前 8 个 Token（2 个完整块），验证前缀命中与 slot_mapping 跳过已缓存块
    seq1 = Sequence([10, 11, 12, 13, 20, 21, 22, 23, 40, 41], max_tokens=3)
    cached1 = bm.can_allocate(seq1)
    assert cached1 == 2, f"Expected 2 cached blocks, got {cached1}"
    bm.allocate(seq1, cached1)
    seq1.num_scheduled_tokens = seq1.num_tokens - seq1.num_cached_tokens
    slots1 = compute_prefill_slot_mapping(seq1, block_size=4)
    print(
        f"[核验 3] seq1 命中 {cached1} 个前缀块: block_table={seq1.block_table}, "
        f"共享块 ref_count={[bm.blocks[b].ref_count for b in seq1.block_table]}, "
        f"仅新 Token 计算 slot_mapping={slots1}"
    )
    assert seq1.block_table[:2] == [0, 1]
    assert bm.blocks[0].ref_count == 2 and bm.blocks[1].ref_count == 2
    assert slots1 == [12, 13]

    # 4. 模拟再加入 seq2 耗尽剩余空闲块，随后在 Decode 跨块边界时触发抢占（preempt）
    seq2 = Sequence([50, 51, 52, 53, 60, 61, 62, 63], max_tokens=3)
    cached2 = bm.can_allocate(seq2)
    bm.allocate(seq2, cached2)
    assert len(bm.free_block_ids) == 0
    print(f"[核验 4] seq2 分配后空闲块耗尽: free_block_ids={list(bm.free_block_ids)}")

    # seq0 连续生成 Token 直到第 13 个 Token 需要新块（len(seq0) % 4 == 1）
    for tok in (31, 32, 33, 34):
        seq0.append_token(tok)
    assert len(seq0) == 13 and len(seq0) % 4 == 1
    assert not bm.can_append(seq0)

    # 触发抢占队尾的 seq2（对应 Scheduler.preempt）
    seq2.status = SequenceStatus.WAITING
    seq2.is_prefill = True
    bm.deallocate(seq2)
    assert bm.can_append(seq0)
    bm.may_append(seq0)
    decode_slot = compute_decode_slot(seq0, block_size=4)
    print(
        f"[核验 5] 显存不足触发抢占 seq2 -> 释放后 free_block_ids={list(bm.free_block_ids)} -> "
        f"seq0 成功追加新块 block_table={seq0.block_table}, 新 Token 写入物理槽位 decode_slot={decode_slot}"
    )
    print("\n[PASS] 实验 02 全部断言核验通过。")


if __name__ == "__main__":
    run_experiment()
