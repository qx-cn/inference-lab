#!/usr/bin/env python3
"""
实验 03：mini-sglang 基数树前缀缓存（RadixPrefixCache）节点分裂、引用计数保护与最小堆 LRU 驱逐
对应计划：第 3 周（精确对齐 mini-sglang/python/minisgl/kvcache/radix_cache.py）
运行方式：先复制整个实验目录，再在副本目录运行 python3 lab03_minisgl_radix_tree_cache.py（见 README.md）。
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import time
from typing import Callable


def align_down(val: int, alignment: int) -> int:
    return (val // alignment) * alignment


class RadixTreeNode:
    """对齐 mini-sglang/python/minisgl/kvcache/radix_cache.py 中的 RadixTreeNode。"""

    counter: int = 0

    def __init__(self, key_fn: Callable[[ tuple[int, ...] ], tuple[int, ...]], tic: int | None = None) -> None:
        self.key_fn = key_fn
        self.children: dict[tuple[int, ...], RadixTreeNode] = {}
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0
        self.uuid = RadixTreeNode.counter
        RadixTreeNode.counter += 1
        self.timestamp = tic if tic is not None else time.monotonic_ns()
        self._key: tuple[int, ...] = ()
        self._value: tuple[int, ...] = ()
        self._length: int = 0

    def set_key_value(self, key: tuple[int, ...], value: tuple[int, ...]) -> None:
        assert len(key) == len(value)
        self._key = tuple(key)
        self._value = tuple(value)
        self._length = len(key)

    def set_parent(self, parent: "RadixTreeNode") -> None:
        self._parent = parent
        parent.children[self.key_fn(self._key)] = self

    @property
    def length(self) -> int:
        return self._length

    @property
    def parent(self) -> "RadixTreeNode":
        assert self._parent is not None
        return self._parent

    @property
    def value(self) -> tuple[int, ...]:
        return self._value

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_match_len(self, input_ids: tuple[int, ...]) -> int:
        limit = min(len(self._key), len(input_ids))
        for i in range(limit):
            if self._key[i] != input_ids[i]:
                return i
        return limit

    def split_at(self, pos: int) -> "RadixTreeNode":
        assert 0 < pos < self.length
        parent = self.parent
        new_node = RadixTreeNode(self.key_fn, self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count

        self.set_key_value(self._key[pos:], self._value[pos:])
        self.set_parent(new_node)
        return new_node

    def __lt__(self, other: "RadixTreeNode") -> bool:
        return self.timestamp < other.timestamp


@dataclass(frozen=True)
class RadixCacheHandle:
    cached_len: int
    node: RadixTreeNode


class RadixPrefixCache:
    """对齐 mini-sglang/python/minisgl/kvcache/radix_cache.py 中的 RadixPrefixCache。"""

    def __init__(self, page_size: int = 1) -> None:
        self.page_size = page_size
        self.key_fn = lambda x: tuple(x[:page_size])
        self.evictable_size = 0
        self.protected_size = 0
        self.root_node = RadixTreeNode(self.key_fn)
        self.root_node.ref_count = 1

    def lock_handle(self, handle: RadixCacheHandle, unlock: bool = False) -> None:
        node = handle.node
        if unlock:
            while not node.is_root():
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                node = node.parent
        else:
            while not node.is_root():
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
                node = node.parent

    def match_prefix(self, input_ids: tuple[int, ...]) -> RadixCacheHandle:
        node, prefix_len = self._tree_walk(input_ids)
        return RadixCacheHandle(prefix_len, node)

    def insert_prefix(self, input_ids: tuple[int, ...], indices: tuple[int, ...]) -> tuple[int, RadixCacheHandle]:
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, indices = input_ids[:insert_len], indices[:insert_len]
        node, prefix_len = self._tree_walk(input_ids)
        if prefix_len != insert_len:
            new_node = RadixTreeNode(self.key_fn)
            new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:])
            new_node.set_parent(node)
            self.evictable_size += new_node.length
            node = new_node
        return prefix_len, RadixCacheHandle(insert_len, node)

    def evict(self, size: int) -> tuple[int, ...]:
        if size == 0:
            return ()
        assert size <= self.evictable_size
        leave_nodes = self._collect_leave_nodes_for_evict()
        heapq.heapify(leave_nodes)
        evicted_indices: list[int] = []
        evicted_size = 0

        while evicted_size < size:
            node = heapq.heappop(leave_nodes)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            evicted_size += node.length
            evicted_indices.extend(node.value)
            self.evictable_size -= node.length
            parent = node.parent
            del parent.children[self.key_fn(node._key)]
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent)
        return tuple(evicted_indices)

    def _collect_leave_nodes_for_evict(self) -> list[RadixTreeNode]:
        nodes = [self.root_node]
        leave_nodes: list[RadixTreeNode] = []
        while nodes:
            node = nodes.pop()
            if node.is_leaf():
                if node.ref_count == 0:
                    leave_nodes.append(node)
            else:
                for child in node.children.values():
                    nodes.append(child)
        return leave_nodes

    def _tree_walk(self, input_ids: tuple[int, ...]) -> tuple[RadixTreeNode, int]:
        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns()
        while prefix_len < indice_len:
            child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child_node is None:
                return node, prefix_len
            node = child_node
            match_len = align_down(node.get_match_len(input_ids[prefix_len:]), self.page_size)
            prefix_len += match_len
            if match_len != node.length:
                node = node.split_at(match_len)
                node.timestamp = tic
                return node, prefix_len
            node.timestamp = tic
        return node, prefix_len

    def dump_tree(self, node: RadixTreeNode | None = None, indent: int = 0) -> None:
        if node is None:
            node = self.root_node
        label = "ROOT" if node.is_root() else "".join(chr(c) for c in node._key)
        print(f"{'  ' * indent}- [{label}] (len={node.length}, ref={node.ref_count}, leaf={node.is_leaf()})")
        for child in node.children.values():
            self.dump_tree(child, indent + 1)


def to_ascii_tuple(text: str) -> tuple[int, ...]:
    return tuple(ord(c) for c in text)


def run_experiment() -> None:
    print("=== [实验 03] mini-sglang RadixPrefixCache 节点分裂与 LRU 驱逐实验 ===")
    cache = RadixPrefixCache(page_size=1)

    # 1. 插入 "Hello world"
    seq1 = to_ascii_tuple("Hello world")
    _, h1 = cache.insert_prefix(seq1, tuple(range(100, 100 + len(seq1))))
    print("[步骤 1] 插入 'Hello world' 后树结构：")
    cache.dump_tree()
    assert cache.evictable_size == 11 and cache.protected_size == 0

    # 2. 插入 "Hellofriend" -> 触发在 "Hello" (pos=5) 处调用 split_at 分裂节点
    time.sleep(0.001)
    seq2 = to_ascii_tuple("Hellofriend")
    matched_len2, h2 = cache.insert_prefix(seq2, tuple(range(200, 200 + len(seq2))))
    assert matched_len2 == 5
    print("\n[步骤 2] 插入 'Hellofriend' 触发 split_at(5) 后树结构：")
    cache.dump_tree()

    # 3. 插入 "Hello world!" 并给 "Hello world!" 加锁（模拟正在运行中的请求）
    time.sleep(0.001)
    seq3 = to_ascii_tuple("Hello world!")
    matched_len3, h3 = cache.insert_prefix(seq3, tuple(range(300, 300 + len(seq3))))
    assert matched_len3 == 11
    cache.lock_handle(h3, unlock=False)
    print("\n[步骤 3] 插入并锁定 'Hello world!' 后树结构（祖先链 ref_count 全部 +1）：")
    cache.dump_tree()
    assert cache.protected_size == 12
    assert cache.evictable_size == 6  # 仅有 'friend' 分支 (6 个字符) 可被驱逐

    # 4. 驱逐 6 个槽位 -> 只能驱逐 ref_count == 0 的叶子节点 'friend'
    evicted = cache.evict(6)
    assert len(evicted) == 6
    print(f"\n[步骤 4] 请求驱逐 6 个槽位 -> 成功驱逐 'friend' 分支，剩余 evictable_size={cache.evictable_size}：")
    cache.dump_tree()

    # 5. 解锁 'Hello world!' 后再驱逐全部剩余缓存（验证子叶驱逐后父节点级联入堆）
    cache.lock_handle(h3, unlock=True)
    assert cache.protected_size == 0 and cache.evictable_size == 12
    evicted_all = cache.evict(12)
    assert len(evicted_all) == 12 and cache.evictable_size == 0
    print("\n[步骤 5] 解锁 'Hello world!' 并级联驱逐全部 12 个槽位后树结构：")
    cache.dump_tree()
    print("\n[PASS] 实验 03 全部断言核验通过。")


if __name__ == "__main__":
    run_experiment()
