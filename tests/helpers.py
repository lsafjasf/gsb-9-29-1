"""测试辅助：构造归档、模拟各种读块调度、逐块对拍。"""

from __future__ import annotations

import io
import os
import sys
import zlib
from typing import List, Sequence, Tuple, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miniarc.archive import HEADER_SIZE, Packer  # noqa: E402

Schedule = Union[int, Sequence[int]]


class ChunkedReader:
    """模拟「按固定块大小读取」的底层流。

    - schedule 为 int：每次 read 最多返回该字节数（读块边界与压缩块
      边界不对齐的典型情形）。
    - schedule 为序列：第 i 次 read 最多返回 schedule[i % len] 字节，
      用于构造「恰好落在块边界」等精确调度。
    """

    def __init__(self, data: bytes, schedule: Schedule):
        self._data = data
        if isinstance(schedule, int):
            schedule = (schedule,)
        if not schedule or any(c <= 0 for c in schedule):
            raise ValueError("schedule chunks must be positive")
        self._schedule = tuple(schedule)
        self._pos = 0
        self._reads = 0

    def read(self, n: int = -1) -> bytes:
        cap = self._schedule[self._reads % len(self._schedule)]
        self._reads += 1
        if n is None or n < 0 or n > cap:
            n = cap
        chunk = self._data[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk


class AlignedReader:
    """每次 read 都在条目边界处截断的「幸运」读法。

    旧实现只有在这种读块恰好与条目边界对齐时才表现正常——
    对应线上「重跑有时又正常」。
    """

    def __init__(self, data: bytes, entry_ends: Sequence[int]):
        self._data = data
        self._ends = sorted(entry_ends)
        self._pos = 0

    def read(self, n: int = -1) -> bytes:
        end = self._ends[-1] if self._ends else len(self._data)
        for boundary in self._ends:
            if boundary > self._pos:
                end = boundary
                break
        if n is None or n < 0:
            n = end - self._pos
        else:
            n = min(n, end - self._pos)
        chunk = self._data[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk


class ArchiveLayout:
    """归档字节 + 每个条目的头/载荷偏移，用于边界与损坏注入。"""

    def __init__(self, blob: bytes, entries: List[Tuple[str, bytes]],
                 entry_starts: List[int]):
        self.blob = blob
        self.entries = entries
        self.entry_starts = entry_starts
        self.entry_ends = entry_starts[1:] + [len(blob)]
        self.payload_starts = [
            start + HEADER_SIZE + len(name.encode("utf-8"))
            for start, (name, _) in zip(entry_starts, entries)
        ]

    def reader(self, schedule: Schedule) -> ChunkedReader:
        return ChunkedReader(self.blob, schedule)


def build_archive(entries: List[Tuple[str, bytes]]) -> ArchiveLayout:
    buf = io.BytesIO()
    packer = Packer(buf)
    starts = []
    for name, data in entries:
        starts.append(packer.add_entry(name, data))
    packer.close()
    return ArchiveLayout(buf.getvalue(), entries, starts)


def crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def assert_block_by_block(testcase, got_entries, expected_entries) -> None:
    """逐块对拍：每个条目的名称、内容与 CRC32 都必须与原始数据一致。"""
    got = list(got_entries)
    testcase.assertEqual(
        [e.name for e in got], [name for name, _ in expected_entries],
        "entry names/order mismatch",
    )
    testcase.assertEqual(len(got), len(expected_entries), "entry count mismatch")
    for entry, (name, data) in zip(got, expected_entries):
        testcase.assertEqual(entry.data, data, f"content mismatch for {name!r}")
        testcase.assertEqual(
            crc32(entry.data), crc32(data), f"crc32 mismatch for {name!r}"
        )
