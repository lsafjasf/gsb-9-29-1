"""修复版回归测试。

覆盖：
- 跨块条目、恰好落在块边界的条目、重复出现的块
- 空条目、单字节条目、超长条目、大量微小条目
- 逐块对拍（内容 + CRC32），读块大小 1..96 全扫描 + 随机调度
- 损坏块识别与偏移上报（载荷/CRC/长度/magic/截断/连续损坏）
- 任何损坏都不输出部分内容
- 单字节翻转 fuzz：除名称区域外，任何字节损坏都必须被检出
"""

from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers import (  # noqa: E402
    assert_block_by_block,
    build_archive,
)
from miniarc.archive import (  # noqa: E402
    CorruptArchiveError,
    pack,
    unpack,
)

SAMPLE_ENTRIES = [
    ("empty", b""),
    ("one", b"Q"),
    ("hello.txt", b"hello world"),
    ("random-256", bytes((i * 37 + 11) & 0xFF for i in range(256))),
    ("repeated", b"ABCABCD" * 1000),
    ("zeros", b"\x00" * 5000),
    ("tail", b"last entry"),
]


class BlockByBlockComparisonTest(unittest.TestCase):
    """跨块条目 + 逐块对拍：不同读块大小下内容与校验值必须一致。"""

    def test_cross_chunk_entries_every_chunk_size(self) -> None:
        layout = build_archive(SAMPLE_ENTRIES)
        for chunk in range(1, 97):
            with self.subTest(chunk=chunk):
                got = list(unpack(layout.reader(chunk), chunk_size=chunk))
                assert_block_by_block(self, got, layout.entries)

    def test_large_chunk_sizes(self) -> None:
        layout = build_archive(SAMPLE_ENTRIES)
        for chunk in [128, 256, 1024, 4096, 1 << 20]:
            with self.subTest(chunk=chunk):
                got = list(unpack(layout.reader(chunk), chunk_size=chunk))
                assert_block_by_block(self, got, layout.entries)

    def test_bytes_input_and_custom_schedule(self) -> None:
        layout = build_archive(SAMPLE_ENTRIES)
        got = list(unpack(layout.blob, chunk_size=128))  # 直接喂 bytes
        assert_block_by_block(self, got, layout.entries)
        # 不规则读块调度：3,1,7,2,...
        got = list(unpack(layout.reader([3, 1, 7, 2, 5, 13]), chunk_size=13))
        assert_block_by_block(self, got, layout.entries)


class BoundaryTest(unittest.TestCase):
    """条目恰好落在读块边界上的用例。"""

    def test_entry_start_exactly_on_chunk_boundary(self) -> None:
        entries = [
            ("first", b"a" * 40),
            ("exact", b"boundary entry"),
            ("third", b"c" * 100),
        ]
        layout = build_archive(entries)
        boundary = layout.entry_starts[1]
        # 第二条目头恰好是一个读块的第一个字节（上一块恰好用完）。
        schedule = [boundary] + [37] * 1000
        got = list(unpack(layout.reader(schedule), chunk_size=37))
        assert_block_by_block(self, got, entries)

    def test_entry_payload_exactly_fills_blocks(self) -> None:
        # 第二条目压缩后恰好一个完整读块（先探测再选块大小）。
        entries = [("a", b"x" * 30), ("b", b"y" * 600), ("c", b"z" * 3)]
        layout = build_archive(entries)
        payload_end = layout.entry_ends[1]
        payload_len = payload_end - layout.payload_starts[1]
        schedule = [layout.payload_starts[1], payload_len, 999]
        got = list(unpack(layout.reader(schedule), chunk_size=999))
        assert_block_by_block(self, got, entries)

    def test_whole_archive_in_single_read(self) -> None:
        layout = build_archive(SAMPLE_ENTRIES)
        got = list(unpack(layout.reader(len(layout.blob)), chunk_size=len(layout.blob)))
        assert_block_by_block(self, got, layout.entries)

    def test_archive_shorter_than_chunk(self) -> None:
        layout = build_archive([("tiny", b"")])
        got = list(unpack(layout.reader(4096), chunk_size=4096))
        assert_block_by_block(self, got, layout.entries)


class RepeatedBlocksTest(unittest.TestCase):
    """重复出现的块：相同内容 / 相同压缩字节必须逐条区分、不得串行。"""

    def test_identical_payloads_repeated(self) -> None:
        payload = bytes(range(256))
        entries = [(f"dup-{i:02d}", payload) for i in range(20)]
        layout = build_archive(entries)
        # 所有条目压缩字节完全相同，块边界在重复模式上滑动。
        for chunk in [7, 31, 64, 100, 127, 256]:
            with self.subTest(chunk=chunk):
                got = list(unpack(layout.reader(chunk), chunk_size=chunk))
                assert_block_by_block(self, got, entries)

    def test_alternating_tiny_entries(self) -> None:
        entries = []
        for i in range(300):
            entries.append((f"e{i}", b"" if i % 2 == 0 else bytes([i & 0xFF])))
        layout = build_archive(entries)
        got = list(unpack(layout.reader(9), chunk_size=9))
        assert_block_by_block(self, got, entries)


class EdgeCaseTest(unittest.TestCase):
    def test_empty_archive(self) -> None:
        self.assertEqual(list(unpack(pack([]), chunk_size=1)), [])

    def test_empty_entry(self) -> None:
        layout = build_archive([("empty", b"")])
        for chunk in [1, 8, 4096]:
            with self.subTest(chunk=chunk):
                got = list(unpack(layout.reader(chunk), chunk_size=chunk))
                assert_block_by_block(self, got, [("empty", b"")])

    def test_single_byte_entry(self) -> None:
        layout = build_archive([("byte", b"\xff")])
        for chunk in [1, 2, 3, 4096]:
            with self.subTest(chunk=chunk):
                got = list(unpack(layout.reader(chunk), chunk_size=chunk))
                assert_block_by_block(self, got, [("byte", b"\xff")])

    def test_very_long_entry(self) -> None:
        # 超长条目：1 MiB 随机数据（不可压缩），远大于任何读块。
        rng = random.Random(0xC0FFEE)
        big = bytes(rng.randrange(256) for _ in range(1 << 20))
        entries = [("big", big), ("after", b"still aligned")]
        layout = build_archive(entries)
        for chunk in [7, 4096, 65537]:
            with self.subTest(chunk=chunk):
                got = list(unpack(layout.reader(chunk), chunk_size=chunk))
                assert_block_by_block(self, got, entries)

    def test_long_highly_compressible_entry(self) -> None:
        entries = [("z", b"Z" * (1 << 18))]
        layout = build_archive(entries)
        got = list(unpack(layout.reader(33), chunk_size=33))
        assert_block_by_block(self, got, entries)

    def test_randomized_schedules_seeded(self) -> None:
        rng = random.Random(20261002)
        entries = [
            ("rand-a", bytes(rng.randrange(256) for _ in range(rng.randrange(0, 2000)))),
            ("rand-b", bytes(rng.randrange(256) for _ in range(rng.randrange(0, 200)))),
            ("rand-c", b""),
            ("rand-d", bytes(rng.randrange(256) for _ in range(rng.randrange(0, 3000)))),
        ]
        layout = build_archive(entries)
        for seed in range(40):
            local = random.Random(seed)
            schedule = [local.randrange(1, 300) for _ in range(64)]
            with self.subTest(seed=seed):
                got = list(
                    unpack(layout.reader(schedule), chunk_size=max(schedule))
                )
                assert_block_by_block(self, got, entries)


class CorruptionTest(unittest.TestCase):
    def setUp(self) -> None:
        # 三个条目，便于验证「损坏条目不产出、前置条目完好、偏移准确」。
        self.layout = build_archive([
            ("good-1", b"first entry content"),
            ("bad-one", b"this payload will be damaged"),
            ("good-2", b"third entry content"),
        ])
        self.e1 = self.layout.entry_starts[1]
        self.p1 = self.layout.payload_starts[1]

    def _flip(self, offset: int, delta: int = 0xFF) -> bytes:
        bad = bytearray(self.layout.blob)
        bad[offset] ^= delta
        return bytes(bad)

    def test_corrupt_payload_reports_payload_offset(self) -> None:
        # 翻转载荷中段的一个字节。
        bad = self._flip(self.p1 + 5)
        with self.assertRaises(CorruptArchiveError) as ctx:
            list(unpack(bad, chunk_size=11))
        self.assertEqual(ctx.exception.offset, self.p1)
        self.assertIn("0x%x" % self.p1, str(ctx.exception))

    def test_corrupt_crc_field_reports_entry_offset(self) -> None:
        # crc32 位于头的最后 8 字节（offset 18..25，大端）。
        bad = self._flip(self.e1 + 22)
        with self.assertRaises(CorruptArchiveError) as ctx:
            list(unpack(bad, chunk_size=64))
        self.assertEqual(ctx.exception.offset, self.e1)

    def test_corrupt_declared_length_reports_entry_offset(self) -> None:
        # dlen 位于头偏移 10..17，翻最低位让长度 +1。
        bad = self._flip(self.e1 + 17, 0x01)
        with self.assertRaises(CorruptArchiveError) as ctx:
            list(unpack(bad, chunk_size=5))
        self.assertEqual(ctx.exception.offset, self.e1)

    def test_corrupt_magic_reports_entry_offset(self) -> None:
        bad = self._flip(self.e1)
        with self.assertRaises(CorruptArchiveError) as ctx:
            list(unpack(bad, chunk_size=13))
        self.assertEqual(ctx.exception.offset, self.e1)

    def test_truncated_in_header_and_payload(self) -> None:
        for cut, expected_offset in [
            (self.e1 + 10, self.e1),       # 头只读到一半
            (self.p1 + 1, self.p1),        # 载荷被截断
            (self.layout.entry_starts[2] + 1, self.layout.entry_starts[2]),
            (len(self.layout.blob) - 1, self.layout.payload_starts[2]),
        ]:
            with self.subTest(cut=cut):
                with self.assertRaises(CorruptArchiveError) as ctx:
                    list(unpack(self.layout.blob[:cut], chunk_size=17))
                self.assertEqual(ctx.exception.offset, expected_offset)

    def test_consecutive_corrupt_entries(self) -> None:
        # 连续损坏：第二、第三条目都翻字节。
        p2 = self.layout.payload_starts[2]
        bad = bytearray(self.layout.blob)
        bad[self.p1 + 2] ^= 0xFF
        bad[p2 + 2] ^= 0xFF
        bad = bytes(bad)

        yielded = []
        with self.assertRaises(CorruptArchiveError) as ctx:
            for entry in unpack(bad, chunk_size=29):
                yielded.append(entry)
        # 第一处损坏（第二个条目）即中止：报它的偏移。
        self.assertEqual(ctx.exception.offset, self.p1)
        # 只有第一个完好条目被产出；两个损坏条目都不作为成功输出。
        self.assertEqual([(e.name, e.data) for e in yielded],
                         [self.layout.entries[0]])

    def test_no_partial_content_emitted(self) -> None:
        # 把坏条目的载荷截掉一半：解压器在确认完整性前不得 yield。
        cut = self.layout.entry_ends[1] - 1
        yielded = []
        with self.assertRaises(CorruptArchiveError):
            for entry in unpack(self.layout.blob[:cut], chunk_size=4):
                yielded.append(entry)
        names = [e.name for e in yielded]
        self.assertNotIn("bad-one", names)
        self.assertNotIn("good-2", names)
        self.assertEqual(names, ["good-1"])


class FuzzCorruptionTest(unittest.TestCase):
    def test_single_byte_flip_every_offset_is_detected(self) -> None:
        layout = build_archive([
            ("k", b"short"),
            ("m", bytes(range(100))),
        ])
        # 名称区域不在 CRC 保护范围内（与 zip 等格式一致），fuzz 时排除；
        # 其余所有偏移（头字段 + 压缩载荷）逐一翻转都必须被检出。
        name_ranges = []
        for start, (name, _) in zip(layout.entry_starts, layout.entries):
            n = len(name.encode())
            name_ranges.append((start + 26, start + 26 + n))

        def in_name(off: int) -> bool:
            return any(lo <= off < hi for lo, hi in name_ranges)

        checked = 0
        for off in range(len(layout.blob)):
            if in_name(off):
                continue
            bad = bytearray(layout.blob)
            bad[off] ^= 0x5A
            checked += 1
            try:
                got = list(unpack(bytes(bad), chunk_size=23))
            except CorruptArchiveError:
                continue
            # 未报错的唯一合法情况：输出与原始数据完全一致。
            self.assertEqual(
                [(e.name, e.data) for e in got],
                layout.entries,
                f"offset {off} 的单字节损坏被静默接受",
            )
        self.assertGreater(checked, 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
