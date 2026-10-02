"""稳定复现旧版解压器的三类线上事故。

同一批输入在修复版 ``miniarc.unpack`` 上必须全部正常（对照验证），
而旧版 ``legacy_unpack`` 必须以「缺尾 / 错位 / 吞损坏」的形式出错。
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers import (  # noqa: E402
    AlignedReader,
    ArchiveLayout,
    build_archive,
)
from miniarc.archive import unpack  # noqa: E402
from miniarc.legacy import legacy_unpack  # noqa: E402


class LegacyReproTest(unittest.TestCase):
    def _fixed_is_correct(self, layout: ArchiveLayout, chunk: int) -> None:
        # 修复版在同样的分块调度下必须逐块对拍通过。
        fixed = list(unpack(layout.reader(chunk), chunk_size=chunk))
        self.assertEqual([(e.name, e.data) for e in fixed], layout.entries)

    def test_missing_tail_is_reproduced(self) -> None:
        """缺尾：第二个条目连同末尾数据被静默丢弃，不报错。"""
        layout = build_archive([
            ("first.txt", b"A" * 10),
            ("second.txt", b"B" * 10),
        ])
        chunk = 64  # 旧版一次读块把第二条目的头/载荷吞进 zlib 尾字节
        legacy = legacy_unpack(layout.reader(chunk), chunk_size=chunk)

        self.assertNotEqual(legacy, layout.entries)
        self.assertLess(len(legacy), len(layout.entries),
                        "旧版必须丢失末尾条目（缺尾复现）")
        self.assertNotIn("second.txt", [name for name, _ in legacy])
        # 对照：修复版两个条目都完整。
        self._fixed_is_correct(layout, chunk)

    def test_misaligned_entry_content_is_wrong(self) -> None:
        """错位/缺截：条目载荷跨块后被截断为空，旧版照样报成功。"""
        layout = build_archive([
            ("cross.dat", bytes(range(256)) * 2),
        ])
        chunk = 50  # 压缩载荷明显跨块
        legacy = legacy_unpack(layout.reader(chunk), chunk_size=chunk)

        self.assertEqual(len(legacy), 1)
        name, data = legacy[0]
        self.assertEqual(name, "cross.dat")
        self.assertNotEqual(data, layout.entries[0][1],
                            "旧版必须输出错误/截断内容（错位复现）")
        self.assertNotIn(b"Z", data)  # 空或半截，绝不是完整数据
        self._fixed_is_correct(layout, chunk)

    def test_all_fixed_chunk_schedules_fail(self) -> None:
        """扫描 1..96 的读块大小：旧版没有任何一种固定调度能解对。"""
        layout = build_archive([
            ("a", b"hello"),
            ("b", b"world!" * 7),
            ("c", b"x" * 500),
            ("d", b""),
            ("e", b"\x00"),
        ])
        for chunk in range(1, 97):
            with self.subTest(chunk=chunk):
                legacy = legacy_unpack(layout.reader(chunk), chunk_size=chunk)
                self.assertNotEqual(
                    legacy, layout.entries,
                    f"chunk={chunk} 时旧版意外解对，复现就不稳定了",
                )
        # 修复版所有调度逐块对拍通过。
        for chunk in range(1, 97):
            with self.subTest(fixed_chunk=chunk):
                got = list(unpack(layout.reader(chunk), chunk_size=chunk))
                self.assertEqual(
                    [(e.name, e.data) for e in got], layout.entries
                )

    def test_aligned_reads_appear_fine_rerun_flake(self) -> None:
        """对齐读法下旧版「重跑又正常」：问题表现依赖读块调度。"""
        layout = build_archive([
            ("a", b"hello"),
            ("b", b"world!" * 7),
            ("c", b"x" * 500),
            ("d", b""),
            ("e", b"\x00"),
        ])
        aligned = AlignedReader(layout.blob, layout.entry_ends)
        legacy = legacy_unpack(aligned, chunk_size=4096)
        self.assertEqual(legacy, layout.entries,
                         "对齐读法下旧版恰好正常（线上偶发的来源）")

    def test_corruption_is_silently_accepted(self) -> None:
        """吞损坏：CRC 被改坏，旧版仍把条目当成功返回、不报告偏移。"""
        layout = build_archive([("a", b"important data")])
        # 头部最后 4 字节是 crc32 的低 32 位，翻掉一个字节。
        bad = bytearray(layout.blob)
        bad[22] ^= 0xFF  # 26 字节头中 crc32 的最低有效字节
        bad = bytes(bad)

        stream = AlignedReader(bad, [len(bad)])
        legacy = legacy_unpack(stream, chunk_size=4096)
        self.assertEqual(len(legacy), 1, "旧版不会拒绝损坏条目")
        self.assertEqual(legacy[0][0], "a")
        self.assertEqual(legacy[0][1], b"important data")

        # 修复版必须拒绝并给出条目头偏移 0。
        from miniarc.archive import CorruptArchiveError

        with self.assertRaises(CorruptArchiveError) as ctx:
            list(unpack(bad, chunk_size=37))
        self.assertEqual(ctx.exception.offset, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
