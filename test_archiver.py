"""Regression tests for the blockzip extractor.

Layout used by the crafted cases (BLOCK = 64):

    header = 19 bytes fixed part + name_len bytes

Run with:  python3 -m unittest test_archiver -v
"""

import binascii
import os
import random
import unittest

from archiver import (
    ArchiveCorrupt,
    ArchiveError,
    ArchiveTruncated,
    HEADER,
    METHOD_DEFLATE,
    METHOD_STORED,
    build_archive,
    build_entry,
    extract,
    iter_blocks,
    legacy_extract,
)

BLOCK = 64


def blocks_of(data, size=BLOCK):
    return list(iter_blocks(data, size))


def rand_bytes(rng, n):
    return bytes(rng.getrandbits(8) for _ in range(n))


class ExtractorTestCase(unittest.TestCase):
    def assertEntriesMatch(self, expected, got):
        """Per-entry check: content AND checksum must equal the original."""
        self.assertEqual(len(expected), len(got), "entry count mismatch")
        for name, data in expected:
            self.assertIn(name, got, "missing entry %r" % name)
            self.assertEqual(got[name], data, "content mismatch for %r" % name)
            self.assertEqual(
                binascii.crc32(got[name]) & 0xFFFFFFFF,
                binascii.crc32(data) & 0xFFFFFFFF,
                "crc mismatch for %r" % name,
            )


class ReproductionTests(ExtractorTestCase):
    """Stable reproductions of the production failures on the legacy code,
    plus proof that the fixed extractor handles the same inputs."""

    def test_repro_entry_crossing_block_boundary(self):
        # Layout (stored): a = [0,30), b = [30,150), c = [150,180).
        # Entry b's body crosses the 64- and 128-byte block boundaries.
        entries = [("a", b"A" * 10), ("b", b"B" * 100), ("c", b"C" * 10)]
        archive = build_archive(entries, METHOD_STORED)
        self.assertEqual(len(archive), 180)
        blocks = blocks_of(archive)

        # --- repro on the legacy code: truncated middle entry + missing tail
        legacy = legacy_extract(blocks)
        self.assertEqual(legacy.get("a"), b"A" * 10)
        self.assertNotEqual(legacy.get("b"), b"B" * 100, "repro: truncated content")
        self.assertNotIn("c", legacy, "repro: tail entry lost after desync")

        # --- fixed extractor: exact per-entry match
        self.assertEntriesMatch(entries, extract(blocks))

    def test_entry_exactly_on_block_boundary(self):
        # Each entry is exactly one 64-byte block: 19 + 1 + 44.
        # This is the "rerun sometimes works" case -- even the buggy code
        # is fine when entries happen to align with the read blocks.
        entries = [("a", b"A" * 44), ("b", b"B" * 44), ("c", b"C" * 44)]
        archive = build_archive(entries, METHOD_STORED)
        self.assertEqual(len(archive) % BLOCK, 0)
        blocks = blocks_of(archive)
        self.assertEntriesMatch(entries, legacy_extract(blocks))  # aligned: ok
        self.assertEntriesMatch(entries, extract(blocks))

    def test_repro_repeated_blocks(self):
        # 45-byte names make every header exactly one block, so the two
        # identical 64-byte payloads land on block boundaries:
        #   block0 = header(x)  block1 = payload  block2 = header(y)  block3 = payload
        # block1 == block3 -> the legacy "dedup cache" skips block3.
        name_x, name_y = "x" * 45, "y" * 45
        payload = b"Q" * BLOCK
        entries = [(name_x, payload), (name_y, payload)]
        archive = build_archive(entries, METHOD_STORED)
        blocks = blocks_of(archive)
        self.assertEqual(blocks[1], blocks[3])  # repeated block, by construction

        # --- repro on the legacy code: repeated block skipped, entries wrong
        legacy = legacy_extract(blocks)
        self.assertNotEqual(legacy.get(name_y), payload, "repro: repeated block lost")

        # --- fixed extractor: both entries intact
        self.assertEntriesMatch(entries, extract(blocks))


class RoundTripTests(ExtractorTestCase):
    def test_empty_archive(self):
        self.assertEqual(extract([]), {})
        self.assertEqual(extract(iter_blocks(b"", BLOCK)), {})

    def test_empty_entry(self):
        entries = [("empty", b"")]
        for method in (METHOD_STORED, METHOD_DEFLATE):
            archive = build_archive(entries, method)
            self.assertEntriesMatch(entries, extract(blocks_of(archive)))

    def test_single_byte_entry(self):
        entries = [("one", b"\x00"), ("two", b"\xff")]
        for method in (METHOD_STORED, METHOD_DEFLATE):
            archive = build_archive(entries, method)
            self.assertEntriesMatch(entries, extract(blocks_of(archive)))

    def test_huge_entry(self):
        # 1 MiB of incompressible data crosses thousands of blocks.
        entries = [("small", b"x"), ("huge", os.urandom(1 << 20)), ("tail", b"y" * 7)]
        archive = build_archive(entries, METHOD_DEFLATE)
        self.assertEntriesMatch(entries, extract(blocks_of(archive, 4096)))

    def test_block_size_one(self):
        # Every single byte is its own block: the extreme of misalignment.
        entries = [("a", b"hello"), ("b", b""), ("c", os.urandom(300))]
        archive = build_archive(entries, METHOD_DEFLATE)
        self.assertEntriesMatch(entries, extract(blocks_of(archive, 1)))

    def test_block_larger_than_archive(self):
        entries = [("a", b"data")]
        archive = build_archive(entries, METHOD_DEFLATE)
        self.assertEntriesMatch(entries, extract(blocks_of(archive, 1 << 20)))

    def test_roundtrip_fuzz_many_block_sizes(self):
        rng = random.Random(20260930)
        sizes = [0, 1, 2, 3, 17, 63, 64, 65, 127, 128, 1000, 5000]
        entries = [
            ("e%02d" % i, rand_bytes(rng, rng.choice(sizes)))
            for i in range(40)
        ]
        for method in (METHOD_STORED, METHOD_DEFLATE):
            archive = build_archive(entries, method)
            for block_size in (1, 7, 19, 64, 1000, 65536):
                # pass a lazy generator, like the real transport does
                got = extract(iter_blocks(archive, block_size))
                self.assertEntriesMatch(entries, got)


class CorruptionTests(ExtractorTestCase):
    def _stored_archive(self):
        rng = random.Random(7)
        entries = [("a", rand_bytes(rng, 100)), ("b", rand_bytes(rng, 100))]
        archive = bytearray(build_archive(entries, METHOD_STORED))
        # entry a = [0,120), entry b = [120,240); b's data starts at 140.
        return entries, archive

    def test_corrupt_block_is_reported_with_offset(self):
        entries, archive = self._stored_archive()
        bad_off = 150  # inside entry b's data, inside block at offset 128
        archive[bad_off] ^= 0xFF
        with self.assertRaises(ArchiveCorrupt) as ctx:
            extract(blocks_of(bytes(archive)))
        err = ctx.exception
        # the implicated span covers the corrupted byte...
        self.assertIsNotNone(err.span)
        self.assertLessEqual(err.span[0], bad_off)
        self.assertLess(bad_off, err.span[1])
        # ...and maps back to the block that was corrupted
        self.assertIn(128, err.block_offsets(BLOCK))
        self.assertIn("offset", str(err))

    def test_corrupt_header_reports_exact_offset(self):
        entries, archive = self._stored_archive()
        entry_b_off = 120
        archive[entry_b_off] ^= 0xFF  # smash b's magic
        with self.assertRaises(ArchiveCorrupt) as ctx:
            extract(blocks_of(bytes(archive)))
        self.assertEqual(ctx.exception.offset, entry_b_off)

    def test_consecutive_corrupt_blocks(self):
        entries, archive = self._stored_archive()
        # corrupt two adjacent blocks: one byte in a's data, one in b's data
        archive[70] ^= 0xFF   # block at offset 64, entry a
        archive[150] ^= 0xFF  # block at offset 128, entry b
        with self.assertRaises(ArchiveCorrupt) as ctx:
            extract(blocks_of(bytes(archive)))
        # fails on the FIRST corrupted entry, does not skip ahead or
        # return the good entries as a partial success
        self.assertEqual(ctx.exception.offset, 0)
        self.assertLessEqual(ctx.exception.span[0], 70)
        self.assertLess(70, ctx.exception.span[1])

    def test_no_partial_output_on_corruption(self):
        entries, archive = self._stored_archive()
        archive[150] ^= 0xFF
        result = None
        try:
            result = extract(blocks_of(bytes(archive)))
        except ArchiveError:
            pass
        self.assertIsNone(result, "partial content must not be returned")

    def test_legacy_accepts_corruption_silently(self):
        # Documents why the CRC check matters: the old code returned the
        # corrupted bytes as a successful extraction.
        # Block-aligned entries (64 bytes each) so the legacy parser
        # does not trip over boundaries and the *only* difference is
        # the missing integrity check.
        entries = [("a", b"A" * 44), ("b", b"B" * 44)]
        archive = bytearray(build_archive(entries, METHOD_STORED))
        archive[100] ^= 0xFF  # inside b's payload (block at offset 64)
        legacy = legacy_extract(blocks_of(bytes(archive)))
        self.assertNotEqual(legacy.get("b"), b"B" * 44)
        with self.assertRaises(ArchiveCorrupt):
            extract(blocks_of(bytes(archive)))

    def test_truncated_archive_raises(self):
        entries, archive = self._stored_archive()
        for cut in (1, 17, 100):  # tail missing: mid-data, mid-header
            with self.assertRaises(ArchiveTruncated):
                extract(blocks_of(bytes(archive[:-cut])))

    def test_trailing_garbage_rejected(self):
        entries, archive = self._stored_archive()
        garbage = bytes([0xDE, 0xAD, 0xBE, 0xEF]) * 8
        with self.assertRaises(ArchiveCorrupt):
            extract(blocks_of(bytes(archive) + garbage))


if __name__ == "__main__":
    unittest.main()
