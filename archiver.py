"""blockzip -- streaming extractor for archives read in fixed-size blocks.

Archive format (little-endian), entries concatenated back to back:

    magic      4 bytes   b"BKZ1"
    method     1 byte    0 = stored, 1 = deflate
    name_len   2 bytes   uint16, length of the UTF-8 name
    raw_size   4 bytes   uint32, uncompressed size
    crc32      4 bytes   uint32, CRC-32 of the uncompressed data
    comp_size  4 bytes   uint32, length of comp_data
    name       name_len bytes
    comp_data  comp_size bytes

The archive is *read* in fixed-size blocks (block device / chunked
transport).  Block boundaries are a transport artifact: they may fall
anywhere inside a header or an entry body, and identical block contents
may appear more than once in the stream.

``extract()`` is the fixed implementation.  ``legacy_extract()`` is the
old buggy version, kept only so the regression tests can reproduce the
original production failures (truncated tail, shifted entries, silently
skipped repeated blocks).  Do not use it for anything else.
"""

from __future__ import annotations

import binascii
import hashlib
import struct
import zlib

MAGIC = b"BKZ1"
METHOD_STORED = 0
METHOD_DEFLATE = 1

HEADER = struct.Struct("<4sBHIII")
DEFAULT_BLOCK_SIZE = 64 * 1024


# ------------------------------------------------------------------ errors

class ArchiveError(Exception):
    """Base class for all archive failures."""


class ArchiveTruncated(ArchiveError):
    """The stream ended in the middle of a structure."""

    def __init__(self, offset, needed, available):
        self.offset = offset
        self.needed = needed
        self.available = available
        super().__init__(
            "truncated archive at offset %d: need %d bytes, only %d left"
            % (offset, needed, available)
        )


class ArchiveCorrupt(ArchiveError):
    """A structure or checksum failed validation.

    ``offset`` is the stream offset where the problem was detected.
    ``span`` (when known) is the [start, end) byte range the corrupted
    bytes must lie in, so the failure can be mapped back to the
    fixed-size block(s) that caused it.
    """

    def __init__(self, offset, reason, span=None):
        self.offset = offset
        self.reason = reason
        self.span = span
        super().__init__("corrupt archive at offset %d: %s" % (offset, reason))

    def block_offsets(self, block_size):
        """Offsets of the fixed-size blocks implicated in this failure."""
        start, end = self.span if self.span else (self.offset, self.offset + 1)
        first = start // block_size
        last = max(start, end - 1) // block_size
        return [i * block_size for i in range(first, last + 1)]


# ------------------------------------------------------------------ builder

def build_entry(name, data, method=METHOD_DEFLATE):
    raw = bytes(data)
    name_bytes = name.encode("utf-8")
    if method == METHOD_DEFLATE:
        comp = zlib.compress(raw, 9)
    elif method == METHOD_STORED:
        comp = raw
    else:
        raise ValueError("unknown method %r" % (method,))
    crc = binascii.crc32(raw) & 0xFFFFFFFF
    header = HEADER.pack(MAGIC, method, len(name_bytes), len(raw), crc, len(comp))
    return header + name_bytes + comp


def build_archive(entries, method=METHOD_DEFLATE):
    return b"".join(build_entry(name, data, method) for name, data in entries)


def iter_blocks(data, block_size=DEFAULT_BLOCK_SIZE):
    """Yield the archive the way the transport delivers it: fixed-size blocks."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    for off in range(0, len(data), block_size):
        yield data[off:off + block_size]


# ------------------------------------------------------------------ reader

class StreamReader:
    """Exact-size reads over a sequence of arbitrary-size blocks.

    This is the core of the fix: parsing never happens "per block".
    A logical cursor is maintained across block boundaries, and every
    read either returns exactly the requested number of bytes or raises
    ``ArchiveTruncated`` -- a short block is never mistaken for EOF, and
    a partial structure is never silently dropped.
    """

    def __init__(self, blocks):
        self._it = iter(blocks)
        self._buf = bytearray()
        self.pos = 0  # absolute stream offset of the next unread byte
        self._exhausted = False

    def _fill(self):
        if self._exhausted:
            return False
        try:
            block = next(self._it)
        except StopIteration:
            self._exhausted = True
            return False
        self._buf += bytes(block)
        return True

    def has_more(self):
        while not self._buf and not self._exhausted:
            self._fill()
        return bool(self._buf)

    def read_exact(self, n):
        start = self.pos
        while len(self._buf) < n:
            if not self._fill():
                raise ArchiveTruncated(start, n, len(self._buf))
        out = bytes(self._buf[:n])
        del self._buf[:n]
        self.pos += n
        return out


def _inflate(method, comp, offset):
    if method == METHOD_STORED:
        return bytes(comp)
    if method == METHOD_DEFLATE:
        decomp = zlib.decompressobj()
        try:
            out = decomp.decompress(comp) + decomp.flush()
        except zlib.error as exc:
            raise ArchiveCorrupt(offset, "deflate stream error: %s" % exc) from exc
        if not decomp.eof:
            raise ArchiveCorrupt(offset, "deflate stream truncated")
        if decomp.unused_data:
            raise ArchiveCorrupt(offset, "trailing garbage after deflate stream")
        return out
    raise ArchiveCorrupt(offset, "unknown method %d" % method)


# ------------------------------------------------------------------ extractor

def extract(blocks):
    """Extract an archive delivered as an iterable of fixed-size blocks.

    Returns ``{name: data}``.  The function is all-or-nothing: any
    truncation or corruption raises ``ArchiveError`` (carrying the
    stream offset) and nothing is returned, so partial content can never
    be mistaken for a successful extraction.
    """
    reader = StreamReader(blocks)
    out = {}
    while reader.has_more():
        entry_off = reader.pos
        header = reader.read_exact(HEADER.size)
        magic, method, name_len, raw_size, crc, comp_size = HEADER.unpack(header)
        if magic != MAGIC:
            raise ArchiveCorrupt(entry_off, "bad magic %r" % (magic,))
        if method not in (METHOD_STORED, METHOD_DEFLATE):
            raise ArchiveCorrupt(entry_off, "unknown method %d" % method)
        name = reader.read_exact(name_len)
        data_off = reader.pos
        comp = reader.read_exact(comp_size)
        data = _inflate(method, comp, data_off)
        end_off = reader.pos
        if len(data) != raw_size:
            raise ArchiveCorrupt(
                entry_off,
                "size mismatch: header says %d, decoded %d" % (raw_size, len(data)),
                span=(entry_off, end_off),
            )
        if (binascii.crc32(data) & 0xFFFFFFFF) != crc:
            raise ArchiveCorrupt(
                entry_off,
                "CRC mismatch for entry %r" % (name,),
                span=(entry_off, end_off),
            )
        out[name.decode("utf-8")] = data
    return out


# ------------------------------------------------------------------ legacy (buggy)

def _legacy_inflate(method, comp):
    if method == METHOD_STORED:
        return bytes(comp)
    return zlib.decompress(comp)


def legacy_extract(blocks):
    """The old, buggy extractor.  Kept only for the regression tests.

    Bugs reproduced by the tests:
      * each block is parsed in isolation -- no carry-over, so an entry
        crossing a block boundary is silently truncated and the rest of
        the stream desynchronises (missing tail / shifted entries);
      * blocks are "de-duplicated" by content hash, so a repeated block
        is skipped and its entries are lost;
      * no CRC validation, so corrupted content is returned as success.
    """
    out = {}
    seen = set()
    for block in blocks:
        block = bytes(block)
        digest = hashlib.md5(block).digest()
        if digest in seen:
            continue  # "cache": skip blocks we have already processed
        seen.add(digest)
        buf = block
        while buf:
            if len(buf) < HEADER.size:
                break  # partial header at end of block: dropped
            magic, method, name_len, raw_size, crc, comp_size = HEADER.unpack(
                buf[:HEADER.size]
            )
            if magic != MAGIC:
                break  # desync: rest of the block silently dropped
            need = HEADER.size + name_len + comp_size
            name = buf[HEADER.size:HEADER.size + name_len]
            comp = buf[HEADER.size + name_len:need]  # silently truncated
            try:
                data = _legacy_inflate(method, comp)
            except Exception:
                break
            out[name.decode("utf-8", "replace")] = data  # no CRC check
            buf = buf[need:]
    return out


# ------------------------------------------------------------------ demo

if __name__ == "__main__":
    sample = [
        ("readme.txt", b"hello world\n" * 3),
        ("payload.bin", bytes(range(256)) * 2),
        ("empty", b""),
    ]
    archive = build_archive(sample, METHOD_STORED)
    blocks = list(iter_blocks(archive, 64))
    print("archive: %d bytes in %d blocks of 64" % (len(archive), len(blocks)))

    got = extract(blocks)
    ok = all(got.get(n) == d for n, d in sample) and len(got) == len(sample)
    print("extract:        %s" % ("OK" if ok else "MISMATCH"))

    legacy = legacy_extract(blocks)
    legacy_ok = all(legacy.get(n) == d for n, d in sample) and len(legacy) == len(sample)
    print("legacy_extract: %s" % (
        "OK" if legacy_ok else "BROKEN -> %r" % {k: len(v) for k, v in legacy.items()}
    ))
