"""修复版归档读写实现。

线路格式（大端序）::

    每个条目:
      MAGIC  4B   b'MARC'
      VER    1B   当前 1
      FLAGS  1B   保留，必须为 0
      NLEN   4B   名称 UTF-8 字节数（<=65535）
      DLEN   8B   解压后字节数（<=2**40-1）
      CRC32  8B   解压后数据的 CRC32（低 32 位有效）
      NAME   NLEN B
      ZDATA  变长，一条 *独立* 的 zlib 流；流结束后 decompressobj 可能
             在 unused_data 中带出同一次读块里多取的尾字节（甚至包含
             后续多个条目的头），必须归还到读缓冲前面
    EOF:     下一条目一个字节都读不到才是正常结束；读到一半即截断损坏。

三类线上事故的根因与对策：

1. 固定块读取且块边界不与压缩块对齐，逻辑头部会被切成两半 ——
   ``_BlockStream.read_exact`` 循环精确取足 n 字节，不假设单次
   ``read`` 的返回长度。
2. zlib 流结束后 ``unused_data`` 里的尾字节被旧实现丢弃 ——
   这里压回缓冲前部并回退逻辑偏移，后续条目才不会整体错位 / 丢尾。
3. 旧实现遇损坏吞异常、吐半截内容 —— 这里每条目在
   长度/CRC32/zlib 全部校验通过前只存在于暂存缓冲，失败统一抛
   :class:`CorruptArchiveError`（带偏移），绝不产出该条目。
"""

from __future__ import annotations

import io
import struct
import zlib
from dataclasses import dataclass
from typing import BinaryIO, Iterable, Iterator, Union

CHUNK_SIZE = 4096

MAGIC = b"MARC"
VERSION = 1
_HEADER = struct.Struct(">4sBBIQQ")  # magic, ver, flags, nlen, dlen, crc32
HEADER_SIZE = _HEADER.size  # 26
MAX_NAME_LEN = 65535
MAX_DATA_LEN = (1 << 40) - 1


class CorruptArchiveError(Exception):
    """归档损坏。``offset`` 为损坏被确认处相对归档起始的字节偏移。

    - 头损坏 / 截断 / 长度 / CRC 不符：偏移指向该条目头起始位置。
    - zlib 载荷损坏：偏移指向该条目压缩载荷起始位置。
    """

    def __init__(self, message: str, offset: int):
        super().__init__(f"{message} (offset 0x{offset:x} / {offset})")
        self.offset = offset


@dataclass(frozen=True)
class Entry:
    name: str
    data: bytes


class Packer:
    """流式打包器：``add_entry`` 增量写入，不要求一次性内存缓冲。"""

    def __init__(self, stream: BinaryIO, level: int = zlib.Z_DEFAULT_COMPRESSION):
        self._stream = stream
        self._level = level
        self._closed = False
        self.offset = 0  # 已写入字节数，即下一条目起始偏移

    def add_entry(self, name: str, data: bytes) -> int:
        if self._closed:
            raise ValueError("packer already closed")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data must be bytes-like")
        raw_name = name.encode("utf-8")
        if len(raw_name) > MAX_NAME_LEN:
            raise ValueError("name too long")
        if len(data) > MAX_DATA_LEN:
            raise ValueError("data too long")

        start = self.offset
        header = _HEADER.pack(
            MAGIC, VERSION, 0, len(raw_name), len(data), zlib.crc32(data) & 0xFFFFFFFF
        )
        payload = zlib.compress(bytes(data), self._level)
        blob = header + raw_name + payload
        self._stream.write(blob)
        self.offset += len(blob)
        return start

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "Packer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def pack(
    entries: Iterable[tuple[str, bytes]],
    level: int = zlib.Z_DEFAULT_COMPRESSION,
) -> bytes:
    """便捷函数：把一组 (name, data) 打成内存中的归档字节。"""
    buf = io.BytesIO()
    with Packer(buf, level=level) as packer:
        for name, data in entries:
            packer.add_entry(name, data)
    return buf.getvalue()


class _BlockStream:
    """固定块读取 + 跨块缓冲 + unused_data 归还。

    - ``pos``：解析器逻辑偏移，即缓冲队头字节在归档中的位置；
      消费字节时前进，压回尾字节时回退。
    - ``physical``：已从底层流读出的物理字节数，单调递增。
    """

    def __init__(self, stream: BinaryIO, chunk_size: int):
        self._stream = stream
        self._chunk_size = chunk_size
        self._buf = bytearray()
        self.pos = 0
        self.physical = 0

    def _fill(self) -> bool:
        """保证缓冲非空；干净 EOF 返回 False。"""
        if self._buf:
            return True
        block = self._stream.read(self._chunk_size)
        if not block:
            return False
        self._buf.extend(block)
        self.physical += len(block)
        return True

    def read_exact(self, n: int, entry_start: int, what: str) -> bytes | None:
        """精确读取 n 字节。

        一个字节都拿不到时返回 ``None``（仅允许出现在条目边界，表示
        正常归档结束）；拿到一部分就 EOF 则报截断损坏。
        """
        out = bytearray()
        while len(out) < n:
            if not self._fill():
                if not out:
                    return None
                raise CorruptArchiveError(
                    f"truncated archive while reading {what}", entry_start
                )
            take = min(n - len(out), len(self._buf))
            out.extend(self._buf[:take])
            del self._buf[:take]
            self.pos += take
        return bytes(out)

    def read_payload(
        self, dlen: int, expected_crc: int, entry_start: int, payload_start: int
    ) -> bytes:
        """解压一条独立 zlib 流，校验长度与 CRC32 后返回全部数据。

        zlib 流结束时，同一次喂入块里多取的尾字节（unused_data）
        压回缓冲队头并回退逻辑偏移，供下一条目使用。
        """
        decoder = zlib.decompressobj()
        out = bytearray()
        while not decoder.eof:
            if not self._fill():
                raise CorruptArchiveError(
                    "truncated compressed payload", payload_start
                )
            block = bytes(self._buf)
            self._buf.clear()
            self.pos += len(block)
            try:
                piece = decoder.decompress(block)
            except zlib.error as exc:
                raise CorruptArchiveError(
                    f"invalid compressed data: {exc}", payload_start
                )
            out.extend(piece)
            if len(out) > dlen:
                raise CorruptArchiveError(
                    "decompressed length exceeds declared length", entry_start
                )
            if decoder.unused_data:
                # 关键：多取的字节属于下一条目，必须归还。
                self._buf.extend(decoder.unused_data)
                self.pos -= len(decoder.unused_data)
        if len(out) != dlen:
            raise CorruptArchiveError(
                f"length mismatch: header declares {dlen}, stream yields {len(out)}",
                entry_start,
            )
        actual_crc = zlib.crc32(out) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise CorruptArchiveError(
                f"crc32 mismatch: header declares 0x{expected_crc:08x}, "
                f"actual 0x{actual_crc:08x}",
                entry_start,
            )
        return bytes(out)


def unpack(
    source: Union[BinaryIO, bytes, bytearray],
    chunk_size: int = CHUNK_SIZE,
) -> Iterator[Entry]:
    """流式解压归档，逐条产出 :class:`Entry`。

    条目在全部校验通过后才产出；任何损坏立即抛出
    :class:`CorruptArchiveError`（带偏移）。已产出的前置条目均完整
    有效，损坏条目本身及其后内容绝不会以「成功」形式输出。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if isinstance(source, (bytes, bytearray)):
        stream: BinaryIO = io.BytesIO(bytes(source))
    else:
        stream = source

    bs = _BlockStream(stream, chunk_size)

    while True:
        entry_start = bs.pos
        header = bs.read_exact(HEADER_SIZE, entry_start, "entry header")
        if header is None:
            return  # 条目边界上的干净 EOF

        magic, version, flags, nlen, dlen, crc32_declared = _HEADER.unpack(header)
        if magic != MAGIC:
            raise CorruptArchiveError("bad entry magic", entry_start)
        if version != VERSION:
            raise CorruptArchiveError(f"unsupported version {version}", entry_start)
        if flags != 0:
            raise CorruptArchiveError(f"unsupported flags 0x{flags:x}", entry_start)
        if nlen > MAX_NAME_LEN:
            raise CorruptArchiveError(f"name length too large: {nlen}", entry_start)
        if dlen > MAX_DATA_LEN:
            raise CorruptArchiveError(f"data length too large: {dlen}", entry_start)

        raw_name = bs.read_exact(nlen, entry_start, "entry name")
        if raw_name is None:
            raise CorruptArchiveError("truncated archive while reading entry name", entry_start)
        try:
            name = raw_name.decode("utf-8")
        except UnicodeDecodeError:
            raise CorruptArchiveError("entry name is not valid utf-8", entry_start)

        payload_start = bs.pos
        data = bs.read_payload(dlen, crc32_declared, entry_start, payload_start)
        yield Entry(name, data)
