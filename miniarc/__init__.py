"""miniarc: 固定块读取的 zlib 分块流式归档。

只依赖 Python 3 标准库（zlib/struct/hashlib/dataclasses）。
归档中的每个条目由「定长头 + 独立 zlib 流」组成，因此读块边界与
压缩块边界是否对齐都不影响正确性，关键在解压器必须正确维护
跨读块的字节缓冲并归还 zlib 流结束后多取的尾字节。
"""

from .archive import Packer, Entry, pack, unpack, CorruptArchiveError, CHUNK_SIZE

__all__ = [
    "Packer",
    "Entry",
    "pack",
    "unpack",
    "CorruptArchiveError",
    "CHUNK_SIZE",
]
