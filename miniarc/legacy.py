"""旧版解压器（保留用于复现线上事故，禁止在生产路径使用）。

它解析与 :mod:`miniarc.archive` 相同的 MARC 线路格式，但带着三处
真实缺陷，正是线上「解压后文件末尾缺一截 / 个别条目内容错位 /
重跑有时又正常」的根因：

缺陷 A —— 假设单次 read 一定读满:
    ``stream.read(HEADER_SIZE)`` 直接按一次返回的字节切片解头。
    底层按固定块读取时，头部恰好被块边界切开就会拿到半拉头，
    后续解析整体错位。读块够大、条目够小（或在内存 BytesIO 上
    一次就把整份归档返回）时永远触发不了，所以「重跑有时正常」。

缺陷 B —— 丢弃 zlib unused_data:
    每读到一块就喂给一个新建的一次性 ``zlib.decompress``，流结束后
    同一块里带着的下一条目头/尾字节被整块扔掉。后续条目要么错位，
    要么最后一两条直接消失（缺尾）。

缺陷 C —— 损坏被吞、半截内容当成功:
    zlib 报错时被 ``except`` 吃掉，已解压出的半截数据照样返回；
    声明长度 / CRC 对不上也只是打个招呼继续，调用方拿到「成功」
    结果，且没有任何偏移信息。
"""

from __future__ import annotations

import struct
import zlib
from typing import BinaryIO, List

from .archive import HEADER_SIZE, MAGIC, _HEADER


class LegacyFormatError(Exception):
    pass


def legacy_unpack(stream: BinaryIO, chunk_size: int) -> List[tuple[str, bytes]]:
    """旧版实现：一次性返回全部 (name, data)。返回即代表它自认为成功。"""
    results: List[tuple[str, bytes]] = []

    while True:
        # 缺陷 A：想当然认为 read(n) 一定返回 n 字节。
        header = stream.read(HEADER_SIZE)
        if not header:
            break
        if len(header) < HEADER_SIZE:
            # 旧代码对截断的处理也是「静默结束」——尾部条目凭空消失。
            break
        magic, version, flags, nlen, dlen, crc_declared = _HEADER.unpack(header)
        if magic != MAGIC:
            # 错位后通常会落到这里；旧代码只是停止，不报告偏移。
            break
        name = stream.read(nlen).decode("utf-8", "replace")

        # 缺陷 B：逐块各自 decompress，unused_data 直接丢弃。
        pieces: List[bytes] = []
        broken = False
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            try:
                piece = zlib.decompress(block)  # 每块都当成独立 zlib 流
            except zlib.error:
                # 这一块解不动就当流结束（真正的流结束本来就会抛 error，
                # 旧代码靠这个歪招识别结束），带着的尾字节整块丢掉。
                broken = True
                break
            pieces.append(piece)
            if len(b"".join(pieces)) >= dlen:
                break

        data = b"".join(pieces)[:dlen]

        # 缺陷 C：长度/CRC 对不上也不失败，半截内容照当成功产出。
        # （旧代码里甚至没有检查，这里留一行打印位但调用方无感。）
        _ = (dlen, crc_declared, broken)
        results.append((name, data))

    return results
