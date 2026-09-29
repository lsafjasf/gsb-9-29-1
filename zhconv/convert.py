"""命令行转换工具。

用法::

    python3 -m zhconv.convert -d s2t "他站在门的后面。"
    echo "皇后住在皇宫里。" | python3 -m zhconv.convert -d s2t
    python3 -m zhconv.convert -d s2t --mark-pending "他什么都会干。"
"""

from __future__ import annotations

import argparse
import sys

from .converter import Converter
from .mapping import MappingTable


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="简繁转换")
    parser.add_argument("-d", "--direction", choices=("s2t", "t2s"), required=True)
    parser.add_argument("--mapping-dir", default="mapping", help="映射表目录")
    parser.add_argument("--mark-pending", action="store_true",
                        help="无法消解的字用 ⟦原字|候选1/候选2⟧ 内联标记")
    parser.add_argument("text", nargs="*", help="要转换的文本；缺省读标准输入")
    args = parser.parse_args(argv)

    table = MappingTable.from_file(f"{args.mapping_dir}/{args.direction}.json")
    converter = Converter(table)

    chunks = args.text if args.text else [sys.stdin.read()]
    exit_code = 0
    for chunk in chunks:
        result = converter.convert(chunk, mark_pending=args.mark_pending)
        sys.stdout.write(result.text)
        if args.text:
            sys.stdout.write("\n")
        if result.pending:
            exit_code = 2
            for item in result.pending:
                print(f"待确认: 位置{item.position} '{item.char}' "
                      f"候选 {'/'.join(item.candidates)} 上下文 …{item.context}…",
                      file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
