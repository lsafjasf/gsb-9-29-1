"""命令行入口：

    python -m zhconv check  -m data/mapping.json
    python -m zhconv convert -m data/mapping.json -d s2t "皇后在后面"
"""

from __future__ import annotations

import argparse
import sys

from .converter import DEFAULT_MARK, make_converters
from .mapping import MappingError, load_mapping


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="zhconv", description="简繁转换工具")
    parser.add_argument("-m", "--mapping", required=True, help="映射表 JSON 路径")
    parser.add_argument("--mark", default=DEFAULT_MARK, help="待确认标记格式，须含 {} 占位符")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="仅校验映射表")

    p_convert = sub.add_parser("convert", help="转换文本")
    p_convert.add_argument("-d", "--direction", choices=["s2t", "t2s"], required=True)
    p_convert.add_argument("text", nargs="?", help="要转换的文本；缺省从 stdin 读取")
    p_convert.add_argument("--plain", action="store_true", help="输出去掉待确认标记的文本")

    args = parser.parse_args(argv)

    try:
        mapping = load_mapping(args.mapping)
    except (MappingError, OSError) as exc:
        print(f"映射表加载失败: {exc}", file=sys.stderr)
        return 2

    for warning in mapping.warnings:
        print(f"警告: {warning}", file=sys.stderr)

    if args.command == "check":
        print("映射表校验通过")
        return 0

    text = args.text if args.text is not None else sys.stdin.read()
    converter = make_converters(mapping, mark=args.mark)[args.direction]
    result = converter.convert(text)
    print(result.plain_text if args.plain else result.text)
    for item in result.pending:
        print(
            f"待确认: 位置 {item.index} 字 {item.char!r} 候选 {'/'.join(item.candidates)}",
            file=sys.stderr,
        )
    return 1 if result.pending else 0


if __name__ == "__main__":
    raise SystemExit(main())
