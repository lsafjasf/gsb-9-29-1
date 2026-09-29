"""命令行入口：python -m textnorm ..."""
from __future__ import annotations

import argparse
import json
import sys

from .core import (
    ExemptConfig,
    NormConfig,
    build_manifest,
    find_exempt_spans,
    normalize,
    rollback,
)


def _read(path):
    if path is None or path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _write(path, data):
    if path is None or path == "-":
        sys.stdout.write(data)
    else:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(data)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="textnorm",
        description="文档文本规范化：统一标点/空白/引号，代码块、行内代码、URL、引用段自动豁免。",
    )
    ap.add_argument("file", nargs="?", help="输入文件（缺省读标准输入，'-' 同）")
    ap.add_argument("-o", "--output", help="输出路径（缺省写标准输出）")
    ap.add_argument("--changes", help="改动清单输出路径（JSON）")
    ap.add_argument("--spans", action="store_true", help="打印豁免区域及判定依据后退出")
    ap.add_argument("--check", action="store_true", help="只检查：存在可规范化处则退出码为 1")
    ap.add_argument("--rollback", metavar="MANIFEST.json",
                    help="回滚模式：按清单把输入文本还原为原文")
    ap.add_argument("--no-exempt-code", dest="fenced_code", action="store_false", help="不免除围栏代码块")
    ap.add_argument("--no-exempt-inline", dest="inline_code", action="store_false", help="不免除行内代码")
    ap.add_argument("--no-exempt-url", dest="url", action="store_false", help="不免除 URL")
    ap.add_argument("--no-exempt-quote", dest="blockquote", action="store_false", help="不免除引用段")
    ap.add_argument("--final-newline", action="store_true", help="确保文件以换行结尾")
    args = ap.parse_args(argv)

    text = _read(args.file)

    if args.rollback:
        with open(args.rollback, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        _write(args.output, rollback(text, manifest))
        return 0

    exempt = ExemptConfig(
        fenced_code=args.fenced_code,
        inline_code=args.inline_code,
        url=args.url,
        blockquote=args.blockquote,
    )
    cfg = NormConfig(exempt=exempt, final_newline=args.final_newline)

    if args.spans:
        for sp in find_exempt_spans(text, exempt):
            print(f"[{sp.kind}] {sp.start}..{sp.end}  {sp.reason}")
        return 0

    result = normalize(text, cfg)
    manifest = build_manifest(result, cfg)

    if args.changes:
        _write(args.changes, json.dumps(manifest, ensure_ascii=False, indent=2))

    if args.check:
        if result.changes:
            print(f"textnorm: {len(result.changes)} 处可规范化", file=sys.stderr)
            return 1
        return 0

    _write(args.output, result.text)
    if not args.output:
        return 0
    print(f"textnorm: {len(result.changes)} 处改动 -> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
