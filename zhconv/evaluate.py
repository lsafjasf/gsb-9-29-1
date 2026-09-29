"""对拍评估：转换结果 vs 标注语料，输出逐句准确率与错误分类。

语料格式（JSONL，每行一条）::

    {"id": "s2t-001", "direction": "s2t", "category": "pure_simplified",
     "source": "他站在门的后面。", "target": "他站在門的後面。", "note": "可选备注"}

错误分类：
- unresolved        无法消解被保留原字（待确认），但标注要求转换
- under_conversion  应转未转（非待确认）
- over_conversion   不该转却转了
- wrong_target      转了，但转成了错误的字

用法::

    python3 -m zhconv.evaluate --mapping-dir mapping --corpus corpus/annotated.jsonl
    python3 -m zhconv.evaluate --mapping-dir mapping --corpus corpus/annotated.jsonl \
        --json-out report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .converter import Converter
from .mapping import MappingTable

UNRESOLVED = "unresolved"
UNDER_CONVERSION = "under_conversion"
OVER_CONVERSION = "over_conversion"
WRONG_TARGET = "wrong_target"
ERROR_CLASSES = (UNRESOLVED, UNDER_CONVERSION, OVER_CONVERSION, WRONG_TARGET)

ERROR_CLASS_NAMES = {
    UNRESOLVED: "未消解(待确认)",
    UNDER_CONVERSION: "应转未转",
    OVER_CONVERSION: "误转(不该转)",
    WRONG_TARGET: "转错目标字",
}


@dataclass
class CharError:
    position: int
    source: str
    expected: str
    got: str
    cls: str

    def to_dict(self):
        return {
            "position": self.position,
            "source": self.source,
            "expected": self.expected,
            "got": self.got,
            "class": self.cls,
            "class_name": ERROR_CLASS_NAMES[self.cls],
        }


@dataclass
class SentenceReport:
    id: str
    direction: str
    category: str
    source: str
    expected: str
    got: str
    errors: list[CharError] = field(default_factory=list)
    pending: int = 0

    @property
    def exact(self) -> bool:
        return not self.errors

    @property
    def char_accuracy(self) -> float:
        total = max(len(self.expected), 1)
        return (total - len(self.errors)) / total

    def to_dict(self):
        return {
            "id": self.id,
            "direction": self.direction,
            "category": self.category,
            "source": self.source,
            "expected": self.expected,
            "got": self.got,
            "exact": self.exact,
            "char_accuracy": round(self.char_accuracy, 4),
            "pending": self.pending,
            "errors": [e.to_dict() for e in self.errors],
        }


def load_corpus(path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            for key in ("direction", "source", "target"):
                if key not in rec:
                    raise ValueError(f"{path}:{lineno} 缺少字段 {key!r}")
            rec.setdefault("id", f"line-{lineno}")
            rec.setdefault("category", "uncategorized")
            records.append(rec)
    return records


def load_converters(mapping_dir) -> dict[str, Converter]:
    mapping_dir = Path(mapping_dir)
    converters = {}
    for direction in ("s2t", "t2s"):
        path = mapping_dir / f"{direction}.json"
        if path.exists():
            converters[direction] = Converter(MappingTable.from_file(path))
    if not converters:
        raise FileNotFoundError(f"{mapping_dir} 下没有找到 s2t.json / t2s.json")
    return converters


def classify(source_ch, expected_ch, got_ch, is_pending) -> str:
    if is_pending and got_ch == source_ch:
        return UNRESOLVED
    if got_ch == source_ch:
        return UNDER_CONVERSION
    if expected_ch == source_ch:
        return OVER_CONVERSION
    return WRONG_TARGET


def evaluate_record(converter: Converter, rec: dict) -> SentenceReport:
    result = converter.convert(rec["source"])
    got, expected, source = result.text, rec["target"], rec["source"]
    pending_positions = {p.position for p in result.pending}
    report = SentenceReport(
        id=rec["id"], direction=rec["direction"], category=rec["category"],
        source=source, expected=expected, got=got, pending=len(result.pending),
    )
    if len(got) != len(expected):
        # 转换保证与源文本等长；此处说明标注与源文本不等长，属语料问题
        raise ValueError(
            f"{rec['id']}: 源文本({len(source)})与标注({len(expected)})长度不一致，"
            f"无法逐字对拍")
    for pos, (src_ch, exp_ch, got_ch) in enumerate(zip(source, expected, got)):
        if got_ch == exp_ch:
            continue
        report.errors.append(CharError(
            position=pos, source=src_ch, expected=exp_ch, got=got_ch,
            cls=classify(src_ch, exp_ch, got_ch, pos in pending_positions),
        ))
    return report


@dataclass
class EvalReport:
    sentences: list[SentenceReport]

    def summary(self) -> dict:
        total = len(self.sentences)
        exact = sum(1 for s in self.sentences if s.exact)
        class_counts = Counter()
        for s in self.sentences:
            for e in s.errors:
                class_counts[e.cls] += 1
        per_category = defaultdict(lambda: [0, 0])  # category -> [exact, total]
        for s in self.sentences:
            per_category[s.category][1] += 1
            if s.exact:
                per_category[s.category][0] += 1
        return {
            "total_sentences": total,
            "exact_match": exact,
            "sentence_accuracy": round(exact / total, 4) if total else 0.0,
            "mean_char_accuracy": round(
                sum(s.char_accuracy for s in self.sentences) / total, 4) if total else 0.0,
            "total_char_errors": sum(class_counts.values()),
            "error_classes": {cls: class_counts.get(cls, 0) for cls in ERROR_CLASSES},
            "per_category": {
                cat: {"exact": ex, "total": tot,
                      "accuracy": round(ex / tot, 4)}
                for cat, (ex, tot) in sorted(per_category.items())
            },
        }

    def to_dict(self) -> dict:
        return {
            "summary": self.summary(),
            "sentences": [s.to_dict() for s in self.sentences],
        }


def evaluate(records, converters) -> EvalReport:
    sentences = []
    for rec in records:
        converter = converters.get(rec["direction"])
        if converter is None:
            raise ValueError(f"{rec['id']}: 没有方向 {rec['direction']!r} 的映射表")
        sentences.append(evaluate_record(converter, rec))
    return EvalReport(sentences=sentences)


def format_text_report(report: EvalReport) -> str:
    lines = []
    summary = report.summary()
    lines.append("=== 逐句结果 ===")
    for s in report.sentences:
        mark = "OK  " if s.exact else "FAIL"
        lines.append(f"[{mark}] {s.id} ({s.category}/{s.direction}) "
                     f"字准确率 {s.char_accuracy:.2%} 待确认 {s.pending}")
        for e in s.errors:
            lines.append(f"       位置{e.position}: 源'{e.source}' "
                         f"期望'{e.expected}' 实际'{e.got}' "
                         f"[{ERROR_CLASS_NAMES[e.cls]}]")
    lines.append("")
    lines.append("=== 汇总 ===")
    lines.append(f"句子总数: {summary['total_sentences']}")
    lines.append(f"整句命中: {summary['exact_match']} "
                 f"({summary['sentence_accuracy']:.2%})")
    lines.append(f"平均字准确率: {summary['mean_char_accuracy']:.2%}")
    lines.append(f"字级错误总数: {summary['total_char_errors']}")
    lines.append("错误分类:")
    for cls in ERROR_CLASSES:
        lines.append(f"  {ERROR_CLASS_NAMES[cls]}({cls}): "
                     f"{summary['error_classes'][cls]}")
    lines.append("分类别整句准确率:")
    for cat, stat in summary["per_category"].items():
        lines.append(f"  {cat}: {stat['exact']}/{stat['total']} "
                     f"({stat['accuracy']:.2%})")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="简繁转换对拍评估")
    parser.add_argument("--mapping-dir", default="mapping", help="映射表目录")
    parser.add_argument("--corpus", required=True, help="标注语料(JSONL)")
    parser.add_argument("--json-out", help="把完整报告写为 JSON 文件")
    args = parser.parse_args(argv)

    converters = load_converters(args.mapping_dir)
    records = load_corpus(args.corpus)
    report = evaluate(records, converters)
    print(format_text_report(report))
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        print(f"\nJSON 报告已写入 {args.json_out}")
    return 0 if report.summary()["total_char_errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
