import gc
import platform
import random
import sys
import time
from datetime import datetime

from morphology import (
    StructuringElement,
    closing,
    dilate,
    erode,
    opening,
)


HEIGHT = 64
WIDTH = 64
SE_HEIGHT = 31
SE_WIDTH = 31
BRUTE_REPEATS = 2
FAST_REPEATS = 5


def timed(callable_value, repeats):
    times = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        result = callable_value()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return min(times), result


def main():
    random.seed(20260930)
    element = StructuringElement.rectangle(SE_HEIGHT, SE_WIDTH)
    binary = [[random.randrange(2) for _ in range(WIDTH)] for _ in range(HEIGHT)]
    grayscale = [[random.randrange(256) for _ in range(WIDTH)] for _ in range(HEIGHT)]

    datasets = [
        ("binary", binary),
        ("grayscale", grayscale),
    ]
    operations = [
        ("erosion", erode),
        ("dilation", dilate),
        ("opening", opening),
        ("closing", closing),
    ]

    rows = []
    for dataset_name, image in datasets:
        for operation_name, operation in operations:
            brute_time, brute_result = timed(
                lambda operation=operation, image=image: operation(image, element, method="brute"),
                BRUTE_REPEATS,
            )
            fast_time, fast_result = timed(
                lambda operation=operation, image=image: operation(image, element, method="separable"),
                FAST_REPEATS,
            )
            assert brute_result == fast_result
            rows.append(
                {
                    "dataset": dataset_name,
                    "operation": operation_name,
                    "brute": brute_time,
                    "fast": fast_time,
                    "speedup": brute_time / fast_time,
                }
            )

    lines = [
        "# Morphology benchmark",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Python: {platform.python_version()} ({platform.platform()})",
        f"- Image: {HEIGHT}x{WIDTH}",
        f"- Structuring element: {SE_HEIGHT}x{SE_WIDTH} flat rectangle ({SE_HEIGHT * SE_WIDTH} members)",
        f"- Timing: minimum of {BRUTE_REPEATS} brute runs and {FAST_REPEATS} separable runs",
        "- All brute-force and separable results are asserted exactly equal",
        "",
        "| Dataset | Operation | Brute (s) | Separable (s) | Speedup |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['operation']} | {row['brute']:.6f} | "
            f"{row['fast']:.6f} | {row['speedup']:.1f}x |"
        )

    totals = {
        dataset: {
            "brute": sum(row["brute"] for row in rows if row["dataset"] == dataset),
            "fast": sum(row["fast"] for row in rows if row["dataset"] == dataset),
        }
        for dataset, _ in datasets
    }
    lines.extend(["", "| Dataset | Four-operation total brute (s) | Total separable (s) | Total speedup |", "| --- | ---: | ---: | ---: |"])
    for dataset, _ in datasets:
        brute_total = totals[dataset]["brute"]
        fast_total = totals[dataset]["fast"]
        lines.append(
            f"| {dataset} | {brute_total:.6f} | {fast_total:.6f} | {brute_total / fast_total:.1f}x |"
        )

    report = "\n".join(lines) + "\n"
    with open("benchmark_results.md", "w", encoding="utf-8") as output:
        output.write(report)
    print(report)


if __name__ == "__main__":
    sys.exit(main())
