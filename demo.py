"""演示：生成测试图像，对比 直接量化 / 有序抖动 / 误差扩散。

运行：python3 demo.py [输出目录]   （默认 ./out）

产出：
  - 终端打印各用例的量化误差统计（含总亮度守恒数据）
  - 终端打印渐变的 ASCII 伪影对比
  - 输出目录下写入 PGM 灰度图，可用任意看图工具打开对比
"""

import os
import random
import sys

from dither import (
    error_diffusion,
    error_stats,
    ordered_dither,
    quantize_plain,
    write_png,
)

ASCII_RAMP = " .:-=+*#%@"


def make_gradient(width, height):
    return [[255.0 * x / (width - 1) for x in range(width)] for _ in range(height)]


def make_solid(width, height, value):
    return [[float(value)] * width for _ in range(height)]


def make_noise(width, height, seed=42):
    rng = random.Random(seed)
    return [[rng.uniform(0, 255) for _ in range(width)] for _ in range(height)]


def ascii_preview(image, cols=72, rows=24):
    """把灰度图降采样成 ASCII 便于终端直接对比伪影。"""
    height = len(image)
    width = len(image[0])
    lines = []
    for r in range(rows):
        y0 = r * height // rows
        y1 = max(y0 + 1, (r + 1) * height // rows)
        line = []
        for c in range(cols):
            x0 = c * width // cols
            x1 = max(x0 + 1, (c + 1) * width // cols)
            acc = n = 0.0
            for y in range(y0, min(y1, height)):
                for x in range(x0, min(x1, width)):
                    acc += image[y][x]
                    n += 1
            line.append(ASCII_RAMP[min(9, int(acc / n / 256 * 10))])
        lines.append("".join(line))
    return "\n".join(lines)


def report(name, before, after, levels):
    s = error_stats(before, after)
    step = 255.0 / (levels - 1)
    bound = step / 2
    ok = "OK " if abs(s["total_diff"]) <= bound + 1e-6 else "超出"
    print(
        "  %-26s 像素=%6d  总亮度 %12.1f -> %12.1f  差=%+9.4f"
        "  [守恒界(误差扩散) ±%.1f: %s]" % (name, s["pixels"], s["total_before"],
                                            s["total_after"], s["total_diff"], bound, ok)
    )
    print(
        "  %-26s MAE=%7.3f  RMSE=%7.3f  最大单像素误差=%6.1f"
        % ("", s["mean_abs_error"], s["rmse"], s["max_abs_error"])
    )
    return s


def run_case(case_name, image, levels, outdir, preview=False):
    print("\n== 用例: %s  (量化级数 levels=%d) ==" % (case_name, levels))
    results = {
        "plain": quantize_plain(image, levels),
        "ordered": ordered_dither(image, levels, matrix_size=4),
        "diffusion": error_diffusion(image, levels),
    }
    for method, out in results.items():
        report(method, image, out, levels)
        write_png(out, os.path.join(outdir, "%s_%s_l%d.png" % (case_name, method, levels)))
    if preview:
        labels = {"plain": "直接量化（可见色带）",
                  "ordered": "Bayer 有序抖动（周期性网格纹理）",
                  "diffusion": "误差扩散（无周期纹理，局部颗粒）"}
        for method, out in results.items():
            print("\n  -- %s --" % labels[method])
            print(ascii_preview(out))


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "out"
    os.makedirs(outdir, exist_ok=True)

    print("量化误差统计（总亮度差越接近 0 说明误差守恒越好；")
    print("误差扩散的守恒界为 ±step/2，step = 255/(levels-1)）")

    run_case("gradient", make_gradient(256, 64), 4, outdir, preview=True)
    run_case("gradient2", make_gradient(256, 64), 2, outdir)
    run_case("solid", make_solid(64, 64, 100), 2, outdir)
    run_case("noise", make_noise(64, 64, seed=42), 2, outdir)
    run_case("col_1px", make_solid(1, 64, 100), 2, outdir)   # 单像素宽（竖）
    run_case("row_1px", make_solid(64, 1, 100), 2, outdir)   # 单像素宽（横）
    run_case("one_px", make_solid(1, 1, 100), 2, outdir)     # 1x1

    print("\nPNG 样例已写入 %s/ （*_plain 直接量化、*_ordered 有序抖动、*_diffusion 误差扩散）" % outdir)


if __name__ == "__main__":
    main()
