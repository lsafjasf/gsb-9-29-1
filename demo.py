"""Demo: artifact samples, size/error stats, streaming memory data.

Run: python3 demo.py   (writes samples/ and prints report tables)
"""

import os
import tracemalloc
import zlib

from ditherlib import (
    StreamingDitherer,
    diffuse,
    error_stats,
    fixed_palette,
    median_cut_palette,
    ordered_dither_image,
    pack_rows,
    quantize_image,
)
from ditherlib import gen
from ditherlib.imageio import (
    indices_to_color_rows,
    indices_to_gray_rows,
    write_pgm,
    write_png,
    write_ppm,
)

SAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")


def run_method(name, rows, width, palette):
    if name == "quant":
        return quantize_image(rows, palette)
    if name == "ordered":
        return ordered_dither_image(rows, palette)
    if name == "diffusion":
        out, _ = diffuse(rows, width, palette)
        return out
    raise ValueError(name)


def report_case(image_name, rows, bits, palette, methods):
    width = len(rows[0])
    height = len(rows)
    results = []
    for method in methods:
        out = run_method(method, rows, width, palette)
        packed = pack_rows(out, bits)
        compressed = zlib.compress(packed, 9)
        stats = error_stats(rows, out, palette)
        results.append({
            "image": image_name,
            "method": method,
            "bits": bits,
            "palette": palette.name,
            "packed": len(packed),
            "zlib": len(compressed),
            "mae": stats["mae"],
            "total": stats["total_lum_diff"],
            "max_cum": stats["max_abs_cumulative"],
            "max_row": stats["max_abs_row_error"],
            "out": out,
        })
    return results


def print_table(title, rows, columns):
    print(f"\n== {title} ==")
    header = " | ".join(columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(" | ".join(str(row.get(c, "")) for c in columns))


def composite_gray(panels, gap=4):
    height = max(len(p) for p in panels)
    width = sum(len(p[0]) for p in panels) + gap * (len(panels) - 1)
    canvas = [[128] * width for _ in range(height)]
    x = 0
    for panel in panels:
        for y, row in enumerate(panel):
            canvas[y][x:x + len(row)] = row
        x += len(panel[0]) + gap
    return canvas


def composite_color(left, right, gap=4):
    width = len(left[0]) + gap + len(right[0])
    canvas = []
    for la, rb in zip(left, right):
        canvas.append(list(la) + [(128, 128, 128)] * gap + list(rb))
    return canvas


def memory_peak(fn):
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def streaming_run(width, height, bits):
    palette = fixed_palette(bits)
    ditherer = StreamingDitherer(width, palette)
    row = [(120, 60, 200)] * width
    sink = 0
    for _ in range(height):
        sink += sum(ditherer.process_row(row))
    return sink


def batch_run(width, height, bits):
    palette = fixed_palette(bits)
    rows = [[(120, 60, 200)] * width for _ in range(height)]
    out, _ = diffuse(rows, width, palette)
    return len(pack_rows(out, bits))


def main():
    os.makedirs(SAMPLES, exist_ok=True)

    gradient = gen.gradient(256, 256)
    pal1 = fixed_palette(1)
    g_quant = quantize_image(gradient, pal1)
    g_ordered = ordered_dither_image(gradient, pal1)
    g_diff, _ = diffuse(gradient, 256, pal1)
    write_pgm(os.path.join(SAMPLES, "gradient_original.pgm"),
              [[p[0] for p in row] for row in gradient])
    write_pgm(os.path.join(SAMPLES, "gradient_1bit_quant.pgm"),
              indices_to_gray_rows(g_quant, pal1))
    write_pgm(os.path.join(SAMPLES, "gradient_1bit_ordered.pgm"),
              indices_to_gray_rows(g_ordered, pal1))
    write_pgm(os.path.join(SAMPLES, "gradient_1bit_diffusion.pgm"),
              indices_to_gray_rows(g_diff, pal1))
    write_pgm(os.path.join(SAMPLES, "gradient_1bit_comparison.pgm"),
              composite_gray([
                  indices_to_gray_rows(g_quant, pal1),
                  indices_to_gray_rows(g_ordered, pal1),
                  indices_to_gray_rows(g_diff, pal1),
              ]))
    write_png(os.path.join(SAMPLES, "gradient_1bit_comparison.png"),
              composite_gray([
                  indices_to_gray_rows(g_quant, pal1),
                  indices_to_gray_rows(g_ordered, pal1),
                  indices_to_gray_rows(g_diff, pal1),
              ]))

    zones = gen.color_zones(192, 128)
    flat = [p for row in zones for p in row]
    adaptive4 = median_cut_palette(flat, 4)
    z_fixed = ordered_dither_image(zones, fixed_palette(4))
    z_adapt = ordered_dither_image(zones, adaptive4)
    write_ppm(os.path.join(SAMPLES, "zones_original.ppm"), zones)
    write_ppm(os.path.join(SAMPLES, "zones_4bit_fixed.ppm"),
              indices_to_color_rows(z_fixed, fixed_palette(4)))
    write_ppm(os.path.join(SAMPLES, "zones_4bit_adaptive.ppm"),
              indices_to_color_rows(z_adapt, adaptive4))
    z_fixed_rows = indices_to_color_rows(z_fixed, fixed_palette(4))
    z_adapt_rows = indices_to_color_rows(z_adapt, adaptive4)
    write_png(os.path.join(SAMPLES, "zones_4bit_comparison.png"),
              composite_color(z_fixed_rows, z_adapt_rows))

    cases = []
    cases += report_case("gradient 256x256", gradient, 1, fixed_palette(1),
                         ("quant", "ordered", "diffusion"))
    cases += report_case("gradient 256x256", gradient, 4, fixed_palette(4),
                         ("quant", "ordered", "diffusion"))
    cases += report_case("solid 64x64", gen.solid(64, 64, (90, 90, 90)), 2,
                         fixed_palette(2), ("quant", "ordered", "diffusion"))
    cases += report_case("noise 128x128", gen.noise(128, 128, seed=3), 2,
                         fixed_palette(2), ("quant", "ordered", "diffusion"))
    cases += report_case("1px wide 1x512", gen.vertical_gradient(1, 512), 2,
                         fixed_palette(2), ("ordered", "diffusion"))
    cases += report_case("wide 4096x2", gen.noise(4096, 2, seed=4), 2,
                         fixed_palette(2), ("ordered", "diffusion"))
    cases += report_case("tall 2x4096", gen.noise(2, 4096, seed=5), 2,
                         fixed_palette(2), ("ordered", "diffusion"))
    cases += report_case("zones 192x128 fixed", zones, 4, fixed_palette(4),
                         ("ordered", "diffusion"))
    cases += report_case("zones 192x128 adaptive", zones, 4, adaptive4,
                         ("ordered", "diffusion"))

    columns = ("image", "method", "bits", "palette", "packed", "zlib",
               "mae", "total", "max_cum", "max_row")
    table_rows = [{
        "image": r["image"], "method": r["method"], "bits": r["bits"],
        "palette": r["palette"], "packed": r["packed"], "zlib": r["zlib"],
        "mae": f'{r["mae"]:.2f}', "total": f'{r["total"]:.1f}',
        "max_cum": f'{r["max_cum"]:.1f}', "max_row": f'{r["max_row"]:.1f}',
    } for r in cases]
    print_table("size / error stats (packed bytes, zlib bytes, luminance)",
                table_rows, columns)

    vgrad = gen.vertical_gradient(64, 256)
    out_v, _ = diffuse(vgrad, 64, fixed_palette(1))
    stats_v = error_stats(vgrad, out_v, fixed_palette(1))
    csv_path = os.path.join(SAMPLES, "cumulative_error.csv")
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("row,row_error,cumulative_error\n")
        for y, (re_, ce_) in enumerate(
                zip(stats_v["per_row_error"], stats_v["cumulative"])):
            fh.write(f"{y},{re_:.6f},{ce_:.6f}\n")
    print(f"\nper-row cumulative error written to {csv_path}")
    head = ", ".join(f"{c:.1f}" for c in stats_v["cumulative"][:8])
    tail = ", ".join(f"{c:.1f}" for c in stats_v["cumulative"][-4:])
    print(f"cumulative[0:8] = [{head}] ... cumulative[-4:] = [{tail}]")
    print(f"max |cumulative| = {stats_v['max_abs_cumulative']:.2f}, "
          f"max |row error| = {stats_v['max_abs_row_error']:.2f}")

    mem_rows = []
    for height in (2000, 8000, 32000):
        peak_s = memory_peak(lambda: streaming_run(64, height, 2))
        peak_b = memory_peak(lambda: batch_run(64, height, 2))
        mem_rows.append({
            "height": height,
            "streaming_peak_KiB": f"{peak_s / 1024:.1f}",
            "batch_peak_KiB": f"{peak_b / 1024:.1f}",
        })
    print_table("tracemalloc peak memory vs height (width=64, 2bpp)",
                mem_rows, ("height", "streaming_peak_KiB", "batch_peak_KiB"))

    print(f"\nsamples written to {SAMPLES}/")


if __name__ == "__main__":
    main()
