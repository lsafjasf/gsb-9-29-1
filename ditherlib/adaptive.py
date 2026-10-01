"""Adaptive palette generation via median-cut colour quantization."""

from .palettes import Palette


def median_cut_palette(pixels, bits, name="median-cut"):
    """Build a palette of exactly 2**bits colours from RGB pixels.

    Repeatedly splits the colour box with the largest channel range at its
    count-weighted median, then averages each box weighted by population.
    """
    ncolors = 1 << bits
    hist = {}
    for pixel in pixels:
        key = (int(pixel[0]), int(pixel[1]), int(pixel[2]))
        hist[key] = hist.get(key, 0) + 1
    items = [(color, count) for color, count in hist.items()]
    if not items:
        items = [((0, 0, 0), 1)]
    boxes = [items]
    while len(boxes) < ncolors:
        best_i = -1
        best_range = -1
        best_ch = 0
        for i, box in enumerate(boxes):
            if len(box) < 2:
                continue
            for ch in range(3):
                lo = min(c[ch] for c, _ in box)
                hi = max(c[ch] for c, _ in box)
                if hi - lo > best_range:
                    best_range = hi - lo
                    best_i = i
                    best_ch = ch
        if best_i < 0:
            break
        box = boxes.pop(best_i)
        box.sort(key=lambda cn: cn[0][best_ch])
        total = sum(n for _, n in box)
        acc = 0
        split = len(box) - 1
        for j, (_, n) in enumerate(box):
            acc += n
            if acc >= total / 2:
                split = j + 1
                break
        split = max(1, min(split, len(box) - 1))
        boxes.append(box[:split])
        boxes.append(box[split:])
    colors = []
    for box in boxes:
        total = sum(n for _, n in box)
        r = sum(c[0] * n for c, n in box) / total
        g = sum(c[1] * n for c, n in box) / total
        b = sum(c[2] * n for c, n in box) / total
        colors.append((round(r), round(g), round(b)))
    while len(colors) < ncolors:
        colors.append(colors[-1] if colors else (0, 0, 0))
    return Palette(colors, bits, name=name)
