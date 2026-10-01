"""Palette types: fixed (gray ramps, RGB332) plus nearest-colour lookup."""


def clamp8(value):
    if value < 0.0:
        return 0.0
    if value > 255.0:
        return 255.0
    return value


def luminance(color):
    r, g, b = color
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


class Palette:
    """An ordered list of RGB colours; pixel values are indices into it."""

    def __init__(self, colors, bits, nearest=None, channel_steps=None,
                 name="palette"):
        if len(colors) > (1 << bits):
            raise ValueError(
                f"{len(colors)} colours do not fit in {bits} bits")
        self.colors = [tuple(int(c) for c in color) for color in colors]
        self.bits = bits
        self.name = name
        self._nearest_fn = nearest
        self._cache = {} if nearest is None else None
        if channel_steps is None:
            steps = []
            for ch in range(3):
                vals = sorted({c[ch] for c in self.colors})
                gaps = [b - a for a, b in zip(vals, vals[1:]) if b > a]
                steps.append(sum(gaps) / len(gaps) if gaps else 255.0)
            channel_steps = tuple(steps)
        self.channel_steps = tuple(channel_steps)

    def __len__(self):
        return len(self.colors)

    @property
    def max_gray_step(self):
        vals = sorted({round(luminance(c)) for c in self.colors})
        gaps = [b - a for a, b in zip(vals, vals[1:]) if b > a]
        return max(gaps) if gaps else 255.0

    def nearest_index(self, r, g, b):
        r = clamp8(r)
        g = clamp8(g)
        b = clamp8(b)
        if self._nearest_fn is not None:
            return self._nearest_fn(r, g, b)
        key = (int(r) >> 3, int(g) >> 3, int(b) >> 3)
        idx = self._cache.get(key)
        if idx is None:
            best = 0
            best_d = None
            for i, (pr, pg, pb) in enumerate(self.colors):
                d = (r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2
                if best_d is None or d < best_d:
                    best_d = d
                    best = i
            idx = best
            if len(self._cache) < 40000:
                self._cache[key] = idx
        return idx


def gray_palette(bits):
    """Fixed palette: 2**bits evenly spaced gray levels."""
    n = 1 << bits
    if n == 1:
        colors = [(0, 0, 0)]
    else:
        colors = [(round(i * 255 / (n - 1)),) * 3 for i in range(n)]

    def nearest(r, g, b, n=n):
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        if lum <= 0.0:
            return 0
        if lum >= 255.0:
            return n - 1
        return int(lum / 255.0 * (n - 1) + 0.5)

    step = 255.0 / (n - 1) if n > 1 else 255.0
    return Palette(colors, bits, nearest=nearest,
                   channel_steps=(step, step, step), name=f"gray{n}")


def rgb332_palette():
    """Fixed 8-bit palette: 3 bits red, 3 bits green, 2 bits blue."""
    colors = []
    for ri in range(8):
        for gi in range(8):
            for bi in range(4):
                colors.append((round(ri * 255 / 7),
                               round(gi * 255 / 7),
                               round(bi * 255 / 3)))

    def nearest(r, g, b):
        ri = min(7, max(0, int(r / 255.0 * 7 + 0.5)))
        gi = min(7, max(0, int(g / 255.0 * 7 + 0.5)))
        bi = min(3, max(0, int(b / 255.0 * 3 + 0.5)))
        return (ri << 5) | (gi << 2) | bi

    return Palette(colors, 8, nearest=nearest,
                   channel_steps=(255 / 7, 255 / 7, 255 / 3), name="rgb332")


def fixed_palette(bits):
    if bits in (1, 2, 4):
        return gray_palette(bits)
    if bits == 8:
        return rgb332_palette()
    raise ValueError(f"no fixed palette for bits={bits!r}")
