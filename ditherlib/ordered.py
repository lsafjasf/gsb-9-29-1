"""Bayer ordered dithering: threshold matrix applied before quantization."""


def _bayer(n):
    m = [[0, 2], [3, 1]]
    size = 2
    while size < n:
        big = [[0] * (2 * size) for _ in range(2 * size)]
        for y in range(2 * size):
            for x in range(2 * size):
                corner = ((0, 2), (3, 1))[y // size][x // size]
                big[y][x] = 4 * m[y % size][x % size] + corner
        m = big
        size *= 2
    return m


BAYER4 = tuple(tuple(row) for row in _bayer(4))
BAYER8 = tuple(tuple(row) for row in _bayer(8))


def ordered_dither_row(row, y, palette, matrix=BAYER8):
    """Dither one row; stateless, so rows can be processed independently."""
    n = len(matrix)
    norm = 1.0 / (n * n)
    sr, sg, sb = palette.channel_steps
    nearest = palette.nearest_index
    my = matrix[y % n]
    out = []
    for x, (pr, pg, pb) in enumerate(row):
        t = (my[x % n] + 0.5) * norm - 0.5
        out.append(nearest(pr + t * sr, pg + t * sg, pb + t * sb))
    return out


def ordered_dither_image(rows, palette, matrix=BAYER8):
    return [ordered_dither_row(row, y, palette, matrix)
            for y, row in enumerate(rows)]
