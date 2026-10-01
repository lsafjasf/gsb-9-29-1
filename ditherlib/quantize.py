"""Plain nearest-colour quantization without dithering."""


def quantize_row(row, palette):
    return [palette.nearest_index(*pixel) for pixel in row]


def quantize_image(rows, palette):
    return [quantize_row(row, palette) for row in rows]
