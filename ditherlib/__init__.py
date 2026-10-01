"""ditherlib: dithering, quantization, adaptive palettes, bit packing."""

from .packing import (
    pack_indices,
    unpack_indices,
    pack_rows,
    unpack_rows,
    packed_row_size,
)
from .palettes import (
    Palette,
    gray_palette,
    rgb332_palette,
    fixed_palette,
    luminance,
)
from .adaptive import median_cut_palette
from .ordered import ordered_dither_row, ordered_dither_image, BAYER4, BAYER8
from .diffusion import StreamingDitherer, diffuse
from .quantize import quantize_row, quantize_image
from .metrics import error_stats

__all__ = [
    "pack_indices",
    "unpack_indices",
    "pack_rows",
    "unpack_rows",
    "packed_row_size",
    "Palette",
    "gray_palette",
    "rgb332_palette",
    "fixed_palette",
    "luminance",
    "median_cut_palette",
    "ordered_dither_row",
    "ordered_dither_image",
    "BAYER4",
    "BAYER8",
    "StreamingDitherer",
    "diffuse",
    "quantize_row",
    "quantize_image",
    "error_stats",
]
