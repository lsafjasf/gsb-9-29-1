"""Stdlib-only Canny-style edge detection package."""

from .autothresh import auto_thresholds, otsu_threshold
from .core import (count_edges, hysteresis, non_max_suppression,
                   sobel_gradient)
from .pgm import edges_to_image, read_pgm, write_pgm
from .pipeline import (detect_edges, detect_from_candidates,
                       gradient_and_candidates)
from .sensitivity import sensitivity, to_csv

__all__ = ["auto_thresholds", "otsu_threshold", "count_edges",
           "hysteresis", "non_max_suppression", "sobel_gradient",
           "read_pgm", "write_pgm", "edges_to_image",
           "detect_edges", "detect_from_candidates",
           "gradient_and_candidates", "sensitivity", "to_csv"]
