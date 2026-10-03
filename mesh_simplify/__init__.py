"""Mesh simplification via edge collapse with quadric error metrics and
feature-edge protection. Pure Python 3 standard library."""
from .mesh import Mesh
from .simplify import simplify, classify_features
from .validate import validate_topology, assert_valid
from .distance import hausdorff_distance
from . import generators

__all__ = [
    "Mesh",
    "simplify",
    "classify_features",
    "validate_topology",
    "assert_valid",
    "hausdorff_distance",
    "generators",
]
