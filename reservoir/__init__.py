"""Streaming sampling with exact weights: uniform reservoir, weighted
reservoir (A-Res), exact k=1 weighted sampling, Chao's PPS reservoir, and
exact shard merging.  Standard library only; randomness is injectable."""
from .weights import parse_weight
from .uniform import UniformReservoirSampler
from .weighted_one import WeightedOneSampler, merge_weighted_one
from .ares import AResWeightedSampler
from .pps import PPSReservoirSampler
from .merge import merge_uniform, merge_ares
from .exact import ares_inclusion_probabilities, pps_targets

__all__ = [
    "parse_weight",
    "UniformReservoirSampler",
    "WeightedOneSampler",
    "merge_weighted_one",
    "AResWeightedSampler",
    "PPSReservoirSampler",
    "merge_uniform",
    "merge_ares",
    "ares_inclusion_probabilities",
    "pps_targets",
]
