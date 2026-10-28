"""Attractor landscapes, addressability and operations in effective binary models.

The distinction between raw asymmetry and diagonal symmetrizability matters for
energy arguments. Native fixed points are not equivalent to learned arbitrary
patterns. These numerical models do not establish experimental memory capacity."""

from .landscape import find_attractors, landscape_metrics, count_scaling
from .addressability import write_state, addressable_capacity, injective_outputs

__all__ = [
    "find_attractors",
    "landscape_metrics",
    "count_scaling",
    "write_state",
    "addressable_capacity",
    "injective_outputs",
]
