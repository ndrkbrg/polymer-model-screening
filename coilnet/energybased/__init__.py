"""Energy-based numerical models, constrained RBMs and sampling diagnostics.

Generic RBM weight constraints and update-error models are computational
comparisons, not demonstrated mappings onto polymer material devices."""

from __future__ import annotations

from .glauber import GlauberNetwork, recall
from .metrics import bars_and_stripes, bas_validity, boltzmann_kl_smallN, recall_accuracy
from .rbm import RBM, local, low_rank, nonnegative

__all__ = [
    "GlauberNetwork",
    "recall",
    "RBM",
    "nonnegative",
    "low_rank",
    "local",
    "recall_accuracy",
    "boltzmann_kl_smallN",
    "bars_and_stripes",
    "bas_validity",
]
