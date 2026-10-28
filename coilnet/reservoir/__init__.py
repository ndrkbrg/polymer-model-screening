"""Phase 1.5: reservoir computing on the physically-derived coupling J.

Phases 1/1b measured the coil-coil coupling matrix J and found it ASYMMETRIC (a receiver-vs-
source split baked into the physics, see ``coilnet.physics.multibody.build_coupling``). That
rules out a Hopfield energy (``coilnet.hopfield``, ``coilnet.coupling``) -- there is no
guarantee the network "rolls downhill" -- but it does not rule out neural computation
altogether. Reservoir computing is the natural next hypothesis to test: it needs a rich,
input-driven dynamical system with fading memory, and does not care whether the recurrent
wiring is symmetric.

This package provides:

  * ``engine``  -- ``CoilReservoir``, the dynamical system that drives J (or any coupling
                   matrix) with an external input, plus spectral-radius helpers.
  * ``readout`` -- the trained linear readout layer (ridge regression) and error/accuracy
                   metrics.
  * ``tasks``   -- standard RC benchmark tasks (NARMA-10, Mackey-Glass prediction, delayed
                   XOR) to drive the reservoir and score the readout against.
  * ``metrics`` -- task-independent reservoir-quality diagnostics (memory capacity, kernel /
                   generalization rank, echo state property, Lyapunov exponent).

Design principle carried over from Phase 1: always validate new machinery against a textbook
reference before trusting it on the physical J. Here that reference is a standard leaky tanh
Echo State Network (``CoilReservoir(..., dynamics="esn")``), for which reservoir-computing
theory makes concrete, checkable predictions (see ``tests/test_reservoir_metrics.py``).
"""

from .engine import CoilReservoir, spectral_radius, rescale_spectral_radius
from .readout import (
    add_bias,
    ridge_fit,
    ridge_predict,
    rmse,
    nrmse,
    accuracy,
    accuracy_binary,
)
from .tasks import (
    narma10,
    make_narma10,
    mackey_glass,
    make_mackey_glass_prediction,
    delayed_xor,
    make_delayed_xor,
)
from .metrics import (
    memory_capacity,
    kernel_rank,
    generalization_rank,
    echo_state_property,
    lyapunov_estimate,
)

__all__ = [
    "CoilReservoir",
    "spectral_radius",
    "rescale_spectral_radius",
    "add_bias",
    "ridge_fit",
    "ridge_predict",
    "rmse",
    "nrmse",
    "accuracy",
    "accuracy_binary",
    "narma10",
    "make_narma10",
    "mackey_glass",
    "make_mackey_glass_prediction",
    "delayed_xor",
    "make_delayed_xor",
    "memory_capacity",
    "kernel_rank",
    "generalization_rank",
    "echo_state_property",
    "lyapunov_estimate",
]
