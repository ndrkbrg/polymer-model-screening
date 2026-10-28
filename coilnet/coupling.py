"""The coupling matrix W and its symmetry diagnostic.

W is the wiring diagram of the polymer "neural network": W[i, j] is how strongly coil j
influences coil i. Everything about whether the solution behaves like a Hopfield memory
depends on W.

The single most important quantity in the whole project (Phase 1) is the **asymmetry of
W**. A Hopfield memory needs W symmetric (W[i, j] == W[j, i]). The physical mechanism is
probably NOT symmetric — a big swollen coil pushes on a small neighbour more than the
reverse — so measuring how asymmetric a physically-derived W is *is the flagship result*.

IMPORTANT: ``placeholder_redistribution_coupling`` below is a deliberately-labelled
PLACEHOLDER for Phase 0. It is NOT the real physics. Phase 1 replaces it with a W derived
from the modified Poisson–Boltzmann electrostatics of Paper 3. The placeholder exists only
to exercise the engine and to *illustrate* why we expect asymmetry.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "symmetric_part",
    "antisymmetric_part",
    "asymmetry_index",
    "is_lyapunov_safe",
    "random_symmetric",
    "placeholder_redistribution_coupling",
]


def symmetric_part(W):
    """The balanced part of W: (W + Wᵀ) / 2."""
    W = np.asarray(W, dtype=float)
    return 0.5 * (W + W.T)


def antisymmetric_part(W):
    """The one-directional part of W: (W - Wᵀ) / 2."""
    W = np.asarray(W, dtype=float)
    return 0.5 * (W - W.T)


def asymmetry_index(W):
    """A single number in [0, 1]: 0 = perfectly symmetric, 1 = perfectly antisymmetric.

    Defined as ||W - Wᵀ|| / ||W + Wᵀ|| (Frobenius norm). This is the headline metric of
    Phase 1: how far the physical wiring is from the symmetric form a Hopfield memory needs.
    """
    W = np.asarray(W, dtype=float)
    num = np.linalg.norm(W - W.T)
    den = np.linalg.norm(W + W.T)
    if den == 0.0:
        return 0.0 if num == 0.0 else 1.0
    return float(num / den)


def is_lyapunov_safe(W, tol=1e-6):
    """True if W is symmetric enough that the Hopfield energy is a valid Lyapunov function."""
    return asymmetry_index(W) <= tol


def random_symmetric(N, rng=None, scale=1.0):
    """A random symmetric coupling matrix (zero diagonal) — handy for tests/demos."""
    rng = np.random.default_rng() if rng is None else rng
    A = rng.normal(scale=scale, size=(N, N))
    W = 0.5 * (A + A.T)
    np.fill_diagonal(W, 0.0)
    return W


def placeholder_redistribution_coupling(positions, volumes=None, debye_length=1.0, strength=1.0):
    """PLACEHOLDER coupling (NOT the real physics — see module docstring).

    A crude stand-in for the "concentration redistribution" idea: coil j influences coil i
    with a strength that grows with j's size (volume) and decays with the distance between
    them over a Debye-length scale::

        W[i, j] = strength * volumes[j] * exp(-distance(i, j) / debye_length)

    The key teaching point baked in on purpose: because the strength depends on the *source*
    coil's size (``volumes[j]``), W is **asymmetric whenever coils differ in size** — which
    is exactly the physical reason we expect the real, Phase-1 W to be non-symmetric.

    Parameters
    ----------
    positions : (N, d) array
        Coil centre positions.
    volumes : (N,) array, optional
        Relative coil sizes (swelling). Defaults to all-equal (which makes W symmetric).
    debye_length : float
        Screening length that sets how far a coil's influence reaches.
    strength : float
        Overall coupling scale.
    """
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    N = positions.shape[0]
    volumes = np.ones(N) if volumes is None else np.asarray(volumes, dtype=float)
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dist = np.linalg.norm(positions[i] - positions[j])
            W[i, j] = strength * volumes[j] * np.exp(-dist / debye_length)
    return W
