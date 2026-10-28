"""Hopfield network engine (the reusable substrate).

A Hopfield network is the specific neural-network model the corpus claims a polymer
solution imitates. Neurons take states in {-1, +1}. Every neuron connects to every other
through the coupling matrix ``W`` (see ``coupling.py`` for what W means). There is no
self-coupling, so the diagonal of W is forced to zero.

Key fact this module exists to make concrete and testable:
    * If W is SYMMETRIC, the energy E(s) = -1/2 sᵀ W s - bᵀ s can only DECREASE (or stay
      equal) under asynchronous updates. The network reliably "rolls downhill" into a
      stored memory (an attractor). This is the guarantee the whole polymer-brain claim
      relies on.
    * If W is ASYMMETRIC, that guarantee is gone: energy can increase and the network can
      oscillate. Phase 1 asks whether the *physically-derived* W is symmetric enough for
      this guarantee to hold. This module gives us the tools to answer that.
"""

from __future__ import annotations

import numpy as np

__all__ = ["sign_keep", "HopfieldNetwork"]


def sign_keep(field, current):
    """Sign activation that HOLDS the current state when the field is exactly zero.

    Holding on ties is what preserves the "energy never increases" guarantee for
    symmetric W (a coin-flip on ties could nudge energy upward).
    """
    field = np.asarray(field, dtype=float)
    current = np.asarray(current, dtype=float)
    return np.where(field > 0, 1.0, np.where(field < 0, -1.0, current))


class HopfieldNetwork:
    """A Hopfield network defined by a coupling matrix W and optional bias.

    Parameters
    ----------
    W : (N, N) array
        Coupling matrix (the "wiring"). The diagonal is zeroed on construction.
    bias : (N,) array, optional
        External drive on each neuron (e.g. temperature/salt inputs). Defaults to zero.
    """

    def __init__(self, W, bias=None):
        W = np.array(W, dtype=float)
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("W must be a square 2-D matrix")
        self.N = W.shape[0]
        W = W.copy()
        np.fill_diagonal(W, 0.0)
        self.W = W
        self.bias = np.zeros(self.N) if bias is None else np.asarray(bias, dtype=float).copy()

    # ---- learning rules (used to VALIDATE the engine against known theory) ----

    @classmethod
    def from_hebbian(cls, patterns, bias=None):
        """Classic Hebbian storage: W = (1/N) Σ_p ξ_p ξ_pᵀ.

        Note: the ~0.138 N capacity result applies to THIS rule with random patterns.
        It does NOT apply to a physics-derived structured W (see capacity.py).
        """
        P = np.atleast_2d(np.asarray(patterns, dtype=float))
        _, N = P.shape
        W = (P.T @ P) / N
        return cls(W, bias)

    @classmethod
    def from_pseudo_inverse(cls, patterns, bias=None):
        """Projection / pseudo-inverse rule: makes the stored patterns exact fixed points."""
        P = np.atleast_2d(np.asarray(patterns, dtype=float))
        _, N = P.shape
        Q = (P @ P.T) / N
        W = (P.T @ np.linalg.pinv(Q) @ P) / N
        return cls(W, bias)

    # ---- dynamics ----

    def local_field(self, s):
        s = np.asarray(s, dtype=float)
        return self.W @ s + self.bias

    def energy(self, s):
        """E(s) = -1/2 sᵀ W s - biasᵀ s.  (Only the symmetric part of W affects this.)"""
        s = np.asarray(s, dtype=float)
        return float(-0.5 * s @ self.W @ s - self.bias @ s)

    def step_sync(self, s):
        """One synchronous update of all neurons."""
        return sign_keep(self.local_field(s), s)

    def is_fixed_point(self, s):
        s = np.asarray(s, dtype=float)
        return bool(np.array_equal(self.step_sync(s), s))

    def update_async(self, s, rng=None, max_sweeps=100, record_energy=False):
        """Asynchronous updates (one neuron at a time) until convergence.

        Returns the final state, or ``(state, energy_trace)`` if ``record_energy`` is set.
        The energy trace records E after every single-neuron update, which is what lets
        tests assert the monotonic-descent guarantee for symmetric W.
        """
        rng = np.random.default_rng() if rng is None else rng
        s = np.asarray(s, dtype=float).copy()
        energies = [self.energy(s)] if record_energy else None
        for _ in range(max_sweeps):
            changed = False
            for i in rng.permutation(self.N):
                hi = self.W[i] @ s + self.bias[i]
                si_new = 1.0 if hi > 0 else (-1.0 if hi < 0 else s[i])
                if si_new != s[i]:
                    s[i] = si_new
                    changed = True
                if record_energy:
                    energies.append(self.energy(s))
            if not changed:
                break
        return (s, np.asarray(energies)) if record_energy else s

    def recall(self, cue, **kwargs):
        """Alias for update_async: settle from a (noisy/partial) cue into the nearest attractor."""
        return self.update_async(cue, **kwargs)
