"""Glauber (heat-bath) dynamics: the stochastic, finite-temperature sibling of the
deterministic Hopfield network in ``coilnet.hopfield``.

Phase 1/1b/1.5/1.6 built up a consistent picture of the polymer substrate: it is
STOCHASTIC (thermal noise is not a nuisance but a required ingredient -- Phase 1.5's echo
state property only holds with noise switched on) and MULTISTABLE (many-valued, not just
binary), with a physically-derived coupling that is ASYMMETRIC. Deterministic Hopfield
recall (``hopfield.HopfieldNetwork.update_async``, "always roll downhill in energy") FAILED
on that asymmetric coupling, because "energy never increases" is only guaranteed for
symmetric W.

Phase 1.7 asks the natural next question: if the substrate is going to be noisy anyway, can
we turn that noise into a FEATURE rather than fight it, by using an explicitly
finite-temperature, PROBABILISTIC update rule instead of the deterministic
"settle-to-the-nearest-attractor" rule? Glauber (heat-bath) dynamics is the textbook answer:

    P(s_i = +1 | s_{-i}) = sigmoid(2 * h_i / T),   h_i = local field on neuron i

where ``h_i = (W @ s)_i + bias_i`` is exactly the same local field used by the Hopfield
network. The key theoretical fact this module exists to make concrete and testable:

    * For SYMMETRIC W, Glauber updates satisfy DETAILED BALANCE with respect to the
      Boltzmann distribution P(s) ∝ exp(-E(s) / T), where E is the SAME energy function as
      ``hopfield.HopfieldNetwork.energy``. So repeatedly applying ``sample_step`` samples
      from that distribution -- this is what ``metrics.boltzmann_kl_smallN`` checks by brute
      -force enumeration on a small network.
    * At T -> 0 the heat-bath rule collapses to the deterministic greedy sign rule (the
      Hopfield update). At T > 0 the network can escape shallow local minima and, in
      principle, average over many attempts at recall -- which is the mechanism by which
      finite temperature might "rescue" recall on a coupling whose deterministic dynamics
      gets stuck or oscillates (the asymmetric, physics-derived case).
"""

from __future__ import annotations

import numpy as np

__all__ = ["GlauberNetwork", "recall"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class GlauberNetwork:
    """A stochastic binary ({-1, +1}) network updated by Glauber (heat-bath) dynamics.

    Parameters
    ----------
    W : (N, N) array
        Coupling matrix. The diagonal is zeroed on construction (no self-coupling), same
        convention as ``hopfield.HopfieldNetwork``. W need NOT be symmetric -- that is
        exactly what this module is built to probe.
    bias : (N,) array, optional
        External drive on each neuron. Defaults to zero.
    temperature : float
        The "T" in ``P(s_i=+1|...) = sigmoid(2 h_i / T)``. T -> 0 recovers the deterministic
        greedy rule; larger T flattens the distribution towards a coin flip.
    rng : numpy.random.Generator, optional
        Source of randomness for ``sample_step`` / ``sample_chain``.
    """

    def __init__(self, W, bias=None, temperature=1.0, rng=None):
        W = np.array(W, dtype=float)
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("W must be a square 2-D matrix")
        self.N = W.shape[0]
        W = W.copy()
        np.fill_diagonal(W, 0.0)
        self.W = W
        self.bias = np.zeros(self.N) if bias is None else np.asarray(bias, dtype=float).copy()
        self.temperature = float(temperature)
        self.rng = np.random.default_rng() if rng is None else rng

    def local_field(self, s):
        s = np.asarray(s, dtype=float)
        return self.W @ s + self.bias

    def energy(self, s):
        """E(s) = -1/2 sᵀ W s - biasᵀ s -- identical definition to HopfieldNetwork.energy, so
        the two models are directly comparable at the same W."""
        s = np.asarray(s, dtype=float)
        return float(-0.5 * s @ self.W @ s - self.bias @ s)

    def sample_step(self, s, unit=None):
        """One Glauber (heat-bath) update.

        Picks a single unit ``i`` (uniformly at random unless ``unit`` is given) and sets
        ``s_i = +1`` with probability ``sigmoid(2 * h_i / T)`` where ``h_i`` is unit i's
        local field under the CURRENT state of every other unit -- the exact single-site
        conditional of the Boltzmann distribution ``P(s) ∝ exp(-E(s)/T)``. At T <= 0 this
        degenerates to the deterministic sign rule (with a tie held at the current value),
        matching ``hopfield.sign_keep`` in that limit.

        Returns a NEW array (the input is not mutated).
        """
        s = np.asarray(s, dtype=float).copy()
        i = int(self.rng.integers(self.N)) if unit is None else int(unit)
        hi = float(self.W[i] @ s + self.bias[i])
        if self.temperature <= 0:
            s[i] = 1.0 if hi > 0 else (-1.0 if hi < 0 else s[i])
        else:
            p_up = _sigmoid(2.0 * hi / self.temperature)
            s[i] = 1.0 if self.rng.random() < p_up else -1.0
        return s

    def sample_chain(self, s0, n_steps, burn_in=0):
        """Run ``n_steps`` single-unit Glauber updates from ``s0``; return the states kept
        after ``burn_in`` as an ``(n_steps - burn_in, N)`` array (one row per single-unit
        update, not per full sweep).

        For SYMMETRIC W, each single-unit update satisfies detailed balance with respect to
        the Boltzmann distribution, so -- given enough steps for the chain to mix -- the
        empirical distribution of the kept states converges to ``P(s) ∝ exp(-E(s)/T)``. This
        is the "sampler" half of the recall mechanism in ``recall`` below, and is what
        ``metrics.boltzmann_kl_smallN`` validates directly.
        """
        s = np.asarray(s0, dtype=float).copy()
        kept = []
        for step in range(n_steps):
            s = self.sample_step(s)
            if step >= burn_in:
                kept.append(s)
        return np.asarray(kept) if kept else np.empty((0, self.N))


def recall(W, patterns, cue, temperature=1.0, n_steps=400, burn_in=None, rng=None):
    """Probabilistic associative recall: run Glauber dynamics from a noisy ``cue`` and
    report the sample-averaged state, compared against the given stored ``patterns``.

    This is the finite-temperature analogue of ``HopfieldNetwork.recall`` (which greedily
    "rolls downhill" to the nearest fixed point). Here, instead of ONE deterministic
    trajectory, we run a Glauber chain, discard a ``burn_in`` prefix, and average the kept
    samples -- the sign of that average (``majority_state``) is the network's probabilistic
    "best guess". We also report which of the given ``patterns`` it is closest to (by
    normalized overlap), since with a noisy/oscillating dynamics the majority state need not
    match any single pattern well.

    Returns a dict with ``mean_state`` (soft, in [-1, 1]), ``majority_state`` (hard, {-1,
    +1}), ``best_match_idx`` (index into ``patterns`` of the closest stored pattern), and
    ``best_overlap`` (its normalized overlap with ``majority_state``, in [-1, 1]).
    """
    rng = np.random.default_rng() if rng is None else rng
    burn_in = n_steps // 2 if burn_in is None else burn_in
    net = GlauberNetwork(W, temperature=temperature, rng=rng)
    samples = net.sample_chain(cue, n_steps=n_steps, burn_in=burn_in)
    mean_state = samples.mean(axis=0) if len(samples) else np.asarray(cue, dtype=float)
    majority_state = np.where(mean_state >= 0, 1.0, -1.0)

    P = np.atleast_2d(np.asarray(patterns, dtype=float))
    overlaps = (P @ majority_state) / P.shape[1]
    best_idx = int(np.argmax(overlaps))
    return {
        "mean_state": mean_state,
        "majority_state": majority_state,
        "best_match_idx": best_idx,
        "best_overlap": float(overlaps[best_idx]),
    }
