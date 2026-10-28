"""Metrics for the energy-based-network experiments: recall accuracy, a sampler-correctness
check (does our Glauber sampler actually obey detailed balance?), and the Bars-and-Stripes
generative benchmark used to validate the RBM.
"""

from __future__ import annotations

import itertools

import numpy as np

from . import glauber

__all__ = ["recall_accuracy", "boltzmann_kl_smallN", "bars_and_stripes", "bas_validity"]


def recall_accuracy(recovered, target):
    """Normalized overlap between a recovered state and a target {-1, +1} pattern, mapped
    onto [0, 1]: 1.0 = perfect recall, 0.5 = chance (no information), 0.0 = perfect
    anti-recall (every bit flipped). Works for a hard {-1, +1} ``recovered`` state as well as
    a soft, continuous one in [-1, 1] (e.g. a Glauber sample-average)."""
    recovered = np.asarray(recovered, dtype=float)
    target = np.asarray(target, dtype=float)
    overlap = float(recovered @ target) / len(target)
    return (overlap + 1.0) / 2.0


def boltzmann_kl_smallN(net, temperature, n_steps=20000, burn_in=2000, rng=None):
    """Validate the Glauber sampler on a SMALL network (N <= 10) by brute force: run a long
    Glauber chain at the given ``temperature`` and compare the empirical distribution of the
    states it visits to the EXACT Boltzmann distribution ``P(s) ∝ exp(-E(s)/T)``, computed by
    enumerating all 2^N states directly. Returns ``KL(empirical || exact)``.

    This is the direct test of the theoretical claim underpinning ``recall``: for SYMMETRIC
    W, single-unit Glauber updates satisfy detailed balance with respect to the Boltzmann
    distribution, so a long enough chain's empirical distribution should match it (small KL).
    For ASYMMETRIC W there is no guarantee -- this function can be run on such a net too, as
    a way of quantifying how far the sampler's stationary behaviour drifts from "the"
    Boltzmann distribution defined by the (symmetric-part-only) energy.

    Parameters
    ----------
    net : glauber.GlauberNetwork
        Provides the coupling (``net.W``, ``net.bias``); its own ``temperature`` attribute is
        ignored in favour of the ``temperature`` argument (so the same net can be re-checked
        at several temperatures without rebuilding it).
    """
    if net.N > 10:
        raise ValueError("boltzmann_kl_smallN only supports small N (<=10, needs 2^N enumeration)")
    rng = np.random.default_rng(0) if rng is None else rng
    sampler = glauber.GlauberNetwork(net.W, bias=net.bias, temperature=temperature, rng=rng)

    s0 = rng.choice([-1.0, 1.0], size=net.N)
    samples = sampler.sample_chain(s0, n_steps=n_steps, burn_in=burn_in)

    states = np.array(list(itertools.product([-1.0, 1.0], repeat=net.N)))
    energies = np.array([sampler.energy(s) for s in states])
    logw = -energies / temperature
    logw -= logw.max()
    w = np.exp(logw)
    exact_p = w / w.sum()

    index = {tuple(s.tolist()): k for k, s in enumerate(states)}
    counts = np.zeros(len(states))
    for s in samples:
        counts[index[tuple(s.tolist())]] += 1
    total = counts.sum()
    empirical_p = counts / total if total > 0 else np.full(len(states), 1.0 / len(states))

    eps = 1e-12
    kl = float(np.sum(empirical_p * np.log((empirical_p + eps) / (exact_p + eps))))
    return kl


def bars_and_stripes(size):
    """The classic small generative benchmark (MacKay): binary ``size`` x ``size`` images,
    flattened to length ``size**2`` vectors of {0, 1}, where every row is a single constant
    value ("stripes") OR every column is a single constant value ("bars"). There are
    ``2 * 2**size - 2`` distinct valid patterns (the two all-one-colour images belong to
    both families and are counted once). A generative model has "learned" this dataset if
    the images it invents overwhelmingly fall in this set.
    """
    patterns = set()
    for bits in itertools.product([0, 1], repeat=size):
        stripes = np.array([[b] * size for b in bits], dtype=float)  # every row constant
        bars = stripes.T  # every column constant
        patterns.add(tuple(stripes.flatten().tolist()))
        patterns.add(tuple(bars.flatten().tolist()))
    return np.array(sorted(patterns))


def bas_validity(samples, size):
    """Fraction of ``samples`` (binary vectors of length ``size**2``) that are valid
    bars-or-stripes patterns (every row constant, or every column constant) -- the standard
    generative-quality score for a model trained on ``bars_and_stripes``."""
    samples = np.atleast_2d(np.asarray(samples, dtype=float))
    samples = (samples > 0.5).astype(int)
    n_valid = 0
    for s in samples:
        img = s.reshape(size, size)
        rows_const = bool(np.all(img == img[:, [0]]))
        cols_const = bool(np.all(img == img[[0], :]))
        if rows_const or cols_const:
            n_valid += 1
    return n_valid / len(samples)
