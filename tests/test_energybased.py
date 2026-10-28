"""Fast, deterministic tests for the Phase 1.7 energy-based module (``coilnet.energybased``).

Kept tiny and quick (a few seconds total): a brute-force sampler-correctness check, a
single-pattern recall sanity check, and a much-shorter-than-production RBM training run that
still clears a well-above-chance bar on Bars-and-Stripes.
"""

import numpy as np

from coilnet.energybased import glauber, metrics
from coilnet.energybased.rbm import RBM
from coilnet.hopfield import HopfieldNetwork


def test_boltzmann_kl_small_for_symmetric_net():
    """Glauber dynamics on a small SYMMETRIC net should sample close to the exact Boltzmann
    distribution (brute-force-checkable by enumerating all 2^N states)."""
    rng = np.random.default_rng(0)
    patterns = rng.choice([-1.0, 1.0], size=(2, 6))
    W = HopfieldNetwork.from_hebbian(patterns).W
    np.fill_diagonal(W, 0.0)
    net = glauber.GlauberNetwork(W, temperature=1.0, rng=rng)
    kl = metrics.boltzmann_kl_smallN(
        net, temperature=1.0, n_steps=20000, burn_in=2000, rng=np.random.default_rng(1)
    )
    assert kl < 0.1


def test_glauber_recall_recovers_single_pattern_at_low_temperature():
    """A single stored Hebbian pattern, lightly noised (~10% flipped bits), should be recovered
    with high overlap by Glauber recall at low (but nonzero) temperature."""
    rng = np.random.default_rng(2)
    pattern = rng.choice([-1.0, 1.0], size=30)
    W = HopfieldNetwork.from_hebbian(pattern[None, :]).W
    np.fill_diagonal(W, 0.0)

    cue = pattern.copy()
    flip_idx = rng.choice(30, size=3, replace=False)
    cue[flip_idx] *= -1.0

    res = glauber.recall(
        W, pattern[None, :], cue, temperature=0.25, n_steps=300, rng=np.random.default_rng(3)
    )
    assert res["best_overlap"] > 0.8


def test_rbm_learns_bars_and_stripes_above_untrained_baseline():
    """A shortened (but still nontrivial) CD-1 training run on Bars-and-Stripes(3) should clear
    a comfortably-above-chance BAS validity bar -- random-chance validity is 14/512 ~= 0.027, so
    0.4 is a strong, unambiguous signal that learning happened, without needing the full
    production epoch count (2500) that gets to ~0.94."""
    data = metrics.bars_and_stripes(3)
    rng = np.random.default_rng(4)
    rbm = RBM(9, 16, rng=rng)
    rbm.train(data, epochs=800, lr=0.5, k=1, batch_size=14)
    samples = rbm.gibbs_sample(200, 500)
    validity = metrics.bas_validity(samples, 3)
    assert validity > 0.4
