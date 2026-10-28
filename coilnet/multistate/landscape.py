"""Attractor-landscape discovery and characterisation.

A "stable state" of the coil cluster is a binary fixed point of the async Hopfield dynamics
(``coilnet.hopfield.HopfieldNetwork``) under the physics-derived coupling J -- it does not
require J to be symmetric, only that ``update_async`` converges. We discover the landscape by
brute force: relax many random starting configurations and record which fixed point each one
falls into (its "basin"). From that empirical landscape we compute the numbers that separate
three qualitatively different pictures:

  TRIVIAL    -- effectively 2 states (a single global up/down switch; a "bit", not a "digit").
  STRUCTURED -- a handful of well-separated states, most starts land in a few big, repeatable
               basins (a usable multi-valued alphabet).
  GLASSY     -- attractor count grows exponentially with system size N and basins are tiny and
               scattered (classic spin-glass rugged landscape -- states exist, but nobody could
               ever dial one up on purpose).

``count_scaling`` is the key discriminator: fitting the slope of ln(mean attractor count) vs.
N turns "many states" into a falsifiable, quantitative claim -- exponential growth (slope
clearly > 0, roughly constant per added coil) is the spin-glass signature; a flat or mildly
growing count is not.
"""

from __future__ import annotations

import numpy as np

from ..hopfield import HopfieldNetwork
from ..physics import multibody as mb

__all__ = ["find_attractors", "landscape_metrics", "count_scaling"]


def find_attractors(J, trials=500, max_sweeps=80, rng=None):
    """Discover fixed points of the async Hopfield dynamics under J by random restarts.

    Relaxes ``trials`` random +/-1 starting configurations under ``HopfieldNetwork(J)`` and
    groups them by which fixed point they converge to (a start that has not settled into a true
    fixed point within ``max_sweeps`` sweeps is counted as non-converged, not assigned a basin).

    Returns ``(basins, n_nonconverged)`` where ``basins`` maps each distinct fixed-point state
    (a tuple of ints in {-1, 1}) to the number of random starts that fell into it (its basin
    size, out of ``trials``).
    """
    rng = np.random.default_rng() if rng is None else rng
    net = HopfieldNetwork(J)
    N = net.N
    basins = {}
    n_nonconverged = 0
    for _ in range(trials):
        s0 = rng.choice([-1.0, 1.0], size=N)
        s_final = net.update_async(s0, rng=rng, max_sweeps=max_sweeps)
        if net.is_fixed_point(s_final):
            key = tuple(int(x) for x in s_final)
            basins[key] = basins.get(key, 0) + 1
        else:
            n_nonconverged += 1
    return basins, n_nonconverged


def landscape_metrics(J, trials=500, rng=None):
    """Characterise the attractor landscape of J with numbers that distinguish STRUCTURED from
    GLASSY (see module docstring).

    Returns a dict with:
      n_attractors        -- number of distinct fixed points found.
      n_nonconverged       -- random starts that never settled (out of ``trials``).
      effective_n          -- exp(Shannon entropy of the basin-SIZE distribution): the
                               "effective number of states" weighting by how often each is
                               actually reached. effective_n == n_attractors only when every
                               basin is exactly equal size; a few giant basins plus a long tail
                               of tiny ones give effective_n << n_attractors.
      top1_basin_frac      -- fraction of ALL trials that landed in the single largest basin.
      top3_basin_frac      -- fraction of all trials landing in the 3 largest basins combined.
      mean_pairwise_hamming -- mean normalized Hamming distance between distinct attractors
                               (0 = identical, 1 = perfect complements); low + few attractors
                               with big basins reads as STRUCTURED, near-1/2 with many tiny
                               basins reads as GLASSY (attractors scattered like random points).
      hamming_std          -- std of those pairwise distances (a REGULAR code -- e.g. a small
                               set of states all roughly equidistant -- has low std; a glassy
                               random scatter has high std).
    """
    rng = np.random.default_rng() if rng is None else rng
    basins, n_nonconverged = find_attractors(J, trials=trials, rng=rng)
    n_attractors = len(basins)

    counts = np.array(sorted(basins.values(), reverse=True), dtype=float)
    if counts.sum() > 0:
        p = counts / counts.sum()
        entropy = float(-np.sum(p * np.log(p)))
        effective_n = float(np.exp(entropy))
        top1_basin_frac = float(counts[0] / trials)
        top3_basin_frac = float(counts[:3].sum() / trials)
    else:
        effective_n = 0.0
        top1_basin_frac = 0.0
        top3_basin_frac = 0.0

    states = [np.array(k, dtype=float) for k in basins.keys()]
    if len(states) >= 2:
        dists = [
            float(np.mean(states[i] != states[j]))
            for i in range(len(states))
            for j in range(i + 1, len(states))
        ]
        mean_pairwise_hamming = float(np.mean(dists))
        hamming_std = float(np.std(dists))
    else:
        mean_pairwise_hamming = 0.0
        hamming_std = 0.0

    return {
        "n_attractors": n_attractors,
        "n_nonconverged": n_nonconverged,
        "effective_n": effective_n,
        "top1_basin_frac": top1_basin_frac,
        "top3_basin_frac": top3_basin_frac,
        "mean_pairwise_hamming": mean_pairwise_hamming,
        "hamming_std": hamming_std,
    }


def count_scaling(system, N_list, seeds=(0, 1, 2), salt_M=0.01, trials=400):
    """How does the attractor count grow with system size N, for a physics-derived J?

    For each N in ``N_list``, builds J via ``coilnet.physics.multibody.build_coupling`` for
    each of ``seeds`` (a fresh coil population per seed) and averages ``n_attractors`` /
    ``effective_n`` over seeds. Also fits ``log_growth_slope`` = the slope of
    ln(mean n_attractors) vs. N by least squares: a clearly positive, roughly constant-per-coil
    slope is the EXPONENTIAL growth signature of a spin glass; a flat or small slope means the
    count is NOT blowing up combinatorially with size -- a structured (or trivial) landscape.

    Returns a dict: ``{"rows": [...], "log_growth_slope": float, "N_list": [...], "seeds": [...]}``.
    Each row is ``{"N", "mean_n_attractors", "mean_effective_n", "n_attractors_per_seed"}``.
    """
    rows = []
    for N in N_list:
        n_attr_list = []
        eff_n_list = []
        for seed in seeds:
            rng_j = np.random.default_rng(seed)
            J = mb.build_coupling(N, rng_j, system=system, spread=1.0, salt_M=salt_M, grid_n=40)[
                "J"
            ]
            rng_t = np.random.default_rng(10_000 + 37 * N + seed)
            m = landscape_metrics(J, trials=trials, rng=rng_t)
            n_attr_list.append(m["n_attractors"])
            eff_n_list.append(m["effective_n"])
        rows.append(
            {
                "N": int(N),
                "mean_n_attractors": float(np.mean(n_attr_list)),
                "mean_effective_n": float(np.mean(eff_n_list)),
                "n_attractors_per_seed": [int(x) for x in n_attr_list],
            }
        )

    Ns = np.array([r["N"] for r in rows], dtype=float)
    # floor mean counts at 1 before taking log: a system that always finds >=1 attractor should
    # not blow up the fit if some cell happens to average just under 1 due to nonconvergence.
    mean_counts = np.array([max(r["mean_n_attractors"], 1e-9) for r in rows], dtype=float)
    log_growth_slope = float(np.polyfit(Ns, np.log(mean_counts), 1)[0]) if len(Ns) >= 2 else 0.0

    return {
        "system": system,
        "rows": rows,
        "log_growth_slope": log_growth_slope,
        "N_list": [int(n) for n in N_list],
        "seeds": [int(s) for s in seeds],
        "trials": trials,
    }
