"""Addressability -- the decisive register-vs-glass test.

``landscape.py`` answers "how many stable states exist, and how are they arranged". That alone
cannot tell register from glass: a spin glass also has many stable states, it just has no way
to reach a CHOSEN one on demand. The test that actually distinguishes them is whether a target
state can be WRITTEN (driven there deliberately) and then HELD once the driving is removed --
exactly what a multi-valued logic value would need to do to be usable.

The write/hold protocol, modelled directly on ``coilnet.hopfield.HopfieldNetwork``:

  1. WRITE  -- add an external bias field ``b = beta * target`` to the coupling (a
               ``HopfieldNetwork(J, bias=b)``) and relax from a random start. The bias tilts
               the landscape so the target configuration is favoured everywhere, regardless of
               where the state started.
  2. HOLD   -- remove the bias (back to plain ``HopfieldNetwork(J)``) and relax again, carrying
               the state over from step 1. If the target is a genuine, addressable attractor of
               J, the state stays there (or very near it) once the write field is gone. If it is
               not -- if it was only ever propped up by the external field, or if the true
               nearby attractor is subtly different -- the state drifts away during HOLD.

A state that survives HOLD with high overlap, repeatably across several noisy write attempts
(different random starts / update orders), counts as ADDRESSABLE. The number of candidate
attractors that pass this test is the usable "alphabet size" -- the number of logic values the
substrate could actually support, as opposed to the raw (landscape) attractor count.
"""

from __future__ import annotations

import numpy as np

from ..hopfield import HopfieldNetwork

__all__ = ["write_state", "addressable_capacity", "injective_outputs"]


def write_state(J, target, beta, settle_sweeps=40, hold_sweeps=40, rng=None):
    """Attempt to WRITE ``target`` into J via a bias field, then HOLD it with the bias removed.

    WRITE: relax a random +/-1 start under ``HopfieldNetwork(J, bias=beta*target)`` for up to
    ``settle_sweeps`` sweeps. HOLD: take that settled state and relax it again under the plain
    ``HopfieldNetwork(J)`` (no bias) for up to ``hold_sweeps`` sweeps. Returns the final held
    state (an (N,) array of +/-1). Randomizing the start (rather than starting exactly at
    ``target``) and the async update order is what makes repeated calls a genuine, noisy
    "attempt" -- not a deterministic replay.
    """
    rng = np.random.default_rng() if rng is None else rng
    target = np.asarray(target, dtype=float)
    N = target.shape[0]

    write_net = HopfieldNetwork(J, bias=beta * target)
    s0 = rng.choice([-1.0, 1.0], size=N)
    s_written = write_net.update_async(s0, rng=rng, max_sweeps=settle_sweeps)

    hold_net = HopfieldNetwork(J)
    s_held = hold_net.update_async(s_written, rng=rng, max_sweeps=hold_sweeps)
    return s_held


def addressable_capacity(
    J, candidate_states, beta=1.0, repeats=5, overlap_thresh=0.95, rng=None, majority_frac=0.6
):
    """For each candidate target state, is it a genuinely ADDRESSABLE attractor of J?

    Attempts ``write_state`` ``repeats`` times per candidate (each a fresh random start / update
    order -- a "noisy attempt"), at a small internal sweep of bias strengths (``beta``, ``2*beta``,
    ``4*beta``; the best of the three is kept per target, since a fixed beta need not be strong
    enough for every target). A target counts as ADDRESSABLE if, at its best beta, the held
    state's overlap with the target is >= ``overlap_thresh`` on a strong majority
    (>= ``majority_frac``) of the ``repeats`` attempts -- i.e. it is REPEATABLY reachable, not
    just occasionally stumbled into.

    Returns a dict:
      n_addressable        -- count of candidates that pass (the usable alphabet size).
      addressable_fraction -- n_addressable / len(candidate_states).
      per_target           -- list of {"target", "success_rate", "beta", "addressable"}.
    """
    rng = np.random.default_rng() if rng is None else rng
    candidates = [np.asarray(c, dtype=float) for c in candidate_states]
    betas = sorted({float(beta), float(beta) * 2.0, float(beta) * 4.0})

    per_target = []
    n_addressable = 0
    for target in candidates:
        best_rate = 0.0
        best_beta = betas[0]
        for b in betas:
            successes = 0
            for _ in range(repeats):
                child_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
                held = write_state(J, target, b, rng=child_rng)
                overlap = float(np.mean(held == target))
                if overlap >= overlap_thresh:
                    successes += 1
            rate = successes / repeats
            if rate > best_rate:
                best_rate = rate
                best_beta = b
        addressable = best_rate >= majority_frac
        if addressable:
            n_addressable += 1
        per_target.append(
            {
                "target": tuple(int(x) for x in target),
                "success_rate": float(best_rate),
                "beta": float(best_beta),
                "addressable": bool(addressable),
            }
        )

    n_candidates = len(candidates)
    return {
        "n_addressable": n_addressable,
        "addressable_fraction": (n_addressable / n_candidates) if n_candidates else 0.0,
        "per_target": per_target,
        "betas_tried": list(betas),
        "overlap_thresh": overlap_thresh,
        "repeats": repeats,
        "majority_frac": majority_frac,
    }


def injective_outputs(J, n_inputs=200, beta=1.0, rng=None, settle_sweeps=40, hold_sweeps=40):
    """A looser "distinguishable outputs" capacity: how many distinct states does J actually
    reach when driven by many different (unplanned) bias patterns?

    Applies ``n_inputs`` distinct random bias fields (continuous, not necessarily any discovered
    attractor) through the same write/hold protocol as ``write_state``, and counts how many
    DISTINCT held states result. Also checks REPEATABILITY: for each bias pattern, the
    write/hold is run twice (different random starts / update order); if J's dynamics are doing
    something meaningfully input-driven (as opposed to being dominated by noise-sensitive
    chaos), the same bias should land in the same held state both times.

    Returns a dict with ``n_distinct_outputs``, ``distinct_fraction`` (of ``n_inputs``),
    ``repeatability`` (fraction of the ``n_inputs`` bias patterns whose two independent
    attempts agreed), and ``basin_size_histogram`` (sorted output-frequency counts, largest
    first -- how concentrated the reachable outputs are).
    """
    rng = np.random.default_rng() if rng is None else rng
    J = np.asarray(J, dtype=float)
    N = J.shape[0]
    free_net = HopfieldNetwork(J)

    seen = {}
    repeat_matches = 0
    for _ in range(n_inputs):
        bias = beta * rng.normal(size=N)
        write_net = HopfieldNetwork(J, bias=bias)

        s0_a = rng.choice([-1.0, 1.0], size=N)
        s_written_a = write_net.update_async(s0_a, rng=rng, max_sweeps=settle_sweeps)
        s_held_a = free_net.update_async(s_written_a, rng=rng, max_sweeps=hold_sweeps)
        key_a = tuple(int(x) for x in s_held_a)
        seen[key_a] = seen.get(key_a, 0) + 1

        s0_b = rng.choice([-1.0, 1.0], size=N)
        s_written_b = write_net.update_async(s0_b, rng=rng, max_sweeps=settle_sweeps)
        s_held_b = free_net.update_async(s_written_b, rng=rng, max_sweeps=hold_sweeps)
        key_b = tuple(int(x) for x in s_held_b)
        if key_b == key_a:
            repeat_matches += 1

    n_distinct = len(seen)
    return {
        "n_inputs": n_inputs,
        "n_distinct_outputs": n_distinct,
        "distinct_fraction": (n_distinct / n_inputs) if n_inputs else 0.0,
        "repeatability": (repeat_matches / n_inputs) if n_inputs else 0.0,
        "basin_size_histogram": sorted(seen.values(), reverse=True),
    }
