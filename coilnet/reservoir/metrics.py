"""Reservoir-quality metrics, independent of any particular task.

A reservoir's usefulness is largely determined by properties of its FREE-RUNNING dynamics,
before any readout is trained. These are the standard ones from the reservoir-computing /
"computational power of dynamical systems" literature (in particular the CHARC framework,
Dale, Miller & Stepney, "Reservoir Computing as a Model for In-Materio Computing", 2019, and
Jaeger's original echo-state-network reports):

  * memory capacity      -- how many past inputs can a linear readout recover from the state?
  * kernel rank          -- how well does the reservoir SEPARATE distinct inputs (task-independent
                             proxy for "can a linear readout tell these apart at all")?
  * generalization rank  -- how well does it GENERALISE (collapse near-identical inputs together,
                             rather than amplifying noise)? Kernel rank and generalization rank
                             trade off against each other (Dale et al. call this the
                             kernel-quality / generalization-quality trade-off).
  * echo state property  -- does the reservoir have fading memory (forget its own initial
                             condition), which is the basic precondition for reservoir computing
                             to make sense at all?
  * Lyapunov exponent     -- is the free-running dynamics contractive, chaotic, or right at the
                             "edge of chaos" (~0), which is where memory and expressivity both
                             tend to be maximised?

These are all measured on the reservoir's OWN dynamics -- they say nothing about whether J came
from a symmetric or asymmetric physical mechanism, which is exactly why they are the right
common ground for comparing the physically-derived, asymmetric J against a textbook ESN.
"""

from __future__ import annotations

import numpy as np

from .readout import add_bias, ridge_fit, ridge_predict

__all__ = [
    "memory_capacity",
    "kernel_rank",
    "generalization_rank",
    "echo_state_property",
    "lyapunov_estimate",
]


def memory_capacity(reservoir, T=2000, max_delay=40, washout=200, reg=1e-6, rng=None):
    """Linear memory capacity (Jaeger, 2001): how many past inputs can be read out linearly.

    Drives the reservoir with a single i.i.d. Uniform[-1, 1] input stream. For each delay
    ``k = 1..max_delay``, fits a ridge readout that tries to reconstruct ``u(t - k)`` from the
    state ``x(t)``, and scores it by the squared Pearson correlation (``r**2``, the standard
    "forgetting curve" score) between prediction and target on a held-out half of the run.

    Returns
    -------
    dict with keys:
        "MC"        -- total memory capacity, sum_k r_k**2 (bounded above by the reservoir's
                       state dimension N for a linear reservoir; see Jaeger 2001).
        "per_delay" -- list of the individual r_k**2 values (the "forgetting curve").
        "max_delay" -- the max delay that was tested.
    """
    rng = np.random.default_rng() if rng is None else rng
    u = rng.uniform(-1.0, 1.0, size=T)
    X_full = reservoir.run(u, washout=0, reset=True)  # (T, N), aligned with u by index

    per_delay = []
    for k in range(1, max_delay + 1):
        start = washout + k
        if start >= T:
            per_delay.append(0.0)
            continue
        X = add_bias(X_full[start:])
        y = u[start - k : T - k]  # = u(t - k) for t = start..T-1

        n = len(y)
        split = n // 2
        if split < 1 or (n - split) < 1:
            per_delay.append(0.0)
            continue
        Xtr, Xte = X[:split], X[split:]
        ytr, yte = y[:split], y[split:]

        Wout = ridge_fit(Xtr, ytr, reg=reg)
        pred = ridge_predict(Xte, Wout)

        if np.std(yte) < 1e-12 or np.std(pred) < 1e-12:
            r2 = 0.0
        else:
            r = np.corrcoef(pred, yte)[0, 1]
            r2 = float(r**2) if np.isfinite(r) else 0.0
        per_delay.append(max(0.0, r2))

    return {"MC": float(np.sum(per_delay)), "per_delay": per_delay, "max_delay": max_delay}


def _final_states(reservoir, streams, washout, common_noise=True, noise_rng=None):
    """Run the reservoir (fresh, reset each time) on each input stream; collect final states.

    ``common_noise`` (same "common random numbers" trick as ``echo_state_property``, default
    True): for a STOCHASTIC reservoir (``noise > 0``), every stream is, by default, reseeded to
    the SAME noise realization before it runs. This matters because ``kernel_rank`` and
    ``generalization_rank`` are meant to measure how much the reservoir's FINAL STATE is
    separated by DIFFERENCES IN THE INPUT -- not by independent noise. With independent noise
    per stream (``common_noise=False``), every stream's final state differs from every other
    purely because of its own unshared noise draw, regardless of what the input actually was --
    so the measured rank saturates near N (full rank) even for a reservoir that barely reacts to
    its input at all (e.g. a linear, ``a=0`` reservoir, which cannot genuinely separate inputs
    that way). Freezing the noise realization across streams isolates the input's effect, which
    is the quantity these metrics are meant to report. Has no effect when
    ``reservoir.noise == 0`` (nothing to freeze), and the reservoir's original ``rng`` object is
    restored before this function returns either way. ``noise_rng`` supplies the single seed
    draw used to build the shared noise realization (kept separate from whatever ``rng`` the
    caller uses to build the streams themselves).
    """
    noise_rng = np.random.default_rng() if noise_rng is None else noise_rng
    use_common_noise = bool(common_noise) and float(getattr(reservoir, "noise", 0.0)) != 0.0
    common_seed = int(noise_rng.integers(0, 2**31 - 1)) if use_common_noise else None
    orig_rng = reservoir.rng

    try:
        states = []
        for u in streams:
            if use_common_noise:
                reservoir.rng = np.random.default_rng(common_seed)
            X = reservoir.run(u, washout=washout, reset=True)
            states.append(X[-1])
        return np.array(states)
    finally:
        reservoir.rng = orig_rng


def kernel_rank(
    reservoir, n_streams=None, T=200, washout=100, rng=None, tol=None, common_noise=True
):
    """Kernel (separation) rank: how many linearly-independent responses N distinct inputs get.

    Drives the reservoir with ``n_streams`` DISTINCT i.i.d. Uniform[-1, 1] input streams (each
    from a freshly-reset state), collects the final state reached under each stream, stacks
    them into an ``(n_streams, N)`` matrix, and returns its numerical rank (SVD-based, via
    ``numpy.linalg.matrix_rank``). A rich, high-dimensional reservoir maps distinct inputs to
    (close to) linearly-independent states -- giving a linear readout the best chance of
    separating them; a "frozen" reservoir collapses everything together and the rank saturates
    at 1. Defaults to ``n_streams = reservoir.N`` (enough streams to probe full rank).

    ``common_noise`` (default True): see ``_final_states``. For a stochastic reservoir, this
    must be True for the returned rank to reflect INPUT-driven separation rather than noise --
    with independent noise per stream, the rank saturates near N regardless of the input.
    """
    rng = np.random.default_rng() if rng is None else rng
    n_streams = reservoir.N if n_streams is None else n_streams
    streams = [rng.uniform(-1.0, 1.0, size=T) for _ in range(n_streams)]
    S = _final_states(reservoir, streams, washout, common_noise=common_noise, noise_rng=rng)
    return int(np.linalg.matrix_rank(S, tol=tol))


def generalization_rank(
    reservoir,
    n_streams=None,
    T=200,
    washout=100,
    perturb=1e-3,
    rng=None,
    tol=None,
    common_noise=True,
):
    """Generalization rank: how much the reservoir separates NOISY VARIANTS of one input.

    Same construction as ``kernel_rank``, but every stream is a small perturbation of a single
    shared base stream rather than ``n_streams`` independent streams (following the CHARC
    definitions, Dale, Miller & Stepney 2019). A reservoir that generalises well treats
    near-identical inputs as near-identical -- collapsing them to a LOW-rank set of final
    states -- rather than amplifying the tiny input differences. So, unlike kernel rank, LOWER
    generalization rank is better (it signals the reservoir isn't just amplifying noise).
    Defaults to ``n_streams = reservoir.N``.

    ``common_noise`` (default True): see ``_final_states``. For a stochastic reservoir, this
    must be True for the returned rank to reflect the input PERTURBATION's effect rather than
    the reservoir's own unshared noise -- with independent noise per stream, the rank saturates
    near N regardless of how small ``perturb`` is.
    """
    rng = np.random.default_rng() if rng is None else rng
    n_streams = reservoir.N if n_streams is None else n_streams
    base = rng.uniform(-1.0, 1.0, size=T)
    streams = [base + perturb * rng.standard_normal(T) for _ in range(n_streams)]
    S = _final_states(reservoir, streams, washout, common_noise=common_noise, noise_rng=rng)
    return int(np.linalg.matrix_rank(S, tol=tol))


def echo_state_property(
    reservoir, T=500, washout=100, n_starts=5, tol=1e-3, rng=None, common_noise=True
):
    """Empirically test the echo state property (fading memory of the initial condition).

    Drives the SAME input sequence from ``n_starts`` different random initial states. If the
    reservoir has fading memory, the influence of the (different) initial conditions should
    decay away and the trajectories should converge to (nearly) the same state regardless of
    where they started -- the defining property that makes "the state is a function of the
    input history" a sensible statement at all.

    ``common_noise`` (CHARC / RC-literature "common random numbers" trick, default True):
    for a STOCHASTIC reservoir (``noise > 0``), each of the ``n_starts`` trajectories is, by
    default, reseeded to the SAME noise realization before it runs. This matters specifically
    for substrates like ours: a bistable (double-well) reservoir can be MULTISTABLE under its
    own deterministic dynamics (different initial conditions get permanently trapped in
    different wells, so ESP legitimately fails), yet still have fading memory of the initial
    condition once thermal noise is strong enough to let trajectories hop between wells. That
    fading-memory question is about whether a SHARED noise realization erases the initial
    condition -- not about whether independent noise happens to average out. With independent
    noise per start (``common_noise=False``), each trajectory is perturbed along its own random
    direction forever, so the final states never actually coincide even when the underlying
    (noise-free) dynamics is strongly contractive -- ESP would (wrongly) always read as failed
    for any reservoir that needs noise to escape multistability. Freezing the realization
    isolates the question ESP is meant to ask. Has no effect when ``reservoir.noise == 0``
    (nothing to freeze), and the reservoir's original ``rng`` object is restored before this
    function returns either way.

    Returns
    -------
    dict with keys:
        "max_distance" -- largest pairwise Euclidean distance between the final states reached
                           from the different initial conditions.
        "has_esp"       -- True if that distance is below ``tol`` (converged).
    """
    rng = np.random.default_rng() if rng is None else rng
    u = rng.uniform(-1.0, 1.0, size=T)

    use_common_noise = bool(common_noise) and float(getattr(reservoir, "noise", 0.0)) != 0.0
    common_seed = int(rng.integers(0, 2**31 - 1)) if use_common_noise else None
    orig_rng = reservoir.rng

    try:
        finals = []
        for _ in range(n_starts):
            init_state = rng.normal(scale=1.0, size=reservoir.N)
            reservoir.reset(init_state)
            if use_common_noise:
                reservoir.rng = np.random.default_rng(common_seed)
            X = reservoir.run(u, washout=washout, reset=False)
            finals.append(X[-1])
        finals = np.array(finals)

        dmax = 0.0
        for i in range(len(finals)):
            for j in range(i + 1, len(finals)):
                d = float(np.linalg.norm(finals[i] - finals[j]))
                dmax = max(dmax, d)

        return {"max_distance": dmax, "has_esp": bool(dmax < tol)}
    finally:
        reservoir.rng = orig_rng


def lyapunov_estimate(reservoir, T=1000, washout=200, perturb=1e-7, rng=None):
    """Estimate the largest Lyapunov exponent of the reservoir's free-running (driven) map.

    Uses Benettin's method: run one trajectory, clone it with a tiny state perturbation
    ``perturb`` at the end of the washout, then drive BOTH clones with the identical
    subsequent input, step by step, renormalising the perturbed trajectory back to distance
    ``perturb`` (in the direction it actually diverged/converged) after every step so the
    estimate reflects the LOCAL expansion rate throughout the run rather than saturating once
    the two trajectories are macroscopically different. The returned value is the mean of
    ``log(distance_after_step / perturb)`` across all steps -- the standard finite-time
    Lyapunov-exponent estimator.

    A positive rate means nearby trajectories diverge exponentially (chaotic); negative means
    they contract (fading memory dominates, "boring" but safe); ~0 is the "edge of chaos"
    where reservoirs are often most computationally useful.

    Noise is temporarily set to 0 for this measurement: the exponent is a property of the
    (driven) deterministic skeleton of the dynamics, and independent noise added to the two
    clones would inject its own spurious divergence unrelated to the map's sensitivity.
    """
    rng = np.random.default_rng() if rng is None else rng
    inputs = rng.uniform(-1.0, 1.0, size=T)

    orig_noise = reservoir.noise
    orig_state = reservoir.state.copy()
    try:
        reservoir.noise = 0.0
        reservoir.reset()
        if washout > 0:
            reservoir.run(inputs[:washout], washout=0, reset=False)
        base_state = reservoir.state.copy()

        direction = rng.standard_normal(reservoir.N)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm

        state_a = base_state.copy()
        state_b = base_state + perturb * direction

        log_rates = []
        for t in range(washout, T):
            u = inputs[t]
            reservoir.state = state_a
            state_a = reservoir.step(u)
            reservoir.state = state_b
            state_b = reservoir.step(u)

            diff = state_b - state_a
            dist = float(np.linalg.norm(diff))
            dist = max(dist, 1e-300)
            log_rates.append(np.log(dist / perturb))

            # Renormalise: keep the perturbation magnitude constant, direction preserved.
            state_b = state_a + (perturb / dist) * diff

        return float(np.mean(log_rates)) if log_rates else 0.0
    finally:
        reservoir.noise = orig_noise
        reservoir.reset(orig_state)
