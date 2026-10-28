"""Shared reservoir construction and chronological train/test helpers.

Legacy operating-point settings are retained to reproduce the original cohort.
They are numerical choices, not experimentally calibrated material parameters."""

from __future__ import annotations

import numpy as np

from coilnet.physics import multibody as mb
from coilnet.reservoir import CoilReservoir, add_bias, nrmse, ridge_fit, ridge_predict
from coilnet.reservoir.engine import rescale_spectral_radius

__all__ = ["operating_point", "physics_reservoir", "train_eval_task"]


def operating_point():
    """A good default operating point for the bistable, physics-derived reservoir.

    These bistable (double-well) coils are MULTISTABLE under purely deterministic dynamics
    (different initial conditions get trapped in different wells forever, so the echo state
    property fails), but fading memory of the initial condition is RESTORED once there is
    enough thermal noise to let trajectories hop between wells -- confirmed empirically by
    ``run_phase15_esp_sweep.py`` across gain in {0.2, 0.5, 1.0} x noise in {0.0, 0.1, 0.3, 0.5}.
    ``gain=0.5`` keeps the recurrent coupling sub-critical; ``noise=0.3`` sits robustly inside
    the ESP-holding region found by that sweep (not merely at its edge), at a memory capacity
    that is representative of -- not exceptional among -- the ESP-satisfying cells (the sweep
    finds MC is highest right where noise is near zero and ESP still fails; every cell where
    ESP holds pays a similar, substantial MC cost). Noise is a FUNCTIONAL ingredient of this
    substrate, not a nuisance to be minimised.
    """
    return dict(dynamics="langevin", a=0.3, b=1.0, dt=0.05, gain=0.5, noise=0.3)


def physics_reservoir(system="PAA", spread=1.0, N=40, radius=0.9, seed=0, **overrides):
    """Build a ``CoilReservoir`` from the physically-derived coupling J, at the operating point.

    ``J`` comes from ``coilnet.physics.multibody.build_coupling`` (asymmetric: a receiver-vs-
    source split baked into the physics -- see Phase 1/1b) and is rescaled to a target spectral
    radius, the standard reservoir-computing "richness" knob; the raw physical radius has no
    special RC meaning of its own, so it is measured and then rescaled rather than trusted
    as-is (mirroring ``rescale_spectral_radius``'s own docstring reasoning).

    ``overrides`` are merged on top of ``operating_point()`` and forwarded to
    ``CoilReservoir(...)`` (e.g. ``gain=``, ``noise=`` for the ESP/MC sweep). A dedicated RNG
    seeded from ``seed`` drives the reservoir's own randomness (input weights, noise draws)
    unless the caller supplies ``rng=`` explicitly in ``overrides``.
    """
    rng_j = np.random.default_rng(seed)
    J = mb.build_coupling(N, rng_j, system=system, spread=spread, salt_M=0.01, grid_n=40)["J"]
    J = rescale_spectral_radius(J, radius)

    kwargs = operating_point()
    kwargs.update(overrides)
    kwargs.setdefault("rng", np.random.default_rng(seed + 1000))
    return CoilReservoir(J, **kwargs)


def train_eval_task(reservoir, u, y, washout=200, reg=1e-6, split=0.5):
    """Run ``reservoir`` on input ``u``, ridge-fit a linear readout to ``y``, score by NRMSE.

    Drives the reservoir on the FULL input series ``u`` (from a reset state, recording the
    input alongside the state via ``record_input=True`` so the readout can use "state + current
    input" as its feature vector -- standard RC practice), discards the first ``washout``
    samples of the resulting state trajectory AND of the target ``y`` (so the two stay aligned
    index-for-index), adds a bias column, splits the remainder into a training prefix and a
    held-out test suffix (fraction ``split`` used for training), ridge-fits ``Wout`` on the
    training split only, and reports NRMSE on both splits.

    Returns
    -------
    dict with keys:
        "nrmse_train", "nrmse_test" -- NRMSE (``coilnet.reservoir.readout.nrmse``) on the train
                                        / test split.
        "pred_test", "target_test"  -- the test-split prediction and target arrays (handy for
                                        plotting predicted-vs-target).
        "n_train", "n_test"         -- sizes of the two splits.
    """
    X = reservoir.run(u, washout=washout, reset=True, record_input=True)
    X = add_bias(X)
    y = np.asarray(y, dtype=float)[washout:]
    if len(y) != len(X):
        raise ValueError(f"washed-out target length {len(y)} != state length {len(X)}")

    n = len(y)
    split_idx = int(round(split * n))
    split_idx = max(1, min(n - 1, split_idx))
    Xtr, Xte = X[:split_idx], X[split_idx:]
    ytr, yte = y[:split_idx], y[split_idx:]

    Wout = ridge_fit(Xtr, ytr, reg=reg)
    pred_tr = ridge_predict(Xtr, Wout)
    pred_te = ridge_predict(Xte, Wout)

    return {
        "nrmse_train": nrmse(pred_tr, ytr),
        "nrmse_test": nrmse(pred_te, yte),
        "pred_test": pred_te,
        "target_test": yte,
        "n_train": int(split_idx),
        "n_test": int(n - split_idx),
    }
