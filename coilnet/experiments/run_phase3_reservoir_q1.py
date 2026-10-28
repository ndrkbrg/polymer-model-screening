"""Confirmatory reservoir-computing comparison and sensitivity sweeps."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from coilnet.experiments._rc_common import train_eval_task
from coilnet.physics import multibody
from coilnet.reservoir import (
    CoilReservoir,
    echo_state_property,
    make_mackey_glass_prediction,
    make_narma10,
    memory_capacity,
    rescale_spectral_radius,
)

from .q1_common import (
    append_unique,
    holm_adjust,
    paired_comparison,
    parser,
    read_csv,
    resolved_config,
    summarize,
    write_csv,
)


MODELS = (
    "bistable_langevin",
    "graded_langevin",
    "physical_tanh",
    "random_tanh",
    "symmetric_tanh",
    "shuffled_tanh",
)


def make_narma20(T, rng):
    """Bounded NARMA-20 using the standard coefficients and U(0, 0.2) input."""
    u = rng.uniform(0.0, 0.2, size=T)
    y = np.zeros(T)
    for t in range(19, T - 1):
        y[t + 1] = (
            0.3 * y[t] + 0.05 * y[t] * np.sum(y[t - 19 : t + 1]) + 1.5 * u[t - 19] * u[t] + 0.1
        )
        if not np.isfinite(y[t + 1]):
            raise FloatingPointError("bounded NARMA-20 produced a non-finite target")
    return u, y


def _safe_narma10(T, seed):
    """Deterministically reject the rare divergent standard NARMA-10 realization."""
    for attempt in range(100):
        u, y = make_narma10(T, np.random.default_rng(seed + attempt))
        if np.all(np.isfinite(y)):
            return u, y, seed + attempt
    raise FloatingPointError("could not generate a finite NARMA-10 realization")


def _matrices(system, n, spread, radius, seed, resp_w=1.0):
    result = multibody.build_coupling(
        n,
        np.random.default_rng(seed),
        system=system,
        spread=spread,
        salt_M=0.01,
        grid_n=40 if n >= 40 else 28,
        resp_w=resp_w,
    )
    W = rescale_spectral_radius(result["J"], radius)
    sym = rescale_spectral_radius((W + W.T) / 2.0, radius)
    rng = np.random.default_rng(80_000 + seed)
    shuffled = W.copy()
    mask = ~np.eye(n, dtype=bool)
    values = shuffled[mask].copy()
    rng.shuffle(values)
    shuffled[mask] = values
    shuffled = rescale_spectral_radius(shuffled, radius)
    random = rescale_spectral_radius(rng.normal(size=(n, n)), radius)
    return W, sym, shuffled, random


def _reservoir(model, matrices, seed, tau=1.0, gain=None):
    W, sym, shuffled, random = matrices
    rng = np.random.default_rng(90_000 + seed)
    if model == "bistable_langevin":
        return CoilReservoir(
            W,
            dynamics="langevin",
            a=0.3,
            b=1.0,
            gain=0.5 if gain is None else gain,
            dt=0.02,
            relaxation_time=tau,
            noise=0.3,
            rng=rng,
        )
    if model == "graded_langevin":
        return CoilReservoir(
            W,
            dynamics="langevin",
            a=-1.0,
            b=1.0,
            gain=0.5 if gain is None else gain,
            dt=0.02,
            relaxation_time=tau,
            noise=0.1,
            rng=rng,
        )
    matrix = {
        "physical_tanh": W,
        "random_tanh": random,
        "symmetric_tanh": sym,
        "shuffled_tanh": shuffled,
    }[model]
    return CoilReservoir(
        matrix, dynamics="esn", gain=1.0 if gain is None else gain, leak=1.0, noise=1e-3, rng=rng
    )


def _score_main(config, cfg):
    rows = []
    T, washout = cfg["sequence_length"], cfg["washout"]
    for system in cfg["systems"]:
        for seed in config["main_seeds"]:
            matrices = _matrices(system, cfg["size"], cfg["spread"], cfg["radius"], seed)
            u10, y10, task_seed = _safe_narma10(T, 100_000 + seed * 100)
            u20, y20 = make_narma20(T, np.random.default_rng(110_000 + seed))
            umg, ymg = make_mackey_glass_prediction(T, rng=np.random.default_rng(120_000 + seed))
            for model in MODELS:
                task_values = {}
                status = "ok"
                for task, u, y in (
                    ("narma10", u10, y10),
                    ("narma20", u20, y20),
                    ("mackey_glass", umg, ymg),
                ):
                    try:
                        with np.errstate(over="raise", invalid="raise"):
                            value = train_eval_task(
                                _reservoir(model, matrices, seed), u, y, washout=washout
                            )["nrmse_test"]
                        if not np.isfinite(value):
                            raise FloatingPointError
                    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
                        value, status = np.nan, "nonfinite_dynamics"
                    task_values[f"{task}_nrmse"] = value
                try:
                    with np.errstate(over="raise", invalid="raise"):
                        mc = memory_capacity(
                            _reservoir(model, matrices, seed),
                            T=cfg["memory_length"],
                            max_delay=cfg["max_delay"],
                            washout=washout,
                            rng=np.random.default_rng(130_000 + seed),
                        )["MC"]
                        esp = echo_state_property(
                            _reservoir(model, matrices, seed),
                            T=max(300, washout + 100),
                            washout=washout,
                            n_starts=5,
                            tol=0.5 if "langevin" in model else 1e-3,
                            rng=np.random.default_rng(140_000 + seed),
                            common_noise=True,
                        )
                except (FloatingPointError, ValueError, np.linalg.LinAlgError):
                    mc, esp, status = (
                        np.nan,
                        {"max_distance": np.nan, "has_esp": False},
                        "nonfinite_dynamics",
                    )
                rows.append(
                    {
                        "experiment": "main",
                        "system": system,
                        "seed": seed,
                        "model": model,
                        "N": cfg["size"],
                        "spread": cfg["spread"],
                        "radius": cfg["radius"],
                        **task_values,
                        "memory_capacity": mc,
                        "narma10_task_seed": task_seed,
                        "esp_max_distance": esp["max_distance"],
                        "has_esp": float(esp["has_esp"]),
                        "status": status,
                    }
                )
    return rows


def _score_sensitivity(config, cfg):
    rows = []
    system = "PAA"
    # Full response/timescale phase grid.
    for seed in config["sensitivity_seeds"]:
        u, y, task_seed = _safe_narma10(cfg["sequence_length"], 150_000 + seed * 100)
        for width in cfg["response_width"]:
            matrices = _matrices(
                system, cfg["size"], cfg["spread"], cfg["radius"], seed, resp_w=width
            )
            for tau in cfg["tau"]:
                for gain in cfg["gain"]:
                    status = "ok"
                    try:
                        with np.errstate(over="raise", invalid="raise"):
                            score = train_eval_task(
                                _reservoir("graded_langevin", matrices, seed, tau=tau, gain=gain),
                                u,
                                y,
                                washout=cfg["washout"],
                            )["nrmse_test"]
                            esp = echo_state_property(
                                _reservoir("graded_langevin", matrices, seed, tau=tau, gain=gain),
                                T=max(300, cfg["washout"] + 100),
                                washout=cfg["washout"],
                                n_starts=4,
                                tol=0.5,
                                rng=np.random.default_rng(160_000 + seed),
                                common_noise=True,
                            )
                    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
                        score, esp, status = (
                            np.nan,
                            {"max_distance": np.nan, "has_esp": False},
                            "nonfinite_dynamics",
                        )
                    rows.append(
                        {
                            "experiment": "phase_map",
                            "system": system,
                            "seed": seed,
                            "model": "graded_langevin",
                            "N": cfg["size"],
                            "response_width": width,
                            "tau": tau,
                            "gain": gain,
                            "radius": cfg["radius"],
                            "washout": cfg["washout"],
                            "narma10_nrmse": score,
                            "esp_max_distance": esp["max_distance"],
                            "has_esp": float(esp["has_esp"]),
                            "narma10_task_seed": task_seed,
                            "status": status,
                        }
                    )
        # One-factor size/radius/washout sweeps for the physical tanh control.
        for n in cfg["sizes"]:
            matrices = _matrices(system, n, cfg["spread"], cfg["radius"], seed)
            score = train_eval_task(
                _reservoir("physical_tanh", matrices, seed),
                u,
                y,
                washout=cfg["washout"],
            )["nrmse_test"]
            mc = memory_capacity(
                _reservoir("physical_tanh", matrices, seed),
                T=max(cfg["memory_length"], n * 12),
                max_delay=min(n, cfg["max_delay"]),
                washout=cfg["washout"],
                rng=np.random.default_rng(170_000 + seed),
            )["MC"]
            rows.append(
                {
                    "experiment": "size",
                    "system": system,
                    "seed": seed,
                    "model": "physical_tanh",
                    "N": n,
                    "radius": cfg["radius"],
                    "washout": cfg["washout"],
                    "narma10_nrmse": score,
                    "memory_capacity": mc,
                    "status": "ok",
                }
            )
        for radius in cfg["spectral_radius"]:
            matrices = _matrices(system, cfg["size"], cfg["spread"], radius, seed)
            score = train_eval_task(
                _reservoir("physical_tanh", matrices, seed),
                u,
                y,
                washout=cfg["washout"],
            )["nrmse_test"]
            rows.append(
                {
                    "experiment": "radius",
                    "system": system,
                    "seed": seed,
                    "model": "physical_tanh",
                    "N": cfg["size"],
                    "radius": radius,
                    "washout": cfg["washout"],
                    "narma10_nrmse": score,
                    "status": "ok",
                }
            )
        matrices = _matrices(system, cfg["size"], cfg["spread"], cfg["radius"], seed)
        for washout in cfg["washouts"]:
            if washout >= len(u) - 2:
                continue
            score = train_eval_task(
                _reservoir("physical_tanh", matrices, seed), u, y, washout=washout
            )["nrmse_test"]
            rows.append(
                {
                    "experiment": "washout",
                    "system": system,
                    "seed": seed,
                    "model": "physical_tanh",
                    "N": cfg["size"],
                    "radius": cfg["radius"],
                    "washout": washout,
                    "narma10_nrmse": score,
                    "status": "ok",
                }
            )
    return rows


def run(config: dict, out: Path, resume=False, force=False):
    cfg = config["reservoir"]
    raw_path = out / "reservoir_raw.csv"
    existing = read_csv(raw_path) if resume and not force else []
    expected_rows = len(cfg["systems"]) * len(config["main_seeds"]) * len(MODELS) + len(
        config["sensitivity_seeds"]
    ) * (
        len(cfg["response_width"]) * len(cfg["tau"]) * len(cfg["gain"])
        + len(cfg["sizes"])
        + len(cfg["spectral_radius"])
        + len(cfg["washouts"])
    )
    if len(existing) == expected_rows:
        merged = existing
    else:
        rows = _score_main(config, cfg) + _score_sensitivity(config, cfg)
        merged = append_unique(
            raw_path,
            rows,
            [
                "experiment",
                "system",
                "seed",
                "model",
                "N",
                "response_width",
                "tau",
                "gain",
                "radius",
                "washout",
            ],
            resume=resume and not force,
        )
    summary = summarize(
        merged,
        [
            "experiment",
            "system",
            "model",
            "N",
            "response_width",
            "tau",
            "gain",
            "radius",
            "washout",
        ],
        [
            "narma10_nrmse",
            "narma20_nrmse",
            "mackey_glass_nrmse",
            "memory_capacity",
            "esp_max_distance",
            "has_esp",
        ],
        n_resamples=config["bootstrap_resamples"],
    )
    summary_path = write_csv(out / "reservoir_summary.csv", summary)
    main = [r for r in merged if r.get("experiment") == "main"]
    comparisons = []
    for system in cfg["systems"]:
        subset = [r for r in main if r.get("system") == system]
        for a, b in (
            ("graded_langevin", "bistable_langevin"),
            ("physical_tanh", "random_tanh"),
            ("physical_tanh", "symmetric_tanh"),
            ("physical_tanh", "shuffled_tanh"),
        ):
            comparison = paired_comparison(
                subset, "model", a, b, "narma10_nrmse", ["system", "seed"]
            )
            comparison["system"] = system
            comparisons.append(comparison)
    comparisons_path = write_csv(out / "reservoir_comparisons.csv", holm_adjust(comparisons))
    return [raw_path, summary_path, comparisons_path]


def main(argv=None):
    p = parser(__doc__)
    args = p.parse_args(argv)
    run(resolved_config(args), args.out, resume=args.resume, force=args.force)


if __name__ == "__main__":
    main()
