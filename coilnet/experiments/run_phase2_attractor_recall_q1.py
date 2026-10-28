"""Direct storage-and-recall tests on physics-derived polymer coupling matrices."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from coilnet.physics import multibody

from .q1_common import append_unique, parser, read_csv, resolved_config, summarize, write_csv


def _energy(W, bias, state):
    return float(-0.5 * state @ W @ state - bias @ state)


def _async_fixed(W, bias, start, order, max_sweeps, energy_W=None, energy_bias=None):
    state = np.asarray(start, dtype=float).copy()
    seen = {tuple(state): 0}
    trace = [
        _energy(
            energy_W if energy_W is not None else W,
            energy_bias if energy_bias is not None else bias,
            state,
        )
    ]
    updates = 0
    for sweep in range(1, max_sweeps + 1):
        changed = False
        for i in order:
            field = W[i] @ state + bias[i]
            new = 1.0 if field > 0 else (-1.0 if field < 0 else state[i])
            changed |= new != state[i]
            state[i] = new
            updates += 1
            trace.append(
                _energy(
                    energy_W if energy_W is not None else W,
                    energy_bias if energy_bias is not None else bias,
                    state,
                )
            )
        if not changed:
            return state, "fixed", sweep, np.asarray(trace), updates
        key = tuple(state)
        if key in seen:
            return state, "cycle", sweep, np.asarray(trace), updates
        seen[key] = sweep
    return state, "timeout", max_sweeps, np.asarray(trace), updates


def _discover(W, starts, max_attractors, max_sweeps, rng):
    counts = Counter()
    for _ in range(starts):
        state = rng.choice([-1.0, 1.0], size=len(W))
        order = rng.permutation(len(W))
        final, status, *_ = _async_fixed(W, np.zeros(len(W)), state, order, max_sweeps)
        if status == "fixed":
            counts[tuple(final)] += 1
    return [(np.asarray(s, dtype=float), c) for s, c in counts.most_common(max_attractors)]


def _symmetrized(result):
    source = np.asarray(result["source"], dtype=float)
    receiver = np.asarray(result["receiver"], dtype=float)
    D = source / receiver
    K = D[:, None] * np.asarray(result["J"], dtype=float)
    np.fill_diagonal(K, 0.0)
    return D, K


def _diagonal_symmetry_error(W):
    W = np.asarray(W, dtype=float)
    n = len(W)
    equations, target = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] * W[j, i] > 1e-20:
                row = np.zeros(n)
                row[i], row[j] = 1.0, -1.0
                equations.append(row)
                target.append(np.log(abs(W[j, i] / W[i, j])))
    if not equations:
        return 1.0
    A = np.vstack(equations)
    A = np.vstack([A, np.ones(n)])
    b = np.r_[target, 0.0]
    logd, *_ = np.linalg.lstsq(A, b, rcond=None)
    D = np.exp(np.clip(logd, -30, 30))
    DW = D[:, None] * W
    return float(np.linalg.norm(DW - DW.T) / (np.linalg.norm(DW) + 1e-30))


def _corrupt(pattern, fraction, rng):
    cue = pattern.copy()
    count = max(1, int(round(len(cue) * fraction)))
    cue[rng.choice(len(cue), count, replace=False)] *= -1
    return cue


def run(config: dict, out: Path, resume=False, force=False):
    cfg = config["attractor"]
    raw_path = out / "attractor_recall_raw.csv"
    existing = read_csv(raw_path) if resume and not force else []
    expected_rows = len(cfg["systems"]) * len(cfg["sizes"]) * len(cfg["spreads"]) * len(
        config["main_seeds"]
    ) * len(cfg["threshold_fractions"]) * len(cfg["corruptions"]) * 2 + len(cfg["systems"]) * len(
        config["sensitivity_seeds"]
    ) * len(cfg["perturbations"])
    if len(existing) == expected_rows:
        merged = existing
        summary = summarize(
            merged,
            [
                "experiment",
                "system",
                "N",
                "spread",
                "threshold_fraction",
                "corruption",
                "representation",
                "perturbation",
            ],
            [
                "perfect_recall_rate",
                "mean_overlap",
                "mean_sweeps",
                "convergence_rate",
                "cycle_rate",
                "timeout_rate",
                "weighted_energy_violations",
                "ordinary_energy_violations",
                "symmetrizability_error",
                "exact_break",
                "practical_break",
            ],
            n_resamples=config["bootstrap_resamples"],
        )
        summary_path = write_csv(out / "attractor_recall_summary.csv", summary)
        return [raw_path, summary_path]
    rows = []
    for system in cfg["systems"]:
        for n in cfg["sizes"]:
            for spread in cfg["spreads"]:
                for seed in config["main_seeds"]:
                    rng_build = np.random.default_rng(10_000 + seed)
                    result = multibody.build_coupling(
                        n,
                        rng_build,
                        system=system,
                        spread=spread,
                        salt_M=0.01,
                        grid_n=cfg["grid_n"],
                    )
                    W = np.asarray(result["J"], dtype=float).copy()
                    np.fill_diagonal(W, 0.0)
                    D, K = _symmetrized(result)
                    rng_discover = np.random.default_rng(20_000 + seed)
                    attractors = _discover(
                        W,
                        cfg["discovery_starts"],
                        cfg["max_attractors"],
                        cfg["max_sweeps"],
                        rng_discover,
                    )
                    field_scale = float(np.median(np.sum(np.abs(W), axis=1))) + 1e-12
                    for threshold_fraction in cfg["threshold_fractions"]:
                        threshold = threshold_fraction * field_scale * np.ones(n)
                        for corruption in cfg["corruptions"]:
                            for representation in ("raw", "weighted"):
                                metrics = {
                                    "perfect": [],
                                    "overlap": [],
                                    "sweeps": [],
                                    "fixed": [],
                                    "cycle": [],
                                    "timeout": [],
                                    "energy_violations": [],
                                    "ordinary_energy_violations": [],
                                }
                                for pidx, (target, _basin) in enumerate(attractors):
                                    for repeat in range(cfg["cue_repeats"]):
                                        rng = np.random.default_rng(
                                            30_000_000 + seed * 10_000 + pidx * 100 + repeat
                                        )
                                        cue = _corrupt(target, corruption, rng)
                                        order = rng.permutation(n)
                                        matrix = W if representation == "raw" else K
                                        bias = (
                                            -threshold
                                            if representation == "raw"
                                            else -D * threshold
                                        )
                                        final, status, sweeps, trace, _ = _async_fixed(
                                            matrix,
                                            bias,
                                            cue,
                                            order,
                                            cfg["max_sweeps"],
                                            energy_W=K,
                                            energy_bias=-D * threshold,
                                        )
                                        ordinary_trace = [
                                            _energy(W, -threshold, cue),
                                            _energy(W, -threshold, final),
                                        ]
                                        overlap = float(final @ target / n)
                                        metrics["perfect"].append(
                                            float(np.array_equal(final, target))
                                        )
                                        metrics["overlap"].append(overlap)
                                        metrics["sweeps"].append(sweeps)
                                        for label in ("fixed", "cycle", "timeout"):
                                            metrics[label].append(float(status == label))
                                        metrics["energy_violations"].append(
                                            float(np.sum(np.diff(trace) > 1e-9))
                                        )
                                        metrics["ordinary_energy_violations"].append(
                                            float(np.sum(np.diff(ordinary_trace) > 1e-9))
                                        )

                                def mean(name):
                                    return (
                                        float(np.mean(metrics[name])) if metrics[name] else np.nan
                                    )

                                rows.append(
                                    {
                                        "experiment": "recall",
                                        "seed": seed,
                                        "system": system,
                                        "N": n,
                                        "spread": spread,
                                        "threshold_fraction": threshold_fraction,
                                        "corruption": corruption,
                                        "representation": representation,
                                        "n_attractors": len(attractors),
                                        "perfect_recall_rate": mean("perfect"),
                                        "mean_overlap": mean("overlap"),
                                        "mean_sweeps": mean("sweeps"),
                                        "convergence_rate": mean("fixed"),
                                        "cycle_rate": mean("cycle"),
                                        "timeout_rate": mean("timeout"),
                                        "weighted_energy_violations": mean("energy_violations"),
                                        "ordinary_energy_violations": mean(
                                            "ordinary_energy_violations"
                                        ),
                                        "status": "ok" if attractors else "no_fixed_attractor",
                                    }
                                )
    # Non-factorizable sensitivity at the representative N=30 (or closest available), spread=1.
    rep_n = min(cfg["sizes"], key=lambda x: abs(x - 30))
    for system in cfg["systems"]:
        for seed in config["sensitivity_seeds"]:
            result = multibody.build_coupling(
                rep_n,
                np.random.default_rng(40_000 + seed),
                system=system,
                spread=1.0,
                salt_M=0.01,
                grid_n=cfg["grid_n"],
            )
            W = np.asarray(result["J"], dtype=float).copy()
            np.fill_diagonal(W, 0)
            for eps in cfg["perturbations"]:
                rng = np.random.default_rng(50_000 + seed)
                noise = rng.normal(size=W.shape)
                np.fill_diagonal(noise, 0)
                noise *= np.linalg.norm(W) / (np.linalg.norm(noise) + 1e-30)
                error = _diagonal_symmetry_error(W + eps * noise)
                rows.append(
                    {
                        "experiment": "symmetry_sensitivity",
                        "seed": seed,
                        "system": system,
                        "N": rep_n,
                        "spread": 1.0,
                        "perturbation": eps,
                        "symmetrizability_error": error,
                        "exact_break": float(error > 1e-8),
                        "practical_break": float(error > 0.01),
                        "status": "ok",
                    }
                )
    key = [
        "experiment",
        "seed",
        "system",
        "N",
        "spread",
        "threshold_fraction",
        "corruption",
        "representation",
        "perturbation",
    ]
    merged = append_unique(raw_path, rows, key, resume=resume and not force)
    summary = summarize(
        merged,
        [
            "experiment",
            "system",
            "N",
            "spread",
            "threshold_fraction",
            "corruption",
            "representation",
            "perturbation",
        ],
        [
            "perfect_recall_rate",
            "mean_overlap",
            "mean_sweeps",
            "convergence_rate",
            "cycle_rate",
            "timeout_rate",
            "weighted_energy_violations",
            "ordinary_energy_violations",
            "symmetrizability_error",
            "exact_break",
            "practical_break",
        ],
        n_resamples=config["bootstrap_resamples"],
    )
    summary_path = write_csv(out / "attractor_recall_summary.csv", summary)
    return [raw_path, summary_path]


def main(argv=None):
    p = parser(__doc__)
    args = p.parse_args(argv)
    config = resolved_config(args)
    run(config, args.out, resume=args.resume, force=args.force)


if __name__ == "__main__":
    main()
