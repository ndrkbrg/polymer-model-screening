"""Constrained 3x3/4x4 RBMs and two implementation-route simulations."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from coilnet.energybased.metrics import bars_and_stripes, bas_validity
from coilnet.energybased.physical import fabricate, train_degraded_local
from coilnet.energybased.rbm import RBM

from .q1_common import (
    parser,
    read_csv,
    resolved_config,
    summarize,
    write_csv,
)


def _local_mask(shape, bandwidth=0.25):
    nv, nh = shape
    vi = np.arange(nv)[:, None] / max(nv - 1, 1)
    hj = np.arange(nh)[None, :] / max(nh - 1, 1)
    return np.abs(vi - hj) <= bandwidth


def _projector(nonnegative=False, local=False, rank=None, iterations=12):
    def project(W):
        X = np.asarray(W, dtype=float).copy()
        mask = _local_mask(X.shape)
        for _ in range(iterations):
            if rank is not None:
                U, S, Vt = np.linalg.svd(X, full_matrices=False)
                X = (U[:, :rank] * S[:rank]) @ Vt[:rank]
            if nonnegative:
                X = np.maximum(X, 0.0)
            if local:
                X *= mask
        return X

    return project


def _constraint_specs(ranks):
    specs = [
        ("unconstrained", False, False, None),
        ("cooperative", True, False, None),
        ("local", False, True, None),
        ("cooperative+local", True, True, None),
    ]
    for rank in ranks:
        specs.extend(
            [
                (f"rank_{rank}", False, False, rank),
                (f"cooperative+rank_{rank}", True, False, rank),
                (f"local+rank_{rank}", False, True, rank),
                (f"cooperative+local+rank_{rank}", True, True, rank),
            ]
        )
    return specs


def _chain_samples(rbm, n_samples, burn_in):
    v = (rbm.rng.random((1, rbm.n_visible)) < 0.5).astype(float)
    for _ in range(burn_in):
        _, h = rbm.sample_hidden(v)
        _, v = rbm.sample_visible(h)
    output = np.empty((n_samples, rbm.n_visible))
    for i in range(n_samples):
        _, h = rbm.sample_hidden(v)
        _, v = rbm.sample_visible(h)
        output[i] = v[0]
    return output


def _visible_distribution(rbm):
    states = np.array(list(itertools.product([0.0, 1.0], repeat=rbm.n_visible)))
    logp = -rbm.free_energy(states)
    logp -= logsumexp(logp)
    return states, logp


def _mixing_time(rbm, steps):
    samples = _chain_samples(rbm, steps, 100)
    signal = samples.mean(axis=1)
    signal -= signal.mean()
    var = float(signal @ signal)
    if var <= 1e-20:
        return float(steps)
    tau = 1.0
    for lag in range(1, min(steps // 2, 500)):
        ac = float(signal[:-lag] @ signal[lag:] / var)
        if ac <= 0:
            break
        tau += 2.0 * ac
    return float(tau)


def _constraint_residuals(W, nonnegative, local, rank):
    norm = np.linalg.norm(W) + 1e-30
    negative = float(np.linalg.norm(np.minimum(W, 0.0)) / norm) if nonnegative else 0.0
    outside = float(np.linalg.norm(W * ~_local_mask(W.shape)) / norm) if local else 0.0
    rank_resid = 0.0
    if rank is not None and rank < min(W.shape):
        s = np.linalg.svd(W, compute_uv=False)
        rank_resid = float(s[rank] / (s[0] + 1e-30))
    return negative, outside, rank_resid


def _evaluate(rbm, size, cfg):
    data = bars_and_stripes(size)
    samples = _chain_samples(rbm, cfg["gibbs_samples"], cfg["gibbs_steps"])
    states, logp = _visible_distribution(rbm)
    index = {tuple(s): i for i, s in enumerate(states)}
    data_idx = [index[tuple(s)] for s in data]
    target_kl = float(np.mean(np.log(1.0 / len(data)) - logp[data_idx]))
    nll = float(-np.mean(logp[data_idx]))
    valid_set = {tuple(s.astype(int)) for s in data}
    valid_samples = [tuple(s.astype(int)) for s in samples if tuple(s.astype(int)) in valid_set]
    coverage = len(set(valid_samples)) / len(valid_set)
    duplicate = 1.0 - len(set(map(tuple, samples.astype(int)))) / len(samples)
    return {
        "valid_fraction": float(bas_validity(samples, size)),
        "kl_target_to_model": target_kl,
        "negative_log_likelihood": nll,
        "mode_coverage": float(coverage),
        "duplicate_rate": float(duplicate),
        "mixing_time": _mixing_time(rbm, cfg["mixing_steps"]),
    }


def _train(size, hidden, constraint, seed, cfg):
    data = bars_and_stripes(size)
    rbm = RBM(size * size, hidden, rng=np.random.default_rng(seed), weight_constraint=constraint)
    epochs = cfg["epochs_3"] if size == 3 else cfg["epochs_4"]
    rbm.train(data, epochs=epochs, lr=cfg["learning_rate"], k=1, batch_size=len(data))
    return rbm


def _rbm_rows(config, cfg):
    specs = _constraint_specs(cfg["ranks"])
    selected_4 = {
        "unconstrained",
        "cooperative",
        "local",
        "rank_4",
        "rank_8",
        "rank_12",
        "cooperative+local",
        "cooperative+local+rank_8",
    }
    for size, hidden in ((3, 16), (4, 32)):
        for name, nonnegative, local, rank in specs:
            seeds = (
                config["main_seeds"]
                if size == 3 or name in selected_4
                else config["sensitivity_seeds"]
            )
            for seed in seeds:
                constraint = (
                    None if name == "unconstrained" else _projector(nonnegative, local, rank)
                )
                rbm = _train(size, hidden, constraint, seed + size * 100_000, cfg)
                metrics = _evaluate(rbm, size, cfg)
                negative, outside, rank_resid = _constraint_residuals(
                    rbm.W, nonnegative, local, rank
                )
                yield {
                    "benchmark": f"bas_{size}x{size}",
                    "size": size,
                    "seed": seed,
                    "constraint": name,
                    "rank_target": "" if rank is None else rank,
                    **metrics,
                    "negative_residual": negative,
                    "local_residual": outside,
                    "rank_residual": rank_resid,
                    "approximate_intersection": float(max(negative, outside, rank_resid) > 1e-6),
                    "status": "ok",
                }


def _route_rows(config, cfg):
    data = bars_and_stripes(3)
    for seed in config["main_seeds"]:
        ideal = _train(3, 16, None, 300_000 + seed, cfg)
        ideal_metrics = _evaluate(ideal, 3, cfg)
        yield {
            "route": "ideal",
            "seed": seed,
            "error": 0.0,
            "levels": "",
            "setting": "baseline",
            **ideal_metrics,
        }
        # The predeclared principal Route-B setting is confirmed with 30 seeds.
        error, levels = 0.20, 8
        fabricated = RBM(9, 16, rng=np.random.default_rng(310_000 + seed))
        fabricated.W = fabricate(
            ideal.W,
            rel_noise=error,
            n_levels=levels,
            rng=np.random.default_rng(320_000 + seed),
        )
        fabricated.vbias, fabricated.hbias = ideal.vbias.copy(), ideal.hbias.copy()
        yield {
            "route": "offline_fabrication",
            "seed": seed,
            "error": error,
            "levels": levels,
            "setting": "confirmatory",
            **_evaluate(fabricated, 3, cfg),
        }
        # Confirmatory in-material settings.
        settings = [
            ("zero_mean_noise", dict(update_noise=0.5)),
            ("biased_update", dict(potentiation_ratio=0.5)),
            (
                "state_dependent",
                dict(memristive=True, mem_gamma=2.0, mem_dep_ratio=0.5, mem_wmax=2.5),
            ),
        ]
        for label, kwargs in settings:
            rbm = RBM(9, 16, rng=np.random.default_rng(330_000 + seed))
            train_degraded_local(
                rbm,
                data,
                epochs=cfg["epochs_3"],
                lr=cfg["learning_rate"],
                k=1,
                batch_size=len(data),
                rng=np.random.default_rng(340_000 + seed),
                **kwargs,
            )
            yield {
                "route": "in_material",
                "seed": seed,
                "error": "",
                "levels": "",
                "setting": label,
                **_evaluate(rbm, 3, cfg),
            }
    # The multidimensional Route-B sensitivity grid uses the 10 predeclared
    # sensitivity seeds, with a common trained target within each seed.
    for seed in config["sensitivity_seeds"]:
        ideal = _train(3, 16, None, 370_000 + seed, cfg)
        for error in cfg["fabrication_errors"]:
            for levels in cfg["quantization_levels"]:
                fabricated = RBM(9, 16, rng=np.random.default_rng(380_000 + seed))
                fabricated.W = fabricate(
                    ideal.W,
                    rel_noise=error,
                    n_levels=levels,
                    rng=np.random.default_rng(390_000 + seed),
                )
                fabricated.vbias, fabricated.hbias = ideal.vbias.copy(), ideal.hbias.copy()
                yield {
                    "route": "offline_fabrication",
                    "seed": seed,
                    "error": error,
                    "levels": levels,
                    "setting": "factorial",
                    **_evaluate(fabricated, 3, cfg),
                }
    # Multidimensional route-A sensitivity on 10 seeds.
    for seed in config["sensitivity_seeds"]:
        for noise in (0.0, 0.1, 0.25, 0.5, 1.0):
            for gamma in (1.0, 2.0, 3.0):
                for dep in (1.0, 0.7, 0.4, 0.2):
                    rbm = RBM(9, 16, rng=np.random.default_rng(350_000 + seed))
                    train_degraded_local(
                        rbm,
                        data,
                        epochs=cfg["epochs_3"],
                        lr=cfg["learning_rate"],
                        k=1,
                        batch_size=len(data),
                        update_noise=noise,
                        memristive=True,
                        mem_gamma=gamma,
                        mem_dep_ratio=dep,
                        mem_wmax=2.5,
                        rng=np.random.default_rng(360_000 + seed),
                    )
                    yield {
                        "route": "in_material_sensitivity",
                        "seed": seed,
                        "setting": "grid",
                        "update_noise": noise,
                        "mem_gamma": gamma,
                        "mem_dep_ratio": dep,
                        **_evaluate(rbm, 3, cfg),
                    }


def _checkpoint_rows(path, generated, key_columns, existing=(), every=10):
    rows = list(existing)
    keys = {tuple(str(row.get(k, "")) for k in key_columns) for row in rows}
    additions = 0
    for row in generated:
        key = tuple(str(row.get(k, "")) for k in key_columns)
        if key in keys:
            continue
        rows.append(row)
        keys.add(key)
        additions += 1
        if additions % every == 0:
            write_csv(path, rows)
    write_csv(path, rows)
    return rows


def run(config: dict, out: Path, resume=False, force=False):
    cfg = config["rbm"]
    raw_path = out / "rbm_raw.csv"
    selected_4 = {
        "unconstrained",
        "cooperative",
        "local",
        "rank_4",
        "rank_8",
        "rank_12",
        "cooperative+local",
        "cooperative+local+rank_8",
    }
    specs = _constraint_specs(cfg["ranks"])
    expected_rbm_rows = sum(
        len(config["main_seeds"])
        if size == 3 or name in selected_4
        else len(config["sensitivity_seeds"])
        for size in (3, 4)
        for name, *_ in specs
    )
    existing_rbm = read_csv(raw_path) if resume and not force else []
    if len(existing_rbm) == expected_rbm_rows:
        rbm_rows = existing_rbm
    else:
        rbm_rows = _checkpoint_rows(
            raw_path,
            _rbm_rows(config, cfg),
            ["benchmark", "seed", "constraint"],
            existing=existing_rbm,
        )
    rbm_summary = summarize(
        rbm_rows,
        ["benchmark", "size", "constraint", "rank_target"],
        [
            "valid_fraction",
            "kl_target_to_model",
            "negative_log_likelihood",
            "mode_coverage",
            "duplicate_rate",
            "mixing_time",
            "negative_residual",
            "local_residual",
            "rank_residual",
            "approximate_intersection",
        ],
        n_resamples=config["bootstrap_resamples"],
    )
    rbm_summary_path = write_csv(out / "rbm_summary.csv", rbm_summary)
    routes_path = out / "routes_raw.csv"
    expected_route_rows = len(config["main_seeds"]) * 5 + len(config["sensitivity_seeds"]) * (
        len(cfg["fabrication_errors"]) * len(cfg["quantization_levels"]) + 5 * 3 * 4
    )
    existing_routes = read_csv(routes_path) if resume and not force else []
    if len(existing_routes) == expected_route_rows:
        route_rows = existing_routes
    else:
        route_rows = _checkpoint_rows(
            routes_path,
            _route_rows(config, cfg),
            [
                "route",
                "seed",
                "setting",
                "error",
                "levels",
                "update_noise",
                "mem_gamma",
                "mem_dep_ratio",
            ],
            existing=existing_routes,
        )
    route_summary = summarize(
        route_rows,
        ["route", "setting", "error", "levels", "update_noise", "mem_gamma", "mem_dep_ratio"],
        [
            "valid_fraction",
            "kl_target_to_model",
            "negative_log_likelihood",
            "mode_coverage",
            "duplicate_rate",
            "mixing_time",
        ],
        n_resamples=config["bootstrap_resamples"],
    )
    route_summary_path = write_csv(out / "routes_summary.csv", route_summary)
    return [raw_path, rbm_summary_path, routes_path, route_summary_path]


def main(argv=None):
    p = parser(__doc__)
    args = p.parse_args(argv)
    run(resolved_config(args), args.out, resume=args.resume, force=args.force)


if __name__ == "__main__":
    main()
