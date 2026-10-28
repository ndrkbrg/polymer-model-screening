"""Declared revision-stage controls; preserve all original scientific code and results."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from coilnet.experiments._rc_common import train_eval_task
from coilnet.experiments.run_phase2_attractor_recall_q1 import _diagonal_symmetry_error
from coilnet.experiments.run_phase3_reservoir_q1 import _safe_narma10
from coilnet.experiments.q1_common import write_csv, write_json
from coilnet.physics import multibody
from coilnet.reservoir import CoilReservoir, memory_capacity, rescale_spectral_radius

from coilnet.paths import ROOT, CONFIGS, WORK, output_directory

DEFAULT_OUT = WORK / "revision-run"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(array):
    return hashlib.sha256(np.asarray(array, dtype="<f8").tobytes()).hexdigest()


def reservoir_conditions(cfg, sensitivity):
    rows = []
    for response, a in cfg["response"].items():
        for noise in cfg["noise"]:
            rows.append(
                dict(
                    condition=f"{response}_D{noise:g}",
                    group="response_noise",
                    response=response,
                    a=a,
                    noise=noise,
                    topology="native",
                    tau=1.0,
                    input_scale=1.0,
                    driven_fraction=1.0,
                )
            )
    if sensitivity:
        base = next(r for r in rows if r["condition"] == "graded_D0.1")
        for factor, values in (
            ("topology", ["symmetric", "shuffled", "random"]),
            ("tau", [v for v in cfg["tau"] if v != 1]),
            ("input_scale", [v for v in cfg["input_scale"] if v != 1]),
            ("driven_fraction", [v for v in cfg["driven_fraction"] if v != 1]),
        ):
            for value in values:
                rows.append(
                    {**base, "condition": f"{factor}_{value}", "group": factor, factor: value}
                )
    return rows


def matrix_controls(W, radius, seed):
    native = np.asarray(W, dtype=float).copy()
    np.fill_diagonal(native, 0)
    symmetric = (native + native.T) / 2
    shuffled = native.copy()
    mask = ~np.eye(len(native), dtype=bool)
    shuffled[mask] = np.random.default_rng(810_000 + seed).permutation(native[mask])
    random = np.random.default_rng(820_000 + seed).normal(size=native.shape)
    np.fill_diagonal(random, 0)
    return {
        k: rescale_spectral_radius(v, radius)
        for k, v in dict(
            native=native, symmetric=symmetric, shuffled=shuffled, random=random
        ).items()
    }


def input_weights(n, seed, scale=1.0, fraction=1.0):
    base = np.random.default_rng(830_000 + seed).uniform(-1, 1, (n, 1))
    mask = np.zeros(n, dtype=bool)
    order = np.random.default_rng(840_000 + seed).permutation(n)
    mask[order[: round(n * fraction)]] = True
    return base * scale * mask[:, None]


def make_reservoir(condition, matrices, cfg, seed):
    return CoilReservoir(
        matrices[condition["topology"]],
        dynamics="langevin",
        a=condition["a"],
        b=cfg["b"],
        gain=cfg["gain"],
        dt=cfg["dt"],
        noise=condition["noise"],
        relaxation_time=condition["tau"],
        rng=np.random.default_rng(860_000 + seed),
        input_weights=input_weights(
            cfg["N"], seed, condition["input_scale"], condition["driven_fraction"]
        ),
    )


def safe_score(call):
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = call()
        if not all(np.all(np.isfinite(v)) for v in result.values()):
            raise FloatingPointError("non-finite result")
        return result, "ok"
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
        return None, f"{type(error).__name__}: {error}"


def reservoir_seed(config, system, seed, out):
    cfg = config["reservoir"]
    result = multibody.build_coupling(
        cfg["N"],
        np.random.default_rng(800_000 + seed),
        system=system,
        spread=cfg["spread"],
        salt_M=0.01,
        grid_n=cfg["grid_n"],
    )
    matrices = matrix_controls(result["J"], cfg["radius"], seed)
    u, y, task_seed = _safe_narma10(cfg["sequence_length"], 850_000 + seed * 100)
    memory_input = np.random.default_rng(870_000 + seed).uniform(-1, 1, cfg["memory_length"])
    saved = {
        **matrices,
        "raw_J": result["J"],
        "input": u,
        "target": y,
        "memory_input": memory_input,
        "input_weights": input_weights(cfg["N"], seed),
    }
    rows, delays = [], []
    for condition in reservoir_conditions(cfg, seed in config["sensitivity_seeds"]):
        score, score_status = safe_score(
            lambda: train_eval_task(
                make_reservoir(condition, matrices, cfg, seed),
                u,
                y,
                washout=cfg["washout"],
                reg=cfg["ridge"],
            )
        )
        mem, memory_status = safe_score(
            lambda: memory_capacity(
                make_reservoir(condition, matrices, cfg, seed),
                T=cfg["memory_length"],
                max_delay=cfg["max_delay"],
                washout=cfg["washout"],
                reg=cfg["ridge"],
                rng=np.random.default_rng(870_000 + seed),
            )
        )
        rows.append(
            {
                "system": system,
                "seed": seed,
                **condition,
                "N": cfg["N"],
                "narma10_nrmse": None if score is None else score["nrmse_test"],
                "nrmse_train": None if score is None else score["nrmse_train"],
                "n_train": None if score is None else score["n_train"],
                "n_test": None if score is None else score["n_test"],
                "memory_capacity": None if mem is None else mem["MC"],
                "task_status": score_status,
                "memory_status": memory_status,
                "narma10_task_seed": task_seed,
                "geometry_seed": 800_000 + seed,
                "noise_seed": 860_000 + seed,
                "memory_seed": 870_000 + seed,
                "matrix_sha256": array_hash(matrices[condition["topology"]]),
                "input_sha256": array_hash(u),
                "input_weights_sha256": array_hash(
                    input_weights(
                        cfg["N"], seed, condition["input_scale"], condition["driven_fraction"]
                    )
                ),
            }
        )
        if score is not None:
            saved[condition["condition"] + "_prediction"] = score["pred_test"]
        for delay in range(1, cfg["max_delay"] + 1):
            delays.append(
                {
                    "system": system,
                    "seed": seed,
                    "condition": condition["condition"],
                    "delay": delay,
                    "r2": None if mem is None else mem["per_delay"][delay - 1],
                    "status": memory_status,
                }
            )
    np.savez_compressed(out / f"reservoir-{system}-{seed}.npz", **saved)
    return {"reservoir": rows, "memory_delays": delays}


def fixed_trajectory(W, bias, start, order, max_sweeps):
    """Same updates/status as the original _async_fixed, omitting unused energy traces."""
    state = np.asarray(start, dtype=float).copy()
    seen = {tuple(state)}
    for sweep in range(1, max_sweeps + 1):
        changed = False
        for i in order:
            field = W[i] @ state + bias[i]
            new = 1.0 if field > 0 else (-1.0 if field < 0 else state[i])
            changed |= new != state[i]
            state[i] = new
        if not changed:
            return state, "fixed", sweep
        key = tuple(state)
        if key in seen:
            return state, "cycle", sweep
        seen.add(key)
    return state, "timeout", max_sweeps


def pattern_id(state):
    return "".join("1" if x > 0 else "0" for x in state)


def state_of(identifier):
    return np.array([1.0 if bit == "1" else -1.0 for bit in identifier])


def cue_and_order(identifier, seed, repeat, corruption):
    # Target identity, not its changing position in a top-k list, determines each cue.
    entropy = int.from_bytes(hashlib.sha256(identifier.encode()).digest()[:8], "little")
    rng = np.random.default_rng(np.random.SeedSequence([880_000 + seed, entropy, repeat]))
    cue = state_of(identifier)
    cue[rng.choice(len(cue), max(1, round(len(cue) * corruption)), replace=False)] *= -1
    return cue, rng.permutation(len(cue))


def discovery_bank(W, starts, max_sweeps, rng):
    records = []
    for i in range(starts):
        initial = rng.choice([-1.0, 1.0], len(W))
        order = rng.permutation(len(W))
        final, status, sweeps = fixed_trajectory(W, np.zeros(len(W)), initial, order, max_sweeps)
        records.append(
            {"start": i, "target": pattern_id(final), "status": status, "sweeps": sweeps}
        )
    return records


def target_scores(W, bias, identifier, seed, cfg):
    target = state_of(identifier)
    fields = W @ target + bias
    stable = bool(np.all(fields * target >= 0))
    rows = []
    for repeat in range(cfg["cue_repeats"]):
        cue, order = cue_and_order(identifier, seed, repeat, cfg["corruption"])
        final, status, sweeps = fixed_trajectory(W, bias, cue, order, cfg["max_sweeps"])
        rows.append(
            {
                "repeat": repeat,
                "perfect": float(status == "fixed" and np.array_equal(final, target)),
                "overlap": float(final @ target / len(target)),
                "status": status,
                "sweeps": sweeps,
                "target_fixed": float(stable),
            }
        )
    return rows


def aggregate_targets(identifiers, counts, cache):
    keys = ("perfect", "overlap", "sweeps", "target_fixed")
    result = {"n_targets": len(identifiers), "n_cues": sum(len(cache[k]) for k in identifiers)}
    if not identifiers:
        return {
            **result,
            **{k: None for k in keys},
            "weighted_recovery": None,
            "convergence_rate": None,
            "cycle_rate": None,
            "timeout_rate": None,
        }
    for key in keys:
        result[key] = float(np.mean([np.mean([r[key] for r in cache[k]]) for k in identifiers]))
    result["weighted_recovery"] = float(
        np.average(
            [np.mean([r["perfect"] for r in cache[k]]) for k in identifiers],
            weights=[counts[k] for k in identifiers],
        )
    )
    for label, name in (
        ("fixed", "convergence_rate"),
        ("cycle", "cycle_rate"),
        ("timeout", "timeout_rate"),
    ):
        result[name] = float(np.mean([r["status"] == label for k in identifiers for r in cache[k]]))
    return result


def attractor_seed(config, system, seed, out):
    cfg = config["attractor"]
    built = multibody.build_coupling(
        cfg["N"],
        np.random.default_rng(10_000 + seed),
        system=system,
        spread=cfg["spread"],
        salt_M=0.01,
        grid_n=cfg["grid_n"],
    )
    W = np.asarray(built["J"]).copy()
    np.fill_diagonal(W, 0)
    discovery = discovery_bank(
        W, max(cfg["budgets"]), cfg["max_sweeps"], np.random.default_rng(20_000 + seed)
    )
    validation = discovery_bank(
        W, cfg["validation_starts"], cfg["max_sweeps"], np.random.default_rng(21_000 + seed)
    )
    zero = np.zeros(len(W))
    all_ids = sorted({r["target"] for r in discovery if r["status"] == "fixed"})
    cache = {identifier: target_scores(W, zero, identifier, seed, cfg) for identifier in all_ids}
    banks = [
        {"system": system, "seed": seed, "bank": name, **r}
        for name, records in (("discovery", discovery), ("validation", validation))
        for r in records
    ]
    cue_rows = [
        {
            "system": system,
            "seed": seed,
            "experiment": "unperturbed",
            "value": 0,
            "target": identifier,
            **r,
        }
        for identifier, records in cache.items()
        for r in records
    ]
    summaries = []
    previous_ids = set()
    for budget in cfg["budgets"]:
        counts = Counter(r["target"] for r in discovery[:budget] if r["status"] == "fixed")
        ranked = sorted(counts, key=lambda k: (-counts[k], k))
        observed = len(counts)
        for top_k in cfg["top_k"]:
            selected = ranked[:top_k] if top_k else ranked
            summaries.append(
                {
                    "system": system,
                    "seed": seed,
                    "budget": budget,
                    "top_k": top_k,
                    "observed_attractors": observed,
                    "new_attractors": len(set(counts) - previous_ids),
                    "discovery_convergence": sum(counts.values()) / budget,
                    "retained_discovery_fraction": sum(counts[k] for k in selected) / budget,
                    "validation_known_fraction": sum(
                        r["status"] == "fixed" and r["target"] in counts for r in validation
                    )
                    / len(validation),
                    "validation_selected_fraction": sum(
                        r["status"] == "fixed" and r["target"] in selected for r in validation
                    )
                    / len(validation),
                    **aggregate_targets(selected, counts, cache),
                }
            )
        previous_ids = set(counts)
    counts = Counter(r["target"] for r in discovery if r["status"] == "fixed")
    top = sorted(counts, key=lambda k: (-counts[k], k))[:5]
    field_scale = float(np.median(np.sum(np.abs(W), axis=1))) + 1e-12
    noise = np.random.default_rng(50_000 + seed).normal(size=W.shape)
    np.fill_diagonal(noise, 0)
    noise *= np.linalg.norm(W) / (np.linalg.norm(noise) + 1e-30)
    sensitivity = []
    for experiment, values in (
        ("threshold", cfg["thresholds"]),
        ("perturbation", cfg["perturbations"]),
    ):
        for value in values:
            matrix = W if experiment == "threshold" else W + value * noise
            bias = -value * field_scale * np.ones(len(W)) if experiment == "threshold" else zero
            local = {
                identifier: target_scores(matrix, bias, identifier, seed, cfg) for identifier in top
            }
            row = {
                "system": system,
                "seed": seed,
                "experiment": experiment,
                "value": value,
                "field_scale": field_scale,
                **aggregate_targets(top, counts, local),
            }
            if experiment == "perturbation":
                row["symmetry_residual"] = _diagonal_symmetry_error(matrix)
            sensitivity.append(row)
            cue_rows.extend(
                {
                    "system": system,
                    "seed": seed,
                    "experiment": experiment,
                    "value": value,
                    "target": identifier,
                    **r,
                }
                for identifier, records in local.items()
                for r in records
            )
    np.savez_compressed(
        out / f"attractor-{system}-{seed}.npz",
        W=W,
        noise_direction=noise,
        source=built["source"],
        receiver=built["receiver"],
    )
    return {
        "discovery": summaries,
        "attractor_sensitivity": sensitivity,
        "discovery_bank": banks,
        "attractor_cues": cue_rows,
    }


def worker(task):
    kind, config, system, seed, out = task
    with threadpool_limits(limits=1):
        return (
            kind,
            system,
            seed,
            (reservoir_seed if kind == "reservoir" else attractor_seed)(
                config, system, seed, Path(out)
            ),
        )


def freeze_inputs(out, config_path, config):
    inputs = sorted((ROOT / "coilnet").rglob("*.py"))
    inputs += sorted((ROOT / "data/original").glob("*"))
    inputs = [p for p in inputs if p.is_file()]
    inputs += [config_path, ROOT / "docs/revision-protocol.md"]
    fingerprints = {
        p.resolve().relative_to(ROOT).as_posix()
        if p.resolve().is_relative_to(ROOT)
        else "external-config/" + p.name: sha(p)
        for p in inputs
    }
    path = out / "run_manifest.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old["input_hashes"] != fingerprints:
            raise RuntimeError(
                "Input/code/config changed; do not combine runs. Preserve this run and use a new output directory."
            )
        return old
    manifest = {
        "status": "running",
        "declared_date": config["analysis_date"],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "input_hashes": fingerprints,
        "starting_revision_hashes": {},
        "release_candidate": True,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "config": config,
        "original_outputs_overwritten": False,
    }
    write_json(path, manifest)
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=CONFIGS / "revision.json")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--kind", choices=("reservoir", "attractor", "both"), default="both")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--pilot", action="store_true")
    args = p.parse_args(argv)
    args.out = output_directory(args.out)
    if args.jobs < 1:
        p.error("--jobs must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.pilot:
        args.out = args.out / "pilot"
        config = {
            **config,
            "main_seeds": [config["main_seeds"][0]],
            "sensitivity_seeds": [config["sensitivity_seeds"][0]],
        }
    args.out = output_directory(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoints = args.out / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    manifest = freeze_inputs(args.out, args.config, config)
    kinds = ("reservoir", "attractor") if args.kind == "both" else (args.kind,)
    tasks = []
    for kind in kinds:
        for system in config["systems"]:
            for seed in (
                config["main_seeds"] if kind == "reservoir" else config["sensitivity_seeds"]
            ):
                if not (checkpoints / f"{kind}-{system}-{seed}.json").exists():
                    tasks.append((kind, config, system, seed, str(checkpoints)))
    start = time.monotonic()
    print(f"Starting {len(tasks)} seed jobs; {args.jobs} workers; pilot={args.pilot}", flush=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(worker, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            kind, system, seed, data = future.result()
            write_json(checkpoints / f"{kind}-{system}-{seed}.json", data)
            print(
                f"{i}/{len(tasks)} {kind}/{system}/{seed}; {time.monotonic() - start:.1f}s",
                flush=True,
            )
    tables = {}
    for checkpoint in sorted(checkpoints.glob("*.json")):
        for name, rows in json.loads(checkpoint.read_text(encoding="utf-8")).items():
            tables.setdefault(name, []).extend(rows)
    for name, rows in tables.items():
        write_csv(args.out / (name + "_raw.csv"), rows)
    manifest["status"] = (
        "complete"
        if all(
            (checkpoints / f"{kind}-{system}-{seed}.json").exists()
            for kind in ("reservoir", "attractor")
            for system in config["systems"]
            for seed in (
                config["main_seeds"] if kind == "reservoir" else config["sensitivity_seeds"]
            )
        )
        else "partial"
    )
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["raw_rows"] = {k: len(v) for k, v in tables.items()}
    manifest["output_hashes"] = {
        str(p.relative_to(args.out)): sha(p)
        for p in [
            *args.out.glob("*_raw.csv"),
            *checkpoints.glob("*.npz"),
            *checkpoints.glob("*.json"),
        ]
    }
    write_json(args.out / "run_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "rows": manifest["raw_rows"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
