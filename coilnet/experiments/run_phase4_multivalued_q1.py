"""Quantitative register-versus-processor tests for multi-state polymer models."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from coilnet.hopfield import HopfieldNetwork
from coilnet.multistate import addressability, landscape, operations
from coilnet.physics import multibody

from .q1_common import append_unique, parser, read_csv, resolved_config, summarize, write_csv


def _alphabet(J, starts, max_alphabet, seed):
    basins, nonconverged = landscape.find_attractors(
        J, trials=starts, max_sweeps=80, rng=np.random.default_rng(seed)
    )
    ranked = sorted(basins.items(), key=lambda kv: -kv[1])
    states = []
    for key, count in ranked:
        state = np.asarray(key, dtype=float)
        if all(np.mean(state != old) >= 0.15 for old in states):
            states.append(state)
        if len(states) >= max_alphabet:
            break
    total = max(1, sum(basins.values()))
    basin_fracs = np.asarray([v / total for _, v in ranked], dtype=float)
    entropy = float(-np.sum(basin_fracs * np.log(basin_fracs + 1e-30)))
    return states, len(basins), nonconverged, entropy, float(max(basin_fracs, default=0.0))


def _nearest(state, alphabet):
    distances = [np.mean(state != target) for target in alphabet]
    idx = int(np.argmin(distances))
    return idx, float(distances[idx])


def _settle(J, state, field, rng):
    return HopfieldNetwork(J, bias=field).update_async(state, rng=rng, max_sweeps=60)


def _is_single_cycle(mapping):
    n = len(mapping)
    if sorted(mapping) != list(range(n)):
        return False
    visited, cur = set(), 0
    for _ in range(n):
        if cur in visited:
            break
        visited.add(cur)
        cur = mapping[cur]
    return len(visited) == n and cur == 0


def _candidate_fields(kind, J, fraction, trials, rng):
    n = len(J)
    scale = operations.field_scale(J, rng=rng)
    if kind == "scalar":
        return [(fraction * scale * np.ones(n),)]
    if kind == "random":
        output = []
        for _ in range(trials):
            v = rng.normal(size=n)
            output.append((fraction * scale * v / (np.linalg.norm(v) / np.sqrt(n) + 1e-30),))
        return output
    U, _, _ = np.linalg.svd(J, full_matrices=False)
    if kind == "structured":
        output = []
        basis = U[:, : min(3, n)]
        for _ in range(trials):
            v = basis @ rng.normal(size=basis.shape[1])
            output.append((fraction * scale * v / (np.linalg.norm(v) / np.sqrt(n) + 1e-30),))
        return output
    output = []
    basis = U[:, : min(3, n)]
    for _ in range(trials):
        pair = []
        for _step in range(2):
            v = basis @ rng.normal(size=basis.shape[1])
            pair.append(fraction * scale * v / (np.linalg.norm(v) / np.sqrt(n) + 1e-30))
        output.append(tuple(pair))
    return output


def _evaluate_field(J, alphabet, fields, repeats, seed):
    n = len(alphabet)
    outcomes = [[Counter() for _ in range(n)] for _ in range(1)]
    for origin, state in enumerate(alphabet):
        for repeat in range(repeats):
            current = state.copy()
            rng = np.random.default_rng(seed + origin * 100 + repeat)
            for field in fields:
                current = _settle(J, current, field, rng)
            target, distance = _nearest(current, alphabet)
            outcomes[0][origin][target] += int(distance <= 0.08)
            if distance > 0.08:
                outcomes[0][origin][-1] += 1
    mapping, consistencies = [], []
    for counter in outcomes[0]:
        target, count = counter.most_common(1)[0]
        mapping.append(target)
        consistencies.append(count / repeats)
    success = float(np.mean(consistencies))
    increment = bool(-1 not in mapping and _is_single_cycle(mapping) and success >= 0.90)
    identity = mapping == list(range(n))
    constant = -1 not in mapping and len(set(mapping)) == 1
    nontrivial = bool(-1 not in mapping and not identity and not constant)
    return mapping, success, increment, nontrivial, outcomes[0]


def run(config: dict, out: Path, resume=False, force=False):
    cfg = config["multivalued"]
    raw_path = out / "multivalued_raw.csv"
    confusion_path = out / "multivalued_confusion.csv"
    existing = read_csv(raw_path) if resume and not force else []
    expected_rows = len(cfg["systems"]) * (
        len(config["main_seeds"]) + len(cfg["spreads"]) * len(config["sensitivity_seeds"])
    )
    if len(existing) == expected_rows and confusion_path.exists():
        merged = existing
        summary = summarize(
            merged,
            ["experiment", "system", "N", "spread"],
            [
                "stable_state_count",
                "nonconverged_starts",
                "basin_entropy",
                "top_basin_fraction",
                "alphabet_size",
                "addressability_score",
                "fields_tested",
                "transition_success_rate",
                "nontrivial_transition_found",
                "increment_found",
            ],
            n_resamples=config["bootstrap_resamples"],
        )
        summary_path = write_csv(out / "multivalued_summary.csv", summary)
        return [raw_path, summary_path, confusion_path]
    rows, confusion = [], []
    conditions = []
    for seed in config["main_seeds"]:
        conditions.append((cfg["spread"], seed, "main"))
    for spread in cfg["spreads"]:
        for seed in config["sensitivity_seeds"]:
            conditions.append((spread, seed, "sensitivity"))
    for system in cfg["systems"]:
        for spread, seed, cohort in conditions:
            result = multibody.build_coupling(
                cfg["size"],
                np.random.default_rng(200_000 + seed),
                system=system,
                spread=spread,
                salt_M=0.01,
                grid_n=32,
            )
            J = np.asarray(result["J"], dtype=float)
            alphabet, n_states, nonconv, entropy, top_basin = _alphabet(
                J, cfg["landscape_starts"], cfg["max_alphabet"], 210_000 + seed
            )
            field_scale = operations.field_scale(J, rng=np.random.default_rng(220_000 + seed))
            address = (
                addressability.addressable_capacity(
                    J,
                    alphabet,
                    beta=0.25 * field_scale,
                    repeats=cfg["repeats"],
                    overlap_thresh=0.95,
                    rng=np.random.default_rng(230_000 + seed),
                )
                if alphabet
                else {"fraction_addressable": 0.0}
            )
            best = {
                "success": 0.0,
                "increment": False,
                "kind": "",
                "fraction": np.nan,
                "mapping": [],
                "counts": [],
                "nontrivial": False,
            }
            fields_tested = 0
            if len(alphabet) >= 3:
                for kind in ("scalar", "random", "structured", "multistep"):
                    for fraction in cfg["field_fractions"]:
                        candidates = _candidate_fields(
                            kind,
                            J,
                            fraction,
                            cfg["field_trials"],
                            np.random.default_rng(240_000 + seed + int(fraction * 1000)),
                        )
                        for cidx, fields in enumerate(candidates):
                            mapping, success, increment, nontrivial, counts = _evaluate_field(
                                J,
                                alphabet,
                                fields,
                                cfg["repeats"],
                                250_000 + seed * 1000 + cidx,
                            )
                            fields_tested += 1
                            if increment or (nontrivial and success > best["success"]):
                                best = {
                                    "success": success,
                                    "increment": increment,
                                    "kind": kind,
                                    "fraction": fraction,
                                    "mapping": mapping,
                                    "counts": counts,
                                    "nontrivial": nontrivial,
                                }
            status = (
                "increment_found"
                if best["increment"]
                else ("completed_no_gate" if len(alphabet) >= 3 else "no_usable_alphabet")
            )
            rows.append(
                {
                    "experiment": cohort,
                    "system": system,
                    "seed": seed,
                    "N": cfg["size"],
                    "spread": spread,
                    "stable_state_count": n_states,
                    "nonconverged_starts": nonconv,
                    "basin_entropy": entropy,
                    "top_basin_fraction": top_basin,
                    "alphabet_size": len(alphabet),
                    "addressability_score": address.get("addressable_fraction", 0.0),
                    "fields_tested": fields_tested,
                    "best_field_kind": best["kind"],
                    "best_field_fraction": best["fraction"],
                    "transition_success_rate": best["success"],
                    "nontrivial_transition_found": float(best["nontrivial"]),
                    "increment_found": float(best["increment"]),
                    "status": status,
                }
            )
            if best["counts"]:
                for origin, counter in enumerate(best["counts"]):
                    for target, count in sorted(counter.items()):
                        confusion.append(
                            {
                                "system": system,
                                "seed": seed,
                                "spread": spread,
                                "field_kind": best["kind"],
                                "field_fraction": best["fraction"],
                                "origin_state": origin,
                                "target_state": target,
                                "count": count,
                                "repeats": cfg["repeats"],
                            }
                        )
    merged = append_unique(
        raw_path,
        rows,
        ["experiment", "system", "seed", "N", "spread"],
        resume=resume and not force,
    )
    summary = summarize(
        merged,
        ["experiment", "system", "N", "spread"],
        [
            "stable_state_count",
            "nonconverged_starts",
            "basin_entropy",
            "top_basin_fraction",
            "alphabet_size",
            "addressability_score",
            "fields_tested",
            "transition_success_rate",
            "nontrivial_transition_found",
            "increment_found",
        ],
        n_resamples=config["bootstrap_resamples"],
    )
    summary_path = write_csv(out / "multivalued_summary.csv", summary)
    confusion_path = write_csv(confusion_path, confusion)
    return [raw_path, summary_path, confusion_path]


def main(argv=None):
    p = parser(__doc__)
    args = p.parse_args(argv)
    run(resolved_config(args), args.out, resume=args.resume, force=args.force)


if __name__ == "__main__":
    main()
