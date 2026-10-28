"""Independent CSV-level recomputation of nested-bank and per-cue results."""

from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / "data/revision"
OLD = ROOT / "data/original"


def read(name):
    with (C / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def equal(label, actual, expected):
    global checks
    checks += 1
    if not math.isclose(float(actual), float(expected), abs_tol=1e-11, rel_tol=1e-10):
        errors.append({"label": label, "actual": actual, "expected": expected})


def aggregate(r, selected, counts, experiment, value):
    groups = [cues[r["system"], r["seed"], experiment, value, t] for t in selected]
    flattened = [c for g in groups for c in g]
    equal("targets", r["n_targets"], len(groups))
    equal("cues", r["n_cues"], len(flattened))
    for metric in ["perfect", "overlap", "sweeps", "target_fixed"]:
        equal(metric, r[metric], statistics.mean(float(c[metric]) for c in flattened))
    for status, metric in [
        ("fixed", "convergence_rate"),
        ("cycle", "cycle_rate"),
        ("timeout", "timeout_rate"),
    ]:
        equal(metric, r[metric], statistics.mean(c["status"] == status for c in flattened))
    weighted = sum(
        counts[t] * statistics.mean(float(c["perfect"]) for c in g)
        for t, g in zip(selected, groups)
    ) / sum(counts[t] for t in selected)
    equal("weighted", r["weighted_recovery"], weighted)


def main(revision_dir, original_dir):
    from coilnet.paths import output_directory

    global C, OLD, banks, cues, checks, errors
    C = output_directory(revision_dir)
    OLD = Path(original_dir)
    banks = defaultdict(list)
    cues = defaultdict(list)
    for r in read("discovery_bank_raw.csv"):
        banks[r["system"], r["seed"], r["bank"]].append(r)
    for r in read("attractor_cues_raw.csv"):
        cues[r["system"], r["seed"], r["experiment"], float(r["value"]), r["target"]].append(r)
    checks = 0
    errors = []

    for r in read("discovery_raw.csv"):
        bank = sorted(banks[r["system"], r["seed"], "discovery"], key=lambda x: int(x["start"]))
        budget = int(r["budget"])
        counts = Counter(c["target"] for c in bank[:budget] if c["status"] == "fixed")
        ranked = sorted(counts, key=lambda t: (-counts[t], t))
        selected = ranked[: int(r["top_k"])] if int(r["top_k"]) else ranked
        valid = banks[r["system"], r["seed"], "validation"]
        equal("observed", r["observed_attractors"], len(counts))
        equal(
            "retained", r["retained_discovery_fraction"], sum(counts[t] for t in selected) / budget
        )
        equal(
            "validation",
            r["validation_known_fraction"],
            statistics.mean(c["status"] == "fixed" and c["target"] in counts for c in valid),
        )
        aggregate(r, selected, counts, "unperturbed", 0.0)
    for r in read("attractor_sensitivity_raw.csv"):
        counts = Counter(
            c["target"]
            for c in banks[r["system"], r["seed"], "discovery"]
            if c["status"] == "fixed"
        )
        selected = sorted(counts, key=lambda t: (-counts[t], t))[:5]
        aggregate(r, selected, counts, r["experiment"], float(r["value"]))
    with (OLD / "rbm_raw.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            targets = 14 if r["size"] == "3" else 30
            equal(
                "RBM KL identity",
                float(r["negative_log_likelihood"]) - float(r["kl_target_to_model"]),
                math.log(targets),
            )
    report = {
        "passed": not errors,
        "scalar_comparisons": checks,
        "errors": errors,
        "scope": "All nested-discovery and sensitivity summaries recomputed from raw bank/cue records; exact RBM NLL-KL identity.",
    }
    (C / "raw-aggregate-verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report))
    if errors:
        raise AssertionError(errors)

    return report
