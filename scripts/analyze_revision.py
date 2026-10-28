"""Reproduce Phase C audits, seed summaries, contrasts and figures from saved raw outputs."""

from __future__ import annotations
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import csv
from collections import defaultdict, Counter
import hashlib
import json
import math
import numpy as np
from scipy.stats import wilcoxon
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from coilnet.experiments.q1_common import bootstrap_ci, holm_adjust, write_csv, write_json
from scripts.validate_tables import validate_summary, validate_claims
from coilnet.paths import output_directory

OUT = ROOT / "data/revision"
OLD = ROOT / "data/original"
CFG = json.loads((ROOT / "configs/revision.json").read_text(encoding="utf-8"))
FIG = OUT / "figures"
METRICS = ("narma10_nrmse", "memory_capacity")
COLORS = {"PAA": "#2269a5", "NVPVPE": "#c36a16"}


def read(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def stats(values):
    x = np.array([num(v) for v in values])
    finite = x[np.isfinite(x)]
    if not len(finite):
        return {"n_total": len(x), "n": 0, "failed_or_missing": len(x)}
    lo, hi = bootstrap_ci(finite, n_resamples=CFG["bootstrap_resamples"], seed=20261020)
    return dict(
        n_total=len(x),
        n=len(finite),
        failed_or_missing=len(x) - len(finite),
        mean=float(finite.mean()),
        std=float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
        median=float(np.median(finite)),
        ci95_low=lo,
        ci95_high=hi,
        minimum=float(finite.min()),
        maximum=float(finite.max()),
    )


def summarize(rows, keys, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, group in sorted(groups.items()):
        for metric in metrics:
            result.append(
                {**dict(zip(keys, key)), "metric": metric, **stats([r.get(metric) for r in group])}
            )
    return result


def index_unique(rows, keys):
    index = {}
    for row in rows:
        key = tuple(row.get(k, "") for k in keys)
        if key in index:
            raise AssertionError(f"Duplicate key {key}")
        index[key] = row
    return index


def contrasts(rows):
    index = index_unique(rows, ["system", "seed", "condition"])
    primary, secondary, differences = [], [], []
    for system in CFG["systems"]:
        for metric in METRICS:
            definitions = [
                (f"graded_minus_bistable_D{d:g}", [(f"graded_D{d:g}", 1), (f"bistable_D{d:g}", -1)])
                for d in CFG["reservoir"]["noise"]
            ]
            definitions.append(
                (
                    "response_noise_interaction_D0.3_minus_D0.1",
                    [
                        ("graded_D0.3", 1),
                        ("bistable_D0.3", -1),
                        ("graded_D0.1", -1),
                        ("bistable_D0.1", 1),
                    ],
                )
            )
            for name, terms in definitions:
                vals = []
                for seed in CFG["main_seeds"]:
                    values = [
                        num(index[(system, str(seed), cond)][metric]) * sign for cond, sign in terms
                    ]
                    value = sum(values) if all(np.isfinite(values)) else np.nan
                    vals.append(value)
                    differences.append(
                        dict(
                            system=system,
                            seed=seed,
                            metric=metric,
                            comparison=name,
                            difference=value,
                        )
                    )
                result = stats(vals)
                finite = np.array(vals)[np.isfinite(vals)]
                # Pass computed differences directly; auto method handles ties/zeros.
                pvalue = (
                    1.0
                    if np.all(finite == 0)
                    else float(wilcoxon(finite, alternative="two-sided", method="auto").pvalue)
                )
                primary.append(
                    dict(system=system, metric=metric, comparison=name, p_value=pvalue, **result)
                )
            for condition in sorted(
                {r["condition"] for r in rows if r["group"] != "response_noise"}
            ):
                vals = [
                    num(index[(system, str(seed), condition)][metric])
                    - num(index[(system, str(seed), "graded_D0.1")][metric])
                    for seed in CFG["sensitivity_seeds"]
                ]
                secondary.append(
                    dict(
                        system=system,
                        metric=metric,
                        comparison=condition + "_minus_graded_D0.1",
                        **stats(vals),
                    )
                )
    return holm_adjust(primary), secondary, differences


def savefig(fig, name):
    fig.savefig(FIG / (name + ".png"), dpi=220, bbox_inches="tight")
    fig.savefig(FIG / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(FIG / (name + ".svg"), bbox_inches="tight")
    plt.close(fig)


def all_figures(new, original, discovery, sensitivity, rbm, primary, secondary):
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    models = [
        "bistable_langevin",
        "graded_langevin",
        "physical_tanh",
        "random_tanh",
        "symmetric_tanh",
        "shuffled_tanh",
    ]
    labels = ["Bistable", "Graded", "Native tanh", "Random tanh", "Sym. tanh", "Shuffled tanh"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for i, system in enumerate(CFG["systems"]):
        for j, metric in enumerate(METRICS):
            ax = axes[i, j]
            data = [
                [num(r[metric]) for r in original if r["system"] == system and r["model"] == m]
                for m in models
            ]
            ax.boxplot(data, tick_labels=labels, showfliers=False)
            for x, vals in enumerate(data, 1):
                jitter = np.linspace(-0.16, 0.16, len(vals))
                ax.scatter(x + jitter, sorted(vals), s=10, color=COLORS[system], alpha=0.65)
            ax.tick_params(axis="x", labelrotation=35)
            ax.set_title(f"{system}; original 30-seed cohort")
            ax.set_ylabel("Test NRMSE" if j == 0 else "Memory capacity")
    fig.tight_layout()
    savefig(fig, "figure-c1-original-distributions")

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for i, system in enumerate(CFG["systems"]):
        for j, metric in enumerate(METRICS):
            ax = axes[i, j]
            for response, marker in (("bistable", "o"), ("graded", "s")):
                means, lows, highs = [], [], []
                for noise in CFG["reservoir"]["noise"]:
                    rr = [
                        r
                        for r in new
                        if r["system"] == system and r["condition"] == f"{response}_D{noise:g}"
                    ]
                    s = stats([r[metric] for r in rr])
                    means.append(s["mean"])
                    lows.append(s["mean"] - s["ci95_low"])
                    highs.append(s["ci95_high"] - s["mean"])
                ax.errorbar(
                    [0, 0.1, 0.3],
                    means,
                    yerr=[lows, highs],
                    marker=marker,
                    capsize=3,
                    label=response,
                )
            ax.set(
                xlabel="Noise parameter D",
                ylabel="Test NRMSE" if j == 0 else "Memory capacity",
                title=f"{system}; matched 30-seed comparisons",
            )
            if j == 0:
                ax.set_yscale("log")
            ax.legend()
    fig.tight_layout()
    savefig(fig, "figure-c2-matched-response-noise")

    names = sorted({r["comparison"] for r in secondary})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for j, metric in enumerate(METRICS):
        ax = axes[j]
        for i, system in enumerate(CFG["systems"]):
            rows = {
                r["comparison"]: r
                for r in secondary
                if r["metric"] == metric and r["system"] == system
            }
            vals = [rows[n]["mean"] for n in names]
            err = [
                [rows[n]["mean"] - rows[n]["ci95_low"] for n in names],
                [rows[n]["ci95_high"] - rows[n]["mean"] for n in names],
            ]
            ax.errorbar(
                vals,
                np.arange(len(names)) + (i - 0.5) * 0.22,
                xerr=err,
                fmt="o",
                capsize=2,
                label=system,
                color=COLORS[system],
            )
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_yticks(
            np.arange(len(names)),
            [n.replace("_minus_graded_D0.1", "").replace("_", " ") for n in names],
        )
        ax.set_xlabel("Paired change in " + ("test NRMSE" if j == 0 else "memory capacity"))
        ax.set_title("Secondary ablations; 10 paired seeds per family")
        ax.legend()
    fig.tight_layout()
    savefig(fig, "figure-c3-reservoir-sensitivity")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for system in CFG["systems"]:
        for j, metric in enumerate(["observed_attractors", "validation_known_fraction"]):
            ax = axes[0, j]
            rr = [r for r in discovery if r["system"] == system and r["top_k"] == "5"]
            ss = [
                stats([r[metric] for r in rr if int(r["budget"]) == b])
                for b in CFG["attractor"]["budgets"]
            ]
            ax.errorbar(
                CFG["attractor"]["budgets"],
                [s["mean"] for s in ss],
                yerr=[
                    [s["mean"] - s["ci95_low"] for s in ss],
                    [s["ci95_high"] - s["mean"] for s in ss],
                ],
                marker="o",
                capsize=3,
                color=COLORS[system],
                label=system,
            )
            ax.set(
                xlabel="Discovery starts",
                ylabel=["Observed fixed points", "Independent starts reaching a known state"][j],
            )
            ax.legend()
    for i, system in enumerate(CFG["systems"]):
        ax = axes[1, i]
        for top_k, label in [(5, "Top 5"), (10, "Top 10"), (0, "All discovered")]:
            ss = [
                stats(
                    [
                        r["perfect"]
                        for r in discovery
                        if r["system"] == system
                        and int(r["top_k"]) == top_k
                        and int(r["budget"]) == b
                    ]
                )
                for b in CFG["attractor"]["budgets"]
            ]
            ax.errorbar(
                CFG["attractor"]["budgets"],
                [s["mean"] for s in ss],
                yerr=[
                    [s["mean"] - s["ci95_low"] for s in ss],
                    [s["ci95_high"] - s["mean"] for s in ss],
                ],
                marker="o",
                capsize=2,
                label=label,
            )
        ax.set(
            xlabel="Discovery starts",
            ylabel="Uniform-target recovery at 20% corruption",
            title=f"{system}; 10 seeds",
        )
        ax.legend()
    fig.tight_layout()
    savefig(fig, "figure-c4-discovery-and-recovery")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for system in CFG["systems"]:
        rr = [r for r in sensitivity if r["system"] == system and r["experiment"] == "perturbation"]
        for metric, ax, label in [
            ("perfect", axes[0], "Original-target recovery"),
            ("convergence_rate", axes[1], "Fixed-point convergence"),
        ]:
            ax.scatter(
                [num(r["symmetry_residual"]) for r in rr],
                [num(r[metric]) for r in rr],
                s=16,
                alpha=0.65,
                label=system,
                color=COLORS[system],
            )
            ax.set_xscale("symlog", linthresh=1e-8)
            ax.axvline(0.01, color="grey", lw=0.8, ls="--")
            ax.set(xlabel="Continuous symmetry residual", ylabel=label)
            ax.legend()
        vals = CFG["attractor"]["thresholds"]
        means = [
            stats(
                [
                    r["perfect"]
                    for r in sensitivity
                    if r["system"] == system
                    and r["experiment"] == "threshold"
                    and num(r["value"]) == t
                ]
            )
            for t in vals
        ]
        axes[2].errorbar(
            vals,
            [s["mean"] for s in means],
            yerr=[
                [s["mean"] - s["ci95_low"] for s in means],
                [s["ci95_high"] - s["mean"] for s in means],
            ],
            marker="o",
            capsize=2,
            label=system,
            color=COLORS[system],
        )
        axes[2].set(xlabel="Threshold / median absolute row sum", ylabel="Original-target recovery")
        axes[2].legend()
    fig.tight_layout()
    savefig(fig, "figure-c5-residual-and-threshold")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, size in enumerate(["3", "4"]):
        rr = [r for r in rbm if r["size"] == size]
        scatter = axes[i].scatter(
            [num(r["valid_fraction"]) for r in rr],
            [num(r["mode_coverage"]) for r in rr],
            c=[num(r["negative_log_likelihood"]) for r in rr],
            s=13,
            cmap="viridis",
            alpha=0.75,
        )
        axes[i].set(
            xlabel="Valid-sample fraction",
            ylabel="Valid modes observed / all valid modes",
            title=f"Bars-and-Stripes {size} x {size}; all saved seeds",
            xlim=(-0.03, 1.03),
            ylim=(-0.03, 1.03),
        )
        fig.colorbar(scatter, ax=axes[i], label="Exact target NLL")
    fig.tight_layout()
    savefig(fig, "figure-c6-rbm-joint-quality")


def main(output_dir, original_dir=None):
    global OUT, OLD, FIG
    OUT = output_directory(output_dir)
    OLD = Path(original_dir) if original_dir is not None else ROOT / "data/original"
    FIG = OUT / "figures"
    run = json.loads((OUT / "run_manifest.json").read_text(encoding="utf-8"))
    checks = {"run_complete": run["status"] == "complete"}
    from reproduce import verify_historical_inputs, historical_input_report, legacy_path

    checks["historical_inputs_or_documented_adapters"] = verify_historical_inputs(
        run["input_hashes"]
    )
    checks["raw_and_checkpoint_hashes_match"] = all(
        sha(OUT / p) == h for p, h in run["output_hashes"].items()
    )
    old_manifest = json.loads((OLD / "manifest.json").read_text(encoding="utf-8"))
    raw_manifest = [
        e for e in old_manifest["files"] if e["path"].replace("\\", "/").endswith("_raw.csv")
    ]
    old_hash_audit = []
    for entry in raw_manifest:
        path = legacy_path(entry["path"])
        blob = path.read_bytes()
        crlf = blob.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        old_hash_audit.append(
            {
                "path": str(path.relative_to(ROOT)),
                "current_sha256": sha(path),
                "historical_sha256": entry["sha256"],
                "byte_identical": sha(path) == entry["sha256"],
                "matches_after_LF_to_CRLF_only": hashlib.sha256(crlf).hexdigest()
                == entry["sha256"],
            }
        )
    checks["original_raw_historical_hashes_match_after_line_endings"] = all(
        r["byte_identical"] or r["matches_after_LF_to_CRLF_only"] for r in old_hash_audit
    )
    original_errors = []
    for name in ["reservoir", "attractor_recall", "multivalued", "rbm", "routes"]:
        original_errors += validate_summary(
            OLD / (name + "_raw.csv"), OLD / (name + "_summary.csv")
        )
    original_errors += validate_claims(OLD, ROOT / "configs/result_claims.json")
    checks["original_five_raw_summary_pairs_and_saved_claims"] = not original_errors
    new, delays, discovery, sensitivity, banks, cues = [
        read(OUT / (name + "_raw.csv"))
        for name in [
            "reservoir",
            "memory_delays",
            "discovery",
            "attractor_sensitivity",
            "discovery_bank",
            "attractor_cues",
        ]
    ]
    original = [r for r in read(OLD / "reservoir_raw.csv") if r["experiment"] == "main"]
    rbm, routes = read(OLD / "rbm_raw.csv"), read(OLD / "routes_raw.csv")
    index_unique(new, ["system", "seed", "condition"])
    index_unique(discovery, ["system", "seed", "budget", "top_k"])
    index_unique(sensitivity, ["system", "seed", "experiment", "value"])
    index_unique(banks, ["system", "seed", "bank", "start"])
    index_unique(cues, ["system", "seed", "experiment", "value", "target", "repeat"])
    checks["new_counts"] = (
        len(new),
        len(delays),
        len(discovery),
        len(sensitivity),
        len(banks),
    ) == (520, 20800, 240, 240, 25600)
    checks["all_new_scores_finite"] = all(np.isfinite(num(r[m])) for r in new for m in METRICS)
    checks["failed_runs_reported"] = all(
        r["task_status"] == r["memory_status"] == "ok" for r in new
    )
    checks["chronological_sample_counts"] = all(r["n_train"] == r["n_test"] == "650" for r in new)
    grouped = defaultdict(list)
    for r in new:
        grouped[r["system"], r["seed"]].append(r)
    checks["paired_task_hashes"] = all(
        len({r["input_sha256"] for r in group}) == 1 for group in grouped.values()
    )
    checks["paired_response_weights_and_matrix"] = all(
        len({r[key] for r in group if r["group"] == "response_noise"}) == 1
        for group in grouped.values()
        for key in ["input_weights_sha256", "matrix_sha256"]
    )
    delay_index = defaultdict(list)
    for r in delays:
        delay_index[r["system"], r["seed"], r["condition"]].append(num(r["r2"]))
    checks["memory_sums_match_40_delays"] = all(
        len(delay_index[r["system"], r["seed"], r["condition"]]) == 40
        and np.isclose(
            sum(delay_index[r["system"], r["seed"], r["condition"]]), num(r["memory_capacity"])
        )
        for r in new
    )
    checks["discovery_statuses_exhaustive"] = all(
        r["status"] in {"fixed", "cycle", "timeout"} for r in banks + cues
    )
    summaries = {
        "reservoir_summary": summarize(new, ["system", "condition", "group"], METRICS),
        "original_reservoir_main_summary": summarize(
            original,
            ["system", "model"],
            [
                "narma10_nrmse",
                "narma20_nrmse",
                "mackey_glass_nrmse",
                "memory_capacity",
                "esp_max_distance",
                "has_esp",
            ],
        ),
        "discovery_summary": summarize(
            discovery,
            ["system", "budget", "top_k"],
            [
                "observed_attractors",
                "new_attractors",
                "retained_discovery_fraction",
                "validation_known_fraction",
                "validation_selected_fraction",
                "perfect",
                "weighted_recovery",
                "convergence_rate",
                "cycle_rate",
                "timeout_rate",
            ],
        ),
        "attractor_sensitivity_summary": summarize(
            sensitivity,
            ["system", "experiment", "value"],
            [
                "perfect",
                "target_fixed",
                "convergence_rate",
                "cycle_rate",
                "timeout_rate",
                "symmetry_residual",
            ],
        ),
        "rbm_joint_summary": summarize(
            rbm,
            ["size", "constraint"],
            [
                "valid_fraction",
                "mode_coverage",
                "negative_log_likelihood",
                "kl_target_to_model",
                "duplicate_rate",
                "mixing_time",
                "approximate_intersection",
            ],
        ),
        "routes_joint_summary": summarize(
            routes,
            ["route", "setting", "error", "levels", "update_noise", "mem_gamma", "mem_dep_ratio"],
            [
                "valid_fraction",
                "mode_coverage",
                "negative_log_likelihood",
                "kl_target_to_model",
                "duplicate_rate",
                "mixing_time",
            ],
        ),
    }
    primary, secondary, diffs = contrasts(new)
    checks["sixteen_primary_contrasts_30_pairs"] = len(primary) == 16 and all(
        r["n"] == 30 for r in primary
    )
    summaries.update(
        primary_contrasts=primary, secondary_contrasts=secondary, paired_differences_raw=diffs
    )
    cutoff_rows = []
    for r in sensitivity:
        if r["experiment"] == "perturbation":
            for cutoff in CFG["attractor"]["cutoffs"]:
                cutoff_rows.append(
                    {
                        "system": r["system"],
                        "seed": r["seed"],
                        "perturbation": r["value"],
                        "cutoff": str(cutoff),
                        "exceeds": float(num(r["symmetry_residual"]) > cutoff),
                    }
                )
    summaries["cutoff_summary"] = summarize(
        cutoff_rows, ["system", "perturbation", "cutoff"], ["exceeds"]
    )
    for name, rows in summaries.items():
        write_csv(OUT / (name + ".csv"), rows)
    write_csv(OUT / "original_reservoir_seed_metrics.csv", original)
    write_csv(OUT / "cutoff_raw.csv", cutoff_rows)
    # Preserve all 1,440 RBM and 1,050 route seed records in their original sources.
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "original_errors": original_errors,
        "runtime_tests": "Not run by this analysis script; see validation/test log for a fresh run.",
        "new_failures": Counter((r["task_status"] + " | " + r["memory_status"]) for r in new),
        "primary_contrasts": primary,
        "historical_line_ending_audit": old_hash_audit,
        "retained_historical_inputs": historical_input_report(run["input_hashes"]),
        "limits": [
            "Pointwise bootstrap CIs; Holm family covers 16 primary Wilcoxon tests.",
            "Secondary 10-seed checks are conditional, not exhaustive or optimized.",
            "Original run had a dirty checkout; historical provenance is not independently reconstructed.",
            "Saved mixing_time is a finite-chain activity-autocorrelation diagnostic, not global mixing time.",
        ],
        "visual_review": "pending",
    }
    write_json(OUT / "analysis-verification.json", report)
    if not report["passed"]:
        raise AssertionError(json.dumps(checks, indent=2))
    all_figures(new, original, discovery, sensitivity, rbm, primary, secondary)
    outputs = [
        *OUT.glob("*summary.csv"),
        OUT / "primary_contrasts.csv",
        OUT / "secondary_contrasts.csv",
        *FIG.glob("*"),
    ]
    write_json(
        OUT / "analysis-output-manifest.json",
        {
            "analysis_script_sha256": sha(__file__),
            "outputs": {str(p.relative_to(OUT)): sha(p) for p in outputs},
            "statistics_reference": "https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.stats.wilcoxon.html",
        },
    )
    print(
        json.dumps(
            {"passed": True, "checks": len(checks), "figures": 6, "primary_contrasts": primary},
            indent=2,
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Working copy of revision data")
    parser.add_argument("--original", type=Path)
    args = parser.parse_args()
    main(args.out, args.original)
