"""Verify the release, regenerate saved results, or run separate smoke simulations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["MPLBACKEND"] = "Agg"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify():
    manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))["files"]
    errors = [
        name
        for name, digest in manifest.items()
        if not (ROOT / name).is_file() or sha(ROOT / name) != digest
    ]
    if errors:
        raise AssertionError(f"Missing or changed release files: {errors}")
    return {"passed": True, "files_checked": len(manifest)}


def legacy_path(name):
    """Map historical manifest paths to the cleaned repository layout."""
    mapping = json.loads((ROOT / "provenance/extraction.json").read_text(encoding="utf-8"))
    return ROOT / mapping["copied"][name.replace("\\", "/")]["path"]


def historical_input_report(inputs):
    """Check retained inputs and explicitly identify excluded historical helpers."""
    records = json.loads((ROOT / "provenance/historical-inputs.json").read_text(encoding="utf-8"))
    normalized = {name.replace("\\", "/"): digest for name, digest in inputs.items()}
    if set(normalized) != set(records):
        raise AssertionError("Historical input inventory differs from the recorded inventory.")
    checked, excluded = [], []
    for name, expected in normalized.items():
        record = records[name]
        if record["historical_sha256"] != expected:
            raise AssertionError(f"Historical hash changed: {name}")
        if record["path"] is None:
            excluded.append(name)
        elif sha(ROOT / record["path"]) != record["release_sha256"]:
            raise AssertionError(f"Retained input changed: {record['path']}")
        else:
            checked.append(record["path"])
    return {
        "passed": True,
        "retained_files_checked": len(checked),
        "excluded_historical_files": excluded,
        "scope": "Integrity of retained inputs and documented cleanup mapping. "
        "Excluded legacy helpers are not reconstructed or executed.",
    }


def verify_historical_inputs(inputs):
    return historical_input_report(inputs)["passed"]


def compare(left_path, right_path):
    left, right = read(left_path), read(right_path)
    if len(left) != len(right):
        raise AssertionError(f"Row-count difference: {left_path}: {len(left)} != {len(right)}")
    checks = 0
    for index, (a, b) in enumerate(zip(left, right), start=1):
        if a.keys() != b.keys():
            raise AssertionError(f"Column difference: {left_path}")
        for key, value in a.items():
            checks += 1
            if value == b[key]:
                continue
            try:
                x, y = float(value), float(b[key])
                if (math.isnan(x) and math.isnan(y)) or math.isclose(
                    x, y, rel_tol=1e-9, abs_tol=1e-11
                ):
                    continue
            except (ValueError, TypeError):
                pass
            raise AssertionError(f"{Path(left_path).name} row {index} {key}: {value} != {b[key]}")
    return checks


def write_report(path, report):
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def regenerate(out):
    from coilnet.experiments.q1_common import (
        holm_adjust,
        paired_comparison,
        summarize,
        write_csv,
    )
    from coilnet.paths import ORIGINAL, REVISION
    from coilnet.paper import make_q1_figures as figures
    from scripts import analyze_revision, check_aggregates

    out.mkdir(parents=True, exist_ok=False)
    original, revision = out / "original", out / "revision"
    shutil.copytree(ORIGINAL, original)
    shutil.copytree(REVISION, revision)
    config = json.loads((original / "config_used.json").read_text(encoding="utf-8"))
    cells, compared = 0, []
    for name in ("reservoir", "attractor_recall", "multivalued", "rbm", "routes"):
        expected = read(ORIGINAL / f"{name}_summary.csv")
        statistical = {"metric", "n", "mean", "std", "median", "ci95_low", "ci95_high"}
        groups = [key for key in expected[0] if key not in statistical]
        metrics = list(dict.fromkeys(row["metric"] for row in expected))
        rows = summarize(
            read(ORIGINAL / f"{name}_raw.csv"),
            groups,
            metrics,
            n_resamples=config["bootstrap_resamples"],
        )
        path = original / f"{name}_summary.csv"
        write_csv(path, rows)
        cells += compare(ORIGINAL / path.name, path)
        compared.append("original/" + path.name)

    comparisons = []
    main = [row for row in read(ORIGINAL / "reservoir_raw.csv") if row["experiment"] == "main"]
    for system in config["reservoir"]["systems"]:
        subset = [row for row in main if row["system"] == system]
        for a, b in (
            ("graded_langevin", "bistable_langevin"),
            ("physical_tanh", "random_tanh"),
            ("physical_tanh", "symmetric_tanh"),
            ("physical_tanh", "shuffled_tanh"),
        ):
            result = paired_comparison(subset, "model", a, b, "narma10_nrmse", ["system", "seed"])
            result["system"] = system
            comparisons.append(result)
    write_csv(original / "reservoir_comparisons.csv", holm_adjust(comparisons))
    cells += compare(ORIGINAL / "reservoir_comparisons.csv", original / "reservoir_comparisons.csv")
    compared.append("original/reservoir_comparisons.csv")

    analyze_revision.main(revision, original)
    aggregate_report = check_aggregates.main(revision, original)
    derived = {
        "primary_contrasts.csv",
        "secondary_contrasts.csv",
        "paired_differences_raw.csv",
        "cutoff_raw.csv",
        "original_reservoir_seed_metrics.csv",
    }
    for path in sorted(REVISION.glob("*.csv")):
        if path.name.endswith("_summary.csv") or path.name in derived:
            cells += compare(path, revision / path.name)
            compared.append("revision/" + path.name)

    main_figures = out / "main-figures"
    source_tables = original / "figure_sources"
    figures._figure1(main_figures, "en")
    for number in range(2, 8):
        getattr(figures, f"_figure{number}")(original, main_figures, "en", source_tables)
    for path in sorted((ORIGINAL / "figure_sources").glob("*.csv")):
        cells += compare(path, source_tables / path.name)
        compared.append("figure_sources/" + path.name)

    report = {
        "passed": True,
        "csv_files_compared": len(compared),
        "csv_cells_compared": cells,
        "comparisons": compared,
        "numeric_tolerance": {"rtol": 1e-9, "atol": 1e-11},
        "independent_aggregate_checks": aggregate_report["scalar_comparisons"],
        "main_figures": 7,
        "supporting_figures": 6,
        "scope": "Regenerated from archived raw results, not full new simulations. "
        "Fonts and image rendering can vary by platform.",
    }
    verify()
    write_report(out / "regeneration-verification.json", report)
    return report


def smoke(out):
    from coilnet.paths import CONFIGS

    out.mkdir(parents=True, exist_ok=False)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "coilnet.experiments.run_q1_upgrade",
            "--quick",
            "--out",
            str(out / "original"),
            "--jobs",
            "1",
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "coilnet.experiments.run_revision_phase_c",
            "--pilot",
            "--config",
            str(CONFIGS / "revision.json"),
            "--out",
            str(out / "revision"),
            "--jobs",
            "2",
        ],
        cwd=ROOT,
        check=True,
    )
    return validate_smoke(out)


def validate_smoke(out):
    from coilnet.experiments.q1_common import write_csv
    from coilnet.paths import REVISION
    from scripts.validate_tables import validate_summary

    revision = json.loads((out / "revision/pilot/run_manifest.json").read_text(encoding="utf-8"))
    if revision["status"] != "complete":
        raise AssertionError("Revision pilot did not complete.")
    if revision["config"]["main_seeds"] != [30] or revision["config"]["sensitivity_seeds"] != [30]:
        raise AssertionError("Pilot must use archived seed 30.")
    original = json.loads((out / "original/manifest.json").read_text(encoding="utf-8"))
    if original["status"] != "complete":
        raise AssertionError("Original quick run did not complete.")
    for name in ("reservoir", "attractor_recall", "multivalued", "rbm", "routes"):
        errors = validate_summary(
            out / "original" / f"{name}_raw.csv", out / "original" / f"{name}_summary.csv"
        )
        if errors:
            raise AssertionError(errors)
    expected = out / "expected-seed-30"
    expected.mkdir(exist_ok=True)
    cells = 0
    for name in (
        "reservoir",
        "memory_delays",
        "discovery",
        "discovery_bank",
        "attractor_sensitivity",
        "attractor_cues",
    ):
        rows = [row for row in read(REVISION / f"{name}_raw.csv") if row["seed"] == "30"]
        write_csv(expected / f"{name}_raw.csv", rows)
        cells += compare(expected / f"{name}_raw.csv", out / "revision/pilot" / f"{name}_raw.csv")
    verify()
    report = {
        "passed": True,
        "original": "Fresh quick simulations; five raw/summary pairs checked.",
        "revision": "Fresh seed-30 simulations in both model families.",
        "revision_rows": revision["raw_rows"],
        "revision_matches_archived_seed_30": True,
        "revision_csv_cells_compared": cells,
        "full_manuscript_runs_repeated": False,
    }
    write_report(out / "smoke-verification.json", report)
    return report


def main(argv=None):
    from coilnet.paths import output_directory

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "regenerate", "smoke"))
    parser.add_argument(
        "--out", type=Path, help="New directory under work/ or outside the repository."
    )
    args = parser.parse_args(argv)
    verified = verify()
    if args.command == "verify":
        result = verified
    else:
        out = output_directory(args.out or ROOT / "work" / args.command)
        result = (regenerate if args.command == "regenerate" else smoke)(out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
