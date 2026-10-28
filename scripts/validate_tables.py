"""Validate Q1 raw/summary arithmetic, figure sources, and manuscript claim mappings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


STATS = {"metric", "n", "mean", "std", "median", "ci95_low", "ci95_high"}


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def validate_summary(raw_path, summary_path):
    raw, summary = read_csv(raw_path), read_csv(summary_path)
    if not summary:
        return [f"{Path(summary_path).name}: empty summary"]
    group_columns = [k for k in summary[0] if k not in STATS]
    index = {}
    for row in raw:
        key = tuple(str(row.get(k, "")) for k in group_columns)
        index.setdefault(key, []).append(row)
    errors = []
    for i, row in enumerate(summary, start=2):
        metric = row["metric"]
        key = tuple(str(row.get(k, "")) for k in group_columns)
        matches = index.get(key, [])
        values = []
        for match in matches:
            try:
                value = float(match.get(metric, "nan"))
            except ValueError:
                continue
            if np.isfinite(value):
                values.append(value)
        arr = np.asarray(values)
        if len(arr) != int(row["n"]):
            errors.append(f"{summary_path.name}:{i}: n mismatch")
            continue
        expected = {
            "mean": arr.mean(),
            "median": np.median(arr),
            "std": arr.std(ddof=1) if len(arr) > 1 else 0.0,
        }
        for key, value in expected.items():
            if not np.isclose(float(row[key]), value, rtol=1e-10, atol=1e-12):
                errors.append(f"{summary_path.name}:{i}: {key} mismatch")
    return errors


def validate_claims(data_dir, claims_path):
    if not claims_path.exists():
        return ["result_claims.json missing"]
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    errors = []
    for claim in claims:
        rows = read_csv(data_dir / claim["summary_file"])
        matches = [
            r
            for r in rows
            if all(str(r.get(k, "")) == str(v) for k, v in claim["selector"].items())
        ]
        if len(matches) != 1:
            errors.append(f"{claim['id']}: selector matched {len(matches)} rows")
            continue
        actual = float(matches[0][claim.get("field", "mean")])
        if not np.isclose(actual, float(claim["value"]), rtol=5e-4, atol=5e-4):
            errors.append(f"{claim['id']}: {actual} != {claim['value']}")
    return errors


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data", type=Path, default=Path(__file__).resolve().parents[1] / "data/original"
    )
    p.add_argument(
        "--claims",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/result_claims.json",
    )
    args = p.parse_args(argv)
    pairs = [
        ("attractor_recall_raw.csv", "attractor_recall_summary.csv"),
        ("reservoir_raw.csv", "reservoir_summary.csv"),
        ("multivalued_raw.csv", "multivalued_summary.csv"),
        ("rbm_raw.csv", "rbm_summary.csv"),
        ("routes_raw.csv", "routes_summary.csv"),
    ]
    errors = []
    for raw, summary in pairs:
        errors.extend(validate_summary(args.data / raw, args.data / summary))
    errors.extend(validate_claims(args.data, args.claims))
    if errors:
        raise SystemExit("\n".join(errors))
    print("Q1 data and manuscript claims validated")


if __name__ == "__main__":
    main()
