"""Shared, deterministic I/O and statistics for the Engineered Science upgrade."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from coilnet.paths import ROOT, CONFIGS, WORK, output_directory


BOOTSTRAP_SEED = 20260706


def load_config(path: str | Path) -> dict:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_config_path"] = str(path.resolve())
    return data


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--out", type=Path, default=WORK / "original-run")
    p.add_argument("--config", type=Path, default=CONFIGS / "original_full.json")
    p.add_argument("--jobs", type=int, default=min(4, max(1, (os.cpu_count() or 2) - 1)))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--quick", action="store_true")
    return p


def resolved_config(args) -> dict:
    args.out = output_directory(args.out)
    path = CONFIGS / "original_quick.json" if args.quick else args.config
    config = load_config(path)
    config["_jobs"] = int(args.jobs)
    config["_profile"] = "quick" if args.quick else "full"
    return config


def write_json(path: Path, data: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
    )
    tmp.replace(path)
    return path


def _json_default(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)
    return path


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def append_unique(
    path: Path, new_rows: list[dict], key_columns: list[str], resume: bool
) -> list[dict]:
    old = read_csv(path) if resume else []
    keys = {tuple(row.get(k, "") for k in key_columns) for row in old}
    merged = list(old)
    for row in new_rows:
        key = tuple(str(row.get(k, "")) for k in key_columns)
        if key not in keys:
            merged.append(row)
            keys.add(key)
    write_csv(path, merged)
    return merged


def bootstrap_ci(values, n_resamples=10_000, seed=BOOTSTRAP_SEED):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    if len(x) == 1:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    chunk = 1000
    means = []
    for start in range(0, n_resamples, chunk):
        n = min(chunk, n_resamples - start)
        idx = rng.integers(0, len(x), size=(n, len(x)))
        means.append(x[idx].mean(axis=1))
    dist = np.concatenate(means)
    lo, hi = np.quantile(dist, [0.025, 0.975])
    return float(lo), float(hi)


def summarize(
    rows: list[dict], group_columns: list[str], metric_columns: list[str], n_resamples=10_000
) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(str(row.get(k, "")) for k in group_columns), []).append(row)
    output = []
    for group_key, group_rows in sorted(groups.items()):
        base = dict(zip(group_columns, group_key))
        for metric in metric_columns:
            vals = []
            for row in group_rows:
                try:
                    value = float(row.get(metric, "nan"))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    vals.append(value)
            if not vals:
                continue
            arr = np.asarray(vals)
            lo, hi = bootstrap_ci(
                arr, n_resamples=n_resamples, seed=BOOTSTRAP_SEED + sum(map(ord, metric))
            )
            output.append(
                {
                    **base,
                    "metric": metric,
                    "n": len(arr),
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    "median": float(np.median(arr)),
                    "ci95_low": lo,
                    "ci95_high": hi,
                }
            )
    return output


def paired_comparison(
    rows: list[dict],
    condition_column: str,
    condition_a: str,
    condition_b: str,
    metric: str,
    pair_columns: list[str],
) -> dict:
    index = {}
    for row in rows:
        if str(row.get(condition_column)) not in {condition_a, condition_b}:
            continue
        key = tuple(str(row.get(k, "")) for k in pair_columns)
        index.setdefault(key, {})[str(row[condition_column])] = float(row[metric])
    pairs = [
        (v[condition_a], v[condition_b])
        for v in index.values()
        if condition_a in v
        and condition_b in v
        and np.isfinite(v[condition_a])
        and np.isfinite(v[condition_b])
    ]
    if not pairs:
        return {"condition_a": condition_a, "condition_b": condition_b, "metric": metric, "n": 0}
    a, b = np.asarray(pairs).T
    diff = a - b
    try:
        stat, pvalue = wilcoxon(a, b, alternative="two-sided")
    except ValueError:
        stat, pvalue = 0.0, 1.0
    lo, hi = bootstrap_ci(diff)
    return {
        "condition_a": condition_a,
        "condition_b": condition_b,
        "metric": metric,
        "n": len(diff),
        "mean_paired_difference": float(diff.mean()),
        "ci95_low": lo,
        "ci95_high": hi,
        "wilcoxon_statistic": float(stat),
        "p_value": float(pvalue),
    }


def holm_adjust(rows: list[dict]) -> list[dict]:
    valid = [(i, float(r["p_value"])) for i, r in enumerate(rows) if "p_value" in r]
    ordered = sorted(valid, key=lambda x: x[1])
    running = 0.0
    for rank, (idx, pval) in enumerate(ordered):
        adjusted = min(1.0, (len(ordered) - rank) * pval)
        running = max(running, adjusted)
        rows[idx]["p_holm"] = running
    return rows


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def software_info(config: dict) -> dict:
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return "unknown"

    packages = {}
    for name in ("numpy", "scipy", "matplotlib", "python-docx", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "git_dirty": bool(git("status", "--porcelain")),
        "profile": config.get("_profile"),
        "config_path": config.get("_config_path"),
    }


def finalize_manifest(out: Path, config: dict, required: list[Path]) -> Path:
    missing = [str(p) for p in required if not p.exists()]
    manifest = {
        "status": "complete" if not missing else "incomplete",
        "software": software_info(config),
        "config_sha256": sha256(Path(config["_config_path"])),
        "files": [
            {"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size}
            for p in required
            if p.exists()
        ],
        "missing": missing,
    }
    path = write_json(out / "manifest.json", manifest)
    if missing:
        raise RuntimeError(f"Required Q1 artifacts missing: {missing}")
    return path
