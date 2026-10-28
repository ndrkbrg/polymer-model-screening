import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from coilnet.energybased.metrics import bars_and_stripes
from coilnet.experiments.q1_common import summarize
from coilnet.experiments.run_phase2_attractor_recall_q1 import _async_fixed
from coilnet.experiments.run_phase3_reservoir_q1 import make_narma20
from coilnet.experiments.run_phase5_rbm_q1 import _constraint_residuals, _projector
from coilnet.experiments.run_q1_upgrade import main as run_upgrade
from coilnet.paper.make_q1_figures import main as make_q1_figures


ROOT = Path(__file__).resolve().parents[1]


def test_full_config_has_confirmatory_seed_counts():
    cfg = json.loads((ROOT / "configs/original_full.json").read_text(encoding="utf-8"))
    assert cfg["main_seeds"] == list(range(30))
    assert cfg["sensitivity_seeds"] == list(range(10))


def test_bounded_narma20_is_finite_and_reproducible():
    a = make_narma20(2000, np.random.default_rng(3))
    b = make_narma20(2000, np.random.default_rng(3))
    assert np.all(np.isfinite(a[1]))
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_bars_and_stripes_3x3_has_exactly_14_patterns():
    patterns = bars_and_stripes(3)
    assert patterns.shape == (14, 9)
    assert len({tuple(row) for row in patterns}) == 14


def test_weighted_energy_is_nonincreasing_for_symmetric_case():
    W = np.array([[0.0, 1.0, -0.2], [1.0, 0.0, 0.3], [-0.2, 0.3, 0.0]])
    _, status, _, trace, _ = _async_fixed(
        W, np.zeros(3), np.array([1.0, -1.0, 1.0]), np.array([0, 1, 2]), 20
    )
    assert status == "fixed"
    assert np.all(np.diff(trace) <= 1e-12)


def test_cycle_and_fixed_status_are_distinct():
    W_cycle = np.array([[0.0, 1.0], [-1.0, 0.0]])
    _, status_cycle, *_ = _async_fixed(W_cycle, np.zeros(2), np.ones(2), np.array([0, 1]), 20)
    _, status_fixed, *_ = _async_fixed(
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.zeros(2),
        np.ones(2),
        np.array([0, 1]),
        20,
    )
    _, status_timeout, *_ = _async_fixed(W_cycle, np.zeros(2), np.ones(2), np.array([0, 1]), 0)
    assert status_cycle == "cycle"
    assert status_fixed == "fixed"
    assert status_timeout == "timeout"


def test_combined_constraint_residuals_are_reported():
    W = np.random.default_rng(2).normal(size=(9, 16))
    projected = _projector(True, True, 4)(W)
    residuals = _constraint_residuals(projected, True, True, 4)
    assert all(np.isfinite(residuals))
    assert residuals[0] <= 1e-12
    assert residuals[1] <= 1e-12


def test_summary_is_computed_from_raw_values():
    rows = [{"group": "a", "score": x} for x in (1.0, 2.0, 3.0)]
    result = summarize(rows, ["group"], ["score"], n_resamples=100)
    assert result[0]["n"] == 3
    assert result[0]["mean"] == 2.0
    assert np.isclose(result[0]["std"], 1.0)


@pytest.mark.integration
def test_quick_master_creates_expected_outputs_without_duplicates(tmp_path):
    args = ["--quick", "--out", str(tmp_path), "--force"]
    run_upgrade(args)
    expected = [
        "attractor_recall_raw.csv",
        "attractor_recall_summary.csv",
        "reservoir_raw.csv",
        "reservoir_summary.csv",
        "multivalued_raw.csv",
        "multivalued_summary.csv",
        "rbm_raw.csv",
        "rbm_summary.csv",
        "routes_raw.csv",
        "routes_summary.csv",
        "manifest.json",
    ]
    for name in expected:
        assert (tmp_path / name).exists(), name
    before = {}
    for name in expected:
        if name.endswith(".csv"):
            with (tmp_path / name).open(encoding="utf-8", newline="") as fh:
                before[name] = len(list(csv.DictReader(fh)))
    run_upgrade(["--quick", "--out", str(tmp_path), "--resume"])
    for name, count in before.items():
        with (tmp_path / name).open(encoding="utf-8", newline="") as fh:
            assert len(list(csv.DictReader(fh))) == count
    figure_root = tmp_path / "figures"
    make_q1_figures(["--data", str(tmp_path), "--out", str(figure_root)])
    for language_root in (figure_root,):
        for number in range(1, 8):
            png = next(language_root.glob(f"figure{number}_*.png"))
            assert next(language_root.glob(f"figure{number}_*.pdf")).exists()
            assert next(language_root.glob(f"figure{number}_*.svg")).exists()
            dpi = Image.open(png).info.get("dpi", (0, 0))
            assert min(dpi) >= 599
    manifest = json.loads((figure_root / "figure_manifest.json").read_text(encoding="utf-8"))
    assert all("source_tables" in row for row in manifest)
