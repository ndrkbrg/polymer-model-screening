"""Scientific invariants for the declared Phase C analyses."""

import json
from pathlib import Path

import numpy as np
import pytest

from coilnet.experiments._rc_common import train_eval_task
from coilnet.experiments.run_phase2_attractor_recall_q1 import _async_fixed
from coilnet.experiments.run_revision_phase_c import (
    aggregate_targets,
    cue_and_order,
    discovery_bank,
    fixed_trajectory,
    input_weights,
    make_reservoir,
    matrix_controls,
    reservoir_conditions,
    safe_score,
    target_scores,
)

CONFIG = json.loads(
    (Path(__file__).resolve().parents[1] / "configs/revision.json").read_text(encoding="utf-8")
)


def test_fast_trajectory_matches_original_fixed_cycle_timeout_and_ties():
    rng = np.random.default_rng(123)
    cases = [
        (np.array([[0.0, 1.0], [-1.0, 0.0]]), np.ones(2), np.arange(2), 0),
        (np.zeros((3, 3)), np.ones(3), np.arange(3), 10),
    ]
    for _ in range(40):
        W = rng.normal(size=(8, 8))
        np.fill_diagonal(W, 0)
        cases.append((W, rng.choice([-1.0, 1.0], 8), rng.permutation(8), 80))
    for W, start, order, limit in cases:
        bias = np.zeros(len(W))
        original = _async_fixed(W, bias, start, order, limit)
        fast = fixed_trajectory(W, bias, start, order, limit)
        assert np.array_equal(original[0], fast[0])
        assert original[1:3] == fast[1:3]


def test_discovery_banks_are_prefix_nested_and_cues_do_not_depend_on_rank():
    W = np.ones((8, 8)) - np.eye(8)
    short = discovery_bank(W, 8, 20, np.random.default_rng(52))
    long = discovery_bank(W, 32, 20, np.random.default_rng(52))
    assert short == long[:8]
    for target in ("01011010", "11001010"):
        cue, order = cue_and_order(target, 30, 2, 0.2)
        cue_again, order_again = cue_and_order(target, 30, 2, 0.2)
        assert np.array_equal(cue, cue_again)
        assert np.array_equal(order, order_again)
        assert np.count_nonzero(cue != np.array([1 if b == "1" else -1 for b in target])) == 2


def test_matched_noise_streams_and_input_masks_are_shared():
    cfg = CONFIG["reservoir"]
    matrices = matrix_controls(np.ones((40, 40)), 0.9, 30)
    conditions = reservoir_conditions(cfg, True)
    a = next(r for r in conditions if r["condition"] == "bistable_D0.1")
    b = next(r for r in conditions if r["condition"] == "graded_D0.1")
    ra, rb = make_reservoir(a, matrices, cfg, 30), make_reservoir(b, matrices, cfg, 30)
    # From zero state, only the a*x term differs and is zero. Same step proves same innovations.
    assert np.array_equal(ra.step(0.2), rb.step(0.2))
    full = input_weights(40, 30)
    half = input_weights(40, 30, fraction=0.5)
    mask = half[:, 0] != 0
    assert np.count_nonzero(mask) == 20
    assert np.array_equal(half[mask], full[mask])
    assert np.array_equal(input_weights(40, 30, scale=2), 2 * full)
    assert len(conditions) == 14
    for W in matrices.values():
        assert np.all(np.diag(W) == 0)
        assert np.isclose(np.max(np.abs(np.linalg.eigvals(W))), 0.9)


def test_task_test_targets_do_not_leak_into_readout_fit():
    cfg = CONFIG["reservoir"]
    matrices = matrix_controls(np.eye(40, k=1) + np.eye(40, k=-1), 0.9, 30)
    condition = reservoir_conditions(cfg, False)[0]
    u = np.random.default_rng(24).uniform(0, 0.5, 1500)
    y = np.sin(np.arange(1500) / 13)
    changed = y.copy()
    changed[850:] += 99
    a = train_eval_task(make_reservoir(condition, matrices, cfg, 30), u, y)
    b = train_eval_task(make_reservoir(condition, matrices, cfg, 30), u, changed)
    assert (a["n_train"], a["n_test"]) == (650, 650)
    assert np.array_equal(a["pred_test"], b["pred_test"])
    assert a["nrmse_train"] == b["nrmse_train"]


def test_failures_remain_missing_and_target_denominators_are_explicit():
    value, status = safe_score(lambda: {"score": float("nan")})
    assert value is None and status.startswith("FloatingPointError")
    empty = aggregate_targets([], {}, {})
    assert empty["n_targets"] == empty["n_cues"] == 0
    assert empty["perfect"] is None
    cfg = {**CONFIG["attractor"], "cue_repeats": 2, "max_sweeps": 0}
    records = target_scores(np.zeros((3, 3)), np.zeros(3), "111", 30, cfg)
    assert all(r["status"] == "timeout" and r["perfect"] == 0 for r in records)


def test_configuration_keeps_revision_cohort_disjoint_from_original():
    assert CONFIG["main_seeds"] == list(range(30, 60))
    assert CONFIG["sensitivity_seeds"] == list(range(30, 40))
    assert CONFIG["attractor"]["budgets"] == [128, 256, 512, 1024]
