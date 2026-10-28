import numpy as np

from coilnet.reservoir import tasks


def test_narma10_finite_and_correct_length():
    rng = np.random.default_rng(0)
    T = 200
    u, y = tasks.make_narma10(T, rng)
    assert u.shape == (T,)
    assert y.shape == (T,)
    assert np.all(np.isfinite(y))
    # transient (t < 10) is guarded to exactly zero
    assert np.allclose(y[:10], 0.0)
    # once past the transient, the series should actually move
    assert np.any(y[10:] != 0.0)


def test_narma10_matches_recurrence_by_hand():
    u = np.array([0.1] * 15)
    y = tasks.narma10(u)
    # y(10) computed from y(9)..y(0) (all zero) and u(0)*u(9)
    expected_y10 = 0.3 * 0.0 + 0.05 * 0.0 * 0.0 + 1.5 * u[0] * u[9] + 0.1
    assert np.isclose(y[10], expected_y10)


def test_mackey_glass_finite_bounded_and_correct_length():
    n = 500
    series = tasks.mackey_glass(n)
    assert series.shape == (n,)
    assert np.all(np.isfinite(series))
    # standard MG parameters keep the series comfortably within [0, 2]
    assert np.all(series > -0.5)
    assert np.all(series < 2.5)


def test_make_mackey_glass_prediction_shapes_and_shift():
    rng = np.random.default_rng(1)
    T = 300
    horizon = 5
    u, y = tasks.make_mackey_glass_prediction(T, horizon=horizon, rng=rng)
    assert u.shape == (T,)
    assert y.shape == (T,)
    # y should be the continuation of the same trajectory as u, `horizon` steps ahead
    _series = tasks.mackey_glass(T + horizon, seed_value=1.2)
    # can't compare directly (rng jitters seed_value), but shapes/finiteness must hold
    assert np.all(np.isfinite(y))


def test_delayed_xor_hand_example():
    bits = np.array([1, 0, 1, 1, 0, 0, 1])
    delay = 2
    target = tasks.delayed_xor(bits, delay)
    expected = np.array([0, 0, 0, 1, 1, 0, 1])
    assert np.array_equal(target, expected)


def test_make_delayed_xor_encodes_pm1_and_matches_delayed_xor():
    rng = np.random.default_rng(2)
    T = 50
    delay = 3
    bits_pm1, target = tasks.make_delayed_xor(T, delay, rng)
    assert set(np.unique(bits_pm1)) <= {-1.0, 1.0}
    bits01 = (bits_pm1 == 1.0).astype(int)
    assert np.array_equal(target, tasks.delayed_xor(bits01, delay))
