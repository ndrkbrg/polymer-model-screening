import numpy as np

from coilnet.reservoir import readout


def test_add_bias_shape_and_values():
    X = np.arange(12, dtype=float).reshape(4, 3)
    Xb = readout.add_bias(X)
    assert Xb.shape == (4, 4)
    assert np.allclose(Xb[:, 0], 1.0)
    assert np.allclose(Xb[:, 1:], X)


def test_ridge_fit_exactly_recovers_noiseless_linear_map_1d_target():
    rng = np.random.default_rng(0)
    n_features = 5
    T = 200
    X = rng.normal(size=(T, n_features))
    true_w = rng.normal(size=n_features)
    y = X @ true_w
    Xb = readout.add_bias(X)
    Wout = readout.ridge_fit(Xb, y, reg=1e-10)
    pred = readout.ridge_predict(Xb, Wout)
    assert readout.nrmse(pred, y) < 1e-6


def test_ridge_fit_recovers_noiseless_linear_map_2d_target():
    rng = np.random.default_rng(1)
    n_features = 6
    n_outputs = 3
    T = 300
    X = rng.normal(size=(T, n_features))
    true_W = rng.normal(size=(n_features, n_outputs))
    Y = X @ true_W
    Wout = readout.ridge_fit(X, Y, reg=1e-10)
    pred = readout.ridge_predict(X, Wout)
    assert Wout.shape == (n_features, n_outputs)
    for j in range(n_outputs):
        assert readout.nrmse(pred[:, j], Y[:, j]) < 1e-6


def test_nrmse_perfect_prediction_is_zero():
    target = np.array([1.0, 2.0, 3.0, 4.0])
    assert readout.nrmse(target, target) == 0.0


def test_rmse_known_value():
    pred = np.array([0.0, 0.0])
    target = np.array([3.0, 4.0])
    assert np.isclose(readout.rmse(pred, target), np.sqrt((9 + 16) / 2))


def test_accuracy_binary():
    pred = np.array([0.1, 0.6, 0.49, 0.9])
    target = np.array([0.0, 1.0, 0.0, 1.0])
    assert readout.accuracy_binary(pred, target) == 1.0


def test_accuracy_multiclass_argmax():
    pred = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.4]])
    target = np.array([[1, 0], [0, 1], [0, 1]])
    # third row: predicted class 0, true class 1 -> wrong
    assert np.isclose(readout.accuracy(pred, target), 2.0 / 3.0)
