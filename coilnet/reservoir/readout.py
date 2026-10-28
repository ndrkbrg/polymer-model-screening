"""The readout layer: the ONLY part of a reservoir computer that gets trained.

Everything upstream (``engine.py``) is a fixed, untrained dynamical system driven by the
input. The reservoir-computing bet is that if that system is rich and "echoey" enough
(``metrics.py``), a plain LINEAR readout on top of its state trajectory is enough to
approximate a wide variety of nonlinear, history-dependent target functions. Linear readouts
are used deliberately: they make training a convex, closed-form least-squares problem (no
backprop through the recurrence at all), and they keep any success or failure attributable to
the reservoir's dynamics rather than to a powerful readout compensating for a poor one.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "add_bias",
    "ridge_fit",
    "ridge_predict",
    "rmse",
    "nrmse",
    "accuracy",
    "accuracy_binary",
]


def add_bias(X):
    """Prepend a column of ones to X, so the linear readout can learn an intercept."""
    X = np.asarray(X, dtype=float)
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])


def ridge_fit(X, Y, reg=1e-6):
    """Closed-form ridge regression: Wout = (XᵀX + reg·I)⁻¹ XᵀY.

    Ridge (rather than plain least squares) is standard in reservoir computing because the
    reservoir state columns are highly correlated (they are all driven by the same input
    through overlapping dynamics), which makes XᵀX close to singular; a little L2
    regularisation keeps the fit numerically stable.

    ``Y`` may be 1-D (a single target series) or 2-D ``(T, n_outputs)``; ``Wout`` comes back
    with the matching shape convention (``(n_features,)`` or ``(n_features, n_outputs)``), so
    ``X @ Wout`` (see ``ridge_predict``) reproduces the same shape as Y.

    Uses ``np.linalg.solve`` rather than an explicit matrix inverse (faster, more stable).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n_features = X.shape[1]
    A = X.T @ X + reg * np.eye(n_features)
    B = X.T @ Y
    return np.linalg.solve(A, B)


def ridge_predict(X, Wout):
    """Apply a trained readout: prediction = X @ Wout."""
    X = np.asarray(X, dtype=float)
    return X @ Wout


def rmse(pred, target):
    """Root-mean-squared error."""
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def nrmse(pred, target):
    """RMSE normalised by the target's standard deviation.

    This is the standard reservoir-computing error metric (used e.g. for NARMA-10 and
    Mackey-Glass benchmarks) because it is scale-free: nrmse ~ 1 means "no better than
    predicting the mean", nrmse ~ 0 means a perfect fit, independent of the target's units.
    """
    target = np.asarray(target, dtype=float)
    denom = np.std(target)
    if denom < 1e-12:
        denom = 1e-12
    return rmse(pred, target) / denom


def accuracy_binary(pred, target):
    """Classification accuracy for a single binary output, thresholded at 0.5.

    ``pred`` is thresholded to {0, 1} (>= 0.5 -> 1); ``target`` is assumed already in {0, 1}
    (or {-1, 1}-like values are NOT handled here -- threshold at 0.5 assumes a 0/1 target).
    """
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    pred_labels = (pred >= 0.5).astype(float)
    return float(np.mean(pred_labels == target))


def accuracy(pred, target):
    """Classification accuracy, thresholding a single output or arg-maxing multiple outputs.

    If ``pred`` has more than one output column, the predicted class is ``argmax`` per row
    (and ``target`` is argmax'd too if it is one-hot); otherwise falls back to
    ``accuracy_binary``.
    """
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    if pred.ndim > 1 and pred.shape[1] > 1:
        pred_labels = np.argmax(pred, axis=1)
        target_labels = np.argmax(target, axis=1) if target.ndim > 1 else target
        return float(np.mean(pred_labels == target_labels))
    return accuracy_binary(pred, target)
