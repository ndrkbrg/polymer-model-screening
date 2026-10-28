"""Standard reservoir-computing benchmark tasks.

These are the yardsticks the RC literature uses to compare reservoirs, independent of any
particular physical substrate. Running them against the physically-derived J
(``coilnet.physics.multibody.build_coupling``) via ``CoilReservoir`` and comparing against a
textbook tanh ESN (``metrics.py``) is how we ask "does the polymer wiring behave like a
competent, generic reservoir, or is it doing something degenerate?".

  * NARMA-10 -- a nonlinear auto-regressive moving-average task requiring deep memory (order
    10) AND nonlinear mixing of the input history. The classic RC "can you do both memory and
    nonlinearity" benchmark.
  * Mackey-Glass -- a chaotic time-delay differential equation, used here in one-step-ahead
    PREDICTION form. Tests whether the reservoir's trajectory carries enough information to
    extrapolate a chaotic signal it has never seen.
  * Delayed XOR -- a minimal, exactly-solvable nonlinear memory task (XOR of two input bits a
    fixed delay apart) used to sanity-check nonlinear temporal integration on hand-computable
    examples.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "narma10",
    "make_narma10",
    "mackey_glass",
    "make_mackey_glass_prediction",
    "delayed_xor",
    "make_delayed_xor",
]


def narma10(u):
    """The NARMA-10 target series for input series ``u`` (values expected in [0, 0.5]).

    Standard recurrence::

        y(t+1) = 0.3 y(t) + 0.05 y(t) * sum_{i=0}^{9} y(t-i) + 1.5 u(t-9) u(t) + 0.1

    This needs 10 past y-values and a 9-step-delayed u, so it is only well-defined once
    ``t >= 9``; the initial transient (indices 0..9) is left at zero (guarded) rather than
    extrapolated, matching standard NARMA-10 generator implementations. Callers should apply a
    washout of at least 10 samples before using ``y`` as a training target.
    """
    u = np.asarray(u, dtype=float)
    T = len(u)
    y = np.zeros(T)
    for t in range(9, T - 1):
        y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * np.sum(y[t - 9 : t + 1]) + 1.5 * u[t - 9] * u[t] + 0.1
    return y


def make_narma10(T, rng):
    """Draw a fresh NARMA-10 (input, target) pair: u ~ Uniform[0, 0.5], y = narma10(u)."""
    u = rng.uniform(0.0, 0.5, size=T)
    y = narma10(u)
    return u, y


def mackey_glass(n, tau=17, beta=0.2, gamma=0.1, nexp=10, dt=1.0, seed_value=1.2, discard=250):
    """Generate ``n`` points of the discrete Mackey-Glass chaotic time series.

    Euler-integrates the delay differential equation

        dx/dt = beta * x(t - tau) / (1 + x(t - tau)**nexp) - gamma * x(t)

    with a constant history ``seed_value`` for ``t <= 0``, then discards the first ``discard``
    post-history points so the returned series has settled onto the chaotic attractor (the
    initial constant history is not itself part of the attractor).

    With the default parameters (tau=17, beta=0.2, gamma=0.1, nexp=10) this is the classic
    ESN-literature Mackey-Glass benchmark series, bounded and deterministic-chaotic (no random
    driving) -- useful for testing a reservoir's ability to model chaotic dynamics rather than
    i.i.d.-input tasks like NARMA-10.
    """
    tau_steps = int(round(tau / dt))
    total = n + discard + tau_steps + 1
    x = np.zeros(total)
    x[: tau_steps + 1] = seed_value
    for t in range(tau_steps, total - 1):
        x_tau = x[t - tau_steps]
        x[t + 1] = x[t] + dt * (beta * x_tau / (1.0 + x_tau**nexp) - gamma * x[t])
    start = tau_steps + discard
    return x[start : start + n]


def make_mackey_glass_prediction(T, horizon=1, rng=None, **mg):
    """(u, y) pair for one-step(-ish)-ahead Mackey-Glass prediction: y(t) = u(t + horizon).

    Mackey-Glass is a deterministic chaotic system (no i.i.d. driving), so ``rng`` is only
    used to jitter the constant initial history slightly (when the caller does not pin
    ``seed_value`` explicitly), giving reproducible-but-varied series across calls with
    different rng state.
    """
    rng = np.random.default_rng() if rng is None else rng
    if "seed_value" not in mg:
        mg["seed_value"] = 1.2 + 0.01 * rng.standard_normal()
    series = mackey_glass(T + horizon, **mg)
    u = series[:T]
    y = series[horizon : horizon + T]
    return u, y


def delayed_xor(bits, delay):
    """target(t) = bits[t - delay] XOR bits[t - delay - 1], the minimal nonlinear memory task.

    ``bits`` is a 1-D array of {0, 1}. Needs both delayed taps to exist (``t >= delay + 1``),
    so entries before that are guarded to 0 (undefined transient) rather than wrapping/extrapolating.
    """
    bits = np.asarray(bits)
    T = len(bits)
    target = np.zeros(T, dtype=int)
    for t in range(delay + 1, T):
        target[t] = int(bits[t - delay]) ^ int(bits[t - delay - 1])
    return target


def make_delayed_xor(T, delay, rng):
    """Draw a fresh delayed-XOR (input, target) pair.

    Returns the bits encoded as +-1 (a more RC-friendly input scale than raw {0, 1}) alongside
    the {0, 1}-valued XOR target.
    """
    bits01 = rng.integers(0, 2, size=T)
    target = delayed_xor(bits01, delay)
    bits_pm1 = np.where(bits01 == 1, 1.0, -1.0)
    return bits_pm1, target
