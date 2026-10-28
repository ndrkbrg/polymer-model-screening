"""The reservoir engine: drive the physically-derived coupling J as a dynamical system.

Phase 1/1b showed the coil-coil coupling J is markedly ASYMMETRIC (receiver != source
influence). That rules out a Hopfield energy (``hopfield.py``, ``coupling.py``), but
asymmetry is not a bug for every neural-network model of a physical substrate — it is
exactly the ingredient RESERVOIR COMPUTING wants.

Reservoir computing (Jaeger's "echo state network", Maass's "liquid state machine") does not
train the recurrent wiring at all. Instead it:

  1. Fixes some large, generic, possibly-asymmetric recurrent matrix (here: the physical J)
     and lets an input signal drive it.
  2. Reads out a LINEAR combination of the resulting high-dimensional trajectory to
     approximate a target function of the input's history.

Only the readout is trained (``readout.py``); the reservoir itself just needs to satisfy the
"echo state property" (fading memory: the state eventually forgets its own initial condition
and depends only on the input history) and to nonlinearly mix the input across many timescales.
Both requirements are agnostic to J's symmetry, which is why the flagship asymmetry result from
Phase 1 does not kill the neural-network story -- it just changes which flavour of
neural-computation is testable.

This module provides the dynamical system itself (``CoilReservoir``) in two flavours:

  * ``"langevin"`` -- the physically-motivated one: each coil sits in a bistable
    (double-well) potential ``a x - b x**3`` (same cubic nonlinearity used elsewhere in this
    project to model a bistable coil), coupled through ``gain * J`` with thermal noise. This is
    the candidate mechanism for the real material.
  * ``"esn"`` -- a textbook leaky-integrator tanh Echo State Network. This is the VALIDATION
    reference: reservoir-computing theory (memory capacity bounds, echo-state-property
    conditions on the spectral radius, etc.) is derived for this exact update rule, so we use
    it in the tests to check our metrics (``metrics.py``) reproduce known results before
    trusting them on the physical ``"langevin"`` dynamics.
"""

from __future__ import annotations

import numpy as np

__all__ = ["CoilReservoir", "spectral_radius", "rescale_spectral_radius"]


def spectral_radius(J):
    """The largest |eigenvalue| of J -- the standard reservoir-computing "richness" knob.

    For the tanh ESN, spectral radius < 1 (roughly) is the classic sufficient condition for
    the echo state property; near 1 is the "edge of chaos" where memory and expressivity both
    peak. It is diagnostic, not prescriptive, for the physical ``J`` -- we MEASURE it, not
    assume it.
    """
    J = np.asarray(J, dtype=float)
    eigvals = np.linalg.eigvals(J)
    return float(np.max(np.abs(eigvals)))


def rescale_spectral_radius(J, target):
    """Return a copy of J scaled so its spectral radius equals ``target``.

    Guards the degenerate zero-radius case (an all-zero or nilpotent J) by returning J
    unchanged -- there is nothing to rescale a zero matrix to.
    """
    J = np.asarray(J, dtype=float)
    radius = spectral_radius(J)
    if radius < 1e-12:
        return J.copy()
    return J * (target / radius)


class CoilReservoir:
    """A reservoir of N coupled coils driven by an external input, with a fixed J.

    Parameters
    ----------
    J : (N, N) array
        The coupling matrix, used EXACTLY as given -- not symmetrized and not zero-diagonalled.
        The physics that produced it (``coilnet.physics.multibody.build_coupling``) already
        decided what belongs on the diagonal and how asymmetric it is; that asymmetry is the
        whole point of testing it as a reservoir instead of a Hopfield memory.
    input_weights : (N, n_inputs) array, optional
        The input-coupling matrix Win. If None, a random matrix uniform in
        ``[-input_scale, input_scale]`` is drawn, then rows for coils that are NOT driven are
        zeroed out (a random ``driven_fraction`` of the N coils receive the input directly --
        the rest only feel it indirectly through J).
    dynamics : {"langevin", "esn"}
        Update rule -- see the module docstring.
    gain : float
        Overall scale on the recurrent term ``J @ x`` (the reservoir-computing "spectral
        scaling" knob, applied on top of whatever spectral radius J itself has).
    dt : float
        Euler step for the "langevin" dynamics.
    relaxation_time : float
        Positive physical relaxation timescale for the "langevin" dynamics.  The drift and
        diffusion accumulated during one sample are scaled by ``dt / relaxation_time``.
        The default ``1.0`` preserves the update rule used before this parameter was exposed.
    a, b : float
        Bistable double-well coefficients (``a*x - b*x**3``) for "langevin".
    noise : float
        Thermal/stochastic noise level (variance-ish scale; see ``step`` for the exact term).
    leak : float
        Leak rate for the "esn" leaky-integrator update (1.0 = no leak, a plain ESN).
    n_inputs : int
        Dimensionality of the external input u.
    input_scale : float
        Range of the randomly-drawn Win when ``input_weights`` is None.
    driven_fraction : float
        Fraction of coils that receive the input directly when Win is randomly drawn.
    rng : numpy.random.Generator, optional
        Source of randomness (default Win, noise terms). Defaults to ``np.random.default_rng()``.

    Attributes
    ----------
    state : (N,) array
        Current reservoir state, initialised to zeros.
    Win : (N, n_inputs) array
        The (possibly generated) input-coupling matrix.
    """

    def __init__(
        self,
        J,
        input_weights=None,
        dynamics="langevin",
        gain=1.0,
        dt=0.05,
        relaxation_time=1.0,
        a=1.0,
        b=1.0,
        noise=0.01,
        leak=1.0,
        n_inputs=1,
        input_scale=1.0,
        driven_fraction=1.0,
        rng=None,
    ):
        J = np.asarray(J, dtype=float)
        if J.ndim != 2 or J.shape[0] != J.shape[1]:
            raise ValueError("J must be a square 2-D matrix")
        if dynamics not in ("langevin", "esn"):
            raise ValueError('dynamics must be "langevin" or "esn"')
        if not np.isfinite(relaxation_time) or relaxation_time <= 0:
            raise ValueError("relaxation_time must be a positive finite number")
        self.J = J.copy()
        self.N = J.shape[0]
        self.dynamics = dynamics
        self.gain = gain
        self.dt = dt
        self.relaxation_time = float(relaxation_time)
        self.a = a
        self.b = b
        self.noise = noise
        self.leak = leak
        self.n_inputs = n_inputs
        self.input_scale = input_scale
        self.driven_fraction = driven_fraction
        self.rng = np.random.default_rng() if rng is None else rng

        if input_weights is None:
            Win = self.rng.uniform(-input_scale, input_scale, size=(self.N, n_inputs))
            n_driven = int(round(np.clip(driven_fraction, 0.0, 1.0) * self.N))
            driven_idx = self.rng.choice(self.N, size=n_driven, replace=False)
            mask = np.zeros(self.N, dtype=bool)
            mask[driven_idx] = True
            Win[~mask, :] = 0.0
            self.Win = Win
        else:
            Win = np.asarray(input_weights, dtype=float)
            if Win.ndim == 1:
                Win = Win[:, None]
            if Win.shape != (self.N, n_inputs):
                raise ValueError(f"input_weights must have shape ({self.N}, {n_inputs})")
            self.Win = Win.copy()

        self.state = np.zeros(self.N)

    def reset(self, state=None):
        """Reset the reservoir state (to zeros, or to a given (N,) array)."""
        if state is None:
            self.state = np.zeros(self.N)
        else:
            state = np.asarray(state, dtype=float)
            if state.shape != (self.N,):
                raise ValueError(f"state must have shape ({self.N},)")
            self.state = state.copy()
        return self.state

    def step(self, u):
        """Advance the reservoir by one timestep under input ``u``; return a copy of the new state.

        ``u`` may be a scalar (when ``n_inputs == 1``) or an (n_inputs,) array.
        """
        u = np.atleast_1d(np.asarray(u, dtype=float))
        if u.shape[0] != self.n_inputs:
            raise ValueError(f"u must have {self.n_inputs} component(s), got {u.shape[0]}")

        x = self.state
        recurrent = self.gain * (self.J @ x)
        drive = recurrent + self.Win @ u

        if self.dynamics == "langevin":
            # Bistable double-well coil dynamics + coupling + input, Euler-Maruyama integrated.
            deterministic = self.a * x - self.b * x**3 + drive
            scaled_dt = self.dt / self.relaxation_time
            kick = np.sqrt(2.0 * self.noise * scaled_dt) * self.rng.standard_normal(self.N)
            x_new = x + scaled_dt * deterministic + kick
        else:  # "esn"
            # Standard leaky-integrator tanh Echo State Network update.
            kick = np.sqrt(self.noise) * self.rng.standard_normal(self.N)
            x_new = (1.0 - self.leak) * x + self.leak * np.tanh(drive) + kick

        self.state = x_new
        return x_new.copy()

    def run(self, inputs, washout=0, reset=True, record_input=False):
        """Drive the reservoir through a sequence of inputs and collect the state trajectory.

        Parameters
        ----------
        inputs : (T,) or (T, n_inputs) array
            The input sequence.
        washout : int
            Number of initial states to discard (lets transients from the starting condition
            decay before the trajectory is used for readout training -- standard RC practice).
        reset : bool
            If True (default), reset the state to zero before driving.
        record_input : bool
            If True, horizontally stack the (post-washout) input columns onto the returned
            state matrix, so a readout can use "state + current input" as its feature vector.

        Returns
        -------
        X : (T - washout, N) array (or (T - washout, N + n_inputs) if ``record_input``)
        """
        inputs = np.asarray(inputs, dtype=float)
        inputs_2d = inputs[:, None] if inputs.ndim == 1 else inputs
        T = inputs_2d.shape[0]

        if reset:
            self.reset()

        X = np.zeros((T, self.N))
        for t in range(T):
            X[t] = self.step(inputs_2d[t])

        X = X[washout:]
        if record_input:
            X = np.hstack([X, inputs_2d[washout:]])
        return X
