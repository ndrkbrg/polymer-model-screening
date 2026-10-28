import numpy as np

from coilnet.reservoir import CoilReservoir, rescale_spectral_radius, spectral_radius


def test_state_shape_and_step():
    rng = np.random.default_rng(0)
    N = 20
    J = rng.normal(scale=0.1, size=(N, N))
    res = CoilReservoir(J, rng=rng)
    assert res.state.shape == (N,)
    new_state = res.step(0.5)
    assert new_state.shape == (N,)
    assert np.array_equal(new_state, res.state)


def test_step_returns_a_copy_not_a_view():
    rng = np.random.default_rng(1)
    N = 10
    J = rng.normal(scale=0.1, size=(N, N))
    res = CoilReservoir(J, rng=rng, noise=0.0)
    s1 = res.step(0.1)
    s1[0] = 999.0
    assert res.state[0] != 999.0


def test_relaxation_time_defaults_to_legacy_langevin_update():
    J = np.array([[0.0, 0.2], [-0.1, 0.0]])
    Win = np.ones((2, 1))
    kwargs = dict(
        input_weights=Win,
        dynamics="langevin",
        gain=0.4,
        dt=0.05,
        a=-1.0,
        b=1.0,
        noise=0.0,
    )
    legacy = CoilReservoir(J, **kwargs)
    explicit = CoilReservoir(J, relaxation_time=1.0, **kwargs)
    initial = np.array([0.2, -0.3])
    legacy.reset(initial)
    explicit.reset(initial)
    assert np.array_equal(legacy.step(0.4), explicit.step(0.4))


def test_relaxation_time_changes_graded_langevin_response():
    J = np.zeros((1, 1))
    Win = np.ones((1, 1))
    fast = CoilReservoir(
        J,
        input_weights=Win,
        dynamics="langevin",
        a=-1.0,
        b=1.0,
        dt=0.05,
        relaxation_time=0.5,
        noise=0.0,
    )
    slow = CoilReservoir(
        J,
        input_weights=Win,
        dynamics="langevin",
        a=-1.0,
        b=1.0,
        dt=0.05,
        relaxation_time=2.0,
        noise=0.0,
    )
    assert fast.step(1.0)[0] > slow.step(1.0)[0]


def test_relaxation_time_must_be_positive_and_finite():
    J = np.zeros((2, 2))
    for invalid in (0.0, -1.0, np.inf, np.nan):
        try:
            CoilReservoir(J, relaxation_time=invalid)
        except ValueError as exc:
            assert "relaxation_time" in str(exc)
        else:
            raise AssertionError(f"accepted invalid relaxation_time={invalid!r}")


def test_run_shape_and_washout():
    rng = np.random.default_rng(2)
    N = 15
    T = 50
    washout = 10
    J = rng.normal(scale=0.1, size=(N, N))
    res = CoilReservoir(J, rng=rng)
    inputs = rng.uniform(-1, 1, size=T)
    X = res.run(inputs, washout=washout)
    assert X.shape == (T - washout, N)


def test_run_record_input_stacks_columns():
    rng = np.random.default_rng(3)
    N = 8
    T = 30
    J = rng.normal(scale=0.1, size=(N, N))
    res = CoilReservoir(J, rng=rng, n_inputs=1)
    inputs = rng.uniform(-1, 1, size=T)
    X = res.run(inputs, washout=5, record_input=True)
    assert X.shape == (T - 5, N + 1)


def test_run_2d_inputs():
    rng = np.random.default_rng(4)
    N = 8
    T = 25
    n_inputs = 3
    J = rng.normal(scale=0.1, size=(N, N))
    res = CoilReservoir(J, rng=rng, n_inputs=n_inputs)
    inputs = rng.uniform(-1, 1, size=(T, n_inputs))
    X = res.run(inputs, washout=0)
    assert X.shape == (T, N)


def test_driven_fraction_zeros_non_driven_rows():
    rng = np.random.default_rng(5)
    N = 40
    J = np.zeros((N, N))
    res = CoilReservoir(J, rng=rng, driven_fraction=0.25, input_scale=1.0)
    n_nonzero_rows = np.count_nonzero(np.any(res.Win != 0.0, axis=1))
    assert n_nonzero_rows == 10


def test_explicit_input_weights_are_used_as_given():
    N = 5
    J = np.zeros((N, N))
    Win = np.arange(N, dtype=float).reshape(N, 1)
    res = CoilReservoir(J, input_weights=Win)
    assert np.array_equal(res.Win, Win)


def test_spectral_radius_of_known_matrix():
    J = np.diag([0.5, -2.0, 1.0])
    assert np.isclose(spectral_radius(J), 2.0)


def test_rescale_spectral_radius_hits_target():
    rng = np.random.default_rng(6)
    J = rng.normal(size=(12, 12))
    rescaled = rescale_spectral_radius(J, 0.7)
    assert np.isclose(spectral_radius(rescaled), 0.7)


def test_rescale_spectral_radius_guards_zero_matrix():
    J = np.zeros((5, 5))
    rescaled = rescale_spectral_radius(J, 1.0)
    assert np.allclose(rescaled, 0.0)


def test_esn_with_subunit_spectral_radius_has_echo_state_property():
    """Classic ESN theory: spectral radius < 1 is (generically) sufficient for fading memory --
    trajectories started from different random states, driven by the same input, should
    converge to (nearly) the same state."""
    rng = np.random.default_rng(7)
    N = 30
    J = rng.normal(scale=1.0, size=(N, N))
    J = rescale_spectral_radius(J, 0.8)
    res = CoilReservoir(J, dynamics="esn", gain=1.0, noise=0.0, rng=np.random.default_rng(8))

    T = 300
    washout = 100
    u = rng.uniform(-1, 1, size=T)

    finals = []
    for _ in range(4):
        init_state = rng.normal(scale=2.0, size=N)
        res.reset(init_state)
        X = res.run(u, washout=washout, reset=False)
        finals.append(X[-1])

    finals = np.array(finals)
    max_dist = max(
        np.linalg.norm(finals[i] - finals[j])
        for i in range(len(finals))
        for j in range(i + 1, len(finals))
    )
    assert max_dist < 1e-3
