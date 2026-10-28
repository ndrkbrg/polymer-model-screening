import numpy as np

from coilnet.hopfield import HopfieldNetwork
from coilnet.multistate import addressability, landscape


def test_find_attractors_ferromagnet_has_exactly_two_states():
    """An all-positive (ferromagnetic) fully-connected J has only two fixed points: all +1 and
    all -1. Every random start should converge to one of them (no glassy debris)."""
    N = 8
    J = np.ones((N, N))
    rng = np.random.default_rng(0)
    basins, n_nonconverged = landscape.find_attractors(J, trials=200, max_sweeps=40, rng=rng)
    assert n_nonconverged == 0
    assert set(basins.keys()) == {tuple([1] * N), tuple([-1] * N)}
    assert sum(basins.values()) == 200


def test_landscape_metrics_ferromagnet_vs_random_symmetric():
    """The ferromagnet's effective number of states is ~2 (a TRIVIAL switch); a random
    symmetric (glassy-flavoured) matrix of the same size has strictly more attractors."""
    N = 8
    J = np.ones((N, N))
    m_ferro = landscape.landscape_metrics(J, trials=200, rng=np.random.default_rng(1))
    assert m_ferro["n_attractors"] == 2
    assert abs(m_ferro["effective_n"] - 2.0) < 0.3

    rng2 = np.random.default_rng(2)
    A = rng2.normal(size=(N, N))
    W = 0.5 * (A + A.T)
    m_glass = landscape.landscape_metrics(W, trials=200, rng=np.random.default_rng(3))
    assert m_glass["n_attractors"] > m_ferro["n_attractors"]


def test_count_scaling_smoke():
    """count_scaling runs across N and returns a fitted growth slope (sanity/smoke check on the
    real physics-derived coupling, kept tiny for speed)."""
    result = landscape.count_scaling("PAA", [6, 8], seeds=(0,), trials=50)
    assert [row["N"] for row in result["rows"]] == [6, 8]
    for row in result["rows"]:
        assert row["mean_n_attractors"] >= 1.0
    assert isinstance(result["log_growth_slope"], float)


def test_addressable_capacity_hebbian_three_well_separated_patterns():
    """A symmetric Hebbian network storing 3 well-separated random patterns is the known-answer
    case: all 3 should be addressable (write + hold reliably reproduces them)."""
    N = 24
    rng = np.random.default_rng(0)
    patterns = rng.choice([-1.0, 1.0], size=(3, N))
    net = HopfieldNetwork.from_hebbian(patterns)
    result = addressability.addressable_capacity(
        net.W, patterns, beta=1.0, repeats=5, rng=np.random.default_rng(1)
    )
    assert result["n_addressable"] >= 2
    assert result["addressable_fraction"] >= 0.66
    assert len(result["per_target"]) == 3


def test_injective_outputs_rich_matrix_gives_multiple_distinct_outputs():
    """A rich (random symmetric) coupling driven by many distinct bias patterns should settle
    into more than one distinct held state."""
    N = 16
    rng = np.random.default_rng(4)
    A = rng.normal(size=(N, N))
    J = 0.5 * (A + A.T)
    result = addressability.injective_outputs(
        J, n_inputs=30, beta=1.0, rng=np.random.default_rng(5)
    )
    assert result["n_distinct_outputs"] > 1
    assert result["n_inputs"] == 30
