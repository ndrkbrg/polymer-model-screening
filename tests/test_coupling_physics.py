import numpy as np

from coilnet import coupling
from coilnet.physics import coupling_physics as cp


def test_identical_coils_give_symmetric_W():
    """spread = 0 -> all coils identical -> derived W is symmetric (the honest null)."""
    rng = np.random.default_rng(0)
    coils = cp.sample_coils(30, rng, size_spread=0.0)
    res = cp.derive_coupling(coils, ionic_strength_M=0.01)
    assert coupling.asymmetry_index(res["W"]) < 1e-6


def test_heterogeneous_coils_give_asymmetric_W():
    rng = np.random.default_rng(1)
    coils = cp.sample_coils(40, rng, size_spread=0.6)
    res = cp.derive_coupling(coils, ionic_strength_M=0.01)
    assert coupling.asymmetry_index(res["W"]) > 1e-2


def test_asymmetry_grows_with_spread():
    def asym(spread, seed):
        rng = np.random.default_rng(seed)
        coils = cp.sample_coils(40, rng, size_spread=spread)
        return coupling.asymmetry_index(cp.derive_coupling(coils, ionic_strength_M=0.01)["W"])

    low = np.mean([asym(0.1, s) for s in range(3)])
    high = np.mean([asym(0.9, s) for s in range(3)])
    assert high > low


def test_contacts_exist():
    rng = np.random.default_rng(2)
    coils = cp.sample_coils(40, rng, size_spread=0.5)
    res = cp.derive_coupling(coils, ionic_strength_M=0.01)
    assert res["n_contacts"] > 0
