import numpy as np

import pytest

from coilnet import coupling
from coilnet.experiments.run_phase1b_multibody import diagonal_symmetrization
from coilnet.physics import multibody as mb


@pytest.mark.parametrize("system", ["PAA", "NVPVPE"])
def test_interaction_matrix_is_symmetric(system):
    rng = np.random.default_rng(0)
    r = mb.build_coupling(24, rng, system=system, spread=0.8, grid_n=40)
    assert coupling.asymmetry_index(r["M"]) < 1e-9


@pytest.mark.parametrize("system", ["PAA", "NVPVPE"])
def test_identical_coils_give_symmetric_J(system):
    rng = np.random.default_rng(1)
    r = mb.build_coupling(24, rng, system=system, spread=0.0, grid_n=40)
    assert coupling.asymmetry_index(r["J"]) < 1e-9


@pytest.mark.parametrize("system", ["PAA", "NVPVPE"])
def test_heterogeneous_coils_make_J_asymmetric(system):
    rng = np.random.default_rng(2)
    r = mb.build_coupling(30, rng, system=system, spread=1.5, grid_n=40)
    assert coupling.asymmetry_index(r["J"]) > 1e-2


def test_asymmetry_grows_with_spread():
    def asym(spread, seed):
        rng = np.random.default_rng(seed)
        r = mb.build_coupling(30, rng, system="NVPVPE", spread=spread, grid_n=40)
        return coupling.asymmetry_index(r["J"])

    low = np.mean([asym(0.2, s) for s in range(3)])
    high = np.mean([asym(1.5, s) for s in range(3)])
    assert high > low


@pytest.mark.parametrize("system", ["PAA", "NVPVPE"])
def test_factorized_coupling_is_diagonally_symmetrizable(system):
    result = mb.build_coupling(20, np.random.default_rng(11), system=system, spread=1.0, grid_n=32)
    K, asymmetry, sign_equivalent = diagonal_symmetrization(result)
    assert asymmetry < 1e-12
    assert sign_equivalent
    assert np.allclose(K, K.T)


@pytest.mark.parametrize("resp_w", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("salt_M", [0.005, 0.05])
def test_invariants_hold_across_variations(resp_w, salt_M):
    """Robustness: whatever the response width or salt, identical coils stay symmetric and
    heterogeneous coils become asymmetric (for the electrostatic PAA system)."""
    rng0 = np.random.default_rng(0)
    r0 = mb.build_coupling(
        24, rng0, system="PAA", spread=0.0, salt_M=salt_M, grid_n=40, resp_w=resp_w
    )
    assert coupling.asymmetry_index(r0["M"]) < 1e-9  # M reciprocal
    assert coupling.asymmetry_index(r0["J"]) < 1e-9  # symmetric null
    rng1 = np.random.default_rng(1)
    r1 = mb.build_coupling(
        30, rng1, system="PAA", spread=2.0, salt_M=salt_M, grid_n=40, resp_w=resp_w
    )
    # clearly nonzero vs the ~1e-9 null (magnitude is smallest in the short-screening/broad-response corner)
    assert coupling.asymmetry_index(r1["J"]) > 2e-3  # heterogeneity -> asymmetry


def test_contact_matrix_symmetric_and_local():
    centers = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (200.0, 0.0, 0.0)]
    radii = [10.0, 10.0, 10.0]
    M = mb.contact_matrix(centers, radii, extra_range_nm=1.0)
    assert np.allclose(M, M.T)
    assert M[0, 1] > 0  # coils 0,1 overlap
    assert M[0, 2] == 0  # coil 2 is far away
