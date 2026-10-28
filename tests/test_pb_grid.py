import numpy as np

from coilnet.physics import pb_grid as g
from coilnet.physics.constants import kappa_per_nm


def _grid():
    return g.make_grid(160.0, 48)


def test_greens_matrix_is_symmetric():
    """Electrostatic reciprocity: the many-body Green's matrix must be symmetric."""
    grid = _grid()
    kappa = kappa_per_nm(0.01)
    rng = np.random.default_rng(0)
    centers = [tuple(rng.uniform(40, 120, 3)) for _ in range(5)]
    sigmas = [rng.uniform(6, 12) for _ in range(5)]
    G = g.greens_matrix(centers, sigmas, grid, kappa)
    assert np.linalg.norm(G - G.T) / max(np.linalg.norm(G + G.T), 1e-30) < 1e-9


def test_anion_potential_is_negative():
    grid = _grid()
    kappa = kappa_per_nm(0.01)
    rho = g.gaussian_density(grid, (80.0, 80.0, 80.0), -100.0, 10.0)
    phi = g.solve_field(rho, grid, kappa)
    assert phi.min() < 0
    # potential is strongest near the coil centre and decays away from it
    centre = phi[24, 24, 24]
    edge = phi[5, 5, 5]
    assert abs(centre) > abs(edge)


def test_charge_normalization():
    grid = _grid()
    rho = g.gaussian_density(grid, (80.0, 80.0, 80.0), 50.0, 8.0)
    integral = rho.sum() * grid["dx"] ** 3
    assert abs(integral - 50.0) / 50.0 < 1e-6


def test_nonlinearity_estimate_small_for_small_phi():
    assert g.nonlinearity_estimate(0.4) < 0.05
    assert g.nonlinearity_estimate(0.0) == 0.0
    assert g.nonlinearity_estimate(2.0) > g.nonlinearity_estimate(0.4)


def test_nonlinear_solver_matches_linear_for_small_charge():
    """In the low-potential regime the nonlinear PB stays close to the linear solution and converges."""
    grid = _grid()
    kappa = kappa_per_nm(0.01)
    rho = g.gaussian_density(grid, (80.0, 80.0, 80.0), -100.0, 10.0)
    phi_lin = g.solve_field(rho, grid, kappa)
    phi_nl, info = g.solve_field_nonlinear(rho, grid, kappa)
    rel = np.max(np.abs(phi_nl - phi_lin)) / (np.max(np.abs(phi_lin)) + 1e-30)
    assert rel < 0.05
    assert info["final_change"] < 1e-5


def test_nonlinear_correction_grows_with_charge():
    grid = _grid()
    kappa = kappa_per_nm(0.01)

    def rel_diff(charge):
        rho = g.gaussian_density(grid, (80.0, 80.0, 80.0), -charge, 8.0)
        lin = g.solve_field(rho, grid, kappa)
        nl, _ = g.solve_field_nonlinear(rho, grid, kappa)
        return np.max(np.abs(nl - lin)) / (np.max(np.abs(lin)) + 1e-30)

    assert rel_diff(400.0) > rel_diff(50.0)
