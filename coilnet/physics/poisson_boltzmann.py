"""Modified Poisson-Boltzmann solver for a single weak-polyacid coil (Paper 3, reduced).

We solve the linearized (Debye-Huckel) screened-Poisson equation for the reduced electrostatic
potential phi(r) = e*psi/(kB*T) around one coil, with a fixed dissociated-charge source distributed
as a Gaussian (denser at the centre, per Paper 3):

    laplacian(phi) - kappa^2 * phi = -4*pi*l_B * n_fix(r)

In radial form, with u = r*phi this becomes the simple linear ODE

    u'' - kappa^2 * u = -4*pi*l_B * r * n_fix(r)

solved as a boundary-value problem with u(0)=0 (phi finite at the centre) and u(r_max)=0
(phi -> 0 far away). Outputs the potential profile plus two derived quantities used downstream:

  * surface_potential  -> the coil's "source strength" felt by a touching neighbour,
  * tail_coeff A       -> the far-field amplitude phi ~ A*exp(-kappa*r)/r (a screened Coulomb).

Validation (see tests): the solver is LINEAR in the charge, and in the weak-screening limit its
far-field amplitude approaches the analytic Coulomb value A -> l_B * Z.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.integrate import solve_bvp

from .constants import BJERRUM_LENGTH_NM, kappa_per_nm, debye_length_nm

__all__ = ["solve_single_coil", "source_strength"]


def _gaussian_number_density(r, total, sigma):
    """Gaussian charge-number density normalized so integral over 4*pi*r^2 dr = total."""
    norm = total / ((2.0 * np.pi * sigma**2) ** 1.5)
    return norm * np.exp(-(r**2) / (2.0 * sigma**2))


def solve_single_coil(
    radius_nm=20.0,
    n_monomers=1000,
    ionization=0.10,
    ionic_strength_M=0.01,
    l_b_nm=BJERRUM_LENGTH_NM,
    r_max_factor=6.0,
    n_mesh=400,
    sign=-1.0,
):
    """Solve the modified PB equation for one coil. Returns a dict of profile + derived quantities."""
    kappa = kappa_per_nm(ionic_strength_M)
    debye = debye_length_nm(ionic_strength_M)
    Z = ionization * n_monomers  # number of ionized (carboxylate) groups
    sigma = radius_nm / 2.0  # Gaussian half-width (denser at centre)
    r_max = max(radius_nm, debye) * r_max_factor + radius_nm
    r0 = 1e-3 * radius_nm
    mesh = np.linspace(r0, r_max, n_mesh)

    def rhs(r, y):
        n_fix = sign * _gaussian_number_density(r, Z, sigma)  # negative for an anionic coil
        return np.vstack([y[1], kappa**2 * y[0] - 4.0 * np.pi * l_b_nm * r * n_fix])

    def bc(ya, yb):
        return np.array([ya[0], yb[0]])  # u(r0)=0, u(r_max)=0

    y_init = np.zeros((2, mesh.size))
    sol = solve_bvp(rhs, bc, mesh, y_init, max_nodes=40000, tol=1e-6)

    r = np.linspace(r0, r_max, 600)
    sol_vals = sol.sol(r)
    u = sol_vals[0]
    phi = u / r

    phi_surface = float(np.interp(radius_nm, r, phi))
    # phi(0) = u'(0) (since u = r*phi and u(0)=0); reading u/r at the boundary node gives a false 0.
    phi_center = float(sol.sol(np.array([r0]))[1][0])

    # Estimate the far-field amplitude A (phi ~ A*exp(-kappa r)/r) in a BOUNDED window just outside
    # the coil, where phi is still resolved. Sampling too far out multiplies ~0 by a huge exp(kappa r)
    # and gives numerical garbage.
    lo = radius_nm + 0.5 * debye
    hi = min(radius_nm + 4.0 * debye, 0.9 * r_max)
    tail_mask = (r >= lo) & (r <= hi)
    if tail_mask.any():
        tail_coeff = float(np.median(phi[tail_mask] * r[tail_mask] * np.exp(kappa * r[tail_mask])))
    else:
        tail_coeff = float("nan")

    return {
        "r": r,
        "phi": phi,
        "kappa": kappa,
        "debye_nm": debye,
        "radius_nm": radius_nm,
        "Z": Z,
        "surface_potential": phi_surface,
        "center_potential": phi_center,
        "tail_coeff": tail_coeff,
        "success": bool(sol.success),
    }


@lru_cache(maxsize=8192)
def _solve_cached(radius_bin, n_monomers, ionization, ionic_strength_bin):
    return solve_single_coil(
        radius_nm=radius_bin,
        n_monomers=n_monomers,
        ionization=ionization,
        ionic_strength_M=ionic_strength_bin,
    )


def source_strength(radius_nm, n_monomers, ionization, ionic_strength_M):
    """|surface potential| of a coil = how strongly it drives a touching neighbour.

    Cached on 0.5-nm radius bins so building W for many coils stays fast.
    """
    radius_bin = round(radius_nm * 2.0) / 2.0
    ionic_bin = round(float(ionic_strength_M), 4)
    return abs(_solve_cached(radius_bin, n_monomers, ionization, ionic_bin)["surface_potential"])
