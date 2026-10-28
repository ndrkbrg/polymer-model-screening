"""Multi-coil Poisson-Boltzmann on a 3-D grid (double-layer overlap, real geometry).

Phase-1b upgrade over the single-coil radial solver. We place every coil's fixed-charge cloud on a
periodic 3-D grid at its real position and solve the screened-Poisson (linearized PB) equation

    laplacian(phi) - kappa^2 * phi = -4*pi*l_B * rho_fix(r)

for the *whole configuration at once* via FFT. Because all clouds are present together, the solution
includes genuine many-body superposition and double-layer OVERLAP on the real geometry — not the
pairwise ``exp(-gap*kappa)`` kernel of the v1 model.

Why linearized PB is justified here: the reduced potentials are small (|phi| < ~0.5), so
sinh(phi) ~ phi to a few percent (quantified by ``nonlinearity_estimate``). An optional Picard
correction for the sinh term is provided but is a small correction in this regime.

Key object: ``greens_matrix`` returns G[i,j] = potential coil i feels from coil j's *unit* charge,
averaged over coil i's body. G is SYMMETRIC (electrostatic reciprocity) — a fact we test. The
asymmetry of the effective coupling therefore comes from the physics wrapped around G (source vs
receiver), not from the electrostatics itself.
"""

from __future__ import annotations

import numpy as np

from .constants import BJERRUM_LENGTH_NM

__all__ = [
    "make_grid",
    "gaussian_density",
    "solve_field",
    "solve_field_nonlinear",
    "greens_matrix",
    "nonlinearity_estimate",
]


def make_grid(box_nm, n):
    axis = np.arange(n) * (box_nm / n)
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    return {"X": X, "Y": Y, "Z": Z, "dx": box_nm / n, "box": float(box_nm), "n": int(n)}


def _min_image(delta, box):
    return delta - box * np.round(delta / box)


def gaussian_density(grid, center, total, sigma):
    """A Gaussian cloud (periodic minimum-image) normalized so its volume integral == ``total``."""
    dx = _min_image(grid["X"] - center[0], grid["box"])
    dy = _min_image(grid["Y"] - center[1], grid["box"])
    dz = _min_image(grid["Z"] - center[2], grid["box"])
    g = np.exp(-(dx * dx + dy * dy + dz * dz) / (2.0 * sigma**2))
    vol = grid["dx"] ** 3
    integral = g.sum() * vol
    return g * (total / integral)


def solve_field(rho, grid, kappa, l_b=BJERRUM_LENGTH_NM):
    """Solve laplacian(phi) - kappa^2 phi = -4 pi l_b rho on the periodic grid (FFT)."""
    n, dx = grid["n"], grid["dx"]
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    k2 = KX**2 + KY**2 + KZ**2
    phi_hat = 4.0 * np.pi * l_b * np.fft.fftn(rho) / (k2 + kappa**2)
    return np.real(np.fft.ifftn(phi_hat))


def solve_field_nonlinear(
    rho, grid, kappa, l_b=BJERRUM_LENGTH_NM, iters=120, omega=0.3, tol=1e-6, clip=6.0
):
    """Solve the FULL (nonlinear) PB: laplacian(phi) - kappa^2 phi = -4 pi l_b rho + kappa^2 (sinh phi - phi).

    Damped Picard iteration reusing the FFT linear solve. The sinh argument is clipped to avoid
    overflow when potentials are large (strong nonlinearity). Returns ``(phi, info)`` with the
    iteration count, final change, and a ``converged`` flag (Picard can fail to converge in strongly
    nonlinear regimes — reported honestly rather than silently returning garbage).
    """
    n, dx = grid["n"], grid["dx"]
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    denom = KX**2 + KY**2 + KZ**2 + kappa**2

    phi = solve_field(rho, grid, kappa, l_b)  # linearized solution as the starting guess
    change = float("nan")
    converged = False
    it = 0
    for it in range(iters):
        phi_c = np.clip(phi, -clip, clip)
        source = 4.0 * np.pi * l_b * rho - kappa**2 * (np.sinh(phi_c) - phi_c)
        phi_lin = np.real(np.fft.ifftn(np.fft.fftn(source) / denom))
        phi_next = (1.0 - omega) * phi + omega * phi_lin
        denom_change = np.max(np.abs(phi_next)) + 1e-30
        change = float(np.max(np.abs(phi_next - phi)) / denom_change)
        phi = phi_next
        if not np.isfinite(change):
            break
        if change < tol:
            converged = True
            break
    return phi, {"iterations": it + 1, "final_change": change, "converged": converged}


def greens_matrix(centers, sigmas, grid, kappa, l_b=BJERRUM_LENGTH_NM):
    """G[i,j] = body-averaged potential at coil i from coil j's UNIT charge cloud.

    Symmetric by construction (electrostatic reciprocity). Captures finite size, overlap and
    many-body geometry. Diagonal (self-term) is included but not used by the coupling model.
    """
    n = len(centers)
    vol = grid["dx"] ** 3
    fields = [
        solve_field(gaussian_density(grid, centers[j], 1.0, sigmas[j]), grid, kappa, l_b)
        for j in range(n)
    ]
    probes = [
        gaussian_density(grid, centers[i], 1.0, sigmas[i]) * vol for i in range(n)
    ]  # sum -> 1
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            G[i, j] = float(np.sum(fields[j] * probes[i]))
    return G


def nonlinearity_estimate(phi_max):
    """Relative size of the neglected nonlinear term: (sinh(phi)-phi)/sinh(phi) at |phi|=phi_max."""
    p = abs(float(phi_max))
    if p == 0:
        return 0.0
    return float((np.sinh(p) - p) / np.sinh(p))
