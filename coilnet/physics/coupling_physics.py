"""Build the coupling matrix W from physics — the heart of Phase 1.

For each ordered pair of coils (i <- j) we model the wiring weight as

    W[i, j] = sensitivity_i * source_j * exp(-gap_ij * kappa)

where
  * source_j       = |surface potential| of coil j (from the Poisson-Boltzmann solve) — how hard
                     coil j drives its surroundings. It depends on j's size/charge STATE: at fixed
                     chain charge, a more-swollen coil dilutes its charge over a bigger volume and so
                     has a WEAKER surface potential. The point is only that source varies from coil
                     to coil when they differ in size.
  * sensitivity_i  = how strongly coil i's swelling responds to a local change (receiver side).
  * gap_ij         = surface-to-surface distance between coils i and j; the screened kernel
                     exp(-gap*kappa) makes coupling matter only for coils in contact / within a
                     few Debye lengths (as Paper 3 states).

Why this is the point: `source` depends on the SOURCE coil (j) while `sensitivity` depends on the
RECEIVER coil (i). When coils differ in size, source_j != source_i, so W[i,j] != W[j,i] — the wiring
is ASYMMETRIC. A [[Hopfield]] memory needs symmetric wiring, so this is exactly the flagship test.
When all coils are identical, W is symmetric (the honest null).

MODELING NOTE (v1): receiver `sensitivity` is treated as ~uniform here (a function of the shared
salt), so the derived asymmetry comes from the source side (coil-size heterogeneity). This is a
deliberate, documented simplification; the Phase-1 study sweeps parameters rather than trusting a
single choice.
"""

from __future__ import annotations

import numpy as np

from .constants import kappa_per_nm, DEFAULT_COIL
from .poisson_boltzmann import source_strength
from .swelling import swelling_sensitivity, radius_from_swelling

__all__ = ["sample_coils", "derive_coupling"]


def sample_coils(
    n,
    rng,
    mean_swelling=None,
    size_spread=0.5,
    volume_fraction=0.30,
    n_monomers=None,
    ionization=None,
    r_dry_nm=None,
):
    """Sample a population of coils with a controllable size spread.

    ``size_spread`` = 0 makes every coil identical (the symmetric null); larger values make the
    swelling (hence size, hence source strength) more heterogeneous. Positions fill a box sized so
    the coils sit near their overlap concentration, guaranteeing contacts.
    """
    mean_swelling = DEFAULT_COIL["mean_swelling"] if mean_swelling is None else mean_swelling
    n_monomers = DEFAULT_COIL["n_monomers"] if n_monomers is None else n_monomers
    ionization = DEFAULT_COIL["ionization"] if ionization is None else ionization
    r_dry_nm = DEFAULT_COIL["r_dry_nm"] if r_dry_nm is None else r_dry_nm

    swelling = mean_swelling * np.exp(size_spread * rng.normal(size=n))
    swelling = np.clip(swelling, 0.2, None)
    radius = radius_from_swelling(swelling, r_dry_nm)

    mean_R = float(radius.mean())
    box_nm = (n * (4.0 / 3.0) * np.pi * mean_R**3 / volume_fraction) ** (1.0 / 3.0)
    positions = rng.uniform(0.0, box_nm, size=(n, 3))

    return {
        "positions": positions,
        "swelling": swelling,
        "radius": radius,
        "box_nm": box_nm,
        "n_monomers": n_monomers,
        "ionization": ionization,
    }


def derive_coupling(coils, ionic_strength_M=0.01, contact_factor=3.0):
    """Compute W (and diagnostics) from a coil population using the physics model above."""
    pos = np.asarray(coils["positions"], dtype=float)
    R = np.asarray(coils["radius"], dtype=float)
    n = len(R)
    kappa = kappa_per_nm(ionic_strength_M)
    debye = 1.0 / kappa

    source = np.array(
        [
            source_strength(R[i], coils["n_monomers"], coils["ionization"], ionic_strength_M)
            for i in range(n)
        ]
    )
    # Receiver sensitivity is uniform in this v1 (function of the shared salt); see module note.
    sensitivity = np.full(n, swelling_sensitivity(ionic_strength_M))

    W = np.zeros((n, n))
    n_contacts = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            gap = float(np.linalg.norm(pos[i] - pos[j]) - (R[i] + R[j]))
            if gap > contact_factor * debye:
                continue
            W[i, j] = sensitivity[i] * source[j] * np.exp(-max(gap, 0.0) * kappa)
            n_contacts += 1

    return {
        "W": W,
        "source": source,
        "sensitivity": sensitivity,
        "kappa": kappa,
        "debye_nm": debye,
        "n_contacts": n_contacts,
        "ionic_strength_M": ionic_strength_M,
    }
