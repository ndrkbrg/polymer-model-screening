"""Phase-1b: many-body coupling with receiver heterogeneity, real geometry, two polymer systems.

This hardens the flagship test. The effective neural-network weight between coils is built as

    J[i, j] = receiver_i * M[i, j] * source_j

where M is a SYMMETRIC interaction matrix (reciprocal physics) and the asymmetry of J comes only
from the receiver (i) vs source (j) split — which we can therefore attribute cleanly to physics,
not to a modelling artefact.

Two systems, two mechanisms — showing the asymmetry finding is mechanism-independent:

  * "PAA"    — poly(acrylic acid): electrostatic coupling. M = the many-body screened-Poisson
               Green's function on real geometry (``pb_grid.greens_matrix``, symmetric to machine
               precision, with genuine double-layer overlap). source_j = coil j's charge;
               receiver_i = how sensitively coil i's swelling responds (varies across coils).
  * "NVPVPE" — the N-vinylpyrrolidone/vinyl-propyl-ether copolymer (Paper 2): HYDROPHOBIC
               association, not electrostatics. M = a short-range geometric contact matrix
               (symmetric). source_j = coil j's hydrophobic "stickiness" (grows as it collapses,
               which salt/temperature promote — Paper 2); receiver_i = its response sensitivity.

Receiver heterogeneity: each coil sits at a different point on its (sigmoidal) swelling/collapse
response, so its local slope ds/d(drive) differs — coils near their transition are very responsive,
saturated ones are not. size spread = 0 makes all coils identical (the symmetric null).
"""

from __future__ import annotations

import numpy as np

from .constants import kappa_per_nm, debye_length_nm, DEFAULT_COIL
from .swelling import radius_from_swelling
from . import pb_grid

__all__ = ["build_coupling", "contact_matrix", "SYSTEMS"]

SYSTEMS = ("PAA", "NVPVPE")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _response(u, lo, hi, w):
    """State value and its sensitivity ds/du (the receiver term) at drive u."""
    sig = _sigmoid(u / w)
    state = lo + (hi - lo) * sig
    sensitivity = (hi - lo) / w * sig * (1.0 - sig)
    return state, sensitivity


def _box_for(radii, n, volume_fraction):
    mean_R = float(np.mean(radii))
    return (n * (4.0 / 3.0) * np.pi * mean_R**3 / volume_fraction) ** (1.0 / 3.0)


def contact_matrix(centers, radii, extra_range_nm=1.0):
    """Symmetric short-range contact overlap (for hydrophobic coupling)."""
    n = len(radii)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(np.asarray(centers[i]) - np.asarray(centers[j])))
            overlap = (radii[i] + radii[j] + extra_range_nm) - d
            if overlap > 0:
                M[i, j] = M[j, i] = overlap
    return M


def _build_paa(u, rng, salt_M, grid_n, volume_fraction, resp_w=1.0):
    swelling, receiver = _response(u, lo=1.5, hi=6.0, w=resp_w)  # swelling ratio + ds/du
    radii = radius_from_swelling(swelling, DEFAULT_COIL["r_dry_nm"])
    charge = np.full(len(u), DEFAULT_COIL["ionization"] * DEFAULT_COIL["n_monomers"])  # ~uniform
    box = _box_for(radii, len(u), volume_fraction)
    centers = rng.uniform(0.0, box, size=(len(u), 3))
    kappa = kappa_per_nm(salt_M)
    grid = pb_grid.make_grid(box, grid_n)
    M = pb_grid.greens_matrix([tuple(c) for c in centers], (radii / 2.0), grid, kappa)
    J = receiver[:, None] * M * charge[None, :]
    return {
        "system": "PAA",
        "J": J,
        "M": M,
        "source": charge,
        "receiver": receiver,
        "radii": radii,
        "state": swelling,
        "box_nm": box,
        "debye_nm": debye_length_nm(salt_M),
    }


def _build_nvpvpe(u, rng, salt_M, temperature_C, volume_fraction, resp_w=1.0):
    # NaCl lowers the transition temperature (Paper 2): collapse driven by (T - T0(salt)).
    t0 = 47.0 - 0.6 * (salt_M * 1000.0)  # qualitative: more salt -> lower T0
    drive = u + 0.25 * (temperature_C - t0)  # warmer/saltier -> more collapse
    collapse, receiver = _response(
        drive, lo=0.05, hi=1.0, w=resp_w
    )  # collapse fraction + sensitivity
    swelling = 6.0 - 4.5 * collapse  # collapsed coils are smaller
    radii = radius_from_swelling(swelling, DEFAULT_COIL["r_dry_nm"])
    stickiness = collapse  # exposed hydrophobic patches grow with collapse
    box = _box_for(radii, len(u), volume_fraction)
    centers = rng.uniform(0.0, box, size=(len(u), 3))
    M = contact_matrix([tuple(c) for c in centers], radii, extra_range_nm=1.0)
    J = receiver[:, None] * M * stickiness[None, :]
    return {
        "system": "NVPVPE",
        "J": J,
        "M": M,
        "source": stickiness,
        "receiver": receiver,
        "radii": radii,
        "state": collapse,
        "box_nm": box,
        "t0_C": t0,
    }


def build_coupling(
    n,
    rng,
    system="PAA",
    spread=0.5,
    salt_M=0.01,
    temperature_C=40.0,
    grid_n=48,
    volume_fraction=0.35,
    resp_w=1.0,
):
    """Assemble the effective coupling J for a heterogeneous coil population.

    ``spread`` controls coil heterogeneity (0 = identical coils = symmetric null).
    ``resp_w`` is the width of the swelling/collapse response (varied in the robustness study).
    Returns a dict with J, the symmetric interaction M, per-coil source/receiver, and geometry.
    """
    if system not in SYSTEMS:
        raise ValueError(f"system must be one of {SYSTEMS}")
    u = spread * rng.normal(size=n)
    if system == "PAA":
        return _build_paa(u, rng, salt_M, grid_n, volume_fraction, resp_w)
    return _build_nvpvpe(u, rng, salt_M, temperature_C, volume_fraction, resp_w)
