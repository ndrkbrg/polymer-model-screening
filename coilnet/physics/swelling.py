"""Swelling model: how much a coil swells vs salt, and how sensitively it responds.

This is the "receiver" side of the coupling. A polyelectrolyte coil swells more at low salt (less
screening -> stronger internal charge repulsion -> expansion) and collapses as salt rises. We use a
simple, monotone, clearly-phenomenological model:

    s(I) = s_min + (s_max - s_min) / (1 + (I / I_half)^n)

`swelling_sensitivity` is |ds/dI| — how strongly a coil's swelling reacts to a change in local salt,
which is what turns a neighbour's perturbation into a change of state.

These are modeling choices (v1), varied in the Phase-1 sensitivity study rather than trusted blindly.
"""

from __future__ import annotations

import numpy as np

__all__ = ["swelling_ratio", "swelling_sensitivity", "radius_from_swelling"]


def swelling_ratio(ionic_strength_M, s_min=1.0, s_max=6.0, i_half=0.05, n=1.0):
    """Volume swelling ratio as a decreasing function of ionic strength."""
    I = np.asarray(ionic_strength_M, dtype=float)
    return s_min + (s_max - s_min) / (1.0 + (I / i_half) ** n)


def swelling_sensitivity(ionic_strength_M, **kwargs):
    """|d(swelling)/d(ionic strength)| via a central difference (always positive)."""
    I = float(ionic_strength_M)
    h = max(I * 1e-3, 1e-6)
    lo = max(I - h, 0.0)
    hi = I + h
    s_hi = float(swelling_ratio(hi, **kwargs))
    s_lo = float(swelling_ratio(lo, **kwargs))
    return abs((s_hi - s_lo) / (hi - lo))


def radius_from_swelling(swelling, r_dry_nm=8.0):
    """Coil radius from its volume swelling ratio (R = R_dry * swelling^(1/3))."""
    return r_dry_nm * np.cbrt(np.asarray(swelling, dtype=float))
