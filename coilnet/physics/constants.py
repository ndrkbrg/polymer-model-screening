"""Physical constants and default parameters (reduced units).

We work in nanometres and in the "reduced potential" phi = e*psi/(kB*T), which keeps all numbers
O(1) and makes the electrostatics clean. Poly(acrylic acid) in water at ~25 C is the target system
(chosen in the plan as the cleanest theory — see docs/).
"""

from __future__ import annotations

# Bjerrum length in water at 25 C: the distance at which two unit charges interact with energy kB*T.
BJERRUM_LENGTH_NM = 0.71


def debye_length_nm(ionic_strength_M):
    """Debye screening length (nm) for a 1:1 salt at 25 C.

    kappa^-1 = 0.304 / sqrt(I) nm, with I in mol/L. This is the reach of a coil's electric
    influence — beyond a few Debye lengths, coils barely feel each other.
    """
    I = max(float(ionic_strength_M), 1e-9)
    return 0.304 / (I**0.5)


def kappa_per_nm(ionic_strength_M):
    """Inverse Debye length kappa (1/nm)."""
    return 1.0 / debye_length_nm(ionic_strength_M)


# Default poly(acrylic acid) coil parameters (order-of-magnitude; the flagship result depends on
# RELATIVE quantities across coils, not these absolute values).
DEFAULT_COIL = {
    "n_monomers": 1000,  # degree of polymerisation
    "ionization": 0.10,  # fraction of carboxyl groups dissociated (weak acid -> small)
    "r_dry_nm": 8.0,  # radius of the fully collapsed coil
    "mean_swelling": 3.0,  # typical swelling ratio (volume) in the swollen state
}
