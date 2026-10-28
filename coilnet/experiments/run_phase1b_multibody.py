"""Phase 1b (hardened) experiment: many-body coupling, real geometry, two polymer systems.

Upgrades the flagship test:
  * PAA coupling from the full many-body screened-Poisson solve (double-layer overlap, real
    geometry) instead of a pairwise kernel;
  * receiver-side heterogeneity made explicit;
  * a second polymer system (NVP-VPE, Paper 2) coupled by HYDROPHOBIC contact, not electrostatics.

Key finding it establishes: the interaction matrix M is symmetric (reciprocal) to machine
precision, so any asymmetry in the effective memory-weight J is attributable ENTIRELY to the
receiver-vs-source split (response heterogeneity) — modest for PAA, substantial for NVP-VPE, and
zero only for identical coils.
"""

from __future__ import annotations

import numpy as np

from coilnet import coupling, figs, report
from coilnet.hopfield import HopfieldNetwork
from coilnet.physics import multibody as mb

SPREADS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
THRESHOLD = 0.05
N = 30
SALT_M = 0.01


def energy_fail_rate(J, n_starts=60):
    net = HopfieldNetwork(J)
    n = J.shape[0]
    srng = np.random.default_rng(9)
    fails = 0
    for k in range(n_starts):
        s0 = srng.choice([-1.0, 1.0], size=n)
        _, e = net.update_async(s0, rng=np.random.default_rng(k), record_energy=True, max_sweeps=40)
        fails += int(np.any(np.diff(e) > 1e-9))
    return fails / n_starts


def diagonal_symmetrization(result):
    """Construct a symmetric coupling with the same zero-threshold sign dynamics."""
    source = np.asarray(result["source"], dtype=float)
    receiver = np.asarray(result["receiver"], dtype=float)
    if np.any(source <= 0.0) or np.any(receiver <= 0.0):
        raise ValueError("diagonal symmetrization requires positive factors")
    K = (source / receiver)[:, None] * np.asarray(result["J"], dtype=float)
    rng = np.random.default_rng(2718)
    sign_equivalent = all(
        np.array_equal(
            np.sign(result["J"] @ (state := rng.choice([-1.0, 1.0], size=K.shape[0]))),
            np.sign(K @ state),
        )
        for _ in range(100)
    )
    return K, float(coupling.asymmetry_index(K)), bool(sign_equivalent)


def main():
    series = {}
    details = {}
    for system in mb.SYSTEMS:
        curve, interaction_asym = [], []
        for sp in SPREADS:
            aJ, aM = [], []
            for s in range(3):
                rng = np.random.default_rng(100 + s)
                r = mb.build_coupling(N, rng, system=system, spread=sp, salt_M=SALT_M, grid_n=44)
                aJ.append(coupling.asymmetry_index(r["J"]))
                aM.append(coupling.asymmetry_index(r["M"]))
            curve.append(float(np.mean(aJ)))
            interaction_asym.append(float(np.mean(aM)))
        series[system] = curve

        rng = np.random.default_rng(7)
        rep = mb.build_coupling(N, rng, system=system, spread=1.0, salt_M=SALT_M, grid_n=44)
        fail = energy_fail_rate(rep["J"])
        K, weighted_asymmetry, sign_equivalent = diagonal_symmetrization(rep)
        weighted_fail = energy_fail_rate(K)
        figs.plot_heatmap(
            rep["J"],
            name=f"phase1b_J_{system}",
            title=f"{system}: effective coupling J (spread 1.0)",
        )

        cross = next((SPREADS[k] for k, v in enumerate(curve) if v > THRESHOLD), None)
        details[system] = {
            "asymmetry_curve_J": curve,
            "interaction_asymmetry_M": float(np.mean(interaction_asym)),
            "asymmetry_at_spread_0p5": curve[SPREADS.index(0.5)],
            "crosses_threshold_at_spread": cross,
            "energy_fail_rate_at_spread_1": fail,
            "diagonally_symmetrized_asymmetry": weighted_asymmetry,
            "weighted_energy_fail_rate_at_spread_1": weighted_fail,
            "zero_threshold_sign_dynamics_equivalent": sign_equivalent,
        }
        report.log(
            f"[{system}] asym(J)={['%.3f' % v for v in curve]}; asym(M)~{np.mean(interaction_asym):.1e}; "
            f"crosses {THRESHOLD} at spread {cross}; ordinary-energy-fail={fail * 100:.0f}%; "
            f"weighted-asym={weighted_asymmetry:.1e}; weighted-energy-fail={weighted_fail * 100:.0f}%; "
            f"sign-equivalent={sign_equivalent}"
        )

    figs.plot_multibody_sweep(SPREADS, series, threshold=THRESHOLD)

    verdict = (
        "Interaction M is reciprocal while effective coupling J is asymmetric when source and "
        "receiver factors differ. Because all factors in these models are positive, "
        "diag(source/receiver) J is symmetric and has the same zero-threshold sign dynamics as J. "
        "The ordinary energy written directly with J is therefore not the appropriate diagnostic: "
        "a weighted Lyapunov function exists for this factorized model. Heterogeneity invalidates "
        "the naive identification of J as a symmetric weight matrix, but does not by itself rule "
        "out attractor dynamics."
    )
    report.log("VERDICT: " + verdict)

    report.save_json(
        "phase1b_multibody",
        {
            "n_coils": N,
            "salt_M": SALT_M,
            "spreads": SPREADS,
            "threshold": THRESHOLD,
            "systems": details,
            "verdict": verdict,
            "note": (
                "PAA: many-body screened-Poisson Green's function (double-layer overlap, real "
                "geometry). NVPVPE: geometric hydrophobic contact (Paper 2 mechanism). Both: J = "
                "diag(receiver)·M·diag(source), M symmetric. Linearized PB (~3% nonlinearity at these "
                "low potentials). Not calibrated to data — a physical consistency test."
            ),
        },
    )


if __name__ == "__main__":
    main()
