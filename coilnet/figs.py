"""Matplotlib figure helpers (headless-safe).

Uses the non-interactive Agg backend so figures render without a display (e.g. in scripts,
CI, or the dashboard). Every helper saves a PNG into ``results/`` and returns its path.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import report  # noqa: E402

__all__ = [
    "plot_energy_traces",
    "plot_capacity",
    "plot_hysteresis",
    "plot_heatmap",
    "plot_bars",
    "plot_potential",
    "plot_asymmetry_sweep",
    "plot_multibody_sweep",
]


def _save(fig, name):
    path = report.fig_path(name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    report.log(f"saved figure {path.name}")
    return path


def plot_energy_traces(traces, labels, name="energy", title="Hopfield energy vs updates"):
    fig, ax = plt.subplots(figsize=(6, 4))
    for trace, label in zip(traces, labels):
        ax.plot(trace, label=label)
    ax.set_xlabel("single-neuron update step")
    ax.set_ylabel("energy E")
    ax.set_title(title)
    ax.legend()
    return _save(fig, name)


def plot_capacity(results, N, name="capacity"):
    loads = [r["load"] for r in results]
    frac = [r["stable_fraction"] for r in results]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(loads, frac, "o-", label="fraction of patterns stable")
    ax.axvline(0.138, ls="--", color="red", label="0.138 (validation reference)")
    ax.set_xlabel("load  P / N")
    ax.set_ylabel("stable fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Hebbian capacity check (N = {N})")
    ax.legend()
    return _save(fig, name)


def plot_hysteresis(drives, states, name="hysteresis"):
    n = len(drives) // 2
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(drives[:n], states[:n], "-o", ms=3, label="drive rising")
    ax.plot(drives[n:], states[n:], "-o", ms=3, label="drive falling")
    ax.set_xlabel("drive")
    ax.set_ylabel("coil state (0 = collapsed, 1 = swollen)")
    ax.set_title("Hysteresis loop = binary memory cell")
    ax.legend()
    return _save(fig, name)


def plot_heatmap(W, name="W", title="coupling matrix W"):
    W = np.asarray(W, dtype=float)
    vmax = np.max(np.abs(W)) if np.max(np.abs(W)) > 0 else 1.0
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(W, cmap="RdBu", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("source coil j")
    ax.set_ylabel("target coil i")
    return _save(fig, name)


def plot_bars(labels, values, name, title="", xlabel="", ylabel=""):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(values)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return _save(fig, name)


def plot_potential(res, name="phase1_single_coil"):
    """Plot the reduced electrostatic potential phi(r) around one coil (from the PB solve)."""
    r = np.asarray(res["r"])
    phi = np.asarray(res["phi"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r, phi, label="phi(r)  (reduced potential)")
    ax.axvline(
        res["radius_nm"], ls="--", color="green", label=f"coil radius = {res['radius_nm']:.0f} nm"
    )
    ax.axvline(
        res["radius_nm"] + res["debye_nm"],
        ls=":",
        color="orange",
        label=f"+1 Debye length ({res['debye_nm']:.1f} nm)",
    )
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("distance from coil centre r (nm)")
    ax.set_ylabel("phi = e·psi / kT")
    ax.set_title("Charged skin of a single polyacid coil (modified Poisson–Boltzmann)")
    ax.legend(fontsize=8)
    return _save(fig, name)


def plot_asymmetry_sweep(spreads, asymmetry, threshold=0.05, name="phase1_asymmetry_sweep"):
    """Plot how the derived coupling's asymmetry grows with coil-size heterogeneity."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(spreads, asymmetry, "o-", label="asymmetry of physics-derived W")
    ax.axhline(threshold, ls="--", color="red", label=f"falsification threshold ({threshold})")
    ax.set_xlabel("coil-size spread (0 = identical coils)")
    ax.set_ylabel("asymmetry index (0 = Hopfield-safe)")
    ax.set_ylim(bottom=-0.02)
    ax.set_title("Does the physics give a symmetric (memory-capable) W?")
    ax.legend(fontsize=8)
    return _save(fig, name)


def plot_multibody_sweep(spreads, series, threshold=0.05, name="phase1b_asymmetry"):
    """Compare asymmetry-vs-heterogeneity across polymer systems (many-body model)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, vals in series.items():
        ax.plot(spreads, vals, "o-", label=label)
    ax.axhline(threshold, ls="--", color="red", label=f"threshold ({threshold})")
    ax.set_xlabel("coil heterogeneity (spread)")
    ax.set_ylabel("asymmetry of effective coupling J")
    ax.set_ylim(bottom=-0.02)
    ax.set_title("Many-body coupling: asymmetry from receiver heterogeneity")
    ax.legend(fontsize=8)
    return _save(fig, name)
