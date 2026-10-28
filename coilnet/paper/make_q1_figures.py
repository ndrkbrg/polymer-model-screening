"""Build bilingual Engineered Science figures solely from saved Q1 CSV files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from coilnet.paths import ORIGINAL, WORK, output_directory

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


COLORS = ["#3366A6", "#D47A2A", "#3A8D5D", "#9A5FA5", "#6E7B8B", "#C44E52"]


def _read(path):
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _float(row, key, default=np.nan):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _write_source(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _save(fig, root, name, lang):
    # Wording used in the revised article; numerical figure data are unchanged.
    if lang == "en":
        if name.startswith("figure1"):
            titles = [
                "A  Native-attractor recovery",
                "B  Numerical reservoir",
                "C  Multi-state storage/control",
                "D  Generic constrained RBM",
            ]
            for ax, title in zip(fig.axes, titles):
                ax.set_title(title, loc="left", fontsize=9, weight="bold")
        if name.startswith("figure4"):
            fig.axes[0].set_xticks(
                range(6),
                ["bist.", "graded", "model", "random", "sym.", "shuffle"],
                rotation=30,
                ha="right",
            )
        if name.startswith("figure7"):
            fig.axes[0].set_xlabel("One-time weight error, %")
        if name.startswith(("figure1", "figure4", "figure7")):
            fig.tight_layout()
    target = root if lang == "en" else root / "ru"
    target.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "svg", "png"):
        path = target / f"{name}.{ext}"
        fig.savefig(path, dpi=600 if ext == "png" else None, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def _figure1(root, lang):
    ru = lang == "ru"
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.2))
    titles = (
        ["A  Память", "B  Резервуар", "C  Многозначные состояния", "D  Энергетическая модель"]
        if ru
        else [
            "A  Attractor memory",
            "B  Reservoir",
            "C  Multi-valued states",
            "D  Energy-based model",
        ]
    )
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, loc="left", fontsize=10, weight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    # A: cue, energy landscape, and recovered pattern.
    ax = axes[0, 0]
    cue = np.array([[1, 1, 1], [1, 0, 1], [0, 1, 0]])
    recovered = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    for offset, image in ((0.06, cue), (0.78, recovered)):
        for i in range(3):
            for j in range(3):
                ax.add_patch(
                    plt.Rectangle(
                        (offset + 0.055 * j, 0.38 + 0.075 * (2 - i)),
                        0.05,
                        0.067,
                        facecolor="#3366A6" if image[i, j] else "white",
                        edgecolor="#455A64",
                        lw=0.6,
                    )
                )
    x = np.linspace(-1.2, 1.2, 100)
    y = 0.48 + 0.18 * (x**2 - 0.7) ** 2
    ax.plot(0.5 + 0.16 * x, y, color="#D47A2A", lw=1.4)
    ax.plot([0.61], [0.49], "o", color="#C44E52", ms=5)
    ax.annotate(
        "",
        xy=(0.40, 0.50),
        xytext=(0.25, 0.50),
        arrowprops={"arrowstyle": "->", "color": "#455A64"},
    )
    ax.annotate(
        "",
        xy=(0.76, 0.50),
        xytext=(0.63, 0.50),
        arrowprops={"arrowstyle": "->", "color": "#455A64"},
    )
    labels = (
        ("повреждённая\nподсказка", "энергетический\nландшафт", "аттрактор")
        if ru
        else ("corrupted\ncue", "energy\nlandscape", "attractor")
    )
    for xpos, text in zip((0.14, 0.50, 0.86), labels):
        ax.text(xpos, 0.25, text, ha="center", va="top", fontsize=7)

    # B: temporal drive, recurrent reservoir, and trained read-out.
    ax = axes[0, 1]
    tx = np.linspace(0.05, 0.27, 80)
    ax.plot(tx, 0.53 + 0.11 * np.sin(np.linspace(0, 3.5 * np.pi, 80)), color="#3366A6", lw=1.4)
    nodes = np.array(
        [[0.40, 0.40], [0.48, 0.62], [0.57, 0.42], [0.64, 0.65], [0.69, 0.34], [0.53, 0.52]]
    )
    for i, j in ((0, 1), (0, 5), (1, 3), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)):
        ax.plot(nodes[[i, j], 0], nodes[[i, j], 1], color="#9AA7AD", lw=0.7)
    ax.scatter(
        nodes[:, 0],
        nodes[:, 1],
        s=45,
        color="#F1B36A",
        edgecolor="#455A64",
        linewidth=0.6,
        zorder=3,
    )
    ax.add_patch(plt.Rectangle((0.80, 0.41), 0.15, 0.18, facecolor="#E5F2E9", edgecolor="#455A64"))
    ax.text(0.875, 0.50, r"$\hat{y}_t$", ha="center", va="center", fontsize=10)
    for start, end in (((0.28, 0.52), (0.36, 0.52)), ((0.72, 0.52), (0.79, 0.52))):
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#455A64"})
    labels = (
        ("вход $u_t$", "фиксированный\nрезервуар", "обучаемое\nсчитывание")
        if ru
        else ("input $u_t$", "fixed\nreservoir", "trained\nread-out")
    )
    for xpos, text in zip((0.16, 0.54, 0.875), labels):
        ax.text(xpos, 0.25, text, ha="center", va="top", fontsize=7)

    # C: addressable states and the desired cyclic operation.
    ax = axes[1, 0]
    state_xy = np.array([[0.14, 0.62], [0.14, 0.36], [0.34, 0.62], [0.34, 0.36]])
    ax.scatter(state_xy[:, 0], state_xy[:, 1], s=150, color="#EAF2F8", edgecolor="#3366A6")
    for idx, (x0, y0) in enumerate(state_xy):
        ax.text(x0, y0, str(idx), ha="center", va="center", fontsize=8)
    ax.annotate(
        "",
        xy=(0.50, 0.50),
        xytext=(0.39, 0.50),
        arrowprops={"arrowstyle": "->", "color": "#455A64"},
    )
    ax.text(0.445, 0.57, "field" if not ru else "поле", ha="center", fontsize=7)
    circle = np.array([[0.67, 0.68], [0.83, 0.68], [0.83, 0.38], [0.67, 0.38]])
    ax.scatter(circle[:, 0], circle[:, 1], s=130, color="#E5F2E9", edgecolor="#3A8D5D")
    for idx, (x0, y0) in enumerate(circle):
        ax.text(x0, y0, str(idx), ha="center", va="center", fontsize=8)
    for i, j in ((0, 1), (1, 2), (2, 3), (3, 0)):
        ax.annotate(
            "",
            xy=circle[j],
            xytext=circle[i],
            arrowprops={"arrowstyle": "->", "color": "#3A8D5D", "shrinkA": 8, "shrinkB": 8},
        )
    ax.text(0.24, 0.22, "устойчивые состояния" if ru else "stable states", ha="center", fontsize=7)
    ax.text(0.75, 0.22, "требуемый цикл" if ru else "required cycle", ha="center", fontsize=7)

    # D: bipartite RBM and stochastic generation.
    ax = axes[1, 1]
    visible = np.array([[0.13, y] for y in np.linspace(0.30, 0.72, 4)])
    hidden = np.array([[0.39, y] for y in np.linspace(0.35, 0.67, 3)])
    for v in visible:
        for h in hidden:
            ax.plot([v[0], h[0]], [v[1], h[1]], color="#B0BEC5", lw=0.5)
    ax.scatter(visible[:, 0], visible[:, 1], s=55, color="#3366A6", edgecolor="#455A64", zorder=3)
    ax.scatter(hidden[:, 0], hidden[:, 1], s=55, color="#D47A2A", edgecolor="#455A64", zorder=3)
    ax.annotate(
        "",
        xy=(0.66, 0.51),
        xytext=(0.47, 0.51),
        arrowprops={"arrowstyle": "->", "color": "#455A64"},
    )
    sample = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]])
    for i in range(3):
        for j in range(3):
            ax.add_patch(
                plt.Rectangle(
                    (0.73 + 0.055 * j, 0.39 + 0.075 * (2 - i)),
                    0.05,
                    0.067,
                    facecolor="#3A8D5D" if sample[i, j] else "white",
                    edgecolor="#455A64",
                    lw=0.6,
                )
            )
    ax.text(0.26, 0.22, "видимые ↔ скрытые" if ru else "visible ↔ hidden", ha="center", fontsize=7)
    ax.text(0.56, 0.59, "выборка" if ru else "sampling", ha="center", fontsize=7)
    ax.text(0.81, 0.22, "образец" if ru else "sample", ha="center", fontsize=7)
    fig.tight_layout()
    return _save(fig, root, "figure1_computational_regimes", lang)


def _figure2(data, root, lang, source_dir):
    rows = [
        r
        for r in _read(data / "attractor_recall_summary.csv")
        if r.get("experiment") == "symmetry_sensitivity"
        and r.get("metric") == "symmetrizability_error"
    ]
    _write_source(source_dir / "figure2.csv", rows)
    fig, ax = plt.subplots(figsize=(7.1, 3.7))
    for color, system in zip(COLORS, ("PAA", "NVPVPE")):
        subset = sorted(
            [r for r in rows if r.get("system") == system], key=lambda r: _float(r, "perturbation")
        )
        ax.plot(
            [100 * _float(r, "perturbation") for r in subset],
            [_float(r, "mean") for r in subset],
            "o-",
            color=color,
            label=system,
        )
        ax.fill_between(
            [100 * _float(r, "perturbation") for r in subset],
            [_float(r, "ci95_low") for r in subset],
            [_float(r, "ci95_high") for r in subset],
            color=color,
            alpha=0.18,
        )
    ax.axhline(
        0.01,
        ls="--",
        color="black",
        lw=1,
        label="практический порог" if lang == "ru" else "practical threshold",
    )
    ax.set(
        xlabel="Нефакторизуемое возмущение, %"
        if lang == "ru"
        else "Non-factorizable perturbation, %",
        ylabel="Ошибка симметризации" if lang == "ru" else "Symmetrizability error",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, root, "figure2_coupling_symmetrizability", lang)


def _figure3(data, root, lang, source_dir):
    rows = [
        r
        for r in _read(data / "attractor_recall_summary.csv")
        if r.get("experiment") == "recall"
        and r.get("metric") == "perfect_recall_rate"
        and _float(r, "spread") == 1.0
        and _float(r, "threshold_fraction") == 0.0
        and r.get("representation") == "weighted"
    ]
    if rows:
        nvals = sorted({_float(r, "N") for r in rows})
        target_n = min(nvals, key=lambda x: abs(x - 30))
        rows = [r for r in rows if _float(r, "N") == target_n]
    _write_source(source_dir / "figure3.csv", rows)
    fig, ax = plt.subplots(figsize=(7.1, 3.8))
    for color, system in zip(COLORS, ("PAA", "NVPVPE")):
        subset = sorted(
            [r for r in rows if r.get("system") == system], key=lambda r: _float(r, "corruption")
        )
        ax.errorbar(
            [100 * _float(r, "corruption") for r in subset],
            [_float(r, "mean") for r in subset],
            yerr=[
                [max(0, _float(r, "mean") - _float(r, "ci95_low")) for r in subset],
                [max(0, _float(r, "ci95_high") - _float(r, "mean")) for r in subset],
            ],
            marker="o",
            capsize=3,
            color=color,
            label=system,
        )
    ax.set(
        xlabel="Повреждённые биты, %" if lang == "ru" else "Corrupted bits, %",
        ylabel="Доля полного восстановления" if lang == "ru" else "Perfect-recall fraction",
        ylim=(-0.03, 1.03),
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, root, "figure3_attractor_recall", lang)


def _figure4(data, root, lang, source_dir):
    allrows = _read(data / "reservoir_summary.csv")
    rows = [
        r for r in allrows if r.get("experiment") == "main" and r.get("metric") == "narma10_nrmse"
    ]
    phase = [
        r
        for r in allrows
        if r.get("experiment") == "phase_map"
        and r.get("metric") == "narma10_nrmse"
        and _float(r, "gain") == 0.6
    ]
    _write_source(source_dir / "figure4.csv", rows + phase)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.8))
    models = [
        "bistable_langevin",
        "graded_langevin",
        "physical_tanh",
        "random_tanh",
        "symmetric_tanh",
        "shuffled_tanh",
    ]
    paa = [
        next((r for r in rows if r.get("system") == "PAA" and r.get("model") == m), None)
        for m in models
    ]
    axes[0].bar(
        range(len(models)),
        [_float(r or {}, "mean") for r in paa],
        yerr=[
            [max(0, _float(r or {}, "mean") - _float(r or {}, "ci95_low")) for r in paa],
            [max(0, _float(r or {}, "ci95_high") - _float(r or {}, "mean")) for r in paa],
        ],
        capsize=2,
        color=COLORS,
    )
    model_labels = (
        ["двуст.", "плавн.", "физ.", "случ.", "сим.", "перем."]
        if lang == "ru"
        else ["bist.", "graded", "phys.", "random", "sym.", "shuffle"]
    )
    axes[0].set_xticks(range(len(models)), model_labels, rotation=30, ha="right")
    axes[0].set_ylabel("NARMA-10 NRMSE")
    widths = sorted({_float(r, "response_width") for r in phase})
    taus = sorted({_float(r, "tau") for r in phase})
    grid = np.full((len(widths), len(taus)), np.nan)
    for r in phase:
        grid[widths.index(_float(r, "response_width")), taus.index(_float(r, "tau"))] = _float(
            r, "mean"
        )
    im = axes[1].imshow(grid, aspect="auto", origin="lower", cmap="viridis")
    axes[1].set_xticks(range(len(taus)), taus)
    axes[1].set_yticks(range(len(widths)), widths)
    axes[1].set(
        xlabel="Время релаксации" if lang == "ru" else "Relaxation time",
        ylabel="Ширина отклика" if lang == "ru" else "Response width",
    )
    fig.colorbar(im, ax=axes[1], label="NRMSE")
    fig.tight_layout()
    return _save(fig, root, "figure4_reservoir_regimes", lang)


def _figure5(data, root, lang, source_dir):
    rows = [
        r
        for r in _read(data / "multivalued_summary.csv")
        if r.get("experiment") == "main"
        and r.get("metric")
        in {"stable_state_count", "addressability_score", "transition_success_rate"}
    ]
    _write_source(source_dir / "figure5.csv", rows)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.2))
    metrics = ["stable_state_count", "addressability_score", "transition_success_rate"]
    labels = (
        ["Устойчивые состояния", "Адресуемость", "Надёжность перехода"]
        if lang == "ru"
        else ["Stable states", "Addressability", "Transition reliability"]
    )
    for ax, metric, label in zip(axes, metrics, labels):
        vals = [
            next((r for r in rows if r.get("system") == s and r.get("metric") == metric), {})
            for s in ("PAA", "NVPVPE")
        ]
        ax.bar(
            [0, 1],
            [_float(v, "mean", 0) for v in vals],
            yerr=[
                [max(0, _float(v, "mean", 0) - _float(v, "ci95_low", 0)) for v in vals],
                [max(0, _float(v, "ci95_high", 0) - _float(v, "mean", 0)) for v in vals],
            ],
            capsize=2,
            color=COLORS[:2],
        )
        ax.set_xticks([0, 1], ["PAA", "NVP–VPE"])
        ax.set_title(label, fontsize=9)
    fig.tight_layout()
    return _save(fig, root, "figure5_multivalued_storage", lang)


def _figure6(data, root, lang, source_dir):
    rows = [
        r
        for r in _read(data / "rbm_summary.csv")
        if r.get("metric") in {"valid_fraction", "mode_coverage"}
    ]
    _write_source(source_dir / "figure6.csv", rows)
    selected_names_3 = [
        "unconstrained",
        "cooperative",
        "local",
        "rank_2",
        "rank_4",
        "rank_6",
        "cooperative+local+rank_8",
    ]
    selected_names_4 = [
        "unconstrained",
        "cooperative",
        "local",
        "rank_4",
        "rank_8",
        "cooperative+local",
        "cooperative+local+rank_8",
    ]
    available_3 = {
        r.get("constraint")
        for r in rows
        if r.get("benchmark") == "bas_3x3" and r.get("metric") == "valid_fraction"
    }
    available_4 = {
        r.get("constraint")
        for r in rows
        if r.get("benchmark") == "bas_4x4" and r.get("metric") == "valid_fraction"
    }
    selected_names_3 = [name for name in selected_names_3 if name in available_3]
    selected_names_4 = [name for name in selected_names_4 if name in available_4]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.8))
    selected = [
        next(
            r
            for r in rows
            if r.get("benchmark") == "bas_3x3"
            and r.get("metric") == "valid_fraction"
            and r.get("constraint") == name
        )
        for name in selected_names_3
    ]
    axes[0].bar(
        range(len(selected)),
        [_float(r, "mean") for r in selected],
        yerr=[
            [max(0, _float(r, "mean") - _float(r, "ci95_low")) for r in selected],
            [max(0, _float(r, "ci95_high") - _float(r, "mean")) for r in selected],
        ],
        capsize=2,
        color="#3366A6",
    )
    label_map = {
        "unconstrained": "без огр." if lang == "ru" else "unrestr.",
        "cooperative": "неотр." if lang == "ru" else "nonneg.",
        "local": "локал." if lang == "ru" else "local",
        "rank_2": "ранг 2" if lang == "ru" else "rank 2",
        "rank_4": "ранг 4" if lang == "ru" else "rank 4",
        "rank_6": "ранг 6" if lang == "ru" else "rank 6",
        "rank_8": "ранг 8" if lang == "ru" else "rank 8",
        "cooperative+local": ("неотр.+локал." if lang == "ru" else "nonneg.+local"),
        "cooperative+local+rank_4": (
            "кооп.+локал.+ранг 4" if lang == "ru" else "coop.+local+rank 4"
        ),
        "cooperative+local+rank_8": ("неотр.+локал.+р8" if lang == "ru" else "nonneg.+local+r8"),
    }
    axes[0].set_xticks(
        range(len(selected)),
        [label_map.get(r["constraint"], r["constraint"]) for r in selected],
        rotation=25,
        ha="right",
    )
    axes[0].set_ylabel("Доля допустимых образцов" if lang == "ru" else "Valid sample fraction")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("A  Bars-and-Stripes 3 × 3", loc="left", fontsize=9)
    valid4 = [
        next(
            r
            for r in rows
            if r.get("benchmark") == "bas_4x4"
            and r.get("metric") == "valid_fraction"
            and r.get("constraint") == name
        )
        for name in selected_names_4
    ]
    coverage4 = [
        next(
            r
            for r in rows
            if r.get("benchmark") == "bas_4x4"
            and r.get("metric") == "mode_coverage"
            and r.get("constraint") == name
        )
        for name in selected_names_4
    ]
    x = np.arange(len(selected_names_4))
    axes[1].bar(
        x - 0.18,
        [_float(r, "mean") for r in valid4],
        0.36,
        yerr=[
            [max(0, _float(r, "mean") - _float(r, "ci95_low")) for r in valid4],
            [max(0, _float(r, "ci95_high") - _float(r, "mean")) for r in valid4],
        ],
        capsize=2,
        color="#3366A6",
        label="допустимые" if lang == "ru" else "valid fraction",
    )
    axes[1].bar(
        x + 0.18,
        [_float(r, "mean") for r in coverage4],
        0.36,
        yerr=[
            [max(0, _float(r, "mean") - _float(r, "ci95_low")) for r in coverage4],
            [max(0, _float(r, "ci95_high") - _float(r, "mean")) for r in coverage4],
        ],
        capsize=2,
        color="#D47A2A",
        label="покрытие мод" if lang == "ru" else "mode coverage",
    )
    axes[1].set_xticks(
        x,
        [label_map.get(name, name) for name in selected_names_4],
        rotation=25,
        ha="right",
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_title("B  Bars-and-Stripes 4 × 4", loc="left", fontsize=9)
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return _save(fig, root, "figure6_rbm_constraints", lang)


def _figure7(data, root, lang, source_dir):
    rows = _read(data / "routes_summary.csv")
    offline = [
        r
        for r in rows
        if r.get("route") == "offline_fabrication"
        and r.get("setting") == "factorial"
        and r.get("metric") == "valid_fraction"
    ]
    inmat_rows = [
        r for r in rows if r.get("route") == "in_material" and r.get("metric") == "valid_fraction"
    ]
    inmat = [
        next(r for r in inmat_rows if r.get("setting") == setting)
        for setting in ("zero_mean_noise", "biased_update", "state_dependent")
    ]
    _write_source(source_dir / "figure7.csv", offline + inmat)
    errors = sorted({_float(r, "error") for r in offline})
    levels = sorted({_float(r, "levels") for r in offline})
    grid = np.full((len(levels), len(errors)), np.nan)
    for r in offline:
        grid[levels.index(_float(r, "levels")), errors.index(_float(r, "error"))] = _float(
            r, "mean"
        )
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.7))
    im = axes[0].imshow(grid, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axes[0].set_xticks(range(len(errors)), [f"{100 * x:.0f}" for x in errors])
    axes[0].set_yticks(range(len(levels)), [f"{x:.0f}" for x in levels])
    axes[0].set(
        xlabel="Ошибка, %" if lang == "ru" else "Fabrication error, %",
        ylabel="Уровни" if lang == "ru" else "Quantization levels",
    )
    fig.colorbar(
        im,
        ax=axes[0],
        label="доля допустимых" if lang == "ru" else "valid fraction",
    )
    setting_labels = {
        "zero_mean_noise": "шум с нулевым средним" if lang == "ru" else "zero-mean noise",
        "biased_update": "смещённое обновление" if lang == "ru" else "biased update",
        "state_dependent": "зависимость от состояния" if lang == "ru" else "state-dependent",
    }
    axes[1].bar(
        range(len(inmat)),
        [_float(r, "mean") for r in inmat],
        yerr=[
            [max(0, _float(r, "mean") - _float(r, "ci95_low")) for r in inmat],
            [max(0, _float(r, "ci95_high") - _float(r, "mean")) for r in inmat],
        ],
        capsize=3,
        color=COLORS[: len(inmat)],
    )
    axes[1].set_xticks(
        range(len(inmat)),
        [setting_labels.get(r.get("setting", ""), r.get("setting", "")) for r in inmat],
        rotation=25,
        ha="right",
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Доля допустимых" if lang == "ru" else "Valid fraction")
    fig.tight_layout()
    return _save(fig, root, "figure7_implementation_routes", lang)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ORIGINAL)
    p.add_argument("--out", type=Path, default=WORK / "figures")
    p.add_argument("--language", choices=("en", "ru", "both"), default="en")
    args = p.parse_args(argv)
    args.out = output_directory(args.out)
    source_dir = args.out / "source_tables"
    builders = [
        lambda lang: _figure1(args.out, lang),
        lambda lang: _figure2(args.data, args.out, lang, source_dir),
        lambda lang: _figure3(args.data, args.out, lang, source_dir),
        lambda lang: _figure4(args.data, args.out, lang, source_dir),
        lambda lang: _figure5(args.data, args.out, lang, source_dir),
        lambda lang: _figure6(args.data, args.out, lang, source_dir),
        lambda lang: _figure7(args.data, args.out, lang, source_dir),
    ]
    paths = []
    for lang in ("en", "ru") if args.language == "both" else (args.language,):
        for builder in builders:
            paths.extend(builder(lang))
    source_files = sorted(source_dir.glob("*.csv"))
    source_manifest = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in source_files
    ]
    manifest = []
    for path in paths:
        manifest.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_tables": source_manifest if "figure1_" not in path.name else [],
            }
        )
    (args.out / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved {len(paths)} figure files under {args.out}")


if __name__ == "__main__":
    main()
