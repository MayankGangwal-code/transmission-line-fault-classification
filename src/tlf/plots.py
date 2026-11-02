"""Figures for the report (matplotlib, PNG)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from . import CLASSES  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
PHASE_COLOURS = {"a": "#e34948", "b": "#eda100", "c": "#2a78d6"}  # the usual R-Y-B colour code
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GROUP_COLOURS = {"no fault": "#898781", "LG": "#2a78d6", "LL": "#eb6834", "LLG": "#1baf7a", "LLL": "#6250d6"}
BLUES = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#5598e7", "#1c5cab", "#0d366b"])

plt.rcParams.update(
    {
        "font.family": "sans-serif", "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titleweight": "bold", "axes.titlesize": 12, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 160, "savefig.bbox": "tight",
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    }
)


def _save(fig, path):
    if path is None:
        return fig
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_waveforms(rec, title: str, path=None, trigger: int | None = None, decision_end: int | None = None):
    """Voltages and currents of one event with the fault instant, trigger and decision window marked."""
    t_ms = rec.t * 1e3
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 6), sharex=True)
    for k, p in enumerate("abc"):
        ax1.plot(t_ms, rec.v[:, k] / 1e3, color=PHASE_COLOURS[p], lw=1.1, label=f"V{p}")
        ax2.plot(t_ms, rec.i[:, k], color=PHASE_COLOURS[p], lw=1.1, label=f"I{p}")
    for ax in (ax1, ax2):
        ax.axvline(t_ms[rec.fault_index], color=INK, lw=1.0, ls="--")
        if trigger is not None:
            ax.axvline(t_ms[trigger], color="#008300", lw=1.0, ls=":")
        if trigger is not None and decision_end is not None:
            n = rec.system.samples_per_cycle
            ax.axvspan(t_ms[max(decision_end - n, 0)], t_ms[min(decision_end, len(t_ms) - 1)], color="#2a78d6", alpha=0.08, lw=0)
    ax1.set_ylabel("bus voltage (kV)")
    ax2.set_ylabel("line current (A)")
    ax2.set_xlabel("time (ms)")
    ax1.set_title(title, loc="left", pad=10)
    ax1.text(t_ms[rec.fault_index] + 1, ax1.get_ylim()[1] * 0.92, "fault", fontsize=8, color=INK)
    if trigger is not None:
        ax2.text(t_ms[trigger] + 1, ax2.get_ylim()[1] * 0.92, "relay trigger", fontsize=8, color="#008300")
        if decision_end is not None:
            ax2.text(t_ms[max(decision_end - rec.system.samples_per_cycle, 0)] + 1, ax2.get_ylim()[0] * 0.92, "decision window (1 cycle)", fontsize=8, color="#2a78d6")
    ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, borderaxespad=0)
    ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, borderaxespad=0)
    fig.tight_layout()
    return _save(fig, path)


def plot_sequence_angle_map(table: pd.DataFrame, labels, path=None, title="The relay's phase-selection map: angles between sequence currents"):
    """Scatter of ∠(I2/I0) vs ∠(I2/ΔI1) coloured by true class — the sequence rule's decision space."""
    labels = np.asarray(labels).astype(str)
    a20 = np.degrees(np.arctan2(table["sin_a20"], table["cos_a20"]))
    a21 = np.degrees(np.arctan2(table["sin_a21"], table["cos_a21"]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))
    ground = [c for c in CLASSES if c.endswith("G")]
    nog = ["AB", "BC", "CA", "ABC"]
    markers = {"AG": "o", "BG": "s", "CG": "^", "ABG": "o", "BCG": "s", "CAG": "^", "AB": "o", "BC": "s", "CA": "^", "ABC": "D"}
    colours = {"AG": "#2a78d6", "BG": "#eda100", "CG": "#e34948", "ABG": "#1baf7a", "BCG": "#6250d6", "CAG": "#e87ba4",
               "AB": "#1baf7a", "BC": "#6250d6", "CA": "#e87ba4", "ABC": "#898781"}
    for c in ground:
        m = labels == c
        ax1.scatter(a20[m], a21[m], s=14, marker=markers[c], color=colours[c], alpha=0.7, label=f"{c} (n={m.sum()})", edgecolor="none")
    for x in (-120, 0, 120):
        ax1.axvline(x, color=AXIS, lw=0.8)
    for y in (-120, -60, 0, 60, 120, 180):
        ax1.axhline(y, color=AXIS, lw=0.8)
    ax1.set_xlabel("∠(I₂ / I₀)  (degrees)")
    ax1.set_ylabel("∠(I₂ / ΔI₁)  (degrees)")
    ax1.set_title("Ground faults", loc="left", pad=10)
    ax1.legend(fontsize=8, loc="lower left", ncol=2)
    ax1.set_xlim(-180, 180)
    ax1.set_ylim(-180, 180)
    for c in nog:
        m = labels == c
        ax2.scatter(table.loc[m, "r_i2_di1"], a21[m], s=14, marker=markers[c], color=colours[c], alpha=0.7, label=f"{c} (n={m.sum()})", edgecolor="none")
    for y in (-60, 60, 180):
        ax2.axhline(y, color=AXIS, lw=0.8)
    from .classical import BALANCED_I2

    ax2.axvline(BALANCED_I2, color=INK, lw=1.0, ls="--")
    ax2.text(BALANCED_I2 * 1.08, 185, f"balanced-event threshold {BALANCED_I2:.2f}", fontsize=8, color=INK, va="top")
    ax2.set_xscale("log")
    ax2.set_xlabel("|I₂| / |ΔI₁|   (log scale)")
    ax2.set_ylabel("∠(I₂ / ΔI₁)  (degrees)")
    ax2.set_title("Phase-to-phase and three-phase faults", loc="left", pad=10)
    ax2.legend(fontsize=8, loc="lower left")
    ax2.set_ylim(-200, 200)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, path)


def plot_confusion(cm: pd.DataFrame, path=None, title="Confusion matrix"):
    fig, ax = plt.subplots(figsize=(0.62 * cm.shape[1] + 2.0, 0.62 * cm.shape[0] + 1.6))
    vals = cm.to_numpy()
    norm = vals / np.maximum(vals.sum(axis=1, keepdims=True), 1)
    ax.imshow(norm, cmap=BLUES, vmin=0, vmax=1, aspect="equal")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if vals[i, j]:
                ax.text(j, i, str(vals[i, j]), ha="center", va="center", fontsize=8.5, color="#ffffff" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(cm.shape[1]), cm.columns, rotation=45, ha="right")
    ax.set_yticks(range(cm.shape[0]), cm.index)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", pad=10)
    fig.tight_layout()
    return _save(fig, path)


def plot_bars(values: pd.Series, path=None, title="", xlabel="accuracy", fmt="{:.1%}", xmax=1.0, highlight=None):
    s = values.sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(s) + 1.4))
    colours = [SERIES[0] if (highlight is None or n == highlight) else "#9ec5f4" for n in s.index]
    bars = ax.barh(s.index, s.values, color=colours, height=0.6)
    for b, v in zip(bars, s.values):
        ax.text(v + xmax * 0.012, b.get_y() + b.get_height() / 2, fmt.format(v), va="center", ha="left", fontsize=9, color=INK_2)
    ax.set_xlim(0, xmax * 1.15)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.yaxis.grid(False)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", pad=10)
    fig.tight_layout()
    return _save(fig, path)


def plot_lines(table: pd.DataFrame, path=None, title="", xlabel="", ylabel="accuracy", ylim=(0.5, 1.0), xlog=False, fmt_pct=True):
    """One line per column of ``table`` (index = x values)."""
    fig, ax = plt.subplots(figsize=(8, 4.3))
    for k, col in enumerate(table.columns):
        ax.plot(table.index, table[col], marker="o", ms=4, lw=1.5, color=SERIES[k % len(SERIES)], label=col)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    if xlog:
        ax.set_xscale("log")
    if fmt_pct:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(fontsize=8.5)
    ax.set_title(title, loc="left", pad=10)
    fig.tight_layout()
    return _save(fig, path)


def plot_location_errors(df: pd.DataFrame, methods: list[str], path=None, title="Fault-location error versus fault resistance"):
    """df: columns 'rf' (ohm) and '<method>_err' (km).  Binned medians with 90th-percentile whiskers."""
    bins = [0, 0.5, 2, 5, 10, 20, 60]
    labels = ["<0.5", "0.5–2", "2–5", "5–10", "10–20", "20–60"]
    cat = pd.cut(df["rf"], bins=bins, labels=labels, include_lowest=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(labels))
    w = 0.8 / len(methods)
    for k, m in enumerate(methods):
        g = df.groupby(cat, observed=False)[f"{m}_err"]
        med, p90 = g.median().reindex(labels), g.quantile(0.9).reindex(labels)
        xs = x + (k - (len(methods) - 1) / 2) * (w + 0.02)
        ax.bar(xs, med.values, width=w, color=SERIES[k], label=m)
        ax.vlines(xs, med.values, p90.values, color=SERIES[k], lw=1.2)
        for xi, v in zip(xs, med.values):
            if np.isfinite(v):
                ax.text(xi, v + 0.25, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK_2,
                        bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.6))
    ax.set_xticks(x, labels)
    ax.set_xlabel("total fault resistance (Ω)")
    ax.set_ylabel("|error| (km): median bar, line to 90th percentile")
    ax.legend(fontsize=8.5, labels=["Reactance method", "Takagi method", "ML regressor"])
    ax.set_title(title, loc="left", pad=10)
    fig.tight_layout()
    return _save(fig, path)


def plot_feature_importance(imp: pd.Series, path=None, title="Feature importance", top=15, xlabel="importance (mean decrease in impurity)"):
    s = imp.sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(s) + 1.4))
    bars = ax.barh(s.index, s.values, color=SERIES[0], height=0.6)
    for b, v in zip(bars, s.values):
        ax.text(v + s.max() * 0.015, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", ha="left", fontsize=8.5, color=INK_2)
    ax.set_xlim(0, max(s.max(), 1e-3) * 1.18)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.yaxis.grid(False)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", pad=10)
    fig.tight_layout()
    return _save(fig, path)


def plot_phasors(V: np.ndarray, I: np.ndarray, path=None, title="Phasors in the decision window"):
    """Polar phasor diagram of the three voltages and currents (currents scaled to the voltage circle)."""
    fig, ax = plt.subplots(figsize=(5.2, 5.2), subplot_kw={"projection": "polar"})
    vmax = max(np.abs(V).max(), 1e-9)
    imax = max(np.abs(I).max(), 1e-9)
    for k, p in enumerate("abc"):
        ax.annotate("", xy=(np.angle(V[k]), np.abs(V[k]) / vmax), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=PHASE_COLOURS[p], lw=2))
        ax.annotate("", xy=(np.angle(I[k]), np.abs(I[k]) / imax), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=PHASE_COLOURS[p], lw=1.2, ls="--"))
        ax.text(np.angle(V[k]), np.abs(V[k]) / vmax + 0.08, f"V{p}", color=PHASE_COLOURS[p], fontsize=9, ha="center")
        ax.text(np.angle(I[k]), np.abs(I[k]) / imax + 0.08, f"I{p}", color=PHASE_COLOURS[p], fontsize=8, ha="center", alpha=0.8)
    ax.set_rticks([])
    ax.set_title(title + "\n(solid = voltage, dashed = current, each to its own scale)", fontsize=9, pad=12)
    fig.tight_layout()
    return _save(fig, path)
