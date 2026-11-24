"""Matplotlib figures for the noise survey and frequency checks."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from .noise import NoiseSpectrum

# Categorical slots, always assigned in this order (validated palette, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#9b9a95"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "dmz_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titlelocation": "left",
    "axes.labelsize": 10,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
    "font.size": 10,
}

_PREFIX = [(1e-15, "f"), (1e-12, "p"), (1e-9, "n"), (1e-6, "µ"), (1e-3, "m"), (1.0, "")]


def eng(x: float, unit: str = "", digits: int = 3) -> str:
    """Engineering-notation string: eng(3.2e-8, 'V') -> '32 nV'."""
    if not np.isfinite(x) or x == 0:
        return f"{x:g} {unit}".strip()
    for scale, p in reversed(_PREFIX):
        if abs(x) >= scale:
            return f"{x / scale:.{digits}g} {p}{unit}".strip()
    return f"{x / 1e-15:.{digits}g} f{unit}".strip()


def _spectra_and_lines(obj) -> tuple:
    if hasattr(obj, "spectra"):
        return obj.spectra, getattr(obj, "lines", {}) or {}
    if isinstance(obj, NoiseSpectrum):
        return {obj.name or "ch": obj}, {}
    return dict(obj), {}


def plot_survey(survey: Any, fmin: Optional[float] = None, fmax: Optional[float] = None,
                n_labels: int = 6, figsize: Optional[tuple] = None):
    """One panel per channel: input noise density vs frequency.

    Dashed: mains harmonics. Orange dots: strongest lines.
    `survey` is a Survey, a {name: NoiseSpectrum} dict or one NoiseSpectrum.
    """
    spectra, lines = _spectra_and_lines(survey)
    mains = getattr(survey, "mains", None) or next(iter(spectra.values())).meta.get("mains", 60.0)
    lo = fmin if fmin is not None else min(s.f[0] for s in spectra.values())
    hi = fmax if fmax is not None else max(s.f[-1] for s in spectra.values())
    n = len(spectra)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(n, 1, sharex=True, figsize=figsize or (10, 2.6 * n + 0.6),
                                 squeeze=False)
        for ax, (name, s) in zip(axes[:, 0], spectra.items()):
            m = (s.f >= lo) & (s.f <= hi)
            for k in range(1, int(hi // mains) + 1):
                if k * mains >= lo:
                    ax.axvline(k * mains, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
            ax.semilogy(s.f[m], s.asd[m], color=SERIES[0], lw=0.9, zorder=2)
            ax.set_title(f"{name}   floor ≈ {eng(s.floor(max(lo, 1.0), hi), s.unit)}/√Hz", pad=4)
            ax.set_ylabel(f"{s.unit}/√Hz")
            shown = [ln for ln in (lines.get(name) or []) if lo <= ln["f"] <= hi][:n_labels]
            for ln in shown:
                ax.plot(ln["f"], ln["peak_asd"], "o", ms=4, color=SERIES[1], zorder=3,
                        mec=SURFACE, mew=0.8)
                lbl = f"{ln['f']:.2f}" + (f" ({ln['tag']})" if ln["tag"] else "")
                ax.annotate(lbl, (ln["f"], ln["peak_asd"]), xytext=(3, 3),
                            textcoords="offset points", fontsize=8, color=INK_2)
        axes[0, 0].text(1.0, 1.02, f"dashed: {mains:g} Hz harmonics   dots: strongest lines",
                        transform=axes[0, 0].transAxes, ha="right", va="bottom", fontsize=8,
                        color=INK_2)
        axes[-1, 0].set_xlabel("input frequency (Hz)")
        axes[-1, 0].set_xlim(lo, hi)
        fig.tight_layout()
    return fig


def plot_ranking(rank: Any, figsize: Optional[tuple] = None):
    """Predicted noise of one reading vs reference frequency, one panel per channel.

    Line: the statistic used for ranking. Band: 10-90 % of the time blocks (wide band =
    noise that comes and goes). Grey: mains-related, skipped. Triangles: the picks.
    """
    names = list(rank.noise)
    n = len(names)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(n, 1, sharex=True, figsize=figsize or (10, 2.4 * n + 0.6),
                                 squeeze=False)
        for i, (ax, name) in enumerate(zip(axes[:, 0], names)):
            st = rank.noise[name]
            col = SERIES[i % len(SERIES)]
            _shade(ax, rank.f, rank.excluded)
            if rank.block:
                ax.fill_between(rank.f, st["p10"], st["p90"], color=col, alpha=0.22, lw=0, zorder=1)
                if rank.stat != "median":
                    ax.plot(rank.f, st["median"], color=col, lw=0.8, alpha=0.7, zorder=2)
            ax.semilogy(rank.f, st[rank.stat], color=col, lw=1.5, zorder=2)
            tr = ax.get_xaxis_transform()
            for j, b in enumerate(rank.best, start=1):
                ax.plot(b["f"], b[name], "^", ms=7, color=INK, mec=SURFACE, mew=0.8, zorder=4)
                ax.annotate(str(j), (b["f"], b[name]), xytext=(0, -13), textcoords="offset points",
                            ha="center", fontsize=8, color=INK)
            ax.set_title(name, pad=4)
            ax.set_ylabel(rank.units[name] + " rms")
        how = (f"thick: {rank.stat}   thin: median   band: 10-90 % of {rank.block:g} s blocks"
               if rank.block else f"line: {rank.stat}")
        axes[0, 0].text(1.0, 1.02, f"{rank.filter.describe()}   {how}   ▲ picks   grey: mains-related",
                        transform=axes[0, 0].transAxes, ha="right", va="bottom", fontsize=8,
                        color=INK_2)
        axes[-1, 0].set_xlabel("reference frequency (Hz)")
        axes[-1, 0].set_xlim(rank.f[0], rank.f[-1])
        fig.tight_layout()
    return fig


def _shade(ax, f, mask, label=None):
    if not np.any(mask):
        return
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts, stops = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
    for k, (a, b) in enumerate(zip(starts, stops)):
        ax.axvspan(f[a], f[b - 1], color=GRID, lw=0, zorder=0, label=label if k == 0 else None)


def plot_spectrogram(t: np.ndarray, f: np.ndarray, psd: np.ndarray, name: str = "",
                     unit: str = "V", fmin: float = 0.0, fmax: Optional[float] = None,
                     figsize=(10, 4)):
    """Time-resolved noise density: look for bursts (compressor cycles, pumps, ...)."""
    m = (f >= fmin) & (f <= (fmax or f[-1]))
    asd = np.sqrt(psd[m])
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=figsize)
        pos = asd[asd > 0]
        norm = LogNorm(vmin=np.percentile(pos, 5), vmax=np.percentile(pos, 99.9))
        pc = ax.pcolormesh(t, f[m], asd, cmap=SEQ_BLUE, norm=norm, shading="auto")
        cb = fig.colorbar(pc, ax=ax, pad=0.01)
        cb.set_label(f"{unit}/√Hz")
        cb.outline.set_visible(False)
        ax.grid(False)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("input frequency (Hz)")
        ax.set_title(f"{name} spectrogram", pad=4)
        fig.tight_layout()
    return fig


def plot_check(check: Any, figsize: Optional[tuple] = None):
    """Allan deviation of X per channel (one panel each), one line per candidate frequency.

    Falling like 1/sqrt(time) = averaging still helps; flattening or rising = drift.
    """
    freqs = list(check.allan)
    names = list(check.allan[freqs[0]]) if freqs else []
    n = len(names)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, n, figsize=figsize or (4.2 * n + 0.5, 3.6), squeeze=False)
        for ax, name in zip(axes[0], names):
            for i, f in enumerate(freqs):
                t, a = check.allan[f][name]
                ax.loglog(t, a, color=SERIES[i % len(SERIES)], lw=1.5, label=f"{f:.2f} Hz")
            ax.set_title(name, pad=4)
            ax.set_xlabel("averaging time (s)")
        axes[0, 0].set_ylabel("Allan deviation of X (input units)")
        axes[0, -1].legend(loc="best")
        fig.tight_layout()
    return fig


def plot_stream(d, gain: float = 1.0, unit: str = "V", figsize=(10, 3)):
    """Quick look at a streamed X(t), Y(t) record."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(d.t, d.x / gain, color=SERIES[0], lw=0.8, label="X")
        ax.plot(d.t, d.y / gain, color=SERIES[1], lw=0.8, label="Y")
        ax.set_xlabel("time (s)")
        ax.set_ylabel(unit)
        ax.set_title(d.name, pad=4)
        ax.legend(loc="upper right", ncol=2)
        fig.tight_layout()
    return fig
