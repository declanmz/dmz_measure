"""Streamed sweeps: the engine behind Station.stream_sweep / stream_megasweep, re-averaging
saved runs (refilter) and comparing runs.

How a streamed point is made: the lock-ins run with a short time constant and the sync
filter and stream continuously. For each point the gate is set; the host clock time when
the set call returns is recorded; after `settle` seconds the next `dwell` seconds of
samples are averaged (Hann-weighted or plain). Samples are tied to host time through the
arrival time of the first packet (the stream itself has no timestamps), good to ~1 ms.

The raw stream of every line is kept (decimated to >= 16 f_ref) in the run's
``stream.npz`` (a sweep) or ``stream_00000.npz``, ``stream_00001.npz``, ... (a map: one file
per line, written when the line ends), so :func:`refilter` can redo the averaging with other
settings, also while a map is still running.
"""
from __future__ import annotations

import glob
import io as _io
import os
import time
import zipfile
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy import signal

from . import lockin as lk
from . import noise as nz
from .instruments.sr860 import LockinIO, apply_settings, preserved_settings
from .instruments.sr860_stream import (DEFAULT_PORT, SR860Stream, StreamTimeout, decode_packet)
from .io import get_basedir, load_run

COMPONENTS = ("X", "Y", "R", "P")


# --------------------------------------------------------------------------
# live streams with host-time lookup
# --------------------------------------------------------------------------

class _Buffer:
    """Incrementally decodes one SR860Stream's packets into a growing complex array."""

    def __init__(self, stream: SR860Stream):
        self.st = stream
        self.i = 0
        self.prev = None
        self.z = np.empty(1 << 16, np.complex64)
        self.n = 0
        self.fs = None
        self.t_first = None

    def _append(self, a: np.ndarray) -> None:
        if self.n + len(a) > len(self.z):
            grown = np.empty(max(2 * len(self.z), self.n + len(a)), np.complex64)
            grown[: self.n] = self.z[: self.n]
            self.z = grown
        self.z[self.n:self.n + len(a)] = a
        self.n += len(a)

    def update(self) -> None:
        packets = self.st._packets
        k = len(packets)
        for buf in packets[self.i:k]:
            hdr, vals = decode_packet(buf)
            if self.fs is None:
                self.fs = hdr.sample_rate
                self.t_first = self.st._t_first - hdr.samples_per_packet / self.fs
            if self.prev is not None:
                gap = (hdr.counter - self.prev - 1) % 256
                if gap:
                    self._append(np.full(gap * hdr.samples_per_packet, np.nan, np.complex64))
            self.prev = hdr.counter
            self._append(vals[:, 0] + 1j * vals[:, 1])
        self.i = k

    @property
    def t_end(self) -> float:
        return -np.inf if self.fs is None else self.t_first + self.n / self.fs

    def window(self, t_a: float, t_b: float) -> np.ndarray:
        ja = max(int(np.ceil((t_a - self.t_first) * self.fs)), 0)
        jb = min(int(np.floor((t_b - self.t_first) * self.fs)), self.n)
        return self.z[ja:jb]


def weighted_mean(seg: np.ndarray, window: str) -> complex:
    """Average of a stretch of samples: 'hann'-weighted or plain ('boxcar'); NaNs skipped."""
    if len(seg) < 2:
        return complex(np.nan, np.nan)
    w = signal.get_window("hann", len(seg)) if window == "hann" else np.ones(len(seg))
    ok = np.isfinite(seg)
    if ok.sum() < 2:
        return complex(np.nan, np.nan)
    return complex(np.sum(w[ok] * seg[ok]) / np.sum(w[ok]))


class LiveStreams:
    """Stream several SR860s at once and average them over host-time windows while running."""

    def __init__(self, ios: Dict[str, LockinIO], rate: float):
        self.ios = ios
        self.streams = {n: SR860Stream(io, port=DEFAULT_PORT + i) for i, (n, io) in enumerate(ios.items())}
        for st in self.streams.values():
            st.configure(rate=rate)
        self.buffers: Dict[str, _Buffer] = {}

    def start(self, lead: float = 0.25) -> None:
        for st in self.streams.values():
            st.start()
        try:
            for st in self.streams.values():
                st.wait_first_packet()
        except BaseException:
            self.stop()
            raise
        self.buffers = {n: _Buffer(st) for n, st in self.streams.items()}
        time.sleep(lead)

    def average(self, t_a: float, t_b: float, window: str = "hann",
                timeout: float = 3.0) -> Dict[str, complex]:
        """Average of each lock-in over host times [t_a, t_b] (waits for the packets to arrive)."""
        out = {}
        for n, b in self.buffers.items():
            t_end = time.monotonic() + timeout
            b.update()
            while b.t_end < t_b and time.monotonic() < t_end:
                time.sleep(0.003)
                b.update()
            out[n] = weighted_mean(b.window(t_a, t_b), window)
        return out

    def stop(self) -> Dict[str, Dict[str, Any]]:
        """Stop streaming; return each lock-in's full record (z, fs, t_first, flags)."""
        for st in self.streams.values():
            try:
                st.io.write("STREAM OFF")
            except Exception:
                pass
        raw = {}
        for n, st in self.streams.items():
            d = st.stop(allow_empty=True, send_off=False)
            b = self.buffers.get(n)
            if b is not None:
                b.update()
            if d is None or b is None or b.fs is None:
                raw[n] = None
                continue
            raw[n] = {"z": b.z[: b.n].copy(), "fs": b.fs, "t_first": b.t_first,
                      "dropped": d.n_dropped, "overload": d.n_overload, "error": d.n_error}
        return raw


class LockinSetup:
    """Context manager: short tau + sync on the streamed lock-ins, restored on exit."""

    def __init__(self, ios: Dict[str, LockinIO], tau: Optional[float] = None, slope: int = 24,
                 advanced: Optional[bool] = None, sync: bool = True, rate: Optional[float] = None):
        self.ios, self._tau, self.slope, self._adv, self.sync, self._rate = ios, tau, slope, advanced, sync, rate

    def __enter__(self) -> "LockinSetup":
        ios = list(self.ios.values())
        self._ctx = preserved_settings(ios, verbose=False)
        self._ctx.__enter__()
        try:
            self.f = ios[0].detection_frequency()
            for io in ios:
                apply_settings(io, tau=self._tau if self._tau is not None else lk.recommended_tau(self.f),
                               slope_db=self.slope, sync=self.sync, advanced=self._adv)
            self.tau = ios[0].time_constant()
            self.advanced = bool(ios[0].ask_int("ADVFILT?"))
            self.settle = lk.settling_time(self.tau, self.f, self.slope, self.advanced, self.sync)
            self.rate = self._rate or lk.recommended_stream_rate(self.tau)
        except BaseException:
            self._ctx.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc) -> None:
        self._ctx.__exit__(*exc)

    def describe(self, settle: float, dwell: float, window: str) -> Dict[str, Any]:
        return {"f_ref": self.f, "tau": self.tau, "slope": self.slope, "advanced": self.advanced,
                "sync": self.sync, "rate": self.rate, "settle": settle, "dwell": dwell,
                "window": window}


def segment_record(t_set0: Sequence[float], t_set1: Sequence[float], rows: Sequence[int],
                   raw: Dict[str, Optional[Dict[str, Any]]], f_ref: float) -> Dict[str, Any]:
    """What is kept of one line for refiltering: step times and the decimated streams."""
    seg = {"t_set0": np.asarray(t_set0), "t_set1": np.asarray(t_set1), "rows": np.asarray(rows),
           "streams": {}}
    for n, r in raw.items():
        if r is None:
            continue
        q = int(2 ** max(0, np.floor(np.log2(r["fs"] / (16 * f_ref)))))
        seg["streams"][n] = {"z": nz.decimate(r["z"], q).astype(np.complex64), "fs": r["fs"] / q,
                             "t_first": r["t_first"]}
    return seg


def segments_to_npz(segments: List[Dict[str, Any]]) -> bytes:
    arrays = {"n_segments": np.array(len(segments))}
    for k, s in enumerate(segments):
        for key in ("t_set0", "t_set1", "rows"):
            arrays[f"{k}_{key}"] = s[key]
        for n, st in s["streams"].items():
            arrays[f"{k}_{n}_z"] = st["z"]
            arrays[f"{k}_{n}_fs"] = np.array(st["fs"])
            arrays[f"{k}_{n}_t_first"] = np.array(st["t_first"])
    b = _io.BytesIO()
    np.savez(b, **arrays)
    return b.getvalue()


def _npz_to_segments(path: str, names: Sequence[str]) -> List[Dict[str, Any]]:
    with np.load(path) as a:
        segs = []
        for k in range(int(a["n_segments"])):
            s = {key: a[f"{k}_{key}"] for key in ("t_set0", "t_set1", "rows")}
            s["streams"] = {n: {"z": a[f"{k}_{n}_z"], "fs": float(a[f"{k}_{n}_fs"]),
                                "t_first": float(a[f"{k}_{n}_t_first"])}
                            for n in names if f"{k}_{n}_z" in a.files}
            segs.append(s)
    return segs


def load_segments(run_dir: str, names: Sequence[str]) -> List[Dict[str, Any]]:
    """The saved stream of a run, one segment per line: ``stream.npz`` (sweeps, and maps saved
    by dmz_measure 0.7.0), or the per-line files ``stream_00000.npz``, ... of a map. A line
    file that can't be read yet (still being written or synced) is skipped."""
    one = os.path.join(run_dir, "stream.npz")
    if os.path.exists(one):
        return _npz_to_segments(one, names)
    segs: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(run_dir, "stream_[0-9]*.npz"))):
        try:
            segs.extend(_npz_to_segments(path, names))
        except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile):
            continue
    return segs


# --------------------------------------------------------------------------
# re-averaging saved runs
# --------------------------------------------------------------------------

def component(z: complex, comp: str) -> float:
    if comp == "X":
        return z.real
    if comp == "Y":
        return z.imag
    if comp == "R":
        return abs(z)
    return float(np.degrees(np.angle(z)))  # 'P'


def refilter(run_id: int, dwell: Optional[float] = None, settle: Optional[float] = None,
             window: Optional[str] = None, basedir: Optional[str] = None) -> Dict[str, Any]:
    """Redo the per-point averaging of a streamed run with other settle / dwell / window.

    Returns the data like dm.pload: {'xs': ..., ('ys': ...), column: array},
    2D runs reshaped to (slow, fast). Nothing is re-measured and the saved run is unchanged.
    """
    run = load_run(run_id, basedir)
    md = run.metadata
    if "stream" not in md:
        raise ValueError(f"run {run_id} is not a streamed run")
    st = md["stream"]
    settle = st["settle"] if settle is None else settle
    dwell = st["dwell"] if dwell is None else dwell
    window = st["window"] if window is None else window
    cols = {c: v for c, v in st["columns"].items()}
    names = sorted({v[0] for v in cols.values()})
    table = {k: v.copy() for k, v in run.table.items()}
    for seg in load_segments(run.dir, names):
        a = seg["t_set1"] + settle
        b = np.minimum(a + dwell, np.append(seg["t_set0"][1:], np.inf))
        for i, row in enumerate(seg["rows"]):
            vals = {}
            for n, s in seg["streams"].items():
                zz = s["z"]
                ja = max(int(np.ceil((a[i] - s["t_first"]) * s["fs"])), 0)
                jb = min(int(np.floor((b[i] - s["t_first"]) * s["fs"])), len(zz))
                vals[n] = weighted_mean(zz[ja:jb], window)
            for c, (n, comp, gain) in cols.items():
                if n in vals and c in table and row < len(table[c]):   # (a running map's rows may lag)
                    table[c][row] = component(vals[n], comp) / gain
    out: Dict[str, Any] = {"measurement_config": md.get("measurement_config"),
                           "settings": dict(st, settle=settle, dwell=dwell, window=window)}
    if md["type"] == "2D":
        xs, ys = np.array(md["fast_setpoints"]), np.array(md["slow_setpoints"])
        out.update(xs=xs, ys=ys)
        for c in md.get("columns", []):                  # points not measured (yet) are NaN
            v = np.asarray(table.get(c, np.empty(0)), dtype=float)[: len(xs) * len(ys)]
            v = np.pad(v, (0, len(xs) * len(ys) - len(v)), constant_values=np.nan)
            out[c] = v.reshape(len(ys), len(xs))
    else:
        out["xs"] = np.array(md["setpoints"])
        out.update(table)
    return out


# --------------------------------------------------------------------------
# comparing runs
# --------------------------------------------------------------------------

def point_noise(y: np.ndarray) -> float:
    """Point-to-point noise of a 1D curve, estimated from the curve itself.

    Residuals from a local 7th-order fit over 11 points (Savitzky-Golay) remove real
    curvature; their white-noise variance is a known fraction of sigma^2, divided out. The
    median absolute deviation keeps sharp features from inflating it. Lag in a stepped
    sweep correlates neighbouring points, which hides part of its noise from this estimate.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    win, order = 11, 7
    if len(y) < win:
        return np.nan
    h = signal.savgol_coeffs(win, order)
    c = 1 - 2 * h[win // 2] + np.sum(h ** 2)
    r = y - signal.savgol_filter(y, win, order)
    return float(1.4826 * np.median(np.abs(r - np.median(r))) / np.sqrt(c))


def _run_curves(run_id, current, basedir):
    run = load_run(run_id, basedir)
    md, t = run.metadata, run.table
    if md.get("type") != "1D":
        raise ValueError(f"run {run_id} is not a 1D sweep")
    x = t[md["param"]]
    names = [c[:-2] for c in t if c.endswith("_X")]

    def z(n):
        return t[f"{n}_X"] + 1j * t.get(f"{n}_Y", np.zeros_like(x))

    curves = {}
    for n in names:
        if current is not None and n != current and current in names:
            curves[n] = (np.real(z(n) / z(current)), "R", "Ω")
        elif current is not None and n == current:
            curves[n] = (np.real(z(n)), "I", "A")
        else:
            curves[n] = (np.real(z(n)), "X", "V")
    dur = md.get("end_time", np.nan) - md.get("start_time", np.nan)
    arrow = "↑" if x[-1] >= x[0] else "↓"
    label = f"{run_id}: {md.get('function', '?')} {arrow}"
    return x, curves, dur / max(len(x), 1), label, md["param"]


def compare(run_ids: Sequence[int], current: Optional[str] = None,
            channels: Optional[Sequence[str]] = None, basedir: Optional[str] = None,
            labels: Optional[Sequence[str]] = None):
    """Overlay 1D runs and print time per point, point noise and forward/reverse mismatch.

    current: lock-in measuring the current, so voltage channels are shown as R = V/I.
    Runs of the same function swept in opposite directions are paired for 'fwd/rev'.
    """
    import matplotlib.pyplot as plt

    from .plotting import SERIES, STYLE

    runs = [_run_curves(r, current, basedir) for r in run_ids]
    names = list(channels or [n for n in runs[0][1] if n != current] or runs[0][1])
    funcs = [load_run(r, basedir).metadata.get("function") for r in run_ids]
    head = f"{'run':<22} {'ms/point':>9}"
    for n in names:
        head += f" | {n + ' noise':>14} {'fwd/rev':>8}"
    print(head)
    for i, (x, curves, tpp, label, _) in enumerate(runs):
        line = f"{label:<22} {tpp * 1e3:9.0f}"
        partner = next((runs[j] for j in range(len(runs)) if j != i and funcs[j] == funcs[i]
                        and (runs[j][0][-1] - runs[j][0][0]) * (x[-1] - x[0]) < 0), None)
        for n in names:
            v = curves[n][0]
            med = np.nanmedian(np.abs(v))
            mism = np.nan
            if partner is not None:
                px, pc = partner[0], partner[1][n][0]
                o = np.argsort(px)
                mism = np.sqrt(np.nanmean((v - np.interp(x, px[o], pc[o])) ** 2)) / med
            line += f" | {100 * point_noise(v) / med:>13.3g}% {100 * mism:>7.3g}%"
        print(line)
    print("noise = point-to-point scatter around a local fit; fwd/rev = rms difference between "
          "up and down sweeps of the same kind (lag/smearing + noise; ~1.4x the noise without "
          "lag); both relative to the median value")
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(names), 1, sharex=True, squeeze=False,
                                 figsize=(10, 2.8 * len(names) + 0.6))
        for ax, n in zip(axes[:, 0], names):
            for i, (x, curves, tpp, label, _) in enumerate(runs):
                v, q, u = curves[n]
                lab = labels[i] if labels else f"{label} ({tpp * 1e3:.0f} ms/pt)"
                ax.plot(x, v, "-", marker="o", ms=2.5, lw=1.0, color=SERIES[i % len(SERIES)], label=lab)
            ax.set_title(n, pad=4)
            ax.set_ylabel(f"{curves[n][1]} ({curves[n][2]})")
        axes[0, 0].legend(loc="best", fontsize=8)
        axes[-1, 0].set_xlabel(runs[0][4])
        fig.tight_layout()
    return fig
