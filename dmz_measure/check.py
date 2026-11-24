"""Frequency check analysis. No instruments: works from the run ID of ``s.check_frequencies``.

    check = dm.load_check(run_id, current='SR860_A', dwell=0.6, window='hann', block=30)
    check                  # table per frequency: R = V/I, noise of one data point, worst stretch, phase
    check.plot()           # Allan deviation of R: falls as 1/sqrt(time) while averaging helps
    check.plot_traces()    # R vs time at each frequency (one point per dwell)

One data point = the `window`-weighted average of `dwell` seconds of the stream, the same
host-time stretch for every lock-in, so R = V/I is taken point by point. noise = scatter of
these points relative to their mean; worst = the noisiest `block`-second stretch. The
averaging can be redone with other dwell / window / block at any time: nothing is re-measured.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import lockin as lk
from . import noise as nz
from .io import load_run
from .plotting import eng, hz, pct
from .sweeps import _npz_to_segments, weighted_mean


@dataclass
class Check:
    rows: List[Dict[str, Any]]
    names: List[str]
    current: str
    dwell: float
    block: float
    window: str
    allan: List[Dict[str, Tuple[np.ndarray, np.ndarray]]]   # per frequency, per channel (relative)
    traces: List[Dict[str, Tuple[np.ndarray, np.ndarray]]]  # per frequency, per channel: t, value
    run_id: Optional[int] = None
    comments: List[str] = field(default_factory=list)

    @property
    def frequencies(self) -> List[float]:
        return [r["f"] for r in self.rows]

    def quantity(self, name: str) -> Tuple[str, str]:
        return ("I", "A") if name == self.current else ("R", "Ω")

    def plot(self):
        """Allan deviation of R (and I) relative to its value, one line per frequency."""
        from . import plotting

        return plotting.plot_check(self)

    def plot_traces(self):
        """R (and I) vs time at each frequency, one point per dwell."""
        from . import plotting

        return plotting.plot_check_traces(self)

    def __repr__(self) -> str:
        L = [f"Frequency check{f' (run {self.run_id})' if self.run_id is not None else ''}"]
        for c in self.comments:
            L.append(f"  comment: {c}")
        L.append(f"  noise = scatter of {self.dwell:g} s {'Hann-weighted ' if self.window == 'hann' else ''}"
                 f"averages of R = V/I (I for {self.current}), relative to the value; "
                 f"worst = worst {self.block:g} s stretch")
        head = f"{'f (Hz)':>12} {'tau':>7} {'adv':>4} {'sync':>4} {'s':>5}"
        for n in self.names:
            q, _ = self.quantity(n)
            head += f" | {n + ' ' + q:>13} {'noise':>8} {'worst':>8} {'phase':>7}"
        L.append(head)
        for r in self.rows:
            line = (f"{hz(r['f']):>12} {lk._fmt_time(r['tau']):>7} {'on' if r['advanced'] else 'off':>4} "
                    f"{'on' if r['sync'] else 'off':>4} {r['duration']:>5.0f}")
            for n in self.names:
                _, u = self.quantity(n)
                line += (f" | {eng(r[n + '_value'], u):>13} {pct(r[n + '_rel_noise']):>8} "
                         f"{pct(r[n + '_rel_noise_worst']):>8} {r[n + '_phase_deg']:>6.1f}°")
            L.append(line)
        flags = [f"{hz(r['f'])} Hz {n}: {r[n + '_flags']}" for r in self.rows for n in self.names
                 if r.get(n + "_flags")]
        if flags:
            L.append("Flags: " + "; ".join(flags))
        return "\n".join(L)


def _window(s: Dict[str, Any], a: float, b: float) -> np.ndarray:
    ja = max(int(np.ceil((a - s["t_first"]) * s["fs"])), 0)
    jb = min(int(np.floor((b - s["t_first"]) * s["fs"])), len(s["z"]))
    return s["z"][ja:jb]


def _aligned(s: Dict[str, Any], ref: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Samples of two streams at the same host times (same sample rate assumed)."""
    k = int(round((s["t_first"] - ref["t_first"]) * s["fs"]))  # s starts k samples after ref
    a, b = (s["z"], ref["z"][k:]) if k >= 0 else (s["z"][-k:], ref["z"])
    m = min(len(a), len(b))
    return a[:m], b[:m]


def _flags_text(flags: Dict[str, int]) -> str:
    names = {"dropped": "dropped packets", "overload": "OVERLOAD", "error": "UNLOCK/SYNC"}
    return ", ".join(names[k] for k, v in flags.items() if v)


def load_check(run_id: int, current: str, dwell: float = 0.3, window: str = "hann",
               block: float = 30.0, basedir: Optional[str] = None) -> Check:
    """Analyse a saved s.check_frequencies run.

    current : the lock-in that measures the current (the others give R = V/I)
    dwell   : averaging time of one data point (s); window: 'hann' or 'boxcar' weighting
    block   : stretch length (s) for the 'worst' column (noise that comes and goes)
    """
    run = load_run(run_id, basedir)
    md = run.metadata
    if md.get("function") != "check_frequencies":
        if md.get("type") == "frequency_check":
            raise ValueError(f"run {run_id} was saved by dmz_measure 0.6, which kept only its "
                             "summary table and no stream: it can't be re-analysed")
        raise ValueError(f"run {run_id} is not a frequency check")
    ss = md["stream"]
    gains = ss["gains"]
    names = list(gains)
    if current not in names:
        raise ValueError(f"current= must be one of {names}")
    segs = _npz_to_segments(os.path.join(run.dir, "stream.npz"), names)
    rows, allan, traces = [], [], []
    for info, seg in zip(ss["segments"], segs):
        st = seg["streams"]
        t_a = max(s["t_first"] for s in st.values())
        t_b = min(s["t_first"] + len(s["z"]) / s["fs"] for s in st.values())
        starts = np.arange(t_a, t_b - dwell + 1e-9, dwell)
        pts = {n: np.array([weighted_mean(_window(st[n], a, a + dwell), window) for a in starts])
               / gains[n] for n in names if n in st}
        row: Dict[str, Any] = {"f": info["f_ref"], "tau": info["tau"], "advanced": info["advanced"],
                               "sync": info["sync"], "duration": t_b - t_a}
        al: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        tr: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for n in names:
            row[n + "_flags"] = _flags_text(info.get("flags", {}).get(n, {}))
            p = pts[n] if n == current else pts[n] / pts[current]
            ok = np.isfinite(p)
            mean = p[ok].mean() if ok.sum() else complex(np.nan, np.nan)
            ref = abs(mean.real) if mean.real else np.nan
            row[n + "_value"] = float(mean.real)
            row[n + "_phase_deg"] = float(np.degrees(np.angle(mean)))
            row[n + "_rel_noise"] = float(p[ok].real.std(ddof=1) / ref) if ok.sum() > 2 else np.nan
            k = int(block / dwell)
            v = p.real
            groups = [v[i:i + k] for i in range(0, len(v) - k + 1, k)] if k >= 3 else []
            worst = [np.nanstd(g, ddof=1) for g in groups if np.isfinite(g).sum() > 2]
            row[n + "_rel_noise_worst"] = float(max(worst) / ref) if worst else np.nan
            tr[n] = (starts + dwell / 2 - t_a, v)
            # Allan deviation from the full-rate stream, relative to the value
            s = st[n]
            if n == current:
                x = s["z"] / gains[n]
            elif abs(s["fs"] - st[current]["fs"]) < 1e-6 * s["fs"]:
                zv, zi = _aligned(s, st[current])
                x = (zv / gains[n]) / (zi / gains[current])
            else:
                continue
            if len(x) > 100 and np.isfinite(ref):
                try:
                    al[n] = nz.allan_deviation(x.real / ref, s["fs"])
                except ValueError:  # nothing finite left
                    pass
        rows.append(row)
        allan.append(al)
        traces.append(tr)
    return Check(rows, names, current, dwell, block, window, allan, traces, run_id,
                 list(md.get("comments") or []))
