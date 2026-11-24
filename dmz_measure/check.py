"""Step 3: measure the noise at chosen frequencies, at real settings, excitation on.

    check = dm.check_frequencies(measurement_config, [17.5, 23.85, SR860_A.frequency()], gains,
                                 current='SR860_A', tau=None, sync=True, duration=120, dwell=0.3)

For each frequency the reference lock-in is set to it, every lock-in gets the
given filter settings, and all of them stream for `duration` seconds. Reported per
channel: the value (R = V/I when `current` is given), the noise of one data point
(scatter of `dwell`-second averages, relative), the worst `block`-second stretch,
and the phase. The excitation amplitude is never changed; settings are restored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from . import lockin as lk
from . import noise as nz
from .instruments.sr860 import apply_settings, preserved_settings, wait_for_lock
from .instruments.sr860_stream import acquire_many
from .io import RunWriter, get_basedir
from .plotting import eng
from .survey import lockins_from_config, reference_lockins, start_streams


@dataclass
class Check:
    rows: List[Dict[str, Any]]
    names: List[str]
    current: Optional[str]
    dwell: float
    block: float
    window: str = "boxcar"
    allan: Dict[float, Dict[str, Any]] = field(default_factory=dict)
    run_id: Optional[int] = None

    def plot(self):
        """Allan deviation per channel: falls as 1/sqrt(time) while averaging helps."""
        from . import plotting

        return plotting.plot_check(self)

    def _quantity(self, name: str):
        if self.current is None:
            return "X", "V"
        return ("I", "A") if name == self.current else ("R", "Ω")

    def __repr__(self) -> str:
        L = [f"Frequency check{f' (run {self.run_id})' if self.run_id is not None else ''}: "
             f"noise = scatter of {self.dwell:g} s {'Hann-weighted ' if self.window == 'hann' else ''}"
             f"averages relative to the value; worst = worst {self.block:g} s stretch"]
        head = f"{'f (Hz)':>9} {'tau':>7} {'adv':>4} {'sync':>4}"
        for n in self.names:
            q, _ = self._quantity(n)
            head += f" | {n + ' ' + q:>13} {'noise':>8} {'worst':>8} {'phase':>6}"
        L.append(head)
        for r in self.rows:
            line = (f"{r['f']:9.4f} {lk._fmt_time(r['tau']):>7} {'on' if r['advanced'] else 'off':>4} "
                    f"{'on' if r['sync'] else 'off':>4}")
            for n in self.names:
                _, u = self._quantity(n)
                line += (f" | {eng(r[n + '_value'], u):>13} {_pct(r[n + '_rel_noise']):>8} "
                         f"{_pct(r[n + '_rel_noise_worst']):>8} {r[n + '_phase_deg']:>5.1f}°")
            L.append(line)
        flags = [f"{r['f']:.2f} Hz {n}: {r[n + '_flags']}" for r in self.rows for n in self.names
                 if r[n + "_flags"]]
        if flags:
            L.append("Flags: " + "; ".join(flags))
        return "\n".join(L)


def _pct(x: float) -> str:
    return f"{100 * x:.3g}%" if np.isfinite(x) else "-"


def check_frequencies(measurement_config: Mapping[str, Any], frequencies: Sequence[float],
                      gains: Optional[Mapping[str, float]] = None, current: Optional[str] = None,
                      *, tau: Optional[float] = None, slope: int = 24,
                      advanced: Optional[bool] = None, sync: bool = True,
                      duration: float = 120.0, dwell: float = 0.3, window: str = "hann",
                      block: float = 30.0,
                      basedir: Optional[str] = None, comment: str = "") -> Check:
    """Step 3: measure each frequency with the excitation ON (set it by hand first).

    frequencies : any list, e.g. best.frequencies[:3] + [SR860_A.frequency()]
    tau         : lock-in time constant (s); None = largest SR860 value <= 1/(30 f) per frequency
    slope, sync : filter slope (dB/oct) and sync filter
    advanced    : SR860 Advanced filter on/off; None = leave the lock-ins as they are
    duration    : seconds streamed per frequency
    dwell       : averaging time of one data point (s); noise = scatter of these averages
    window      : weighting within each dwell: 'boxcar' (plain mean) or 'hann'
    block       : stretch length (s) for the 'worst' column (time variation)
    """
    lockins = lockins_from_config(measurement_config, gains, current)
    ios = [l.io for l in lockins.values()]
    rows: List[Dict[str, Any]] = []
    allan: Dict[float, Dict[str, Any]] = {}
    with preserved_settings(ios) as snaps:
        refs = reference_lockins(lockins, snaps)
        for f in [float(x) for x in frequencies]:
            t = tau if tau is not None else lk.recommended_tau(f)
            for io in refs:
                apply_settings(io, freq=f)
            for io in ios:
                apply_settings(io, tau=t, slope_db=slope, sync=sync, advanced=advanced)
            t = ios[0].time_constant()
            adv = bool(ios[0].ask_int("ADVFILT?"))
            wait_for_lock(ios, minimum=2.0 + lk.settling_time(t, f, slope, advanced=adv, sync=sync))
            streams = start_streams(lockins, lk.recommended_stream_rate(t))
            print(f"\n{f:.4f} Hz: tau {lk._fmt_time(t)}, {slope} dB/oct{' advanced' if adv else ''}, "
                  f"sync {'on' if sync else 'off'}, {duration:g} s")
            data = acquire_many(streams, duration)
            row, allan[f] = _analyse(f, t, sync, data, lockins, current, dwell, block, window)
            row["advanced"] = adv
            rows.append(row)
    check = Check(rows, list(lockins), current, dwell, block, window, allan)

    base = get_basedir(basedir, required=False)
    if base:
        _save_check(check, base, comment or "frequency check", measurement_config)
    print()
    print(check)
    return check


def _analyse(f, tau, sync, data, lockins, current, dwell, block, window):
    skip = 2.0 / f + 10 * tau  # filter warm-up
    row: Dict[str, Any] = {"f": f, "tau": tau, "sync": sync}
    allan, points = {}, {}
    for n, l in lockins.items():
        d = data[n]
        z = d.z[int(skip * d.fs):] / l.gain
        points[n] = (d, nz.block_means(z, max(1, int(round(dwell * d.fs))), window))
        allan[n] = nz.allan_deviation(z.real, d.fs)
        row[n + "_flags"] = ", ".join(
            s for s, k in (("dropped packets", d.n_dropped), ("OVERLOAD", d.n_overload),
                           ("UNLOCK/SYNC", d.n_error)) if k)
    for n in lockins:
        d, p = points[n]
        if current is not None and n != current:
            di, pi = points[current]
            k = int(round((d.t0 - di.t0) / dwell)) if np.isfinite(d.t0 - di.t0) else 0
            pi = pi[k:] if k > 0 else pi
            p = p[-k:] if k < 0 else p
            m = min(len(p), len(pi))
            p = p[:m] / pi[:m]  # R = V/I point by point
        mean = p.mean()
        ref = abs(mean.real) if mean.real else np.nan
        row[n + "_value"] = float(mean.real)
        row[n + "_phase_deg"] = float(np.degrees(np.angle(mean)))
        row[n + "_rel_noise"] = float(p.real.std(ddof=1) / ref)
        k = int(block / dwell)
        groups = [p.real[i:i + k] for i in range(0, len(p) - k + 1, k)] if k >= 3 else []
        row[n + "_rel_noise_worst"] = (float(max(g.std(ddof=1) for g in groups) / ref)
                                       if groups else np.nan)
    return row, allan


def _save_check(check: Check, base, comment, measurement_config) -> None:
    import matplotlib.pyplot as plt

    cols = ["f", "tau"] + [f"{n}_{k}" for n in check.names
                           for k in ("value", "rel_noise", "rel_noise_worst", "phase_deg")]
    table = np.array([[float(r[c]) for c in cols] for r in check.rows])
    with RunWriter(base, "frequency_check", comment, dict(measurement_config)) as w:
        w.metadata.update(rows=check.rows, current=check.current, dwell=check.dwell,
                          block=check.block, window=check.window)
        w.write_table(cols, table)
        w.save_arrays("allan.npz", **{f"{n}_{f:.4f}Hz_{k}": v for f, per in check.allan.items()
                                      for n, (t, a) in per.items() for k, v in (("tau", t), ("adev", a))})
        fig = check.plot()
        w.save_figure(fig)
        plt.close(fig)
        check.run_id = w.id
    print(f"Saved run {check.run_id} in {base}")
