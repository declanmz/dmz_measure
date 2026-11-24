"""Steps 1 and 2: measure the noise spectrum, then choose frequencies from it.

    survey = dm.noise_survey(measurement_config, gains, current='SR860_A', f_ref=100, duration=600)
    best = dm.best_frequencies(survey, fmin=10, fmax=40, tau=0.3)
    dm.expected_noise(survey, [13.2943, 17.5], tau=0.3)

The survey streams every SR860 in ``measurement_config`` with the reference at
f_ref, a short time constant and the sync filter off. The complex FFT of X + iY
is the input spectrum shifted by f_ref, so one record gives each channel's
input-referred noise density over the band [fmin, fmax] (anything 0 < f < f_ref +
fs/2 is unambiguous).

Choosing frequencies is pure analysis of a survey (live or reloaded with
:func:`load_survey`). The predicted noise of one lock-in reading at reference f is

    sigma^2 = 1/2 * integral S(f') (|H(f' - f)|^2 + |H(f' + f)|^2) df'

with H the filter that makes one data point (tau, slope, sync, dwell). To catch
noise that comes and goes, the record is also cut into time blocks and sigma is
evaluated for each block: a frequency that is quiet on average but noisy in some
blocks shows a large spread (max/median) and is pushed down by ranking on the
90th percentile instead of the mean.

The code never changes the excitation amplitude. It changes the reference
frequency and filter settings, and restores them at the end.
"""
from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from . import lockin as lk
from . import noise as nz
from .instruments.sr860 import (LockinIO, apply_settings, find_instrument, is_sr860,
                                preserved_settings, wait_for_lock)
from .instruments.sr860_stream import DEFAULT_PORT, SR860Stream, acquire_many
from .io import RunWriter, get_basedir, load_run
from .plotting import eng


# --------------------------------------------------------------------------
# shared helpers (also used by check.py)
# --------------------------------------------------------------------------

@dataclass
class _Lockin:
    name: str
    io: LockinIO
    gain: float
    unit: str


def lockins_from_config(measurement_config: Mapping[str, Any],
                        gains: Optional[Mapping[str, float]],
                        current: Optional[str]) -> Dict[str, _Lockin]:
    """The open SR860s named in measurement_config (other entries are ignored)."""
    gains = dict(gains or {})
    out: Dict[str, _Lockin] = {}
    for name in measurement_config:
        inst = find_instrument(name)
        if inst is None:
            continue  # e.g. 'Vbg2': a label, not an instrument
        if not is_sr860(inst):
            if "SR830" in type(inst).__name__:
                print(f"{name}: SR830s can't stream, skipped")
            continue
        out[name] = _Lockin(name, LockinIO(inst, name), float(gains.get(name, 1.0)),
                            "A" if name == current else "V")
    if not out:
        raise RuntimeError("none of the measurement_config entries is an open SR860")
    if current is not None and current not in out:
        raise ValueError(f"current={current!r} is not one of the lock-ins {list(out)}")
    return out


def reference_lockins(lockins: Dict[str, _Lockin], snaps: Dict[str, Dict]) -> List[LockinIO]:
    """Lock-ins on internal reference: they set the frequency, the others follow."""
    refs = [l.io for l in lockins.values() if snaps[l.name]["RSRC"] == 0]
    if not refs:
        raise RuntimeError("no lock-in in measurement_config is on internal reference")
    return refs


def start_streams(lockins: Dict[str, _Lockin], rate: float) -> List[SR860Stream]:
    streams = [SR860Stream(l.io, port=DEFAULT_PORT + i) for i, l in enumerate(lockins.values())]
    for st in streams:
        st.configure(rate=rate)
    return streams


# --------------------------------------------------------------------------
# Step 1: survey
# --------------------------------------------------------------------------

@dataclass
class Survey:
    spectra: Dict[str, nz.NoiseSpectrum]   # whole-record spectra, input referred
    z: Dict[str, np.ndarray]               # X + iY, input referred, decimated
    fs: Dict[str, float]                   # sample rate of z
    lines: Dict[str, List[Dict[str, Any]]]
    mains: float
    settings: Dict[str, Any]
    run_id: Optional[int] = None
    _blocks: Dict[float, Dict[str, List[nz.NoiseSpectrum]]] = field(default_factory=dict, repr=False)

    def plot(self, fmin: Optional[float] = None, fmax: Optional[float] = None):
        """Noise density of each channel, mains harmonics and the strongest lines marked."""
        from . import plotting

        return plotting.plot_survey(self, fmin=fmin, fmax=fmax)

    def spectrogram(self, channel: str, block: float = 30.0, fmin: Optional[float] = None,
                    fmax: Optional[float] = None):
        """Noise density vs time and frequency for one channel: look for bursts."""
        from . import plotting

        s = self.spectra[channel]
        t, f, P = nz.spectrogram(self.z[channel], self.fs[channel], s.f_ref, segment=block,
                                 fmax=fmax or s.f[-1], orientation=s.meta["orientation"]["used"])
        return plotting.plot_spectrogram(t, f, P, name=channel, unit=s.unit,
                                         fmin=fmin if fmin is not None else s.f[0], fmax=fmax)

    def block_spectra(self, block: float) -> Dict[str, List[nz.NoiseSpectrum]]:
        if block not in self._blocks:
            self._blocks[block] = {n: nz.block_spectra(self.z[n], self.fs[n], s, block)
                                   for n, s in self.spectra.items()}
        return self._blocks[block]

    def __repr__(self) -> str:
        st = self.settings
        L = [f"Noise survey{f' (run {self.run_id})' if self.run_id is not None else ''}: "
             f"{', '.join(self.spectra)}",
             f"  f_ref {st['f_ref']:g} Hz, {st['duration']:g} s, band {st['fmin']:g}-{st['fmax']:g} Hz, "
             f"bins {next(iter(self.spectra.values())).df:.3g} Hz, tau {lk._fmt_time(st['tau'])}, "
             f"mains {self.mains:g} Hz"]
        for n, s in self.spectra.items():
            L.append(f"  {n}: floor {eng(s.floor(), s.unit)}/√Hz ({s.n_segments} averages)")
        seen: List[Dict[str, Any]] = []
        for ln in sorted((x for v in self.lines.values() for x in v), key=lambda x: -x["snr_db"]):
            if all(abs(ln["f"] - s["f"]) > 0.3 for s in seen):
                seen.append(ln)
        if seen:
            L.append("  strongest lines (Hz): " + ", ".join(
                f"{ln['f']:.2f}" + (f" ({ln['tag']})" if ln["tag"] else "") for ln in seen[:8]))
        return "\n".join(L)


def noise_survey(measurement_config: Mapping[str, Any], gains: Optional[Mapping[str, float]] = None,
                 current: Optional[str] = None, *, f_ref: float = 100.0, duration: float = 600.0,
                 fmin: float = 0.0, fmax: Optional[float] = None, resolution: float = 0.05,
                 tau: float = 100e-6, rate: float = 10e3, basedir: Optional[str] = None,
                 comment: str = "") -> Survey:
    """Step 1: stream all SR860s in measurement_config and take the FFT. Turn the excitation off first.

    f_ref      : reference frequency during the survey (Hz)
    duration   : seconds of streaming
    fmin, fmax : band of input frequencies to keep (Hz); fmax defaults to 2 f_ref
    resolution : FFT bin width (Hz); Welch segments are 1/resolution s, 50 % overlap
    tau        : lock-in time constant during the survey (keep it <= 100 us so the band is flat)
    rate       : stream sample rate (Hz); ~100x the band is plenty against aliasing
    gains      : preamp gains to divide out; current: the lock-in that measures current
    basedir    : data folder (default: the one from sweep.set_basedir)
    """
    fmax = 2 * f_ref if fmax is None else fmax
    lockins = lockins_from_config(measurement_config, gains, current)
    ios = [l.io for l in lockins.values()]
    with preserved_settings(ios) as snaps:
        refs = reference_lockins(lockins, snaps)
        for io in refs:
            apply_settings(io, freq=f_ref)
        for io in ios:
            apply_settings(io, tau=tau, slope_db=24, sync=False)
        wait_for_lock(ios, minimum=3.0)
        f_det = {n: l.io.detection_frequency() for n, l in lockins.items()}
        tau_set = lockins[next(iter(lockins))].io.time_constant()
        streams = start_streams(lockins, rate)
        print(f"Streaming {', '.join(lockins)} for {duration:g} s at {streams[0].rate:g} Hz "
              f"(f_ref {f_ref:g} Hz, tau {lk._fmt_time(tau_set)}) ...")
        data = acquire_many(streams, duration)

    # keep only what the band needs: decimate to >= 4x the largest offset from f_ref
    half = max(abs(fmin - f_ref), abs(fmax - f_ref), 1.0)
    z, fs = {}, {}
    for n, l in lockins.items():
        d = data[n]
        q = int(2 ** max(0, np.floor(np.log2(d.fs / (4 * half)))))
        z[n] = (nz.decimate(d.z, q) / l.gain).astype(np.complex64)
        fs[n] = d.fs / q
    settings = dict(f_ref=f_ref, duration=duration, fmin=fmin, fmax=fmax, resolution=resolution,
                    tau=tau_set, rate=streams[0].rate, f_det=f_det,
                    units={n: l.unit for n, l in lockins.items()},
                    gains={n: l.gain for n, l in lockins.items()},
                    dropped={n: d.n_dropped for n, d in data.items()},
                    overload={n: d.n_overload for n, d in data.items()})
    del data
    survey = _build_survey(z, fs, settings)

    base = get_basedir(basedir, required=False)
    if base:
        _save_survey(survey, base, comment or f"noise survey f_ref {f_ref:g} Hz",
                     measurement_config)
    else:
        print("Not saved: no data folder (call sweep.set_basedir(...) or pass basedir=...)")
    print(survey)
    return survey


def _build_survey(z, fs, st, run_id=None) -> Survey:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # orientation notes are reported in the summary below
        spectra = {n: nz.input_spectrum(z[n], fs[n], st["f_det"][n], unit=st["units"][n],
                                        resolution=st["resolution"], fmin=st["fmin"],
                                        fmax=st["fmax"], tau=st["tau"], name=n) for n in z}
    mains = Counter(s.meta["mains"] for s in spectra.values()).most_common(1)[0][0]
    for s in spectra.values():
        s.meta["mains"] = mains
    lines = {n: nz.find_lines(s) for n, s in spectra.items()}
    for n, s in spectra.items():
        on = [ln for ln in lines[n] if abs(ln["f"] - s.f_ref) < 3 * s.df]
        if on:
            print(f"NOTE {n}: strong signal at f_ref = {s.f_ref:g} Hz "
                  f"({eng(on[0]['rms'], s.unit)} rms) - is the excitation still on?")
    return Survey(spectra, z, fs, lines, mains, st, run_id)


def _save_survey(sv: Survey, base, comment, measurement_config) -> None:
    import matplotlib.pyplot as plt

    names = list(sv.spectra)
    f0 = sv.spectra[names[0]].f
    cols = ["f_Hz"] + [f"{n}_asd_{sv.spectra[n].unit}_per_rtHz" for n in names]
    table = [f0] + [np.interp(f0, sv.spectra[n].f, sv.spectra[n].asd) for n in names]
    with RunWriter(base, "noise_survey", comment, dict(measurement_config)) as w:
        w.metadata.update(settings=sv.settings, mains=sv.mains, lines=sv.lines)
        w.write_table(cols, np.column_stack(table))
        w.save_arrays("timeseries.npz", **{f"{n}_z": sv.z[n] for n in names},
                      **{f"{n}_fs": np.array(sv.fs[n]) for n in names})
        fig = sv.plot()
        w.save_figure(fig)
        plt.close(fig)
        sv.run_id = w.id
    print(f"Saved run {sv.run_id} in {base}")


def load_survey(run_id: int, basedir: Optional[str] = None) -> Survey:
    """Reload a saved survey (spectra are recomputed from the stored time series)."""
    run = load_run(run_id, basedir)
    if run.metadata.get("type") != "noise_survey":
        raise ValueError(f"run {run_id} is a {run.metadata.get('type')!r}, not a noise_survey")
    st = run.metadata["settings"]
    arr = run.arrays("timeseries")
    names = list(st["units"])
    return _build_survey({n: arr[f"{n}_z"] for n in names},
                         {n: float(arr[f"{n}_fs"]) for n in names}, st, run_id)


# --------------------------------------------------------------------------
# Step 2: choosing frequencies (analysis only)
# --------------------------------------------------------------------------

STATS = ("mean", "median", "p90", "max")


def _noise_stats(survey: Survey, f: np.ndarray, filt: lk.MeasurementFilter,
                 block: Optional[float]) -> Dict[str, Dict[str, np.ndarray]]:
    """Predicted reading noise per channel: whole record ('mean') and over time blocks."""
    out = {}
    blocks = survey.block_spectra(block) if block else {}
    for n, s in survey.spectra.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full = nz.predicted_noise(s, f, filt)
            B = np.array([nz.predicted_noise(b, f, filt) for b in blocks.get(n, [])])
        st = {"mean": full}
        if len(B) >= 3:
            st.update(p10=np.nanpercentile(B, 10, axis=0), median=np.nanmedian(B, axis=0),
                      p90=np.nanpercentile(B, 90, axis=0), max=np.nanmax(B, axis=0))
        else:
            st.update(p10=full, median=full, p90=full, max=full)
        st["spread"] = st["max"] / st["median"]
        out[n] = st
    return out


def _filter(tau, slope, sync, dwell, advanced, window) -> lk.MeasurementFilter:
    return lk.MeasurementFilter(tau=tau, slope_db=slope, sync=sync, dwell=dwell,
                                advanced=advanced, window=window)


@dataclass
class Ranking:
    f: np.ndarray
    noise: Dict[str, Dict[str, np.ndarray]]
    score: np.ndarray
    excluded: np.ndarray
    best: List[Dict[str, Any]]
    filter: lk.MeasurementFilter
    stat: str
    block: Optional[float]
    units: Dict[str, str]
    fmin: float = 0.0
    fmax: float = 0.0

    @property
    def frequencies(self) -> List[float]:
        return [b["f"] for b in self.best]

    def plot(self):
        from . import plotting

        return plotting.plot_ranking(self)

    @property
    def typical_spread(self) -> float:
        """Median spread over the range: what steady noise gives with these blocks."""
        return float(np.nanmedian([st["spread"] for st in self.noise.values()]))

    def __repr__(self) -> str:
        names = list(self.noise)
        how = f"{self.stat} over {self.block:g} s blocks" if self.block else self.stat
        L = [f"Best frequencies {self.fmin:g}-{self.fmax:g} Hz for {self.filter.describe()}, "
             f"ranked by {how}",
             "  noise = predicted rms of one reading (input referred)"]
        if self.block:
            L.append(f"  spread = worst block / typical block; steady noise gives about "
                     f"{self.typical_spread:.2f} here, clearly more = intermittent noise")
        head = f"{'#':>3} {'f (Hz)':>8}"
        for n in names:
            head += f" | {n + ' noise':>15} {'spread':>6}"
        L.append(head)
        for j, b in enumerate(self.best, start=1):
            row = f"{j:>3} {b['f']:8.2f}"
            for n in names:
                row += f" | {eng(b[n], self.units[n]):>15} {b[n + '_spread']:>6.2f}"
            L.append(row)
        return "\n".join(L)


def best_frequencies(survey: Survey, fmin: float, fmax: float, *, tau: float = 0.3,
                     slope: int = 24, advanced: bool = False, sync: bool = False,
                     dwell: Optional[float] = None, window: str = "boxcar", n: int = 5, min_spacing: float = 1.0, avoid_mains: bool = True,
                     block: Optional[float] = 60.0, stat: str = "p90") -> Ranking:
    """Step 2: the quietest reference frequencies in [fmin, fmax] for the given lock-in settings.

    tau, slope, sync : lock-in time constant (s), slope (dB/oct) and sync filter you will use
    advanced         : SR860 Advanced filter on (Gaussian stages for tau <= 3 s)
    dwell            : extra software averaging per data point (s); None = one reading
    window           : weighting within the dwell: 'boxcar' (plain mean) or 'hann'
    n, min_spacing   : how many to return and how far apart (Hz)
    avoid_mains      : skip mains * p/q, q <= 4 (15, 20, 30, 40, 45 ... Hz for 60 Hz)
    block            : time-block length (s) for the time-variation analysis; None = off
    stat             : rank by 'mean' (whole record), 'median', 'p90' (exceeded 10 % of
                       the time) or 'max' over blocks
    Channels are combined with a geometric mean of noise / median-over-range, since
    they share one reference.
    """
    if stat not in STATS:
        raise ValueError(f"stat must be one of {STATS}")
    filt = _filter(tau, slope, sync, dwell, advanced, window)
    step = min(s.df for s in survey.spectra.values())
    f = np.arange(fmin, fmax + step / 2, step)
    f = f[f <= fmax + 1e-9]
    stats = _noise_stats(survey, f, filt, block)
    rel = [np.log(st[stat] / np.nanmedian(st[stat])) for st in stats.values()]
    score = np.exp(np.mean(rel, axis=0))
    tol = max(0.5, 3 * filt.enbw(float(np.mean([fmin, fmax]))))
    excluded = nz.mains_related(f, survey.mains, 4, tol) if avoid_mains else np.zeros(f.shape, bool)

    best: List[Dict[str, Any]] = []
    for i in np.argsort(np.where(np.isfinite(score) & ~excluded, score, np.inf)):
        if not np.isfinite(score[i]) or excluded[i]:
            break
        if any(abs(f[i] - b["f"]) < min_spacing for b in best):
            continue
        row: Dict[str, Any] = {"f": float(f[i]), "score": float(score[i])}
        for name, st in stats.items():
            row[name] = float(st[stat][i])
            row[name + "_spread"] = float(st["spread"][i])
        best.append(row)
        if len(best) >= n:
            break
    units = {name: s.unit for name, s in survey.spectra.items()}
    ranking = Ranking(f, stats, score, excluded, best, filt, stat, block, units, fmin, fmax)
    print(ranking)
    return ranking


def expected_noise(survey: Survey, frequencies, *, tau: float = 0.3, slope: int = 24,
                   advanced: bool = False, sync: bool = False, dwell: Optional[float] = None,
                   window: str = "boxcar", block: Optional[float] = 60.0) -> List[Dict[str, Any]]:
    """Predicted noise of one reading at the given frequencies, for the given lock-in settings.

    Prints, per channel: the whole-record value, the 90th percentile and the worst
    `block`-second block. Returns the rows.
    """
    fr = np.atleast_1d(np.asarray(frequencies, dtype=float))
    filt = _filter(tau, slope, sync, dwell, advanced, window)
    stats = _noise_stats(survey, fr, filt, block)
    names = list(stats)
    units = {n: s.unit for n, s in survey.spectra.items()}
    rows = []
    for i, f in enumerate(fr):
        row: Dict[str, Any] = {"f": float(f)}
        for n in names:
            for k in ("mean", "p90", "max"):
                row[f"{n}_{k}"] = float(stats[n][k][i])
        rows.append(row)
    how = f"; p90 and worst over {block:g} s blocks" if block else ""
    print(f"Expected noise for {filt.describe()} (rms of one reading, input referred{how})")
    head = f"{'f (Hz)':>9}"
    for n in names:
        head += f" | {n + ' mean':>14} {'p90':>9} {'worst':>9}"
    print(head)
    for r in rows:
        line = f"{r['f']:9.3f}"
        for n in names:
            u = units[n]
            line += (f" | {eng(r[n + '_mean'], u):>14} {eng(r[n + '_p90'], u):>9} "
                     f"{eng(r[n + '_max'], u):>9}")
        print(line)
    return rows
