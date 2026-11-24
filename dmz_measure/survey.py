"""Noise survey analysis and choosing the excitation frequency. No instruments: everything
works from the run ID of a survey measured with ``s.noise_survey``.

    survey = dm.load_survey(run_id, current='SR860_A')      # spectra from the saved stream
    survey.plot(); survey.spectrogram('SR860_B')
    best = dm.best_frequencies(survey, resistances={'SR860_B': 106}, excitation=2e-9,
                               fmin=1, fmax=49, tau=0.3)
    dm.expected_noise(survey, [13.2943, 17.5623], resistances={'SR860_B': 106},
                      excitation=2e-9, tau=0.3)

Spectra: the lock-ins streamed X + iY at reference f_ref with a short time constant. The
complex FFT of that record is the input spectrum shifted by f_ref, so one record gives each
lock-in's input-referred noise density from 0 to 2 f_ref.

Resistance noise: a data point at reference frequency f reads the voltage V and the current I
with noise sigma_V(f) and sigma_I(f): the part of the measured spectra that passes the
lock-in filter you will use (tau, slope, sync, dwell; see noise.predicted_noise). For
R = V / I, excitation current I and a typical resistance R of the channel, with the voltage
and current noise independent of each other,

    sigma_R = sqrt(sigma_V^2 + (R sigma_I)^2) / I

Frequencies are ranked by the worst sigma_R / R among the voltage channels you list. To catch
noise that comes and goes, the survey is cut into `block`-second pieces and sigma_R is worked
out for each piece: ranking on the 90th percentile ('p90') pushes down frequencies that are
quiet on average but noisy part of the time.
"""
from __future__ import annotations

import os
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import lockin as lk
from . import noise as nz
from .io import load_run
from .plotting import eng, hz, pct
from .sweeps import _npz_to_segments


# --------------------------------------------------------------------------
# Loading a survey
# --------------------------------------------------------------------------

@dataclass
class Survey:
    spectra: Dict[str, nz.NoiseSpectrum]   # whole-record spectra, input referred
    z: Dict[str, np.ndarray]               # X + iY, input referred (gain divided out)
    fs: Dict[str, float]                   # sample rate of z
    lines: Dict[str, List[Dict[str, Any]]]
    mains: float
    settings: Dict[str, Any]
    run_id: Optional[int] = None
    comments: List[str] = field(default_factory=list)
    _blocks: Dict[float, Dict[str, List[nz.NoiseSpectrum]]] = field(default_factory=dict, repr=False)

    @property
    def current(self) -> str:
        """The lock-in that measures the current."""
        return self.settings["current"]

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
             f"{', '.join(self.spectra)}; current: {self.current}"]
        for c in self.comments:
            L.append(f"  comment: {c}")
        L.append(f"  f_ref {hz(st['f_ref'])} Hz, {st['duration']:.0f} s, band {hz(st['fmin'])}-"
                 f"{hz(st['fmax'])} Hz, bins {next(iter(self.spectra.values())).df:.4g} Hz, "
                 f"tau {lk._fmt_time(st['tau'])}, mains {self.mains:g} Hz")
        for n, s in self.spectra.items():
            L.append(f"  {n}: floor {eng(s.floor(), s.unit)}/√Hz ({s.n_segments} averages)")
        seen: List[Dict[str, Any]] = []
        for ln in sorted((x for v in self.lines.values() for x in v), key=lambda x: -x["snr_db"]):
            if all(abs(ln["f"] - s["f"]) > 0.3 for s in seen):
                seen.append(ln)
        if seen:
            L.append("  strongest lines (Hz): " + ", ".join(
                f"{ln['f']:.3f}" + (f" ({ln['tag']})" if ln["tag"] else "") for ln in seen[:8]))
        return "\n".join(L)


def load_survey(run_id: int, current: Optional[str] = None, resolution: float = 0.05,
                fmin: float = 0.0, fmax: Optional[float] = None,
                basedir: Optional[str] = None) -> Survey:
    """Spectra of a saved noise survey (s.noise_survey, or a survey from dmz_measure 0.6).

    current    : the lock-in that measures the current (its spectrum is in A/√Hz)
    resolution : FFT bin width (Hz); Welch segments of 1/resolution s, 50 % overlap
    fmin, fmax : band of input frequencies to keep (Hz); fmax defaults to 2 f_ref
    """
    run = load_run(run_id, basedir)
    md = run.metadata
    if md.get("function") == "noise_survey":
        ss = md["stream"]
        gains = ss["gains"]
        seg = _npz_to_segments(os.path.join(run.dir, "stream.npz"), list(gains))[0]
        z = {n: s["z"] / gains[n] for n, s in seg["streams"].items()}
        fs = {n: s["fs"] for n, s in seg["streams"].items()}
        n0 = next(iter(z))
        st = {"f_ref": ss["f_ref"], "tau": ss["tau"], "f_det": ss["f_det"], "gains": gains,
              "rate": ss.get("rate"), "flags": ss.get("flags", {}),
              "duration": len(z[n0]) / fs[n0]}
    elif md.get("type") == "noise_survey":   # saved by dmz_measure 0.6: already input referred
        st = dict(md["settings"])
        arr = run.arrays("timeseries")
        z = {n: arr[f"{n}_z"] for n in st["units"]}
        fs = {n: float(arr[f"{n}_fs"]) for n in st["units"]}
        if current is None:
            current = next((n for n, u in st["units"].items() if u == "A"), None)
    else:
        raise ValueError(f"run {run_id} is not a noise survey")
    if current not in z:
        raise ValueError(f"current= must be the lock-in that measures the current: one of {list(z)}")
    st.update(current=current, units={n: ("A" if n == current else "V") for n in z},
              resolution=resolution, fmin=fmin, fmax=2 * st["f_ref"] if fmax is None else fmax)
    return _build_survey(z, fs, st, run_id, list(md.get("comments") or []))


def _build_survey(z, fs, st, run_id=None, comments=()) -> Survey:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # orientation notes are reported in the summary
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
            print(f"NOTE {n}: strong signal at f_ref = {hz(s.f_ref)} Hz "
                  f"({eng(on[0]['rms'], s.unit)} rms) - was the excitation on?")
    return Survey(spectra, z, fs, lines, mains, st, run_id, list(comments))


# --------------------------------------------------------------------------
# Resistance noise from the spectra
# --------------------------------------------------------------------------

STATS = ("mean", "median", "p90", "max")


def _filter(tau, slope, sync, dwell, advanced, window) -> lk.MeasurementFilter:
    return lk.MeasurementFilter(tau=tau, slope_db=slope, sync=sync, dwell=dwell,
                                advanced=advanced, window=window)


def _reading_noise(survey: Survey, names: Sequence[str], f: np.ndarray, filt: lk.MeasurementFilter,
                   block: Optional[float]) -> Dict[str, Tuple[np.ndarray, Dict[int, np.ndarray]]]:
    """Predicted noise of one reading per channel: (whole record, {block number: per block})."""
    blocks = survey.block_spectra(block) if block else {}
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n in names:
            full = nz.predicted_noise(survey.spectra[n], f, filt)
            per = {b.meta["block"]: nz.predicted_noise(b, f, filt) for b in blocks.get(n, [])}
            out[n] = (full, per)
    return out


def _stats(full: np.ndarray, B: np.ndarray) -> Dict[str, np.ndarray]:
    st = {"mean": full}
    if len(B) >= 3:
        st.update(p10=np.nanpercentile(B, 10, axis=0), median=np.nanmedian(B, axis=0),
                  p90=np.nanpercentile(B, 90, axis=0), max=np.nanmax(B, axis=0))
    else:
        st.update(p10=full, median=full, p90=full, max=full)
    st["spread"] = st["max"] / st["median"]
    return st


def _resistance_noise(survey: Survey, f: np.ndarray, filt: lk.MeasurementFilter,
                      block: Optional[float], resistances: Mapping[str, float],
                      excitation: float) -> Dict[str, Dict[str, np.ndarray]]:
    """sigma_R (Ohm) per voltage channel: whole record ('mean') and statistics over blocks."""
    cur = survey.current
    for n in resistances:
        if n not in survey.spectra or n == cur:
            raise ValueError(f"resistances: {n!r} is not a voltage channel of this survey "
                             f"({[m for m in survey.spectra if m != cur]})")
    if not excitation or excitation <= 0:
        raise ValueError("excitation: the excitation current in A (e.g. 2e-9)")
    pred = _reading_noise(survey, [cur] + list(resistances), f, filt, block)
    full_i, per_i = pred[cur]
    out = {}
    for n, R in resistances.items():
        full_v, per_v = pred[n]

        def sig(v, i, R=abs(R)):
            return np.sqrt(v ** 2 + (R * i) ** 2) / excitation

        common = sorted(set(per_v) & set(per_i))
        B = np.array([sig(per_v[k], per_i[k]) for k in common]) if common else np.empty((0, len(f)))
        out[n] = _stats(sig(full_v, full_i), B)
    return out


def _check_args(stat: str, resistances: Mapping[str, float]) -> Dict[str, float]:
    if stat not in STATS:
        raise ValueError(f"stat must be one of {STATS}")
    if not resistances:
        raise ValueError("resistances: {'SR860_B': R in Ohm, ...} for the voltage channels to rank by")
    return {n: float(R) for n, R in resistances.items()}


# --------------------------------------------------------------------------
# Choosing frequencies
# --------------------------------------------------------------------------

@dataclass
class Ranking:
    f: np.ndarray
    noise: Dict[str, Dict[str, np.ndarray]]   # sigma_R (Ohm) per channel and statistic
    resistances: Dict[str, float]
    excitation: float
    current: str
    score: np.ndarray                         # worst sigma_R / R over the channels
    excluded: np.ndarray
    best: List[Dict[str, Any]]
    filter: lk.MeasurementFilter
    stat: str
    block: Optional[float]
    fmin: float = 0.0
    fmax: float = 0.0
    run_id: Optional[int] = None

    @property
    def frequencies(self) -> List[float]:
        """The picks (Hz), rounded to 0.1 mHz (4 decimals)."""
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
        src = f" (survey run {self.run_id})" if self.run_id is not None else ""
        L = [f"Best frequencies {hz(self.fmin)}-{hz(self.fmax)} Hz{src} for {self.filter.describe()}; "
             f"ranked by the worst δR/R ({how})",
             f"  δR = predicted rms noise of one data point of R = V/I, for I = "
             f"{eng(self.excitation, 'A')} ({self.current}) and R = "
             + ", ".join(f"{eng(R, 'Ω')} ({n})" for n, R in self.resistances.items())]
        if self.block:
            L.append(f"  spread = worst block / typical block; steady noise gives about "
                     f"{self.typical_spread:.2f} here, clearly more = noise that comes and goes")
        head = f"{'#':>3} {'f (Hz)':>9}"
        for n in names:
            head += f" | {n + ' δR':>13} {'δR/R':>8} {'spread':>6}"
        L.append(head)
        for j, b in enumerate(self.best, start=1):
            row = f"{j:>3} {b['f']:9.4f}"
            for n in names:
                row += f" | {eng(b[n], 'Ω'):>13} {pct(b[n + '_rel']):>8} {b[n + '_spread']:>6.2f}"
            L.append(row)
        L.append("frequencies = [" + ", ".join(f"{b['f']:.4f}" for b in self.best) + "]   # Hz")
        return "\n".join(L)


def best_frequencies(survey: Survey, resistances: Mapping[str, float], excitation: float, *,
                     fmin: Optional[float] = None, fmax: Optional[float] = None,
                     tau: float = 0.3, slope: int = 24, advanced: bool = False,
                     sync: bool = False, dwell: Optional[float] = None, window: str = "boxcar",
                     n: int = 5, min_spacing: float = 1.0, avoid_mains: bool = True,
                     block: Optional[float] = 60.0, stat: str = "p90") -> Ranking:
    """The reference frequencies in [fmin, fmax] with the lowest resistance noise.

    resistances : {voltage lock-in: typical |R| in Ohm}, e.g. {'SR860_B': 106, 'SR860_C': 16};
                  ranked by the worst δR/R among them
    excitation  : excitation current (A), e.g. SR860_A's reading with the excitation on
    fmin, fmax  : range to search (Hz); default: the survey's band
    tau, slope, sync : lock-in time constant (s), slope (dB/oct) and sync filter you will use
    advanced    : SR860 Advanced filter on (Gaussian stages for tau <= 3 s)
    dwell       : extra software averaging per data point (s); None = one reading
    window      : weighting within the dwell: 'boxcar' (plain mean) or 'hann'
    n, min_spacing : how many to return and how far apart (Hz)
    avoid_mains : skip mains * p/q, q <= 4 (15, 20, 30, 40, 45 ... Hz for 60 Hz)
    block       : time-block length (s) for noise that comes and goes; None = off
    stat        : rank by 'mean' (whole record), 'median', 'p90' (exceeded 10 % of the
                  time) or 'max' over blocks
    Frequencies are rounded to 0.1 mHz and printed with all 4 decimals.
    """
    resistances = _check_args(stat, resistances)
    fmin = survey.settings["fmin"] if fmin is None else fmin
    fmax = survey.settings["fmax"] if fmax is None else fmax
    filt = _filter(tau, slope, sync, dwell, advanced, window)
    step = min(s.df for s in survey.spectra.values())
    f = np.round(np.arange(fmin, fmax + step / 2, step), 4)
    f = f[f <= fmax + 1e-9]
    stats = _resistance_noise(survey, f, filt, block, resistances, excitation)
    score = np.max([stats[c][stat] / abs(R) for c, R in resistances.items()], axis=0)
    tol = max(0.5, 3 * filt.enbw(float(np.mean([fmin, fmax]))))
    excluded = nz.mains_related(f, survey.mains, 4, tol) if avoid_mains else np.zeros(f.shape, bool)

    best: List[Dict[str, Any]] = []
    for i in np.argsort(np.where(np.isfinite(score) & ~excluded, score, np.inf)):
        if not np.isfinite(score[i]) or excluded[i]:
            break
        if any(abs(f[i] - b["f"]) < min_spacing for b in best):
            continue
        row: Dict[str, Any] = {"f": float(f[i]), "score": float(score[i])}
        for c, R in resistances.items():
            row[c] = float(stats[c][stat][i])
            row[c + "_rel"] = float(stats[c][stat][i] / abs(R))
            row[c + "_spread"] = float(stats[c]["spread"][i])
        best.append(row)
        if len(best) >= n:
            break
    ranking = Ranking(f, stats, resistances, float(excitation), survey.current, score, excluded,
                      best, filt, stat, block, fmin, fmax, survey.run_id)
    print(ranking)
    return ranking


def expected_noise(survey: Survey, frequencies, resistances: Mapping[str, float],
                   excitation: float, *, tau: float = 0.3, slope: int = 24,
                   advanced: bool = False, sync: bool = False, dwell: Optional[float] = None,
                   window: str = "boxcar", block: Optional[float] = 60.0) -> List[Dict[str, Any]]:
    """Predicted resistance noise of one data point at the given frequencies (Hz).

    Same arguments as best_frequencies. Prints, per voltage channel: δR over the whole record,
    its 90th percentile and worst value over `block`-second blocks, and the 90th percentile
    relative to R. Returns the rows.
    """
    resistances = _check_args("p90", resistances)
    fr = np.atleast_1d(np.asarray(frequencies, dtype=float))
    filt = _filter(tau, slope, sync, dwell, advanced, window)
    stats = _resistance_noise(survey, fr, filt, block, resistances, excitation)
    rows = []
    for i, f in enumerate(fr):
        row: Dict[str, Any] = {"f": float(f)}
        for n, R in resistances.items():
            for k in ("mean", "p90", "max"):
                row[f"{n}_{k}"] = float(stats[n][k][i])
            row[f"{n}_rel_p90"] = float(stats[n]["p90"][i] / abs(R))
        rows.append(row)
    how = f"; p90 and worst over {block:g} s blocks" if block else ""
    print(f"Expected noise of one data point of R = V/I for {filt.describe()}, I = "
          f"{eng(excitation, 'A')} ({survey.current}){how}")
    head = f"{'f (Hz)':>12}"
    for n in resistances:
        head += f" | {n + ' δR':>13} {'p90':>9} {'worst':>9} {'p90/R':>8}"
    print(head)
    for r in rows:
        line = f"{hz(r['f']):>12}"
        for n in resistances:
            line += (f" | {eng(r[n + '_mean'], 'Ω'):>13} {eng(r[n + '_p90'], 'Ω'):>9} "
                     f"{eng(r[n + '_max'], 'Ω'):>9} {pct(r[n + '_rel_p90']):>8}")
        print(line)
    return rows
