"""High-level procedures for choosing the excitation frequency.

Step 1 - :func:`noise_survey`
    Reference at f_ref (default 100 Hz), tau <= 100 us, sync off, excitation
    off, stream X/Y from every lock-in at once for a few minutes. The complex FFT
    gives each channel's input noise spectrum over 0 - 2 f_ref.

Step 2 - :func:`candidate_check`
    For a few candidate frequencies: real settings (tau ~ 1/(30 f), sync on),
    excitation on, a representative gate point, stream for a few minutes and
    measure the scatter of dwell-length averages (= the noise of one map pixel),
    the Allan deviation and the phase.

Both snapshot every lock-in's settings first and restore them at the end, also
on errors or a KeyboardInterrupt.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from . import lockin as lk
from . import noise as nz
from .instruments.sr860 import LockinIO, apply_settings, as_io, preserved_settings, wait_for_lock
from .instruments.sr860_stream import DEFAULT_PORT, SR860Stream, StreamData, acquire_many
from .io import RunWriter


@dataclass
class Channel:
    """One lock-in to stream.

    gain : preamp gain to divide out (100 for a x100 voltage preamp, 1e6 for a
           1e6 V/A current preamp). Same meaning as the gain in ``s.fp(SR860.X, gain=...)``.
    unit : input-referred unit after dividing by gain ('V' or 'A')
    """

    lockin: Any
    gain: float = 1.0
    unit: str = "V"
    label: str = ""


def _channels(channels: Mapping[str, Union[Channel, Tuple, Any]]) -> Dict[str, Channel]:
    out = {}
    for name, ch in channels.items():
        if isinstance(ch, Channel):
            out[name] = ch
        elif isinstance(ch, tuple):
            out[name] = Channel(*ch)
        else:
            out[name] = Channel(ch)
    return out


# --------------------------------------------------------------------------
# Step 1
# --------------------------------------------------------------------------

@dataclass
class SurveyResult:
    spectra: Dict[str, nz.NoiseSpectrum]
    lines: Dict[str, List[Dict[str, Any]]]
    streams: Dict[str, StreamData]
    settings: Dict[str, Any]
    run_id: Optional[int] = None
    run_dir: Optional[str] = None

    def rank(self, filt: Optional[lk.MeasurementFilter] = None, **kw) -> nz.FrequencyRanking:
        """Shortcut for :func:`dmz_measure.noise.rank_frequencies` on these spectra."""
        return nz.rank_frequencies(self.spectra, filt, lines=self.lines, **kw)


def noise_survey(channels: Mapping[str, Any], master: Any, f_ref: float = 100.0,
                 tau: float = 100e-6, duration: float = 300.0, target_rate: float = 40e3,
                 excitation_off: bool = True, resolution: float = 0.05, mains: float = 60.0,
                 settle: float = 3.0, base_port: int = DEFAULT_PORT,
                 basedir: Optional[str] = None, comment: str = "",
                 measurement_config: Optional[Mapping[str, Any]] = None,
                 save_timeseries: bool = True, line_threshold: float = 10.0,
                 fundamentals: Optional[Mapping[str, float]] = None) -> SurveyResult:
    """Step 1: broadband noise survey of all channels from one streamed record.

    channels : {'SR860_B': Channel(SR860_B, gain=100), ...} - every lock-in on the
               shared reference you want spectra for (the master may be one of them)
    master   : the lock-in that owns the reference and the excitation (internal ref)
    f_ref    : survey reference frequency; the spectra cover 0 - 2 f_ref
    tau      : lock-in time constant during the survey (<= 100 us keeps the band flat)
    duration : seconds of streaming; resolution * duration >~ 10 gives a decent average
    excitation_off : set the master's sine out to 1 nV during the survey
    basedir  : data folder; None = don't save
    """
    chans = _channels(channels)
    ios = {name: as_io(ch.lockin, name) for name, ch in chans.items()}
    master_io = ios.get(getattr(master, "name", None)) or as_io(master)
    all_ios = list(ios.values()) + ([master_io] if master_io not in ios.values() else [])

    settings: Dict[str, Any] = {"f_ref_requested": f_ref, "tau_requested": tau,
                                "excitation_off": excitation_off, "duration": duration,
                                "target_rate": target_rate}
    with preserved_settings(all_ios) as snaps:
        settings["original_settings"] = snaps
        if master_io.reference_source() == "INT":
            apply_settings(master_io, freq=f_ref)
        else:
            warnings.warn(f"{master_io.name} is not on internal reference; surveying at its "
                          "current frequency")
        if excitation_off:
            apply_settings(master_io, amplitude=1e-9)
        for io in ios.values():
            apply_settings(io, tau=tau, slope_db=24, sync=False)
        wait_for_lock(all_ios, minimum=max(settle, 20 * tau))
        per_ch = {}
        for name, io in ios.items():
            per_ch[name] = {"f_det": io.detection_frequency(), "tau": io.time_constant(),
                            "rsrc": io.reference_source()}
        settings["channels"] = per_ch
        streams = [SR860Stream(io, port=base_port + i) for i, io in enumerate(ios.values())]
        for st in streams:
            st.configure(rate=target_rate)
        print(f"Streaming {len(streams)} lock-in(s) for {duration:g} s at "
              f"{streams[0].rate:g} Hz (f_ref = {per_ch[streams[0].name]['f_det']:.4f} Hz) ...")
        data = acquire_many(streams, duration)

    spectra, lines = {}, {}
    for name, ch in chans.items():
        d = data[name]
        f_det = per_ch[name]["f_det"]
        if d.n_overload:
            warnings.warn(f"{name}: {d.n_overload} packets flagged OVERLOAD - input range too small?")
        spec = nz.input_spectrum(d.z, d.fs, f_det, gain=ch.gain, unit=ch.unit,
                                 resolution=resolution, tau=per_ch[name]["tau"], mains=mains,
                                 name=name)
        spectra[name] = spec
        lines[name] = nz.find_lines(spec, threshold=line_threshold,
                                    fundamentals=fundamentals or {"mains": mains})
    res = SurveyResult(spectra, lines, data, settings)
    if basedir is not None:
        _save_survey(res, chans, basedir, comment, measurement_config)
    return res


def _save_survey(res: SurveyResult, chans: Dict[str, Channel], basedir: str, comment: str,
                 measurement_config) -> None:
    from . import plotting

    names = list(res.spectra)
    f0 = res.spectra[names[0]].f
    cols = ["f_Hz"] + [f"{n}_asd_{res.spectra[n].unit}_per_rtHz" for n in names]
    table = [f0] + [np.interp(f0, res.spectra[n].f, res.spectra[n].asd) for n in names]
    with RunWriter(basedir, "noise_survey", comment, measurement_config) as w:
        md = w.metadata
        md["settings"] = res.settings
        md["gains"] = {n: chans[n].gain for n in names}
        md["units"] = {n: chans[n].unit for n in names}
        md["spectra"] = {n: {"f_ref": s.f_ref, "df": s.df, "n_segments": s.n_segments,
                             "floor": s.floor(), "orientation": s.meta.get("orientation")}
                         for n, s in res.spectra.items()}
        md["streams"] = {n: {"fs": d.fs, "samples": len(d), "t0": d.t0, "packets": d.n_packets,
                             "dropped": d.n_dropped, "overload": d.n_overload, "error": d.n_error}
                         for n, d in res.streams.items()}
        md["lines"] = res.lines
        w.write_table(cols, np.column_stack(table))
        arrays = {}
        for n, d in res.streams.items():
            # keep enough bandwidth for the 0 - 2 f_ref band: decimate to >= 4 f_ref
            f_ref = res.spectra[n].f_ref
            q = int(2 ** max(0, np.floor(np.log2(d.fs / (4 * f_ref)))))
            arrays[f"{n}_z"] = nz.decimate(d.z, q).astype(np.complex64)
            arrays[f"{n}_fs"] = np.array(d.fs / q)
            arrays[f"{n}_f_ref"] = np.array(f_ref)
        if arrays:
            w.save_arrays("timeseries.npz", **arrays)
        fig = plotting.plot_survey(res)
        w.save_figure(fig, "plot.png")
        import matplotlib.pyplot as plt

        plt.close(fig)
        res.run_id, res.run_dir = w.id, w.dir
    print(f"Saved run {res.run_id} -> {res.run_dir}")


# --------------------------------------------------------------------------
# Step 2
# --------------------------------------------------------------------------

@dataclass
class CheckResult:
    rows: List[Dict[str, Any]]
    allan: Dict[float, Dict[str, Tuple[np.ndarray, np.ndarray]]]
    streams: Dict[float, Dict[str, StreamData]] = field(default_factory=dict)
    run_id: Optional[int] = None
    run_dir: Optional[str] = None


def candidate_check(channels: Mapping[str, Any], master: Any, frequencies: Sequence[float],
                    duration: float = 120.0, dwell: float = 0.3, tau: Optional[float] = None,
                    sync: bool = True, slope_db: int = 24, target_rate: Optional[float] = None,
                    settle: Optional[float] = None, current: Optional[str] = None,
                    base_port: int = DEFAULT_PORT, basedir: Optional[str] = None,
                    comment: str = "", measurement_config: Optional[Mapping[str, Any]] = None,
                    keep_streams: bool = False) -> CheckResult:
    """Step 2: measure each candidate frequency at real settings, excitation ON.

    frequencies : candidates (Hz), e.g. the top few from SurveyResult.rank()
    duration    : seconds streamed per candidate (several minutes is better)
    dwell       : averaging time of one map pixel (s); pixel noise = scatter of these averages
    tau         : lock-in time constant; None = largest SR860 value <= 1/(30 f) (short-tau recipe)
    sync        : sync filter on (removes the 2f ripple in ~1 period)
    current     : name of the channel that measures the current; then R = V/I is evaluated too
    """
    chans = _channels(channels)
    ios = {name: as_io(ch.lockin, name) for name, ch in chans.items()}
    master_io = ios.get(getattr(master, "name", None)) or as_io(master)
    all_ios = list(ios.values()) + ([master_io] if master_io not in ios.values() else [])
    if master_io.reference_source() != "INT":
        raise RuntimeError(f"{master_io.name} must be on internal reference to set the frequency")

    rows: List[Dict[str, Any]] = []
    allan: Dict[float, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    kept: Dict[float, Dict[str, StreamData]] = {}
    with preserved_settings(all_ios) as snaps:
        for f in frequencies:
            t_c = tau if tau is not None else lk.recommended_tau(f)
            apply_settings(master_io, freq=f)
            for io in ios.values():
                apply_settings(io, tau=t_c, slope_db=slope_db, sync=sync)
            t_c = list(ios.values())[0].time_constant()
            wait_s = settle if settle is not None else 2.0 + lk.settling_time(t_c, f, slope_db, sync=sync)
            wait_for_lock(all_ios, minimum=wait_s)
            streams = [SR860Stream(io, port=base_port + i) for i, io in enumerate(ios.values())]
            rate = target_rate or lk.recommended_stream_rate(t_c)
            for st in streams:
                st.configure(rate=rate)
            print(f"\n{f:.4f} Hz: tau = {lk._fmt_time(t_c)}, sync {'on' if sync else 'off'}, "
                  f"stream {streams[0].rate:g} Hz, {duration:g} s")
            data = acquire_many(streams, duration)
            if keep_streams:
                kept[f] = data
            row: Dict[str, Any] = {"f": f, "tau": t_c, "sync": sync, "dwell": dwell}
            allan[f] = {}
            discard = 2 / f  # sync filter warm-up
            for name, ch in chans.items():
                d = data[name]
                z = d.z / ch.gain
                pn = nz.pixel_noise(z, d.fs, dwell, discard=discard)
                mean = pn["mean_x"] + 1j * pn["mean_y"]
                row[f"{name}_X"] = pn["mean_x"]
                row[f"{name}_Y"] = pn["mean_y"]
                row[f"{name}_phase_deg"] = float(np.degrees(np.angle(mean)))
                row[f"{name}_sigma"] = pn["sigma_x"]
                row[f"{name}_rel_noise"] = pn["sigma_x"] / abs(mean) if abs(mean) > 0 else np.nan
                row[f"{name}_flags"] = _flags(d)
                allan[f][name] = nz.allan_deviation(z.real[int(discard * d.fs):], d.fs)
            if current is not None:
                _add_resistance(row, data, chans, current, dwell, discard)
            rows.append(row)
    res = CheckResult(rows, allan, kept)
    if basedir is not None:
        _save_check(res, list(chans), basedir, comment, measurement_config, snaps, current)
    return res


def _flags(d: StreamData) -> str:
    out = []
    if d.n_dropped:
        out.append(f"{d.n_dropped} dropped")
    if d.n_overload:
        out.append("OVERLOAD")
    if d.n_error:
        out.append("UNLOCK/SYNC")
    return ", ".join(out)


def _add_resistance(row, data, chans, current, dwell, discard) -> None:
    """R = V/I per pixel; streams from different lock-ins are aligned by arrival time."""
    di = data[current]
    zi = di.z / chans[current].gain
    for name, ch in chans.items():
        if name == current:
            continue
        dv = data[name]
        zv = dv.z / ch.gain
        if dv.fs != di.fs:
            continue
        shift = int(round((dv.t0 - di.t0) * di.fs)) if np.isfinite(dv.t0 - di.t0) else 0
        if shift > 0:
            zi_al, zv_al = zi[shift:], zv
        else:
            zi_al, zv_al = zi, zv[-shift:]
        n = min(len(zi_al), len(zv_al))
        k = max(1, int(round(dwell * di.fs)))
        s0 = int(discard * di.fs)
        vb = nz.block_means(zv_al[s0:n], k)
        ib = nz.block_means(zi_al[s0:n], k)
        m = min(len(vb), len(ib))
        r = vb[:m] / ib[:m]
        rm = r.mean()
        row[f"{name}_R"] = float(rm.real)
        row[f"{name}_R_phase_deg"] = float(np.degrees(np.angle(rm)))
        row[f"{name}_R_sigma"] = float(r.real.std(ddof=1))
        row[f"{name}_R_rel_noise"] = float(r.real.std(ddof=1) / abs(rm.real)) if rm.real else np.nan


def _save_check(res: CheckResult, names, basedir, comment, measurement_config, snaps, current):
    from . import plotting

    num_cols = [k for k in res.rows[0] if not k.endswith("_flags") and k != "sync"]
    table = np.array([[float(r.get(k, np.nan)) for k in num_cols] for r in res.rows])
    with RunWriter(basedir, "frequency_check", comment, measurement_config) as w:
        w.metadata["rows"] = res.rows
        w.metadata["original_settings"] = snaps
        w.metadata["current_channel"] = current
        w.write_table(num_cols, table)
        arrays = {}
        for f, per in res.allan.items():
            for n, (t, a) in per.items():
                arrays[f"{n}_{f:.4f}Hz_tau"] = t
                arrays[f"{n}_{f:.4f}Hz_adev"] = a
        w.save_arrays("allan.npz", **arrays)
        fig = plotting.plot_check(res)
        w.save_figure(fig, "plot.png")
        import matplotlib.pyplot as plt

        plt.close(fig)
        res.run_id, res.run_dir = w.id, w.dir
    print(f"Saved run {res.run_id} -> {res.run_dir}")
