"""measureme's Station, plus measurements that stream the SR860s.

``dm.Station`` / ``dm.AsyncStation`` are measureme's Station / AsyncStation (included in this
package as ``dmz_measure.measureme``): ``fp``, ``plot``, ``add_comment``, ``sweep``,
``megasweep``, ``watch``, ``ramp``, ``read``, the live plots and the data files work exactly as
in measureme. Added, in the same style (one run ID, the station's comments, live plots of the
followed parameters, the interrupt button stops at the next point and the call returns
normally so the rest of the cell - e.g. ramping the gates down - still runs):

    s.stream_sweep(Kbg2.volt, Vbg, dwell=0.3)                 # like s.sweep
    s.stream_megasweep(Kbg2.volt, Vbg, Ktg2.volt, Vtg, slow_delay=3, dwell=0.3)
    s.noise_survey(duration=600, f_ref=25)                    # excitation OFF
    s.check_frequencies([13.2943, 17.5623], duration=60)      # excitation ON

They only measure and save. The analysis works from the run ID, without instruments:
dm.load_survey, dm.best_frequencies, dm.expected_noise, dm.load_check, dm.refilter,
dm.compare, dm.step_response.

Followed SR860 X / Y / R / P values come from the lock-in's data stream; every other followed
parameter (gate leakage etc.) is read as usual. Lock-in settings that are changed (reference
frequency, time constant, slope, sync / advanced filter) are printed before and put back
afterwards, also after the interrupt button or an error. No gate is moved, except the swept
parameter of stream_sweep / stream_megasweep (as in s.sweep). The excitation amplitude is
never changed.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm.auto import tqdm

from . import lockin as lk
from .instruments.sr860 import (LockinIO, apply_settings, is_sr860, preserved_settings,
                                wait_for_lock)
from .measureme import db as _db
from .measureme import sweep as _mm
from .measureme.sweep import SweepResult, _interruptible, _sec_to_str
from .plotting import hz
from .sweeps import (COMPONENTS, LiveStreams, LockinSetup, component, segment_record,
                     segments_to_npz)


class _Streaming:
    """Methods added to measureme's stations."""

    # ---- which followed parameters come from the stream -------------------------
    def _stream_plan(self) -> Tuple[Dict[str, LockinIO], List[Tuple]]:
        ios: Dict[str, LockinIO] = {}
        plan = []
        for p, gain in self._params:
            inst = getattr(p, "instrument", None)
            comp = getattr(p, "name", "")
            if inst is not None and is_sr860(inst) and comp in COMPONENTS:
                ios.setdefault(inst.name, LockinIO(inst, inst.name))
                plan.append(("stream", inst.name, comp, gain))
            else:
                plan.append(("read", p, None, gain))
        if not ios:
            raise ValueError("no SR860 X/Y/R/P parameter is followed: use s.fp(SR860.X, gain=...)")
        return ios, plan

    @staticmethod
    def _stream_gains(plan) -> Dict[str, float]:
        gains: Dict[str, float] = {}
        for kind, name, comp, gain in plan:
            if kind == "stream":
                gains.setdefault(name, float(gain))
        return gains

    def _stream_values(self, plan, vals: Dict[str, complex], reads: Dict[int, float]) -> List[float]:
        out = []
        for i, (kind, a, comp, gain) in enumerate(plan):
            if kind == "stream":
                out.append(component(vals[a], comp) / gain)
            else:
                out.append(reads[i] / gain)
        return out

    def _start_reads(self, pool, plan):
        """Read the non-stream parameters in the background.

        AsyncStation: one task per instrument, in parallel (as its _measure does);
        Station: one task reading them in order."""
        idx = [i for i, row in enumerate(plan) if row[0] == "read"]
        if not idx:
            return []
        if hasattr(self, "_ps_by_inst"):
            groups: Dict[Any, List[int]] = {}
            for i in idx:
                groups.setdefault(getattr(plan[i][1], "instrument", i), []).append(i)
            groups_l = list(groups.values())
        else:
            groups_l = [idx]
        return [pool.submit(lambda g=g: {i: plan[i][1]() for i in g}) for g in groups_l]

    # ---- run bookkeeping (as measureme's sweeps) ----------------------------------
    def _begin(self, w, function: str, lead_columns: List[str], **extra) -> None:
        md = w.metadata
        md["comments"] = self._comments
        md["type"] = "1D"
        md["function"] = function
        md["columns"] = lead_columns + self._col_names()
        md["measurement_config"] = self._measurement_config
        md["interrupted"] = False
        md["start_time"] = time.time()
        md.update(extra)

    def _end(self, w, p, interrupted: bool) -> None:
        if interrupted:
            self.logger.warning(f"ID {w.id} INTERRUPTED")
            self._interrupted = True
            w.metadata["interrupted"] = True
        w.metadata["end_time"] = time.time()
        self._finish_plot(w, p)

    def _finish_plot(self, w, p) -> None:
        image = p.send_image()
        if image is not None:
            w.add_blob("plot.png", image)
            from IPython import display

            display.display(display.Image(data=image, format="png"))

    def _log_done(self, w) -> None:
        self.logger.info(f"Completed in {_sec_to_str(w.metadata['end_time'] - w.metadata['start_time'])}")
        self.logger.info(f"Data saved in {w.datapath}")

    @staticmethod
    def _references(ios: Dict[str, LockinIO], snaps: Dict[str, Dict]) -> List[LockinIO]:
        """The followed lock-ins on internal reference: they set the frequency, the others follow."""
        refs = [io for n, io in ios.items() if snaps[n]["RSRC"] == 0]
        if not refs:
            raise RuntimeError("none of the followed SR860s is on internal reference, so the "
                               "reference frequency can't be set")
        return refs

    @staticmethod
    def _flags(raw: Dict[str, Optional[Dict[str, Any]]]) -> Dict[str, Dict[str, int]]:
        return {n: {k: int(r[k]) for k in ("dropped", "overload", "error")}
                for n, r in raw.items() if r is not None}

    @staticmethod
    def _print_streams(raw: Dict[str, Optional[Dict[str, Any]]]) -> None:
        for n, r in raw.items():
            if r is None:
                print(f"  {n}: no data")
                continue
            flags = [f"{r[k]} {k}" for k in ("dropped", "overload", "error") if r[k]]
            print(f"  {n}: {len(r['z'])} samples, {len(r['z']) / r['fs']:.1f} s at {r['fs']:g} Hz, "
                  + (", ".join(flags) if flags else "no drops/flags"))

    # ---- stream while writing a row every `delay` s ---------------------------------
    def _watch_streams(self, w, p, live: LiveStreams, plan, prefix: List[float],
                       duration: float, delay: float, desc: str) -> bool:
        """Every `delay` s for `duration` s, write and live-plot one row: time, `prefix`, then
        the followed parameters (SR860 X/Y/R/P: plain mean of the stream over the last `delay`
        s; the others read at the end of each interval). Returns True if interrupted."""
        n = max(1, int(round(duration / delay)))
        pool = ThreadPoolExecutor(max_workers=max(1, len(plan)))
        try:
            t0 = time.time()
            for k in tqdm(range(1, n + 1), desc=desc):
                t = t0 + k * delay
                while time.time() < t:
                    if self.interrupt_requested:
                        return True
                    time.sleep(min(0.05, max(t - time.time(), 0.0)))
                self._run_run_befores(station=self, writer=w)
                reads: Dict[int, float] = {}
                for f in self._start_reads(pool, plan):
                    reads.update(f.result())
                vals = live.average(t - delay, t, "boxcar")
                data = [time.time()] + list(prefix) + self._stream_values(plan, vals, reads)
                w.add_point(data)
                p.add_point(data)
                self._run_run_afters(station=self, data=data, writer=w)
                if self.interrupt_requested:
                    return True
            return False
        finally:
            pool.shutdown(wait=True)

    # ---- noise survey --------------------------------------------------------------
    @_interruptible
    def noise_survey(self, duration: float = 600.0, f_ref: float = 25.0, tau: float = 100e-6,
                     rate: float = 10e3, delay: float = 1.0) -> SweepResult:
        """Noise survey. Switch the excitation OFF by hand first.

        Streams every followed SR860 for `duration` s with the reference at `f_ref` (Hz) and
        saves the stream. dm.load_survey(run_id) then gives each lock-in's input noise spectrum
        from 0 to 2 f_ref (0 - 50 Hz for f_ref = 25 Hz). Nothing is analysed or plotted here.

        Changed during the survey, and put back afterwards: the reference frequency (FREQ) of
        the followed lock-ins on internal reference (the others follow on external reference),
        and on every followed SR860 the time constant (`tau`, s; <= 100 us keeps the band
        flat), slope 24 dB/oct and sync filter off. The excitation amplitude is never changed.

        Every `delay` s one row is written and live-plotted: time, then each followed
        parameter (SR860 X/Y: mean of the stream over the last `delay` s; all others, e.g.
        gate leakage, read then). The interrupt button stops at the next row; the stream so
        far is saved. rate: stream sample rate (Hz).
        """
        self._check_interrupted()
        ios, plan = self._stream_plan()
        raw: Dict[str, Any] = {}
        interrupted = False
        with _db.Writer(self._basedir) as w, self._plotter as p:
            self.logger.info(f"Starting noise_survey with ID {w.id}")
            self._begin(w, "noise_survey", ["time"], delay=delay, duration=duration)
            p.set_cols(w.metadata["columns"])
            w.update_metadata()
            info: Dict[str, Any] = {"kind": "noise_survey", "f_ref": float(f_ref),
                                    "gains": self._stream_gains(plan)}
            try:
                with preserved_settings(ios.values()) as snaps:
                    refs = self._references(ios, snaps)
                    for io in refs:
                        apply_settings(io, freq=f_ref)
                    for io in ios.values():
                        apply_settings(io, tau=tau, slope_db=24, sync=False)
                    tau_set = next(iter(ios.values())).time_constant()
                    print(f"Survey settings: FREQ {hz(f_ref)} Hz on {', '.join(io.name for io in refs)}; "
                          f"tau {lk._fmt_time(tau_set)}, 24 dB/oct, sync off for {', '.join(ios)}")
                    wait_for_lock(ios.values(), minimum=3.0)
                    info.update(f_det={n: io.detection_frequency() for n, io in ios.items()},
                                tau=tau_set, slope=24, sync=False)
                    live = LiveStreams(ios, rate)
                    live.start()
                    try:
                        interrupted = self._watch_streams(w, p, live, plan, [], duration, delay,
                                                          "noise survey")
                    finally:
                        raw = live.stop()
            finally:
                if raw:
                    self._print_streams(raw)
                    info.update(flags=self._flags(raw),
                                rate=next((r["fs"] for r in raw.values() if r is not None), None))
                    w.add_blob("stream.npz", segments_to_npz([segment_record([], [], [], raw, f_ref)]))
                w.metadata["stream"] = info
            self._end(w, p, interrupted)
        self._log_done(w)
        return SweepResult(self._basedir, w.id, w.metadata, w.datapath)

    # ---- frequency check ----------------------------------------------------------
    @_interruptible
    def check_frequencies(self, frequencies: Sequence[float], duration: float = 60.0,
                          tau: Optional[float] = None, slope: int = 24,
                          advanced: Optional[bool] = None, sync: bool = True,
                          rate: Optional[float] = None, delay: float = 1.0) -> SweepResult:
        """Frequency check. Switch the excitation ON by hand first.

        For each frequency (Hz) in turn: write it as the reference frequency (FREQ) of the
        followed lock-ins on internal reference - this moves the excitation to that frequency,
        its amplitude is never changed - set the filter on every followed SR860, wait for lock
        and settling, then stream for `duration` s. dm.load_check(run_id) gives R = V/I, its
        noise, drift and phase at each frequency. Nothing is analysed here.

        tau      : time constant while streaming (s); None = largest SR860 value <= 1/(30 f)
        slope    : filter slope (dB/oct); sync: sync filter on/off
        advanced : Advanced filter on/off; None = leave it as it is
        rate     : stream rate (Hz); None = 10 / tau
        delay    : every `delay` s a row (time, f_ref, followed parameters) is written and
                   live-plotted, as in noise_survey

        Frequencies and filter settings are put back at the end, also after the interrupt
        button (which stops at the next row) or an error.
        """
        self._check_interrupted()
        ios, plan = self._stream_plan()
        freqs = [float(f) for f in frequencies]
        segments: List[Dict[str, Any]] = []
        done: List[Dict[str, Any]] = []
        interrupted = False
        with _db.Writer(self._basedir) as w, self._plotter as p:
            self.logger.info(f"Starting check_frequencies with ID {w.id}")
            self._begin(w, "check_frequencies", ["time", "f_ref"], param="f_ref",
                        setpoints=freqs, delay=delay, duration=duration)
            p.set_cols(w.metadata["columns"])
            w.update_metadata()
            try:
                with preserved_settings(ios.values()) as snaps:
                    refs = self._references(ios, snaps)
                    first = next(iter(ios.values()))
                    for f in freqs:
                        t = tau if tau is not None else lk.recommended_tau(f)
                        for io in refs:
                            apply_settings(io, freq=f)
                        for io in ios.values():
                            apply_settings(io, tau=t, slope_db=slope, sync=sync, advanced=advanced)
                        f_now = refs[0].ask_float("FREQ?")
                        t = first.time_constant()
                        adv = bool(first.ask_int("ADVFILT?"))
                        print(f"\n{hz(f_now)} Hz (FREQ on {', '.join(io.name for io in refs)}): tau "
                              f"{lk._fmt_time(t)}, {slope} dB/oct{', advanced' if adv else ''}, "
                              f"sync {'on' if sync else 'off'} for {', '.join(ios)}; {duration:g} s")
                        wait_for_lock(ios.values(),
                                      minimum=2.0 + lk.settling_time(t, f_now, slope, advanced=adv, sync=sync))
                        if self.interrupt_requested:
                            interrupted = True
                            break
                        seg_info = {"f_set": f, "f_ref": f_now, "tau": t, "slope": slope,
                                    "advanced": adv, "sync": sync,
                                    "f_det": {n: io.detection_frequency() for n, io in ios.items()}}
                        live = LiveStreams(ios, rate or lk.recommended_stream_rate(t))
                        live.start()
                        try:
                            interrupted = self._watch_streams(w, p, live, plan, [f_now], duration,
                                                              delay, f"{hz(f_now)} Hz")
                        finally:
                            raw = live.stop()
                            self._print_streams(raw)
                            seg_info.update(flags=self._flags(raw),
                                            rate=next((r["fs"] for r in raw.values() if r is not None), None))
                            segments.append(segment_record([], [], [], raw, f_now))
                            done.append(seg_info)
                        if interrupted:
                            break
            finally:
                if segments:
                    w.add_blob("stream.npz", segments_to_npz(segments))
                w.metadata["stream"] = {"kind": "check_frequencies", "gains": self._stream_gains(plan),
                                        "segments": done}
            self._end(w, p, interrupted)
        self._log_done(w)
        return SweepResult(self._basedir, w.id, w.metadata, w.datapath)

    # ---- streamed sweeps -----------------------------------------------------------
    def _stream_metadata(self, w, plan, setup: LockinSetup, settle, dwell, window) -> None:
        cols = {}
        for (kind, a, comp, gain), name in zip(plan, self._col_names()):
            if kind == "stream":
                cols[name] = [a, comp, gain]
        w.metadata["stream"] = dict(setup.describe(settle, dwell, window), columns=cols)

    def _stream_line(self, w, p, live_args, plan, prefix, param, setpoints, settle, dwell, window,
                     row0: int, f_ref: float, desc_kw: Dict[str, Any]):
        """One line: start streams, step `param` through `setpoints`, average, stop.

        Returns (segment record, number of points written, interrupted?)."""
        live = LiveStreams(*live_args)
        live.start()
        t0s, t1s, rows = [], [], []
        interrupted = False
        pool = ThreadPoolExecutor(max_workers=max(1, len(plan)))
        try:
            for k, sp in enumerate(tqdm(setpoints, **desc_kw)):
                a = time.time()
                param(sp)
                b = time.time()
                wait = b + settle - time.time()
                if wait > 0:
                    time.sleep(wait)
                self._run_run_befores(station=self, writer=w)
                futures = self._start_reads(pool, plan)     # other parameters, during the dwell
                wait = b + settle + dwell - time.time()
                if wait > 0:
                    time.sleep(wait)
                reads: Dict[int, float] = {}
                for f in futures:
                    reads.update(f.result())
                vals = live.average(b + settle, b + settle + dwell, window)
                data = [time.time()] + prefix + [sp] + self._stream_values(plan, vals, reads)
                w.add_point(data)
                p.add_point(data)
                t0s.append(a)
                t1s.append(b)
                rows.append(row0 + k)
                if self.interrupt_requested:
                    interrupted = True
                    break
                self._run_run_afters(station=self, data=data, writer=w)
        finally:
            pool.shutdown(wait=True)
            raw = live.stop()
        return segment_record(t0s, t1s, rows, raw, f_ref), len(rows), interrupted

    @_interruptible
    def stream_sweep(self, param, setpoints, dwell: float = 0.3, settle: Optional[float] = None,
                     window: str = "hann", *, tau: Optional[float] = None, slope: int = 24,
                     advanced: Optional[bool] = None, sync: bool = True,
                     rate: Optional[float] = None) -> SweepResult:
        """Like ``sweep(param, setpoints, delay)``, but the lock-ins stream and each point is the
        `window`-weighted average of the `dwell` seconds that start `settle` s after the set.

        settle   : None = lock-in settling (7-10 tau) + one reference period; add more if the
                   gate line or the device is slower
        window   : 'hann' (rejects lines away from f much better) or 'boxcar' (plain mean)
        tau      : lock-in time constant during the sweep; None = largest value <= 1/(30 f)
        slope, advanced, sync : lock-in filter during the sweep (advanced None = as set)
        rate     : stream rate; None = 10 / tau
        The lock-in settings are restored afterwards; the excitation is never changed. The
        stream is saved with the run, so dm.refilter(run_id, ...) can redo the averaging.
        """
        self._check_interrupted()
        ios, plan = self._stream_plan()
        with _db.Writer(self._basedir) as w, self._plotter as p:
            self.logger.info(f"Starting stream_sweep with ID {w.id}")
            self._begin(w, "stream_sweep", ["time", param.full_name], param=param.full_name,
                        setpoints=list(setpoints))
            p.set_cols(w.metadata["columns"])
            with LockinSetup(ios, tau, slope, advanced, sync, rate) as setup:
                settle_s = setup.settle if settle is None else settle
                self._stream_metadata(w, plan, setup, settle_s, dwell, window)
                w.metadata["delay"] = settle_s + dwell
                w.update_metadata()
                self.logger.info(f"tau {setup.tau:g} s, settle {settle_s * 1e3:.0f} ms, dwell {dwell:g} s "
                                 f"{window}; minimum duration {_sec_to_str(len(setpoints) * (settle_s + dwell))}")
                param(setpoints[0])
                wait_for_lock(ios.values(), minimum=max(1.0, 3 * settle_s))
                seg, _, interrupted = self._stream_line(
                    w, p, (ios, setup.rate), plan, [], param, setpoints, settle_s, dwell, window,
                    0, setup.f, {})
            w.add_blob("stream.npz", segments_to_npz([seg]))
            self._end(w, p, interrupted)
        self._log_done(w)
        return SweepResult(self._basedir, w.id, w.metadata, w.datapath)

    @_interruptible
    def stream_megasweep(self, slow_param, slow_v, fast_param, fast_v, slow_delay: float = 0.0,
                         dwell: float = 0.3, settle: Optional[float] = None, window: str = "hann",
                         init_delay: bool = True, *, tau: Optional[float] = None, slope: int = 24,
                         advanced: Optional[bool] = None, sync: bool = True,
                         rate: Optional[float] = None) -> SweepResult:
        """Like ``megasweep(slow_param, slow_v, fast_param, fast_v, slow_delay, fast_delay)``,
        with each fast line streamed (see stream_sweep for dwell / settle / window / filter)."""
        self._check_interrupted()
        ios, plan = self._stream_plan()
        segments = []
        with _db.Writer(self._basedir) as w, self._plotter as p:
            self.logger.info(f"Starting stream_megasweep with ID {w.id}")
            self._begin(w, "stream_megasweep", ["time", slow_param.full_name, fast_param.full_name],
                        slow_delay=slow_delay, slow_param=slow_param.full_name,
                        fast_param=fast_param.full_name, slow_setpoints=list(slow_v),
                        fast_setpoints=list(fast_v))
            w.metadata["type"] = "2D"
            p.set_cols(w.metadata["columns"])
            interrupted = False
            with LockinSetup(ios, tau, slope, advanced, sync, rate) as setup:
                settle_s = setup.settle if settle is None else settle
                self._stream_metadata(w, plan, setup, settle_s, dwell, window)
                w.metadata["fast_delay"] = settle_s + dwell
                w.update_metadata()
                n_done = 0
                for i, ov in enumerate(tqdm(slow_v, position=0)):
                    slow_param(ov)
                    fast_param(fast_v[0])
                    if init_delay or i > 0:
                        self._interruptable_sleep(slow_delay)
                    if i == 0:
                        wait_for_lock(ios.values(), minimum=max(0.0, 3 * settle_s))
                    if self.interrupt_requested:
                        interrupted = True
                        break
                    seg, n, interrupted = self._stream_line(
                        w, p, (ios, setup.rate), plan, [ov], fast_param, fast_v, settle_s, dwell,
                        window, n_done, setup.f, dict(position=1, leave=False))
                    segments.append(seg)
                    n_done += n
                    if interrupted:
                        break
            w.add_blob("stream.npz", segments_to_npz(segments))
            self._end(w, p, interrupted)
        self._log_done(w)
        return SweepResult(self._basedir, w.id, w.metadata, w.datapath)


class Station(_Streaming, _mm.Station):
    """measureme's Station + stream_sweep, stream_megasweep, noise_survey, check_frequencies."""


class AsyncStation(_Streaming, _mm.AsyncStation):
    """measureme's AsyncStation (parallel reads per instrument) + the streaming methods."""
