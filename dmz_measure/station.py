"""measureme's Station, plus streaming.

``dm.Station`` / ``dm.AsyncStation`` build on measureme's Station / AsyncStation, which are
included in this package (``dmz_measure.measureme``, copied from measureme; nothing imports
an installed ``sweep``). Everything works exactly as in measureme: ``fp``, ``plot``,
``add_comment``, ``sweep``, ``megasweep``, ``watch``, ``ramp``, ``read``, the live plots and
the data files.

Added:

    s.stream_sweep(Kbg2.volt, Vbg, dwell=0.3)                         # like s.sweep
    s.stream_megasweep(Kbg2.volt, Vbg, Ktg2.volt, Vtg, slow_delay=3, dwell=0.3)  # like s.megasweep
    survey = s.noise_survey(current='SR860_A', f_ref=100, duration=600)
    check = s.check_frequencies([17.5, 23.85], current='SR860_A')

Streamed sweeps write the same columns as the normal ones: followed X / Y / R / P
parameters of SR860s come from the stream (averaged over `dwell` after `settle`, Hann
weighted by default), every other followed parameter is read as usual at each point.
The run also stores the stream itself (``stream.npz``), so ``dm.refilter(run_id, ...)``
can redo the averaging later.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from .measureme import db as _db
from .measureme import sweep as _mm
from .measureme.sweep import SweepResult, _interruptible, _sec_to_str
from tqdm.auto import tqdm

from . import check as _check
from . import survey as _survey
from .instruments.sr860 import LockinIO, is_sr860, wait_for_lock
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

    def _stream_values(self, plan, vals: Dict[str, complex], reads: Dict[int, float]) -> List[float]:
        out = []
        for i, (kind, a, comp, gain) in enumerate(plan):
            if kind == "stream":
                out.append(component(vals[a], comp) / gain)
            else:
                out.append(reads[i] / gain)
        return out

    def _start_reads(self, pool, plan):
        """Read the non-stream parameters in the background (during the dwell).

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

    # ---- streamed 1D -----------------------------------------------------------
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
        The lock-in settings are restored afterwards; the excitation is never changed.
        """
        self._check_interrupted()
        ios, plan = self._stream_plan()
        with _db.Writer(self._basedir) as w, self._plotter as p:
            self.logger.info(f"Starting stream_sweep with ID {w.id}")
            w.metadata["comments"] = self._comments
            w.metadata["type"] = "1D"
            w.metadata["function"] = "stream_sweep"
            w.metadata["param"] = param.full_name
            w.metadata["columns"] = ["time", param.full_name] + self._col_names()
            w.metadata["measurement_config"] = self._measurement_config
            w.metadata["setpoints"] = list(setpoints)
            w.metadata["interrupted"] = False
            w.metadata["start_time"] = time.time()
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
            if interrupted:
                self.logger.warning(f"ID {w.id} INTERRUPTED")
                self._interrupted = True
                w.metadata["interrupted"] = True
            w.add_blob("stream.npz", segments_to_npz([seg]))
            w.metadata["end_time"] = time.time()
            self._finish_plot(w, p)
        self._log_done(w)
        return SweepResult(self._basedir, w.id, w.metadata, w.datapath)

    # ---- streamed 2D -----------------------------------------------------------
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
            w.metadata["comments"] = self._comments
            w.metadata["type"] = "2D"
            w.metadata["function"] = "stream_megasweep"
            w.metadata["slow_delay"] = slow_delay
            w.metadata["slow_param"] = slow_param.full_name
            w.metadata["fast_param"] = fast_param.full_name
            w.metadata["columns"] = ["time", slow_param.full_name, fast_param.full_name] + self._col_names()
            w.metadata["measurement_config"] = self._measurement_config
            w.metadata["slow_setpoints"] = list(slow_v)
            w.metadata["fast_setpoints"] = list(fast_v)
            w.metadata["interrupted"] = False
            w.metadata["start_time"] = time.time()
            p.set_cols(w.metadata["columns"])
            with LockinSetup(ios, tau, slope, advanced, sync, rate) as setup:
                settle_s = setup.settle if settle is None else settle
                self._stream_metadata(w, plan, setup, settle_s, dwell, window)
                w.metadata["fast_delay"] = settle_s + dwell
                w.update_metadata()
                n_done = 0
                interrupted = False
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
            if interrupted:
                self.logger.warning(f"ID {w.id} INTERRUPTED")
                self._interrupted = True
                w.metadata["interrupted"] = True
            w.add_blob("stream.npz", segments_to_npz(segments))
            w.metadata["end_time"] = time.time()
            self._finish_plot(w, p)
        self._log_done(w)
        return SweepResult(self._basedir, w.id, w.metadata, w.datapath)

    def _finish_plot(self, w, p) -> None:
        image = p.send_image()
        if image is not None:
            w.add_blob("plot.png", image)
            from IPython import display

            display.display(display.Image(data=image, format="png"))

    def _log_done(self, w) -> None:
        self.logger.info(f"Completed in {_sec_to_str(w.metadata['end_time'] - w.metadata['start_time'])}")
        self.logger.info(f"Data saved in {w.datapath}")

    # ---- timing -------------------------------------------------------------------
    def timing(self, param, setpoints, n: int = 20, delay: Optional[float] = None,
               settle: Optional[float] = None, dwell: Optional[float] = None):
        """Where the time of a sweep point goes (gate set, each read) and the real total.

        Sets the first n setpoints and reads the followed parameters, then returns `param`
        to where it was. See dmz_measure.timing.measure_timing."""
        from .timing import measure_timing

        return measure_timing(self, param, setpoints, n, delay, settle, dwell)

    def step_response(self, param, v1: float, v2: float, n: int = 10, hold: float = 1.0,
                      tol: float = 0.01, **filter_kw):
        """Step `param` between v1 and v2 while streaming; how long until it has settled.

        The gate moves as in a sweep (its qcodes step / inter_delay, fast_ramp if on), so
        pick |v2 - v1| like a sweep point. The result's .settle is the value to pass as
        stream_sweep(settle=...). Lock-in settings are the streaming ones (tau ~ 1/(30 f),
        sync on) and are restored after. See dmz_measure.timing.step_response."""
        from .timing import step_response

        return step_response(self, param, v1, v2, n, hold, tol, **filter_kw)

    # ---- choosing the frequency --------------------------------------------------
    def _lockin_gains(self) -> Dict[str, float]:
        gains: Dict[str, float] = {}
        for p, gain in self._params:
            inst = getattr(p, "instrument", None)
            if inst is not None and is_sr860(inst) and getattr(p, "name", "") in ("X", "Y", "R"):
                gains.setdefault(inst.name, gain)
        return gains

    def noise_survey(self, current: Optional[str] = None, **kw) -> "_survey.Survey":
        """dm.noise_survey for the SR860s in this station's measurement_config, with the gains
        given in s.fp(...). Saved in this station's data folder with its comments."""
        return _survey.noise_survey(self._measurement_config, self._lockin_gains(), current,
                                    basedir=self._basedir, comment="; ".join(self._comments), **kw)

    def check_frequencies(self, frequencies: Sequence[float], current: Optional[str] = None,
                          **kw) -> "_check.Check":
        """dm.check_frequencies for this station's lock-ins (see there for the options)."""
        return _check.check_frequencies(self._measurement_config, frequencies, self._lockin_gains(),
                                        current, basedir=self._basedir,
                                        comment="; ".join(self._comments), **kw)


class Station(_Streaming, _mm.Station):
    """measureme's Station + stream_sweep, stream_megasweep, noise_survey, check_frequencies."""


class AsyncStation(_Streaming, _mm.AsyncStation):
    """measureme's AsyncStation (parallel reads per instrument) + the streaming methods."""
