"""Where the time of a sweep point goes, and how long a gate step really takes to settle.

    s.timing(Kbg2.volt, Vbg, n=20, delay=0.6, dwell=0.3)   # per-point time budget, total estimate
    st = s.step_response(Kbg2.volt, 0.0, 0.2)              # measured settle after a gate step
    with dm.gate_steps(Kbg2.volt, max_step=0.01):          # one command per sweep point
        s.stream_sweep(Kbg2.volt, Vbg, settle=st.settle)

qcodes ramps a parameter that has `step` / `inter_delay`: a change is cut into chunks of
at most `step`, each chunk is one instrument command (for the Keithley 2450 with
block_during_ramp also a *OPC? wait), and before each command qcodes waits until
`inter_delay` has passed since the previous one. One point therefore costs about
    chunks x round trip + (chunks - 1) x inter_delay
which is often much more than the `delay` you asked for.
"""
from __future__ import annotations

import math
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import lockin as lk
from .plotting import eng


def _chunks(param, dv: float) -> int:
    step = getattr(param, "step", None)
    if not step or step <= 0:
        return 1
    return max(1, int(math.ceil(abs(dv) / step - 1e-9)))


@contextmanager
def gate_steps(*params, max_step: float):
    """Temporarily let `params` move up to `max_step` per command.

    The qcodes `inter_delay` is scaled by the same factor, so a long ramp (e.g. to the
    start of a sweep) is never faster than step / inter_delay allows now; inside a sweep
    whose increment is <= max_step every point is then a single command. Restored on exit.
    """
    saved = []
    for p in params:
        step, delay = getattr(p, "step", None), getattr(p, "inter_delay", 0.0) or 0.0
        saved.append((p, step, delay))
        if step and max_step > step:
            p.step = max_step
            p.inter_delay = delay * max_step / step
    try:
        yield
    finally:
        for p, step, delay in saved:
            p.step = step
            p.inter_delay = delay


class fast_ramp:
    """Same ramp, one confirmation: keep qcodes' `step` / `inter_delay`, drop the per-chunk wait.

        dm.fast_ramp(Kbg2.volt, Ktg2.volt)          # from now on (call .undo() to revert)
        with dm.fast_ramp(Kbg2.volt):               # or only inside the block
            s.sweep(Kbg2.volt, Vbg, delay=0.6)

    With `block_during_ramp(True)` (Keithley 2450 in qcodes), every chunk of a ramp is a
    write followed by a *OPC? round trip. Here block_during_ramp is switched off, so each
    chunk is only a write, and a single *OPC? is asked once the whole ramp has been sent:
    the set call still returns only when the instrument has applied the final value.
    Unchanged: the chunk size (`step`, e.g. 5 mV) and the minimum time between chunks
    (`inter_delay`, timed from when the previous command was accepted), so the ramp is never
    faster than before - it just stops paying a round trip per chunk.
    """

    def __init__(self, *params):
        self._saved = []
        for p in params:
            inst = getattr(p, "instrument", None)
            root = getattr(p, "root_instrument", None) or inst
            blk = getattr(inst, "block_during_ramp", None)
            if blk is None or root is None or not hasattr(root, "ask"):
                raise ValueError(f"{getattr(p, 'full_name', p)}: no block_during_ramp to switch "
                                 "(fast_ramp is for Keithley 2450 source levels)")
            was = blk()
            orig_set = p.set

            def set_then_confirm(value, _orig=orig_set, _root=root, **kw):
                _orig(value, **kw)
                _root.ask("*OPC?")

            blk(False)
            p.set = set_then_confirm
            self._saved.append((p, blk, was, orig_set))

    def undo(self) -> None:
        for p, blk, was, orig_set in self._saved:
            p.set = orig_set
            blk(was)
        self._saved = []

    def __enter__(self) -> "fast_ramp":
        return self

    def __exit__(self, *exc) -> None:
        self.undo()


def _stats(x: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    return {"median": float(np.median(x)), "max": float(np.max(x)), "mean": float(np.mean(x))}


def measure_timing(station, param, setpoints, n: int = 20, delay: Optional[float] = None,
                   settle: Optional[float] = None, dwell: Optional[float] = None) -> Dict[str, Any]:
    """Time n points of a sweep of `param` (first n setpoints): set, each read, all reads.

    Prints a per-point time budget and the estimated total for the full `setpoints` with
    s.sweep(delay) and s.stream_sweep(settle, dwell). The parameter is returned to its
    starting value afterwards. Nothing is saved.
    """
    setpoints = np.asarray(setpoints, dtype=float)
    pts = setpoints[: max(2, min(n, len(setpoints)))]
    start = param()
    param(pts[0])
    t_set, chunks = [], []
    t_read: Dict[str, List[float]] = {p.full_name: [] for p, _ in station._params}
    t_all = []
    prev = pts[0]
    for v in pts[1:]:
        chunks.append(_chunks(param, v - prev))
        prev = v
        a = time.perf_counter()
        param(v)
        t_set.append(time.perf_counter() - a)
        for p, _ in station._params:
            a = time.perf_counter()
            p()
            t_read[p.full_name].append(time.perf_counter() - a)
        a = time.perf_counter()
        station._measure()
        t_all.append(time.perf_counter() - a)
    param(start)

    inter = getattr(param, "inter_delay", 0.0) or 0.0
    k = float(np.median(chunks))
    set_med = float(np.median(t_set))
    rtt = max(set_med - (k - 1) * inter, 0.0) / k
    read_all = float(np.median(t_all))
    N = len(setpoints)
    out = {"set": _stats(t_set), "chunks": k, "round_trip": rtt, "inter_delay": inter,
           "reads": {k_: _stats(v) for k_, v in t_read.items()}, "read_all": _stats(t_all)}

    print(f"Timing of {len(t_set)} points of {param.full_name} "
          f"(increment {abs(np.median(np.diff(pts))):.4g}; qcodes step {getattr(param, 'step', None)}, "
          f"inter_delay {inter * 1e3:.0f} ms)")
    print(f"  set   {set_med * 1e3:7.1f} ms median ({max(t_set) * 1e3:.1f} max) = {k:g} command(s) "
          f"x {rtt * 1e3:.1f} ms" + (f" + {k - 1:g} x {inter * 1e3:.0f} ms inter_delay" if k > 1 else ""))
    for name, v in t_read.items():
        print(f"  read  {np.median(v) * 1e3:7.1f} ms  {name}")
    print(f"  all reads together (station._measure): {read_all * 1e3:.1f} ms")
    if delay is not None:
        per = set_med + delay + read_all
        print(f"s.sweep(delay={delay:g}): {per * 1e3:.0f} ms/point -> {N} points in {_hms(N * per)} "
              f"(measureme's estimate, delay only: {_hms(N * delay)})")
        out["sweep_per_point"] = per
    if dwell is not None:
        s_ = settle if settle is not None else 0.0
        per = set_med + s_ + dwell
        print(f"s.stream_sweep(settle={s_:g}, dwell={dwell:g}): {per * 1e3:.0f} ms/point -> {N} points in "
              f"{_hms(N * per)} (reads overlap the dwell"
              + (")" if read_all < dwell else f"; they take longer than the dwell, add {read_all - dwell:.2f} s)"))
        out["stream_per_point"] = per
    gate = param.full_name.split("_")[0]
    blk = getattr(getattr(param, "instrument", None), "block_during_ramp", None)
    root = getattr(param, "root_instrument", None) or getattr(param, "instrument", None)
    if blk is not None and blk() and k > 1:
        t_opc = []
        for _ in range(5):
            a = time.perf_counter()
            root.ask("*OPC?")
            t_opc.append(time.perf_counter() - a)
        opc = float(np.median(t_opc))
        out["opc"] = opc
        print(f"  -> each of the {k:g} chunks waits for a *OPC? reply (~{opc * 1e3:.0f} ms). "
              f"dm.fast_ramp({gate}.volt) keeps the {getattr(param, 'step', 0) * 1e3:g} mV chunks and "
              f"confirms once per set: saves ~{(k - 1) * opc * 1e3:.0f} ms/point")
    if k > 1:
        print(f"  -> dm.gate_steps({gate}.volt, max_step={abs(np.median(np.diff(pts))):.4g}) instead "
              f"makes each point one bigger command: saves ~{(set_med - rtt) * 1e3:.0f} ms/point")
    return out


def _hms(t: float) -> str:
    return f"{int(t // 3600)}h {int(t // 60) % 60}m {int(t) % 60}s"


@dataclass
class StepResponse:
    t: np.ndarray                      # s after the set call returned
    response: Dict[str, np.ndarray]    # normalised step response per lock-in (0 before, 1 settled)
    settle: float                      # s: recommended settle (all channels within tol)
    settle_by: Dict[str, float]
    lockin_settle: float               # s: what the lock-in filter alone needs
    set_time: float                    # s: median duration of the set call
    tol: float

    def plot(self):
        import matplotlib.pyplot as plt

        from .plotting import INK_2, MUTED, SERIES, STYLE

        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=(10, 3.6))
            ax.axhspan(1 - self.tol, 1 + self.tol, color="#e6e5e1", lw=0, zorder=0)
            for i, (n, r) in enumerate(self.response.items()):
                ax.plot(self.t * 1e3, r, color=SERIES[i % len(SERIES)], lw=1.5, label=n)
            ax.axvline(self.lockin_settle * 1e3, color=MUTED, ls=(0, (4, 3)), lw=1)
            ax.axvline(self.settle * 1e3, color=INK_2, lw=1)
            ax.text(self.settle * 1e3, 0.02, f" settled ({self.tol:.0%}): {self.settle * 1e3:.0f} ms",
                    color=INK_2, fontsize=8)
            ax.text(self.lockin_settle * 1e3, 0.12, f" lock-in alone: {self.lockin_settle * 1e3:.0f} ms",
                    color=MUTED, fontsize=8)
            ax.set_xlabel("time after the gate set returned (ms)")
            ax.set_ylabel("normalised response")
            ax.set_title("Step response (gate line + device + lock-in)", pad=4)
            ax.legend(loc="lower right")
            fig.tight_layout()
        return fig

    def __repr__(self) -> str:
        lines = [f"Step response: set call {self.set_time * 1e3:.0f} ms; settled to {self.tol:.0%} "
                 f"{self.settle * 1e3:.0f} ms after it returned (lock-in filter alone: "
                 f"{self.lockin_settle * 1e3:.0f} ms)"]
        for n, s in self.settle_by.items():
            lines.append(f"  {n}: {s * 1e3:.0f} ms")
        lines.append(f"-> use settle={math.ceil(self.settle * 200) / 200:g} in stream_sweep")
        return "\n".join(lines)


def step_response(station, param, v1: float, v2: float, n: int = 10, hold: float = 1.0,
                  tol: float = 0.01, tau: Optional[float] = None, slope: int = 24,
                  advanced: Optional[bool] = None, sync: bool = True,
                  single_command: bool = False) -> StepResponse:
    """Step `param` between v1 and v2 `n` times while streaming; measure the settling.

    The gate moves exactly as it does in a sweep (qcodes step / inter_delay, fast_ramp if
    on), and the clock starts when the set call returns - so choose |v2 - v1| like a sweep
    point (one or a few increments) that still changes the signal clearly. With
    single_command=True each step is one jump instead (dm.gate_steps). Uses the station's
    followed SR860s with the streaming filter (tau ~ 1/(30 f), sync on, as stream_sweep).
    dm.step_response(run_id) measures the same from a streamed sweep's own steps.
    """
    from .sweeps import LiveStreams, LockinSetup

    ios, _ = station._stream_plan()
    t_steps = []
    with (gate_steps(param, max_step=abs(v2 - v1)) if single_command else nullcontext()):
        param(v1)
        with LockinSetup(ios, tau, slope, advanced, sync) as setup:
            time.sleep(max(1.0, 5 * setup.settle))
            live = LiveStreams(ios, setup.rate)
            live.start(lead=min(hold, 0.5))
            try:
                for k in range(n):
                    a = time.time()
                    param(v2 if k % 2 == 0 else v1)
                    b = time.time()
                    t_steps.append((a, b))
                    time.sleep(hold)
            finally:
                raw = live.stop()
            lockin_settle = setup.settle
        param(v1)
    segments = [(name, r["z"], r["fs"], r["t_first"]) for name, r in raw.items() if r is not None]
    windows = [(a, b, b + 0.95 * hold, a - 0.1 * hold, a) for a, b in t_steps]
    res = _combine(_collect(segments, windows), tol, lockin_settle,
                   float(np.median([b - a for a, b in t_steps])))
    print(res)
    return res


def _collect(segments, windows) -> Dict[str, List]:
    """Normalised step responses with their weights.

    segments: [(channel, z, fs, t_first)]; windows: [(t_set0, t_set1, t_end, before_a, before_b)]
    - the response runs from t_set1 to t_end, 'before' is the mean over [before_a, before_b]
    and 'after' the mean over the last quarter of the response window. Steps smaller than
    10x the noise are dropped; the rest are weighted by |step|^2 (small steps are noisy).
    """
    out: Dict[str, List] = {}
    for name, z, fs, tf in segments:
        for a, b, e, ba, bb in windows:
            j0, j1 = int(round((b - tf) * fs)), int(round((e - tf) * fs))
            ka, kb = int(round((ba - tf) * fs)), int(round((bb - tf) * fs))
            if ka < 0 or kb <= ka or j1 > len(z) or j1 - j0 < 8:
                continue
            tail = z[j0 + 3 * (j1 - j0) // 4:j1]
            before, after = np.nanmean(z[ka:kb]), np.nanmean(tail)
            step = after - before
            if not np.isfinite(step) or step == 0 or abs(step) < 10 * np.nanstd(tail):
                continue
            out.setdefault(name, []).append((np.real((z[j0:j1] - before) / step), abs(step) ** 2, fs))
    return out


def _combine(collected: Dict[str, List], tol, lockin_settle, set_time) -> StepResponse:
    responses, settle_by, t_axis = {}, {}, None
    for name, items in collected.items():
        if len(items) < 2:
            continue
        m = min(len(r) for r, _, _ in items)
        w = np.array([wt for _, wt, _ in items])
        resp = np.sum([wt * r[:m] for r, wt, _ in items], axis=0) / w.sum()
        t = np.arange(m) / items[0][2]
        bad = np.nonzero(np.abs(resp - 1) > tol)[0]
        settle_by[name] = 0.0 if not len(bad) else float(t[min(bad[-1] + 1, m - 1)])
        responses[name] = resp
        if t_axis is None or len(t) < len(t_axis):
            t_axis = t
    if not responses:
        raise RuntimeError("no channel changed clearly at the steps: use a larger step")
    responses = {k: v[: len(t_axis)] for k, v in responses.items()}
    return StepResponse(t_axis, responses, max(settle_by.values()), settle_by, lockin_settle,
                        set_time, tol)


def step_response_from_run(run_id: int, tol: float = 0.01, basedir: Optional[str] = None) -> StepResponse:
    """Settling measured from a streamed run's own gate steps (the most faithful measurement).

    Every point is a step: its response runs from the moment the set call returned, and is
    normalised between the previous point's average and the end of this point. All steps
    (of all lines, for 2D runs) that change the signal clearly are averaged, weighted by
    their size squared. If more settle is needed than the run used, fix the data with
    dm.refilter(run_id, settle=...).
    """
    from .io import load_run
    from .sweeps import _npz_to_segments

    run = load_run(run_id, basedir)
    st = run.metadata["stream"]
    names = sorted({v[0] for v in st["columns"].values()})
    collected: Dict[str, List] = {}
    set_times: List[float] = []
    for seg in _npz_to_segments(f"{run.dir}/stream.npz", names):
        t0, t1 = seg["t_set0"], seg["t_set1"]
        set_times += list(t1 - t0)
        windows = [(t0[i], t1[i],
                    t0[i + 1] if i + 1 < len(t0) else t1[i] + st["settle"] + st["dwell"],
                    t1[i - 1] + st["settle"], t0[i]) for i in range(1, len(t1))]
        for name, sd in seg["streams"].items():
            for k, v in _collect([(name, sd["z"], sd["fs"], sd["t_first"])], windows).items():
                collected.setdefault(k, []).extend(v)
    lockin_settle = lk.settling_time(st["tau"], st["f_ref"], st["slope"], st["advanced"], st["sync"])
    res = _combine(collected, tol, lockin_settle, float(np.median(set_times)))
    print(res)
    if res.settle > 1.1 * st["settle"] + 0.005:
        print(f"   (this run used settle={st['settle']:.3g} s: dm.refilter({run_id}, "
              f"settle={math.ceil(res.settle * 200) / 200:g}) re-averages it)")
    return res
