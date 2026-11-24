"""How long the signal takes to settle after a gate step, measured from a streamed sweep's own
steps. Analysis only: works from the run ID, no instruments.

    st = dm.step_response(run_id)     # run of s.stream_sweep / s.stream_megasweep
    st.plot()

Every point of a streamed sweep is a step: its response runs from the moment the gate set call
returned, normalised between the previous point's average and the end of this point. If the
signal needs more settle than the run used, dm.refilter(run_id, settle=...) re-averages it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from . import lockin as lk


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
    from .sweeps import load_segments

    run = load_run(run_id, basedir)
    st = run.metadata["stream"]
    names = sorted({v[0] for v in st["columns"].values()})
    collected: Dict[str, List] = {}
    set_times: List[float] = []
    for seg in load_segments(run.dir, names):
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
