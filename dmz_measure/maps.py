"""Gate maps: split an up/down map into its up and down maps, and bin maps. Analysis only.

An up/down map is one ``s.megasweep`` or ``s.stream_megasweep`` whose fast setpoints are a grid
followed by the same grid reversed, e.g. ``np.r_[Vtg, Vtg[::-1]]``: every slow line sweeps up
and then straight back down, so there is no ramp back to the start of each line. It is one
run ID; the data file is measureme's, so ``dm.pload2d`` still reads it (with 2 x len(Vtg) fast
points per line). load_updown also reads a map that is still running.

    up, down = dm.load_updown(run_id)                     # as measured
    up, down = dm.load_updown(run_id, dwell=0.24, window='boxcar')   # streamed: re-averaged
    up['xs'], up['ys']                                    # fast grid (ascending), slow setpoints
    up['SR860_B_X'][i, j]                                 # at slow ys[i], fast xs[j]: as pload2d
    down['SR860_B_X'][i, j]                               # the same (ys[i], xs[j]) point, swept down
    b = dm.bin_map(up, fast=2, slow=2)                    # mean of 2 x 2 blocks
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .io import Run, get_basedir, load_run
from .sweeps import refilter


def _as_map(run: Run) -> Dict[str, Any]:
    """A 2D run's table laid out like dm.pload2d ([slow, fast]); points not measured are NaN."""
    md = run.metadata
    xs = np.asarray(md["fast_setpoints"], dtype=float)
    ys = np.asarray(md["slow_setpoints"], dtype=float)
    n = len(xs) * len(ys)
    out: Dict[str, Any] = {"measurement_config": md.get("measurement_config"), "xs": xs, "ys": ys}
    for c in md.get("columns", []):
        v = np.asarray(run.table.get(c, np.empty(0)), dtype=float)[:n]
        out[c] = np.pad(v, (0, n - len(v)), constant_values=np.nan).reshape(len(ys), len(xs))
    return out


def load_updown(run_id: int, basedir: Optional[str] = None, dwell: Optional[float] = None,
                settle: Optional[float] = None, window: Optional[str] = None
                ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The up and the down map of an up/down gate map, on the same grid.

    Returns (up, down), each laid out like ``dm.pload2d``: 'xs' = the fast grid in ascending
    order, 'ys' = the slow setpoints, and every column as an array [slow, fast]. The down map
    is reordered to the same ascending grid, so up[c][i, j] and down[c][i, j] are the same
    gate point, and up - down is the hysteresis. 'direction' says which is which. Points not
    measured (an interrupted map) are NaN.

    dwell / settle / window (streamed maps only): re-average the saved stream instead of
    using the values as measured (as dm.refilter). Nothing is re-measured.

    Works while the map is still running (e.g. from another computer through Dropbox): the
    points measured so far are filled in, the rest are NaN, and 'finished' is False.
    """
    basedir = get_basedir(basedir)
    run = load_run(run_id, basedir)
    md = run.metadata
    if md.get("type") != "2D":
        raise ValueError(f"run {run_id} is not a 2D map")
    if dwell is None and settle is None and window is None:
        d = _as_map(run)
    else:
        d = refilter(run_id, dwell=dwell, settle=settle, window=window, basedir=basedir)
    xs = np.asarray(d["xs"], dtype=float)
    n = len(xs) // 2
    if len(xs) % 2 or n < 2 or not np.allclose(xs[:n], xs[n:][::-1]):
        raise ValueError(f"run {run_id} is not an up/down map: its fast setpoints are not a grid "
                         "followed by the same grid reversed")
    grid = xs[:n]
    if not (np.all(np.diff(grid) > 0) or np.all(np.diff(grid) < 0)):
        raise ValueError(f"run {run_id}: the fast grid is not monotonic")
    first_up = grid[-1] > grid[0]
    order = slice(None) if first_up else slice(None, None, -1)   # -> ascending grid

    def half(second: bool) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 2 * n:
                part = v[:, n:][:, ::-1] if second else v[:, :n]   # second half back in grid order
                out[k] = part[:, order]
            elif k not in ("xs",):
                out[k] = v
        out["xs"] = grid[order]
        out["ys"] = np.asarray(d["ys"], dtype=float)
        out["run_id"] = run_id
        out["finished"] = "end_time" in md
        return out

    first, second = half(False), half(True)
    first["direction"] = "up" if first_up else "down"
    second["direction"] = "down" if first_up else "up"
    return (first, second) if first_up else (second, first)


def bin_map(d: Dict[str, Any], fast: int = 1, slow: int = 1) -> Dict[str, Any]:
    """Average a map (pload2d / load_updown layout) over blocks of `slow` x `fast` points.

    Every [slow, fast] array becomes the mean of each block (NaNs skipped); 'xs' and 'ys' become
    the block centres. Points left over at the end of an axis are dropped. Bin the X and Y
    columns and then take R = V/I, or take R first and bin it: with a steady current the two
    agree well within the noise.
    """
    xs, ys = np.asarray(d["xs"], dtype=float), np.asarray(d["ys"], dtype=float)
    nx, ny = len(xs) // fast, len(ys) // slow
    if nx < 1 or ny < 1:
        raise ValueError("bins larger than the map")

    def blocks(a: np.ndarray) -> np.ndarray:
        a = a[: ny * slow, : nx * fast].reshape(ny, slow, nx, fast)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN blocks stay NaN
            return np.nanmean(a, axis=(1, 3))

    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape == (len(ys), len(xs)):
            out[k] = blocks(v)
        elif k not in ("xs", "ys"):
            out[k] = v
    out["xs"] = xs[: nx * fast].reshape(nx, fast).mean(axis=1)
    out["ys"] = ys[: ny * slow].reshape(ny, slow).mean(axis=1)
    out["binned"] = (slow, fast)
    return out
