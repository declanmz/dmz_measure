"""The data folder, and loading runs for analysis.

Runs are measureme's numbered folders in the data folder (``metadata.json``, ``data.tsv.gz``
and, for streamed runs, ``stream.npz`` or, for streamed maps, one ``stream_00000.npz`` per
line), written by the stations. While a run is still going its points are in ``data.tsv``
(measureme compresses it at the end), so a run can be loaded while it is being measured.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


def set_basedir(path: str) -> None:
    """Data folder for all runs: dm.set_basedir (measureme's set_basedir)."""
    from .measureme import sweep as _mm

    _mm.set_basedir(path)


def get_basedir(basedir: Optional[str] = None, required: bool = True) -> Optional[str]:
    """basedir argument, else the folder set with dm.set_basedir."""
    from .measureme import sweep as _mm

    path = basedir or _mm.BASEDIR
    if path is None:
        if required:
            raise ValueError("no data folder: call dm.set_basedir(...) or pass basedir=...")
        return None
    return os.path.expanduser(path)


@dataclass
class Run:
    id: int
    dir: str
    metadata: Dict[str, Any]
    table: Dict[str, np.ndarray]

    def arrays(self, name: str) -> Dict[str, np.ndarray]:
        p = os.path.join(self.dir, name if name.endswith(".npz") else name + ".npz")
        with np.load(p) as z:
            return {k: z[k] for k in z.files}


def _read_rows(d: str, n_cols: int) -> Optional[np.ndarray]:
    """The data rows of a run: data.tsv.gz once it has ended, else data.tsv (still being
    written, a row per point). Only complete lines of data.tsv are used: a last row cut off in
    the middle of being written (or synced) is left out, never read as a shorter number."""
    gz, plain = os.path.join(d, "data.tsv.gz"), os.path.join(d, "data.tsv")
    if os.path.exists(gz) and os.path.getsize(gz) > 20:
        try:
            return np.loadtxt(gz, delimiter="\t", ndmin=2)
        except (EOFError, OSError, ValueError):   # still being compressed: data.tsv is complete
            pass
    if not os.path.exists(plain):
        return None
    with open(plain, "rt", newline="") as f:
        text = f.read()
    lines = text.split("\n")[:-1]                  # the part after the last newline is unfinished
    rows = [r.rstrip("\r").split("\t") for r in lines if r.strip()]
    rows = [r for r in rows if len(r) == n_cols]
    return np.array(rows, dtype=float).reshape(-1, n_cols) if rows else None


def load_run(run_id: int, basedir: Optional[str] = None) -> Run:
    """A run's metadata and its data table as {column: array}, also while it is running."""
    d = os.path.join(get_basedir(basedir), str(run_id))
    with open(os.path.join(d, "metadata.json")) as f:
        md = json.load(f)
    table: Dict[str, np.ndarray] = {}
    cols = md.get("columns", [])
    data = _read_rows(d, len(cols)) if cols else None
    if data is not None:
        for i, c in enumerate(cols):
            table[c] = data[:, i]
    return Run(run_id, d, md, table)
