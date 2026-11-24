"""Run storage, compatible with measureme's (``sweep``) folder layout.

Each run gets the lowest free integer folder in ``basedir`` (the same scheme as
``sweep.db.Writer``), so dmz_measure and sweep can share one data folder and the
ID numbers stay in one sequence. Every run folder contains

* ``metadata.json``  with at least ``type``, ``interrupted``, ``start_time``,
  ``end_time``, ``columns``, ``comments`` (so ``sweep.list_measurements`` lists it)
* ``data.tsv.gz``    a small numeric table (np.loadtxt-able), e.g. the spectra
* optional ``*.npz`` arrays (time series) and ``*.png`` figures
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

_BASEDIR: Optional[str] = None


def set_basedir(path: str) -> None:
    """Default data folder for runs (otherwise sweep's folder from sweep.set_basedir is used)."""
    global _BASEDIR
    _BASEDIR = path


def get_basedir(basedir: Optional[str] = None, required: bool = True) -> Optional[str]:
    """basedir argument > dmz_measure.set_basedir > sweep.set_basedir."""
    path = basedir or _BASEDIR or getattr(sys.modules.get("sweep.sweep"), "BASEDIR", None)
    if path is None:
        if required:
            raise ValueError("no data folder: call sweep.set_basedir(...) or pass basedir=...")
        return None
    return os.path.expanduser(path)


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, complex):
        return [o.real, o.imag]
    if hasattr(o, "__dict__"):
        return {k: v for k, v in vars(o).items() if not k.startswith("_")}
    return str(o)


class RunWriter:
    """Create the next numbered run folder and write into it.

    >>> with RunWriter(basedir, 'noise_survey', comment='ID 44: survey') as w:
    ...     w.metadata['f_ref'] = 100
    ...     w.write_table(['f_Hz', 'asd'], np.c_[f, asd])
    """

    def __init__(self, basedir: Optional[str], kind: str, comment: str = "",
                 measurement_config: Optional[Dict[str, Any]] = None, max_id: int = 1_000_000):
        base = get_basedir(basedir)
        os.makedirs(base, exist_ok=True)
        i = 0
        while True:
            try:
                os.mkdir(os.path.join(base, str(i)))
                break
            except FileExistsError:
                i += 1
                if i > max_id:
                    raise RuntimeError("could not pick a run id")
        self.id = i
        self.dir = os.path.join(base, str(i))
        self.metadata: Dict[str, Any] = {
            "type": kind,
            "comments": [comment] if comment else [],
            "interrupted": False,
            "start_time": time.time(),
            "columns": [],
            "library": "dmz_measure",
        }
        if measurement_config:
            self.metadata["measurement_config"] = dict(measurement_config)
        self._table_written = False
        self.write_metadata()

    def __enter__(self) -> "RunWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(interrupted=exc_type is not None)

    def path(self, name: str) -> str:
        return os.path.join(self.dir, name)

    def write_metadata(self) -> None:
        with open(self.path("metadata.json"), "wt") as f:
            json.dump(self.metadata, f, indent=4, default=_json_default)

    def write_table(self, columns: Sequence[str], data: np.ndarray, fmt: str = "%.9g") -> None:
        data = np.atleast_2d(np.asarray(data, dtype=float))
        if data.shape[1] != len(columns):
            raise ValueError(f"{len(columns)} columns but data has {data.shape[1]}")
        with gzip.open(self.path("data.tsv.gz"), "wt") as f:
            np.savetxt(f, data, delimiter="\t", fmt=fmt)
        self.metadata["columns"] = list(columns)
        self._table_written = True

    def save_arrays(self, name: str, **arrays: np.ndarray) -> str:
        p = self.path(name if name.endswith(".npz") else name + ".npz")
        np.savez(p, **arrays)
        return p

    def save_figure(self, fig, name: str = "plot.png", dpi: int = 120) -> str:
        p = self.path(name)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        return p

    def close(self, interrupted: bool = False) -> None:
        if not self._table_written:
            # sweep's Reader expects data.tsv.gz to exist
            with gzip.open(self.path("data.tsv.gz"), "wt"):
                pass
        self.metadata["interrupted"] = bool(interrupted)
        self.metadata["end_time"] = time.time()
        self.write_metadata()


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


def load_run(run_id: int, basedir: Optional[str] = None) -> Run:
    d = os.path.join(get_basedir(basedir), str(run_id))
    with open(os.path.join(d, "metadata.json")) as f:
        md = json.load(f)
    table: Dict[str, np.ndarray] = {}
    p = os.path.join(d, "data.tsv.gz")
    if os.path.exists(p) and os.path.getsize(p) > 20:
        data = np.atleast_2d(np.loadtxt(p, delimiter="\t"))
        for i, c in enumerate(md.get("columns", [])):
            table[c] = data[:, i]
    return Run(run_id, d, md, table)
