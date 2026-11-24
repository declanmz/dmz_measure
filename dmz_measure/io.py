"""The data folder, and loading runs for analysis.

Runs are measureme's numbered folders in the data folder (``metadata.json``, ``data.tsv.gz``
and, for streamed runs, ``stream.npz``), written by the stations.
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


def load_run(run_id: int, basedir: Optional[str] = None) -> Run:
    """A run's metadata and its data table as {column: array}."""
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
