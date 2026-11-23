"""Small helpers."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _fmt(v: Any) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else ""
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "-"
        a = abs(v)
        if a != 0 and (a < 1e-3 or a >= 1e5):
            return f"{v:.3e}"
        return f"{v:.4g}"
    return str(v)


def format_table(rows: Sequence[Dict[str, Any]], columns: Optional[List[str]] = None,
                 headers: Optional[Dict[str, str]] = None) -> str:
    """Plain-text table from a list of dicts (pandas not required)."""
    if not rows:
        return "(empty)"
    columns = columns or list(rows[0])
    headers = headers or {}
    cells = [[headers.get(c, c) for c in columns]] + [[_fmt(r.get(c, "")) for c in columns]
                                                       for r in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(columns))]
    lines = ["  ".join(s.rjust(w) for s, w in zip(row, widths)) for row in cells]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)
