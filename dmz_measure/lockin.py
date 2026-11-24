"""SR860 lock-in filter math.

Time-constant tables, equivalent noise bandwidths, settling times and the filter
response models used to predict how much of a measured noise spectrum ends up in
a lock-in reading.

Numbers quoted from the SR860/SR865 manuals:

* ENBW of the RC filters: 1/(4 tau), 1/(8 tau), 3/(32 tau), 5/(64 tau) for
  6/12/18/24 dB/oct.
* Settling to 1 %: RC filters 4.6/6.6/8.4/10 tau (6-24 dB/oct). With the
  advanced filter at 24 dB/oct: 7.1 tau (Gaussian FIR, tau <= 3 s) and 5.4 tau
  (linear-phase IIR, tau >= 10 s). The manual is silent for 3 s < tau < 10 s,
  so 7.1 tau is used there.
* The sync filter averages over one reference period and settles in ~1/f.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# OFLT index -> time constant in seconds: 1 us, 3 us, 10 us, ..., 10 ks, 30 ks
TIME_CONSTANTS = np.array(
    [(1.0 if i % 2 == 0 else 3.0) * 10.0 ** (i // 2 - 6) for i in range(22)]
)
SLOPES_DB = (6, 12, 18, 24)  # OFSL index 0..3

_ENBW_FACTOR = {6: 1 / 4, 12: 1 / 8, 18: 3 / 32, 24: 5 / 64}
_SETTLE_RC_1PCT = {6: 4.6, 12: 6.6, 18: 8.4, 24: 10.0}


def tau_index(tau: float, mode: str = "nearest") -> int:
    """OFLT index for a time constant.

    mode: 'nearest' (in log), 'floor' (largest table value <= tau) or
    'ceil' (smallest table value >= tau).
    """
    tau = float(tau)
    if mode == "nearest":
        return int(np.argmin(np.abs(np.log(TIME_CONSTANTS) - np.log(tau))))
    if mode == "floor":
        ok = np.nonzero(TIME_CONSTANTS <= tau * (1 + 1e-9))[0]
        return int(ok[-1]) if len(ok) else 0
    if mode == "ceil":
        ok = np.nonzero(TIME_CONSTANTS >= tau * (1 - 1e-9))[0]
        return int(ok[0]) if len(ok) else len(TIME_CONSTANTS) - 1
    raise ValueError(f"unknown mode {mode!r}")


def tau_from_index(i: int) -> float:
    return float(TIME_CONSTANTS[int(i)])


def slope_index(slope_db: int) -> int:
    if slope_db not in SLOPES_DB:
        raise ValueError(f"slope must be one of {SLOPES_DB} dB/oct")
    return SLOPES_DB.index(slope_db)


def n_poles(slope_db: int) -> int:
    slope_index(slope_db)
    return slope_db // 6


def enbw_rc(tau: float, slope_db: int = 24) -> float:
    """Equivalent noise bandwidth (Hz) of the SR860 RC output filter."""
    slope_index(slope_db)
    return _ENBW_FACTOR[slope_db] / tau


def settling_time(tau: float, f_ref: Optional[float] = None, slope_db: int = 24,
                  advanced: bool = True, sync: bool = False) -> float:
    """Time (s) for the output to settle to ~1 % after a step.

    Adds one reference period when the sync filter is on.
    """
    slope_index(slope_db)
    if advanced and slope_db == 24:
        n = 5.4 if tau >= 10 else 7.1
    else:
        n = _SETTLE_RC_1PCT[slope_db]
    t = n * tau
    if sync:
        if not f_ref:
            raise ValueError("f_ref is needed when sync=True")
        t += 1.0 / f_ref
    return t


def recommended_tau(f_ref: float, factor: float = 30.0) -> float:
    """tau ~ 1/(factor * f): short-tau + sync recipe for streaming.

    Returns the largest SR860 time constant <= 1/(factor * f_ref).
    """
    return tau_from_index(tau_index(1.0 / (factor * f_ref), mode="floor"))


def recommended_stream_rate(tau: float) -> float:
    """~10 samples per time constant keeps the stream from aliasing."""
    return 10.0 / tau


# --------------------------------------------------------------------------
# Filter response models
# --------------------------------------------------------------------------

def rc_power_response(nu, tau: float, slope_db: int = 24):
    """|H(nu)|^2 of n cascaded single-pole RC stages (n = slope/6)."""
    n = n_poles(slope_db)
    w = 2 * np.pi * np.asarray(nu, dtype=float) * tau
    return (1.0 + w * w) ** (-n)


def gaussian_power_response(nu, tau: float, slope_db: int = 24):
    """|H(nu)|^2 of the SR860 Advanced filter (tau <= 3 s): n Gaussian FIR stages.

    Per the manual each RC pole is replaced by a Gaussian stage with the same noise
    bandwidth, 1/(4 tau). A Gaussian with time-domain std tau/sqrt(pi) has exactly that,
    |H|^2 = exp(-4 pi nu^2 tau^2), and a -3 dB point 1.48x the RC pole's (the manual
    says 1.5x). n stages in series are one Gaussian with std sqrt(n) tau/sqrt(pi).
    """
    n = n_poles(slope_db)
    return np.exp(-4 * np.pi * n * (np.asarray(nu, dtype=float) * tau) ** 2)


def boxcar_power_response(nu, T: float):
    """|H(nu)|^2 of a plain average over a window of length T."""
    return np.sinc(np.asarray(nu, dtype=float) * T) ** 2


def hann_power_response(nu, T: float):
    """|H(nu)|^2 of a Hann-weighted average over a window of length T.

    W = sinc(nu T) / (1 - (nu T)^2): main lobe twice as wide as the plain average's,
    but sidelobes falling as 1/nu^3 instead of 1/nu (first one at -31 dB).
    """
    x = np.asarray(nu, dtype=float) * T
    den = 1.0 - x * x
    near = np.abs(den) < 1e-9
    w = np.where(near, 0.5, np.sinc(x) / np.where(near, 1.0, den))
    return w * w


WINDOWS = ("boxcar", "hann")


@dataclass
class MeasurementFilter:
    """Everything that low-passes the demodulated signal before it becomes one data point.

    tau, slope_db : SR860 time-constant filter (None = none)
    advanced      : SR860 Advanced filter on (ADVFILT 1): Gaussian stages for tau <= 3 s
                    (longer tau: modelled as RC, which has the same slope)
    sync          : SR860 sync filter (an average over one reference period)
    dwell         : software average over this many seconds (None = single reading)
    window        : weighting of the samples in the dwell: 'boxcar' (plain mean) or 'hann'

    Examples
    --------
    Single reading with tau = 300 ms at 24 dB/oct (what a stepped gate map does):
        MeasurementFilter(tau=0.3)                       # RC
        MeasurementFilter(tau=0.3, advanced=True)        # Advanced (Gaussian)
    Streaming recipe: tau ~ 1/(30 f), sync on, Hann-weighted 0.3 s average per point:
        MeasurementFilter(tau=1e-3, sync=True, dwell=0.3, window='hann')
    """

    tau: Optional[float] = 0.3
    slope_db: int = 24
    sync: bool = False
    dwell: Optional[float] = None
    advanced: bool = False
    window: str = "boxcar"

    def __post_init__(self):
        if self.window not in WINDOWS:
            raise ValueError(f"window must be one of {WINDOWS}")

    @property
    def gaussian(self) -> bool:
        return bool(self.advanced and self.tau and self.tau <= 3.0 * (1 + 1e-9))

    def power_response(self, nu, f_ref):
        """|H(nu)|^2 at offset nu (Hz) from a reference f_ref (Hz); broadcasts."""
        nu = np.asarray(nu, dtype=float)
        h2 = np.ones(np.broadcast(nu, np.asarray(f_ref)).shape)
        if self.tau:
            resp = gaussian_power_response if self.gaussian else rc_power_response
            h2 = h2 * resp(nu, self.tau, self.slope_db)
        if self.sync:
            h2 = h2 * np.sinc(nu / np.asarray(f_ref, dtype=float)) ** 2
        if self.dwell:
            resp = hann_power_response if self.window == "hann" else boxcar_power_response
            h2 = h2 * resp(nu, self.dwell)
        return h2

    def enbw(self, f_ref: float) -> float:
        """One-sided equivalent noise bandwidth (Hz), computed numerically."""
        widths = []
        if self.tau:
            widths.append(1 / (4 * self.tau * np.sqrt(n_poles(self.slope_db))) if self.gaussian
                          else enbw_rc(self.tau, self.slope_db))
        if self.sync:
            widths.append(f_ref / 2)
        if self.dwell:
            widths.append((0.75 if self.window == "hann" else 0.5) / self.dwell)
        if not widths:
            return np.inf
        w = min(widths)
        # integrate far into the tails; sinc^2 tails fall as 1/nu^2
        nu = np.linspace(0, 4000 * w, 400_001)
        h2 = self.power_response(nu, f_ref)
        return float(np.trapezoid(h2, nu) if hasattr(np, "trapezoid") else np.trapz(h2, nu))

    def describe(self) -> str:
        parts = []
        if self.tau:
            kind = " advanced" if self.advanced else ""
            parts.append(f"tau={_fmt_time(self.tau)} {self.slope_db} dB/oct{kind}")
        if self.sync:
            parts.append("sync")
        if self.dwell:
            parts.append(f"dwell={_fmt_time(self.dwell)}{' Hann' if self.window == 'hann' else ''}")
        return ", ".join(parts) or "no filter"


def _fmt_time(t: float) -> str:
    if t >= 1:
        return f"{t:g} s"
    if t >= 1e-3:
        return f"{t * 1e3:g} ms"
    return f"{t * 1e6:g} us"
