"""Minimal SR860 command I/O: a thin wrapper, settings snapshots and restore.

dmz_measure talks to the SR860 with raw commands so it works with whatever
object you already have open:

* a qcodes SR860 instrument (``.write`` / ``.ask``)  <- what the CryoFree notebooks use
* a pyvisa resource (``.write`` / ``.query``)
* a VISA address string such as ``'TCPIP::192.168.50.12::inst0::INSTR'``
* :class:`dmz_measure.simulate.FakeSR860` for offline testing

Streaming needs the VXI-11 address form (``TCPIP::<ip>::inst0::INSTR``): the
SR860 sends UDP packets to the IP that sent ``STREAM ON`` over VXI-11.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Optional

from .. import lockin as _lk

# Settings that the noise procedures touch, and the order they are restored in.
# SLVL (excitation) is restored last so the sample never sees an excitation at an
# unexpected frequency.
SETTINGS = ("OFLT", "OFSL", "SYNC", "ADVFILT", "HARM", "FREQ", "SLVL")
_INT_SETTINGS = {"OFLT", "OFSL", "SYNC", "ADVFILT", "HARM", "RSRC"}

RSRC_NAMES = {0: "INT", 1: "EXT", 2: "DUAL", 3: "CHOP"}


class LockinIO:
    """Text command interface to one SR860, whatever object it is wrapped around."""

    def __init__(self, inst: Any, name: Optional[str] = None):
        if isinstance(inst, LockinIO):
            self.inst, self._write, self._ask = inst.inst, inst._write, inst._ask
            self.name = name or inst.name
            return
        if isinstance(inst, str):
            import pyvisa  # only needed for bare addresses

            res = pyvisa.ResourceManager().open_resource(inst)
            res.read_termination = "\n"
            res.write_termination = "\n"
            res.timeout = 10_000
            self.inst = res
            self.name = name or inst
        else:
            self.inst = inst
            self.name = name or getattr(inst, "name", None) or type(inst).__name__
        self._write = getattr(self.inst, "write")
        ask = getattr(self.inst, "ask", None) or getattr(self.inst, "query", None)
        if ask is None:
            raise TypeError(f"{self.inst!r} has no ask() or query() method")
        self._ask = ask

    def __repr__(self) -> str:
        return f"LockinIO({self.name!r})"

    def write(self, cmd: str) -> None:
        self._write(cmd)

    def ask(self, cmd: str) -> str:
        return str(self._ask(cmd)).strip()

    def ask_float(self, cmd: str) -> float:
        return float(self.ask(cmd))

    def ask_int(self, cmd: str) -> int:
        return int(float(self.ask(cmd)))

    # ---- convenience -------------------------------------------------------
    def idn(self) -> str:
        return self.ask("*IDN?")

    def reference_source(self) -> str:
        return RSRC_NAMES.get(self.ask_int("RSRC?"), "?")

    def detection_frequency(self) -> float:
        """Frequency the lock-in is actually demodulating at (Hz)."""
        try:
            return self.ask_float("FREQDET?")
        except Exception:
            return self.ask_float("FREQ?")

    def time_constant(self) -> float:
        return _lk.tau_from_index(self.ask_int("OFLT?"))

    def enbw(self) -> float:
        return self.ask_float("ENBW?")


def as_io(inst: Any, name: Optional[str] = None) -> LockinIO:
    return inst if isinstance(inst, LockinIO) and name is None else LockinIO(inst, name)


def snapshot_settings(io: LockinIO) -> Dict[str, Any]:
    """Read the settings that the procedures change. Safe to call any time."""
    snap: Dict[str, Any] = {"RSRC": io.ask_int("RSRC?")}
    for key in SETTINGS:
        val = io.ask(f"{key}?")
        snap[key] = int(float(val)) if key in _INT_SETTINGS else float(val)
    return snap


def restore_settings(io: LockinIO, snap: Dict[str, Any]) -> None:
    """Put back a snapshot from :func:`snapshot_settings`.

    FREQ is only written on lock-ins with an internal reference (RSRC=INT);
    SLVL last.
    """
    for key in SETTINGS:
        if key not in snap:
            continue
        if key == "FREQ" and snap.get("RSRC", 0) != 0:
            continue
        val = snap[key]
        io.write(f"{key} {val}" if key in _INT_SETTINGS else f"{key} {float(val):.9E}")


def apply_settings(io: LockinIO, *, tau: Optional[float] = None,
                   slope_db: Optional[int] = None, sync: Optional[bool] = None,
                   advanced: Optional[bool] = None, freq: Optional[float] = None,
                   amplitude: Optional[float] = None) -> Dict[str, Any]:
    """Write the given settings; returns what was written (tau is snapped to the SR860 table)."""
    done: Dict[str, Any] = {}
    if tau is not None:
        i = _lk.tau_index(tau)
        io.write(f"OFLT {i}")
        done["tau"] = _lk.tau_from_index(i)
    if slope_db is not None:
        io.write(f"OFSL {_lk.slope_index(slope_db)}")
        done["slope_db"] = slope_db
    if sync is not None:
        io.write(f"SYNC {int(bool(sync))}")
        done["sync"] = bool(sync)
    if advanced is not None:
        io.write(f"ADVFILT {int(bool(advanced))}")
        done["advanced"] = bool(advanced)
    if freq is not None:
        io.write(f"FREQ {freq:.6f}")
        done["freq"] = freq
    if amplitude is not None:
        amp = max(float(amplitude), 1e-9)  # SR860 minimum is 1 nV
        io.write(f"SLVL {amp:.6E}")
        done["amplitude"] = amp
    return done


@contextmanager
def preserved_settings(ios: Iterable[LockinIO], verbose: bool = True):
    """Snapshot every lock-in, run the block, restore them all (also on error/interrupt).

    Yields the dict of snapshots ``{name: snapshot}``.
    """
    ios = list(ios)
    snaps = {io.name: snapshot_settings(io) for io in ios}
    if verbose:
        print("Saved lock-in settings (restored automatically when done):")
        for name, s in snaps.items():
            print(f"  {name}: " + ", ".join(f"{k}={v}" for k, v in s.items()))
    try:
        yield snaps
    finally:
        errors = []
        for io in ios:
            try:
                restore_settings(io, snaps[io.name])
            except Exception as e:  # keep restoring the others
                errors.append(f"{io.name}: {e}")
        if errors:
            print("WARNING: could not restore some settings:\n  " + "\n  ".join(errors))
        elif verbose:
            print("Lock-in settings restored.")


def wait_for_lock(ios: Iterable[LockinIO], minimum: float = 2.0, timeout: float = 20.0) -> None:
    """Wait at least `minimum` seconds, then until no lock-in reports an unlock/error.

    Uses the SR860 LIA status register (LIAS?): bit 3 = unlock / reference error.
    Falls back to the plain wait if the query is unsupported.
    """
    time.sleep(minimum)
    t_end = time.monotonic() + timeout
    ios = list(ios)
    while time.monotonic() < t_end:
        try:
            # read twice: the register latches, the first read clears old events
            for io in ios:
                io.ask("LIAS?")
            time.sleep(0.3)
            bad = [io.name for io in ios if int(float(io.ask("LIAS?"))) & 0b1000]
        except Exception:
            return
        if not bad:
            return
        time.sleep(0.5)
    print(f"WARNING: reference still unlocked after {timeout:.0f} s on: {', '.join(bad)}")
