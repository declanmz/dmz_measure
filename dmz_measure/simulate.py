"""Offline stand-ins for testing without the fridge.

* :class:`NoiseEnvironment` - real-valued input signal: white + 1/f noise,
  sinusoidal pickup lines, and the excitation response.
* :class:`SoftwareLockin`   - chunked digital lock-in (sqrt(2) e^{-i w t} mixing,
  cascaded RC stages, optional sync average) used to make realistic X/Y.
* :class:`FakeSR860`        - answers the SR860 commands dmz_measure sends and
  streams UDP packets to 127.0.0.1 exactly like the real instrument, so the
  notebook can be run end to end on any PC.
"""
from __future__ import annotations

import math
import socket
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal

from . import lockin as _lk
from .instruments.sr860_stream import (BASE_RATE, CHANNEL_CODES, CONTENT_CHANNELS, PACKET_BYTES,
                                       PacketHeader, encode_packet)


class NoiseEnvironment:
    """Phase-continuous generator of the voltage at one lock-in input.

    white        : white noise density (unit/sqrt(Hz))
    flicker_corner: frequency (Hz) where 1/f noise equals the white level (0 = none)
    lines        : [(f_Hz, rms), ...] pickup lines (mains, pulse tube, ...)
    transfer     : complex response to the excitation (input volts per volt of SLVL)
    rc_corner    : if set, the excitation response rolls off as 1/(1 + i f/rc_corner)
    """

    def __init__(self, fs: float, white: float = 1e-6, flicker_corner: float = 0.0,
                 lines: Sequence[Tuple[float, float]] = (), transfer: complex = 0.0,
                 rc_corner: Optional[float] = None, seed: Optional[int] = None):
        self.fs = float(fs)
        self.white = white
        self.rng = np.random.default_rng(seed)
        self.lines = [(float(f), float(a), float(self.rng.uniform(0, 2 * np.pi))) for f, a in lines]
        self.transfer = complex(transfer)
        self.rc_corner = rc_corner
        self.n = 0
        # 1/f noise: sum of AR(1) (Lorentzian) processes, 3 per decade, 0.003 Hz .. fs/4
        self._ar = []
        if flicker_corner > 0:
            fk = np.geomspace(0.003, self.fs / 4, int(3 * np.log10(self.fs / 4 / 0.003)) + 1)
            d = 1 / np.log(fk[1] / fk[0])  # Lorentzians per e-fold
            c = white ** 2 * flicker_corner * 2 / (np.pi * d)
            sigma = np.sqrt(np.pi * c / 2)
            for f0 in fk:
                a = math.exp(-2 * math.pi * f0 / self.fs)
                b = sigma * math.sqrt(1 - a * a)
                x0 = self.rng.normal(0, sigma)
                self._ar.append([a, b, np.array([a * x0])])

    def excitation(self, f_ref: float, amplitude: float) -> complex:
        """Complex rms response to an excitation of `amplitude` V rms at f_ref."""
        h = self.transfer * amplitude
        if self.rc_corner:
            h = h / (1 + 1j * f_ref / self.rc_corner)
        return h

    def chunk(self, n: int, ref_phase: Optional[np.ndarray] = None, f_ref: float = 0.0,
              amplitude: float = 0.0) -> np.ndarray:
        t = (self.n + np.arange(n)) / self.fs
        v = self.rng.normal(0.0, self.white * math.sqrt(self.fs / 2), n)
        for st in self._ar:
            a, b, zi = st
            y, st[2] = signal.lfilter([b], [1, -a], self.rng.normal(0, 1, n), zi=zi)
            v += y
        for f, rms, ph in self.lines:
            v += math.sqrt(2) * rms * np.cos(2 * np.pi * f * t + ph)
        if amplitude and ref_phase is not None:
            h = self.excitation(f_ref, amplitude)
            v += math.sqrt(2) * abs(h) * np.cos(ref_phase + np.angle(h))
        self.n += n
        return v


class SoftwareLockin:
    """Chunked lock-in: z = sqrt(2) * LPF[v * exp(-i phi)] with n RC stages (+ sync average)."""

    def __init__(self, fs: float, tau: float, slope_db: int = 24, sync: bool = False):
        self.fs = float(fs)
        self.set_filter(tau, slope_db, sync)

    def set_filter(self, tau: float, slope_db: int = 24, sync: bool = False) -> None:
        self.tau, self.slope_db, self.sync = tau, slope_db, sync
        alpha = 1 - math.exp(-1 / (self.fs * tau))
        self._ba = ([alpha], [1, -(1 - alpha)])
        self._zi = [np.zeros(1, complex) for _ in range(_lk.n_poles(slope_db))]
        self._tail = np.zeros(0, complex)

    def process(self, v: np.ndarray, phase: np.ndarray, f_ref: float) -> np.ndarray:
        z = math.sqrt(2) * v * np.exp(-1j * phase)
        b, a = self._ba
        for i in range(len(self._zi)):
            z, self._zi[i] = signal.lfilter(b, a, z, zi=self._zi[i])
        if self.sync and f_ref > 0:
            m = max(1, int(round(self.fs / f_ref)))
            buf = np.concatenate([self._tail, z])
            if len(buf) >= m:
                cs = np.concatenate([[0], np.cumsum(buf)])
                avg = (cs[m:] - cs[:-m]) / m
                out = avg[-len(z):] if len(avg) >= len(z) else np.concatenate(
                    [np.full(len(z) - len(avg), avg[0]), avg])
            else:
                out = np.full(len(z), buf.mean())
            self._tail = buf[-(m - 1):] if m > 1 else np.zeros(0, complex)
            z = out
        return z


def cryofree_like_lines(scale: float = 1.0, pulse_tube: float = 1.4) -> List[Tuple[float, float]]:
    """A made-up but plausible pickup line list: mains harmonics, a pulse-tube comb, a few strays."""
    lines = [(60.0 * k, scale * 3e-5 / k) for k in range(1, 5)]
    lines += [(pulse_tube * k, scale * 2e-6 / k ** 0.7) for k in range(1, 25)]
    lines += [(37.53, scale * 1.5e-6), (52.0, scale * 1e-6), (73.2, scale * 8e-7),
              (13.0, scale * 2e-6)]  # e.g. another lock-in's excitation
    return lines


class FakeSR860:
    """Pretend SR860 for the subset of commands dmz_measure uses, with real UDP streaming.

    master: another FakeSR860 whose reference this one follows (like EXT reference
    through the master's TTL out); None = internal reference.
    """

    _BASE_RATE = BASE_RATE

    def __init__(self, name: str, environment: Optional[NoiseEnvironment] = None,
                 master: Optional["FakeSR860"] = None, freq: float = 13.2943,
                 amplitude: float = 1.0, tau: float = 0.3, flip_y: bool = False,
                 env_kwargs: Optional[Dict[str, Any]] = None, seed: Optional[int] = None):
        self.name = name
        self.master = master
        self._freq = freq
        self.slvl = amplitude
        self.oflt = _lk.tau_index(tau)
        self.ofsl = 3
        self.sync = 0
        self.advfilt = 1
        self.harm = 1
        self.flip_y = flip_y
        self.env = environment
        self._env_kwargs = dict(env_kwargs or {})
        self._seed = seed
        self.stream_cfg = {"ch": 1, "fmt": 0, "pckt": 0, "port": 1865, "option": 2, "rate": 0}
        self._streaming = False
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.lias = 0

    def __repr__(self) -> str:
        return f"FakeSR860({self.name!r})"

    # ---- reference ---------------------------------------------------------
    @property
    def rsrc(self) -> int:
        return 0 if self.master is None else 1

    @property
    def freq(self) -> float:
        return self._freq if self.master is None else self.master.freq

    @property
    def excitation_amplitude(self) -> float:
        return self.slvl if self.master is None else self.master.slvl

    def max_stream_rate(self) -> float:
        tau = _lk.tau_from_index(self.oflt)
        n = int(np.clip(np.ceil(np.log2(BASE_RATE * tau / 16)), 0, 31))
        return BASE_RATE / 2 ** n

    # ---- command interface -------------------------------------------------
    def write(self, cmd: str) -> None:
        cmd = cmd.strip()
        head, _, arg = cmd.partition(" ")
        head = head.upper()
        arg = arg.strip()
        if head == "FREQ":
            if self.master is not None:
                raise ValueError(f"{self.name}: FREQ not settable with external reference")
            self._freq = float(arg)
        elif head == "SLVL":
            self.slvl = float(arg)
        elif head == "OFLT":
            self.oflt = int(float(arg))
        elif head == "OFSL":
            self.ofsl = int(float(arg))
        elif head == "SYNC":
            self.sync = int(float(arg))
        elif head == "ADVFILT":
            self.advfilt = int(float(arg))
        elif head == "HARM":
            self.harm = int(float(arg))
        elif head == "STREAMCH":
            self.stream_cfg["ch"] = CHANNEL_CODES[arg.upper()] if arg.upper() in CHANNEL_CODES else int(arg)
        elif head == "STREAMFMT":
            self.stream_cfg["fmt"] = int(arg)
        elif head == "STREAMPCKT":
            self.stream_cfg["pckt"] = int(arg)
        elif head == "STREAMPORT":
            self.stream_cfg["port"] = int(arg)
        elif head == "STREAMOPTION":
            self.stream_cfg["option"] = int(arg)
        elif head == "STREAMRATE":
            self.stream_cfg["rate"] = int(arg)
        elif head == "STREAM":
            on = arg.upper() in ("ON", "1")
            if on and not self._streaming:
                self._start_stream()
            elif not on and self._streaming:
                self._stop_stream()
        else:
            raise ValueError(f"FakeSR860 does not know command {cmd!r}")

    def ask(self, cmd: str) -> str:
        q = cmd.strip().upper()
        if q == "*IDN?":
            return f"Stanford_Research_Systems,SR860,FAKE-{self.name},v1.0"
        if q in ("FREQ?", "FREQDET?", "FREQINT?"):
            return f"{self.freq:.6f}"
        if q == "RSRC?":
            return str(self.rsrc)
        if q == "SLVL?":
            return f"{self.slvl:.9g}"
        if q == "OFLT?":
            return str(self.oflt)
        if q == "OFSL?":
            return str(self.ofsl)
        if q == "SYNC?":
            return str(self.sync)
        if q == "ADVFILT?":
            return str(self.advfilt)
        if q == "HARM?":
            return str(self.harm)
        if q == "ENBW?":
            return f"{_lk.enbw_rc(_lk.tau_from_index(self.oflt), _lk.SLOPES_DB[self.ofsl]):.6g}"
        if q == "LIAS?":
            v, self.lias = self.lias, 0
            return str(v)
        if q == "STREAMRATEMAX?":
            return f"{self.max_stream_rate():.6f}"
        if q == "STREAM?":
            return str(int(self._streaming))
        for key, k in (("STREAMCH?", "ch"), ("STREAMFMT?", "fmt"), ("STREAMPCKT?", "pckt"),
                       ("STREAMPORT?", "port"), ("STREAMOPTION?", "option"), ("STREAMRATE?", "rate")):
            if q == key:
                return str(self.stream_cfg[k])
        raise ValueError(f"FakeSR860 does not know query {cmd!r}")

    query = ask

    def close(self) -> None:
        if self._streaming:
            self._stop_stream()

    # ---- streaming ---------------------------------------------------------
    def _start_stream(self) -> None:
        self._stop.clear()
        self._streaming = True
        self._thread = threading.Thread(target=self._stream_loop, daemon=True, name=f"fake-{self.name}")
        self._thread.start()

    def _stop_stream(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._streaming = False
        self._thread = None

    def _stream_loop(self) -> None:
        cfg = dict(self.stream_cfg)
        rmax = self.max_stream_rate()
        n_rate = int(round(np.log2(BASE_RATE / rmax))) + cfg["rate"]
        fs = BASE_RATE / 2 ** n_rate
        tau = _lk.tau_from_index(self.oflt)
        slope = _lk.SLOPES_DB[self.ofsl]
        content = cfg["ch"] + (4 if cfg["fmt"] == 1 else 0)
        chans = CONTENT_CHANNELS[cfg["ch"]]
        little = bool(cfg["option"] & 1)
        status_static = (0x10 if little else 0) | (0x20 if cfg["option"] & 2 else 0)
        data_bytes = PACKET_BYTES[cfg["pckt"]]
        spp = data_bytes // ((2 if cfg["fmt"] == 1 else 4) * len(chans))

        if self.env is None or self.env.fs != fs:
            kw = {"white": 1e-6, "flicker_corner": 3.0, "lines": cryofree_like_lines(), "seed": self._seed}
            kw.update(self._env_kwargs)
            if self.env is not None:  # keep the configured environment, at the new rate
                kw.update(white=self.env.white, lines=[(f, a) for f, a, _ in self.env.lines],
                          transfer=self.env.transfer, rc_corner=self.env.rc_corner)
            self.env = NoiseEnvironment(fs, **kw)
        li = SoftwareLockin(fs, tau, slope, bool(self.sync))
        phase0 = 0.0
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        counter = 0
        t_start = time.monotonic()
        sent = 0
        pending = np.zeros((0, len(chans)))
        # pre-roll so the filters start settled
        for _ in range(3):
            n = int(10 * tau * fs) + 1
            f = self.freq
            ph = phase0 + 2 * np.pi * f * np.arange(1, n + 1) / fs
            v = self.env.chunk(n, ph, f, self.excitation_amplitude)
            li.process(v, ph, f)
            phase0 = float(ph[-1] % (2 * np.pi))
        try:
            while not self._stop.is_set():
                due = int((time.monotonic() - t_start) * fs) - sent
                if due < spp:
                    time.sleep(0.003)
                    continue
                n = due
                f = self.freq
                ph = phase0 + 2 * np.pi * f * np.arange(1, n + 1) / fs
                phase0 = float(ph[-1] % (2 * np.pi))
                v = self.env.chunk(n, ph, f, self.excitation_amplitude)
                z = li.process(v, ph, f)
                if self.flip_y:
                    z = np.conj(z)
                cols = {"X": z.real, "Y": z.imag, "R": np.abs(z), "THETA": np.degrees(np.angle(z))}
                block = np.column_stack([cols[c] for c in chans])
                pending = np.vstack([pending, block])
                sent += n
                while len(pending) >= spp:
                    vals, pending = pending[:spp], pending[spp:]
                    hdr = PacketHeader(counter=counter, content=content, length_code=cfg["pckt"],
                                       rate_code=n_rate, status=status_static)
                    sock.sendto(encode_packet(hdr, vals.ravel()), ("127.0.0.1", cfg["port"]))
                    counter = (counter + 1) % 256
        finally:
            sock.close()


def fake_cryofree(seed: int = 0) -> Dict[str, FakeSR860]:
    """Three fake lock-ins wired like the MM03 CryoFree setup.

    SR860_A: current through the device via a 1e6 V/A preamp (internal reference, drives the excitation)
    SR860_B, SR860_C: four-probe voltages via x100 preamps (external reference from A)
    """
    # A: 1 V excitation through 10 MOhm -> 100 nA -> x1e6 V/A -> 0.1 V at the input
    a = FakeSR860("SR860_A", freq=13.2943, amplitude=1.0, seed=seed,
                  env_kwargs=dict(white=2e-6, flicker_corner=2.0, transfer=0.1,
                                  lines=cryofree_like_lines(scale=0.5)))
    # B, C: 100 nA x ~1 kOhm -> 100 uV x100 -> 10 mV; cable RC gives a growing phase
    b = FakeSR860("SR860_B", master=a, seed=seed + 1,
                  env_kwargs=dict(white=1e-6, flicker_corner=5.0, transfer=0.01, rc_corner=400.0,
                                  lines=cryofree_like_lines(scale=1.0)))
    c = FakeSR860("SR860_C", master=a, seed=seed + 2,
                  env_kwargs=dict(white=1.5e-6, flicker_corner=8.0, transfer=0.006, rc_corner=250.0,
                                  lines=cryofree_like_lines(scale=2.0) + [(88.0, 3e-7)]))
    return {"SR860_A": a, "SR860_B": b, "SR860_C": c}
