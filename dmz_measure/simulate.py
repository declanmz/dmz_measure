"""Offline stand-ins for testing without the fridge.

* :class:`NoiseEnvironment` - real-valued input signal: white + 1/f noise,
  sinusoidal pickup lines, and the (gate dependent) excitation response.
* :class:`SoftwareLockin`   - chunked digital lock-in: sqrt(2) e^{-i w t} mixing, then
  n RC stages (or the Advanced filter's Gaussian stages) and an optional sync average.
* :class:`FakeSR860`        - runs continuously in real time like the instrument, answers
  the commands dmz_measure sends (incl. SNAP?), and streams UDP packets to 127.0.0.1.
* :class:`FakeGate`         - a gate Keithley stand-in (``Kbg2.volt(1.0)``) with a gate-line RC.
* :func:`fake_cryofree`     - three lock-ins and a back gate wired like the MM03 CryoFree.
"""
from __future__ import annotations

import math
import socket
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal

from . import lockin as _lk
from .instruments.sr860_stream import (BASE_RATE, CHANNEL_CODES, CONTENT_CHANNELS, PACKET_BYTES,
                                       PacketHeader, encode_packet)


class NoiseEnvironment:
    """Phase-continuous generator of the voltage at one lock-in input.

    white         : white noise density (unit/sqrt(Hz))
    flicker_corner: frequency (Hz) where 1/f noise equals the white level (0 = none)
    lines         : [(f_Hz, rms), ...] pickup lines (mains, pulse tube, ...)
    transfer      : complex response to the excitation (input volts per volt of SLVL)
    rc_corner     : if set, the excitation response rolls off as 1/(1 + i f/rc_corner)
    """

    def __init__(self, fs: float, white: float = 1e-6, flicker_corner: float = 0.0,
                 lines: Sequence[Tuple[float, float]] = (), transfer: complex = 0.0,
                 rc_corner: Optional[float] = None, seed: Optional[int] = None):
        self.fs = float(fs)
        self.white = white
        self.flicker_corner = flicker_corner
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

    def params(self) -> Dict[str, Any]:
        return dict(white=self.white, flicker_corner=self.flicker_corner,
                    lines=[(f, a) for f, a, _ in self.lines], transfer=self.transfer,
                    rc_corner=self.rc_corner)

    def excitation(self, f_ref: float, amplitude: float) -> complex:
        """Complex rms response to an excitation of `amplitude` V rms at f_ref."""
        h = self.transfer * amplitude
        if self.rc_corner:
            h = h / (1 + 1j * f_ref / self.rc_corner)
        return h

    def chunk(self, n: int, ref_phase: Optional[np.ndarray] = None, f_ref: float = 0.0,
              amplitude: float = 0.0, multiplier: Optional[np.ndarray] = None) -> np.ndarray:
        """Next n samples. `multiplier` (per sample) scales the excitation response (gate dependence)."""
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
            if multiplier is not None:
                h = h * multiplier
            v += math.sqrt(2) * np.real(h * np.exp(1j * ref_phase))
        self.n += n
        return v


class SoftwareLockin:
    """Chunked lock-in: z = sqrt(2) * LPF[v * exp(-i phi)].

    The low-pass is n RC stages (n = slope/6) or, with advanced=True and tau <= 3 s, the
    SR860 Advanced filter: Gaussian stages of the same noise bandwidth (n of them in
    series = one Gaussian with time-domain std sqrt(n) tau/sqrt(pi)). sync=True adds a
    moving average over one reference period.
    """

    def __init__(self, fs: float, tau: float, slope_db: int = 24, sync: bool = False,
                 advanced: bool = False):
        self.fs = float(fs)
        self.tau, self.slope_db, self.sync = tau, slope_db, sync
        n = _lk.n_poles(slope_db)
        self.gaussian = bool(advanced and tau <= 3.0 * (1 + 1e-9))
        if self.gaussian:
            sigma = math.sqrt(n) * tau / math.sqrt(math.pi) * self.fs  # in samples
            L = int(8 * sigma) + 1
            if L >= 3:
                k = np.arange(L)
                g = np.exp(-0.5 * ((k - 4 * sigma) / sigma) ** 2)
                self._stages = [(g / g.sum(), np.array([1.0]))]
            else:
                self._stages = []
        else:
            alpha = 1 - math.exp(-1 / (self.fs * tau))
            self._stages = [(np.array([alpha]), np.array([1.0, -(1 - alpha)]))] * n
        self._zi = [np.zeros(max(len(b), len(a)) - 1, complex) for b, a in self._stages]
        self._tail = np.zeros(0, complex)

    def prime(self, value: complex, f_ref: float = 0.0) -> None:
        """Start the filters settled at a constant output `value` (no start-up transient)."""
        self._zi = [signal.lfilter_zi(b, a).astype(complex) * value for b, a in self._stages]
        if self.sync and f_ref > 0:
            m = max(1, int(round(self.fs / f_ref)))
            self._tail = np.full(m - 1, value, complex)

    def process(self, v: np.ndarray, phase: np.ndarray, f_ref: float) -> np.ndarray:
        z = math.sqrt(2) * v * np.exp(-1j * phase)
        for i, (b, a) in enumerate(self._stages):
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


# --------------------------------------------------------------------------
# Fake gate and device
# --------------------------------------------------------------------------

class FakeParam:
    """qcodes-style parameter: p() gets, p(value) sets; has name, full_name and instrument.

    Like qcodes, a parameter with `step` set is ramped in chunks of at most `step`, one
    set call each, waiting until `inter_delay` has passed since the previous set call.
    """

    def __init__(self, instrument: Any, name: str, get: Callable, set: Optional[Callable] = None):
        self.instrument = instrument
        self.name = name
        self.full_name = f"{instrument.name}_{name}"
        self._get, self._set = get, set
        self.step: Optional[float] = None
        self.inter_delay = 0.0
        self.post_delay = 0.0
        self._t_last_set = 0.0

    def __call__(self, value: Optional[float] = None):
        if value is None:
            return self._get()
        if self._set is None:
            raise ValueError(f"{self.full_name} is read-only")
        values = [value]
        if self.step:
            start = self._get()
            k = max(1, int(math.ceil(abs(value - start) / self.step - 1e-9)))
            values = list(np.linspace(start, value, k + 1)[1:])
        for v in values:
            wait = self.inter_delay - (time.perf_counter() - self._t_last_set)
            if wait > 0:
                time.sleep(wait)
            self._set(float(v))
            self._t_last_set = time.perf_counter()
            if self.post_delay:
                time.sleep(self.post_delay)

    def get(self):
        return self._get()

    def set(self, value):
        self(value)

    def __repr__(self) -> str:
        return f"<FakeParam {self.full_name}>"


class FakeGate:
    """Gate Keithley 2450 stand-in.

    A set command is a write taking `write_time` s (the voltage is applied at its end), followed,
    when `block_during_ramp` is on (as in the CryoFree notebook), by a *OPC? round trip of
    `opc_time` s. The gate line then settles with RC `rc`. Reading the leakage current takes
    `read_time` s. `volt` ramps like qcodes when its `step` / `inter_delay` are set. The times
    of all set commands are kept in `command_times` (for checking ramp rates).
    """

    def __init__(self, name: str = "Kbg2", rc: float = 0.02, set_time: float = 0.02,
                 read_time: float = 0.02, write_time: float = 0.002):
        self.name, self.rc, self.read_time, self.write_time = name, rc, read_time, write_time
        self.opc_time = max(set_time - write_time, 0.0)
        self._block = True
        self._steps: List[Tuple[float, float]] = [(time.time(), 0.0)]
        self.command_times: List[Tuple[float, float]] = []
        self._lock = threading.Lock()
        self._rng = np.random.default_rng()
        # like the notebook's Kbg2.volt = Kbg2.source.voltage / Kbg2.curr = Kbg2.sense.current
        self.volt = FakeParam(self, "source_voltage", lambda: self.setpoint, self.set)
        self.curr = FakeParam(self, "sense_current", self._read_current)

    @property
    def set_time(self) -> float:
        """Time of one set command with block_during_ramp on (write + *OPC? round trip)."""
        return self.write_time + self.opc_time

    @set_time.setter
    def set_time(self, value: float) -> None:
        self.opc_time = max(value - self.write_time, 0.0)

    def block_during_ramp(self, value: Optional[bool] = None):
        if value is None:
            return self._block
        self._block = bool(value)

    def ask(self, cmd: str) -> str:
        if cmd.strip().upper() == "*OPC?":
            time.sleep(self.opc_time)
            return "1"
        raise ValueError(f"FakeGate does not know {cmd!r}")

    def _read_current(self) -> float:
        time.sleep(self.read_time)
        return float(self._rng.normal(2e-12, 1e-12))

    @property
    def setpoint(self) -> float:
        return self._steps[-1][1]

    def set(self, value: float) -> None:
        time.sleep(self.write_time)
        with self._lock:
            now = time.time()
            self.command_times.append((now, float(value)))
            self._steps.append((now, float(value)))
            while len(self._steps) > 2 and self._steps[1][0] < now - 30 * self.rc:
                self._steps.pop(0)  # long settled: the next step becomes the baseline
        if self._block:
            time.sleep(self.opc_time)

    def effective(self, t: np.ndarray) -> np.ndarray:
        """Gate voltage the device sees at host times t (first-order approach to each setpoint)."""
        with self._lock:
            steps = list(self._steps)
        v = np.full(np.shape(t), steps[0][1], dtype=float)
        for (_, v0), (t1, v1) in zip(steps, steps[1:]):
            dt = np.clip(t - t1, 0, None)
            v += (v1 - v0) * np.where(t >= t1, 1 - np.exp(-dt / self.rc), 0.0)
        return v

    def close(self) -> None:
        pass


def lorentzian_device(gate: FakeGate, peaks: Sequence[Tuple[float, float, float]]) -> Callable:
    """R(V)/R(0) for R(V) = 1 + sum a / (1 + ((V - v0)/w)^2): a charge-neutrality-like peak etc."""
    def r(v):
        return 1 + sum(a / (1 + ((v - v0) / w) ** 2) for v0, a, w in peaks)

    r0 = r(0.0)
    return lambda t: r(gate.effective(t)) / r0


# --------------------------------------------------------------------------
# Fake SR860
# --------------------------------------------------------------------------

class FakeSR860:
    """Pretend SR860: runs continuously in real time, answers commands, streams UDP.

    master : another FakeSR860 whose reference this one follows (EXT reference); None = internal
    device : callable(host_times) -> complex factor on the excitation response (gate dependence)
    """

    registry: Dict[str, "FakeSR860"] = {}  # found by name like qcodes instruments

    def __init__(self, name: str, master: Optional["FakeSR860"] = None, freq: float = 13.2943,
                 amplitude: float = 1.0, tau: float = 0.3, flip_y: bool = False,
                 env_kwargs: Optional[Dict[str, Any]] = None, device: Optional[Callable] = None,
                 seed: Optional[int] = None):
        self.name = name
        self.address = "TCPIP::127.0.0.1::inst0::INSTR"
        FakeSR860.registry[name] = self
        self.master = master
        self._freq = freq
        self.slvl = amplitude
        self.oflt = _lk.tau_index(tau)
        self.ofsl = 3
        self.sync = 0
        self.advfilt = 0
        self.harm = 1
        self.flip_y = flip_y
        self.device = device
        self._env_params = {"white": 1e-6, "flicker_corner": 3.0, "lines": cryofree_like_lines()}
        self._env_params.update(env_kwargs or {})
        self._seed = seed
        self.stream_cfg = {"ch": 1, "fmt": 0, "pckt": 0, "port": 1865, "option": 2, "rate": 0}
        self._stream: Optional[Dict[str, Any]] = None
        self._cfg_version = 0
        self._z_last = 0j
        self._engine: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self.lias = 0
        # qcodes-style parameters, as on the qcodes SR860 driver
        snap = lambda: complex(*map(float, self.ask("SNAP? 0,1").split(",")))
        self.X = FakeParam(self, "X", lambda: snap().real)
        self.Y = FakeParam(self, "Y", lambda: snap().imag)
        self.R = FakeParam(self, "R", lambda: abs(snap()))
        self.P = FakeParam(self, "P", lambda: float(np.degrees(np.angle(snap()))))
        self.frequency = FakeParam(self, "frequency", lambda: self.freq,
                                   lambda v: self.write(f"FREQ {v}"))
        self.amplitude = FakeParam(self, "amplitude", lambda: self.slvl,
                                   lambda v: self.write(f"SLVL {v}"))
        self.time_constant = FakeParam(self, "time_constant", lambda: self.tau,
                                       lambda v: self.write(f"OFLT {_lk.tau_index(v)}"))

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

    @property
    def tau(self) -> float:
        return _lk.tau_from_index(self.oflt)

    def max_stream_rate(self) -> float:
        n = int(np.clip(np.ceil(np.log2(BASE_RATE * self.tau / 16)), 5, 31))
        return BASE_RATE / 2 ** n

    # ---- command interface -------------------------------------------------
    def write(self, cmd: str) -> None:
        self._ensure_engine()
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
        elif head in ("OFLT", "OFSL", "SYNC", "ADVFILT"):
            setattr(self, head.lower(), int(float(arg)))
            self._cfg_version += 1
        elif head == "HARM":
            self.harm = int(float(arg))
        elif head == "STREAMCH":
            self.stream_cfg["ch"] = CHANNEL_CODES[arg.upper()] if arg.upper() in CHANNEL_CODES else int(arg)
        elif head in ("STREAMFMT", "STREAMPCKT", "STREAMPORT", "STREAMOPTION", "STREAMRATE"):
            key = {"STREAMFMT": "fmt", "STREAMPCKT": "pckt", "STREAMPORT": "port",
                   "STREAMOPTION": "option", "STREAMRATE": "rate"}[head]
            self.stream_cfg[key] = int(arg)
        elif head == "STREAM":
            if arg.upper() in ("ON", "1"):
                if self._stream is None:
                    self._stream = {"cfg": dict(self.stream_cfg), "counter": 0, "k0": None,
                                    "pending": np.zeros((0, 0))}
            else:
                self._stream = None
        else:
            raise ValueError(f"FakeSR860 does not know command {cmd!r}")

    def ask(self, cmd: str) -> str:
        self._ensure_engine()
        q = cmd.strip().upper()
        if q == "*IDN?":
            return f"Stanford_Research_Systems,SR860,FAKE-{self.name},v1.0"
        if q in ("FREQ?", "FREQDET?", "FREQINT?"):
            return f"{self.freq:.6f}"
        if q.startswith("SNAP?") or q.startswith("OUTP?"):
            self._ready.wait(1.0)
            z = self._z_last
            vals = {"0": z.real, "X": z.real, "1": z.imag, "Y": z.imag, "2": abs(z), "R": abs(z),
                    "3": math.degrees(math.atan2(z.imag, z.real)),
                    "TH": math.degrees(math.atan2(z.imag, z.real))}
            keys = [k.strip() for k in q.split(" ", 1)[1].split(",")]
            return ",".join(f"{vals[k]:.9g}" for k in keys)
        simple = {"RSRC?": self.rsrc, "SLVL?": f"{self.slvl:.9g}", "OFLT?": self.oflt,
                  "OFSL?": self.ofsl, "SYNC?": self.sync, "ADVFILT?": self.advfilt,
                  "HARM?": self.harm, "STREAM?": int(self._stream is not None),
                  "STREAMRATEMAX?": f"{self.max_stream_rate():.6f}"}
        if q in simple:
            return str(simple[q])
        if q == "ENBW?":
            return f"{_lk.enbw_rc(self.tau, _lk.SLOPES_DB[self.ofsl]):.6g}"
        if q == "LIAS?":
            v, self.lias = self.lias, 0
            return str(v)
        for key, k in (("STREAMCH?", "ch"), ("STREAMFMT?", "fmt"), ("STREAMPCKT?", "pckt"),
                       ("STREAMPORT?", "port"), ("STREAMOPTION?", "option"), ("STREAMRATE?", "rate")):
            if q == key:
                return str(self.stream_cfg[k])
        raise ValueError(f"FakeSR860 does not know query {cmd!r}")

    query = ask

    def close(self) -> None:
        self._stop.set()
        if self._engine is not None:
            self._engine.join(timeout=2)
        self._engine = None

    # ---- the real-time engine --------------------------------------------------
    def _ensure_engine(self) -> None:
        if self._engine is None or not self._engine.is_alive():
            self._stop.clear()
            self._engine = threading.Thread(target=self._run, daemon=True, name=f"fake-{self.name}")
            self._engine.start()

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        phase = 0.0
        try:
            while not self._stop.is_set():
                version = self._cfg_version
                fs_out = self.max_stream_rate()
                # simulate the input at >= 1.2 kHz so 60 Hz harmonics don't alias at long tau
                fs = max(fs_out, BASE_RATE / 2 ** 10)
                q_out = int(round(fs / fs_out))
                env = NoiseEnvironment(fs, seed=self._seed, **self._env_params)
                li = SoftwareLockin(fs, self.tau, _lk.SLOPES_DB[self.ofsl], bool(self.sync),
                                    advanced=bool(self.advfilt))
                li.prime(self._z_last if not self.flip_y else np.conj(self._z_last), self.freq)
                t0, host0, k = time.monotonic(), time.time(), 0
                while not self._stop.is_set() and version == self._cfg_version:
                    n = min(int((time.monotonic() - t0) * fs) - k, int(fs))
                    if n < max(1, q_out):
                        time.sleep(0.002)
                        continue
                    f = self.freq
                    ph = phase + 2 * np.pi * f * np.arange(1, n + 1) / fs
                    phase = float(ph[-1] % (2 * np.pi))
                    mult = self.device(host0 + (k + np.arange(n)) / fs) if self.device else None
                    v = env.chunk(n, ph, f, self.excitation_amplitude, mult)
                    z = li.process(v, ph, f)
                    if self.flip_y:
                        z = np.conj(z)
                    self._z_last = complex(z[-1])
                    self._ready.set()
                    st = self._stream
                    if st is not None:
                        self._feed(st, z, k, q_out, fs_out, sock)
                    k += n
        finally:
            sock.close()

    def _feed(self, st, z, k, q_out, fs_out, sock) -> None:
        cfg = st["cfg"]
        div = q_out * 2 ** cfg["rate"]
        if st["k0"] is None:
            st["k0"] = k
        idx = k + np.arange(len(z))
        z = z[(idx - st["k0"]) % div == 0]
        chans = CONTENT_CHANNELS[cfg["ch"]]
        cols = {"X": z.real, "Y": z.imag, "R": np.abs(z), "THETA": np.degrees(np.angle(z))}
        block = np.column_stack([cols[c] for c in chans])
        pending = block if st["pending"].size == 0 else np.vstack([st["pending"], block])
        spp = PACKET_BYTES[cfg["pckt"]] // ((2 if cfg["fmt"] == 1 else 4) * len(chans))
        rate_code = int(round(np.log2(BASE_RATE / fs_out))) + cfg["rate"]
        status = (0x10 if cfg["option"] & 1 else 0) | (0x20 if cfg["option"] & 2 else 0)
        while len(pending) >= spp:
            vals, pending = pending[:spp], pending[spp:]
            hdr = PacketHeader(counter=st["counter"], content=cfg["ch"] + (4 if cfg["fmt"] == 1 else 0),
                               length_code=cfg["pckt"], rate_code=rate_code, status=status)
            sock.sendto(encode_packet(hdr, vals.ravel()), ("127.0.0.1", cfg["port"]))
            st["counter"] = (st["counter"] + 1) % 256
        st["pending"] = pending


def fake_cryofree(seed: int = 0) -> Dict[str, Any]:
    """Three fake lock-ins and a back gate, wired like the MM03 CryoFree setup.

    SR860_A: current through the device via a 1e6 V/A preamp (internal reference, drives the excitation)
    SR860_B, SR860_C: four-probe voltages via x100 preamps (external reference from A)
    Kbg2: back gate; B and C see a broad peak at +0.25 V and a sharp one at -0.55 V
    (R_B = 1 kOhm and R_C = 600 Ohm at 0 V).
    """
    gate = FakeGate("Kbg2")
    # A: 1 V excitation through 10 MOhm -> 100 nA -> x1e6 V/A -> 0.1 V at the input
    a = FakeSR860("SR860_A", freq=13.2943, amplitude=1.0, seed=seed,
                  env_kwargs=dict(white=2e-6, flicker_corner=2.0, transfer=0.1,
                                  lines=cryofree_like_lines(scale=0.5)))
    # B, C: 100 nA x ~1 kOhm -> 100 uV x100 -> 10 mV; cable RC gives a growing phase
    b = FakeSR860("SR860_B", master=a, seed=seed + 1,
                  device=lorentzian_device(gate, [(0.25, 3.0, 0.15), (-0.55, 0.8, 0.03)]),
                  env_kwargs=dict(white=1e-6, flicker_corner=5.0, transfer=0.01, rc_corner=400.0,
                                  lines=cryofree_like_lines(scale=1.0)))
    c = FakeSR860("SR860_C", master=a, seed=seed + 2,
                  device=lorentzian_device(gate, [(0.25, 2.5, 0.2), (-0.55, 0.6, 0.03)]),
                  env_kwargs=dict(white=1.5e-6, flicker_corner=8.0, transfer=0.006, rc_corner=250.0,
                                  lines=cryofree_like_lines(scale=2.0) + [(88.0, 3e-7)]))
    return {"SR860_A": a, "SR860_B": b, "SR860_C": c, "Kbg2": gate}
