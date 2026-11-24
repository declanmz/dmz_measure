"""SR860 ethernet data streaming (UDP), from the SR860 manual Appendix D.

Packet = 4-byte big-endian header + 128/256/512/1024 data bytes.

Header bits
  31-24 status   bit 24 overload (input / sync / output), bit 25 error
                 (PLL unlock or sync filter out of range), bit 28 little-endian
                 data, bit 29 integrity checking on
  23-16 rate     N -> sample rate = 1.25 MHz / 2**N
  15-12 length   0: 1024, 1: 512, 2: 256, 3: 128 data bytes
  11-8  content  float32: 0 X, 1 XY, 2 R-theta, 3 XYRtheta; int16: 4-7 likewise
  7-0   counter  increments by one per packet, modulo 256

``STREAM ON`` must be sent over VXI-11 from the receiving PC; the instrument
streams to the IP address that sent it. Windows Firewall usually blocks the
inbound UDP port (1865 by default) until a rule allows it.
"""
from __future__ import annotations

import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .sr860 import LockinIO, as_io

BASE_RATE = 1.25e6
DEFAULT_PORT = 1865
CHANNEL_CODES = {"X": 0, "XY": 1, "RT": 2, "XYRT": 3}
PACKET_CODES = {1024: 0, 512: 1, 256: 2, 128: 3}
PACKET_BYTES = {v: k for k, v in PACKET_CODES.items()}
CONTENT_CHANNELS = {0: ("X",), 1: ("X", "Y"), 2: ("R", "THETA"), 3: ("X", "Y", "R", "THETA")}
INT16_FULL_SCALE = 29491  # integer that corresponds to the sensitivity (90 % of 32768)

FIREWALL_HELP = """\
No UDP packets from {name} arrived on port {port} within {timeout:g} s.
Most likely causes:
  1. Windows Firewall is blocking inbound UDP. In an *administrator* PowerShell on this PC run
       New-NetFirewallRule -DisplayName "SR860 streaming" -Direction Inbound -Protocol UDP -LocalPort 1865-1880 -RemoteAddress {subnet} -Action Allow
     (only the instrument subnet {subnet} is let in).
  2. STREAM ON did not come over VXI-11. Open the lock-in as TCPIP::<ip>::inst0::INSTR; the SR860
     streams to the IP address that sent STREAM ON.
  3. Another program (or another open stream) is already using port {port}."""


def _help(name: str, port: int, timeout: float, ip: Optional[str]) -> str:
    subnet = ip.rsplit(".", 1)[0] + ".0/24" if ip else "<instrument subnet, e.g. 192.168.50.0/24>"
    return FIREWALL_HELP.format(name=name, port=port, timeout=timeout, subnet=subnet)


class StreamTimeout(RuntimeError):
    pass


@dataclass(frozen=True)
class PacketHeader:
    counter: int
    content: int
    length_code: int
    rate_code: int
    status: int

    @classmethod
    def from_word(cls, word: int) -> "PacketHeader":
        return cls(
            counter=word & 0xFF,
            content=(word >> 8) & 0x0F,
            length_code=(word >> 12) & 0x0F,
            rate_code=(word >> 16) & 0xFF,
            status=(word >> 24) & 0xFF,
        )

    def to_word(self) -> int:
        return (self.counter & 0xFF) | (self.content & 0x0F) << 8 | (self.length_code & 0x0F) << 12 \
            | (self.rate_code & 0xFF) << 16 | (self.status & 0xFF) << 24

    @property
    def overload(self) -> bool:
        return bool(self.status & 0x01)

    @property
    def error(self) -> bool:
        return bool(self.status & 0x02)

    @property
    def little_endian(self) -> bool:
        return bool(self.status & 0x10)

    @property
    def checksum(self) -> bool:
        return bool(self.status & 0x20)

    @property
    def is_int16(self) -> bool:
        return self.content >= 4

    @property
    def channels(self) -> Tuple[str, ...]:
        return CONTENT_CHANNELS[self.content & 0x03]

    @property
    def data_bytes(self) -> int:
        return PACKET_BYTES[self.length_code]

    @property
    def samples_per_packet(self) -> int:
        return self.data_bytes // ((2 if self.is_int16 else 4) * len(self.channels))

    @property
    def sample_rate(self) -> float:
        return BASE_RATE / 2 ** self.rate_code


def decode_packet(buf: bytes) -> Tuple[PacketHeader, np.ndarray]:
    """Decode one UDP packet -> (header, values[n_samples, n_channels]).

    float32 data come back in instrument units (V, deg); int16 data come back as raw
    integers (multiply by sensitivity / 29491 to get volts).
    """
    (word,) = struct.unpack_from(">I", buf, 0)
    hdr = PacketHeader.from_word(word)
    dt = np.dtype(("<" if hdr.little_endian else ">") + ("i2" if hdr.is_int16 else "f4"))
    n = min(hdr.data_bytes, len(buf) - 4) // dt.itemsize
    vals = np.frombuffer(buf, dtype=dt, count=n, offset=4).astype(np.float32)
    nch = len(hdr.channels)
    return hdr, vals[: (n // nch) * nch].reshape(-1, nch)


def encode_packet(hdr: PacketHeader, values: np.ndarray) -> bytes:
    """Inverse of decode_packet (used by the simulator and tests)."""
    dt = np.dtype(("<" if hdr.little_endian else ">") + ("i2" if hdr.is_int16 else "f4"))
    payload = np.asarray(values, dtype=dt).tobytes()
    return struct.pack(">I", hdr.to_word()) + payload


@dataclass
class StreamData:
    """One continuous record from one lock-in.

    data     : array [n_samples, n_channels]; samples from dropped packets are NaN
    fs       : sample rate (Hz) from the packet headers
    t0       : host clock (time.time()) when the first packet arrived
    f_ref    : detection frequency at the start of the record (Hz), if known
    """

    name: str
    channels: Tuple[str, ...]
    fs: float
    data: np.ndarray
    t0: float
    f_ref: Optional[float] = None
    n_packets: int = 0
    n_dropped: int = 0
    n_overload: int = 0
    n_error: int = 0
    info: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.data)

    @property
    def duration(self) -> float:
        return len(self.data) / self.fs

    @property
    def t(self) -> np.ndarray:
        return np.arange(len(self.data)) / self.fs

    def channel(self, name: str) -> np.ndarray:
        return self.data[:, self.channels.index(name)]

    @property
    def x(self) -> np.ndarray:
        if "X" in self.channels:
            return self.channel("X")
        return self.channel("R") * np.cos(np.radians(self.channel("THETA")))

    @property
    def y(self) -> np.ndarray:
        if "Y" in self.channels:
            return self.channel("Y")
        return self.channel("R") * np.sin(np.radians(self.channel("THETA")))

    @property
    def z(self) -> np.ndarray:
        """Complex X + iY."""
        return self.x + 1j * self.y

    def summary(self) -> str:
        s = (f"{self.name}: {len(self)} samples, {self.duration:.1f} s at {self.fs:g} Hz, "
             f"{self.n_packets} packets")
        flags = []
        if self.n_dropped:
            flags.append(f"{self.n_dropped} dropped")
        if self.n_overload:
            flags.append(f"{self.n_overload} with OVERLOAD")
        if self.n_error:
            flags.append(f"{self.n_error} with unlock/sync ERROR")
        return s + (" (" + ", ".join(flags) + ")" if flags else ", no drops/flags")


def assemble(packets: Sequence[bytes], name: str = "", t0: float = np.nan,
             f_ref: Optional[float] = None) -> StreamData:
    """Turn raw packets into a StreamData, inserting NaNs for dropped packets."""
    if not packets:
        raise ValueError("no packets")
    blocks: List[np.ndarray] = []
    first, _ = decode_packet(packets[0])
    prev = None
    n_drop = n_ovl = n_err = 0
    for buf in packets:
        hdr, vals = decode_packet(buf)
        if hdr.content != first.content or hdr.rate_code != first.rate_code:
            # settings changed mid-stream; keep only the consistent part
            break
        if prev is not None:
            gap = (hdr.counter - prev - 1) % 256
            if gap:
                n_drop += gap
                blocks.append(np.full((gap * hdr.samples_per_packet, vals.shape[1]), np.nan))
        prev = hdr.counter
        n_ovl += hdr.overload
        n_err += hdr.error
        blocks.append(vals)
    data = np.concatenate(blocks, axis=0)
    return StreamData(name=name, channels=first.channels, fs=first.sample_rate, data=data,
                      t0=t0, f_ref=f_ref, n_packets=len(packets), n_dropped=n_drop,
                      n_overload=n_ovl, n_error=n_err,
                      info={"int16": first.is_int16, "little_endian": first.little_endian,
                            "samples_per_packet": first.samples_per_packet})


class SR860Stream:
    """Stream X/Y (or R/theta) from one SR860 over UDP.

    Only packets coming from the lock-in's own IP address are accepted.

    >>> st = SR860Stream(SR860_B, port=1866)        # qcodes SR860 instrument
    >>> st.configure(rate=40e3)                      # picks the largest allowed rate <= 40 kHz
    >>> data = st.acquire(60)                        # 60 s record -> StreamData
    >>> data.z                                       # complex X + iY
    """

    def __init__(self, lockin: Any, name: Optional[str] = None, port: int = DEFAULT_PORT,
                 channels: str = "XY", packet_size: int = 1024, little_endian: bool = False,
                 integrity_check: bool = True, bind_ip: str = "0.0.0.0",
                 recv_buffer: int = 16 * 2 ** 20):
        self.io: LockinIO = as_io(lockin, name)
        self.name = self.io.name
        if channels not in CHANNEL_CODES:
            raise ValueError(f"channels must be one of {list(CHANNEL_CODES)}")
        if packet_size not in PACKET_CODES:
            raise ValueError(f"packet_size must be one of {list(PACKET_CODES)}")
        self.port = int(port)
        self.channels = channels
        self.packet_size = packet_size
        self.little_endian = little_endian
        self.integrity_check = integrity_check
        self.bind_ip = bind_ip
        self.recv_buffer = recv_buffer
        self.rate: Optional[float] = None
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._packets: List[bytes] = []
        self._t_first: Optional[float] = None
        self._f_ref: Optional[float] = None
        self.source_ip: Optional[str] = self.io.host  # None = accept any sender
        self.n_rejected = 0

    def __repr__(self) -> str:
        return f"SR860Stream({self.name!r}, port={self.port})"

    # ---- configuration -----------------------------------------------------
    def max_rate(self) -> float:
        """STREAMRATEMAX?: depends on the time constant (and the sync filter)."""
        return self.io.ask_float("STREAMRATEMAX?")

    def configure(self, rate: Optional[float] = None, divider_exponent: Optional[int] = None) -> float:
        """Set channels, format (float32), packet size, port, options and rate.

        rate: requested sample rate (Hz); the largest allowed rate max/2**n <= rate is used.
        Returns the nominal sample rate.
        """
        io = self.io
        io.write("STREAM OFF")
        io.write(f"STREAMCH {CHANNEL_CODES[self.channels]}")
        io.write("STREAMFMT 0")  # float32: values in volts, no sensitivity bookkeeping
        io.write(f"STREAMPCKT {PACKET_CODES[self.packet_size]}")
        io.write(f"STREAMPORT {self.port}")
        io.write(f"STREAMOPTION {int(self.little_endian) | 2 * int(self.integrity_check)}")
        rmax = self.max_rate()
        if divider_exponent is None:
            if rate is None or rate >= rmax:
                divider_exponent = 0
            else:
                divider_exponent = int(np.ceil(np.log2(rmax / rate) - 1e-9))
        divider_exponent = int(np.clip(divider_exponent, 0, 20))
        io.write(f"STREAMRATE {divider_exponent}")
        self.rate = rmax / 2 ** divider_exponent
        return self.rate

    # ---- start / stop ------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("stream already running")
        if self.rate is None:
            self.configure()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.recv_buffer)
        except OSError:
            pass
        sock.bind((self.bind_ip, self.port))
        sock.settimeout(0.2)
        self._sock = sock
        self._packets = []
        self._t_first = None
        self._stop.clear()
        try:
            self._f_ref = self.io.detection_frequency()
        except Exception:
            self._f_ref = None
        self._thread = threading.Thread(target=self._recv_loop, name=f"stream-{self.name}", daemon=True)
        self._thread.start()
        self.io.write("STREAM ON")

    def _recv_loop(self) -> None:
        sock, packets, src = self._sock, self._packets, self.source_ip
        bufsize = 4 + 1024 + 64
        while not self._stop.is_set():
            try:
                buf, addr = sock.recvfrom(bufsize)
            except socket.timeout:
                continue
            except OSError:
                break
            if src is not None and addr[0] != src:
                self.n_rejected += 1  # not from our lock-in: ignore
                continue
            if self._t_first is None:
                self._t_first = time.time()
            packets.append(buf)

    @property
    def n_packets(self) -> int:
        return len(self._packets)

    def packet_interval(self) -> float:
        """Seconds between packets at the configured rate."""
        spp = self.packet_size // (4 * len(CONTENT_CHANNELS[CHANNEL_CODES[self.channels]]))
        return spp / self.rate if self.rate else 0.0

    def wait_first_packet(self, timeout: float = 3.0) -> None:
        # at long time constants the max stream rate is low and the first packet is slow
        timeout = max(timeout, 3 * self.packet_interval() + 1.0)
        t_end = time.monotonic() + timeout
        while self._t_first is None:
            if time.monotonic() > t_end:
                self.stop(allow_empty=True)
                raise StreamTimeout(_help(self.name, self.port, timeout, self.source_ip))
            time.sleep(0.01)

    def stop(self, allow_empty: bool = False, send_off: bool = True) -> Optional[StreamData]:
        try:
            if send_off:
                self.io.write("STREAM OFF")
        finally:
            time.sleep(0.25)  # let in-flight packets land
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=2)
            if self._sock is not None:
                self._sock.close()
            self._thread, self._sock = None, None
        if not self._packets:
            if allow_empty:
                return None
            raise StreamTimeout(_help(self.name, self.port, 0, self.source_ip))
        return assemble(self._packets, name=self.name, t0=self._t_first or np.nan, f_ref=self._f_ref)

    def acquire(self, duration: float, rate: Optional[float] = None, timeout: float = 3.0,
                progress: bool = True) -> StreamData:
        """Configure (if rate given), stream for `duration` seconds, return the record."""
        return acquire_many([self], duration, rate=rate, timeout=timeout, progress=progress)[self.name]


def acquire_many(streams: Sequence[SR860Stream], duration: float, rate: Optional[float] = None,
                 timeout: float = 3.0, progress: bool = True) -> Dict[str, StreamData]:
    """Stream from several lock-ins at once (each needs its own port)."""
    ports = [s.port for s in streams]
    if len(set(ports)) != len(ports):
        raise ValueError(f"each stream needs its own UDP port, got {ports}")
    if rate is not None:
        for s in streams:
            s.configure(rate=rate)
    started: List[SR860Stream] = []
    try:
        for s in streams:
            s.start()
            started.append(s)
        for s in streams:
            s.wait_first_packet(timeout)
        _wait_with_progress(streams, duration, progress)
    except BaseException:
        for s in started:
            try:
                s.stop(allow_empty=True)
            except Exception:
                pass
        raise
    out: Dict[str, StreamData] = {}
    errors = []
    for s in streams:  # switch all off first so the records end together
        try:
            s.io.write("STREAM OFF")
        except Exception as e:
            errors.append(e)
    for s in streams:
        out[s.name] = s.stop(send_off=False)
    if errors:
        raise errors[0]
    if progress:
        for d in out.values():
            print("  " + d.summary())
    return out


def _wait_with_progress(streams: Sequence[SR860Stream], duration: float, progress: bool) -> None:
    t0 = time.monotonic()
    bar = None
    if progress:
        try:
            from tqdm.auto import tqdm

            bar = tqdm(total=round(duration), unit="s", desc="streaming", leave=False)
        except Exception:
            bar = None
    last = 0
    while True:
        el = time.monotonic() - t0
        if el >= duration:
            break
        time.sleep(min(0.2, duration - el))
        if bar is not None:
            now = min(round(time.monotonic() - t0), round(duration))
            if now > last:
                bar.update(now - last)
                last = now
                bar.set_postfix(packets=sum(s.n_packets for s in streams))
    if bar is not None:
        bar.close()
