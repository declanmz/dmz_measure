"""Noise analysis for picking lock-in excitation frequencies.

The idea
--------
Stream X and Y from an SR860 at a fixed reference f_ref with a short time
constant and no sync filter. The complex output z = X + iY is the input signal
shifted down by f_ref: an input component at frequency f shows up in z at
offset nu = f - f_ref. A complex (two-sided) FFT of z therefore gives the input
spectrum over 0 < f < 2 f_ref (and beyond, up to the filter/Nyquist limit) from
one record, with frequency resolution set only by how long you record.

Normalisation: with z defined so that a sine of rms amplitude A at f_ref reads
|z| = A, the two-sided PSD of z at nu equals the one-sided PSD of the input at
f_ref + nu. So sqrt(PSD) is directly the input noise density in V/sqrt(Hz) (or
A/sqrt(Hz) after dividing by a transimpedance gain), and a lock-in reading X at
reference f with filter |H|^2 has variance

    sigma_X^2 = 1/2 * integral S_in(f') (|H(f' - f)|^2 + |H(f' + f)|^2) df'

which is what :func:`predicted_noise` evaluates for every candidate f.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import ndimage, signal

from .lockin import MeasurementFilter, rc_power_response


# --------------------------------------------------------------------------
# Spectra
# --------------------------------------------------------------------------

def welch(x: np.ndarray, fs: float, nperseg: int, overlap: float = 0.5,
          window: str = "hann") -> Tuple[np.ndarray, np.ndarray, int]:
    """Welch PSD that skips any segment containing NaN (dropped packets).

    Complex input -> two-sided PSD on fftshifted frequencies.
    Real input -> one-sided PSD.
    Returns (freqs, psd, n_segments_used).
    """
    x = np.asarray(x)
    nperseg = int(nperseg)
    if len(x) < nperseg:
        raise ValueError(f"record ({len(x)} samples) shorter than one segment ({nperseg})")
    step = max(1, int(round(nperseg * (1 - overlap))))
    w = signal.get_window(window, nperseg)
    scale = 1.0 / (fs * np.sum(w ** 2))
    is_complex = np.iscomplexobj(x)
    acc = None
    n_used = 0
    for s in range(0, len(x) - nperseg + 1, step):
        seg = x[s:s + nperseg]
        if not np.all(np.isfinite(seg)):
            continue
        spec = np.fft.fft(seg * w) if is_complex else np.fft.rfft(seg * w)
        p = spec.real ** 2 + spec.imag ** 2
        acc = p if acc is None else acc + p
        n_used += 1
    if n_used == 0:
        raise ValueError("no NaN-free segment in the record")
    psd = acc * scale / n_used
    if is_complex:
        f = np.fft.fftshift(np.fft.fftfreq(nperseg, 1 / fs))
        psd = np.fft.fftshift(psd)
    else:
        f = np.fft.rfftfreq(nperseg, 1 / fs)
        psd[1:] *= 2
        if nperseg % 2 == 0:
            psd[-1] /= 2
    return f, psd, n_used


@dataclass
class NoiseSpectrum:
    """Input-referred noise spectrum measured through a lock-in.

    f    : input frequency (Hz)
    psd  : one-sided power spectral density (unit^2/Hz), input referred (gain removed)
    unit : 'V' or 'A'
    """

    f: np.ndarray
    psd: np.ndarray
    unit: str
    f_ref: float
    df: float
    n_segments: int
    name: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def asd(self) -> np.ndarray:
        """Amplitude spectral density, unit/sqrt(Hz)."""
        return np.sqrt(self.psd)

    @property
    def enbw_bin(self) -> float:
        """Noise bandwidth of one bin (Hann window): 1.5 x bin spacing."""
        return 1.5 * self.df

    def band(self, fmin: float, fmax: float) -> "NoiseSpectrum":
        m = (self.f >= fmin) & (self.f <= fmax)
        return NoiseSpectrum(self.f[m], self.psd[m], self.unit, self.f_ref, self.df,
                             self.n_segments, self.name, dict(self.meta))

    def floor(self, fmin: Optional[float] = None, fmax: Optional[float] = None) -> float:
        """Median ASD over a band: a robust 'background level' (lines are ignored)."""
        s = self.band(fmin if fmin is not None else self.f[0], fmax if fmax is not None else self.f[-1])
        return float(np.sqrt(np.nanmedian(s.psd)))

    def at(self, f: float) -> float:
        """ASD at frequency f (nearest bin)."""
        return float(self.asd[np.argmin(np.abs(self.f - f))])


def choose_nperseg(fs: float, n: int, resolution: float, min_segments: int = 4) -> int:
    """Power-of-two segment length for ~`resolution` Hz bins, keeping >= min_segments.

    With 50 % overlap, k segments need n >= nperseg (k + 1) / 2.
    """
    want = 2 ** int(np.round(np.log2(fs / resolution)))
    longest = 2 ** int(np.floor(np.log2(max(2 * n / (min_segments + 1), 16))))
    nper = min(want, longest)
    if nper < want:
        warnings.warn(f"record too short for {resolution:g} Hz resolution; using "
                      f"{fs / nper:.3g} Hz. Record longer to resolve finer.")
    return max(nper, 16)


def input_spectrum(z: np.ndarray, fs: float, f_ref: float, gain: float = 1.0,
                   unit: str = "V", resolution: float = 0.05, fmin: float = 0.0,
                   fmax: Optional[float] = None, tau: Optional[float] = None,
                   slope_db: int = 24, orientation: str = "auto",
                   mains: Optional[float] = None, name: str = "",
                   min_segments: int = 4) -> NoiseSpectrum:
    """Input noise spectrum from a lock-in X+iY record taken at reference f_ref.

    z          : complex X + iY samples (NaN where packets were dropped)
    fs         : sample rate (Hz)
    f_ref      : detection frequency of the record (Hz)
    gain       : preamp gain to divide out (e.g. 100 for a x100 voltage preamp,
                 1e6 for a 1e6 V/A current preamp -> unit='A')
    resolution : target frequency bin spacing (Hz)
    fmin, fmax : input frequency range kept (default 0 - 2 f_ref; anything above 0 up
                 to f_ref + fs/2 is unambiguous)
    tau        : lock-in time constant used for the record; if given, the RC
                 filter roll-off is divided out (small for tau <= 100 us)
    orientation: 'auto' checks mains harmonics to decide whether an input at
                 f_ref + nu appears at +nu ('normal') or -nu ('flipped');
                 or force 'normal' / 'flipped'
    mains      : 50 or 60 Hz; None = detected from the spectrum
    """
    z = np.asarray(z, dtype=complex) / gain
    nper = choose_nperseg(fs, len(z), resolution, min_segments)
    nu, psd, nseg = welch(z, fs, nper)
    df = fs / nper
    if mains is None:
        mains = detect_mains(nu, psd, f_ref, df)

    orient_info: Dict[str, Any] = {"requested": orientation}
    if orientation == "auto":
        score, used = _orientation_score(nu, psd, f_ref, mains, df)
        orient_info.update(score=score, harmonics_used=used)
        if score < -1.0:
            orientation = "flipped"
        else:
            if score < 1.0:
                warnings.warn(
                    f"could not confirm the X/Y sign convention from the {mains:g} Hz "
                    f"harmonics (score {score:.2f}); assuming f = f_ref + nu. Check a known line.")
            orientation = "normal"
    if orientation == "flipped":
        nu = -nu[::-1]
        psd = psd[::-1]
    elif orientation != "normal":
        raise ValueError("orientation must be 'auto', 'normal' or 'flipped'")
    orient_info["used"] = orientation

    if tau:
        psd = psd / rc_power_response(nu, tau, slope_db)

    f = f_ref + nu
    if fmax is None:
        fmax = 2 * f_ref
    fmax = min(fmax, f_ref + 0.45 * fs)
    m = (f > max(fmin, 0.0)) & (f <= fmax)
    meta = {"fs": fs, "nperseg": nper, "gain": gain, "tau": tau, "slope_db": slope_db,
            "orientation": orient_info, "mains": mains}
    return NoiseSpectrum(f[m], psd[m], unit, f_ref, df, nseg, name, meta)


def block_spectra(z: np.ndarray, fs: float, like: NoiseSpectrum, block: float) -> List[NoiseSpectrum]:
    """Spectra of consecutive `block`-second pieces of a record, analysed like `like`.

    Same reference, gain, band, orientation and mains as the full-record spectrum, and
    the same bin width where the block is long enough (a line must not be smeared wider
    than the lock-in filter). Short blocks average fewer segments, so their spectra
    scatter more; blocks of >= 3/resolution seconds give >= 5 averages.
    """
    m = like.meta
    n = int(block * fs)
    res = max(like.df, 1.0 / block)
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(len(z) // n):
            piece = z[i * n:(i + 1) * n]
            if np.isfinite(piece).mean() < 0.5:
                continue
            try:
                spec = input_spectrum(piece, fs, like.f_ref, gain=m["gain"], unit=like.unit,
                                      resolution=res, fmin=like.f[0] - like.df,
                                      fmax=like.f[-1], tau=m["tau"], slope_db=m["slope_db"],
                                      orientation=m["orientation"]["used"],
                                      mains=m["mains"], name=like.name, min_segments=1)
            except ValueError:  # piece lost to dropped packets
                continue
            spec.meta["block"] = i  # block number, to pair blocks of different channels
            out.append(spec)
    return out


def _peak_ratio(nu: np.ndarray, psd: np.ndarray, pos: float, df: float) -> float:
    """Peak power near `pos` divided by the local median background."""
    near = np.abs(nu - pos) <= 2.5 * df
    ring = (np.abs(nu - pos) > 4 * df) & (np.abs(nu - pos) <= max(1.0, 40 * df))
    if not near.any() or ring.sum() < 5:
        return np.nan
    return float(psd[near].max() / np.median(psd[ring]))


def detect_mains(nu: np.ndarray, psd: np.ndarray, f_ref: float, df: float) -> float:
    """50 or 60 Hz: whichever family of harmonics stands out more (60 if neither does)."""
    lo, hi = 0.9 * nu[0], 0.9 * nu[-1]   # stay clear of the decimation filter's edge
    scores = {}
    for m in (50.0, 60.0):
        logs = []
        for k in range(1, 10):
            p = k * m - f_ref
            if abs(p) < 5 * df or not (lo < p < hi and lo < -p < hi):
                continue
            r = max(_peak_ratio(nu, psd, p, df), _peak_ratio(nu, psd, -p, df))  # either X/Y sign
            if np.isfinite(r):
                logs.append(np.log10(r))
        scores[m] = float(np.mean(logs)) if logs else 0.0
    return 50.0 if scores[50.0] > scores[60.0] + 0.3 else 60.0


def _orientation_score(nu, psd, f_ref, mains, df) -> Tuple[float, List[int]]:
    """Sum over usable mains harmonics of log10(power at +nu_k / power at -nu_k).

    Harmonics outside the kept band (above 2 f_ref) count too: the stream reaches far beyond
    it, so a 25 Hz survey (band 0 - 50 Hz) still sees 60, 120, 180 Hz at +35, +95, +155 Hz.
    What sits at -nu_k comes from input frequency |f_ref - nu_k| (directly below 2 f_ref, as
    the image of the sum frequency above), so a harmonic is skipped when that is one too.
    """
    lo, hi = 0.9 * nu[0], 0.9 * nu[-1]   # stay clear of the decimation filter's edge
    harmonics = [k * mains for k in range(1, 64)]
    score, used = 0.0, []
    for k, fk in enumerate(harmonics, start=1):
        p = fk - f_ref  # where harmonic k lands if orientation is normal
        if abs(p) < 5 * df or not (-f_ref < p < hi) or not (lo < -p < hi):
            continue
        # skip if the mirror position is itself fed by (near) another harmonic or DC
        mirror_f = abs(f_ref - p)
        if any(abs(mirror_f - h) < 5 * df for h in harmonics) or mirror_f < 5 * df:
            continue
        r_plus = _peak_ratio(nu, psd, p, df)
        r_minus = _peak_ratio(nu, psd, -p, df)
        if np.isfinite(r_plus) and np.isfinite(r_minus):
            score += np.log10(r_plus / r_minus)
            used.append(k)
        if len(used) >= 6:
            break
    return float(score), used


# --------------------------------------------------------------------------
# Lines
# --------------------------------------------------------------------------

def find_lines(spec: NoiseSpectrum, threshold: float = 10.0, background_width: float = 2.0,
               fundamentals: Optional[Mapping[str, float]] = None, max_lines: int = 60,
               fmin: Optional[float] = None, fmax: Optional[float] = None) -> List[Dict[str, Any]]:
    """Find narrow lines that stick out of the smooth background.

    threshold        : minimum peak power / background power (10 = 10 dB)
    background_width : running-median window (Hz) used as the background
    fundamentals     : {'mains': 60, 'pulse tube': 1.4, ...}; lines within a bin
                       of a harmonic are tagged with it. Defaults to mains.

    Returns a list (strongest first) of dicts with
      f (Hz), rms (input-referred line amplitude), peak_asd, background_asd, snr_db, tag
    """
    fundamentals = dict(fundamentals or {"mains": spec.meta.get("mains", 60.0)})
    f, psd, df = spec.f, spec.psd, spec.df
    # the median window must be much wider than a line (Hann main lobe = 4 bins)
    win = max(15, int(round(background_width / df))) | 1
    bg = np.exp(ndimage.median_filter(np.log(np.maximum(psd, 1e-300)), size=win, mode="nearest"))
    over = psd > threshold * bg
    if fmin is not None:
        over &= f >= fmin
    if fmax is not None:
        over &= f <= fmax
    labels, n = ndimage.label(over)
    lines = []
    for i in range(1, n + 1):
        idx = np.nonzero(labels == i)[0]
        lo, hi = max(idx[0] - 2, 0), min(idx[-1] + 3, len(f))
        k = idx[np.argmax(psd[idx])]
        power = float(np.sum(np.clip(psd[lo:hi] - bg[lo:hi], 0, None)) * df)
        lines.append({
            "f": float(f[k]),
            "rms": np.sqrt(power),
            "peak_asd": float(np.sqrt(psd[k])),
            "background_asd": float(np.sqrt(bg[k])),
            "snr_db": float(10 * np.log10(psd[k] / bg[k])),
            "tag": _tag(f[k], fundamentals, tol=max(2 * df, 0.05)),
        })
    lines.sort(key=lambda d: -d["rms"])
    return lines[:max_lines]


def _tag(f: float, fundamentals: Mapping[str, float], tol: float) -> str:
    for name, f0 in fundamentals.items():
        if f0 <= 0:
            continue
        k = int(round(f / f0))
        if k >= 1 and abs(f - k * f0) <= tol * max(1, k ** 0.5):
            return f"{name} x{k}" if k > 1 else name
    return ""


# --------------------------------------------------------------------------
# Predicting lock-in noise and ranking frequencies
# --------------------------------------------------------------------------

def predicted_noise(spec: NoiseSpectrum, f_candidates: np.ndarray, filt: MeasurementFilter,
                    min_coverage: float = 0.97, chunk: int = 128) -> np.ndarray:
    """Predicted rms noise of one lock-in reading (X, input referred) at each candidate f.

    NaN where the filter reaches outside the measured band (coverage < min_coverage).
    """
    fc = np.atleast_1d(np.asarray(f_candidates, dtype=float))
    f, S, df = spec.f, spec.psd, spec.df
    out = np.full(fc.shape, np.nan)
    enbw_of = _enbw_interpolator(filt, fc)
    # Each PSD bin is the mean density over [f - df/2, f + df/2]; integrate |H|^2 across
    # the bin (sub-sampled) so that bins wider than the filter are handled correctly.
    w = float(np.min(enbw_of(fc)))
    n_sub = int(min(64, max(1, np.ceil(5 * df / w))))
    if df > 2 * w:
        warnings.warn(f"survey bins ({df:.3g} Hz) are wider than the measurement bandwidth "
                      f"({w:.3g} Hz): predictions near lines are approximate. Record longer / "
                      "use a finer resolution.")
    offsets = ((np.arange(n_sub) + 0.5) / n_sub - 0.5) * df
    for i0 in range(0, len(fc), chunk):
        c = fc[i0:i0 + chunk, None]
        h2 = 0.0
        for o in offsets:
            fo = f[None, :] + o
            h2 = h2 + filt.power_response(fo - c, c) + filt.power_response(fo + c, c)
        h2 = h2 / n_sub
        var = 0.5 * np.sum(h2 * S[None, :], axis=1) * df
        cover = 0.5 * np.sum(h2, axis=1) * df
        ok = cover / enbw_of(c[:, 0]) >= min_coverage
        out[i0:i0 + chunk] = np.where(ok, np.sqrt(var), np.nan)
    return out


def _enbw_interpolator(filt: MeasurementFilter, fc: np.ndarray):
    """ENBW as a function of reference frequency (only the sync filter depends on it)."""
    if not filt.sync:
        e = filt.enbw(1.0)
        return lambda f: np.full(np.shape(f), e)
    grid = np.linspace(fc.min(), fc.max(), 16) if fc.max() > fc.min() else fc[:1]
    vals = np.array([filt.enbw(g) for g in grid])
    return lambda f: np.interp(f, grid, vals)


def mains_related(f: np.ndarray, mains: float = 60.0, max_den: int = 4,
                  tol: float = 0.5) -> np.ndarray:
    """True where f is within tol of mains * p/q for q <= max_den (15, 20, 30, 40, 45 ... Hz)."""
    f = np.asarray(f, dtype=float)
    bad = np.zeros(f.shape, bool)
    for q in range(1, max_den + 1):
        step = mains / q
        k = np.round(f / step)
        bad |= (k >= 1) & (np.abs(f - k * step) <= tol)
    return bad


# --------------------------------------------------------------------------
# Time-domain checks (step 2: a candidate at real settings)
# --------------------------------------------------------------------------

def block_means(x: np.ndarray, n: int, window: str = "boxcar") -> np.ndarray:
    """Averages of consecutive blocks of n samples; blocks containing NaN are dropped.

    window='hann' weights each block with a Hann window (normalised to unit sum):
    much less leakage from lines away from f, ~1.2x the white noise of a plain mean.
    """
    x = np.asarray(x)
    m = len(x) // n
    blocks = x[: m * n].reshape(m, n)
    if window == "hann" and n > 2:
        w = signal.get_window("hann", n)
        b = blocks @ (w / w.sum())
    elif window in ("boxcar", "hann"):
        b = blocks.mean(axis=1)
    else:
        raise ValueError("window must be 'boxcar' or 'hann'")
    return b[np.isfinite(b)]


def pixel_noise(x: np.ndarray, fs: float, dwell: float, discard: float = 0.0,
                window: str = "boxcar") -> Dict[str, float]:
    """Scatter of dwell-length averages: the noise of one pixel of a map.

    x may be complex (X + iY): sigma_x and sigma_y are returned separately.
    discard: seconds dropped from the start (settling).
    window : 'boxcar' (plain mean) or 'hann' weighting within each dwell.
    """
    x = np.asarray(x)[int(round(discard * fs)):]
    n = max(1, int(round(dwell * fs)))
    b = block_means(x, n, window)
    if len(b) < 3:
        raise ValueError("record too short for this dwell")
    out = {"n_pixels": int(len(b)), "dwell": n / fs}
    if np.iscomplexobj(b):
        out.update(mean_x=float(b.real.mean()), mean_y=float(b.imag.mean()),
                   sigma_x=float(b.real.std(ddof=1)), sigma_y=float(b.imag.std(ddof=1)))
    else:
        out.update(mean=float(b.mean()), sigma=float(b.std(ddof=1)))
    return out


def allan_deviation(x: np.ndarray, fs: float, taus: Optional[Sequence[float]] = None,
                    n_taus: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Overlapping Allan deviation of a real record (longest NaN-free stretch is used).

    Flat-then-falling as 1/sqrt(tau) = white noise (averaging helps); a floor or rise = drift.
    """
    x = np.asarray(x, dtype=float)
    x = _longest_finite(x)
    N = len(x)
    if taus is None:
        taus = np.geomspace(max(2.0 / fs, 1e-3), N / fs / 4, n_taus)
    ms = np.unique(np.clip(np.round(np.asarray(taus) * fs).astype(int), 1, N // 3))
    cs = np.concatenate([[0.0], np.cumsum(x)])
    adev = []
    for m in ms:
        ybar = (cs[m:] - cs[:-m]) / m
        d = ybar[m:] - ybar[:-m]
        adev.append(np.sqrt(0.5 * np.mean(d * d)))
    return ms / fs, np.array(adev)


def _longest_finite(x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    if ok.all():
        return x
    lab, n = ndimage.label(ok)
    if n == 0:
        raise ValueError("no finite samples")
    sizes = ndimage.sum(ok, lab, range(1, n + 1))
    return x[lab == (np.argmax(sizes) + 1)]


def spectrogram(z: np.ndarray, fs: float, f_ref: float, gain: float = 1.0,
                segment: float = 10.0, fmax: Optional[float] = None,
                orientation: str = "normal",
                averages: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time-resolved input PSD: (t [s], f [Hz], psd[f, t]). Use to spot intermittent bursts.

    The record is cut into blocks of `segment` seconds; each block's spectrum is a Welch
    average of ~`averages` half-overlapping sub-segments (less speckle, coarser bins).
    """
    z = np.asarray(z, dtype=complex) / gain
    if orientation == "flipped":
        z = np.conj(z)
    nblk = int(segment * fs)
    nb = len(z) // nblk
    if nb < 1:
        raise ValueError("record shorter than one segment")
    nper = 2 ** int(np.floor(np.log2(nblk * 2 / (averages + 1))))
    nu = np.fft.fftshift(np.fft.fftfreq(nper, 1 / fs))
    cols = []
    for b in range(nb):
        try:
            _, p, _ = welch(z[b * nblk:(b + 1) * nblk], fs, nper)
        except ValueError:  # block lost to dropped packets
            p = np.full(nper, np.nan)
        cols.append(p)
    P = np.array(cols).T
    t = (np.arange(nb) + 0.5) * segment
    f = f_ref + nu
    fmax = fmax or 2 * f_ref
    m = (f > 0) & (f <= fmax)
    return t, f[m], P[m]


# --------------------------------------------------------------------------
# Decimation for storage
# --------------------------------------------------------------------------

def decimate(z: np.ndarray, q: int) -> np.ndarray:
    """Anti-aliased decimation by q that survives NaN gaps (each finite stretch separately)."""
    z = np.asarray(z)
    if q <= 1:
        return z.copy()
    out = np.full(len(z) // q, np.nan, dtype=np.result_type(z.dtype, np.float64))
    lab, _ = ndimage.label(np.isfinite(z))
    first = True
    for s in ndimage.find_objects(lab):
        a, b = s[0].start, s[0].stop
        a = -(-a // q) * q  # start on the output grid so samples stay time-aligned
        if b - a < 30 * q:
            continue
        d = _decimate_chunk(z[a:b], q)
        start = a // q
        stop = min(start + len(d), len(out))
        out[start:stop] = d[: stop - start]
        if not first and start > 0:
            out[start - 1] = np.nan  # keep every gap visible after decimation
        first = False
    return out


def _decimate_chunk(x: np.ndarray, q: int) -> np.ndarray:
    # cascade stages of <= 8 (scipy's recommendation for the FIR design)
    y = x
    while q > 1:
        k = 8 if q % 8 == 0 else (4 if q % 4 == 0 else (2 if q % 2 == 0 else q))
        y = signal.decimate(y, k, ftype="fir", zero_phase=True)
        q //= k
    return y
