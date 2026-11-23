import warnings

import numpy as np
import pytest

import dmz_measure as dm
from dmz_measure.noise import decimate, mains_related
from dmz_measure.simulate import NoiseEnvironment, SoftwareLockin

FS = 1.25e6 / 2 ** 8  # 4882.8 Hz
WHITE = 1e-6
# (weak lines are fine too, but their fitted amplitude scatters by ~10 % at 7 averages)
LINES = [(60.0, 3e-6), (120.0, 1.5e-6), (37.5, 2e-6), (13.0, 2e-6)]


@pytest.fixture(scope="module")
def input_signal():
    env = NoiseEnvironment(FS, white=WHITE, lines=LINES, seed=3)
    return env.chunk(int(120 * FS))


def demod(v, f, tau, sync=False, flip=False):
    li = SoftwareLockin(FS, tau, 24, sync)
    ph = 2 * np.pi * f * np.arange(1, len(v) + 1) / FS
    z = li.process(v, ph, f)
    return np.conj(z) if flip else z


@pytest.mark.parametrize("flip", [False, True])
def test_spectrum_calibration_and_orientation(input_signal, flip):
    z = demod(input_signal, 100.0, 100e-6, flip=flip) * 100  # x100 preamp
    s = dm.input_spectrum(z, FS, 100.0, gain=100, resolution=0.05, tau=100e-6)
    assert s.meta["orientation"]["used"] == ("flipped" if flip else "normal")
    assert s.floor(1, 199) == pytest.approx(WHITE, rel=0.05)
    lines = dm.find_lines(s)

    def nearest(f):
        ln = min(lines, key=lambda d: abs(d["f"] - f))
        assert abs(ln["f"] - f) < 0.2, f"line {f} Hz not found"
        return ln

    for f, rms in LINES:
        assert nearest(f)["rms"] == pytest.approx(rms, rel=0.1)
    assert nearest(60)["tag"] == "mains" and nearest(120)["tag"] == "mains x2"
    assert s.f.min() > 0 and s.f.max() <= 200


@pytest.mark.parametrize("filt", [dm.MeasurementFilter(tau=0.1),
                                  dm.MeasurementFilter(tau=1e-3, sync=True, dwell=0.2)])
def test_predicted_noise_matches_direct_lockin(input_signal, filt):
    z = demod(input_signal, 100.0, 100e-6)
    s = dm.input_spectrum(z, FS, 100.0, resolution=0.05, tau=100e-6)
    for fc in (24.3, 58.9):  # quiet spot, and next to the 60 Hz line
        pred = dm.predicted_noise(s, [fc], filt)[0]
        x = demod(input_signal, fc, filt.tau, sync=filt.sync).real[int(2 * FS):]
        meas = dm.pixel_noise(x, FS, filt.dwell)["sigma"] if filt.dwell else x.std()
        assert meas == pytest.approx(pred, rel=0.12), (fc, filt)


def test_rank_avoids_lines_and_mains_fractions(input_signal):
    z = demod(input_signal, 100.0, 100e-6)
    s = dm.input_spectrum(z, FS, 100.0, resolution=0.05, tau=100e-6, name="B")
    r = dm.rank_frequencies({"B": s}, dm.MeasurementFilter(tau=0.3), fmin=5, fmax=95)
    assert len(r.best) == 8
    for b in r.best:
        assert min(abs(b["f"] - f) for f, _ in LINES) > 1.0
        assert not mains_related(np.array([b["f"]]), tol=0.5)[0]


def test_mains_related():
    f = np.array([15.0, 20.0, 30.0, 40.0, 45.0, 60.0, 13.3, 17.7, 22.5])
    np.testing.assert_array_equal(mains_related(f, 60, 4, 0.3),
                                  [True, True, True, True, True, True, False, False, False])


def test_pixel_noise_and_allan_white():
    rng = np.random.default_rng(0)
    fs = 1000.0
    x = rng.normal(0, 1, 200_000)
    pn = dm.pixel_noise(x, fs, dwell=0.1)
    assert pn["sigma"] == pytest.approx(1 / np.sqrt(100), rel=0.05)
    t, a = dm.allan_deviation(x, fs)
    slope = np.polyfit(np.log(t[3:-5]), np.log(a[3:-5]), 1)[0]
    assert slope == pytest.approx(-0.5, abs=0.07)


def test_welch_skips_nan_segments():
    rng = np.random.default_rng(1)
    z = (rng.normal(size=100_000) + 1j * rng.normal(size=100_000)) * np.sqrt(FS / 2) * 1e-6
    z[50_000:50_100] = np.nan
    s = dm.input_spectrum(z, FS, 100.0, resolution=1.0, orientation="normal")
    assert np.isfinite(s.psd).all()
    assert s.floor() == pytest.approx(1e-6, rel=0.05)


def test_decimate_keeps_alignment_and_gaps():
    fs = 40_000.0
    t = np.arange(400_000) / fs
    z = np.exp(2j * np.pi * 7.0 * t)
    z[100_000:100_256] = np.nan
    d = decimate(z, 32)
    td = np.arange(len(d)) * 32 / fs
    ok = np.isfinite(d)
    assert (~ok).sum() >= 1
    good = ok & (np.abs(td - 2.5) > 0.2) & (td > 0.2) & (td < td[-1] - 0.2)
    np.testing.assert_allclose(d[good], np.exp(2j * np.pi * 7.0 * td[good]), atol=1e-3)


def test_lockin_helpers():
    assert dm.lockin.tau_from_index(0) == pytest.approx(1e-6)
    assert dm.lockin.tau_from_index(21) == pytest.approx(3e4)
    assert dm.lockin.tau_index(0.1) == 10  # manual: OFLT 10 = 100 ms
    assert dm.recommended_tau(17.0) == pytest.approx(1e-3)
    assert dm.recommended_tau(100.0) == pytest.approx(300e-6)
    assert dm.enbw_rc(0.1, 24) == pytest.approx(5 / 64 / 0.1)
    assert dm.MeasurementFilter(tau=0.1).enbw(10) == pytest.approx(5 / 64 / 0.1, rel=1e-3)
    assert dm.MeasurementFilter(tau=None, dwell=0.2).enbw(10) == pytest.approx(2.5, rel=1e-3)
    assert dm.settling_time(1e-3, 17.0, sync=True) == pytest.approx(7.1e-3 + 1 / 17)
