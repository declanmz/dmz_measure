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


def demod(v, f, tau, sync=False, flip=False, advanced=False):
    li = SoftwareLockin(FS, tau, 24, sync, advanced=advanced)
    ph = 2 * np.pi * f * np.arange(1, len(v) + 1) / FS
    z = li.process(v, ph, f)
    return np.conj(z) if flip else z


@pytest.mark.parametrize("flip", [False, True])
def test_spectrum_calibration_and_orientation(input_signal, flip):
    z = demod(input_signal, 100.0, 100e-6, flip=flip) * 100  # x100 preamp
    s = dm.noise.input_spectrum(z, FS, 100.0, gain=100, resolution=0.05, tau=100e-6)
    assert s.meta["orientation"]["used"] == ("flipped" if flip else "normal")
    assert s.floor(1, 199) == pytest.approx(WHITE, rel=0.05)
    lines = dm.noise.find_lines(s)

    def nearest(f):
        ln = min(lines, key=lambda d: abs(d["f"] - f))
        assert abs(ln["f"] - f) < 0.2, f"line {f} Hz not found"
        return ln

    for f, rms in LINES:
        assert nearest(f)["rms"] == pytest.approx(rms, rel=0.1)
    assert nearest(60)["tag"] == "mains" and nearest(120)["tag"] == "mains x2"
    assert s.f.min() > 0 and s.f.max() <= 200


MF = dm.lockin.MeasurementFilter


@pytest.mark.parametrize("filt", [MF(tau=0.1),
                                  MF(tau=0.03, advanced=True),
                                  MF(tau=1e-3, sync=True, dwell=0.2),
                                  MF(tau=1e-3, sync=True, dwell=0.2, window="hann")])
def test_predicted_noise_matches_direct_lockin(input_signal, filt):
    z = demod(input_signal, 100.0, 100e-6)
    s = dm.noise.input_spectrum(z, FS, 100.0, resolution=0.05, tau=100e-6)
    # quiet spot, 1.1 Hz from the 60 Hz line, and 3 Hz from it
    for fc in (24.3, 58.9, 57.0):
        pred = dm.noise.predicted_noise(s, [fc], filt)[0]
        x = demod(input_signal, fc, filt.tau, sync=filt.sync, advanced=filt.advanced)
        x = x.real[int(2 * FS):]
        meas = (dm.noise.pixel_noise(x, FS, filt.dwell, window=filt.window)["sigma"]
                if filt.dwell else x.std())
        assert meas == pytest.approx(pred, rel=0.12), (fc, filt)


def test_filter_shapes_match_the_manual_and_theory():
    tau = 0.1
    nu = np.linspace(0, 20, 200001)
    rc1 = dm.lockin.rc_power_response(nu, tau, 6)
    g1 = dm.lockin.gaussian_power_response(nu, tau, 6)
    # same noise bandwidth per stage (1/(4 tau)), -3 dB point ~1.5x wider (SR860 manual, App. A)
    assert np.trapezoid(g1, nu) == pytest.approx(1 / (4 * tau), rel=1e-3)
    assert dm.lockin.enbw_rc(tau, 6) == pytest.approx(1 / (4 * tau))
    f3 = lambda h: nu[np.argmax(h < 0.5)]
    assert f3(g1) / f3(rc1) == pytest.approx(1.5, rel=0.03)
    # Hann average: first sidelobe -31.5 dB, ENBW 1.5x the plain average's
    T = 0.3
    x = np.linspace(2.05, 3.0, 20001) / T
    assert 10 * np.log10(dm.lockin.hann_power_response(x, T).max()) == pytest.approx(-31.5, abs=0.3)
    assert MF(tau=None, dwell=T, window="hann").enbw(10) == pytest.approx(0.75 / T, rel=1e-3)
    # the Advanced filter is far stronger than RC 3 Hz away (tau = 0.3 s, 24 dB/oct)
    assert dm.lockin.gaussian_power_response(3.0, 0.3) < 1e-10 < dm.lockin.rc_power_response(3.0, 0.3)


def _survey_from(v, duration):
    from dmz_measure.survey import _build_survey

    z = demod(v, 100.0, 100e-6)
    st = dict(f_ref=100.0, duration=duration, fmin=0.0, fmax=200.0, resolution=0.05,
              tau=100e-6, f_det={"B": 100.0}, units={"B": "V"})
    return _build_survey({"B": z}, {"B": FS}, st)


def test_best_frequencies_avoid_lines_and_mains_fractions(input_signal):
    sv = _survey_from(input_signal, 120)
    r = dm.best_frequencies(sv, 5, 95, tau=0.3, n=5)
    assert len(r.frequencies) == 5
    for f0 in r.frequencies:
        assert min(abs(f0 - f) for f, _ in LINES) > 1.0
        assert not mains_related(np.array([f0]), tol=0.5)[0]
    rows = dm.expected_noise(sv, [24.3], tau=0.3)
    pred = dm.noise.predicted_noise(sv.spectra["B"], [24.3], dm.lockin.MeasurementFilter(tau=0.3))[0]
    assert rows[0]["B_mean"] == pytest.approx(pred)


def test_time_variation_catches_bursty_line():
    # a 27 Hz line that is only there 20 % of the time
    T = 200.0
    n = int(T * FS)
    v = NoiseEnvironment(FS, white=WHITE, seed=7).chunk(n)
    t = np.arange(n) / FS
    gate = (t % 50.0) < 10.0
    v = v + gate * np.sqrt(2) * 3e-6 * np.cos(2 * np.pi * 27.0 * t)
    sv = _survey_from(v, T)
    rows = dm.expected_noise(sv, [27.0, 31.0], tau=0.3, block=10)
    burst, quiet = rows
    assert burst["B_max"] / burst["B_mean"] > 1.5    # worst block much worse than average
    assert quiet["B_max"] / quiet["B_mean"] < 1.6
    r = dm.best_frequencies(sv, 22, 32, tau=0.3, block=10, stat="max", n=3, avoid_mains=False)
    assert all(abs(f - 27.0) > 1.0 for f in r.frequencies)


@pytest.mark.parametrize("mains", [50.0, 60.0])
def test_mains_is_detected(mains):
    lines = [(mains * k, 3e-6 / k) for k in (1, 2, 3)] + [(37.5, 2e-6)]
    v = NoiseEnvironment(FS, white=WHITE, lines=lines, seed=4).chunk(int(40 * FS))
    s = dm.noise.input_spectrum(demod(v, 100.0, 100e-6), FS, 100.0, resolution=0.2, tau=100e-6)
    assert s.meta["mains"] == mains


def test_mains_related():
    f = np.array([15.0, 20.0, 30.0, 40.0, 45.0, 60.0, 13.3, 17.7, 22.5])
    np.testing.assert_array_equal(mains_related(f, 60, 4, 0.3),
                                  [True, True, True, True, True, True, False, False, False])


def test_pixel_noise_and_allan_white():
    rng = np.random.default_rng(0)
    fs = 1000.0
    x = rng.normal(0, 1, 200_000)
    pn = dm.noise.pixel_noise(x, fs, dwell=0.1)
    assert pn["sigma"] == pytest.approx(1 / np.sqrt(100), rel=0.05)
    t, a = dm.noise.allan_deviation(x, fs)
    slope = np.polyfit(np.log(t[3:-5]), np.log(a[3:-5]), 1)[0]
    assert slope == pytest.approx(-0.5, abs=0.07)


def test_welch_skips_nan_segments():
    rng = np.random.default_rng(1)
    z = (rng.normal(size=100_000) + 1j * rng.normal(size=100_000)) * np.sqrt(FS / 2) * 1e-6
    z[50_000:50_100] = np.nan
    s = dm.noise.input_spectrum(z, FS, 100.0, resolution=1.0, orientation="normal")
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
    assert dm.lockin.recommended_tau(17.0) == pytest.approx(1e-3)
    assert dm.lockin.recommended_tau(100.0) == pytest.approx(300e-6)
    assert dm.lockin.enbw_rc(0.1, 24) == pytest.approx(5 / 64 / 0.1)
    assert dm.lockin.MeasurementFilter(tau=0.1).enbw(10) == pytest.approx(5 / 64 / 0.1, rel=1e-3)
    assert dm.lockin.MeasurementFilter(tau=None, dwell=0.2).enbw(10) == pytest.approx(2.5, rel=1e-3)
    assert dm.lockin.settling_time(1e-3, 17.0, sync=True) == pytest.approx(7.1e-3 + 1 / 17)
