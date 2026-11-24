"""dm.Station / dm.AsyncStation: measureme's stations plus streaming, against fake instruments."""
import numpy as np
import pytest

import sweep
import sweep.sweep_load
import dmz_measure as dm
from dmz_measure.simulate import FakeGate, fake_cryofree

CONFIG = {"SR860_A": "9-12", "SR860_B": "15-14", "SR860_C": "15-11", "Vbg2": "13", "Vtg2": "16"}
GAINS = [("SR860_A", 1e6), ("SR860_B", 100), ("SR860_C", 100)]


def model_R(name, v):
    """R the fake device shows at gate v once settled (Ohm)."""
    peaks = {"SR860_B": [(0.25, 3.0, 0.15), (-0.55, 0.8, 0.03)],
             "SR860_C": [(0.25, 2.5, 0.2), (-0.55, 0.6, 0.03)]}[name]
    r0 = {"SR860_B": 1000.0, "SR860_C": 600.0}[name]
    r = lambda x: 1 + sum(a / (1 + ((x - v0) / w) ** 2) for v0, a, w in peaks)
    return r0 * r(np.asarray(v)) / r(0.0)


@pytest.fixture()
def rig():
    fakes = fake_cryofree(seed=21)
    yield fakes
    for f in fakes.values():
        f.close()


def make_station(rig, tmp_path, cls=None):
    s = (cls or dm.Station)(CONFIG, basedir=str(tmp_path), verbose=False)
    for name, gain in GAINS:
        s.fp(rig[name].X, gain=gain)
        s.fp(rig[name].Y, gain=gain)
    s.fp(rig["Kbg2"].curr)
    return s


def R_from(d, name):
    return np.real((d[f"{name}_X"] + 1j * d[f"{name}_Y"]) / (d["SR860_A_X"] + 1j * d["SR860_A_Y"]))


def test_stations_are_measureme_stations():
    assert issubclass(dm.Station, sweep.Station)
    assert issubclass(dm.AsyncStation, sweep.AsyncStation)
    for m in ("fp", "plot", "sweep", "megasweep", "watch", "ramp", "read", "add_comment"):
        assert getattr(dm.Station, m) is getattr(sweep.Station, m)


def test_normal_sweep_is_measureme(rig, tmp_path):
    for n, _ in GAINS:
        rig[n].write("OFLT 8")  # 10 ms, so a short delay settles
    s = make_station(rig, tmp_path, dm.AsyncStation)
    V = np.linspace(-0.2, 0.6, 6)
    r = s.sweep(rig["Kbg2"].volt, V, delay=0.2)
    assert r.metadata["function"] == "sweep"
    d = sweep.sweep_load.pload1d(str(tmp_path), r.id)
    np.testing.assert_allclose(R_from(d, "SR860_B"), model_R("SR860_B", V), rtol=0.05)


def test_stream_sweep_same_columns_restores_and_refilters(rig, tmp_path):
    a = rig["SR860_A"]
    before = (a.freq, a.slvl, a.oflt, a.sync, rig["SR860_B"].oflt)
    s = make_station(rig, tmp_path)
    V = np.linspace(-0.2, 0.6, 9)
    r = s.stream_sweep(rig["Kbg2"].volt, V, dwell=0.15)
    md = r.metadata
    assert md["function"] == "stream_sweep" and md["type"] == "1D"
    assert md["columns"] == ["time", "Kbg2_source_voltage"] + [
        f"{n}_{c}" for n, _ in GAINS for c in "XY"] + ["Kbg2_sense_current"]
    assert md["stream"]["tau"] == pytest.approx(1e-3) and md["stream"]["sync"]
    assert (a.freq, a.slvl, a.oflt, a.sync, rig["SR860_B"].oflt) == before  # restored, excitation untouched
    d = sweep.sweep_load.pload1d(str(tmp_path), r.id)                      # measureme reads it
    np.testing.assert_allclose(R_from(d, "SR860_B"), model_R("SR860_B", V), rtol=0.03)
    assert np.all(np.abs(d["Kbg2_sense_current"]) < 1e-10)                # read normally
    re = dm.refilter(r.id, dwell=0.05, window="boxcar", basedir=str(tmp_path))
    np.testing.assert_array_equal(re["xs"], V)
    np.testing.assert_allclose(R_from(re, "SR860_B"), model_R("SR860_B", V), rtol=0.05)
    same = dm.refilter(r.id, basedir=str(tmp_path))                        # same settings
    np.testing.assert_allclose(R_from(same, "SR860_B"), R_from(d, "SR860_B"), rtol=2e-3)


def test_stream_megasweep(rig, tmp_path):
    top = FakeGate("Ktg2")
    s = make_station(rig, tmp_path, dm.AsyncStation)
    fast = np.linspace(-0.2, 0.6, 5)
    r = s.stream_megasweep(top.volt, [0.0, 0.5], rig["Kbg2"].volt, fast, slow_delay=0.2, dwell=0.1)
    assert r.metadata["function"] == "stream_megasweep" and r.metadata["type"] == "2D"
    d = sweep.sweep_load.pload2d(str(tmp_path), r.id)
    assert d["SR860_B_X"].shape == (2, 5)
    for row in range(2):
        np.testing.assert_allclose(R_from(d, "SR860_B")[row], model_R("SR860_B", fast), rtol=0.04)
    re = dm.refilter(r.id, dwell=0.05, basedir=str(tmp_path))
    assert re["SR860_B_X"].shape == (2, 5)


def test_station_survey_and_check_use_fp_gains(rig, tmp_path):
    rig["SR860_A"].amplitude(0)  # excitation off by hand
    s = make_station(rig, tmp_path)
    sv = s.noise_survey(current="SR860_A", duration=8, resolution=0.3, rate=5e3, fmax=150)
    assert sv.spectra["SR860_B"].floor(30, 55) == pytest.approx(1e-8, rel=0.3)  # gain 100 applied
    rig["SR860_A"].amplitude(1.0)
    ck = s.check_frequencies([17.77], current="SR860_A", duration=2, dwell=0.1, block=1)
    assert ck.rows[0]["SR860_B_value"] == pytest.approx(1000, rel=0.05)
    assert dm.load_run(sv.run_id, str(tmp_path)).metadata["type"] == "noise_survey"


def test_point_noise_ignores_curvature():
    x = np.linspace(-1, 1, 201)
    R = model_R("SR860_B", x)
    for sigma in (1.0, 5.0):
        est = [dm.point_noise(R + np.random.default_rng(k).normal(0, sigma, len(x))) for k in range(10)]
        assert np.median(est) == pytest.approx(sigma, rel=0.2)


def test_fake_gate_ramps_like_qcodes_and_gate_steps():
    g = FakeGate("Kx", set_time=0.01)
    g.volt.step, g.volt.inter_delay = 0.005, 0.005
    t = __import__("time").perf_counter()
    g.volt(0.02)                               # 4 chunks: 4 x 10 ms + 3 x 5 ms
    assert __import__("time").perf_counter() - t == pytest.approx(0.055, abs=0.03)
    with dm.gate_steps(g.volt, max_step=0.02):
        assert g.volt.step == 0.02 and g.volt.inter_delay == pytest.approx(0.02)  # same 1 V/s cap
        __import__("time").sleep(0.05)         # in a sweep the dwell has passed by now
        t = __import__("time").perf_counter()
        g.volt(0.04)                           # one command
        assert __import__("time").perf_counter() - t < 0.03
    assert (g.volt.step, g.volt.inter_delay) == (0.005, 0.005)   # restored


def test_timing_breaks_down_a_point(rig, tmp_path):
    gate = rig["Kbg2"]
    gate.set_time, gate.read_time = 0.02, 0.03
    gate.volt.step, gate.volt.inter_delay = 0.005, 0.005
    s = make_station(rig, tmp_path, dm.AsyncStation)
    t = s.timing(gate.volt, np.linspace(0, 0.2, 21), n=8, delay=0.5, settle=0.1, dwell=0.3)
    assert t["chunks"] == 2                                 # 10 mV increments, 5 mV qcodes step
    assert t["round_trip"] == pytest.approx(0.02, abs=0.01)
    assert t["reads"]["Kbg2_sense_current"]["median"] == pytest.approx(0.03, abs=0.015)
    assert gate.volt() == pytest.approx(0.0)                # returned to the start


def test_step_response_sees_the_gate_line(rig, tmp_path):
    rig["Kbg2"].rc = 0.04                                   # a slow gate line
    s = make_station(rig, tmp_path)
    st = s.step_response(rig["Kbg2"].volt, 0.0, 0.2, n=4, hold=0.6)
    assert st.settle > st.lockin_settle + 0.05              # slower than the lock-in alone
    assert st.settle < 0.45


def test_stream_sweep_reads_during_the_dwell(rig, tmp_path):
    rig["Kbg2"].read_time = 0.15                            # slow leakage reads
    s = make_station(rig, tmp_path, dm.AsyncStation)
    V = np.linspace(0, 0.2, 8)
    r = s.stream_sweep(rig["Kbg2"].volt, V, dwell=0.2, settle=0.1)
    t_set = dm.load_run(r.id, str(tmp_path)).arrays("stream")["0_t_set0"]
    per_point = np.median(np.diff(t_set))
    assert per_point < 0.02 + 0.1 + 0.2 + 0.06             # the 0.15 s reads are hidden in the dwell
    d = sweep.sweep_load.pload1d(str(tmp_path), r.id)
    assert np.all(np.isfinite(d["Kbg2_sense_current"]))


def test_step_response_from_a_sweep_run(rig, tmp_path):
    rig["Kbg2"].rc = 0.05                                   # gate line slower than the lock-in
    s = make_station(rig, tmp_path)
    V = np.linspace(-0.1, 0.5, 16)                          # steep part of the fake device
    r = s.stream_sweep(rig["Kbg2"].volt, V, dwell=0.35, settle=0.03)
    st = dm.step_response(r.id, basedir=str(tmp_path))
    assert st.settle > 0.15                                 # sync period + ~5 x 50 ms gate RC
    assert st.settle < 0.38


def test_fast_ramp_keeps_the_chunks_and_drops_the_waits():
    import time

    g = FakeGate("Kf", set_time=0.02, write_time=0.002)     # 2 ms write + 18 ms *OPC? round trip
    g.volt.step, g.volt.inter_delay = 0.005, 0.005
    t = time.perf_counter()
    g.volt(0.2)                                             # 40 chunks, each waits for *OPC?
    slow = time.perf_counter() - t
    g.command_times.clear()
    time.sleep(0.05)
    with dm.fast_ramp(g.volt):
        assert g.block_during_ramp() is False
        t = time.perf_counter()
        g.volt(0.0)
        fast = time.perf_counter() - t
    assert g.block_during_ramp() is True and g.volt.step == 0.005   # restored, step untouched
    v = np.array([c[1] for c in g.command_times])
    tc = np.array([c[0] for c in g.command_times])
    assert len(v) == 40 and np.max(np.abs(np.diff(np.r_[0.2, v]))) <= 0.005 + 1e-9  # same 5 mV chunks
    assert np.min(np.diff(tc)) >= 0.005 - 1e-3                # still >= inter_delay apart
    assert slow > 0.9 and fast < 0.4                           # ~1.0 s -> ~0.3 s
