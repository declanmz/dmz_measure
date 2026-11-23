import gzip
import json
import os

import numpy as np
import pytest

import dmz_measure as dm
from dmz_measure.simulate import fake_cryofree


@pytest.fixture()
def rig():
    fakes = fake_cryofree(seed=11)
    yield fakes
    for f in fakes.values():
        f.close()


def channels(f):
    return {
        "SR860_A": dm.Channel(f["SR860_A"], gain=1e6, unit="A"),
        "SR860_B": dm.Channel(f["SR860_B"], gain=100),
        "SR860_C": dm.Channel(f["SR860_C"], gain=100),
    }


def test_noise_survey_end_to_end(rig, tmp_path):
    a = rig["SR860_A"]
    before = (a.freq, a.slvl, a.oflt)
    res = dm.noise_survey(channels(rig), master=a, f_ref=100.0, duration=8.0, resolution=0.5,
                          target_rate=10e3, settle=0.5, base_port=18700,
                          basedir=str(tmp_path), comment="test survey")
    # settings restored
    assert (a.freq, a.slvl, a.oflt) == before
    assert rig["SR860_B"].sync == 0 and rig["SR860_B"].oflt == before[2]
    # spectra are input referred: B's white level is 1e-6 V/rtHz at the lock-in -> 1e-8 at the sample
    sB = res.spectra["SR860_B"]
    assert sB.floor(30, 55) == pytest.approx(1e-8, rel=0.3)
    assert res.spectra["SR860_A"].unit == "A"
    assert any(ln["tag"] == "mains" for ln in res.lines["SR860_B"])
    # saved in sweep-compatible form
    run = dm.load_run(res.run_id, str(tmp_path))
    assert run.metadata["type"] == "noise_survey"
    assert run.metadata["interrupted"] is False
    assert "start_time" in run.metadata and "end_time" in run.metadata
    assert run.table["f_Hz"].size > 10
    ts = run.arrays("timeseries")
    assert ts["SR860_B_z"].dtype == np.complex64
    assert os.path.exists(os.path.join(run.dir, "plot.png"))
    # ranking works on the result
    r = res.rank(dm.MeasurementFilter(tau=0.3), fmin=5, fmax=95, n_best=3)
    assert len(r.best) == 3


def test_candidate_check_end_to_end(rig, tmp_path):
    a = rig["SR860_A"]
    before = (a.freq, a.slvl, a.oflt, a.sync)
    res = dm.candidate_check(channels(rig), master=a, frequencies=[17.77, 33.3], duration=3.0,
                             dwell=0.2, current="SR860_A", base_port=18710, settle=0.3,
                             basedir=str(tmp_path))
    assert (a.freq, a.slvl, a.oflt, a.sync) == before
    assert [r["f"] for r in res.rows] == [17.77, 33.3]
    r = res.rows[0]
    # A: 1 V / 10 MOhm = 100 nA; B: 100 nA * 1 kOhm = 100 uV  ->  R_B = 1 kOhm
    assert abs(r["SR860_A_X"]) == pytest.approx(1e-7, rel=0.05)
    assert r["SR860_B_R"] == pytest.approx(1000, rel=0.05)
    assert r["SR860_B_rel_noise"] < 0.05
    assert set(res.allan[17.77]) == {"SR860_A", "SR860_B", "SR860_C"}
    run = dm.load_run(res.run_id, str(tmp_path))
    assert run.metadata["type"] == "frequency_check"
    assert run.table["f"].tolist() == [17.77, 33.3]


def test_runwriter_ids_and_sweep_compat(tmp_path):
    os.mkdir(tmp_path / "0")  # pretend sweep already wrote run 0
    with dm.RunWriter(str(tmp_path), "noise_survey", comment="hi") as w:
        w.write_table(["a", "b"], np.c_[[1, 2], [3, 4]])
    assert w.id == 1
    with open(tmp_path / "1" / "metadata.json") as f:
        md = json.load(f)
    for key in ("type", "interrupted", "start_time", "end_time", "columns", "comments"):
        assert key in md
    with gzip.open(tmp_path / "1" / "data.tsv.gz", "rt") as f:
        assert np.loadtxt(f).shape == (2, 2)
