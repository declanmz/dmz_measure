import gzip
import json
import os

import numpy as np
import pytest

import dmz_measure as dm
from dmz_measure.io import RunWriter
from dmz_measure.simulate import fake_cryofree

CONFIG = {"SR860_A": "9-12", "SR860_B": "15-14", "SR860_C": "15-11", "Vbg2": "13", "Vtg2": "16"}
GAINS = {"SR860_A": 1e6, "SR860_B": 100, "SR860_C": 100}


@pytest.fixture()
def rig():
    fakes = fake_cryofree(seed=11)
    yield fakes
    for f in fakes.values():
        f.close()


def test_survey_pick_expected_reload(rig, tmp_path):
    a = rig["SR860_A"]
    a.amplitude(0)  # excitation off by hand
    before = (a.freq, a.slvl, a.oflt)
    sv = dm.noise_survey(CONFIG, GAINS, current="SR860_A", f_ref=100, duration=12, fmax=150,
                         resolution=0.2, rate=5e3, basedir=str(tmp_path))
    # settings restored, the excitation never touched, 'Vbg2'/'Vtg2' ignored
    assert (a.freq, a.slvl, a.oflt) == before
    assert rig["SR860_B"].sync == 0 and rig["SR860_B"].oflt == before[2]
    assert list(sv.spectra) == ["SR860_A", "SR860_B", "SR860_C"]
    assert sv.mains == 60.0
    assert sv.spectra["SR860_B"].f[-1] <= 150
    # input referred: B's white level 1e-6 V/rtHz at the lock-in -> 1e-8 at the sample
    assert sv.spectra["SR860_B"].floor(30, 55) == pytest.approx(1e-8, rel=0.3)
    assert sv.spectra["SR860_A"].unit == "A"

    best = dm.best_frequencies(sv, 15, 35, tau=0.1, n=3, block=4)
    assert len(best.frequencies) == 3 and all(15 <= f <= 35 for f in best.frequencies)
    rows = dm.expected_noise(sv, [13.2943, 25.0], tau=0.1)
    assert len(rows) == 2 and rows[0]["SR860_B_mean"] > 0

    again = dm.load_survey(sv.run_id, str(tmp_path))
    np.testing.assert_allclose(again.spectra["SR860_B"].psd, sv.spectra["SR860_B"].psd, rtol=1e-6)
    run = dm.load_run(sv.run_id, str(tmp_path))
    assert run.metadata["type"] == "noise_survey" and run.metadata["interrupted"] is False
    assert os.path.exists(os.path.join(run.dir, "plot.png"))


def test_check_frequencies(rig, tmp_path):
    a = rig["SR860_A"]
    before = (a.freq, a.slvl, a.oflt, a.sync)
    ck = dm.check_frequencies(CONFIG, [17.77, a.frequency()], GAINS, current="SR860_A",
                              tau=None, sync=True, duration=3.0, dwell=0.1, block=1.0,
                              basedir=str(tmp_path))
    assert (a.freq, a.slvl, a.oflt, a.sync) == before
    assert [r["f"] for r in ck.rows] == [17.77, pytest.approx(13.2943)]
    r = ck.rows[0]
    # A: 1 V / 10 MOhm = 100 nA; B: 100 nA * 1 kOhm = 100 uV  ->  R_B = 1 kOhm
    assert r["SR860_A_value"] == pytest.approx(1e-7, rel=0.05)
    assert r["SR860_B_value"] == pytest.approx(1000, rel=0.05)
    assert r["SR860_B_rel_noise"] < 0.05
    assert r["SR860_B_rel_noise_worst"] >= 0.5 * r["SR860_B_rel_noise"]
    assert "SR860_B R" in repr(ck)
    assert dm.load_run(ck.run_id, str(tmp_path)).metadata["type"] == "frequency_check"


def test_fixed_tau_is_used(rig):
    ck = dm.check_frequencies(CONFIG, [21.0], GAINS, current="SR860_A", tau=3e-3, sync=False,
                              duration=1.5, dwell=0.1)
    assert ck.rows[0]["tau"] == pytest.approx(3e-3) and ck.rows[0]["sync"] is False


def test_set_basedir_is_one_folder_for_everything(tmp_path, monkeypatch):
    from dmz_measure.measureme import sweep as mm

    monkeypatch.setattr(mm, "BASEDIR", None)
    dm.set_basedir(str(tmp_path / "data"))
    assert dm.get_basedir() == str(tmp_path / "data") == mm.BASEDIR   # stations use the same


def test_does_not_import_an_installed_sweep():
    import sys

    import dmz_measure  # noqa: F401
    assert "sweep" not in sys.modules or not sys.modules["sweep"].__file__.startswith(
        str(__import__("pathlib").Path(dmz_measure.__file__).parent))
    assert dm.Station.__mro__[2].__module__ == "dmz_measure.measureme.sweep"


def test_runwriter_ids_and_sweep_compat(tmp_path):
    os.mkdir(tmp_path / "0")  # pretend sweep already wrote run 0
    with RunWriter(str(tmp_path), "noise_survey", comment="hi") as w:
        w.write_table(["a", "b"], np.c_[[1, 2], [3, 4]])
    assert w.id == 1
    with open(tmp_path / "1" / "metadata.json") as f:
        md = json.load(f)
    for key in ("type", "interrupted", "start_time", "end_time", "columns", "comments"):
        assert key in md
    with gzip.open(tmp_path / "1" / "data.tsv.gz", "rt") as f:
        assert np.loadtxt(f).shape == (2, 2)

