"""End-to-end over real UDP sockets on 127.0.0.1 using FakeSR860."""
import numpy as np
import pytest

import dmz_measure as dm
from dmz_measure.instruments.sr860_stream import SR860Stream, StreamTimeout, acquire_many
from dmz_measure.simulate import FakeSR860, fake_cryofree

PORT = 18650


def test_single_stream_loopback():
    li = FakeSR860("L", freq=100.0, tau=100e-6, seed=1)
    st = SR860Stream(li, port=PORT)
    rate = st.configure(rate=40e3)
    assert rate == pytest.approx(li.max_stream_rate() / 4)
    d = st.acquire(1.5, progress=False)
    assert d.fs == pytest.approx(rate)
    assert d.channels == ("X", "Y")
    assert d.n_dropped == 0
    assert d.f_ref == pytest.approx(100.0)
    assert 1.2 * rate < len(d) < 1.8 * rate
    assert np.isfinite(d.z).all()


def test_many_streams_and_external_reference():
    fakes = fake_cryofree(seed=5)
    for f in fakes.values():
        dm.apply_settings(dm.LockinIO(f), tau=1e-3)
    streams = [SR860Stream(f, port=PORT + 1 + i) for i, f in enumerate(fakes.values())]
    for s in streams:
        s.configure(rate=5e3)
    out = acquire_many(streams, 1.0, progress=False)
    assert set(out) == set(fakes)
    for d in out.values():
        assert d.f_ref == pytest.approx(fakes["SR860_A"].freq)
        assert d.n_dropped == 0


def test_duplicate_ports_rejected():
    a, b = FakeSR860("a"), FakeSR860("b")
    with pytest.raises(ValueError):
        acquire_many([SR860Stream(a, port=PORT), SR860Stream(b, port=PORT)], 0.1)


class _Silent:
    """An instrument that accepts STREAM ON but never sends anything."""
    name = "silent"

    def write(self, cmd):
        pass

    def ask(self, cmd):
        return "1000" if cmd.startswith("STREAMRATEMAX") else "0"


def test_no_packets_raises_with_firewall_help():
    with pytest.raises(StreamTimeout, match="Firewall"):
        SR860Stream(_Silent(), port=PORT + 9).acquire(0.5, timeout=0.5, progress=False)


def test_settings_snapshot_restore():
    f = FakeSR860("x", freq=13.2943, amplitude=0.5, tau=0.3)
    io = dm.LockinIO(f)
    with dm.preserved_settings([io], verbose=False):
        dm.apply_settings(io, tau=1e-4, slope_db=12, sync=True, freq=100.0, amplitude=0)
        assert f.slvl == pytest.approx(1e-9)
    assert f.freq == pytest.approx(13.2943)
    assert f.slvl == pytest.approx(0.5)
    assert dm.lockin.tau_from_index(f.oflt) == pytest.approx(0.3)
    assert f.sync == 0 and f.ofsl == 3
