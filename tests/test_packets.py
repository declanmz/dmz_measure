import numpy as np
import pytest

from dmz_measure.instruments.sr860_stream import (PacketHeader, assemble, decode_packet,
                                                  encode_packet)


def _packet(counter, vals, content=1, length_code=0, rate_code=5, status=0x20):
    hdr = PacketHeader(counter=counter, content=content, length_code=length_code,
                       rate_code=rate_code, status=status)
    return encode_packet(hdr, np.asarray(vals).ravel())


def test_header_fields_roundtrip():
    h = PacketHeader(counter=200, content=3, length_code=2, rate_code=17, status=0x33)
    h2 = PacketHeader.from_word(h.to_word())
    assert h2 == h
    assert h2.overload and h2.error and h2.little_endian and h2.checksum
    assert h2.channels == ("X", "Y", "R", "THETA")
    assert h2.data_bytes == 256
    assert h2.samples_per_packet == 256 // 16
    assert h2.sample_rate == pytest.approx(1.25e6 / 2 ** 17)


def test_header_bit_positions_match_manual():
    # counter 7-0, content 11-8, length 15-12, rate 23-16, status 31-24
    word = (0x5A) | (1 << 8) | (3 << 12) | (4 << 16) | (1 << 24)
    h = PacketHeader.from_word(word)
    assert (h.counter, h.content, h.length_code, h.rate_code) == (0x5A, 1, 3, 4)
    assert h.overload and not h.error and not h.little_endian


@pytest.mark.parametrize("little", [False, True])
def test_float_xy_decode(little):
    vals = np.column_stack([np.arange(128) * 1e-6, -np.arange(128) * 1e-6])
    status = 0x20 | (0x10 if little else 0)
    hdr, out = decode_packet(_packet(7, vals, status=status))
    assert hdr.little_endian == little
    assert out.shape == (128, 2)
    np.testing.assert_allclose(out, vals.astype(np.float32))


def test_int16_decode_raw():
    vals = np.column_stack([np.arange(-50, 50), np.arange(100)]).astype(np.int16)
    hdr = PacketHeader(counter=0, content=5, length_code=3, rate_code=0, status=0)
    out_hdr, out = decode_packet(encode_packet(hdr, np.resize(vals.ravel(), 64)))
    assert out_hdr.is_int16 and out_hdr.channels == ("X", "Y")
    assert out.shape == (32, 2)


def test_assemble_inserts_nan_for_dropped_packets_and_wraps_counter():
    vals = np.ones((128, 2))
    counters = [254, 255, 0, 3, 4]  # 1 and 2 dropped, counter wraps 255 -> 0
    pk = [_packet(c, vals * c) for c in counters]
    d = assemble(pk, name="t")
    assert d.n_dropped == 2
    assert len(d) == (len(counters) + 2) * 128
    assert np.isnan(d.data[3 * 128:5 * 128]).all()
    assert d.fs == pytest.approx(1.25e6 / 32)
    np.testing.assert_allclose(d.z[:128], 254 + 254j)
