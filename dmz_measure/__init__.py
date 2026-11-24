"""dmz_measure: a fork of measureme (``sweep``) with SR860 streaming on top.

Measurements, exactly as with measureme, plus streamed sweeps:
    s = dm.AsyncStation(measurement_config)          # = sweep.AsyncStation + streaming
    s.fp(SR860_B.X, gain=100); s.fp(SR860_B.Y, gain=100); s.fp(Kbg2.curr)
    s.sweep(Kbg2.volt, Vbg, delay=0.6)               # measureme, unchanged
    s.stream_sweep(Kbg2.volt, Vbg, dwell=0.3)        # streamed, same columns and files
    dm.refilter(run_id, dwell=0.1); dm.compare([id1, id2], current='SR860_A')

Choosing the excitation frequency:
    survey = s.noise_survey(current='SR860_A', f_ref=100, duration=600)
    best = dm.best_frequencies(survey, fmin=10, fmax=40, tau=0.3)
    dm.expected_noise(survey, [13.2943], tau=0.3)
    check = s.check_frequencies(best.frequencies[:3], current='SR860_A')

The excitation amplitude is never changed by this package.
"""
from . import lockin, noise, plotting
from .check import Check, check_frequencies
from .instruments.sr860 import LockinIO, apply_settings, preserved_settings
from .instruments.sr860_stream import SR860Stream, StreamData, StreamTimeout, acquire_many
from .io import load_run, set_basedir
from .survey import Ranking, Survey, best_frequencies, expected_noise, load_survey, noise_survey
from .sweeps import compare, point_noise, refilter
from .timing import fast_ramp, gate_steps
from .timing import step_response_from_run as step_response

__version__ = "0.5.0"


def __getattr__(name):
    # The stations import measureme's sweep package (which selects the Qt plot backend);
    # load it only when a station is actually used.
    if name in ("Station", "AsyncStation"):
        from . import station

        return getattr(station, name)
    raise AttributeError(f"module 'dmz_measure' has no attribute {name!r}")
