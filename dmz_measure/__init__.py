"""dmz_measure: measureme's measurement framework plus SR860 streaming, in one package.

measureme (Station, sweeps, data files, loading) is included as dmz_measure.measureme
(copied from measureme, GPL-3.0) - nothing imports an installed ``sweep``, so both can
live on the same computer.

Measuring (stations: one run ID per call, live plots, the interrupt button stops at the next
point and the call returns, so the cell can ramp the gates down):

    import dmz_measure as dm
    dm.set_basedir('./data_03+12')
    s = dm.AsyncStation(measurement_config)
    s.add_comment('ID 44: ...')
    s.fp(SR860_B.X, gain=100); s.fp(SR860_B.Y, gain=100); s.fp(Kbg2.curr)
    s.sweep(Kbg2.volt, Vbg, delay=0.6)               # measureme's sweep
    s.stream_sweep(Kbg2.volt, Vbg, dwell=0.3)        # streamed, same columns and files
    s.noise_survey(duration=600, f_ref=25)           # excitation off
    s.check_frequencies([13.2943, 17.5623])          # excitation on

Analysis (from the run ID, no instruments):

    d = dm.pload1d(dm.get_basedir(), run_id)         # measureme's loader
    d = dm.refilter(run_id, dwell=0.1); dm.compare([id1, id2], current='SR860_A')
    survey = dm.load_survey(run_id, current='SR860_A')
    best = dm.best_frequencies(survey, resistances={'SR860_B': 106}, excitation=2e-9, tau=0.3)
    check = dm.load_check(run_id, current='SR860_A')
    up, down = dm.load_updown(run_id)               # map swept up then down on every line
    b = dm.bin_map(up, fast=2, slow=2)

The excitation amplitude is never changed by this package.
"""
from . import lockin, noise, plotting
from .check import Check, load_check
from .instruments.sr860 import LockinIO, apply_settings, preserved_settings
from .instruments.sr860_stream import SR860Stream, StreamData, StreamTimeout, acquire_many
from .io import get_basedir, load_run, set_basedir
from .maps import bin_map, load_updown
from .measureme import db, raster
from .measureme.sweep import list_measurements, measurement_info
from .measureme.sweep_load import load, load_meta, pload, pload0d, pload1d, pload2d
from .settling import step_response_from_run as step_response
from .station import AsyncStation, Station
from .survey import Ranking, Survey, best_frequencies, expected_noise, load_survey
from .sweeps import compare, point_noise, refilter

__version__ = "0.7.1"
