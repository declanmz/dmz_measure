"""dmz_measure: lock-in streaming and noise tools for low-frequency transport.

First module set: SR860 UDP streaming and the FFT noise survey for choosing the
excitation frequency (see notebooks/CF_noise_survey.ipynb).
"""
from . import lockin, noise, plotting, simulate
from .instruments.sr860 import (LockinIO, apply_settings, preserved_settings,
                                restore_settings, snapshot_settings)
from .instruments.sr860_stream import (SR860Stream, StreamData, StreamTimeout, acquire_many,
                                       decode_packet)
from .io import RunWriter, load_run, set_basedir
from .lockin import MeasurementFilter, enbw_rc, recommended_tau, settling_time
from .noise import (FrequencyRanking, NoiseSpectrum, allan_deviation, find_lines,
                    input_spectrum, pixel_noise, predicted_noise, rank_frequencies, spectrogram)
from .procedures import Channel, CheckResult, SurveyResult, candidate_check, noise_survey
from .utils import format_table

__version__ = "0.1.0"
