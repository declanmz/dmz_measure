# dmz_measure

Lock-in streaming and noise tools for low-frequency (< 100 Hz) transport measurements
on 2D devices. It is meant to sit next to measureme (`sweep`), not replace it: runs go
into the same kind of numbered data folders, and it uses the qcodes SR860 objects you
already have open.

**v0.1 status.** SR860 ethernet streaming, plus the FFT noise survey for choosing the
excitation frequency. It is tested offline against a simulated SR860 that streams real
UDP packets. It has **not been run on the CryoFree yet**: start with
`notebooks/CF_noise_survey.ipynb`.

## Install (measurement PC)

```
pip install -e "path\to\dmz_measure"
```

Dependencies: numpy, scipy and matplotlib. Use qcodes or pyvisa to open the instruments.
Alternatively, leave the folder next to your notebooks: the notebook falls back to
`sys.path` if the package isn't installed.

Two things streaming needs:

* **VXI-11 addresses** (`TCPIP::192.168.50.x::inst0::INSTR`). The SR860 sends UDP to
  the IP address that sent `STREAM ON` over VXI-11.
* **The firewall open for UDP 1865–1880.** Run this once in an administrator PowerShell:
  `New-NetFirewallRule -DisplayName "SR860 streaming" -Direction Inbound -Protocol UDP -LocalPort 1865-1880 -Action Allow`

## Choosing the excitation frequency

```python
import dmz_measure as dm

channels = {'SR860_A': dm.Channel(SR860_A, gain=1e6, unit='A'),   # current
            'SR860_B': dm.Channel(SR860_B, gain=100),              # voltages
            'SR860_C': dm.Channel(SR860_C, gain=100)}

# 1. survey: f_ref = 100 Hz, tau = 100 us, sync off, excitation off -> spectra 0-200 Hz
survey = dm.noise_survey(channels, master=SR860_A, duration=300, basedir='./data_dmz')
dm.plotting.plot_survey(survey)

# 2. rank: predicted noise of one data point at every candidate frequency
rank = survey.rank(dm.MeasurementFilter(tau=0.3), fmin=5, fmax=95)
print(dm.format_table(rank.best))

# 3. check the best few at real settings, excitation on (tau ~ 1/(30 f), sync on)
check = dm.candidate_check(channels, master=SR860_A, current='SR860_A',
                           frequencies=[b['f'] for b in rank.best[:4]] + [13.2943])
```

How it works. With a reference at f_ref, the complex output z = X + iY is the input
signal shifted down by f_ref. An FFT of z therefore shows the input at f = f_ref + ν. With
the lock-in's rms normalisation, the two-sided PSD of z is the input noise density
directly, in V/√Hz. A lock-in reading at frequency f with filter |H|² then has variance

σ² = ½ ∫ S(f′) (|H(f′ − f)|² + |H(f′ + f)|²) df′,

and that is how the candidates are ranked. The X/Y sign convention is checked
automatically against the mains harmonics. Every lock-in setting the procedures touch is
snapshotted and restored at the end, also on errors.

The method is checked with a software lock-in on simulated input (`tests/test_noise.py`).
The recovered noise floor and line amplitudes are within a few percent of what was put
in, and the predicted noise matches the measured noise within ±5% (τ filter, sync, dwell
average).

## Modules

| module | what |
|---|---|
| `instruments.sr860_stream` | `SR860Stream`, `acquire_many`: UDP streaming. Header parsing follows the SR860 manual (App. D). Dropped packets become NaN. |
| `instruments.sr860` | `LockinIO` (wraps qcodes, pyvisa or an address), `preserved_settings`, `apply_settings` |
| `noise` | `input_spectrum`, `find_lines`, `predicted_noise`, `rank_frequencies`, `pixel_noise`, `allan_deviation`, `spectrogram` |
| `lockin` | SR860 time-constant table, ENBW, settling times (7.1τ / 5.4τ + 1/f), `recommended_tau` = 1/(30 f), `MeasurementFilter` |
| `procedures` | `noise_survey`, `candidate_check` |
| `io` | `RunWriter` / `load_run`. Same ID scheme as `sweep`. Writes `metadata.json`, `data.tsv.gz`, `*.npz` and `plot.png`. |
| `simulate` | `FakeSR860` (streams real UDP to 127.0.0.1), `fake_cryofree()`, `SoftwareLockin` |
| `plotting` | survey / ranking / spectrogram / Allan figures |

Runs saved here appear in `sweep.list_measurements()` when both share a data folder.
Their `type` is `noise_survey` or `frequency_check`.

## Tests

```
pytest            # ~30 s; includes real UDP loopback against FakeSR860
```

Set `SIMULATE = True` in the notebook to run all of it without instruments.
`notebooks/CF_noise_survey_SIMULATED.ipynb` is an executed copy of such a run.

## Roadmap (from the SR860 streaming discussion)

- [x] Stream X/Y over UDP; FFT noise survey; frequency ranking; candidate check
- [ ] Triggered capture per line: `CAPTURESTART` with hardware trigger, a TTL from the
      gate ramp, `CAPTUREGET?` after each line, serpentine scans
- [ ] Continuous gate ramps: stream, bin by gate voltage, shift out the filter lag,
      check forward vs reverse
- [ ] Stepped maps on the short-τ recipe: τ ≈ 1/(30 f) + sync, skip ~1 period plus 7τ
      after each step, average the rest
- [ ] Phase diagnostics: flag θ > 5° / 15° / 20°; complex V/I; two-frequency
      extrapolation R_dc ≈ [4X(f) − X(2f)]/3
