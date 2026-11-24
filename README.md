# dmz_measure

A fork of [measureme](https://github.com/sharpelab/measureme) (upstream: `sharpelab/measureme`)
with SR860 streaming on top:
* **Streamed sweeps.** `s.stream_sweep` and `s.stream_megasweep` work like `s.sweep` and
  `s.megasweep`. Each point is averaged from the lock-in data stream, with nothing leaking
  between points, and can be re-averaged later.
* **Excitation frequency.** A noise survey, frequency ranking and frequency check for
  choosing it.

measureme's `sweep` package is included **unchanged**, so existing notebooks
(`import sweep`, `sweep.Station`, `s.sweep`, `sweep_load.pload`) keep working, and upstream
updates merge cleanly. measureme's own README is in `README_measureme.md`. GPL-3.0, like
measureme.

**Status (v0.5).** Tested offline against fake SR860s and a fake gated device that stream
real UDP packets, through measureme's own `Station`. It has **not been run on the CryoFree
yet.**

## Install

This installs both `sweep` (measureme) and `dmz_measure`:

```
pip uninstall sweep                      # a separately installed measureme, if any
git clone https://github.com/declanmz/dmz_measure.git
pip install -e dmz_measure
```

As with measureme, live plots need PyQt5. To pull measureme updates into the fork:
`git fetch upstream && git merge upstream/main`.

## Measuring: exactly measureme, plus streaming

```python
import sweep, dmz_measure as dm
sweep.set_basedir('./data_03+12')

s = dm.AsyncStation(measurement_config)          # = sweep.AsyncStation (dm.Station = sweep.Station)
s.add_comment('back-gate line')
for SR860, gain in SR860s:
    s.fp(SR860.X, gain=gain)
    s.fp(SR860.Y, gain=gain)
    s.plot(Kbg2.volt, SR860.X)
s.fp(Kbg2.curr)

s.sweep(Kbg2.volt, Vbg, delay=0.6)                                  # measureme, unchanged
r = s.stream_sweep(Kbg2.volt, Vbg, dwell=0.3, settle=None, window='hann',
                   tau=None, slope=24, advanced=None, sync=True)     # streamed
s.stream_megasweep(Kbg2.volt, Vbg, Ktg2.volt, Vtg, slow_delay=3, dwell=0.3)

dm.refilter(r.id, dwell=0.1, window='boxcar')    # re-average the saved stream (pload format)
dm.compare([id_up, id_down, ...], current='SR860_A')   # overlay + noise + fwd/rev table
```

What `stream_sweep` does:
* **Same output as `s.sweep`.** Same columns, live plots, `data.tsv.gz`/`metadata.json`
  (type `1D`, function `stream_sweep`), and run IDs.
* **Lock-in values from the stream.** The followed SR860 X/Y/R/P parameters come from the
  stream. Every other followed parameter (gate leakage, SR830s) is read normally at each
  point.
* **Lock-in settings during the sweep.** τ = largest SR860 value ≤ 1/(30 f) unless `tau`
  is given, sync filter on, streaming at 10/τ. Restored afterwards.
* **How a point is made.** After each gate set, skip `settle` seconds (default: lock-in
  settling + one reference period), then average `dwell` seconds. `'hann'` weighting
  rejects interference lines away from f far better than a plain mean (`'boxcar'`), for
  ~1.2× the white noise.
* **Re-averaging later.** The stream is saved as `stream.npz` in the run folder.
* **Timing.** Samples are tied to gate steps via the host clock (the stream has no
  timestamps), good to ~1 ms.
* **The excitation amplitude is never changed.**

## Where the time of a sweep goes

```python
s.timing(Kbg2.volt, Vbg, n=20, delay=0.6, dwell=0.3)    # per-point budget + real totals
with dm.fast_ramp(Kbg2.volt):                            # same 5 mV chunks, one *OPC? per set
    st = s.step_response(Kbg2.volt, 0.0, 0.05); st.plot()   # how long a step takes to settle
    r = s.stream_sweep(Kbg2.volt, Vbg, dwell=0.3, settle=st.settle)
dm.step_response(r.id)                                    # settle measured from the sweep itself
```

* **Why a point takes longer than `delay`.** qcodes cuts every set of a parameter with
  `step` (5 mV on the gates) into chunks, at least `inter_delay` (5 ms) apart. With
  `block_during_ramp(True)`, every chunk also waits for a `*OPC?` reply from the 2450.
  Following the leakage currents adds a read per point.
* **What `s.timing` reports.** It measures each piece, including the `*OPC?` round trip,
  and prints the real time per point and per sweep next to measureme's delay-only
  estimate.
* **`dm.fast_ramp`.** Keeps exactly the same 5 mV chunks, at least 5 ms apart, but sends
  them without waiting for a reply each time. It asks `*OPC?` once when the whole set has
  been sent, so the call returns only when the gate is at the new value. The ramp is never
  faster than before; it just stops paying a round trip per chunk. Use it as
  `with dm.fast_ramp(...):` or call it once (`.undo()` reverts). On a fake 2450 (2 ms
  write, 18 ms `*OPC?`), a 0.2 V ramp went from 1.0 s to 0.3 s.
* **`dm.gate_steps`.** The alternative: one bigger command per sweep point. `inter_delay`
  is scaled so long ramps keep the same rate.
* **Instrument-timed ramps.** Not used. The 2450's built-in sweep could time the 5 ms
  steps itself, but its manual doesn't say where the output sits after the sweep ends or
  is aborted, so it would need a bench test first. With `fast_ramp` it would only save the
  ~1–2 ms write per chunk anyway.
* **`s.step_response`.** Steps the gate back and forth while streaming and measures when
  the signal is within 1% of its final value. That covers the gate line, the device, the
  lock-in filter, and any delay between the set call returning and the voltage changing.
  Each step is made as one command, like a sweep point; a ramped step would mostly be
  over before the set call returned. Use the result as `settle`.
* **`dm.step_response(run_id)`.** Measures the same thing from a streamed sweep's own
  steps. If it shows the sweep needed more settle, `dm.refilter(run_id, settle=...)`
  re-averages the saved stream.
* **Reads in streamed sweeps.** Followed parameters that aren't streamed (leakage
  currents, SR830s) are read during the dwell: in parallel per instrument on an
  `AsyncStation`. They add no time unless they take longer than the dwell.

## Choosing the excitation frequency

```python
# excitation off by hand
survey = s.noise_survey(current='SR860_A', f_ref=100, duration=600, fmin=0, fmax=200,
                        resolution=0.05, tau=100e-6, rate=10e3)
survey.plot(); survey.spectrogram('SR860_B', block=30)

best = dm.best_frequencies(survey, fmin=10, fmax=40, tau=0.3, slope=24, advanced=False,
                           sync=False, dwell=None, window='boxcar', block=60, stat='p90')
dm.expected_noise(survey, [13.2943, 17.777], tau=0.3)                                    # normal
dm.expected_noise(survey, [13.2943, 17.777], tau=1e-3, sync=True, dwell=0.3, window='hann')  # streamed

# excitation on by hand, gates at a representative point
check = s.check_frequencies(best.frequencies[:3] + [SR860_A.frequency()], current='SR860_A',
                            tau=None, sync=True, duration=120, dwell=0.3, window='hann')
```

* **Survey.** With the reference at f_ref, the complex FFT of X + iY is the input
  spectrum shifted by f_ref. Every SR860 in `measurement_config` is surveyed, and the
  gains come from `s.fp`. Mains (50/60 Hz) is detected from the data.
  `dm.load_survey(id)` reloads a survey.
* **Choose.** The predicted noise of one reading at reference f is
  σ² = ½ ∫ S(f′) (|H(f′ − f)|² + |H(f′ + f)|²) df′. H is the filter that makes one data
  point:
  * the SR860 time constant: RC stages, or Gaussian stages with `advanced=True`;
  * the sync filter;
  * a software `dwell` average, as a plain mean or Hann-weighted.

  Every combination is checked against a direct software lock-in to within ±12%,
  including next to strong lines.
* **Time variation.** σ is computed for each `block`-second piece of the survey.
  `stat='p90'` ranks by the noise exceeded 10% of the time, and `spread` (worst block /
  typical block) flags intermittent noise.
* **Check.** For each frequency it reports R = V/I, the noise of one data point, the
  worst `block` stretch, and the phase.

## Notebooks

* `notebooks/CF_noise_survey.ipynb` is for the CryoFree. It loads the same instruments as
  the MM03 measurement notebook, then runs the survey → choose → check steps and the
  back-gate comparison (normal vs streamed, up and down).
* `notebooks/CF_noise_survey_SIMULATED.ipynb` runs the same steps against fake instruments
  on any PC.

## Firewall

The SR860 sends its data as UDP packets to the PC, so Windows Firewall has to let them
in. Allow only the instrument subnet, in an administrator PowerShell, once:

```
New-NetFirewallRule -DisplayName "SR860 streaming" -Direction Inbound -Protocol UDP -LocalPort 1865-1880 -RemoteAddress 192.168.50.0/24 -Action Allow
```

The risk is small:
* The rule only accepts packets from the 192.168.50.x instrument network.
* Nothing listens on those ports except while a stream is running.
* dmz_measure also drops any packet that doesn't come from the lock-in's own IP.
* The packets are read as plain numbers, never executed.

To remove the rule: `Remove-NetFirewallRule -DisplayName "SR860 streaming"`.

## Tests

`pytest` (~80 s; needs PyQt5 because `import sweep` selects the Qt backend). It covers:
* the packet format;
* the spectrum calibration and the noise prediction for every filter type, checked
  against a software lock-in;
* catching bursty noise;
* stations against fake instruments: measureme's own `sweep` unchanged, `stream_sweep`
  with the same columns and readable by `pload1d`, `stream_megasweep`, `refilter`, and
  the survey and check from a station;
* the timing tools against a fake gate that ramps like qcodes, with a set latency and a
  slow gate line.

## Roadmap

- [x] UDP streaming, noise survey, frequency choice with time variation, frequency check
- [x] Streamed `stream_sweep` / `stream_megasweep` on measureme's Station, with re-averaging
- [ ] Triggered capture per gate line (`CAPTURESTART` with a TTL from the ramp)
- [ ] Continuous gate ramps: stream, bin by gate voltage, shift out the filter lag
- [ ] Phase checks and the two-frequency extrapolation R_dc ≈ [4X(f) − X(2f)]/3
