# dmz_measure

**dmz_measure is built on [measureme](https://github.com/sharpelab/measureme)**, the
measurement framework by spxtr, Aaron Sharpe and other contributors (originally
[spxtr/measureme](https://github.com/spxtr/measureme)). Its stations, sweeps, data files
and loaders are measureme's own code, included here as `dmz_measure.measureme`. It was
copied from sharpelab/measureme at commit `2fab7a4` under the GPL-3.0, with only two
changes: its imports point inside this package, and the Qt backend is selected only
inside the live-plot window. measureme's README is kept as
[README_measureme.md](README_measureme.md).

On top of measureme, dmz_measure adds SR860 data streaming:
* **Streamed sweeps.** `s.stream_sweep` and `s.stream_megasweep` work like `s.sweep` and
  `s.megasweep`, but each point is averaged from the lock-in's data stream. Points don't
  blur into each other, and the saved stream can be re-averaged later.
* **Choosing an excitation frequency.** A noise survey of the lock-in inputs, a ranking
  of quiet frequencies that accounts for noise that comes and goes, and a check of the
  frequencies you pick.
* **Faster gate ramps.** `dm.fast_ramp` keeps the same voltage steps but doesn't wait for
  the Keithley to reply after each one.

dmz_measure never changes the lock-in excitation amplitude.

## Install

```
git clone https://github.com/declanmz/dmz_measure.git
pip install -e dmz_measure
```

It can be installed next to measureme (`import sweep`); the two don't share any code.
Live plots need PyQt5, as in measureme.

## Use

```python
import dmz_measure as dm
dm.set_basedir('./data')

s = dm.AsyncStation(measurement_config)       # measureme's AsyncStation, plus streaming
s.fp(SR860_A.X, gain=1e6); s.fp(SR860_A.Y, gain=1e6)
s.fp(SR860_B.X, gain=100); s.fp(SR860_B.Y, gain=100)
s.fp(Kbg2.curr)

s.sweep(Kbg2.volt, Vbg, delay=0.6)                 # measureme's sweep
r = s.stream_sweep(Kbg2.volt, Vbg, dwell=0.3)      # streamed: same columns and files
d = dm.pload1d(dm.get_basedir(), r.id)             # measureme's loader
d = dm.refilter(r.id, dwell=0.1)                   # re-average the saved stream
```

Choosing an excitation frequency (switch the excitation off and on by hand):

```python
survey = s.noise_survey(current='SR860_A', f_ref=100, duration=600, fmax=200)  # excitation off
best = dm.best_frequencies(survey, fmin=10, fmax=40, tau=0.3)
dm.expected_noise(survey, [13.2943, 17.777], tau=0.3)
check = s.check_frequencies(best.frequencies[:3], current='SR860_A')           # excitation on
```

Gate timing:

```python
s.timing(Kbg2.volt, Vbg, delay=0.6)          # where the time per point goes
with dm.fast_ramp(Kbg2.volt):                # same steps, one reply per set
    s.stream_sweep(Kbg2.volt, Vbg, dwell=0.3)
```

Every function's docstring lists its options.

## Firewall

The SR860 streams UDP packets to the PC, so Windows Firewall has to let them in. In an
administrator PowerShell, allow them from your instrument subnet only:

```
New-NetFirewallRule -DisplayName "SR860 streaming" -Direction Inbound -Protocol UDP -LocalPort 1865-1880 -RemoteAddress 192.168.50.0/24 -Action Allow
```

## License

GPL-3.0, like measureme. See [LICENSE](LICENSE).
