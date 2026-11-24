# dmz_measure

**dmz_measure is built on [measureme](https://github.com/sharpelab/measureme)**, the
measurement framework by Aaron Sharpe and other contributors (originally
[spxtr/measureme](https://github.com/spxtr/measureme)). Its stations, sweeps, data files
and loaders are measureme's own code, included here as `dmz_measure.measureme`. It was
copied from sharpelab/measureme at commit `2fab7a4` under the GPL-3.0, with only three
changes: its imports point inside this package, the Qt backend is selected only inside the
live-plot window, and the interrupt handler is put back even when a measurement raises.
measureme's README is kept as [README_measureme.md](README_measureme.md).

On top of measureme, dmz_measure adds SR860 data streaming:
* **Streamed sweeps.** `s.stream_sweep` and `s.stream_megasweep` work like `s.sweep` and
  `s.megasweep`, but each point is averaged from the lock-in's data stream. Points don't
  blur into each other, and the saved stream can be re-averaged later.
* **Up/down gate maps.** A megasweep whose fast axis goes up and straight back down on every
  line (`np.r_[V, V[::-1]]`) measures both sweep directions in one run, with no ramp back
  between lines. `dm.load_updown` splits it into the up and the down map on the same grid, and
  `dm.bin_map` averages blocks of points. A streamed map writes each line's stream when the line
  ends, so it can be loaded (and re-averaged) while it is still running, e.g. from another
  computer through a synced folder.
* **Choosing an excitation frequency.** `s.noise_survey` records the noise of the lock-in
  inputs (excitation off), `dm.best_frequencies` ranks frequencies by the expected noise of
  the resistance, including noise that comes and goes, and `s.check_frequencies` measures the
  frequencies you pick (excitation on).

Measurements are station methods in measureme's style: one run ID, live plots, and the
interrupt button stops at the next point and returns. They only save data. The analysis
functions take a run ID and never touch an instrument.

## Install

```
git clone https://github.com/declanmz/dmz_measure.git
pip install -e dmz_measure
```

It can be installed next to measureme (`import sweep`); the two don't share any code.
Live plots need PyQt5, as in measureme.

## Use

A measurement cell, as in measureme, with the gates ramped back down whatever happens:

```python
import dmz_measure as dm
dm.set_basedir('./data')

s = dm.AsyncStation(measurement_config)       # measureme's AsyncStation, plus streaming
s.add_comment('ID 12: back-gate sweep')
s.fp(SR860_A.X, gain=1e6); s.fp(SR860_A.Y, gain=1e6)
s.fp(SR860_B.X, gain=100); s.fp(SR860_B.Y, gain=100)
s.fp(Kbg2.curr)
s.plot(Kbg2.volt, SR860_B.X); s.plot(Kbg2.volt, Kbg2.curr)
try:
    Kbg2.volt(-1)                                  # qcodes ramp (step / inter_delay)
    s.sweep(Kbg2.volt, Vbg, delay=0.6)             # measureme's sweep, or:
    # s.stream_sweep(Kbg2.volt, Vbg, dwell=0.6)    # streamed: same columns and files
finally:
    Kbg2.volt(0)                                   # also after the interrupt button or an error
```

Choosing an excitation frequency (switch the excitation off and on by hand):

```python
s.noise_survey(duration=600, f_ref=25)                   # excitation off; spectrum 0 - 50 Hz
s.check_frequencies([13.2943, 17.5623], duration=60)     # excitation on
```

Analysis, from the run IDs:

```python
d = dm.pload1d(dm.get_basedir(), 12)                     # measureme's loader
d = dm.refilter(13, dwell=0.1)                           # re-average a streamed run
survey = dm.load_survey(14, current='SR860_A')
best = dm.best_frequencies(survey, {'SR860_B': 106}, excitation=2e-9, tau=0.3)
check = dm.load_check(15, current='SR860_A', dwell=0.6)
up, down = dm.load_updown(16)                            # map swept up then down on every line
b = dm.bin_map(up, fast=2, slow=2)                       # mean of 2 x 2 blocks
```

Every function's docstring lists its options.

## Firewall

The SR860 streams UDP packets to the PC, so Windows Firewall has to let them in. In an
administrator PowerShell, allow them from your instrument subnet only, replacing
`<instrument subnet>` with yours (for example `192.168.1.0/24`):

```
New-NetFirewallRule -DisplayName "SR860 streaming" -Direction Inbound -Protocol UDP -LocalPort 1865-1880 -RemoteAddress <instrument subnet> -Action Allow
```

If streaming times out, the error message prints this command with the lock-in's own
subnet filled in.

## License

GPL-3.0, like measureme. See [LICENSE](LICENSE).
