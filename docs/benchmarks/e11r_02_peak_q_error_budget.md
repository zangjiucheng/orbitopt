# E11R-02 — where the E11-06 entry peak-q difference lives (task #415)

Epic #409. The E11-06 gate record archived the independent point-mass twin
re-flying the sealed continuous-burn script with an entry peak dynamic pressure
**+18.2 %** above the main model's, against the unchanged **12 %**
`reflight.entry_peak_q` band, so that one check is red by design while the
other eight agree (`speed +4.6 %`, `tilt 2.2°`, `peak_g 0.84 g`):
`twin.json` under
[`tests/data/mission/reference/e11-06-starbase-continuous-burn-reports/`](../../tests/data/mission/reference/e11-06-starbase-continuous-burn-reports/README.md).
The archive named the candidates — atmosphere model, grid-fin drag
implementation, angle-of-attack and attitude, integration step, event
extraction — without pricing any of them. This task prices them and leaves the
tolerance alone. The machine reading is
[`e11r_02_peak_q_error_budget.json`](e11r_02_peak_q_error_budget.json);
reproduce with
`E11_ENGINE_STAGES=1 E11_FINS_FIXED=1 python benchmarks/bench_e11r_02_peak_q_error_budget.py --out …`
(≈1 min, CPU only, and the script refuses to run without the gate's two
environment variables) and check with
`PYTHONPATH=src pytest tests/test_e11r_02_peak_q_error_budget.py`.

## Method

The E10-07 budget protocol, one convention changed at a time, every line a
measured twin re-fly rather than a hand calculation, expressed in **points of
the main model's peak q** (main peak q ≡ 100 pts). The closure is exact by
construction, so nothing hides in a residual:

```
delta = (twin − twin-on-main-profile) + (twin-on-main-profile − main)
      =    density-profile form line  +   everything else
```

Re-running the sealed candidate on this stack reproduces the archive:
**+17.81 %** measured against **+18.19 %** archived (0.4 pt of drift, main peak
q 135.57 kPa here vs 136.67 kPa archived), plan feasible, corridor speed
241.0 m/s, tilt 11.9°. The gate still reads red.

## The budget

| line | pts | what was changed |
|---|---|---|
| density-profile form | **+21.10** | twin re-flown reading the main model's density profile |
| … the 0–25 km segment alone | +20.96 | same swap, applied below 25 km only |
| … above 25 km | +0.17 | same swap, applied above 25 km only |
| everything else | −3.29 | the gap left once the two share one profile |
| … integration step 0.10 → 0.05 s | +0.53 | twin step halved |
| … J2 dropped | +0.27 | twin on point-mass gravity |
| … geopotential → geometric height | −0.34 | twin's own US76 read without the layer-table conversion |
| … entry-window definition | 0.00 | main model's own track re-windowed onto the twin's window |
| measured delta | **+17.81** | (closure error 0, to machine precision) |

One convention swap takes the gap from 17.8 pts to 3.3 pts, and it overshoots:
the rest of the twin pulls the *other* way, so the profile line is not merely
the biggest contributor — it is larger than the discrepancy itself, and no
other single difference can be worth 18 pts. The archived candidates are
priced out by that closure: the step is 0.5 pt, gravity 0.3 pt, the height
convention 0.3 pt, and the event-extraction difference is *exactly* zero (the
main model's peak q is the same number whether it is read over its own
ignition→crossing window or over the twin's cutoff→crossing window).

Grid-fin modelling could not be dismissed the same way, because the fins carry
so much of this trajectory: the twin re-flown with its own fin force switched
off dives through the corridor at 617 m/s and its entry peak is 52 pts
*higher*. That line is therefore not a cross-model line — it is reported to
show the size of the lever. The fin candidate closes for the same reason as the
others: with only the density profile shared, everything left is 3.3 pts.

## Where the profiles split

Both implementations are faithful to their published source — this is a
model-*form* difference, not a bug. The twin integrates the US76 layer
equations with its own geopotential conversion (`orbitopt.verify.
full_recovery_reflight._twin_us76`, layer continuity across the 11/20/32/47/51/
71 km bases is asserted in the E10-07 test). The main model flies the published
piecewise-exponential table (`orbitopt.entry._ATMOSPHERE`), whose base heights
below 100 km are 0, 25, 30, … km: **no base between the ground and 25 km**, so
the entire troposphere and lower stratosphere is one exponential with scale
height 7.249 km, while US76 bends at 11 and 20 km.

| h (km) | main ρ (kg/m³) | twin US76 ρ (kg/m³) | main/twin |
|---|---|---|---|
| 0 | 1.225 | 1.225 | 1.000 |
| 5 | 0.6146 | 0.7364 | 0.835 |
| 8 | 0.4063 | 0.5258 | 0.773 |
| 11 | 0.2686 | 0.3648 | **0.736** |
| 15 | 0.1547 | 0.1948 | 0.794 |
| 20 | 0.07761 | 0.08891 | 0.873 |
| 25 | 0.03899 | 0.04008 | 0.973 |
| 40 | 0.003972 | 0.003995 | 0.994 |
| 100 | 5.297e−7 | 5.788e−7 | 0.915 |

The two profiles agree exactly at sea level, disagree by up to 26 % inside the
0–25 km segment (worst near the 11 km tropopause) and by at most 8.5 % above
it — where density no longer matters to peak q. The peak sits inside that
segment in both models: **main 9.96 km, twin 11.34 km** (the height and time of
the peak are now recorded by the twin, `entry_peak_q_height_m` /
`entry_peak_q_t_s`). So the two models are not disagreeing *at* the peak by
18 %: the twin integrates denser air over the whole descent, the air builds
faster along its path, and its peak arrives both higher up and 21 points of q
above the main model's.

## Verdict — band 12 % unchanged, the gate stays red

The discrepancy is a **genuine model-form difference in the density profile,
localised to the single 0–25 km segment of the published piecewise-exponential
table the main model flies**, with 99 % of the profile line carried by that
segment and an exact closure. There is no implementation difference inside this
band to fix, so acceptance branch 2 applies: the difference is retained,
recorded, and now explained, and the 12 % band is untouched — widening it would
hide a 26 % profile disagreement that a table revision at the next vehicle
revision can remove. `tests/test_e11r_02_peak_q_error_budget.py` locks the
band, the closure, the dominance of the profile line and the live profile
comparison, so a future change to either atmosphere model has to move the
attribution as well.

One real implementation difference was found while auditing the twin, and it
does **not** touch this band: the twin's thrust model reads `cmd.throttle` but
ignores `cmd.min_throttle_scale`, the 13 → 5 → 3 flag the gated main model
flies. That flag is set only by the landing law (`landing.py:1867`), i.e. after
the corridor crossing, so it cannot move the entry peak q — but it does bias the
*impact* lines of the same gate, and it is recorded in the report for the
follow-up that grades those.
