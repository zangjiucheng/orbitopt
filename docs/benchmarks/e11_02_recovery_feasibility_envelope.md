# E11-02 — recovery feasibility envelope (task #309)

E11-02 screens the separation state against a recovery feasibility envelope (orbitopt.flightlab.recovery.feasibility) BEFORE the planner searches: one reference rollout plus an analytic fuel floor decides IN/OUT and predicts the flight reason with zero search fevals. Envelope conditions: fuel floor (rocket equation on the separation velocity), corridor reach/miss/tilt/arrival-speed, closed-loop landing forecast, entry q and g, landing propellant reserve margin — every one measured on the planner's own physics. Era note: the DC8 hold-out arms are the E10-03 era (the DC7 sealed POWERED recipe flown on the DC8 vehicle, no grid fins); the envelope that applies to DC8 as authored (E11-01 grid-fin mission) is graded in parts A/B/E.

## 1. Reference verdicts (part A)

| rig | spec | in envelope | governing | predicted reason | unscreened reason | match |
|---|---|---|---|---|---|---|
| `dc7` | `recipe` | IN | None | `None` | `-` | yes |
| `starbase` | `recipe` | IN | None | `None` | `-` | yes |
| `dc8_fins` | `dc8_fins_a60` | OUT | corridor_tilt | `CORRIDOR_MISSED` | `CORRIDOR_MISSED` | yes |
| `dc8` | `recipe` | OUT | landing_forecast | `CORRIDOR_MISSED` | `CORRIDOR_MISSED` | yes |

## 2. Boundary maps (part B)

- `dc7`: 18/49 inside; severity field and governing map in `docs/benchmarks/figures/e11_02_boundary_dc7.png`; grid = downrange ±9 km x speed ±60 m/s at the hot-staging handoff.
- `dc8_fins`: 0/49 inside; severity field and governing map in `docs/benchmarks/figures/e11_02_boundary_dc8_fins.png`; grid = downrange ±9 km x speed ±60 m/s at the hot-staging handoff.

## 3. E10-03 hold-out agreement (part C)

Each sample is flown unscreened (the truth, the E10-03 worker verbatim) and screened (the prediction). A *false reject* — screen OUT but the flight lands — is the only error the screen is not allowed to make.

| arm | n | screen-in & landed | in & not landed | out & not landed | out but landed (false rejects) | agreement | rejected-sample code match | truth vs committed |
|---|---|---|---|---|---|---|---|---|
| `dc7_pinned` | 168 | 44 | 0 | 124 | **0** | 1.0 | 124/124 | 44/44 (match) |
| `starbase_pinned` | 168 | 11 | 0 | 157 | **0** | 1.0 | 157/157 | 11/11 (match) |
| `dc8_robust` | 168 | 0 | 0 | 168 | **0** | 1.0 | 168/168 | 0/0 (match) |
| `dc8_binary` | 168 | 0 | 3 | 165 | **0** | 0.9821 | 165/165 | 0/0 (match) |

Total false rejects (screen OUT, flight landed): **0** — the screen's only unforgivable error is zero.

## 4. Twin boundary reproduction (part D)

E10-07's independent twin re-flies the sealed script from the same separation state on its own atmosphere line, wind, and fin model; bands are calibrated on the baseline runs and then applied to the boundary pair C selects.

| run | boundary side | agreement | q (kPa) | load (g) | speed (m/s) | tilt (deg) | miss (m) | propellant (t) |
|---|---|---|---|---|---|---|---|---|
| `D|base|dc7` | — | no | 147 (in, ±20%) ✗ | 7.6 (in, ±2g) ✓ | 277 (in, ±40%) ✓ | 0.0712 (in, ±3deg) ✓ | 122 (in, ±1500m) ✓ | 166 (in, ±3%) ✗ |
| `D|base|dc8_fins` | — | no | 131 (in, ±20%) ✓ | 7.09 (in, ±2g) ✓ | 279 (in, ±40%) ✗ | 26.1 (out, ±3deg) ✓ | 823 (in, ±1500m) ✓ | 301 (in, ±3%) ✓ |
| `D|pair|in|dc7_pinned|502|1|cal` | IN | yes | 160 (in, ±40%) ✓ | 7.47 (in, ±2g) ✓ | 270 (in, ±40%) ✓ | 0.0444 (in, ±3deg) ✓ | 65.8 (in, ±1500m) ✓ | 154 (in, ±6%) ✓ |
| `D|pair|out|dc7_pinned|501|3|cal` | OUT | yes | 166 (in, ±40%) ✓ | 7.32 (in, ±2g) ✓ | 256 (in, ±40%) ✓ | 0.0326 (in, ±3deg) ✓ | 315 (in, ±1500m) ✓ | 162 (in, ±6%) ✓ |

Baselines carry the band calibration (1.5x the measured twin-vs-planner delta, floored at the E11-01 FIN_TOL values); twin disagreement is reported, never hidden. The twin's arrival-speed channel is the known atmosphere-line quantity (E11-01 measured 24%): when that quantity alone disagrees, the boundary verdict rests on q/g/tilt/miss/prop.

## 5. Search budget (part E)

| state | full search: wall / fevals / verdict | screened: wall / fevals / reason |
|---|---|---|
| `E|dc7|in-nominal` | 59.8 s / 132 fevals / `CORRIDOR_MISSED` | 61.0 s / 132 fevals / accepted |
| `E|dc7|out-corner` | 59.9 s / 0 fevals / `CORRIDOR_MISSED` | 1.5 s / 0 fevals / `CORRIDOR_MISSED` |
| `E|dc8_fins|out` | 40.1 s / 0 fevals / `CORRIDOR_MISSED` | 0.8 s / 0 fevals / `CORRIDOR_MISSED` |

The screened calls spend zero search fevals on an OUT state and return the predicted flight reason; the full search spends its budget (counted through the planner's own search hook, same options as the default path). Below the propellant floor the planner's own guard also skips the search, so there the saving is wall-clock -- one cheap reference rollout against the full-dt grading integrations (40x on the corner cell) -- while inside the floor the screen retires the entire full-budget pass. The DC8-era arm's real budget was the 300+-flight ES training (see the committed E10-03 report), which the screen retires for states outside the envelope.
