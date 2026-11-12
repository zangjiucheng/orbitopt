# ADR flightlab-0002: Closed-loop terminal landing guidance + a gentler FL-302 handoff

- **Status:** Accepted
- **Date:** 2026-11-12
- **Deciders:** Flight Lab E3 (Epic #195), FL-305 (#214), follow-up to FL-304 (#205)
- **Code:** `src/orbitopt/flightlab/recovery/landing.py` (`TerminalGuidance`,
  `terminal_guidance_command`, `fly_landing_burn_closed_loop`),
  `src/orbitopt/flightlab/recovery/pipeline.py` (`NOMINAL_PLAN_GUESS`,
  `plan_options["landing"]`)
- **Evidence:** `docs/benchmarks/fl305_recovery_montecarlo.json`
  (`benchmarks/bench_fl305_recovery_montecarlo.py`),
  `tests/test_flightlab_simulate_recovery.py`,
  `tests/test_flightlab_recovery_terminal_guidance.py`

## Context and problem

FL-304 wired the full recovery chain (simulate → FL-302 boostback/entry →
FL-303 G-FOLD landing burn), but from the textbook RTLS separation (8 km
east, 40 km up, 250 m/s, 65 t) the nominal recovery came back red.

- FL-302 handed the landing burn a corridor state at h ≈ 3.0 km, 307 m/s
  down, **1.76 km west of the pad**, 41.1 t (11.1 t propellant).
- FL-303 flies one convex plan open-loop. The flat model sees no drag, and
  it uses the insertion-altitude thrust envelope while the engine loses
  ~10 % of its thrust to ambient pressure near the ground. Re-measured here:
  touchdown at **−107.7 m/s, 331 m** off the pad.
- FL-304's green E2E only passed because it substituted FL-303's certified
  nominal landing (200 m, 10 m/s, 54 t) for the landing step.
- The engine's 40 % floor is ~1.5× the vehicle's weight near the ground (no
  hover), so the last seconds leave no room for error.

## Options considered

1. **Re-solve G-FOLD in the loop** (re-plan the SOCP from the propagated
   state at altitude triggers, shrinking time of flight, a re-plan stop
   height). FL-304 measured −4.5 to −30 m/s and tilt past 40°. A
   prototype in this task went further: 13 re-solves, envelope and drag
   bias re-evaluated at the current height, and the throttle floor lowered
   to 0.12 so the vehicle could hover. It still touched down at **−8.3 m/s,
   4.9 m off, 653 kg below the reserve**. The min-fuel arcs are bang-coast-bang:
   they pin the throttle at its limits, so every re-plan starts with no
   authority to absorb the drag and thrust mismatch. Near the ground they
   coast too long, then bounce at full thrust (lit, rose to 11 m, fell in).
   Rejected.
2. **Synthetic vehicle numbers** (lower the landing reserve or the throttle
   floor). These would turn the reserve or hover checks green by fiat and
   change the vehicle FL-301/FL-302/FL-303 certified. Not needed (see
   below), so the vehicle YAML is **unchanged**: 4,200 kg reserve, throttle
   0.4–1.0, one engine.
3. **Closed-loop terminal guidance with margin**: an analytic law,
   re-planned every guidance step from the propagated state, that flies with
   throttle authority held in reserve. **Chosen.**
4. **Gentler FL-302 handoff**: hand over closer to the pad. **Chosen, in
   combination.** FL-302's corridor penalty is `(miss / radius)²` with an
   8 km radius, so a 1.76 km miss costs it ~0.05 and it has no reason to fix
   it. A tighter corridor radius was tried (1 km, 0.6 km). At the
   pipeline's 18–30-evaluation Nelder-Mead budget the optimiser cannot step
   the 0.5 s-quantised boostback duration, so the result was
   `CORRIDOR_MISSED`. The fix that works is the warm start.

## Decision

**(a) Closed-loop terminal guidance in FL-303** (`TerminalGuidance`, the
default landing step of the pipeline). Every integration step
(`dt = 0.1 s`), from the propagated pad-ENU state:

- **Vertical: throttle-limited hoverslam.** The guidance computes
  `a_req = (v_z² − v_td²) / 2h`, the constant deceleration that reaches
  `v_td = 1.5 m/s` at pad level. The engine stays off until `a_req` reaches
  `design_throttle = 0.9` of the *sea-level* thrust minus gravity, then
  follows `a_req + g`. That leaves 10 % authority for drag, ambient thrust
  and mass errors. A late, hard burn is also the lowest-gravity-loss burn.
- **Lateral: ZEM/ZEV divert.** The lateral command is
  `6·ZEM/t² − 2·ZEV/t`, using the vertical profile's time-to-go minus a
  2 s settle window. Inside that window a PD loop nulls the residual drift.
  If a burn started now would need more than 70 % of the tilt-limited
  lateral authority, the engine lights early (*divert-aware ignition*).
- **Attitude in the loop.** The lateral command is clipped to a 30° thrust
  cone, below the 40° touchdown check. Thrust is capped at 4.5 g (the check
  limit is 5 g).
- **Cutoff tuned to the no-hover T/W.** Once `a_req` falls below what the
  40 % floor delivers, the profile can no longer be tracked. Below 3 m the
  engine then shuts down for good, as soon as the ballistic arrival speed
  `sqrt(v_z² + 2gh)` is ≤ 2.2 m/s. One ignition, one cutoff, no restarts.

The insertion SOCP is still solved. It remains the feasibility certificate:
an infeasible state is `LANDING_INFEASIBLE` exactly as before, with no
flight. **The referee is unchanged.** One propagation through FL-301 (drag,
J2, rotation, min-throttle clamp, the caller's dispersed physics) is graded
by FL-303's `landing_touchdown_checks`, which the guidance never sees.
`plan_options={"landing": {"mode": "single"}}` keeps FL-303's open-loop flow
available.

**(b) A gentler FL-302 handoff.** `NOMINAL_PLAN_GUESS` changes its boostback
duration from 19.1 s to 17.75 s. FL-302 keeps it as its accepted script: it
is feasible, uses less boostback propellant, and the 18-evaluation
refinement stays on it. The handoff becomes h = 3.0 km, 306 m/s down,
**0.35 km** off, **41.7 t** (11.7 t propellant).

## Results

Nominal chain (`test_slow_e2e_nominal_recovery_is_green_through_the_real_chain`,
real simulate → FL-302 → FL-303, nothing substituted):

| handoff | landing flow | touchdown v_z | pad dist. | lateral | max tilt | reserve margin | verdict |
|---|---|---|---|---|---|---|---|
| FL-304 (1.76 km, 41.1 t) | single plan | −107.7 m/s | 331 m | 72 m/s | 37.5° | +1839 kg | red |
| FL-304 (1.76 km, 41.1 t) | closed loop | −1.6 m/s | 224 m | 39 m/s | 30.0° | −514 kg | red (divert budget) |
| FL-305 (0.35 km, 41.7 t) | single plan | −85.0 m/s | 27 m | 16 m/s | 29.2° | +2921 kg | red |
| **FL-305 (0.35 km, 41.7 t)** | **closed loop** | **−1.6 m/s** | **0.6 m** | **0.04 m/s** | **19.3°** | **+992 kg** | **green** |

Limits: |v_z| ≤ 3 m/s, lateral ≤ 1 m/s, pad distance ≤ 25 m, tilt ≤ 40°,
≤ 5 g (measured 3.5 g), and propellant ≥ the 4,200 kg reserve. Neither
change alone is enough. The guidance is what gets touchdown speed under
control. The handoff is what brings the divert inside the propellant
budget.

### Monte Carlo before / after (FL-304 harness, default dispersions, seed 1, n = 32)

`docs/benchmarks/fl305_recovery_montecarlo.json`. Both runs are
deterministic per seed. "Before" is FL-304 as merged (old warm start,
single plan); "after" is the FL-305 defaults.

| reason | before (FL-304) | after (FL-305) |
|---|---|---|
| `LANDED` | 0 | **14** |
| `LANDING_POSITION_OUT_OF_TOLERANCE` | 23 | 4 |
| `LANDING_PROPELLANT_RESERVE_LOW` | 0 | 5 |
| `INSUFFICIENT_PROPELLANT_RESERVE` (FL-302) | 8 | 8 |
| `CORRIDOR_MISSED` (FL-302) | 1 | 1 |
| **success rate** | **0.000** | **0.438** |

Landed subset (after): pad distance p50 0.86 m / p95 2.2 m, lateral speed p95
0.19 m/s, vertical speed −1.50 m/s.

Every failure that remains carries an honest reason:

- **9 never reach the landing burn.** FL-302 rejects them (separation
  dispersion ±800 m / ±4 m/s), identically before and after.
- **4 position failures.** These handoffs end up kilometres off, beyond
  what a 30° cone can divert from 3 km.
- **5 reserve failures.** The divert or density dispersion costs more than
  the ~1 t of propellant the nominal leaves above the reserve.

Raising the success rate further is an FL-302 / vehicle-sizing question,
not a terminal-guidance one.

## Consequences

- The pipeline's default landing step is the closed loop. The landing
  assumption strings say so, and the run's `landing_start` event still
  carries the insertion plan.
- `NOMINAL_PLAN_GUESS` (also the Monte Carlo warm start) is the 17.75 s
  script.
- FL-304's monkeypatched green tests are now labelled **wiring-only**. They
  pin the merge and statistics plumbing with the single-plan flow and a
  substituted verdict. The physical green E2E substitutes nothing.
- FL-302's corridor penalty is still nearly blind to misses under the
  corridor radius. Coupling the corridor to the landing burn's divert
  budget is the natural next step if the dispersed success rate matters.
- Out of scope, unchanged: density dispersion in the FL-302 optimiser, and
  the OpenTsiolkovsky stub scenario's `INSUFFICIENT_PROPELLANT_BOOSTBACK`
  verdict.
