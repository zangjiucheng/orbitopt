# ADR flightlab-0008: The sampled-data control loop and the RK4 plant (E6-04 numerical control audit)

- **Status:** Accepted
- **Date:** 2026-11-15
- **Deciders:** E6 6-DOF Production Maturity (Epic #265), E6-04 (#276)
- **Code:** `src/orbitopt/flightlab/sixdof/rigidbody.py` (`propagate`,
  `_deriv`), `src/orbitopt/flightlab/sixdof/tvc.py` (`AttitudeTvcLaw`,
  `run_closed_loop_ascent`), `src/orbitopt/flightlab/sixdof/gravity_turn.py`
  (`propagate_point_mass`, `_point_mass_deriv`),
  `src/orbitopt/flightlab/guidance/upper_stage.py`
  (`UpperStageGuidance.dcm_from_state`)
- **Evidence:** `tests/test_flightlab_e6_04_step_convergence.py`,
  `benchmarks/bench_e6_04_step_convergence.py`,
  `docs/benchmarks/e604_step_convergence.json`

## Context and problem

`AttitudeTvcLaw` is a *stateful* controller — a tracker tick, two integrator
states and two rate-limited actuator models — sampled at `dt_c_s` and held
zero-order onto the plant, which integrates with fixed-step RK4 at `dt_s`.
A preventive audit (no defect had been observed) asked the two questions the
two grids make non-trivial: does the controller ever advance at an RK4
mid-stage time, and are the control-step and integration-step errors cleanly
separable and of the orders their names imply.

Pre-audit, `propagate` evaluated the control law at **every** RK4 stage
(the stage derivative recomputed `control(t_stage, y_stage)`), and the
point-mass propagator did the same for `UpperStageGuidance`.

## Findings

**F1 — the tick fired at stage times and its command leaked one stage back.**
With the shipped default `dt_c_s == dt_s == 0.02` the tick fired at the *k4*
stage: the tracker/actuator state advanced with the stage-predicted state
`y + h·k3` rather than the accepted boundary state, and the post-tick command
entered the *current* step's k4 derivative — the next interval's command was
applied a partial step early. Measured consequence (closed loop, single
90→60 deg/12 s ramp, `dt_c_s` fixed 0.04): a **first-order `dt_s` plateau**
— final-altitude error −1.78e-3 / −8.29e-4 / −3.55e-4 / −1.18e-4 m at
`dt_s` = 0.04 / 0.02 / 0.01 / 0.005, halving ratios 2.15 / 2.34 / 3.00 ≈ 2.

**F2 — unaligned grids ticked at mid-stage times.** With `dt_c_s = 0.025`
against `dt_s = 0.02` the tick fired at `t + h/2`, i.e. at a stage estimate,
so the actuator substep and the tracker `dt` were driven by a
non-accepted state.

**F3 — a control tick below the plant step is silently decimated.** A fixed-
step plant cannot resolve a tick faster than its own step; pre-audit
`dt_c_s < dt_s` just ticked once per step with the merged period and nobody
said so.

**F4 — the guidance slew-limiter state was corrupted every stage pair.**
`UpperStageGuidance.dcm_from_state` slewed its command only when the clock
had *strictly advanced*; a re-read at a time it had already answered — which
RK4 guarantees, since k2 and k3 share `t + h/2` — fell through the guard,
returned the **raw un-slewed** pointing *and* re-seeded the slew memory from
it. The command-slew limit (`command_slew_rate_deg_s`) was therefore not
enforced across stage-pair re-reads on both the 6-DOF
(`GuidedUpperStageControl`) and the point-mass (`_point_mass_deriv`) paths.

**F5 — interior program kinks are order-zero plateaus.** A waypoint program
whose slope changes mid-flight is an O(1) discontinuity of the feedforward
command at the kink tick: boundary accumulated-time jitter of one ulp samples
the slope on either side of it. Measured: a 0.137 m final-altitude cliff
between `dt_s ≤ 0.005` and `dt_s ≥ 0.01` runs with *identical* tick
schedules — a plateau no amount of `dt_s` refinement removes. This is not a
code defect (the program itself is discontinuous in `C¹`); it is recorded so
a two-segment ascent program is never used for step-size validation, and the
benchmark scenario is a single ramp for exactly this reason. Actuator
travel/slew limit engagement is the same story in nonlinear form — limit
engagement is a discrete event — so the audit scenario also keeps margin on
both limits.

**F6 — loop contraction masks integration error in the loop output.** With
the audit fix in place, the closed-loop terminal state is invariant to
`dt_s` at the 2.3e-8 m float floor for every `dt_s ≤ 0.04` (the tracker
absorbs the truncation transient like the disturbance it is). The plant's
truncation order therefore has to be measured where it is visible — the
open-loop plant — and it is **4th order** there (vacuum plant, constant
command, measured ratios ≈ 16 per halving). The `dt_s` choice must still
resolve the plant, not just the loop: guidance states, aero and the J2
torsional pendulum do not get a feedback path.

## Decision

**D1 — one law evaluation per integration step.** `propagate` samples the
`ControlLaw` once per step, at the step-boundary state, and holds the
returned `ControlCommand` constant through all four RK4 stages. That is
exactly the law's declared ZOH semantics: a sampled-data controller knows
one instant per tick and no mid-stage estimate may advance the tick, the
tracker integrators or the actuator state. `run_closed_loop_ascent` and
every existing test semantics survive: ticks, history rows and sample-grid
re-evaluations all happen at boundaries.

**D2 — the point-mass path holds its guidance the same way.**
`propagate_point_mass` samples `command_from_state` once per step at the
boundary and passes the held `(Tec2b, throttle)` tuple to all stages. An
open-loop program (a pure function of stage time/state) stays
stage-resolved — holding it would be wrong.

**D3 — a guidance re-read at an already-answered time returns the held
command**, never the raw pointing, and never re-seeds the slew memory (F4).

**D4 — `run_closed_loop_ascent` warns when `dt_c_s < dt_s`** (F3): the
effective control period becomes `dt_s`.

**D5 — the default steps stay (dt_s, dt_c_s) = (0.02, 0.02).** Evidence
(`docs/benchmarks/e604_step_convergence.json`, 12 s single-ramp ascent):

- plant truncation (vacuum, `dt_c_s` fixed, reference `dt_s = 0.0025`):
  4th order — 9.2e-6 / 5.7e-7 / 2.5e-8 m at `dt_s` = 0.04 / 0.02 / 0.01
  (orders 4.01, 4.51; the next pair is at the reference float floor);
- closed-loop output vs `dt_s` (dt_c fixed 0.04): flat,
  ≤ 2.24e-8 m at every `dt_s ≤ 0.04` (F6);
- control discretisation (`dt_s = 0.005` fixed, reference `dt_c_s = 0.005`):
  at-least-first-order ZOH error, +18.9 / +6.8 / +2.3 mm at
  `dt_c_s` = 0.04 / 0.02 / 0.01 — this channel, not `dt_s`, dominates the
  default's total error;
- default vs refined `(0.0025, 0.0025)`: +7.9 mm altitude, +6.1e-4 deg tilt,
  +6.0e-6 rad max tracker error on a ~1 km climb (0.0008 %), fully accounted
  for by the `dt_c_s` term;
- unaligned ticks snap to the grid: every tick period in
  [`dt_c_s`, `dt_c_s` + `dt_s`) with boundary error ≤ 2.2e-13 s, and the
  tick schedule (hence the solution) depends only on the snapped schedule
  (`dt_c_s` = 0.05: 240 ticks at `dt_s` = 0.05 and 0.025 with a 1.9 mm
  final-altitude residual — aero-table crossing times, F5 — vs 200 ticks on
  the snapped `dt_s` = 0.02 grid).

Both defaults sit at the knee of the cost curve: `dt_c_s = 0.02` is the
coarsest common F-04 tick, and at `dt_s = 0.02` the plant truncation is two
orders below the ZOH term. Halving either buys ≤ 1e-5 m closed-loop for ≥ 4×
the steps — production propagation is the F-05 attribution study, whose
pre-declared phase constraints have 1e3–1e6 m margins.

## Consequences

- The law sees one evaluation per step boundary, so a stateful law's notion
  of time is the integration grid; laws with interior time schedules change
  phase by (at most) the half-step they previously sampled at — the ZOH-
  correct phase. Open-loop laws that were tuned against the old mid-stage
  phase shift by ≤ `dt_s`/2; none in the repo are.
- One control evaluation per step instead of four-plus: the closed-loop
  ascent is ~3× cheaper per step in law calls.
- Existing test configs with `dt_c_s < dt_s` (the 0.05/0.02 pair in
  `tests/test_flightlab_sixdof_tvc.py`) now emit the F3 warning; the runs
  are still valid and the warning is the point.
- Step-size convergence is locked by
  `tests/test_flightlab_e6_04_step_convergence.py` against the committed
  benchmark JSON; re-running
  `PYTHONPATH=src python benchmarks/bench_e6_04_step_convergence.py -o
  docs/benchmarks/e604_step_convergence.json` regenerates the curves.
