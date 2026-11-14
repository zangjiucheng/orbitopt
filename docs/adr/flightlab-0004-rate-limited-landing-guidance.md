# ADR flightlab-0004: Rate-limited landing guidance (declared TVC chain + lag-aware law)

- **Status:** Accepted
- **Date:** 2026-11-14
- **Deciders:** Recovery Robustness (Epic #248), E3-01 (#258), follow-up to
  R-05 (ADR flightlab-0003 lineage, `benchmarks/bench_r05_actuator_margins.py`)
- **Code:** `src/orbitopt/flightlab/recovery/landing.py`
  (`TerminalGuidance.tvc_lag_compensation`,
  `TerminalGuidance.tvc_lag_compensation_cap_deg`,
  `_lag_compensated_command`, `terminal_guidance_command`,
  `_ClosedLoopLanding`, `TERMINAL_GUIDANCE_E301`,
  `DECLARED_LANDING_ACTUATOR`), `src/orbitopt/flightlab/recovery/montecarlo.py`
  (`MC_PLAN_OPTIONS["landing"]`)
- **Evidence:** `docs/benchmarks/r05_actuator_margins.json`
  (`benchmarks/bench_r05_actuator_margins.py`),
  `docs/benchmarks/r07_nominal_dispersed.json`,
  `docs/benchmarks/r01_boostback_objective.json`,
  `tests/test_flightlab_recovery_terminal_guidance.py`

## Context and problem

R-05 asked "how much does a 20 deg/s TVC slew limit cost the recovery
stack?" and answered honestly but with a blunt instrument: the limiter was
clipped onto the law's command, and the servo lag the repo's own JSBSim
spike-stage model declares (`gimbal_lag_tau_s = 0.2`, declared alongside
`gimbal_rate_limit_rad_s = 0.35` in
`src/orbitopt/data/flightlab/jsbsim/aircraft/spike_stage/spike_stage.xml`,
`<lag>`, then `<rate_limit>`, then the travel clamp) was left out of the
accounting entirely. E3-01 closes both gaps: the declared chain becomes the
default the recovery benchmarks fly, and the guidance stops treating the
actuator as pure loss.

The chain matters more than its individual elements. Flown naively on the
recovery Monte Carlo (n = 32, seed 1, default dispersions, same warm start
and referee -- `docs/benchmarks/r05_actuator_margins.json`):

| column | landing guidance | success | landed | lateral-speed failures | realised slew max |
| --- | --- | --- | --- | --- | --- |
| baseline_fl305 | FL-305 law, unlimited actuator | 0.531 | 17 | 0 | 215.6 deg/s |
| r05_actuator | + 20 deg/s slew clip | 0.500 | 16 | 3 | 20.0 deg/s |
| e301_lag_naive | + 0.2 s lag (series, no compensation) | 0.375 | 12 | 8 | 20.0 deg/s |
| e301_declared | + lag compensation (this ADR) | 0.438 | 14 | 5 | 20.0 deg/s |

The baseline's "margin" was an artefact of the unlimited-slew assumption --
R-05 measured a mean touchdown-phase demand of 54.9 deg/s (max 167.6) --
and the naive series chain is brutal in a specific, locatable way. Every
added failure is `LANDING_LATERAL_SPEED_EXCEEDED`, and the per-sample
tracks in the benchmark JSON's `failure_records` locate it: the failures
all happen in the last few hundred metres, where the ZEM branch's demand
(`6 (−r − v t) / t²`) reverses hard; a gimbal that needs 0.2 s to arrive
and 20 deg/s to get there overshoots, limit-cycles against the tilt cone,
and cuts off with 1.1--5.3 m/s of lateral speed left. The divert is not
arriving "a bit late" -- inside a response time of the pad the law is
demanding something the declared actuator cannot physically do.

## Options considered

1. **Fly the chain naively, guidance blind** (the `e301_lag_naive`
   column): honest, but 0.375, and it hands the ADR a failure mode the
   guidance can see coming. Kept as a benchmark column, not as the default.
2. **Response-reserved ignition boundary**: shorten the divert's time-to-go
   by the actuator response time so the law ignites/reserves earlier.
   Measured first, rejected -- it front-loads the same un-trackable ZEM
   spike instead of avoiding it and costs landings outright. The ignition
   boundary stays the FL-305 law's (ADR flightlab-0002).
3. **A divert-gain schedule keyed to actuator state** (scale the ZEM/PD
   gains by the tracking error): measured on the nominal case grid, it
   degrades tracking without fixing the terminal phase; the terminal phase
   is a timing problem, not a gain problem. Rejected.
4. **Make the law see the actuator at its two failure points** (adopted):
   the command interface, and the moment the ZEM branch stops being
   trackable. `TERMINAL_GUIDANCE_E301` (rate 20 deg/s, lag 0.2 s,
   `tvc_lag_compensation=True`) is the default in
   `MC_PLAN_OPTIONS["landing"]`.

## Decision

`TerminalGuidance.tvc_lag_compensation` (requires a declared
`tvc_lag_tau_s`; off -- bit-exact FL-305/R-05 -- otherwise) does two things,
both derived from the declared τ alone, no new free knobs beyond a cap:

- **Actuator interface** (`_lag_compensated_command`): the command leads
  the lag state by its own tracking error, `u = d + (d − s)`, capped at
  `tvc_lag_compensation_cap_deg` (default 10 deg) and still slew-limited.
  The error is taken against the *modelled lag state* the wrapper already
  carries, not a demand derivative -- a variable integration step cannot
  blow it up. The realised gimbal tracks the demand instead of trailing it
  one response time, and the first-order lag stays in front of the slew
  limiter exactly where `spike_stage.xml` puts it.
- **Terminal hand-over**: `t_h = t_go − settle_s − τ` under compensation.
  The last response time of the divert is flown by the damped terminal law
  (`lateral_damping_per_s`, `lateral_stiffness_per_s2` -- already an
  overdamped PD, ω_n τ ≈ 0.09, so the lag barely touches it), because a
  ZEM spike inside a response time is exactly the demand the declared
  chain cannot track. Without compensation the FL-305 hand-over point
  stands to the bit.

## Consequences

- All recovery benchmarks now fly the declared actuator set by default
  (`MC_PLAN_OPTIONS["landing"] = DECLARED_LANDING_ACTUATOR`); the R-01 and
  R-07 evidence regenerates with it (R-07 nominal recovers under the
  declared chain in both columns; the stack column matches the declared
  column above). The FL-305-era evidence files stay bit-exact -- those
  benchmarks pin `landing={}` on their baseline columns.
- The declared chain still costs landings versus the unlimited-slew
  baseline (0.438 vs 0.531): the honest reading of the four-column table
  is that the FL-305 stack's 17 landings needed ~216 deg/s to happen. The
  compensation recovers 2 of the 5 landings the naive lag loses and halves
  nothing less than the lateral-speed failure count (8 -> 5); the residual
  5 are deep-corner dispersion draws (records in the benchmark JSON) where
  no 20 deg/s actuator lands softly from that corridor miss -- that is a
  Boostback-objective problem (R-01 lineage), not a landing-law problem.
- The nominal lands with margin under the declared chain: the closed-loop
  textbook-handoff case puts down with realised slew peaking at 18.9 deg/s
  (inside the pin) and the compensation tightens the nominal touchdown
  lateral speed (0.030 -> 0.014 m/s on the handoff case).
- `ActuatorMargin` records `tvc_lag_compensation`, so a run's assumption
  line states which chain it flew; `bench_r05_actuator_margins.py` keeps
  the four-column before/naive-clip/naive-lag/declared comparison and the
  per-sample location of every failure as the standing evidence.
