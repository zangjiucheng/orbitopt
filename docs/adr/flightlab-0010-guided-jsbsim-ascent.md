# ADR flightlab-0010: Guided JSBSim ascent with staging (E6-02)

- **Status:** Accepted
- **Date:** 2026-11-15
- **Deciders:** E6 6-DOF Production Maturity (Epic #265), E6-02 (#274)
- **Code:** `src/orbitopt/flightlab/sixdof/guided_ascent.py`,
  `benchmarks/bench_e6_02_guided_ascent.py`
- **Evidence:** `docs/benchmarks/e6_02_guided_ascent.json`,
  `tests/test_flightlab_sixdof_guided_ascent_jsbsim.py`
- **Builds on:** ADR flightlab-0007 (production aero path), flightlab-0008
  (sampled-data control), flightlab-0009 (engine parity); FL-402 staging
  adapter, FL-403 PID-on-production-path, F-06 two-stage scenario

## Context and problem

Every earlier record stopped short of the full production loop:

- ADR 0007 flew the production airframe under the *native* law, open loop,
  and published residuals; it explicitly left the closed-loop production
  path to E6-02.
- FL-402 proved the JSBSim staging adapter (momentum-conserving property
  injection) but only on a ~150 m/s hop, where its absolute handoff
  tolerances (1e-6 m / 1e-6 m/s) are meaningful.
- F-06's two-stage mission (ADR 0002/0004) closes the loop with staging —
  but only on the native plant.
- FL-403 closed the loop on the production path (PID, TVC telemetry) but
  never reached MECO, let alone separation.

E6-02's acceptance is the intersection: a **closed-loop gravity turn with
TVC and staging on the JSBSim production model**, graded per phase against
declared constraints, with the native comparison attributed by channel
(dynamics / guidance / integration / model).

## Decision

Ship `e602_scenario` / `run_e602_study()` in `guided_ascent.py`:

- **Vehicle:** the production 70 t stack (E6-01 aligned engine, packaged
  `spike_stage_aero` tables), split at t = 95 s into 4 t / 6 t dry and
  0.4 / 0.6 propellant — FL-402's split fractions, on the E1-04 vehicle
  (not F-06's 80 t variant).
- **Guidance:** the F-04/F-06 gravity-turn waypoint program, steepened from
  110 s to **140 s** before hold. Under the production FCS lag the 110 s
  ramp drove AoA to 15.21°, outside the ±15° table envelope (the tabulated
  stall cliff); at 140 s the mission peaks at 11.7° inside it. This is a
  disclosed guidance design choice for the production path.
- **Law:** FL-403's production PID (`PidGains` defaults, no packaged
  production-gain config exists) on both plants' own actuator models; JSBSim
  samples every plant frame, native ticks at 0.05 s.
- **Pad:** 28.5° N / 80.6° W (the F-05/F-06 lineage pad). The equatorial
  pad (FL-403's convenience frame) was tried and the *native* two-gimbal
  tracker azimuth loop drifts ~42° there versus 4.3° at the Cape pad —
  a scenario-choice artefact of that law's azimuth gauge, not a JSBSim-vs-
  native residual, so the study runs where the lineage was tuned.
- **Staging:** commanded MECO at separation (E6-01 semantics), 10 kN·s
  impulse along body +x, FL-402 property-injected children, delayed
  ignition at 103 s on `spike_upper` (which carries no aero — it inherits
  the stage-1 spike engine and burns out ~3 s later; disclosed).
- **Phase constraint table** (`E602_PHASE_CONSTRAINTS`) declared with
  rationales before measurement: powered phases hold 6° peak / 3.5° RMS,
  ±15° AoA on the tabbed airframe (25° on the aero-free child, where the
  binding structural gate is `q·alpha ≤ 5e5` instead), coast phases are
  unactuated (F-06's zero coast command) and therefore gated only through
  max-Q-derived quantities; attitude drift in coast is reported, not gated
  (E1-02 discipline).
- **Staging bookkeeping checks** keep FL-402's absolute position tolerance
  but re-express velocity / momentum / attitude as **round-trip bands scaled
  with the handoff state** (5e-6 relative; 1e-5 rad Euler reprojection near
  pitch 90): FL-402's absolutes presuppose its hop state; at the ~4 km/s
  separation state the property round-trip (m/s ↔ ft/s, NED ↔ ECEF,
  quaternion ↔ Euler-deg) scales with the state. The native split still
  closes these at 1e-12 — the difference is the injection path's own
  accounting, attributed to the **model** channel.
- **Ignition schedule tolerance** is 1.5× each plant's onset *observation*
  period (integration step for the JSBSim hook, sample grid for the native
  study): 0.03 s vs 0.3 s.

## Measured (dt = 0.02 s, sample 0.2 s; all 38 checks pass)

| metric (stack powered) | JSBSim | native |
|---|---|---|
| attitude error max / RMS | 0.44° / 0.22° | 4.26° / 2.57° |
| AoA max / `q·alpha` max | 11.7° / 3.9e5 | 11.6° / 4.0e5 |
| max-Q | 71 353 Pa | 71 180 Pa (0.24 % apart) |
| TVC margin min / saturated | 0.96 / 0 s | 0.99 / 0 s |
| staging mass / momentum | exact / 92 N·s round-trip | exact / exact |

Upper powered phase: 6.13° / 6.15° peak on the two plants — the two laws
agree to 0.02° once the *same* actuator-free child dynamics face them.

## Residual attribution (the acceptance's four channels)

- **dynamics** — 0.076° max cmd−actual deflection lag (production FCS lag
  parameter 5.0, i.e. tau ≈ 0.2 s, rate 0.35 rad/s, vs the native
  law-internal 0.1 s actuator); 55 % command-peak difference for the same
  program: different actuators and per-frame vs constant inertia
  bookkeeping.
- **guidance** — the att-error *pair* (0.44° vs 4.26°) is the guidance
  channel itself: PID-on-live-instrumentation vs 0.05 s tracker ticks. Both
  sit inside their own 6° band, and the native side matches the FL-605
  lineage measurement (4.77°). The cross-law agreement bands (5° peak /
  3° RMS) are declared from those lineage numbers, not from a shared truth.
- **integration** — pad thrust 3e-6 rel, max-Q 0.24 % rel, separation state
  ~0.17 % rel; dt-halving shifts separation altitude by 5.8e-4 (JSBSim) and
  7e-7 (native): both plants are grid-converged at the shipped step.
- **model** — the property-injection round-trip bands above; the coast
  attitude drifts (92° both plants, 0.29° apart) are two unactuated bodies
  under the same central gravity, not a law difference.

## Consequences

- The production path now flies, stages and re-ignites under its own closed
  loop in CI; `tests/test_flightlab_sixdof_guided_ascent_jsbsim.py` guards
  the committed study plus a live short flight.
- FL-402's absolute handoff tolerances are scoped to low-speed handoffs;
  high-speed staging must use the scaled bands (`guided_ascent.py` exports
  the constants).
- Coast attitude remains unactuated by design; any future recovery-mode
  study needs its own coast law and its own constraint table.
- The equatorial-pad azimuth drift of the F-04 tracker is documented; a
  future task that wants the equatorial pad should fix the gauge, not move
  the mission.

## Alternatives considered

- **Native law on the JSBSim plant** — rejected: the acceptance asks for
  TVC *and* the production path; FL-403 already shows the production law
  holds MECO altitude to 0.4° on this airframe.
- **F-06's own 80 t vehicle** — rejected: E6 grades the production-maturity
  vehicle (E1-04 70 t), and F-06's own record stays native-only.
- **110 s program + wider AoA band** — rejected: the ±15° limit is the
  table envelope (physics), not a tuning knob; the program moved instead.
- **Keeping the equatorial pad and widening native bands to 45°** —
  rejected: that would grade the tracker's gauge artefact as if it were a
  plant difference, poisoning the comparison the acceptance asks for.
