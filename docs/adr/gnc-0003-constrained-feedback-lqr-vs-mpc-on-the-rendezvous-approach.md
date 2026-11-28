# ADR gnc-0003: constrained feedback — LQR vs MPC on the rendezvous approach (E13-05)

- **Status:** Accepted
- **Date:** 2026-11-27
- **Deciders:** E13 Closed-Loop GNC & Relative Navigation (Epic #416), E13-05 (#422)
- **Code:** `src/orbitopt/gnc/lqr.py`, `src/orbitopt/gnc/mpc.py`, `src/orbitopt/gnc/constraints.py`,
  `src/orbitopt/rendezvous/constrained_campaign.py`, `benchmarks/bench_e13_05_constrained_control.py`
- **Evidence:** `tests/test_e13_05_constrained_control.py`,
  `docs/benchmarks/e13_05_constrained_control.{json,md}`
- **Builds on:** ADR gnc-0001 (contracts/clock), gnc-0002 (EKF seam), and the E13-04 closed-loop
  baseline documented in `docs/benchmarks/e13_04_closed_loop_guidance.md`

## Context

E13-04 put a CW mid-course retarget law in the closed loop and it works (258/264 on the LEO-RDV-001
holdout). E13-05 asks the next question: *is the schedule-shaped feedback the right shape?* Two
standard replacements were implemented and compared on the identical seam — same truth physics, same
EKF and sensing stack, same planned departure burn, same perturbation and execution draws, same
grading — differing only in the feedback law:

- **LQR** (`CwLqrLaw`): the frozen CW LQR gain from the shared Riccati sequence, saturating at the
  pulse cap, no constraint model at all.
- **MPC** (`CwMpcLaw`): 12-step (120 s) receding-horizon condensed QP over the same CW impulse
  model, with the mission's safety constraints *inside* the optimizer — per-pulse thrust cap,
  remaining Δv budget (smoothed `sum|u|` surrogate), keep-out sphere, cross-track corridor, terminal
  approach cone, and the range-indexed approach-speed ceiling; terminal cost priced by the same
  Riccati sequence so a short horizon still values the tail.

Three structural decisions were forced by the experiment before any method comparison could be
fair; they are the substance of this ADR as much as the numbers.

## Decision

1. **The terminal pair is scheduled, not law-dependent.** A single 3-DOF impulse cannot null both
   position and velocity: measured directly on the seam, the best capped burn 27 s from terminal
   leaves a ~148 m residual because the velocity weight (the tolerance contract, 0.2 m/s vs 50 m)
   rightly dominates. So both arms (and only via the same `ManeuverPlan` pair the baseline schedule
   carries: trim at t_f−40 s / 3 m/s cap / velocity weight 0.25, arrival at t_f−20 s / 8 m/s cap /
   velocity weight 100) fire `scheduled_burns` at fixed epochs; feedback replaces the baseline's
   MID-COURSE retargets only. The laws reserve the unfired burn caps inside their own budget gate
   (`budget_available_m_s`), so a feedback tick can never strand the terminal pair. This makes the
   comparison a clean *mid-course law* comparison and keeps every arm inside the mission's own
   manoeuvre contract.
2. **Feedback tracks the sample's own planned arc, not the hold point.** Both laws regulate
   `ρ − ρ_ref(t)` against the sample's realised planned transfer (the same arc E13-04 flies), so a
   mid-course law is judged on how well it follows the mission plan it was handed, and constraint
   margins are evaluated where the plan puts the vehicle.
3. **MPC falls back to the LQR gain, never to coast.** A failed or non-finite SLSQP solve issues the
   frozen-gain step and increments a counter that the benchmark reports. A guidance law that drops a
   tick costs the mission more than one that risks a single tick's margin.
4. **Prediction is linear-CW inside the QP** — a stated design assumption of the research: over a
   120 s horizon the J2/drag relative effect is mm–cm, two orders below the 500 m keep-out the
   constraint cares about. If a future mission closes that gap, the prediction model (not the QP)
   is the thing to replace.

## Measured (full 264-sample holdout, identical draws; see the JSON)

| arm | success | Wilson 95 % | terminal pos med (m) | Δv med (m/s) | realized family violations | decision ms mean |
|---|---|---|---|---|---|---|
| baseline | 258/264 | [0.951, 0.990] | 42.1 | 11.38 | 0 | 0.011 |
| LQR | 260/264 | [0.962, 0.994] | 45.8 | 11.50 | 0 | 0.36 |
| MPC | 255/264 | [0.936, 0.982] | 45.4 | 15.97 | 0 | 23.4 |

Paired exact-McNemar (Holm): neither arm's success endpoint separates from
the baseline (LQR 3↔1, p = 0.625; MPC 3↔6, p = 0.508; both violation
endpoints 0↔0). Failure *classes* move: baseline loses 6 samples to the
campaign safety checks, MPC to capped-arrival misses (6 `arrival_accuracy`,
worst 176 m) caused by its 6.05 m/s mean mid-course spend eroding the
terminal reserve; the four realized families held in every arm, with MPC
flying tightest to the walls it can see (speed +0.07 m/s, cone +9.8 m).

## Consequences

- **Applicability.** LQR is the right default when the plan's constraint margins are wide and the
  onboard computer is not: it is ~0.36 ms per decision, closed-form, and arrived
  260/264 on the same footing as the schedule it replaced. MPC earns its cost exactly where
  the mission's own families bind — it is the only arm whose optimizer *knows* the keep-out,
  corridor, cone and speed ceiling, and it flew zero realized violations (14 036 fallback ticks, 15.7 % of decisions,
  mean 23 ms/decision — affordable only with a real-time-capable QP solver and a 10 s
  cadence).
- **The speed family is where a schedule-shaped law and a constraint-aware law can disagree about
  feasibility**: the plan itself flies inside +2.5 m/s of its ceiling at the closest approach, and
  a feedback law that spends margin early has nothing left there. MPC's horizon sees that wall
  coming; LQR finds out at the tick it happens.
- **Δv is not free but is not the discriminator either**: both arms land inside the 25 m/s budget
  with headroom; the MPC's mid-course cost premium comes from back-pulses against its own constraint
  shaping, visible in the per-phase Δv table in the artifact.
- **Neither arm changes the terminal statistics materially** — errors are terminal-pair-limited
  (decision 1), so method choice is about constraint handling and compute, not arrival accuracy, on
  this mission.
- The identity guard in the benchmark re-flies the baseline arm and aborts on any disagreement with
  the frozen E13-04 artifact; the three-arm numbers are only comparable because that guard runs.

## Alternatives rejected

- **Terminal burn inside the law** (a shared retarget fired by each arm at window end): rejected —
  it makes arms differ in a dimension the task does not compare, and one 3-DOF burn demonstrably
  cannot close both tolerance channels.
- **MPC falling back to coast on solve failure:** rejected — a dropped tick at 500 m inside the cone
  is the worst case the constraint set exists to prevent.
- **Regulating to the hold point instead of the planned arc:** rejected — it would grade the laws
  against a trajectory the mission never flies and hide plan-constraint interactions.
- **Hard `sum|u|` budget constraint in the QP:** rejected in favour of the smoothed surrogate
  (eps = 1 mm/s ≪ execution error) — the nonsmooth version needs an NLP solver for no operational
  gain.
