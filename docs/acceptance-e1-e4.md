---
title: E1–E4 acceptance review
parent: Engineering
nav_order: 5
description: Scoped verification, resolved findings and evidence boundaries.
---

# E1–E4 acceptance review

**Date:** 2026-11-15. **Initial reviewed revision:**
`d41ae5fb241fa072c8e2109067175339c1d12932`.
**Result: both reproduced blockers resolved in the accompanying fix.**
This is scoped local acceptance, not proof of deployed documentation or remote CI.

The review covers canonical backend identities, JSBSim aero, native upper-stage
guidance, dual-branch integration, recovery actuator/chance/statistics work and
the evidence/acceptance matrix. The initial runs had 599 passes, one skip and
two E2-04 failures, plus the independent-physics counterexample below.

## 1. Ascent physics evidence is dropped

**Resolved P1.** Previously, changing only the frozen ascent's
`physics_verified` flag to false still produced a passing chain, report and
full-mission gate. Neither boolean value appeared as independent evidence.
The chain now emits `ascent.physics_verified` for both boolean values:

| Input | Check / independent evidence | Orbit injection / full mission |
|---|---|---|
| `true` | passed | passed, when other gates pass |
| `false` | failed (`physics_verification_failed`) | failed |
| `null` | explicitly unassessed; no independent check | governed by other applicable checks |

The regression covers all three values with and without recovery and validates
the serialized report. Null does not become positive independent-validation
evidence. A pass on design constraints alone still does not prove an independent
re-flight occurred.

Code: [`two_stage_leo.py`](../src/orbitopt/mission/two_stage_leo.py),
`run_two_stage_leo`; [regression](../tests/test_mission_phases.py).

## 2. Frozen-reference portability is not accepted

**Resolved P2.** The initial macOS run exposed frame roundoff of about
`1.3924e-7 m` and `m/s`, exact equality on numeric event lists, and advisory
landing-plan drift of `4.817e-4 kg` fuel / `3.029e-3 s` duration.
An isolated landing-CI dependency environment also exposed different offline
IERS defaults between Astropy 6 and 8, producing metre-scale frame differences.

The reference now includes a four-row Earth-orientation snapshot from the
original recording data package. Only reference builds/reruns select it; the
caller's table is restored afterwards. Recursive comparisons apply numeric
budgets to arrays while rejecting changed shapes, keys, booleans and non-finite
values. Rebuilt frame components have a `1e-6` absolute budget. Only advisory
landing-start fuel/time fields use `0.001 kg` / `0.01 s` budgets, justified
against solver convergence and integrator resolution.

**Recorded flight outputs are unchanged.** The physical-state limits remain
`0.01 m` and `1e-4 m/s`. Tests explicitly reject larger flown-state drift and
out-of-budget planning drift. The metadata-only update verifies existing hashes
first and preserves original recording provenance. See the
[reference procedure](flightlab-regression-ci.md#e2-04-the-frozen-dual-branch-reference-mission-task-257).

Code: [`e204_reference.py`](../scripts/e204_reference.py);
[regressions](../tests/test_e204_reference.py).

## Fix verification

| Environment / scope | Result |
|---|---|
| macOS Python 3.12, NumPy 2.5.4 / SciPy 1.18.1 / Astropy 8.0.2; reference, phases, mission chain and reports | 142 passed, 1 skipped, 1 deselected |
| Isolated Python 3.12 landing-CI requirements: NumPy 1.26.4 / SciPy 1.13.1 / Astropy 6.1.7; reference and phases | 92 passed |

Additional checks: backend/parity, guidance, recovery and mission/report core
(non-slow): **283 passed, 1 skipped**. Real JSBSim production ascent/comparison,
combined ascent/recovery and numerical/analytic acceptance: **4 passed**.
Additional scenes, contract, CLI/API and offscreen Qt/timeline checks
(non-slow): **280 passed**.
These overlap the table above and are not an additive total.

Both environments ran the real frozen recovery simulation and compared it
against the original recorded output with gfold 0.3.2. The isolated environment
is a dependency compatibility check on macOS, not an executed Linux CI job.
The deselected test is the expensive full reference optimizer; the skipped
negative test expects pygmo to be unavailable, but pygmo is installed. Some general phase fixtures warn about default
IERS data coverage; reference reruns use their frozen snapshot.

For E3-03, the recorded 168 paired rows per arm reproduce all four success
counts, Wilson intervals and six exact McNemar p-values. These are unadjusted
pairwise tests; the 672 recovery flights were not re-run.

## Evidence boundaries

E2-04 uses a synthetic, two-sample ascent handoff and prescribed orbit plus a
real recovery simulation. Parking propagation is out of scope. Passing this
fixture does not establish one continuously propagated ascent, orbit and return.
(E5-01, task #268: that gap is closed on the mission code path -- the full
mission chain's ascent is now computed by OpenTsiolkovsky with event-determined
staging and full time histories, `tests/test_mission_e5_real_ascent.py`; the
synthetic E2-04 fixture remains annotated as a fast-test exception only.)
The native F-07 comparison uses a disclosed modified upper stage.

This review does not establish full expensive optimizer/M-07 launch,
full-repository suite, interactive desktop rendering, CUDA, Windows or Linux CI
acceptance. Historical benchmark files and offscreen widgets do not substitute
for those checks.
