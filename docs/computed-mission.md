---
title: Computed mission workflow
parent: Flight Lab
nav_order: 0
description: Replay the recorded LEO and RTLS mission, add propagation across environments, and inspect evidence.
---

# Computed mission workflow

Choose a reference by its vehicle, flight target and evidence scope. A
nominal landing, a joint physical pass, independent-model agreement and
dispersion reliability are separate results.

| Workflow | What runs | Evidence boundary |
|---|---|---|
| E2-04 | Synthetic ascent endpoints + simulated recovery. | Fast integration fixture; no computed ascent or parking propagation. |
| E5-05 replay | Recorded computed ascent + freshly simulated recovery. | One canonical separation state; parking propagation remains `not_checked`. |
| E5-05 cross-environment | Recorded ascent + TudatPy parking propagation + gfold recovery. | The joined run must evidence all five physical checks. |
| E5-07 replay | Recorded Starship-class computed ascent, no recovery stage. | Physics Qualification Reference; recovery gap reported (`RECOVERY_GAP.md`), `full_mission` not declared. |
| E5-07 cross-environment | Recorded ascent + TudatPy mission leg with parking propagation. | Propagation evidenced in the foreign environment; joint gate NOT-DECLARED by design. |
| E5-08 design variants | Powered-entry RTLS at Kourou and Starbase. | DC7 and Starbase pass the nominal joint physical gate. The original gap remains a negative regression. [Results](e5-08-new-vehicle-report.md). |
| E8-01 reference check | Compares sealed records with raw backend output and closed forms. | No new flight; validates extraction, continuity and stated reference comparisons. |
| E12 rendezvous | Plans and replays a target and chaser in LEO. | Nominal, independent re-flight and dispersion results cover the declared reference scenario. [Replay and evidence](#e12-orbital-rendezvous). |

## E12 orbital rendezvous

The [Rendezvous](rendezvous.md) section carries the quickstart, the relative-motion
theory, the mission configuration and the full verification report.

The packaged **LEO-RDV-001** Mission Studio replay is precomputed under
`examples/runs/leo-rdv-001`. Open the scene without any compute stack:

```bash
orbitopt view examples/runs/leo-rdv-001-scene.json
```

or add the `examples/runs/leo-rdv-001` directory under **Runs** in Mission
Control. Follow the target and chaser, inspect maneuvers and safety checks in
**Flight → Rendezvous**, and select range, relative speed or the radial,
along-track and cross-track components in **Telemetry → Plots**. To rebuild
the run from source: `orbitopt rendezvous studio -o examples/runs/leo-rdv-001`
(`--config PATH` for another mission, `--dt S` for the sample interval). The
run directory includes `mission.json`, `replay.json` and the Rendezvous
panel's `rendezvous.json`; the sidecar carries the consumed `plan_digest`.
See the [replay format](adr/rendezvous-0004-mission-studio-flightlab-replay.md).

Three lines of evidence, kept separate — a nominal pass is neither an
independent pass nor a robust success:

| Evidence | LEO-RDV-001 result |
|---|---|
| Nominal plan (sealed, same-model) | The canonical two-impulse plan uses **10.345 m/s** total Δv (time of flight 3416.7 s) and passes landing, Δv-budget and whole-trajectory safety checks. The Mission Studio replay re-propagates the sealed plan **locally** (`propagate_pair`) — it is a replay, not independent verification. [Safety constraints](adr/rendezvous-0002-safety-constraints-whole-trajectory.md). |
| Independent re-flight (TudatPy, cross-environment) | The recorded nonlinear TudatPy two-spacecraft re-flight **passed** its declared checks, closest approach about **913 m**. It re-flew the sealed plan (10.345 m/s; the evidence carries the sealed plan's maneuver digest), so the packaged studio run reports `physics_verified: true`; evidence whose digest does not match gives `null` (fail-closed). [Re-flight report](benchmarks/e12_05_independent_refly.json). |
| Robust dispersion (Monte Carlo, not a nominal result) | Under the declared stress levels, **14 of 264** hold-out cases passed (5.3%, Wilson 95% CI 3.2–8.7%): 181 arrival-accuracy failures and 69 safety failures. This prices a locked mission design under stress; it says nothing against the nominal pass above, and the nominal pass does not soften it. [Campaign design](adr/rendezvous-0005-dispersion-campaign-not-a-safety-factor.md). |
| Independent robustness re-flight | With the passive contingency checks graded, **13 of 264** hold-out cases pass (4.9%, cluster-bootstrap 95% CI 2.7–7.6%). There are 164 arrival-accuracy failures and 87 safety failures. The hold-point coast is the dominant safety failure. TudatPy re-flew 23 selected samples, each with its own perturbation, and agreed with every main-model verdict. [Re-flight report](benchmarks/e12_10_independent_reflight.json), [design](adr/rendezvous-0004-independent-robustness-reflight.md). |

For the independent re-flight in a TudatPy environment, use
`orbitopt crossenv run --plan src/orbitopt/data/crossenv_e12_05_plan.yaml`.
See the [independent verification design](adr/rendezvous-0003-independent-refly-tudatpy.md)
and [CW validity study](benchmarks/e12_02_cw_validity.md) for the model limits.

## Replay and view the RTLS mission

Use the [landing environment](../envs/README.md), with gfold **0.3.2**, from
the repository root:

```bash
python scripts/e505_reference.py verify
python scripts/e505_reference.py run --out runs/e505
python scripts/e505_reference.py compare runs/e505
orbitopt flight to-scene runs/e505/recovery/run -o runs/e505/scene.json
orbitopt view runs/e505/scene.json
```

The last command needs the local viewer. Computation may run on another
machine: copy the output directory back, then open its scene locally. For
telemetry, add `runs/e505/recovery/run` in Mission Control's **Runs** workspace.
The scene is the combined flight run; the unified mission report is
`runs/e505/mission.verification.json`.

The reference was recorded in `64f64a6`, using code `74f30c8`. Its
`synthetic_two_stage_rtls` vehicle flew a fixed-program OpenTsiolkovsky
ascent from Kourou and reached **265.5 × 278.1 km at 51.0°**. This replaces
the old blocked 400 km/CMA-ES recording proposal. `update` re-flies that
computed ascent and the fixed recovery recipe; routine replay does not
re-optimize or re-fly ascent. See the
[vehicle decision](adr/flightlab-0008-rtls-capable-vehicle-redesign.md) and
[recording procedure](flightlab-regression-ci.md#e5-05-the-reproducible-full-physical-reference-mission-task-272).

The committed single-environment report passes its applicable phase checks,
but its physical gate is **`not_checked`** because parking propagation was
not run. Read that gate rather than treating the overall green report as
proof of a complete physical mission.

## Add parking propagation across environments

Install the checkout in the locked TudatPy Python 3.10 environment and a
separate Python 3.12 landing environment. Supply their interpreter paths:

```bash
orbitopt crossenv run --plan src/orbitopt/data/crossenv_e505_plan.yaml \
  --env tudat=/path/to/orbitopt/bin/python \
  --env gfold=/path/to/landing/bin/python -o runs/crossenv-e505
python scripts/e505_reference.py crossenv-check runs/crossenv-e505
```

Each stage seals its artifacts in a versioned exchange with SHA-256 digests
and interpreter/toolchain provenance. The consumer verifies it before use;
the recovery stage must report the same separation handoff identity. The
reference plan also selects the recorded Earth-orientation snapshot in both
interpreters; stage reports record its digest, which `crossenv-check` verifies. The
joined report must actually pass propagation, landing, insertion, mass and
continuity checks; configuring this plan alone is not evidence of a pass.
Since E5-06 the joined report carries a `joint_qualification` document:
`crossenv-check` re-grades it from the two sealed envelopes and requires all
five physical checks passed, each naming the environment and envelope digest
its evidence came from. The plan declares `full_mission: true`, so a missing
propagation or landing leg grades `not_checked`, never silently green.

## The Physics Qualification Reference (E5-07)

E5-05 above is the **Software Integration Reference**: a synthetic vehicle
with ADR-recorded relaxations, proving the chain -- never engineering-grade
verification. The **Physics Qualification Reference** flies the
public-sourced `starship_class_public_estimate` vehicle against
`leo_starship_class.yaml` (Max-Q 55 kPa public, 3.0 g ASSUMPTION never
relaxed; the flight peaks at 2.89 g), reaching **232.5 × 292.1 km at
51.98°**:

```bash
python scripts/e507_reference.py verify
python scripts/e507_reference.py run --out runs/e507
python scripts/e507_reference.py compare runs/e507
orbitopt crossenv run --plan src/orbitopt/data/crossenv_e507_plan.yaml \
  --env tudat=/path/to/tudatpy/bin/python -o runs/crossenv-e5-07
python scripts/e507_reference.py crossenv-check runs/crossenv-e5-07
```

The booster RTLS leg is **reported, not flown**: the vehicle's restart-2
aerodynamic-entry architecture has no representation in the FL-302 planner
(see `tests/data/mission/reference/e5-07-physics-qualification/RECOVERY_GAP.md`
for the verbatim planner rejections), so the mission declares no recovery,
the physical gate caps at `not_checked`, and the plan does not declare
`full_mission` -- the joint gate reports NOT-DECLARED instead of a relaxed
pass. Details in the
[recording procedure](flightlab-regression-ci.md#e5-07-the-physics-qualification-reference-starship-class-task-291).

## Check the recorded chain without re-flying

```bash
python -m orbitopt.verify.full_chain --out runs/e505/full-chain.verification.json
```

This checks 22 reference comparisons. Its independent-model evidence covers
the chain's extraction of raw OT ascent data and the booster's shared
unpowered window. Touchdown limits are **design input**; no historical
telemetry or independently validated landing is claimed. See
[verification](verification.md#full-chain-reference-validation-e8-01-task-281).

## E10 recovery and guidance studies

| Study | Recorded result |
|---|---|
| E10-01 / E10-02 | The recovery planner supports unpowered entry, and the flight models represent hot staging. The [hot-staging benchmark](benchmarks/e10_02_hot_staging.json) compares the timelines. |
| E10-03 | In the paired 168-case hold-out, DC7 landed in 44 cases and Starbase in 11. The DC8 arms landed in zero cases. [Closure study](benchmarks/e10_03_robust_rtls_closure.md). |
| E10-04 | DC7's independent checks pass after correcting event-state extraction. Starbase's debris impact-speed discrepancy remains outside its band. [Discrepancy audit](e5-08-new-vehicle-report.md#e10-04-closing-the-two-open-discrepancies-task-301). |
| E10-05 / E10-06 | Compare [recovery planning strategies](benchmarks/e10_05_recovery_planner_robust.md) and [graded admission](benchmarks/e10_06_chance_constraint_redesign.md). |
| E10-07 | An independent model re-flies recovery from the actual DC8 separation handoff and reproduces the corridor failure. [Re-flight report](e10-07-full-recovery-reflight.md). |
| E10-08 | 12 of 17 scenarios insert on both native backends, up from six. [Guidance matrix](flightlab.md#6-dof-studies). |
| E10-09 | Cross-environment CI grades the frozen DC7 and Starbase plans and checks that the original E5-08 reproduces its expected failure. [CI evidence rules](verification.md#final-schemes-in-cross-environment-ci-e10-09-task-306). |

## E11 aerodynamic RTLS closure

| Study | Recorded result |
|---|---|
| E11-01 | Grid fins reduced DC8 peak dynamic pressure and cut the best arrival speed from 1,068 to 362 m/s. The two-burn design still missed the 350 m/s speed and 15° tilt limits. [Grid-fin authority](benchmarks/e11_01_grid_fin_authority.md). |
| E11-02 | The separation-state screen made zero false rejects across 672 recorded hold-out cases. [Feasibility envelope](benchmarks/e11_02_recovery_feasibility_envelope.md). |
| E11-03 | Co-design landed in 0 of 168 paired hold-out cases, versus 44 for DC7 and 11 for Starbase. [Recovery co-design](benchmarks/e11_03_robust_recovery_co_design.md). |
| E11-04 | 24 of 32 newly admitted cases landed in counterfactual re-flights. [Admission validation](benchmarks/e11_04_counterfactual_admission.md). |
| E11-05 | Starbase’s independent debris impact-speed discrepancy remained +12.70% against a 12% tolerance. [Aero qualification](benchmarks/e11_05_starbase_aero_qualification.md). |
| E11-06 | The experimental Starbase replay records two booster ignitions, one for boostback and one for continuous entry and landing. Its saved landing and orbit re-flight checks pass. [Inspect the run](flightlab.md#viewing-runs-in-mission-control). [Gate record: missions, re-flights, FL-304 rates with CIs](#e11-06-gate-record-task-390). |
| E11R-01 | Ranking each search's feasible pool on training seeds and flying the best pinned landed 35–36 of 36 hold-out samples for each of 5 search seeds, against 6–33 of 36 for the first-feasible stop of the same runs (significant per seed except seed 5). Against the sealed 33/36 champion the gain is not significant (Holm p = 1), and the pooled 179/180 reuses the same 36 samples for every seed. [Campaign report](../tests/data/mission/reference/e11r-01-robust-candidate-selection/campaign.md). |
| E11R-02 | The twin's 18.2% entry peak-q gap is the density-profile form, inside the main model's single 0–25 km table segment; the 12% band is unchanged. [Peak-q error budget](benchmarks/e11r_02_peak_q_error_budget.md). |

Add the archived `tests/data/mission/reference/e11-06-starbase-continuous-burn/mission-hs/recovery/run`
folder in **Runs** to inspect the hot-staging burn sequence and fin telemetry.

At hot staging, the mission chain can bridge the gap between sampled tracks
using observed acceleration and mass flow while the parent is still firing.
The handoff report records the bridge method and tolerances. The powered
limits were calibrated on one Starbase separation. Separations after cutoff
use the existing coast bridge. E11-06 fin drag uses an assumed calibration,
and the saved replay shows a nominal simulation.

### E11-06 gate record (task #390)

Both archived mission chains pass all 71 listed checks. The separate SECO
orbit re-flights also pass. With the selected design, FL-304 recorded 33/36
landings on each ascent with a pinned script and 36/36 with per-sample
re-planning on the sealed ascent. The independent twin's peak dynamic
pressure differs by **18.2%** against a **12%**
band. E11R-02 attributed that gap to the density-profile form inside the main
model's single 0–25 km table segment, without moving the band: [peak-q error
budget](benchmarks/e11r_02_peak_q_error_budget.md). The
[gate archive](../tests/data/mission/reference/e11-06-starbase-continuous-burn-reports/README.md)
includes the intervals, fin drag assumption and search sensitivity.

## Earlier studies

| Study | Read or reproduce |
|---|---|
| E6: engine parity, guided JSBSim staging, guidance envelope and control-step convergence | [6-DOF studies](flightlab.md#6-dof-studies) and ADRs 0008–0011. |
| E7: enforced joint constraints, probability calibration, coupled risk and paired failure analysis | [Recovery studies](flightlab.md#joint-chance-constrained-planning-e3-02). |
| E8: required-phase qualification, discrepancy attribution and reference portability | [Verification](verification.md) and [tolerance audit](reference-tolerance-audit.md). |

Study scripts export measurement JSON; they do not necessarily produce a
Flight Lab run directory. Recorded results describe their declared vehicles,
controllers and dispersions, not a general flight qualification.
