# Flight Lab

Flight Lab simulates ascent and booster recovery from code or the CLI.
Mission Control's **Runs**, **Vehicles** and **Compare** workspaces display
saved flights, vehicle definitions and run comparisons.

<figure>
  <a href="assets/flightlab-runs-starbase.png"><img src="assets/flightlab-runs-starbase.png" alt="Starbase RTLS replay with a detailed coastal map, ascent and recovery tracks, and per-stage telemetry"></a>
  <figcaption>Sentinel-2 cloudless 2024 by EOX IT Services GmbH, containing modified Copernicus Sentinel data 2024 (CC BY-NC-SA 4.0).</figcaption>
</figure>

## Quick start

Install the [compute environment](quickstart.md#installation), fetch the
pinned binary, then run the packaged point-mass example:

```bash
python scripts/fetch_opentsiolkovsky.py
orbitopt flight validate src/orbitopt/data/flightlab/scenarios/leo_two_stage.yaml
orbitopt flight simulate src/orbitopt/data/flightlab/scenarios/leo_two_stage.yaml \
  --vehicle src/orbitopt/data/flightlab/vehicles/synthetic_two_stage.yaml -o runs/leo/
orbitopt
```

Select **Runs** to open the run. To export a standalone scene, use
`orbitopt flight to-scene runs/leo/ -o scene.json`. The
[command reference](https://zangjiucheng.github.io/orbitopt/api/cli/flight/)
lists all options.

## Viewing runs in Mission Control

- **Run list:** scans `./runs`, `ORBITOPT_FLIGHT_RUNS` (platform path separator)
  and remembered **Add folder…** locations. **Refresh** finds new runs;
  `orbitopt launch -o DIR` produces the listed `DIR/ascent` run.
- **Shared view:** selecting a run loads the 3D scene, timeline, Flight and
  Verification tab. Returning to **Missions** restores the previous mission.
- **Flight tab:** launch inputs, target/achieved orbit, verdicts, seekable
  events, failure explanations and recovery details.
- **Telemetry:** altitude, speed, dynamic pressure, load, mass and throttle;
  clicking a chart seeks the timeline. **Compare** opens two saved runs side by side.

The **Bodies** strip stays above the plots: click Earth, a stage, the booster
or a site to focus and follow it. Stage buttons
are enabled during their recorded lifespans. When a matching target definition
is available, launch/recovery markers stay fixed on the rotating Earth; RTLS
uses one shared marker. An unresolved offshore target is not replaced by the pad.

Plots also show **propellant remaining**, **radial vertical speed** and
recorded **downrange**. Fuel curves use recorded propellant where present;
otherwise `(est.)` means track mass minus dry mass and any attached upper
stages, using the vehicle file matched by the run's digest. Payload already
included in dry mass is not subtracted twice. Unknown dry masses remain missing.

**Double-click a telemetry plot** to open a larger, live-linked window.
Scroll to zoom around the pointer, right-drag to pan, hover for time and
interpolated per-series values, and use **Reset view** to restore the full range.
Left-click or drag still seeks the shared mission clock. The same controls
work on the inline plots and comparison telemetry.

Drag the horizontal divider above **Bodies** to resize the plot pane. In a short window,
use the telemetry scrollbar or **Shift + scroll** to reach lower plots.

In **Distance**, scroll vertically to zoom, then swipe horizontally or
right-drag to move through time. Left-click or drag to seek. Double-click
to fit the full time range.

Separation tooltips show the velocity difference and the applicable limit.
Recorded mission replays use their mission configuration when the run and
input digests match. Other runs use the standalone staging check.

[![Expanded propellant plot showing stage and booster estimates with a precise time readout](assets/flightlab-telemetry-detail.png)](assets/flightlab-telemetry-detail.png)

Targets and vehicles are edited in YAML or through the Python designer API;
the desktop mode renders runs rather than configuring or flying them.

## Vehicle view

Open **Vehicles** in the top workspace bar and select a catalog vehicle. This read-only
page shows its stage profile, masses, thrust, ideal Δv, recovery reserves
and declared thrust/staging schedule. Profiles use declared geometry when
available; otherwise they are labelled **schematic, not to scale**. Hover
source labels for provenance, dates and stated uncertainty.

[![Vehicle view showing the illustrative RTLS stack, declared thrust schedule and recorded E5-05a flight comparison](assets/flightlab-vehicle.png)](assets/flightlab-vehicle.png)

To compare a definition with a flight, select a saved run in **Runs**, then
return to **Vehicles**. The schedule adds flown event markers; **Declared vs
flown** lists timing and mass differences. The header states whether the run
used the exact vehicle file or its digest differs. Missing values appear as
`—`; differences alone are not verification verdicts.

The screenshot uses the illustrative `synthetic_two_stage_rtls` vehicle and
the recorded [E5-05a ascent](computed-mission.md). Ideal Δv excludes gravity,
drag and steering losses. Edit vehicle YAML to change the design; return to
**Runs** for the 3D flight view and telemetry.

## Mission pipeline

`flight-lab` runs an ascent through the unified pipeline. `two-stage-leo`
also grades station order, staging, mass, orbit insertion and optional
propagation/recovery:

```bash
orbitopt run src/orbitopt/data/flightlab/mission_leo_two_stage.yaml -o run/scene.json
orbitopt run src/orbitopt/data/mission_two_stage_leo.yaml
```

After separation, upper-stage and booster tracks have separate timeline
lanes and boundary checks. A central-gravity + J2 coast bridges staging gaps;
the post-insertion **display coast** is not verification.
`ascent.mode: run_dir` replays recorded input rather than optimizing an ascent.

The packaged `src/orbitopt/data/mission_two_stage_leo_recovery.yaml` is the
**E2-04 frozen integration fixture**: two-sample, illustrative ascent arcs
with a prescribed 400 km insertion and 35 t of booster propellant at
separation. Its recovery is simulated, but ascent is not re-flown and parking
propagation is explicitly out of scope. It is not evidence of one physically
propagated launch-to-orbit-and-return trajectory. See
[regression procedure](flightlab-regression-ci.md#e2-04-the-frozen-dual-branch-reference-mission-task-257)
and the [acceptance results](acceptance-e1-e4.md). For the
recorded **computed-ascent RTLS mission**, use the
[E5-05 workflow](computed-mission.md): it has a fixed-program OT ascent,
one shared separation handoff and optional cross-environment parking
propagation.

<span id="fidelity-levels"></span>

## Backend identities and fidelity levels

Canonical identities are recorded in run manifests, verification inputs and
`scene.flightRun.backend`:

| Backend | Scope | Boundary |
|---|---|---|
| `opentsiolkovsky` | Prescribed-attitude point-mass ascent, Cd(Mach), GCRS output. | Default ascent/optimization lineage; no attitude dynamics or TVC. |
| `native-pointmass` | Native US76/J2/rotating-Earth twin with kinematic attitude. | Native comparison plant, distinct from OpenTsiolkovsky. |
| `native-rigidbody` | Native rigid body, coefficient-table aero and lag/rate-limited TVC tracking. | Illustrative reference plant; no complete roll autopilot. |
| `jsbsim` | JSBSim rigid body with live mass/CG/inertia and gimballed TVC, ECEF output. | Base spike aircraft are zero-aero; `spike_stage_aero` is the generated aero-bearing reference. Opt-in, outside ascent optimization. |

`flight simulate --backend opentsiolkovsky|jsbsim` takes a vehicle YAML for
the former and a **JSBSim scenario YAML** for the latter. For the aero-bearing
reference (requires `orbitopt[sixdof]`):

```bash
orbitopt flight simulate src/orbitopt/data/flightlab/scenarios/leo_two_stage.yaml \
  --backend jsbsim --vehicle src/orbitopt/data/flightlab/sixdof/e104_aero_ascent.yaml \
  --no-refly -o runs/e104/
```

This held-vertical ascent exercises the airframe; the 400 km orbit-target
checks are expected to fail. `--no-refly` leaves independent physics
verification unassessed.
`flight mission --backend native-pointmass|native-rigidbody|BOTH` runs the
native comparison. Legacy `3DOF`/`6DOF` aliases resolve within their original
entry point; they are not canonical output identities.

### Aligned-scenario parity (E1-01)

The [parity document](../src/orbitopt/data/flightlab/parity/spike_stage_parity.yaml)
drives native stage construction and checks packaged JSBSim mass, inertia,
propellant, mounting and aero geometry. Since E6-01 the engines share the
propulsion law too: a choked `T_vac/(Isp_vac*g0)` = 599.083 kg/s flow through
the same physical 2 m^2 exit, so both sides pay the same ambient back-pressure
debit and agree on 1.6774 MN of pad thrust. The generated production airframe
adds the aero tables and cancels the fixed-AERORP/live-CG force couple. See
[ADR 0009](adr/flightlab-0009-engine-and-mass-flow-parity.md) for the aligned
residuals, [ADR 0007](adr/flightlab-0007-production-jsbsim-aero-path.md) for
the aero path, [ADR 0001](adr/flightlab-0001-jsbsim-6dof.md) for the base spike.

Recovery remains a separate Earth-fixed 3-DOF chain with J2, rotation,
atmosphere and engine/event models. Selecting a 6-DOF ascent does not make
recovery six-dimensional.

## Verification verdicts

| Run field | Meaning |
|---|---|
| `backend_self_check` | The producing backend's own checks. |
| `physics_verified` | Independent insertion/coast re-flight using TudatPy; JSBSim ECEF output is first bridged to GCRS. |
| `optimizer_converged` | Optimizer verdict, when present. |
| `mission_constraints_passed` | Declared flight-target constraints. |

`null` means unassessed. These fields are independent; a self-check is not an
independent re-flight. See [ADR 0004](adr/flightlab-0004-independent-physics-verification.md).
The mission rollup preserves this flag as independent-model evidence. A false
flag fails orbit injection and the full-mission gate; null remains unassessed.
See the [acceptance results](acceptance-e1-e4.md).

<span id="native-aero-model-and-the-reference-plant-f-02"></span>
<span id="tvc-actuator-and-attitude-tracking-f-04"></span>
<span id="6-dof-vs-3-dof-gravity-turn-ascent-f-05"></span>
<span id="two-stage-separation-and-upper-stage-ignition-f-06"></span>
<span id="mission-outputs-selectable-backends-reports-and-mission-control-f-07"></span>

## 6-DOF studies

The records cover engine parity, sampled control and native/JSBSim guidance
matrices. These are declared study vehicles and controllers, not a single
qualified production vehicle:

| Study | Recorded result | Scope / source |
|---|---|---|
| F-05 / E1-02 | Powered attitude error 4.77° peak, alpha 14.42°, max-Q 71.2 kPa. | Powered and coast gates differ; the later 62.6° alpha peak is in near-vacuum coast. [Record](benchmarks/fl605_gravity_turn_ascent.json). |
| F-07 / E1-03 | Native point mass: 225.3/228.5 km peri/apo; rigid body: 220.5/226.5 km, within ±25 km. | Shared feedback law on a disclosed 1 t dry / 7 t propellant upper stage with 3° gimbal; not original F-06 parity. [Record](benchmarks/fl607_mission_outputs.json). |
| E6-02 | Guided JSBSim staging: powered error 0.44° peak, alpha 11.7°; upper-powered attitude error 6.13° peak / 5.32° RMS. | Illustrative 70 t stack; upper-stage bounds differ from booster bounds. [ADR 0010](adr/flightlab-0010-guided-jsbsim-ascent.md), [record](benchmarks/e6_02_guided_ascent.json). |
| E6-03 / E10-08 | 12 of 17 points insert on both native backends (6 before E10-08), with no unexpected verdict; 180-300 km at every declared inclination. | The terminal-radius radial channel converts the guidance-class misses. The 400 km, heavy-stage, F-06 and A3-servo cells stay misses with named vehicle/actuator causes. [ADR 0011](adr/flightlab-0011-guidance-generalisation-envelope.md), [matrix](benchmarks/e6_03_guidance_matrix.json). |
| E6-05 / E10-08 | JSBSim and native verdicts agree on 16/17 cells, and all 130 parity checks pass; median peri/apo difference 12.0/3.1 km. | Shared frozen separation state. Only the A3 servo-boundary cell differs, and that difference is declared as expected (JSBSim inserts it); propellant agrees to <= 0.11% on co-inserted cells. [Record](benchmarks/e6_05_jsbsim_matrix.json). |
| E6-04 | Stateful guidance/control sampled once per step and held across RK4 stages. | The study justifies `dt_s = dt_c_s = 0.02`; this is not a universal backend step. [Sampled-control ADR](adr/flightlab-0008-sampled-data-control-and-rk4-plant.md), [sweep](benchmarks/e604_step_convergence.json). |

Run the native two-backend mission and open its saved outputs:

```bash
orbitopt flight mission --backend BOTH -o runs/f07-mission
orbitopt
```

For the guided JSBSim study (requires the six-DOF dependency):

```bash
PYTHONPATH=src python benchmarks/bench_e6_02_guided_ascent.py -o runs/e602.json
```

The guidance matrix (native) and its JSBSim applicability twin regenerate from
their benchmarks; both fly every cell independently, so `--jobs` scales with
cores (cells take minutes each):

```bash
PYTHONPATH=src python benchmarks/bench_e6_03_guidance_matrix.py --jobs 4 -o docs/benchmarks/e6_03_guidance_matrix.json
PYTHONPATH=src python benchmarks/bench_e6_05_jsbsim_matrix.py --jobs 4 -o docs/benchmarks/e6_05_jsbsim_matrix.json
```

The study writes measurement JSON rather than a Flight Lab run directory.
The [API reference](api.md#mission-and-qualification-interfaces) covers its
scenario, runner and comparison functions.

## Hard gates

Orbit and touchdown checks apply the declared velocity, position, orbital,
load and propellant limits. Reports record frames, epochs, units, provenance
and failure reason codes. A pass reaches only the modeled checks and limits;
[full-mission phase verdicts](verification.md#full-mission-verification-gate-e2-03-task-256)
show where evidence is absent or a branch failed.

## What is not modelled

The default ascent is point mass. The models omit CFD-grade aero, structural
loads, slosh, stage collisions, turbulence and range safety. Recovery may use
one steady dispersed wind vector; its re-planning is not an onboard-computer
simulation.

## Vehicle data, provenance and uncertainty

The [vehicle catalog](../src/orbitopt/data/flightlab/vehicles/) contains an
illustrative synthetic vehicle and a Falcon-9-class public estimate, not
official SpaceX performance data. Fields record source/date/provenance;
uncertainty bands can supply Monte Carlo defaults.

## Recovery uncertainty (R-04)

```bash
orbitopt flight montecarlo src/orbitopt/data/flightlab/scenarios/leo_two_stage.yaml \
  --vehicle src/orbitopt/data/flightlab/vehicles/synthetic_two_stage.yaml \
  -n 32 --seed 7 --capture-tracks 240 -o montecarlo.json
```

Pass `--uncertainty unc.yaml` to configure sources independently:

```yaml
uncertainty:
  thrust: {enabled: true, distribution: uniform_pct, half_width_pct: 5.0}
  wind: {enabled: true, distribution: normal, sigma_east_m_s: 4.0,
         sigma_north_m_s: 3.0, sigma_up_m_s: 0.5}
```

Wind affects acceptance re-flight, not the boostback optimizer rollout.
Sampling is fixed-seed deterministic and independent of `--jobs`.

## Recovery Monte Carlo statistics (R-06)

Reports include failure classes, Wilson success-rate intervals and per-source
sensitivity. Joint-dispersion associations are not isolated causal effects.
`--solo-sensitivity` adds one sample pass per active source; `--capture-tracks N`
retains at most N ENU samples per trajectory without changing random draws.

## Nominal / dispersed comparison (R-07)

The [32-sample evidence](benchmarks/r07_nominal_dispersed.json) records 14
landings in each arm. The baseline assumes unlimited TVC bandwidth; the stack
flies 20°/s slew plus 0.2 s lag with compensation. They are not an actuator-
matched comparison. The stack's nominal touchdown is 0.03 m from the pad;
position failures rise from 4 to 7. See
[ADR 0006](adr/flightlab-0006-rate-limited-landing-guidance.md) and
[ADR 0003](adr/flightlab-0003-landing-aware-boostback-objective.md).

## Joint chance-constrained planning (E3-02)

`plan_boostback_entry(joint_chance=...)` replaces the R-01 soft objective
terms with joint fuel, divert/scatter and landing-probability constraints,
solved as a **penalized optimization** (worst-violation exact penalty on the
Nelder–Mead search — not a constrained solver). Since E7-01 the constraint
verdict is enforced, not implied by the penalty: a plan whose worst
normalized violation exceeds `JointChanceSpec.violation_tolerance` comes
back `feasible=False` with reason code `JOINT_CHANCE_VIOLATED`. The recorded
32-sample joint arm lands 17; see
[ADR 0005](adr/flightlab-0005-joint-recovery-chance-constraints.md)
and [paired results](benchmarks/e3_02_joint_recovery.json) for assumptions.
Its planning probability is a surrogate, not a guaranteed flight success rate.

## Robust Monte Carlo qualification (E3-03)

The [four-seed study](benchmarks/e3_03_robust_montecarlo.json) uses 168 paired
samples per arm with matched draws and declared 20°/s / 0.2 s actuator physics:

| Arm | Landed / 168 | 95% Wilson interval |
|---|---|---|
| FL-305 baseline | 45 | 20.7–33.9% |
| E3-01 compensation | 49 | 22.8–36.4% |
| R-01 objective | 55 | 26.1–40.2% |
| E3-02 joint chance | 56 | 26.6–40.8% |

Against baseline, exact McNemar p-values are 0.125, 0.0309 and 0.0192,
respectively. These are **unadjusted pairwise** tests, not family-wise
significance claims. E3-02 versus R-01 remains unresolved (two versus one
discordant landings); paired landed-only scatter tests reject nothing.
The [benchmark script](../benchmarks/bench_e3_03_robust_montecarlo.py) records
all sample rows, failure classes and sensitivities. Statistics were recomputed
from those rows in the current acceptance review; the 672 flights were not
re-run.

**The E3-02 row predates E7-01.** It measures the old penalty-only mode.
With feasibility enforced, E7-04 re-flies the same draws and lands **35/168**;
99 plans are rejected as `JOINT_CHANCE_VIOLATED`. The baseline comparison has
McNemar p = 0.110. The old p = 0.0192 does not describe today's enforced mode.

## Probability calibration of p_land (E7-02)

The [calibration report](benchmarks/e7_02_probability_calibration.md)
reuses 168 recorded paired samples. The 0.5–0.8 bin predicts 0.656 but lands
0.076; the 0–0.5 bin lands 0/42. Above 0.8, mean predictions fall within the
observed Wilson intervals, whose small high-bin samples leave wide uncertainty.
This supports only that recorded scenario/dispersion set, not a guaranteed
success probability. API: `orbitopt.flightlab.recovery.calibration`.

## Joint risk modeling (E7-03)

The [joint-risk report](benchmarks/e7_03_joint_risk.md) rejects independent
fuel/divert failure channels (Fisher p = 0.010). A Gaussian copula fitted on
three other arms uses ρ = −0.273; on the held-out joint arm, Brier score
improves 0.222 → 0.214. The mid-band remains overconfident. The shipped
`JointChanceSpec.landing_dependence_rho` default is **0.0** (the product);
the fitted constant is opt-in. API: `orbitopt.flightlab.recovery.joint_risk`.

## Failure-driven optimization (E7-04)

The [paired re-fly report](benchmarks/e7_04_failure_driven.md) compares the
same 168 dispersion draws with TVC lag compensation and an optional lateral
cutoff gate. Lateral-speed failures fall 24 → 18 and landings rise 45 → 49;
this is **not significant after Holm correction** (raw category p = 0.0703,
adjusted 0.3516). The gate alone reproduces baseline verdicts; compensation
carries the measured change. The enforced joint arm stays 35/168 with or
without the fix. Default cutoff and corridor-aim knobs remain off; see
`orbitopt.flightlab.recovery.failure_driven` for the paired analysis.

## Out-of-sample reliability (E7-05)

The [reliability report](benchmarks/e7_05_out_of_sample_reliability.md)
diagnoses E7-04's 99/168 `JOINT_CHANCE_VIOLATED`: reconstructing the three
normalized violations from each recorded verdict (identity checked on all
99 rows) shows **51 bind on the divert budget, 48 on the fuel quantile, 0 on
the landing-probability term** -- and under the shipped spec the probability
term provably cannot bind alone (the sweep test in
`tests/test_flightlab_recovery_reliability.py`). The rejections are physical
(51/99 deep: the dispersion band itself misses the spec), with 20 of 99
rejected worlds landing anyway under FL-305 grading. Re-analysing the E7-03
headline per-seed instead of pooled: Mantel-Haenszel p = 0.017 and a
cluster-bootstrap CI of [-0.185, -0.058] survive, but seed 1 sits at phi = 0,
Fisher-combined p = 0.154 and no per-arm Holm p reaches 0.05. Hold-out flights
on disjoint seeds reproduce the pattern (39/168 landed, Brier skill -0.62,
overconfident mid-bins, 0 probability-bound rejections); widened bands (x1.5)
collapse to 2/84 and an independent ~180 t-class RTLS vehicle to 0/48, all
deep fuel-bound. The probability model generalizes only in shape, not level:
it stays unskilled out of sample. No success-rate improvement is claimed.
API: `orbitopt.flightlab.recovery.reliability`.

## Graded admission (E10-06)

The [chance-constraint redesign report](benchmarks/e10_06_chance_constraint_redesign.md)
replaces the E7-01 binary gate with a graded rule over the same reconstructed
violations: reject only deep violations (worst > 0.5), a second binding
channel (> 0.15), or a calibrated landing probability below the recorded
joint-arm climatology (0.208). The probability is a logistic refit on
fuel-quantile z and divert violation -- monotone by fit, and the only fit
that stays short of severe overconfidence when the dispersion bands shift
x1.5 (skill -0.09, ECE 0.03, against -9.38 / 0.43 for the raw surrogate).
Regrading the recorded grid row-for-row: 99 -> 84 rejections, false
rejections (joint-rejected, FL-305 landed) 20 -> 10, all 51 deep violations
kept, nothing previously accepted re-rejected; on the E7-05 hold-out the
calibrated model scores Brier skill +0.565 (raw: -0.62) with no
high-probability bin significantly below its own prediction. Wired into
FL-302 as `plan_boostback_entry(..., joint_admission=...)`; the search
objective is unchanged and no success-rate improvement is claimed.
API: `orbitopt.flightlab.recovery.redesign`.

## Counterfactual admission validation (E11-04)

The [counterfactual re-flight report](benchmarks/e11_04_counterfactual_admission.md)
replaces the E10-06 landing inference with measurement: every newly admitted
world (15 training, 16 `a_seeds`, 1 `b_bands`) is really re-flown without and
then with the policy on its recorded world. Measured pooled landing rate
**24/32 = 0.750** (Wilson 0.579-0.867), far above the 0.208 floor; the paired
McNemar over the re-flown flips is b/c = 19/0, p = 3.8e-06 -- the calibration
improvement does translate into landings, and no old-rule landing is undone
(b = 0 verified offline too). The training-grid measurement (8/15 = 0.533)
sits below the FL-305 inference of 0.667 without significant difference
(exact binomial p = 0.28; McNemar vs FL-305 3/5, p = 0.73): FL-305 is an
inference proxy, not a counterfactual, and the E7-04-era grid predates the
#287 guidance series (its replays report era drift, while the bit-valid E7-05
hold-out cells reproduce to 1e-9 and measure 15/16 + 1/1 landed; the one
admitted hold-out failure is landing accuracy, the seven training admit
failures are the live gate rejecting on today's physics). No rule update
required.
