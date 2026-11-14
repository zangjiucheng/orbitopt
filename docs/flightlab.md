# Flight Lab

Flight Lab simulates rocket ascent and booster recovery. Runs are made from
code or the CLI; Mission Control's **Flight Lab** mode renders them in the same
3D view, timeline and inspector as a mission.

![A Flight Lab run in Mission Control: the launch summary, key events, telemetry and checks](assets/flightlab-replay.png)

## Quick start

Install the [compute environment](quickstart.md#installation), then fetch the
pinned OpenTsiolkovsky binary and run the included example:

```bash
python scripts/fetch_opentsiolkovsky.py
orbitopt flight validate src/orbitopt/data/flightlab/scenarios/leo_two_stage.yaml
orbitopt flight simulate src/orbitopt/data/flightlab/scenarios/leo_two_stage.yaml \
  --vehicle src/orbitopt/data/flightlab/vehicles/synthetic_two_stage.yaml -o runs/leo/
orbitopt view        # switch to Flight Lab: runs/leo is in the list
```

`orbitopt flight to-scene runs/leo/ -o scene.json` exports the same run as a
plain scene file. See the [command reference](https://zangjiucheng.github.io/orbitopt/api/cli/flight/)
for target export, optimization, simulation and recovery options.

## Viewing runs in Mission Control

Flight Lab only renders runs; it has no designer. Edit the target and vehicle
YAML (or call `orbitopt.flightlab.designer` from code), run the CLI, and the
result shows up in Mission Control.

- **Run list.** The Flight Lab side of the sidebar lists every run directory
  under `./runs`, the folders in `ORBITOPT_FLIGHT_RUNS` (separated by `:`; `;`
  on Windows) and folders added with **Add folder…**, which are remembered.
  `orbitopt launch -o DIR` writes `DIR/ascent`, listed as `DIR`. **Refresh**
  picks up new runs; the marker shows whether the run passed its mission
  constraints.
- **Same view as a mission.** Selecting a run loads it into the 3D view,
  timeline (dragging the playhead pauses), **Inspector** and **Verification**
  tab (staging handoffs included). Switching back to Missions restores the
  mission that was open.
- **Flight tab.** The launch site and coordinates, liftoff epoch and azimuth;
  vehicle and payload; target and achieved insertion orbit; optimizer,
  physics and mission verdicts; the key events (liftoff, max-Q, MECO,
  separation, SECO, …) with altitude and speed at each, which seek the
  timeline when clicked; and **Checks**, **Why** (each failure in plain
  language, with a fix) and **Recovery**.
- **Telemetry tab.** Altitude, speed, dynamic pressure, g-load, mass and
  throttle against mission time, under the view; click a chart to seek.
- **Compare runs…** opens two runs side by side: inputs, outcomes and why
  they differ.

## Mission pipeline

The `flight-lab` mission kind runs an ascent through the unified pipeline. Its
handoff includes frame, epoch, position, velocity and mass; failures produce a
best-effort warning. The `two-stage-leo` reference scenario also grades the
stage boundaries, injection and propagated parking orbit:

```bash
orbitopt run src/orbitopt/data/flightlab/mission_leo_two_stage.yaml -o run/scene.json
orbitopt run src/orbitopt/data/mission_two_stage_leo.yaml
```

Mission Control shows the stages and handoff checks on one timeline. The
post-insertion display coast is not verification. For the reference chain,
only the ascent lineage is graded; a two-body + J2 coast (the gravity the
ascent model flies) bridges the staging gap.
`ascent.mode: run_dir` replays an existing run instead of optimizing it. See
[verification](verification.md) for report meanings.

## Backend identities and fidelity levels

Since E1-01 every physics engine has **one canonical backend identity**,
defined in `orbitopt.flightlab.backends` and recorded on every artefact it
produces: the run-directory manifest (`manifest.json` `backend`), the
verification report (`inputs.backend` + `inputs.backend_fidelity`), and the
Mission Control scene (`flightRun.backend`). The coarse `3DOF`/`6DOF` wording
that used to name *two different engines* (`flight simulate --backend 6DOF`
was JSBSim, `flight mission --backend 6DOF` the native plant) survives only
as a deprecated input alias inside the entry point that historically took it;
it is resolved at the CLI and never written into a run, report or scene.

| Identity | Models | Where its word stops |
|---|---|---|
| `opentsiolkovsky` | Point-mass ascent with a prescribed attitude program and Cd(Mach); GCRS states from the pinned v1.0.0 external binary. | No attitude dynamics, no TVC, no staging forces; the Gate-B pipeline and the FL-304 recovery chain build on this lineage. |
| `native-pointmass` | Point-mass twin on the native plant (US76 + J2 + rotation), kinematic program attitude, q-bar S CA/CN aero. | No attitude states — the program *is* the attitude. The 3-DOF side of the F-05/F-07 comparison, not the OpenTsiolkovsky adapter. |
| `jsbsim` | Full JSBSim rigid-body dynamics with gimballed TVC and propellant-driven mass/CG/inertia on the E4 spike models; ECEF states. | The packaged *base* aircraft fly with **zero aero** (a TVC/mass paper test); since E1-04 the **production path** is the generated `spike_stage_aero` aircraft, compiled from the packaged aero document, and it is the aero-bearing ascent reference. Outside the optimization loop; opt-in `orbitopt[sixdof]` extra. See [ADR 0001](adr/flightlab-0001-jsbsim-6dof.md) and [ADR 0004](adr/flightlab-0004-production-jsbsim-aero-path.md). |
| `native-rigidbody` | Six-DOF rigid body with coefficient-table aero, the F-04 tracker and a lag/rate-limited TVC actuator; US76 + J2 + rotation; ECEF states. | The reference plant of the native studies (F-02/F-04/F-06/F-07): illustrative coefficients, basic tracker without roll control; not the JSBSim backend. |

CLI mapping: `flight simulate --backend opentsiolkovsky|jsbsim` (default
`opentsiolkovsky`; legacy `3DOF`/`6DOF` resolve to exactly these two here);
`flight mission --backend native-pointmass|native-rigidbody|BOTH` (default
`BOTH`; legacy `3DOF`/`6DOF` resolve to exactly these two here).

### Aligned-scenario parity (E1-01)

Backends may differ in fidelity but not about the scenario they claim to
fly. `src/orbitopt/data/flightlab/parity/spike_stage_parity.yaml` is the
shared parameter source for the aligned spike-stage family: the native
studies build their stage pair **from that document**
(`f06_two_stage_scenario` / `f05_scenario` via
`orbitopt.flightlab.parity.native_stage_pair`), and
`orbitopt.flightlab.parity.check_parity` cross-checks it against the
parameters the packaged JSBSim XMLs actually declare. Masses, propellant,
inertias, engine mount station and Isp match exactly; stage-1 thrust agrees
within the documented band (JSBSim derives it from `isp·g0·ṁ`, ~0.15 % from
the canonical 1.88 MN); aero *geometry and reference quantities* are shared,
the zero-aero boundary of the *base* aircraft stays disclosed, and the
generated production aircraft (`jsbsim.production_aircraft`,
[ADR 0004](adr/flightlab-0004-production-jsbsim-aero-path.md)) carries the
aero document's compiled tables. The automated gate is
`tests/test_flightlab_backend_identity_parity.py`. Deliberately *not*
aligned: the OpenTsiolkovsky `synthetic_two_stage` demonstrator and the
FL-402 `spike_upper` stand-in.

Recovery stays the 3-DOF Earth-fixed point-mass chain (J2, rotation,
atmosphere, engines, event detection; Monte Carlo re-plans per dispersed
sample) regardless of the ascent backend.

<span id="native-aero-model-and-the-reference-plant-f-02"></span>
<span id="tvc-actuator-and-attitude-tracking-f-04"></span>
<span id="6-dof-vs-3-dof-gravity-turn-ascent-f-05"></span>
<span id="two-stage-separation-and-upper-stage-ignition-f-06"></span>
<span id="mission-outputs-selectable-backends-reports-and-mission-control-f-07"></span>

## 6-DOF studies

The native plant uses illustrative Mach/angle coefficient tables, US76,
Earth rotation, J2, thrust and a lag- and rate-limited gimbal. Its basic
tracker does not control roll or represent a complete autopilot. In the
[gravity-turn study](benchmarks/fl605_gravity_turn_ascent.json), the maximum
6-DOF minus 3-DOF differences were 2.13 km in altitude, 2.08 m/s in speed
and 0.47° in flight-path angle for that scenario.

The two-stage study applies equal and opposite separation impulses, then
propagates each stage independently. Compare its native point-mass and
rigid-body backends with:

```bash
orbitopt flight mission --backend BOTH -o runs/f07-mission
```

Outputs include FlightResult runs, handoffs, verification reports, scenes and
a comparison report. In the [recorded comparison](benchmarks/fl607_mission_outputs.json),
the upper stages differed by 235 km at the end: the 6-DOF upper stage was
inertially pointed, while the 3-DOF model followed its prescribed program.
These figures are scenario results, not general accuracy bounds.

## Hard gates

Runs record frames, epochs, units and provenance. Orbit requires the declared
velocity, perigee and inclination checks; landing requires all declared
position, speed and load limits. Verdicts are reproducible through the CLI,
and failures include reason codes. A pass applies only to the stated models
and limits.

## What is not modelled

The default ascent has no 6-DOF attitude dynamics. The models omit CFD-grade
aerodynamics, structural loads, propellant slosh, stage collisions, wind
fields and range safety. Recovery samples can include one steady wind vector,
but no turbulence; guidance re-planning is not an onboard computer simulation.

## Vehicle data, provenance and uncertainty

The [vehicle catalog](../src/orbitopt/data/flightlab/vehicles/) includes an
**illustrative** synthetic vehicle and a Falcon-9-class **public estimate**,
not official SpaceX performance data. Numeric fields record source, date and
provenance; uncertainty bands can supply Monte Carlo defaults. The Flight tab
shows the vehicle's provenance.

## Recovery uncertainty (R-04)

Run recovery dispersions with a target and vehicle:

```bash
orbitopt flight montecarlo src/orbitopt/data/flightlab/scenarios/leo_two_stage.yaml \
  --vehicle src/orbitopt/data/flightlab/vehicles/synthetic_two_stage.yaml \
  -n 32 --seed 7 --capture-tracks 240 -o montecarlo.json
```

`--uncertainty unc.yaml` configures wind, density, thrust, Isp, drag and
separation-state distributions individually. For example:

```yaml
uncertainty:
  thrust: {enabled: true, distribution: uniform_pct, half_width_pct: 5.0}
  wind: {enabled: true, distribution: normal, sigma_east_m_s: 4.0,
         sigma_north_m_s: 3.0, sigma_up_m_s: 0.5}
```

The report records the distributions. Wind enters the acceptance re-flight,
not the boostback optimizer rollout. Sampling is deterministic for a fixed
seed and independent of `--jobs`.

## Recovery Monte Carlo statistics (R-06)

`montecarlo.json` reports failure classes, a Wilson success-rate interval and
per-source sensitivity. Joint-dispersion associations are not isolated
causal effects. `--solo-sensitivity` re-flies each active source alone and
costs one additional sample pass per source. `--capture-tracks N` records up
to N site-ENU points per sample for the envelope plots without changing draws.

## Nominal / dispersed comparison (R-07)

In the recorded 32-sample comparison (seed 1, identical dispersions and
referee), the baseline landed 14 and the newer objective landed 17. The
latter's Wilson interval was [0.36, 0.69]; position failures rose from 4 to 7.
See [ADR 0003](adr/flightlab-0003-landing-aware-boostback-objective.md) and
[the recorded results](benchmarks/r07_nominal_dispersed.json) for the setup
and failure breakdown.

For implementation and reproducibility details, see [JSBSim findings](flightlab-jsbsim-spike-findings.md),
[regression CI](flightlab-regression-ci.md) and
[landing guidance](adr/flightlab-0002-closed-loop-landing-guidance.md).
