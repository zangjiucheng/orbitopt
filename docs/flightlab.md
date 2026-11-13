# Flight Lab

Flight Lab designs and simulates rocket ascent and booster recovery. In
Mission Control, switch the sidebar to **Flight Lab**: Designer sets up a
run, Replay displays its trajectory and checks, and Compare puts two runs side
by side. Recent runs and *Open run…* load saved run directories.

## Quick start

From a configured [compute environment](quickstart.md#installation), install
the pinned OpenTsiolkovsky binary and validate the included target:

```bash
python scripts/fetch_opentsiolkovsky.py
orbitopt flight validate src/orbitopt/data/flightlab/scenarios/leo_two_stage.yaml
```

Run the example from the repository root:

```bash
orbitopt flight simulate src/orbitopt/data/flightlab/scenarios/leo_two_stage.yaml \
  --vehicle src/orbitopt/data/flightlab/vehicles/synthetic_two_stage.yaml -o run/

orbitopt flight to-scene run/ -o scene.json
orbitopt view scene.json
```

For a recovery Monte Carlo, pass the **target and vehicle**, not the run
directory:

```bash
orbitopt flight montecarlo src/orbitopt/data/flightlab/scenarios/leo_two_stage.yaml \
  --vehicle src/orbitopt/data/flightlab/vehicles/synthetic_two_stage.yaml \
  -n 32 --seed 7 -o montecarlo.json
```

`orbitopt flight optimize --help` describes ascent optimization. Each command
also has its own `--help`.

## Fidelity levels

| Model | Scope |
|---|---|
| Ascent | OpenTsiolkovsky 3-DOF point mass with a gravity turn or attitude program and a Cd(Mach) model. Displayed attitude is kinematic. |
| Booster recovery | 3-DOF Earth-fixed point mass with J2, rotation, atmosphere, relightable engines and event detection. |
| Recovery Monte Carlo | Plan-level dispersed samples; guidance re-plans with each sample's vehicle parameters. Vehicle uncertainty bands supply defaults; R-04 sources (wind / density / thrust / Isp / drag / initial state) are config-declared and individually switchable. |
| JSBSim spike | Optional 6-DOF tracking cross-check; it is not the default propagator or part of the optimization loop. See [ADR 0001](adr/flightlab-0001-jsbsim-6dof.md). |

## Hard gates

Runs record frames, epochs, units and provenance. Orbit requires the declared
velocity, perigee and inclination checks; landing requires all declared
position, speed and load limits. Verdicts are reproducible through the CLI,
and thresholds are declared before the run. Failures include reason codes.

## What is not modelled

The default model does not simulate 6-DOF attitude dynamics, CFD-grade
aerodynamics, structural loads, propellant slosh, stage collisions, wind
fields or range safety. The recovery Monte Carlo samples a *steady* ENU wind
per sample (R-04, see below), but no wind field or turbulence is modelled.
Recovery guidance re-plans at the model level; it is not an
onboard computer simulation. A passing verdict is conditional on these models
and the declared limits.

## Vehicle data, provenance and uncertainty

The packaged [vehicle catalog](../src/orbitopt/data/flightlab/vehicles/)
contains a synthetic two-stage vehicle tagged **illustrative** and a
Falcon-9-class vehicle tagged **public estimate**. The latter is based on a
community reconstruction and public sources, not official SpaceX performance
data. The Designer displays the provenance tag.

Numeric fields carry a source, date and provenance; uncertainty can be supplied
as a percentage. The Monte Carlo uses vehicle thrust, Isp and Cd uncertainty
bands by default and records its assumptions. The sidecar `data_provenance`
uses the most specific matching path; inline provenance is permitted under
`booster`, while OpenTsiolkovsky stage data remains plain.

See [regression CI](flightlab-regression-ci.md), the
[JSBSim findings](flightlab-jsbsim-spike-findings.md),
[ADR 0002](adr/flightlab-0002-closed-loop-landing-guidance.md) and
[ADR 0003](adr/flightlab-0003-landing-aware-boostback-objective.md) for
engineering and verification details.

## Recovery uncertainty (R-04)

`orbitopt flight montecarlo --uncertainty unc.yaml` declares the perturbation
model per source instead of inheriting the defaults — wind, atmospheric
density, thrust, Isp, drag and initial (separation) state:

```yaml
uncertainty:
  thrust:        {enabled: true, distribution: uniform_pct, half_width_pct: 5.0}
  isp:           {enabled: true, distribution: normal_pct, sigma_pct: 1.5}
  density:       {enabled: true, distribution: uniform_pct, half_width_pct: 8.0}
  drag:          {enabled: false}
  wind:          {enabled: true, distribution: normal, sigma_east_m_s: 4.0,
                  sigma_north_m_s: 3.0, sigma_up_m_s: 0.5}
  initial_state: {enabled: true, distribution: normal, sigma_altitude_m: 300.0,
                  sigma_downrange_m: 800.0, sigma_crossrange_m: 800.0,
                  sigma_speed_m_s: 5.0, sigma_heading_deg: 0.2}
```

Every source has its own `enabled` switch; the declared distributions and
parameters are recorded verbatim in the report's `uncertainty` section
(`montecarlo.json` schema 1.1; `dispersions` stays as the flat legacy view).
The wind is one steady ENU vector per sample — it enters the acceptance
re-integration (propagator and landing burn) but not the FL-302 optimiser
rollout, the same deliberate plan/reality mismatch as the density bias.
Sampling stays deterministic: one `[seed, idx]` RNG stream per sample and a
fixed draw order, so a fixed seed reproduces the report byte-for-byte
regardless of `--jobs`, and pre-R-04 fixed-seed dispersions runs (the plain
`Dispersions` path, wind inactive) remain reproducible.
