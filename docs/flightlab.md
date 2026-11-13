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
| Recovery Monte Carlo | Plan-level dispersed samples; guidance re-plans with each sample's vehicle parameters. Vehicle uncertainty bands supply defaults. |
| 6-DOF ascent (JSBSim) | Opt-in formal backend (`backend="6DOF"`, `pip install orbitopt[sixdof]`): config-driven vehicle/ICs/dt through `orbitopt.flightlab.adapters.sixdof`; mass, CG and inertia tensor follow the propellant load and are checked against the scenario's declared mass model (`mass_model`, F-03). Zero-aero illustrative vehicles so far. Not the default propagator, no aero yet, and not part of the optimization loop. See [ADR 0001](adr/flightlab-0001-jsbsim-6dof.md). |
| 6-DOF reference plant (native) | JSBSim-free RK4 plant in `orbitopt.flightlab.sixdof.rigidbody` (gravity/J2/rotation, US76, throttleable thrust, attitude) flying the coefficient model of `orbitopt.flightlab.sixdof.aero`; reports the q-bar history and max-Q. Reference plant for aero acceptance, not a JSBSim replacement and not a `flight simulate` backend. |
| TVC + attitude tracking (native, F-04) | `orbitopt.flightlab.sixdof.tvc` closes the reference plant's loop: first-order-lag / slew-limited / saturating actuator, a ZOH PID tracker on the FL-403 body-axis attitude error, per-tick history and deflection-margin diagnostics. Basic tracker by design, not GNC. |

## Native aero model and the reference plant (F-02)

The JSBSim backend flies the aerodynamics written inside its aircraft XML, and
the `jsbsim==1.3.3` bindings expose no per-step force/moment injection hook —
so an in-house coefficient model cannot be pushed into it. F-02 therefore
ships the mirror image: an aero model authored here, flown by a native 6-DOF
plant here (`orbitopt.flightlab.sixdof.rigidbody`).

- **Coefficient model** (`sixdof/aero.py`): tables over Mach / angle of attack
  / sideslip per coefficient (`ca, cy, cn, cl, cm, cn_yaw`), N-linear
  interpolation with a per-table extrapolation policy (`clamp`, `linear`,
  `error`), body-axis force/moment assembly with linear rate damping, and max-Q
  extraction (peak sample, parabolic time refinement). Documents validate
  against the packaged `flight-aero-1.0` JSON schema on load.
- **Axisymmetric form**: `kind: axisymmetric` documents carry just
  `ca(M)`, a curve slope, a static margin and damping, and are promoted to full
  tables at load. The packaged [spike-stage model](../src/orbitopt/data/flightlab/aero/spike_stage.yaml)
  (geometry from the task #206 spike, cd(M) from the f9-class catalog table,
  coefficients **illustrative**) is what the acceptance tests fly.
- **Reference plant** (`sixdof/rigidbody.py`): same physics family as the
  recovery propagator plus attitude — ECEF RK4 on `(r, v, q, omega, m)`, US76
  (or an injectable `ExponentialAtmosphere` for analytic baselines),
  ambient-corrected throttleable thrust with a gimballed moment arm, propellant
  bookkeeping floored at dry mass, ground-stop with an interpolated crossing
  time. `to_body_track()` emits contract tracks directly.
- **Max-Q**: reported per run with time, altitude and Mach;
  `tests/test_flightlab_sixdof_rigidbody.py` checks the recorded history
  against the closed form of the exponential atmosphere at the run's own mean
  acceleration (constant-acceleration idealisation, so a ~35 % band against a
  burn-down run).

The JSBSim backend stays the formal 6-DOF engine; this plant is where a real
coefficient table from F-05 lands first.

## TVC actuator and attitude tracking (F-04)

`sixdof/tvc.py` closes the reference plant's attitude loop with deliberately
basic parts:

- **Actuator** (`TvcActuator`): first-order lag (`tau_lag_s`), slew-rate cap
  (`max_slew_rate_rad_s`, exact on ramps) and travel saturation
  (`limit_rad`, default 5°); commands are clamped *before* the lag, so the
  state never leaves the travel. Substepped at ≤ τ/4 for stiff commands.
- **Tracker** (`AttitudeTvcLaw`): digital ZOH PID on the body-axis attitude
  error `so3_log(R_cmdᵀ R_act)` resolved into the body frame, with the
  program's kinematic body rate as feedforward — the FL-403 convention, so
  positive pitch error means "nose up" and near vertical the azimuth DOF
  rides the roll axis. The integrator is clamped at `i_max_alpha/ki`; after
  engine cut-off (`mass ≤ dry`) the command neutralises and the integrator
  freezes. Deflection authority is checked against the plant's own engine
  gimbal clamp at construction.
- **Actuator state is what the plant sees**: the tracker feeds the *post*
  lag/rate/saturation deflection to `RigidBodyVehicle`, so saturation and
  slew limiting are real in the loop, not cosmetic.
- **Diagnostics**: per-tick history (`law.history()`), `law.diagnostics`
  (command/applied peaks, saturated and slew-limited seconds, deflection
  margin) and `attitude_metrics` (error max/RMS, α and rate peaks) are
  recorded on every `run_closed_loop_ascent()`, and `summary()` prints the
  acceptance's margin/max-deflection line.
- **Known scope limit**: two gimbals cannot steer the unactuated roll /
  azimuth-about-vertical DOF. A hard pitch-over through vertical therefore
  leaves a bounded azimuth spin (the aero acceptance runs stay under a
  degree); the thrust line and pitch rate converge onto the program
  regardless — `test_aggressive_initial_offset_proves_saturation_in_the_loop`
  pins this behaviour down. Roll control, steering-to-vertical and a proper
  autopilot are out of scope for F-04 by design.

## Hard gates

Runs record frames, epochs, units and provenance. Orbit requires the declared
velocity, perigee and inclination checks; landing requires all declared
position, speed and load limits. Verdicts are reproducible through the CLI,
and thresholds are declared before the run. Failures include reason codes.

## What is not modelled

The default model does not simulate 6-DOF attitude dynamics, CFD-grade
aerodynamics, structural loads, propellant slosh, stage collisions, wind or
range safety. Recovery guidance re-plans at the model level; it is not an
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
[JSBSim findings](flightlab-jsbsim-spike-findings.md) and
[ADR 0002](adr/flightlab-0002-closed-loop-landing-guidance.md) for engineering
and verification details.
