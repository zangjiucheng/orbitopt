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
  -n 32 --seed 7 --capture-tracks 240 -o montecarlo.json
```

`orbitopt flight optimize --help` describes ascent optimization. Each command
also has its own `--help`.

## Fidelity levels

| Model | Scope |
|---|---|
| Ascent | OpenTsiolkovsky 3-DOF point mass with a gravity turn or attitude program and a Cd(Mach) model. Displayed attitude is kinematic. |
| Booster recovery | 3-DOF Earth-fixed point mass with J2, rotation, atmosphere, relightable engines and event detection. |
| Recovery Monte Carlo | Plan-level dispersed samples; guidance re-plans with each sample's vehicle parameters. Vehicle uncertainty bands supply defaults; R-04 sources (wind / density / thrust / Isp / drag / initial state) are config-declared and individually switchable. R-06 reports failure classes, a success-rate confidence interval and per-source sensitivity. R-07 optionally records each sample's booster track (`--capture-tracks`) for the dispersion-envelope figures. |
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
coefficient table from F-05 lands first (the F-05 ascent study below flies
it with the packaged `spike_stage` tables).

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

## 6-DOF vs. 3-DOF gravity-turn ascent (F-05)

`sixdof/gravity_turn.py` flies the F-02 plant + F-04 tracker against a
point-mass 3-DOF reference built on the *same* vehicle object, the same
ambient-corrected engine law, the same physics switches (US76 + J2 +
rotation) and the same RK4 core — the class of model the OpenTsiolkovsky
adapter drives (point mass, kinematic program attitude, q̄·S·CA/CN aero),
flown natively so the pair differs *only* in the 6-DOF content the study
sets out to quantify; the pinned OT binary itself stays covered by
`tests/test_flightlab_opentsiolkovsky.py`. One scenario — the spike-stage
booster, packaged `spike_stage` tables, vertical rise to 10 s then a linear
pitch-over to 18° by 120 s, full throttle to the 99.85 s burn-out, coast to
180 s — is flown five times:

| rung | physics added over the rung below |
|---|---|
| `3dof_vacuum` | shared-physics baseline (no aero) |
| `3dof_aero` | drag/lift on the point mass — the OT-equivalent ascent |
| `6dof_ideal` | rigid-body inertia + discrete tracker, ideal actuator |
| `6dof_lag` | the real F-04 actuator (τ = 0.1 s, 5.73°/s) |
| `6dof_full` | body-oriented aero closed onto the lagged loop |

**Headline (6-DOF minus 3-DOF, max |difference|):** altitude **2.13 km**,
Earth-relative speed **2.08 m/s**, flight-path angle **0.47°**, mass
**8 g**. The acceptance's four difference curves and these metrics are
committed as `docs/benchmarks/fl605_gravity_turn_ascent.json`
(`PYTHONPATH=src python benchmarks/bench_fl605_gravity_turn.py`), and the
regression gates live in `tests/test_flightlab_sixdof_gravity_turn.py`.

**Where the differences come from** — four attribution terms, each flipping
exactly one physics switch, and the headline difference telescopes over them
*sample-wise exactly*:

- **Aerodynamics dominates everything**: drag costs 34.4 km of the 402 km
  ballistic altitude on both models (`aero_point_mass` 34 422 m vs.
  `aero_rigid_body` 34 453 m) — so in the 3-DOF/6-DOF *comparison* it almost
  cancels; the ~30 m residual between the two is the difference between a
  point-mass q̄S·C law and the same tables on an oriented body with pitch
  rate (weathercock damping `cm_q`, and AoA realised from body attitude).
- **Attitude dynamics is what the headline really measures**: rigid-body
  inertia plus the discrete tracker lag the thrust vector behind the 0.65°/s
  pitch program and the gravity-loss path bends — 2.1 km of altitude, 8 m/s
  of speed, 0.57° of FPA. The tracker never saturates or slew-limits (5°
  deflection margin ≥ 98%), but the 3-axis attitude error reaches ~10°
  through the high-q̄ pitch-over (trim AoA plus F-04's un-actuated
  roll-about-flight-axis DOF) and keeps growing after MECO — by design the
  tracker neutralises at engine cutoff, so the attitude freezes while the
  velocity vector turns on (62° "AoA" at t = 180 s, at q̄ ≈ 0: unloaded,
  and the 3-DOF side sees the same program-vs-velocity angle).
- **Control lag is second-order on this program**: the tracker treats the
  first-order lag as plant dynamics it was tuned to fight, so the actuator
  rung moves the trajectory by < 5 cm and the FPA by ~0.001° — the lag bites
  on *fast* commands, which F-04's slew/saturation tests already pin down.
- **Mass is a shared-law consistency check, not a difference**: the same
  engine law seen through slightly different altitudes differs by 8 g —
  the test gate at 5 kg exists to catch anyone ever breaking that.

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
(`montecarlo.json` schema 1.3 — R-06 adds the `statistics` section described
below and R-07 the optional `track_capture` section described below;
`dispersions` stays as the flat legacy view).
The wind is one steady ENU vector per sample — it enters the acceptance
re-integration (propagator and landing burn) but not the FL-302 optimiser
rollout, the same deliberate plan/reality mismatch as the density bias.
Sampling stays deterministic: one `[seed, idx]` RNG stream per sample and a
fixed draw order, so a fixed seed reproduces the report byte-for-byte
regardless of `--jobs`, and pre-R-04 fixed-seed dispersions runs (the plain
`Dispersions` path, wind inactive) remain reproducible.

## Recovery Monte Carlo statistics (R-06)

Every `montecarlo.json` now carries a `statistics` section
(schema 1.2, still current in 1.3) that reads the sample population instead
of only counting it:

* **failure classes** — every non-landed sample is attributed to exactly one
  family: `planning_infeasible` (no feasible boostback/entry/landing plan),
  `execution_failure` (sample error, no touchdown, entry limits, handoff
  break), `landing_accuracy` (touched down outside the limits) or
  `fuel_insufficient` (a propellant shortfall, wherever reported). An
  unrecognised reason code is reported as `unclassified` rather than dropped,
  so the classes always sum to the failure count;
* **confidence interval** — a Wilson score interval on the success rate
  (`statistics.success_rate_ci`), reported with its point estimate and Wald
  standard error. Samples are independent by construction, so this is the
  honest statement of what `n` samples support: 6/8 landed is 0.75 with a
  95 % interval of [0.41, 0.93], not "75 %";
* **sensitivity** — per source, the association between that sample's own
  perturbation (in sigma-equivalents of its declared distribution) and its
  failure: mean |z| among landed vs failed samples, the failure rate above and
  below 1 sigma, a Spearman rank correlation, and a `ranking`. This is read
  under joint dispersion, so a source can inherit a neighbour's effect.

`--solo-sensitivity` adds the marginal reading: every active source re-flown
alone at the same seed and `n` (`statistics.solo_sensitivity`, one delta in
success rate per source). That costs `1 + n_active` sample passes and is what
isolates one source from the ones dispersed with it. All of it is a pure
function of the flown samples — no RNG — so the same seed still reproduces the
whole report byte-for-byte.

## Nominal / dispersed comparison (R-07)

R-06 counts and classifies the failures; R-07 shows *where and how* the
population lands, against the FL-305 baseline, under identical conditions.
Three pieces, all in `orbitopt.flightlab.recovery.envelope`:

* **nominal verification report** — `nominal_recovery()` flies the one
  unperturbed recovery (the Monte Carlo's own warm start by default, so the
  nominal and the dispersed population share one setup), and
  `nominal_report()` / `write_nominal_report()` render the full verdict:
  every gate and check re-assessed through `checks.evaluate` with its
  governing reason title, the FL-302 corridor assessment, the landing
  summary (with the R-05 `actuator_margin`), the R-02 handoff block and the
  flown booster track in the capture format below. `--reports` JSONs are
  byte-stable (sorted keys).
* **track capture** — `flight montecarlo --capture-tracks N` records each
  landed *and* failed sample's booster track, projected into site-ENU
  (`t_s, east_m, north_m, up_m`, up from the site altitude) and uniformly
  down-sampled to ≤ N rows, first and last row always kept. It rides on the
  per-sample records after the flight, so it never touches the RNG or the
  statistics: the same seed reports identical numbers with and without
  capture (`montecarlo.json` schema 1.3, `track_capture` section records the
  budget and the site used).
* **envelope figures** — `envelope_dataset()` + `plot_envelope()` render one
  run: every captured ground track in the site-ENU plane (landed in blue,
  failures coloured by R-06 class and annotated with `idx REASON`, the max
  touchdown radius as a ring) and the altitude-vs-range-to-site min/max
  envelope with the nominal profile overlaid. `plot_comparison()` overlays
  two runs' nominal tracks, touchdown scatters with 1/3-sigma ellipses, and
  each run's Wilson interval.

The evidence run is `benchmarks/bench_r07_nominal_dispersed.py`
(evidence: `docs/benchmarks/r07_nominal_dispersed.json`, nominal reports
`docs/benchmarks/r07_nominal_{baseline_fl305,r07_stack}.json`, figures
`docs/assets/r07_envelope_{baseline_fl305,r07_stack}.png` and
`docs/assets/r07_comparison.png`). Fairness: same ascent fixture, seed 1,
n 32, dispersions, FL-302 warm start and referee for both configurations —
the guidance/objective differs by exactly what epic #217 changed, the
baseline pinning the R-01 objective terms off as the recorded FL-305
evidence does. Under those conditions the baseline re-run reproduces the
recorded FL-305 evidence exactly (14/32 = 0.4375, identical reason
histogram — the fairness check passing by construction), and the current
stack lands 17/32 = 0.531 (Wilson [0.36, 0.69]): `INSUFFICIENT_PROPELLANT_
RESERVE` failures drop 8 → 2 as the landing-aware objective stops burning
boostback propellant the landing needs, at the price of three more
out-of-tolerance touchdowns (4 → 7) — the envelope figures show those as
`LANDING_POSITION_OUT_OF_TOLERANCE` annotations pushed past the tolerance
ring. Both configurations' nominal flights land at ~0.55 m with identical
touchdown states; the difference is entirely in the tail.
