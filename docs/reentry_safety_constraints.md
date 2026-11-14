# Re-entry safety constraints — assessment and design (task #163)

**Status: proposal.** The lunar free-return targeter checks entry geometry; it
does not assess heating, loads, skip-out or survivability. The atmospheric
assessment described below is not implemented for this targeter.

## 1. What the code guarantees today

`target_free_return_entry_corridor()` and the export-side recheck propagate in
vacuum under Earth, Moon and Sun point-mass gravity. At the downward crossing
of a spherical entry interface (Earth radius 6,378 km plus 121.92 km), they
check the **inertial** flight-path angle against [−7°, −5°] and report the
entry speed. Passing this check establishes only the stated geometry.

### Why the geometric corridor is not enough

Heating and aerodynamic loads depend on atmosphere-relative velocity,
`v_rel = v_inertial − ω × r`, and on the vehicle and guidance. Earth rotation
can shift the relative angle by tenths of a degree; spherical and WGS-84
altitudes can differ by about 21 km near the poles. Heat load and skip-out
require the atmospheric trajectory, not just its initial angle.

## 2. Constraints to include

A proposed 3-DOF atmospheric pass would report these quantities. Limits must
be supplied for the vehicle; a missing limit means **not assessed**.

| Constraint | Metric and proposed model |
|---|---|
| Peak heat flux | Sutton–Graves convective `q̇ = k √(ρ/R_n) V³`, with `k = 1.7415 × 10⁻⁴` in SI units; report the low-fidelity radiative estimate separately. |
| Heat load | Integrate heat flux over the pass. |
| Peak sensed load | `max √(L² + D²) / (m g₀)`, excluding gravity. |
| Skip-out | Detect an upward entry-interface crossing and classify the exit energy. |
| Dynamic pressure | Optional `max ½ρV²`. |

Classify an exit as `skip_bounded` when its Earth-centered **inertial**
specific energy `ε = |v_I|²/2 − μ/|r|` is negative, or `skip_escape` when it
is non-negative. A bounded skip's subsequent guided entry remains unassessed.
If no upward crossing occurs before the terminal condition, classify the
modeled pass as `captured`.

Use `v_rel` for aerodynamics, never for the inertial energy test. Convert an
Earth-fixed exit state to an inertial frame with the full SPICE `sxform` 6×6
state transformation; a position-only rotation omits the transport velocity.
Record the frame in the report.

## 3. Corridor bounds instead of a fixed band (recommended)

At fixed entry position, epoch, speed and heading, proposed root searches
would vary the flight-path angle to find the shallowest captured angle with
full lift-down and the steepest angle within heat and load limits with full
lift-up. The resulting vehicle-specific bounds are necessary conditions under
those constant-bank assumptions, not proof of a feasible guided entry.

## 4. Required inputs and model fidelity

### Inputs

The assessment needs the entry state and epoch; mass, area, drag coefficient,
lift-to-drag ratio, bank angle and nose radius; an atmosphere and Earth
rotation; and user-supplied heat, load and pressure limits. Example capsule
parameters must be marked illustrative.

### Fidelity tiers

| Tier | Model and purpose |
|---|---|
| T0: current | Vacuum entry angle for geometry screening only. |
| T1: analytic | Ballistic/lifting approximations and an exponential atmosphere for sanity checks. |
| T2: proposed | 3-DOF, rotating Earth, US76 atmosphere, constant aero and bank, with heating correlations. |
| T3: future | Guided entry, aerodynamic databases and atmospheric dispersions. |

Accuracy estimates for these correlations are planning assumptions, not
validated results from this repository. T2 would stop at a low-altitude or
speed threshold, or an upward entry-interface crossing.

## 5. Interface with `target_free_return_entry_corridor()`

Keep the existing vacuum targeter unchanged by default. A proposed
`verify/entry_assessment.py` would assess its entry state separately, then
optionally calculate vehicle-specific corridor bounds. Its result would carry
relative speed and angle, heat flux and load, peak g and pressure, skip
classification, assumptions and a per-limit pass/fail/unknown value.

The handoff state is Earth-centered ECLIPJ2000. Its absolute epoch is
`reference_epoch_ephemeris_seconds + initial_epoch + entry_epoch` in TDB
seconds past J2000. Start from the existing spherical-interface state and
report its geodetic altitude; do not silently re-target it. Any new report
fields and schema changes are follow-up work.

## 6. Verification plan for the future implementation

- Compare ballistic cases with Allen–Eggers and check expected scaling with
  ballistic coefficient and nose radius.
- Test angle and bank-response trends, and verify `steepest < shallowest`.
- Test bounded and escaping skips, including an Earth-fixed-to-inertial
  `sxform` round trip at near-escape speed.
- Check an illustrative lunar-return case against physical ranges; do not
  interpret agreement with analytic approximations as flight validation.

## 7. Scope: what this does **not** do

The proposal excludes guided or 6-DOF entry, atmospheric dispersions, detailed
aerodynamic and thermal-protection models, and descent below roughly 30 km.
It does not model parachutes, splashdown or crew physiology. Reports must say
"within modeled limits under the stated assumptions", not "survivable".

## 8. Suggested follow-up tasks

Implement the opt-in assessment and tests, add report/schema fields, then add
vehicle-specific corridor bounds. Any change to the existing interface or
Earth-shape convention requires separate review.
