# Re-entry safety constraints — assessment and design (task #163)

> **摘要（中文）**：当前 `target_free_return_entry_corridor()` 只保证"真空传播下、入口界面处的惯性航迹角落在 (-7°, -5°) 几何走廊内"。这**不等于**可存活再入。本文档列出应纳入的约束（峰值热流、总热载、g 载荷、skip-out 风险）、所需输入数据与模型精度分级、与现有定靶函数的衔接方式，并明确**不做**的部分。任何输出都不得宣称"可存活再入（survivable entry）"，只能表述为"在所列假设与模型下，指标是否在所设限值之内"。

Status: design / research only. No code in this task changes behaviour.

## 1. What the code guarantees today

`orbitopt.verify.differential_correction.target_free_return_entry_corridor()`
(and its export-side re-check `viz.mission_timeline._verify_entry_corridor()`)
establish exactly this, and nothing more:

| Item | Current model |
|---|---|
| Dynamics | TudatPy, point-mass Earth/Moon/Sun, **vacuum** (no atmosphere, no aero forces) |
| Earth shape | sphere, `EARTH_RADIUS_KM = 6378.0` |
| Entry interface (EI) | `|r| = 6378 + 121.92 km` (400,000 ft), root-found by `find_altitude_crossing` |
| Checked quantity | **inertial** flight-path angle at the EI crossing (`flight_path_angle_deg`) |
| Pass criterion | `-7.0 <= fpa <= -5.0` deg (`LUNAR_RETURN_ENTRY_CORRIDOR_DEG`) |
| Also reported | inertial entry speed (`entry_speed_m_s`, ~11 km/s for lunar return) |

`in_corridor == True` therefore means "the vacuum trajectory pierces a
spherical shell at an angle inside a fixed band borrowed from Orion-class
lunar returns". It says nothing about heating, loads, or whether the vehicle
is captured by the atmosphere.

### Why the geometric corridor is not enough

* **The band presupposes a lifting vehicle and guidance.** A ballistic
  (L/D = 0) entry at 11 km/s and -6 deg gives, by Allen–Eggers,
  `a_max ≈ V_E² sin|γ| / (2 e H) ≈ (11 000²)(0.105) / (2·2.718·7 200) ≈ 320 m/s² ≈ 33 g`,
  independent of ballistic coefficient. The same angle is fine for a capsule
  with L/D ≈ 0.3 flying a modulated bank profile. The corridor is a property
  of *vehicle + guidance*, not of the trajectory alone.
* **The angle is inertial; the atmosphere rotates.** Aerodynamic loads and
  heating depend on the Earth-relative velocity `v_rel = v − ω⊕ × r`. Earth
  rotation (up to ~0.46 km/s at the equator) shifts the relative FPA by
  roughly a few tenths of a degree depending on heading — not negligible
  against a 2-deg-wide band.
* **Spherical Earth.** Geodetic altitude on WGS-84 differs from
  `|r| − 6378 km` by up to ~21 km near the poles, so the "121.92 km EI" is
  not at a consistent atmospheric density.
* **Heat load and skip-out are integral/trajectory properties.** They need
  the atmospheric pass itself to be flown, not just its initial condition.

## 2. Constraints to include

All evaluated on an atmospheric 3-DOF trajectory flown from the EI state
(Section 4), using Earth-relative velocity `V` and atmospheric density `ρ(h)`.

| # | Constraint | Metric | Model (first implementation) | Typical limit (configurable, *not* a claim) |
|---|---|---|---|---|
| C1 | Peak stagnation-point heat flux | `max q̇` [W/cm²] | Convective: Sutton–Graves `q̇_c = k √(ρ/R_n) V³`, `k = 1.7415e-4` (SI, Earth air). Radiative (relevant above ~10 km/s): Tauber–Sutton correlation, reported separately and flagged as low-fidelity | vehicle/TPS-specific; user input, no default pass/fail |
| C2 | Total heat load | `Q = ∫ q̇ dt` [J/cm²] | Trapezoid integral of C1 over the pass from EI to end condition | TPS-specific; user input |
| C3 | Peak sensed deceleration | `max |a_aero| / g0` [g], `a_aero = √(L²+D²)/m` | From the 3-DOF aero acceleration (not total acceleration incl. gravity) | e.g. crew ~ 8–10 g short duration; user input |
| C4 | Skip-out risk | (a) any exit back above EI after first atmospheric pass; (b) apogee / geocentric inertial energy at exit; (c) overshoot margin (Section 3) | Event detection on the 3-DOF trajectory with `find_altitude_crossing` semantics (upward crossing) | classification, see below |
| C5 *(optional)* | Peak dynamic pressure | `max ½ρV²` [kPa] | Free from the same trajectory | user input |

Skip-out (C4) classification:

* `captured` — altitude never returns above EI before the end condition
  (e.g. h < 30 km or V_rel < 1 km/s, i.e. parachute regime is *not* modelled).
* `skip_bounded` — exits above EI with negative geocentric **inertial**
  specific energy: a second entry will occur (intentional for Orion-style
  skip entry, but it is a separate, guided problem and is reported as "not
  assessed").
* `skip_escape` — exits with non-negative inertial energy: hard failure.

The energy test uses the geocentric inertial state at the exit crossing,
`ε = ½|v_I|² − μ⊕/|r|`, never the atmosphere-relative speed. Two-body
boundedness is a property of the inertial orbit; `v_rel = v_I − ω⊕ × r` is
smaller than `|v_I|` by up to ~0.47 km/s (equatorial, prograde exit at
EI), so an energy computed from `v_rel` can call an escaping trajectory
bounded: at the EI radius (~6500 km) escape speed is ≈ 11.07 km/s, and an
exit at exactly that inertial speed can have `|v_rel|` ≈ 10.6 km/s.
`v_rel` is for aerodynamics and heating only.

NAIF inertial-state requirement: `r`, `v_I` must be Earth-centred
coordinates in an inertial frame (ECLIPJ2000 or J2000, i.e. the propagation
frame). If the 3-DOF integration runs in the Earth-fixed frame
(`IAU_EARTH`/`ITRF93`), convert the exit state back with the full 6×6 state
transformation (`sxform`, which adds the `ω⊕ × r` transport term), not the
3×3 position rotation (`pxform`) applied to the velocity; a rotated `v_rel`
is still a relative velocity. The report records the frame used for `ε`.

## 3. Corridor bounds instead of a fixed band (recommended)

The physically meaningful corridor for a given vehicle is the classic pair of
boundaries, computed by flying constant-bank 3-DOF entries:

* **Overshoot (shallow) boundary** — the shallowest EI flight-path angle
  that is still `captured` when flying **full lift-down** (bank 180°).
  Shallower ⇒ skip-out risk even with maximum control authority.
* **Undershoot (steep) boundary** — the steepest EI flight-path angle that
  keeps C1/C3 (and C2, if given) inside limits when flying **full lift-up**
  (bank 0°). Steeper ⇒ limits exceeded even with maximum control authority.

Each boundary is found by a 1-D root find (Brent) over the EI FPA at fixed
EI speed/heading/position, so it costs ~10–20 short (minutes-long) 3-DOF
propagations. The output is a vehicle-specific `(steepest, shallowest)`
tuple that can replace `LUNAR_RETURN_ENTRY_CORRIDOR_DEG` (still in the same
shape, so the targeter and export check need no structural change).

These are *necessary* conditions under constant-bank bounds, not proof that
a guided entry exists inside them.

## 4. Required inputs and model fidelity

### Inputs

| Input | Source | Required for |
|---|---|---|
| EI state (6,), epoch | `FreeReturnTargetingResult.entry_state`, `entry_epoch` + reference epoch | all |
| Vehicle mass `m`, reference area `A`, `C_D` → ballistic coefficient `β = m/(C_D A)` | user (`EntryVehicle` dataclass) | C1–C5 |
| Lift-to-drag ratio `L/D` (constant, trim) | user | C3, C4, Section 3 |
| Bank angle: constant value or {lift-up, lift-down} bounds | user / fixed for corridor bounds | C3, C4 |
| Nose radius `R_n` | user | C1, C2 |
| Limits (`q̇_max`, `Q_max`, `n_max`, `q_dyn_max`) | user, **no built-in "safe" defaults** | pass/fail flags |
| Atmosphere | US Standard Atmosphere 1976 (Tudat built-in, or tabulated) | all |
| Earth rotation | ω⊕ = 7.292115e-5 rad/s about the ECLIPJ2000→ITRF pole (via SPICE frame) | relative V/FPA |

A provided example vehicle (e.g. "Apollo-class: β ≈ 300–400 kg/m², L/D ≈ 0.3,
R_n ≈ 4.7 m") must be labelled illustrative, not a validated vehicle model.

### Fidelity tiers

| Tier | Model | Use | Expected accuracy (order of magnitude) |
|---|---|---|---|
| T0 (today) | vacuum, inertial FPA, fixed band | screening the free-return geometry | geometry only |
| T1 | analytic: Allen–Eggers ballistic, Chapman/Loh lifting approximations, exponential atmosphere (H ≈ 7.2 km) | sanity check / unit-test oracle for T2 | peak g ~10–20 %, heat-flux ~20–30 % |
| **T2 (proposed implementation)** | 3-DOF point mass, US76 atmosphere, rotating spherical Earth (optionally J2 + WGS-84 geodetic altitude), constant `C_D`, constant `L/D`, constant bank; Sutton–Graves + Tauber–Sutton | C1–C5 and corridor bounds | relative to T3 truth: peak g ~5–10 %, convective q̇ ~20–30 %, radiative q̇ factor-of-2, Q ~30 % |
| T3 (not done) | guided entry (bank modulation, Apollo/PredGuid/FNPEG), Mach-dependent aero database, NRLMSISE/GRAM dispersions, Monte Carlo | flight design | — |

The accuracy numbers are planning estimates for the stated correlations, not
validated by this repository; tests (Section 6) only check self-consistency
and agreement with T1 analytic limits.

Integration: the T2 trajectory runs from EI for at most ~20–30 min with an
adaptive integrator (Tudat `rkdp_87` with aerodynamic acceleration +
`constant_aerodynamic_coefficients`, or a standalone NumPy/SciPy `DOP853`
ODE for the planar/3-D equations of motion — the latter avoids Tudat
aero-guidance plumbing and is easier to unit-test; either is acceptable as
long as the report records which). Terminal events: h < 30 km, V_rel < 1 km/s,
or an upward EI crossing (skip).

## 5. Interface with `target_free_return_entry_corridor()`

Principle: **keep the targeter a vacuum-geometry targeter.** Its Newton
residual (vacuum return perigee) stays smooth and cheap; atmospheric
assessment is a separate, post-targeting verification stage. Nothing in
the joint Newton changes.

```text
target_free_return_entry_corridor(...)          # unchanged: vacuum, inertial FPA band
        │  FreeReturnTargetingResult.entry_state / entry_epoch
        ▼
assess_entry_constraints(entry_state, entry_epoch_abs, vehicle, limits,
                         bank_deg=..., atmosphere="us76", rotating_earth=True,
                         report_path=None) -> EntryAssessment
        │
        ▼
entry_corridor_bounds(entry_state, entry_epoch_abs, vehicle, limits)  # optional, Section 3
        -> (steepest_deg, shallowest_deg)   # same shape as LUNAR_RETURN_ENTRY_CORRIDOR_DEG
```

Proposed module: `src/orbitopt/verify/entry_assessment.py`.

```python
@dataclass
class EntryVehicle:
    mass_kg: float
    reference_area_m2: float
    drag_coefficient: float
    lift_to_drag: float = 0.0
    nose_radius_m: float = 1.0
    label: str = "user-supplied"

@dataclass
class EntryLimits:            # all optional; None = report value, no pass/fail
    peak_heat_flux_w_cm2: float | None = None
    total_heat_load_j_cm2: float | None = None
    peak_load_g: float | None = None
    peak_dynamic_pressure_kpa: float | None = None

@dataclass
class EntryAssessment:
    relative_entry_speed_m_s: float
    relative_entry_fpa_deg: float
    peak_convective_heat_flux_w_cm2: float
    peak_radiative_heat_flux_w_cm2: float | None   # flagged low-fidelity
    total_heat_load_j_cm2: float
    peak_load_g: float
    peak_dynamic_pressure_kpa: float
    skip_status: str            # "captured" | "skip_bounded" | "skip_escape"
    within_limits: dict          # {constraint: True/False/None}
    model: dict                  # tier, atmosphere, aero, heating correlations, integrator
    report: dict | None = None
```

Wiring (follow-up tasks, all opt-in):

1. `FreeReturnTargetingResult` gains `entry_assessment: EntryAssessment | None = None`;
   `target_free_return_entry_corridor(..., entry_vehicle=None, entry_limits=None)`
   calls `assess_entry_constraints` only when a vehicle is given. Default
   behaviour and existing tests are unchanged.
2. The verification report (`verify.report.build_report`) gets one residual
   entry per constraint, same shape as the existing
   `entry_flight_path_angle_in_corridor` residual
   (`name`, `reference`, `value`, `tolerance`, `within_tolerance`), with
   `within_tolerance: null` when no limit was supplied, plus a `model`
   block recording the tier and every assumption. The JSON schema
   `schemas/verification-report-1.0.json` must accept these.
3. `viz.mission_timeline` keeps refusing to export out-of-corridor
   trajectories; when an assessment exists it adds the metrics as
   timeline facts. **Wording** stays "inside the modelled corridor /
   within modelled limits" — see Section 7.
4. Optionally, `entry_corridor_deg` can be fed from `entry_corridor_bounds()`
   for a specific vehicle instead of the fixed Orion-class band.

Data hand-off details:

* `entry_state` is Earth-centred ECLIPJ2000 (the same frame as the
  propagation). The assessment rotates it to an Earth-fixed frame via SPICE
  (`IAU_EARTH`, or ITRF93 if the high-precision kernel is loaded) to get
  latitude/longitude, `v_rel`, relative FPA and heading.
* The absolute epoch is `reference_epoch_ephemeris_seconds + initial_epoch + entry_epoch`
  (TDB seconds past J2000, consistent with task #159's time handling).
* Because the targeter's EI is a spherical-radius shell, the assessment
  starts integrating from that state as-is and reports the geodetic altitude
  of the start point; it does not re-target the EI.

## 6. Verification plan for the future implementation

* Ballistic (L/D = 0), non-rotating, exponential atmosphere: peak
  deceleration and its altitude match Allen–Eggers within a few percent
  over γ ∈ [-5°, -10°], V ∈ [7.5, 11.5] km/s.
* Peak g independent of β for ballistic entry (Allen–Eggers property);
  peak q̇ scales as √β / √R_n.
* Monotonicity: steeper γ ⇒ higher peak g and q̇, lower Q; lift-up ⇒ lower
  peak g than lift-down for the same γ.
* Skip-out: a lifting vehicle at a sufficiently shallow γ with lift-up
  produces `skip_bounded`; lift-down at the same γ is `captured` or is
  closer to capture (overshoot-boundary existence).
* Skip-out energy frame: a synthetic prograde equatorial exit state with
  inertial speed just above escape at EI (so `|v_rel|` is below it) must be
  `skip_escape`; the same state rebuilt from an Earth-fixed state with
  `sxform` must classify identically.
* Corridor bounds: `steepest < shallowest`, and both boundaries move in the
  expected direction when L/D increases (corridor widens).
* Regression: the converged #144 free-return case assessed with an
  illustrative capsule vehicle gives physically plausible magnitudes (peak
  g single digits for lift-up, q̇ hundreds of W/cm²); asserted only as
  ranges.

## 7. Scope: what this does **not** do

The following are explicitly out of scope and must not be implied by any
output, report field, UI label or doc:

* **No "survivable entry" claim.** Outputs say "within modelled limits
  under the stated model and vehicle assumptions"; with no limits supplied,
  only metrics are reported.
* No entry guidance (bank-angle modulation, skip-entry guidance, footprint /
  landing-site targeting); only constant-bank bounds.
* No 6-DOF, attitude dynamics, trim/stability or RCS.
* No Mach/Knudsen-dependent aerodynamic database, no rarefied-flow regime.
* No TPS sizing, ablation, material response or backshell heating; heating is
  stagnation-point correlations only, radiative heating is indicative.
* No atmospheric dispersions (density, winds), no Monte Carlo, no
  navigation/delivery errors at EI.
* No descent below ~30 km: drogue/main parachutes, splashdown loads, and
  blackout are not modelled.
* No crew physiology model beyond a user-supplied g limit (no
  duration-dependent tolerance curves).

## 8. Suggested follow-up tasks

1. `verify/entry_assessment.py`: `EntryVehicle`, `EntryLimits`,
   `assess_entry_constraints` (T2), T1 analytic oracle + tests (Section 6).
2. Report/schema residuals for C1–C5 and the `model` block.
3. `entry_corridor_bounds()` (Section 3) and opt-in use in the targeter.
4. Optional: WGS-84 geodetic EI and Earth-relative FPA in the existing
   corridor check (changes current numbers by tenths of a degree; needs its
   own review).
