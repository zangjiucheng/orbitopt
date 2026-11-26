# ADR rendezvous-0002: TudatPy two-spacecraft re-flight as independent verification (E12-05)

- **Status:** Accepted
- **Date:** 2026-11-25
- **Deciders:** E12 Orbital Rendezvous & Mission Operations (Epic #389), E12-05 (#395)
- **Code:** `src/orbitopt/verify/rendezvous_refly.py`, `src/orbitopt/verify/tudat_propagate.py`,
  `src/orbitopt/crossenv/` (rendezvous_refly stage)
- **Evidence:** `docs/benchmarks/e12-05-refly-run/` (sealed exchange + stage report),
  `tests/test_e12_05_independent_verification.py`, `benchmarks/bench_e12_05_independent_refly.py`
- **Builds on:** ADR relative-0001 (LVLH frames, `propagate_pair`), ADR rendezvous-0001
  (LEO-RDV-001 config, `verify_plan`), E5-03 crossenv stage protocol, task #261 degraded rule

## Context and problem

E12-05 verifies the LEO-RDV-001 plan with an *independent* propagator: both
spacecraft re-flown in TudatPy with the burns applied as impulses, reusing the
E5-03 crossenv stage so the verification runs in a second environment
(tudatpy 1.0, separate interpreter) and arrives through a sealed, revalidated
exchange. Three constraints shaped the design.

First, the acceptance demands that error sources be *separable*: numerical
propagation error, approximate-model error (CW vs nonlinear), optimizer
nonconvergence and cross-model disagreement are four different liabilities,
and one blended `passed` would hide any of them behind another.

Second, the TudatPy interpreter's SPICE kernel pool defines no GCRS/ICRS frame
at all (`SPICE(UNKNOWNFRAME)`), and a rejected frame makes tudat *abort the
propagation at t = 0 without raising*, handing back a one-sample history —
silent truncation, the worst failure mode for a verification harness.

Third, when the re-flight is forced out of GCRS, the frames are *not* related
by the ~25 mas precession one naively assumes: GCRS and ECLIPJ2000 differ by
the 23.44° obliquity of the ecliptic. A zonal J2 field is axisymmetric only
about the spin axis, so feeding GCRS components in as ecliptic components
misplaces every J2 torque — measured at 133 m of false cross-model
disagreement on the hold point (and a matching min-approach offset), orders of
magnitude above the integration error being certified.

## Decision

1. **Fly in ECLIPJ2000, rotated correctly.** The config's GCRS states and the
   inertial burns are rotated by the IAU76 obliquity matrix about x (verified
   against SPICE `compute_rotation_matrix_between_frames("J2000","ECLIPJ2000",0)`
   to 1e-7) before propagation in ECLIPJ2000 at anchor epoch 0.0. Every graded
   quantity is chief-LVLH or a norm — invariant under that single constant
   rotation applied to both spacecraft — so no comparison needs it undone.
   The abort guard in `run_refly_in_current_environment` refuses any history
   that does not reach the leg end, turning tudat's silent t=0 abort into a
   loud `ReflyBackendError`.

2. **Force the config's exact physics onto the tudat environment.**
   `central_gravity={"gravitational_parameter": mu, "reference_radius": Re,
   "j2": J2}` overrides the SPICE-default gravity so the only differences
   left are gravity *formulation* and *integration*. Residual: **0.16 mm**
   hold-point position error against the main nonlinear J2 model.

3. **Four separate error-source lines, never one blended verdict.**
   `numerical_propagation` (analytic two-body replay + step/tolerance sweep),
   `approximate_model_cw` (CW vs nonlinear reference),
   `optimizer_nonconvergence` (landing residual + Δv stationarity),
   `independent_model_agreement` (refly vs main on the four acceptance
   quantities). Evidence types are `independent_model`, `analytical`,
   `numerical_consistency`; `historical_telemetry` is structurally absent
   (0 checks). Missing backend ⇒ `not_checked` with
   `REFLY_BACKEND_MISSING`, exit code 2 — never a fake pass (task #261).

4. **Compare sweeps at the latest common sample epoch.** A fixed-step RK4
   stores samples on its own grid and *straddles* the requested leg end;
   comparing last samples would measure a time offset (a 60 s grid looks
   2.3e5 m wrong at 7.5 km/s), not integration error. Every sweep run is
   compared against the 3.75 s reference at their latest common epoch ≤ leg
   end; the adaptive DOP853 tolerance axis terminates exactly at leg end
    (residual 0.0 m, order n/a), the step axis shows order **4.10** and a 21.6
    m position / 0.023 m/s velocity bound at 60 s steps, 2.3× below the
    declared 50 m. Each metric is sliced per quantity -- position over both
    crafts' position columns of the concatenated state, velocity over the
    velocity columns; the original baseline normed the mixed pair state
    (`d[:6]` / `d[6:]`), reporting the deputy position difference as a
    velocity, corrected and regenerated under E12-09 (task #411).

5. **The circular-orbit Kepler replay is exact.** Both initial orbits are
   exactly circular (e ≈ 0), where the classical Kepler-element pipeline
   divides by e; `verify/benchmarks.py` now uses the argument of latitude for
   mean anomaly and a rotation about the (unique) angular-momentum axis.
   Residual vs the point-mass re-flight: **1.1e-7 m**.

## Consequences

- Cross-model band 1.0 m / 0.01 m/s stands with >6000× margin (measured
  1.6e-4 m / 5.9e-8 m/s); CW bands 600 m / 0.5 m/s hold at measured 252.7 m /
  0.20 m/s, documenting how much the linearized planner model costs.
- The E12-09 slice correction (task #411) does not move any E12-05 hold-point
  conclusion: the headline agreement compares the acceptance quantities
  directly, never through the sweep slicing, and repeats at 1.6e-4 m /
  passed. Only the numerical line's sweep columns change — the step-axis 60 s
  position bound moves 15.2 → 21.6 m (still 2.3× inside its 50 m band) and
  the step-axis velocity column becomes a real 0.023 m/s instead of the
  deputy position difference it used to show.
- The obliquity lesson generalizes: *any* frame change in a J2 mission is a
  physics change, not a bookkeeping one; the guard comment lives at the
  rotation site so the next frame edit meets it first.
- TudatPy is a dev/verification dependency in a dedicated interpreter, not a
  package requirement: everything degrades to `not_checked` without it.
