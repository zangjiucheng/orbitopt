# ADR rendezvous-0002: safety constraints on the whole trajectory, additive to config v1.0 (E12-04)

- **Status:** Accepted
- **Date:** 2026-11-25
- **Deciders:** E12 Orbital Rendezvous & Mission Operations (Epic #389), E12-04 (#394)
- **Code:** `src/orbitopt/rendezvous/safety.py`, `src/orbitopt/rendezvous/config.py`,
  `src/orbitopt/schemas/rendezvous-mission-1.0.json`, `src/orbitopt/data/leo_rdv_001.yaml`
- **Evidence:** `tests/test_e12_04_safety_constraints.py`
- **Builds on:** ADR rendezvous-0001 (two-impulse J2-targeted Lambert), ADR relative-0001 (`propagate_pair`)

## Context and problem

E12-04 pins the reference mission's safety envelope: a 500 m keep-out zone, an
approach corridor, a range-dependent approach-speed ceiling, the (already
fixed) fuel budget and time windows — checked on the **whole** trajectory,
coast between the burns included, not just at arrival; plus passive safety: an
uncontrolled chaser after a burn abort must not enter the keep-out zone. Two
measurements shaped the design:

1. *The corridor cannot be a V-bar tube at all ranges.* Every 0-rev Lambert arc
   in the feasible window climbs 10 km of radial separation by flying **over**
   the target: the E12-03 reference transfer reaches LOS = 179.9 deg from
   -V-bar at ~1.5 km range (inside-1.2-km LOS stays <= 79.6 deg). A full-range
   aft cone would reject the entire impulsive two-pulse family — the constraint
   set would be a statement that the mission cannot fly, not a safety envelope.
2. *The safety-screened feasible set is not convex.* Keep-out (min range < 500 m)
   screens out tof ~1500–2400 s and the terminal cone screens ~2450–2950 s, so
   the feasible set is two islands: [1400, ~1500] s at dv ~13.98 m/s and
   [~2950, 3600] s down to dv 10.34 m/s. A golden-section search given the
   screen discards the good basin on its first two infeasible samples (it
   reported 13.81 m/s as "optimum" for the reference config).

## Decision

1. **`safety` is an optional additive block inside schema version 1.0 — no
   version bump.** Every pre-E12-04 document validates and behaves exactly as
   before (a config without the block has `safety: None` and keeps the E12-03
   pipeline); a document *with* the block is schema-validated and boundary
   cross-checked by the loader (speed knots strictly increasing, caps
   non-decreasing with range, keep-out sphere must exclude the Hold Point).
   Splitting a 1.1 schema would have forked the loader and forced migration on
   configs that gain nothing from the feature.
2. **Corridor = terminal aft cone + containment.** Within `cone_range_m`
   (1100 m, one Hold-Point-radius beyond arrival) the line of sight from the
   target must stay within `cone_half_angle_deg` (90 deg — "never in front") of
   -V-bar; everywhere, |z| <= `cross_track_max_m` (100 m). The mid-range over-
   pass geometry is governed by the keep-out sphere and the speed profile
   instead. A tighter tube at all ranges is a powered-terminal-guidance
   requirement, out of scope here (ADR rendezvous-0001 scope note).
3. **Approach speed is a piecewise-linear ceiling in range** — constant below
   the first knot, linear between knots, constant above the last — with knots
   [500, 1000, 2500, 9500, 13000] m and caps [9, 9, 12, 22, 26] m/s. The
   profile admits the two-pulse family's measured kinematics (worst reference
   sample: 6.46 m/s inside 1 km, 18.5 m/s crossing 9.5 km) rather than an
   ideal that impulsive transfers cannot meet; it still binds hard on the
   screened-out mid-window arcs and on any synthetic fast approach.
4. **Every constraint is evaluated per epoch on a dense re-propagated grid**
   (`check_sample_dt_s` = 2 s, leg ends and burn epochs always sampled, both
   sides of each burn) over drift leg, transfer leg and the post-insertion
   sample; `fuel_budget` and `time_window` are event checks on the same report.
   Each check carries its signed margin series and the worst epoch + LVLH
   position, so violations are locatable, not just countable. Passive safety is
   three no-thrust coasts on the same propagator — from t0 (no burn executed),
   from the gate (burn-1 aborted), from the Hold Point (post-insertion loss of
   control) — each over `passive_contingency_s` = 5400 s (one orbital period),
   graded against the keep-out margin only.
5. **Safety-constrained optimization = coarse scan + refined basin.**
   `plan_rendezvous_safe` scans the fixed tof window (12 intervals) through a
   per-candidate hard screen (budget + the four trajectory checks on the dense
   transfer leg), then golden-section-polishes around the best feasible scan
   point inside its bracket, then re-verifies the winner independently
   (`verify_plan_safety`, drift leg and passive coasts included). Nothing is
   ever relaxed at runtime; when the screen kills every candidate the error
   carries the collected per-constraint worst margins ("plan safe, or say
   exactly why not").
6. **Reference margins are measured, not guessed.** The packaged numbers give
   min margins: keep-out +413.0 m (t = 3898.6 s), cone +19.99 deg, cross-track
   +79.9 m, approach speed +2.54 m/s (inside 1 km), fuel +14.65 m/s, windows
   +183.3 s, passive +9529 / +9529 / +500 m. The constrained optimum (tof
   3416.7 s, dv 10.345 m/s) matches the E12-03 unconstrained plan — the screen
   removes half the window but not the optimum; the report says so per epoch.

## Consequences

- `plan_rendezvous` keeps E12-03 semantics byte-for-byte; the optional
  `feasible` hook is inert when unused. Safety-planning runtime on the
  reference config is ~18 s (scan + refine + verify + 3 passive coasts).
- `evaluate_trajectory_safety(cfg, times, rho, rho_dot)` is public: any
  measured or constructed LVLH sample series can be graded — this is how the
  acceptance's negative test ("a constructed keep-out crossing must fail") is
  proven without contriving an orbit.
- Grid resolution is a declared fidelity knob: `check_sample_dt_s` = 2 s found
  cone margins 18.8 deg vs 8 deg on the coarse 5 s scan grid — margins between
  burns are only as sharp as the grid that measured them.
- The 2 s grid aliases the true LOS peak (79.6 deg at 1 s vs 70 deg at 2 s for
  the reference); the packaged 90 deg cone keeps >= 19 deg of grid-independent
  margin (measured at 1 s: 71.2 deg inside 1100 m), so the verdict does not
  depend on the aliasing.
- Keep-out is checked at the Hold Point too — it passes only because the Hold
  Point sits at 1000 m; the loader forbids a keep-out radius >= |Hold Point|,
  so a mission can never forbid its own arrival point.
