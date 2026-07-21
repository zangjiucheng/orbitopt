# Mars — Earth to Orbit Insertion: Mission Plan

## Scope

This mission models a Mars probe's entire journey from Earth launch through
Mars orbit insertion (MOI). **It does not model landing.** This framework has
no atmosphere model (point-mass/N-body gravity only, see
`verify/tudat_propagate.py`), so a descent-to-the-surface arc would be
fabricated physics — unlike everything else this framework claims. This
mirrors `geo_raising.py` not drawing the launch ascent a two-body propagator
can't represent, and Artemis II's own "what this does not claim" precedent.
The mission is titled and scoped accordingly: **"Mars — Earth to Orbit
Insertion,"** not "...to Landing."

## Why one continuous heliocentric-km scene

Unlike Artemis II (Earth-Moon, Earth-centered throughout) or GOES
(Earth-centered GTO→GEO), a Mars mission genuinely crosses three physics
regimes — near-Earth, heliocentric cruise, near-Mars — and three
characteristic length scales six orders of magnitude apart (hundreds of km
at each planet vs. hundreds of millions of km in cruise). The scene format
imposes no per-segment frame or scaling (`scene_format.py`'s schema: trail
positions are just numbers, `centralBodyId` is a descriptive label never
read by the renderer), so this mission is exported as **one continuous
trail in a single heliocentric, km-scale frame**: each Earth/Mars-centered
phase is propagated in its own natural frame, then converted to heliocentric
by adding that body's real heliocentric position (from SPICE, via
`bodies.py`/`tudat_propagate.py`) at each sample's absolute epoch — exact
vector addition (ECLIPJ2000 axes are non-rotating, so this is a plain
Galilean frame translation), not an approximation. The tradeoff is that
near-Earth/near-Mars detail is sub-pixel at the whole-mission default zoom;
Mission Control's Track/Focus/Face camera system is the tool for inspecting
either end up close.

## Real reference numbers used

- **Departure window:** real, near-term Earth-Mars Hohmann-class transfer
  window, searched 2026-09 through 2027-02 (Earth-Mars synodic period is
  ~780 days; recent windows are ~2020-07, ~2022-09, ~2024-10, implying the
  next one lands in ~2026 Q4). The porkchop-screen + PSO-optimizer pipeline
  converged to a departure of ≈2026-Oct-30 (MJD2000 9799.78), arrival
  ≈2027-Sep-06, time of flight ≈310.67 days, departure C3 ≈9.2 km²/s²
  (physically realistic for an Earth-Mars transfer) — this is a genuinely
  optimized result, not a hand-picked date.
- **Parking orbit:** 185 km circular, matching the Artemis II mission's own
  convention (`mission_timeline.py`).
- **Earth sphere of influence:** 924,000 km (real value), used as the
  phase-1 hand-off boundary.
- **Target Mars capture orbit:** ~500 km periapsis altitude × 20,000 km
  apoapsis altitude — a representative science-orbit-class ellipse, not a
  specific real mission's exact target (no single real Mars orbiter's
  capture orbit is claimed to be reproduced here).

## Phase-boundary idealizations (stated plainly, not hidden)

1. **Idealized Lambert departure vs. reality.** The Lambert solver's
   departure v∞ is idealized: it assumes an instantaneous departure at
   Earth's exact heliocentric state at t0. The real trajectory departs from
   Earth's *sphere of influence*, a few days later, once Earth's own
   heliocentric velocity direction has rotated a few degrees. Measured (not
   assumed): propagating the real post-departure state under real
   Earth/Mars/Jupiter third-body perturbation misses Mars by **over 1.27
   million km** relative to the idealized Lambert arrival point. This is
   genuine physical sensitivity, not a numerical artifact — Earth's gravity
   at the ~924,000 km SOI boundary is still ~7.9% of the Sun's local pull,
   and this transfer's ~157.8° transfer angle sits in the numerically- and
   physically-sensitive near-180° regime. Every real interplanetary mission
   flies a trajectory-correction maneuver (TCM) for exactly this reason;
   this mission does too (see below).
2. **Trajectory-correction maneuver (TCM).** A fixed-time Newton-Raphson
   differential-correction shooting method (`verify/mars_insertion.py::
   target_mars_approach`, mirroring `verify/differential_correction.py::
   target_lunar_flyby`) targets a specific Mars periapsis radius under real
   perturbed propagation (Sun + Earth + Mars + Jupiter point masses). The
   correction re-refines its target coast duration between outer rounds
   (mirroring `target_lunar_b_plane`'s own documented need for this — a
   large correction can shift the trajectory's true closest-approach time
   enough that a stale fixed-time target stops being meaningful). The
   targeter is not guaranteed to fully converge to the exact target
   periapsis in a bounded number of iterations; the scene's "Trajectory
   correction maneuver" event note reports the actual achieved miss
   distance and whether it converged, rather than silently assuming success.
3. **Mars orbit insertion (MOI).** An apsis-preserving, tangential-only
   capture burn (closed-form vis-viva), applied at the *true* periapsis of
   the hyperbolic approach — located via a dedicated two-stage propagation
   (`verify/mars_insertion.py::locate_mars_periapsis`: a coarse bulk-step
   coast for the long final approach, then a fine-step local window around
   the anticipated encounter). This two-stage approach exists because even
   a step size fine enough for the multi-hundred-day cruise (hundreds to
   thousands of seconds) is too coarse to locate the true periapsis of a
   close, fast planetary encounter — the same class of RK4-step-size gotcha
   already documented for the lunar-flyby case elsewhere in this project.
   A short post-MOI stationkeeping propagation confirms the captured orbit
   actually holds over a few days, the same sanity check `geo_insertion.py`
   performs for GEO.
4. **One numerical trajectory for the whole cruise leg, not two glued
   together.** An earlier version of this pipeline displayed the cruise leg
   from an *independently re-propagated* `propagate_two_body_leg` call
   (3600 s step) glued to `locate_mars_periapsis`'s own bulk+fine result at
   the Mars approach. Even though both used identical physics (same
   perturbing bodies, same central body), the differing step sizes' RK4
   truncation error left the two numerical solutions of the *same* physical
   trajectory measurably apart by the time they reached Mars — a residual of
   ~90,000–110,000 km (an implied ~70–110 km/s "speed" across that one trail
   sample, well above any real heliocentric speed this mission ever
   reaches), visible as a small teleport in the exported trail right at Mars
   arrival. Measured via a dedicated continuity test comparing consecutive
   trail-sample implied speeds, not eyeballed. Fixed by having
   `locate_mars_periapsis` return its own bulk+fine trajectory (already
   heliocentric, since both stages use `central_body="Sun"`) for `viz.
   mars_transfer` to display directly as the cruise leg, and having phase 3
   pick up *exactly* at the periapsis state that trajectory already ends at
   — one continuous numerical solution end to end, with an exact (not
   merely close) seam by construction, not a loosened tolerance.

## Modules

- `viz/mars_transfer.py` — the 3-phase pipeline + scene assembly
  (`_build_mars_mission`, `compute_and_export_mars_mission`,
  `build_from_config`).
- `verify/mars_insertion.py` — the TCM targeter (`target_mars_approach`),
  precise periapsis locator (`locate_mars_periapsis`), and MOI burn +
  stationkeeping verifier (`insert_mars_orbit`, `verify_mars_orbit_insertion`).
- `schemas/mission-mars-transfer-1.0.json` — the `mars-transfer` mission-config
  schema (`orbitopt validate`/`orbitopt run`).
- Registered both as a config-driven kind (`mission.kind: mars-transfer`) and
  as the zero-arg built-in mission `"mars"` (`missions.py`), mirroring
  `geo-raising`/`"goes"`.

## What this does not claim

- No landing/EDL — see Scope above.
- No launch ascent from the pad — the trail starts already in the 185 km
  parking orbit, same as every other mission in this project.
- The TCM/MOI targeting is not guaranteed to fully converge for every
  departure-window/time-of-flight combination a user's config might select;
  the exported scene reports the actual achieved result rather than
  asserting success.
