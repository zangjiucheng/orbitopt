# Engineering notes

Root-caused gotchas from building orbitopt, kept in full detail here instead
of the main [README](../README.md) so that file stays scannable. Each of
these cost real debugging time and is exercised by a test, not just fixed
and forgotten.

## pygmo: `bfe` must be attached before wrapping

`bfe` must be attached to the *raw* UDA (`uda.set_bfe(bfe)`) *before*
wrapping it in `pg.algorithm(uda)`. `pg.algorithm` never exposes `set_bfe`
itself, and calling it on the wrapped object either raises or (if swallowed)
silently falls back to evaluating one candidate at a time -- ~15x slower,
with no error to indicate why. Also, only some algorithms support it at
all: `pg.pso_gen`, `pg.cmaes`, `pg.nsga2` do; `pg.sga`, `pg.sade`, `pg.de` do
not (pygmo 2.19.7). `orbitopt.optimize.runner` handles this correctly; see
its docstring.

## SPICE: `J2000` and `ECLIPJ2000` are not interchangeable

SPICE's `"J2000"` (mean equator/equinox) and `"ECLIPJ2000"`
(ecliptic-and-equinox) frames are related by a ~23.4-degree rotation
(Earth's obliquity) and are NOT interchangeable -- mixing states queried in
one against the other silently produces errors of hundreds of thousands of
km, not a small correction. Every module that touches Earth/Moon/Sun
ephemerides in this codebase uses `ECLIPJ2000` consistently; if you add a
new one, match it.

## RK4 step size that's fine for a coast is not fine for a flyby

A fixed step size that's perfectly fine for a multi-day interplanetary
coast (300-600s) is NOT fine near a close lunar flyby -- an empirical
convergence check found step_size=600s reporting an 81,733 km closest
approach for a trajectory whose true (step_size<=15s converged) closest
approach was 4,379 km. Local trajectory curvature near a
several-thousand-km-altitude flyby is far sharper than during a plain
coast; `verify.differential_correction` defaults to a 60s step for exactly
this reason. If you extend this code to closer flybys (hundreds of km
altitude or less), re-run the convergence check -- 60s may no longer be
fine enough.

## Verification layer: adaptive integrator + event location (task #136)

Screening (`dynamics.nbody_gpu`) stays fixed-step RK4 for GPU batching. The
tudatpy verification layer (`verify.tudat_propagate`) now defaults to
`integrator="dop853"` -- tudat's adaptive RKDP 8(7), rtol 1e-11 / atol 1e-6,
with the old `step_size` reused as the **maximum** step (so saved histories are
never coarser than before) and `terminate_exactly_on_final_condition=True`
(fixed-time-of-flight targeting needs the state exactly at the requested
epoch). `integrator="rk4"` reproduces the legacy results.

Events are roots of event functions solved with Brent's method on a cubic
Hermite dense output (`verify.events`), with the bracketing interval
re-propagated at fine spacing by the same `ReusableCoastPropagator`
(`refine=True`, automatic for `propagate_multi_arc` results):

| event | g(t) | API |
|---|---|---|
| periapsis / closest approach | (r - r_b).(v - v_b) | `find_closest_approach`, `find_periapsides` |
| SOI crossing | \|r - r_b\| - R_SOI | `find_soi_crossings` |
| entry interface | \|r\| - (R + h_EI) | `find_altitude_crossing` |

`find_closest_approach(method="sample")` and
`find_altitude_crossing(method="linear")` keep the legacy behaviour for
comparison.

Measured difference (`examples/11_event_location_comparison.py`, Part A:
Keplerian lunar flyby, true perilune 1837.400 km):

| output step | legacy nearest sample | error | event-located | error |
|---:|---:|---:|---:|---:|
| 15 s | 1837.431 km | +0.031 km | 1837.400000 km | < 1 mm |
| 60 s | 1837.892 km | +0.492 km | 1837.400000 km | < 1 mm |
| 300 s | 1849.652 km | +12.25 km | 1837.400000 km | < 1 mm |
| 600 s | 1885.817 km | +48.42 km | 1837.400000 km | < 1 mm |
| 1800 s | 2227.589 km | +390.2 km | 1837.400000 km | < 1 mm |

The legacy error is always positive (a saved sample can only be farther than
the true minimum) and grows ~quadratically with step; the event-located value
is step-independent. Part B runs the same comparison through tudatpy
(rk4+sample vs dop853+event) and must be run in the conda environment. Known
limitation: a pass shorter than one output step whose range rate does not
change sign between samples is caught only via the sample-wise minimum
fallback (Hermite root, not re-propagated).

## Verification report: every verification emits machine-readable JSON (task #137)

Every high-fidelity verifier (`verify_geo_raising`, `target_mars_approach`,
`verify_mars_orbit_insertion`, `target_lunar_flyby`, `target_lunar_b_plane`,
`compare_to_lambert_prediction`) now attaches a report (`result.report`, see
`orbitopt/verify/report.py`, schema `schemas/verification-report-1.0.json`)
with:

- **residuals**: position/velocity of the numerical solution minus a named
  reference (Lambert arrival, analytic screening orbit, targeter aim point);
- **integrator**: method, rtol/atol, min/max step, realised step statistics,
  and a **step-convergence** check (re-run at tolerances x0.01, or half the
  step for RK4; the final-state difference is the integration-error estimate);
- **conservation**: energy and angular-momentum drift per coast arc. Under
  perturbed dynamics these quantities are not conserved, so that block is
  `applicable: false`; the applicable check re-runs the same initial state
  under the central point mass only;
- **dynamics**: the perturbation inventory (Earth J2/J22 harmonics, Sun/Moon/
  planet point masses, plus what is *not* modelled).

Each report costs up to 2 extra propagations per propagation it covers. It is
byte-reproducible (no timestamps, sorted keys, `digest` = SHA-256 of the
canonical JSON). `orbitopt run` writes it next to the scene as
`<scene>.verification.json` (or `output.verification_report`); set
`ORBITOPT_VERIFICATION_REPORT_DIR` to also write every report from library calls.

## GPU is not always faster -- measured, not assumed

`nbody_gpu.propagate_spacecraft_batch` is GPU-*slower* than CPU below
roughly N~20,000 candidates (N=2000, n_steps=300: CPU 20.2s vs GPU 107.8s),
the opposite of the Lambert solver's crossover. The RK4 wrapper issues
4*n_steps small CuPy kernel launches per call (one Newton-solved Moon/Sun
position lookup per RK4 substage), and per-launch overhead dominates until
the batch is large enough to amortize it -- unlike `lambert.gpu_batch`,
which has no per-step Python loop calling it repeatedly. `problems/free_return.py`
and `examples/05_artemis2_free_return.py` default to `use_gpu=False` for
this reason; don't assume GPU is the right choice for a new batched routine
without measuring at your actual batch size.

**Update (task #139):** the Moon/Sun positions now come from a grid computed
once before integration: one batched Kepler solve over the 2*n_steps+1
half-step epochs, run once for each distinct (epoch0, dt) pair. The
4*n_steps Newton solves per call are gone. The output is bit-identical to
the old per-stage evaluation. N=2000, n_steps=300, CPU (numpy, measured on
the CI container, where the old code took 69.6s instead of 20.2s): **69.6s →
0.83s** (~84x). The GPU path loses the same 4*n_steps×maxiter kernel
launches, but it has not been re-measured (no CUDA device was available).
Re-check the CPU/GPU crossover before you change the `use_gpu=False`
defaults.

**Update (task #143):** profiling counted ~4.5 M array ops (≈ CuPy kernel
launches) per call before #139, which matches the old 107.8 s at about
24 µs per launch. Today the count is 24.8 k, so CuPy should now be roughly
on par with numpy at N=2000; this is an estimate that has not been measured
on a GPU. A fused per-candidate RK4 loop is 20-30x faster than numpy even
on the CPU (Numba). A CUDA fused kernel is not worth it at current batch
sizes. Benchmark, numbers and decision:
[`benchmarks/fused_kernel_evaluation.md`](benchmarks/fused_kernel_evaluation.md)
(`scripts/bench_nbody_backends.py`).

## Undamped Newton diverges on real (imprecise) inputs, not just in theory

Wiring the full pipeline together end-to-end -- not just testing each stage
in isolation -- surfaced a case where a GPU-coarse-screened TLI guess, only
slightly different from a hand-verified working one, sent
`differential_correction.target_lunar_flyby`'s plain Newton step into a
runaway divergence (residual growing to tens of millions of km within a few
iterations). Root cause: the coarse two-body Moon model's own
closest-approach-time estimate was off by more than two days for that
candidate, which anchored the fixed-time targeter to a time with no nearby
feasible solution. Fixed with two changes that are cheap to skip and easy
to regret skipping: (1) sanity-check the refined coast duration against the
caller's own guess before trusting it, falling back to the guess if the
"closest approach" found is implausibly far away in time or distance; (2)
cap and backtrack the Newton step (classic damped Newton) instead of always
taking the full linearized step. Both are exercised by
`tests/test_cislunar.py::test_target_lunar_flyby_converges_from_an_imprecise_gpu_screened_guess`,
using the exact delta-v that used to diverge -- a reminder that a solver
validated only on its own best-case hand-picked input isn't validated for
what an upstream (imprecise, automated) stage will actually hand it.

## Qt/VTK: which screenshot API actually captures the 3D view

Grabbing the Qt widget itself (`QWidget.grab()`) comes back solid black for
the embedded 3D view -- Qt's generic widget compositor doesn't reliably
capture the native OpenGL surface pyvistaqt renders into. `plotter.screenshot()`
(PyVista's own capture path) shows the real content; this is a
testing/screenshotting quirk, not a rendering bug -- the view displays
correctly on screen either way. Also, VTK's native Win32 OpenGL context
fails outright under Qt's `offscreen` platform plugin
(`QT_QPA_PLATFORM=offscreen`), so `tests/test_app.py` runs against a real
(if briefly-shown) window rather than a headless one, unlike this
project's other `off_screen=True` PyVista tests. README.md's own
screenshots are captured this way by `scripts/capture_screenshots.py`
(composite `window.grab()` with `plotter.screenshot()` for just the
viewport region; a real OS-level screen capture was tried first and
rejected after it picked up an unrelated foreground window instead of the
app -- not safe to rely on in an environment where this process can't
reliably hold OS focus). One trap when compositing: the image `grab()`
returns keeps the window's device pixel ratio, so a QPainter on it already
works in logical coordinates -- scaling the viewport rect by the ratio
again pastes the 3D view at twice its offset on a Retina display.

## VTK point markers: constant apparent size, and a bug this sidesteps

Body markers render via VTK's `render_points_as_spheres` point rendering,
which draws at a constant *screen-pixel* size regardless of camera
distance. An earlier, now-removed browser/Three.js iteration of this
viewer hand-rolled that same "constant apparent size" behavior
(recomputing world-space marker scale from distance-to-camera every frame)
and got the scale factor wrong: the Sun's marker stayed large enough in
world-space to visually swallow Mercury's entire orbit even fully zoomed
in. Letting VTK's own point rendering handle it avoids that whole class of
bug for free -- one more reason (on top of "no HTML/CSS/JS to hand-write")
this landed on a Python-native 3D toolkit instead of a from-scratch web
renderer.

## PyVista slider: `interaction_event` defaults to release-only

PyVista's `add_slider_widget` defaults to `interaction_event='end'` -- the
callback only fires on mouse-*release*, not while dragging. A naive test
that manually invoked VTK's `'InteractionEvent'` to simulate a drag
therefore silently did nothing (right event name, wrong one for the
widget's actual default), which looked exactly like the scene simply not
updating. Fixed by passing `interaction_event='always'` explicitly, both
for correctness (this test) and because live feedback while dragging is
the UX you actually want from a timeline scrubber. Covered by
`tests/test_pv_viewer.py::test_mission_timeline_slider_updates_spacecraft_position`.

## Cislunar / Artemis II free-return pipeline, in full

`examples/05_artemis2_free_return.py` chains every layer of the framework:

1. **Patched-conic seed**: an Earth-only 2-body Lambert arc (`lambert.cpu`)
   from a parking-orbit position to the Moon's real position (`bodies.moon_state`,
   sourced from tudatpy/SPICE, not pykep -- pykep's `jpl_lp` has no Moon) at
   the coast time, giving a plane- and energy-appropriate initial TLI burn
   guess instead of an arbitrary kick. (An arbitrarily-oriented parking
   orbit needs a wildly unrealistic 11+ km/s "TLI" burn to reach a
   misaligned Moon position in the allotted time -- this isn't a solver
   bug, it's what happens when the launch geometry doesn't match the
   target, exactly as it constrains real launch windows.)
2. **GPU coarse screening** (`problems.free_return.FreeReturnScreeningProblem`):
   a pygmo population of candidate burns around that seed, batch-evaluated
   through `dynamics.nbody_gpu`'s simplified propagator, searching for the
   burn whose coarse closest-approach to the Moon matches the target
   distance.
3. **tudatpy differential correction** (`verify.differential_correction.target_lunar_flyby`):
   Newton-Raphson refinement of the winning candidate against the real
   SPICE-based Earth+Moon+Sun n-body model, converging to the actual
   Artemis II perilune altitude (6,545 km above the lunar surface) to
   within ~10 km in 3 iterations / a few seconds.
4. **Independent fine-resolution verification**: re-propagate the converged
   solution at a finer step than the corrector used and re-measure closest
   approach from scratch, so the reported number isn't just "what the
   corrector's own residual said" (see the RK4 step-size gotcha above --
   this distinction mattered here in practice, not just in principle).

**What this does not claim:** NASA hasn't published Artemis II's
navigation-grade state vectors or SPICE kernels, so there is no ground
truth to fit against -- this independently re-solves the same free-return
boundary-value problem and lands on a trajectory of the same class and
comparable magnitudes, not a reproduction of the actual flown mission. It
also targets flyby *distance* only, in whatever direction the Lambert seed
happens to miss by; a genuine unpowered free return additionally needs the
B-plane crossing aimed so gravity alone bends the outbound trajectory back
through Earth's atmosphere, which is a 2-parameter aim-point targeting
problem this example doesn't attempt -- expect the example's post-flyby
trajectory to *not* re-enter on its own within the propagated window.
Adding that targeting (vary the B-plane aim point, not just distance, as
the Newton unknowns) is the natural next extension of
`differential_correction.py`.

## Window search: one porkchop minimum is not one basin

`viz/mars_transfer.py` used to take the single lowest cell of a 120x120
porkchop grid and run PSO only inside departure +-15 d / TOF +-30 d around
it. A porkchop plot has several distinct valleys (adjacent launch windows,
type-I vs type-II arcs), and a grid cell's value carries discretization
error of the same order as the gap between valleys -- so a narrow, deeper
valley that falls between grid nodes can rank behind a wide, shallow one
and be excluded before the optimizer ever sees it.

`optimize/multistart.py` replaces that with:

1. `find_diverse_basins` -- every grid local minimum, greedily
   non-max-suppressed so kept basins are at least one separation ellipse
   apart (Mars: 20 d departure x 40 d TOF, up to 4 basins);
2. `multistart_optimize` -- one independent PSO per basin (seed `seed+k`),
   each in the old +-15/+-30 box, plus one PSO over the full window; runs
   converging on the same optimum are merged;
3. a `MultistartReport` -- the basin-0 run *is* the old single-basin
   result (same box, same seed), so the comparison is recorded at no extra
   cost: `baseline`, `best`, `improvement`, `extra_candidates`.

The ranked candidates are then handed to tudatpy in order: the first whose
real flight closes (Earth-SOI escape, TCM converges) is exported; only if
none does (up to `MAX_TUDAT_CANDIDATES`) is the Lambert-best flown
best-effort, as before. The outcome (basins, runs, distinct candidates,
baseline vs flown Lambert dv, rejected candidates) is written into the
exported scene's TMI event note and logged at INFO.

`tests/test_multistart.py` pins the failure mode on a synthetic two-valley
landscape: the single-basin baseline is stuck at 7.0 in the wide valley,
multistart finds the narrow 6.0 valley (improvement 1.0, +1 candidate),
with both a brute-force optimizer and real pygmo PSO.

MGA: `problems/mga.screen_departure_windows` now returns distinct first-leg
basins (`diverse=True`) instead of the `top_k` lowest cells (which were
neighbours in one valley), and `rank_windows_by_full_mga` re-ranks those
windows by short independent full MGA-1DSM runs, so a launch window is no
longer kept or dropped on first-leg dv alone (see
`examples/02_mga_cassini_like.py`).

## Launch constraints in the fitness: measured effect on the optimum (task #153)

`orbitopt.launch_constraints.LaunchConstraints` checks every candidate's
departure v_inf against the selected rocket and launch site:

- **C3** `= |v_inf|^2` must not exceed `vehicle.max_c3(payload_kg)` (the
  task #151 curve; a payload outside the tabulated range is rejected when
  the constraint is built, never extrapolated).
- **DLA** (declination of v_inf in the Earth *equatorial* J2000 frame --
  pykep's `jpl_lp` velocities are ecliptic, so v_inf is rotated by the
  23.439 deg obliquity first) must satisfy `|DLA| <= min(i, 180 - i)` for
  some direct-ascent inclination `i` the site can reach (task #150). Cape
  Canaveral gives `|DLA| <= 59.8 deg`; every packaged site that reaches
  i = 90 deg is unconstrained.

`Transfer2BodyProblem(..., launch_constraints=lc)` applies it in `fitness()`
and in the vectorized `batch_fitness()` used by both the CPU and GPU bfe
(`apply()`/`evaluate()` are array-module agnostic, numpy or cupy).
`lc.with_enabled(False)` (or `launch_constraints=None`) switches it off and
reproduces the unconstrained fitness bit-for-bit
(`tests/test_launch_constraints.py`).

Two handling modes, both measured with
`examples/12_launch_constrained_transfer.py` (Earth->Mars, departures in
2028, TOF 100-400 d, PSO 150 gen x 512, seed 7, Falcon 9 expendable,
4000 kg payload -> C3 <= 9.0 km^2/s^2, Cape Canaveral):

| run | departure | TOF [d] | total dv [m/s] | C3 [km^2/s^2] | DLA [deg] | feasible |
|---|---|---|---|---|---|---|
| constraints off | 2028-11-24 | 300.7 | 5988.9 | 9.09 | 29.3 | no |
| `mode="invalid"` | 2028-11-26 | 305.9 | 6015.7 | 9.00 | 29.9 | yes |
| `mode="penalty"`, weight 1000 (default) | 2028-11-26 | 305.9 | 6015.7 | 9.00 | 29.9 | yes |
| `mode="penalty"`, weight 100 | 2028-11-25 | 301.5 | 5996.8 (+penalty) | 9.07 | 29.6 | no |

Switching the constraints on moves the optimum 2.4 days later with a
5.2 day longer flight and costs +26.8 m/s total dv, landing exactly on the
C3 limit. The DLA limit does not bind for this window (29-30 deg vs. 59.8).

Gotchas found while measuring:

- **Flat `invalid` landscape.** A plain `1e6` sentinel for every violator
  gives a population that starts fully infeasible nothing to climb. Invalid
  candidates therefore score `invalid_fitness + c3_excess + dla_excess`:
  still unmistakably invalid, but graded. With an infeasible request
  (Long March 3B, 2800 kg -> C3 <= 7.8, 2026 window, whose minimum C3 is
  ~9.14) the optimizer now converges on the least-violating trajectory
  (C3 9.14) instead of an arbitrary point (C3 303 before the fix).
- **Penalty weight must beat the objective's slope.** Near a Mars-window
  optimum d(dv)/d(C3) is ~300 m/s per km^2/s^2, so a weight of 100 m/s
  per km^2/s^2 leaves the optimum slightly over the limit (row 4). The
  default is 1000, which matched `invalid` mode exactly.
- **Window edges.** In a 2030-01-01..2031-01-01 search box the optimum sits
  on the upper t0 bound, so constrained/unconstrained comparisons there
  measure the box, not the constraint -- pick a window containing the
  whole launch period.

## Epochs: one leap-second-aware UTC -> TDB conversion (task #159)

`orbitopt.bodies.mjd2000_to_ephemeris_seconds` used to pass MJD2000 straight
through tudat's `julian_day_to_seconds_since_epoch` (no time-scale change, so
UTC was silently treated as TDB), while `orbitopt.injection` hard-coded
`(mjd2000 - 0.5) * 86400 + 69.184` (TT - UTC frozen at its post-2017 value).
For the same input the two disagreed by ~69 s, and the Mission Control date
label assumed a third offset (64.184 s). All three now call
`orbitopt.timescales`: MJD2000 is a **UTC** label, ephemeris seconds are
**TDB** past J2000 noon, TT - UTC comes from the IERS leap-second table, and
TDB - TT from SPICE's DELTET formula. It matches SPICE `str2et` (64.183927 s
at 2000-01-01 12:00 UTC) and ERFA `utctai`/`taitt` to < 10 us
(`tests/test_timescales.py`).

Measured change of the computed ephemeris epoch:

| epoch (UTC)      | `bodies` paths | `injection` paths |
|------------------|----------------|-------------------|
| 2000-01-01 12:00 | +64.184 s      | -5.000 s          |
| 2020-07-30 11:50 | +69.183 s      | -0.7 ms           |
| 2026-08-01 00:00 | +69.183 s      | -0.7 ms           |

* **Launch-to-Injection** (`examples/13`, numpy backend): output is
  bit-identical before/after -- for any post-2017 date only the sub-ms
  TDB - TT term changed. The Mars 2020 replay's injection step (`plan_departure`
  at a 2020 epoch) is in the same post-2017 regime.
* **`bodies`-based paths** (Moon state, `IAU_*` orientations, `mars_transfer`,
  `mission_timeline`, `geo_insertion` reference epochs) now sample the
  ephemeris 69.18 s later than before for the same UTC input: ~70 km of Moon
  motion along its orbit, ~2,060 km of Earth/Mars heliocentric motion, and
  0.29 deg of Earth rotation (~32 km at the equator for a launch site's
  inertial position). These are now the correct values; the Mars 2020 replay
  and tudat-backed verifications need the conda compute stack to re-fly and
  should be re-run there to confirm the downstream trajectory deltas.

## Unified mission pipeline (task #170)

Every config-driven mission runs one lifecycle, implemented in
`src/orbitopt/pipeline.py`:

`mission_config → constraints → search → optimize → select → refine → verify → export`

`run_pipeline(config)` returns a `PipelineResult`; `to_dict()` is a
`pipeline-result-1.0` document (`src/orbitopt/schemas/pipeline-result-1.0.json`)
with the same structure for every mission kind. Each stage reports candidate
trajectories, convergence status, constraint violations, the physical model,
numerical error estimates and wall-clock time. Stages a kind doesn't have are
`skipped`; stages after a failure are `not_run`.

The CLI (`orbitopt run`, which now also writes `<scene>.pipeline.json`), the
Python API (`mission_config.build_scene_from_config`, `pipeline.run_pipeline`)
and the GUI's built-in `goes` / `mars` missions (`missions.builtin_mission_config`)
all call the same `run_pipeline`.

Migration is incremental. A kind's `build_from_config(config, recorder=None)`
reports its stages to the `StageRecorder` it gets (`with recorder.stage("search")
as st: ...`, `recorder.add_time(...)`, `recorder.skip(...)`); the pipeline
handles config validation, the verify summary from verification reports, and
export. `geo-raising` and `mars-transfer` are migrated.

## Candidate Selection and the Verification Gate (task #171)

`src/orbitopt/candidate_selection.py` adds two explicit steps between the
cheap search/optimize stages and the high-fidelity verify stage.

- **Candidate Selection** (`select_candidates`, the `select` pipeline stage):
  drops infeasible (non-finite cost) candidates, removes near-duplicates
  (within a per-coordinate tolerance of a cheaper one), ranks by cost and,
  when the list is cut to `max_candidates`, fills it first with *diverse*
  candidates (at least one `min_separation` ellipse from every cheaper pick).
  Each input candidate is recorded on the `select` stage with its status
  (`selected` / `duplicate` / `infeasible` / `truncated`) and the reason.
  Mars transfer uses 1 d / 2 d duplicate tolerance, the multistart basin
  separation (20 d / 40 d) as the diversity scale and `MAX_TUDAT_CANDIDATES`.
- **Verification Gate** (`VerificationGate`, `walk_candidates`): every selected
  candidate gets a verdict: `rejected` (refinement didn't hold up, with the
  reason), `accepted` (tri-state verification result `passed`), `degraded`
  (flown, but `failed` or `not_checked`, with the failing / unchecked reports)
  or `not_evaluated`. A degraded candidate is never a success, so Mars keeps
  trying the next candidate. If none passes, the best-ranked degraded
  candidate is exported. If every candidate was rejected, the Lambert-best one
  is re-flown without requiring TCM convergence. Either way the export is
  `best_effort` and carries `BEST_EFFORT_WARNING` in the pipeline result's
  `gate.warnings`, the scene's `warnings` and first subtitle line, the CLI
  output and the log.

The pipeline result gains `success` (completed **and** the exported candidate
passed the gate) next to the execution `status`, and a `gate` block with the
per-candidate verdicts. Builders that don't drive the gate get a generic one in
which the exported scene's verification reports decide: `failed` is
best-effort, and `not_checked` is "not verified" (warned, but not a success).

