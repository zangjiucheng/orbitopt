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
screenshots were captured this way -- see the git history of
`docs/assets/mission-control-*.png` for the actual capture script's
approach (composite `window.grab()` with `plotter.screenshot()` for just
the viewport region; a real OS-level screen capture was tried first and
rejected after it picked up an unrelated foreground window instead of the
app -- not safe to rely on in an environment where this process can't
reliably hold OS focus).

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
