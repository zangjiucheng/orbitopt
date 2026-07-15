# orbitopt

Orbital trajectory optimization research framework built on **pykep**
(fast patched-conic trajectory design + pygmo global optimization) and
**tudatpy** (high-fidelity numerical propagation), with **CUDA-accelerated
batch candidate evaluation** via CuPy.

## Why this split

- **pykep + pygmo**: cheap, analytic/patched-conic trajectory models
  (Lambert arcs, multi-gravity-assist sequences) searched with population-based
  global optimizers. Fast to evaluate, good for exploring huge search spaces,
  but ignores perturbations.
- **tudatpy**: numerical, N-body-capable propagation. Expensive per call, used
  to *verify* a pykep/pygmo solution actually holds up when Earth/Mars/Jupiter
  gravity, etc. are included, not to search the space itself.
- **CUDA (CuPy)**: population-based optimization and porkchop-plot generation
  evaluate thousands to millions of *independent* candidate trajectories per
  run. That evaluation is batched into single vectorized array ops instead of
  a Python loop, running on GPU when available and falling back to vectorized
  numpy otherwise -- same code path either way (`orbitopt.core.gpu`).

## Architecture

```
src/orbitopt/
  core/
    gpu.py          numpy/cupy array-module abstraction (the GPU on/off switch)
    problem.py       OrbitOptProblem base class: pygmo UDP + optional batch_fitness()
  lambert/
    gpu_batch.py      batched 0-rev Lambert solver (universal-variable / Vallado),
                       vectorized -- the framework's core CUDA-acceleration point
    cpu.py             thin wrapper around pykep's reference Izzo solver (multi-rev capable)
  bodies.py            pykep.planet ephemeris + epoch helpers + real Moon state (via tudatpy/SPICE)
  dynamics/
    nbody_gpu.py       GPU-batched, simplified Earth-Moon(-Sun) coarse propagator
                        (two-body Kepler batch solve + osculating Moon/Sun ephemeris +
                        fixed-step RK4) for screening cislunar TLI candidates
  problems/
    transfer_2body.py  single Lambert-leg transfer UDP, with a real batch_fitness()
    mga.py             wraps pykep.trajopt.mga_1dsm; GPU-screens launch windows first
    free_return.py     GPU-batched TLI-burn screening UDP (orbitopt.dynamics.nbody_gpu-backed)
  optimize/
    gpu_bfe.py         custom pygmo UDBFE that dispatches to a problem's batch_fitness()
    runner.py          drives pygmo algorithms (pso_gen/cmaes/nsga2) through the GPU bfe
  verify/
    tudat_propagate.py numerically propagate a candidate leg and diff vs. the
                        pykep patched-conic prediction; also multi-arc propagation
                        (coast/impulsive-burn sequences) + post-hoc closest-approach
                        and altitude-crossing event detection
    differential_correction.py  fixed-time Newton-Raphson shooting targeter that
                        refines a coarse TLI guess into a precise lunar-flyby distance
  viz/
    porkchop.py        GPU-batched porkchop grid + plotting
    scene.py           shared SceneData document schema (bodies/orbits/trails/timeline)
                        every viewer-facing exporter below builds through
    solar_system.py    exports the whole solar system as a SceneData document
    mission_timeline.py exports the Artemis II free-return mission as a
                        scrubbable SceneData document
    pv_viewer.py        general-purpose interactive 3D viewer (PyVista/VTK) --
                        renders any SceneData document; no HTML/CSS/JS anywhere
```

The cislunar pieces (`dynamics/nbody_gpu.py`, `problems/free_return.py`,
`verify/differential_correction.py`, plus the Moon ephemeris in `bodies.py`
and the multi-arc extension in `verify/tudat_propagate.py`) implement the
same two-stage pattern as the rest of the framework, applied to an
Artemis-II-like Earth-Moon free-return trajectory: `examples/05_artemis2_free_return.py`
Lambert-seeds a parking-orbit-aligned TLI guess, GPU-batch-screens the burn
vector against a simplified osculating Moon model, then refines the winner
in tudatpy to hit the real Artemis II perilune altitude (6,545 km) to
within ~10 km.

Extending the framework with a new problem type means writing a new
`orbitopt.core.problem.OrbitOptProblem` subclass; if its per-candidate work is
vectorizable, give it a real `batch_fitness()` and it gets GPU accel through
`optimize.runner.run_optimization` for free. If not (e.g. wrapping pykep's own
compiled UDPs, as `problems/mga.py` does), it still works, just CPU-bound --
GPU pre-screening of the search space (see `problems.mga.screen_departure_windows`)
is the fallback acceleration point for that case.

## Setup

tudatpy is conda-forge/tudat-team only (no PyPI wheels) and pins `python=3.10`;
pykep additionally needs `scipy<1.14` (it still calls the since-removed
`scipy.interpolate.interp2d`). See `environment.yml` / `scripts/setup_env.ps1`.

```powershell
.\scripts\setup_env.ps1
conda activate orbitopt
pytest tests/ -v
```

GPU packages (`cupy-cuda13x`, `numba`) are pip-installed inside the conda env;
adjust the `cupy-cudaXXx` package name in `environment.yml` to match your
CUDA driver version (`nvidia-smi`) if not CUDA 13.

## Measured results (RTX A3000 Laptop, 6 GB, this machine)

| Workload | CPU (vectorized numpy) | GPU (CuPy) | Speedup |
|---|---|---|---|
| 2,000,000 independent Lambert solves | 82.8 s (24.2k/s) | 17.7 s (112.8k/s) | 4.7x |
| 90,000-cell Earth-Mars porkchop grid | -- | 1.08 s (83.2k solves/s) | -- |

Speedup grows with batch size (GPU kernel-launch overhead dominates at small
N; at 200k it was ~2.5x, at 2M it's 4.7x) -- porkchop grids and wide
population-based optimizer runs are exactly the "large N, no data
dependencies" shape that benefits.

**pygmo gotcha that cost real debugging time:** `bfe` must be attached to the
*raw* UDA (`uda.set_bfe(bfe)`) *before* wrapping it in `pg.algorithm(uda)`.
`pg.algorithm` never exposes `set_bfe` itself, and calling it on the wrapped
object either raises or (if swallowed) silently falls back to evaluating one
candidate at a time -- ~15x slower, with no error to indicate why. Also, only
some algorithms support it at all: `pg.pso_gen`, `pg.cmaes`, `pg.nsga2` do;
`pg.sga`, `pg.sade`, `pg.de` do not (pygmo 2.19.7). `orbitopt.optimize.runner`
handles this correctly; see its docstring.

**SPICE frame gotcha:** SPICE's `"J2000"` (mean equator/equinox) and
`"ECLIPJ2000"` (ecliptic-and-equinox) frames are related by a ~23.4-degree
rotation (Earth's obliquity) and are NOT interchangeable -- mixing states
queried in one against the other silently produces errors of hundreds of
thousands of km, not a small correction. Every module that touches
Earth/Moon/Sun ephemerides in this codebase uses `ECLIPJ2000` consistently;
if you add a new one, match it.

**RK4 step-size gotcha for close encounters:** a fixed step size that's
perfectly fine for a multi-day interplanetary coast (300-600s) is NOT fine
near a close lunar flyby -- an empirical convergence check found step_size=600s
reporting an 81,733 km closest approach for a trajectory whose true
(step_size<=15s converged) closest approach was 4,379 km. Local trajectory
curvature near a several-thousand-km-altitude flyby is far sharper than
during a plain coast; `verify.differential_correction` defaults to a 60s
step for exactly this reason. If you extend this code to closer flybys
(hundreds of km altitude or less), re-run the convergence check -- 60s may
no longer be fine enough.

**GPU is not always faster -- measured, not assumed:** `nbody_gpu.propagate_spacecraft_batch`
is GPU-*slower* than CPU below roughly N~20,000 candidates (N=2000, n_steps=300:
CPU 20.2s vs GPU 107.8s), the opposite of the Lambert solver's crossover.
The RK4 wrapper issues 4*n_steps small CuPy kernel launches per call (one
Newton-solved Moon/Sun position lookup per RK4 substage), and per-launch
overhead dominates until the batch is large enough to amortize it -- unlike
`lambert.gpu_batch`, which has no per-step Python loop calling it repeatedly.
`problems/free_return.py` and `examples/05_artemis2_free_return.py` default
to `use_gpu=False` for this reason; don't assume GPU is the right choice for
a new batched routine without measuring at your actual batch size.

**Undamped Newton diverges on real (imprecise) inputs, not just in theory:**
wiring the full pipeline together end-to-end -- not just testing each stage
in isolation -- surfaced a case where a GPU-coarse-screened TLI guess, only
slightly different from a hand-verified working one, sent
`differential_correction.target_lunar_flyby`'s plain Newton step into a
runaway divergence (residual growing to tens of millions of km within a few
iterations). Root cause: the coarse two-body Moon model's own
closest-approach-time estimate was off by more than two days for that
candidate, which anchored the fixed-time targeter to a time with no nearby
feasible solution. Fixed with two changes that are cheap to skip and easy to
regret skipping: (1) sanity-check the refined coast duration against the
caller's own guess before trusting it, falling back to the guess if the
"closest approach" found is implausibly far away in time or distance; (2)
cap and backtrack the Newton step (classic damped Newton) instead of always
taking the full linearized step. Both are exercised by
`tests/test_cislunar.py::test_target_lunar_flyby_converges_from_an_imprecise_gpu_screened_guess`,
using the exact delta-v that used to diverge -- a reminder that a solver
validated only on its own best-case hand-picked input isn't validated for
what an upstream (imprecise, automated) stage will actually hand it.

## Examples

- `examples/01_earth_mars_lambert_gpu.py` -- optimize (t0, tof) for an
  Earth->Mars transfer with pygmo PSO, every generation GPU-batched.
- `examples/02_mga_cassini_like.py` -- GPU-screen Earth->Venus launch windows,
  then optimize a 5-body MGA-1DSM sequence (pykep.trajopt.mga_1dsm) seeded
  from the screening result.
- `examples/03_verify_with_tudat.py` -- propagate a pykep Lambert solution
  through tudatpy's N-body dynamics and report the drift vs. the two-body
  prediction.
- `examples/04_porkchop_gpu.py` -- generate and plot a full Earth->Mars
  porkchop grid from one batched GPU call.
- `examples/05_artemis2_free_return.py` -- Lambert-seeded, GPU-screened,
  tudatpy-refined Earth-Moon free-return trajectory targeting the real
  Artemis II perilune altitude (see the "Cislunar / Artemis II" section
  below for what this does and does not claim to reproduce).
- `examples/06_solar_system_explorer_data.py` -- regenerate the solar-system
  SceneData JSON (only needed if you want to cache it; the viewer below can
  also compute it live).
- `examples/07_mission_timeline_data.py` -- regenerate the Artemis II
  mission SceneData JSON, same caveat.
- `examples/08_pyvista_viewer.py` -- open the interactive 3D viewer (see
  below) on either built-in scenario or any scene JSON file.

## Interactive 3D viewer (PyVista)

```
python examples/08_pyvista_viewer.py solar-system
python examples/08_pyvista_viewer.py artemis2
python examples/08_pyvista_viewer.py path/to/some_scene.json
```

A real desktop window (PyVista/VTK -- no browser, no HTML/CSS/JS anywhere
in this package): drag to orbit the camera, scroll/pinch to zoom, click a
body for its info panel. Scenes with a `timeline` (currently just the
Artemis II mission) get a scrubber slider at the bottom -- dragging it
moves the spacecraft/Moon to their position at that mission time, redraws
the "traveled so far" trail, and updates the live distance readout.

Every scene the viewer can open is the same `SceneData` dict (see
`orbitopt/viz/scene.py`): `{title, distanceUnit, bodies: [{id, name, color,
kind, radiusDisplay, orbit?, position?, trail?, info?}], timeline?}`. The
viewer (`orbitopt/viz/pv_viewer.py`) has exactly one render path for that
shape -- adding a new scene (a Lambert transfer, an MGA sequence, anything
else this framework computes) means writing another small exporter that
builds a `SceneData` document through `viz.scene`'s helpers, not touching
the viewer. `load_scene(path)` reads one from disk; `show_scene(scene)`
opens it -- both are plain functions you can call from a script or a
notebook, e.g.:

```python
from orbitopt.viz.pv_viewer import show_scene
from orbitopt.viz.solar_system import export_solar_system_data
show_scene(export_solar_system_data())
```

**Marker sizing, and a bug this sidesteps entirely:** body markers render
via VTK's `render_points_as_spheres` point rendering, which draws at a
constant *screen-pixel* size regardless of camera distance. An earlier,
now-removed browser/Three.js iteration of this viewer hand-rolled that same
"constant apparent size" behavior (recomputing world-space marker scale
from distance-to-camera every frame) and got the scale factor wrong: the
Sun's marker stayed large enough in world-space to visually swallow
Mercury's entire orbit even fully zoomed in. Letting VTK's own point
rendering handle it avoids that whole class of bug for free -- one more
reason (on top of "no HTML/CSS/JS to hand-write") this landed on a
Python-native 3D toolkit instead of a from-scratch web renderer.

**Slider gotcha that cost real debugging time:** PyVista's
`add_slider_widget` defaults to `interaction_event='end'` -- the callback
only fires on mouse-*release*, not while dragging. A naive test that
manually invoked VTK's `'InteractionEvent'` to simulate a drag therefore
silently did nothing (right event name, wrong one for the widget's actual
default), which looked exactly like the scene simply not updating. Fixed
by passing `interaction_event='always'` explicitly, both for correctness
(this test) and because live feedback while dragging is the UX you
actually want from a timeline scrubber.  Covered by
`tests/test_pv_viewer.py::test_mission_timeline_slider_updates_spacecraft_position`.

## Cislunar / Artemis II free-return pipeline

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
comparable magnitudes, not a reproduction of the actual flown mission.
It also targets flyby *distance* only, in whatever direction the Lambert
seed happens to miss by; a genuine unpowered free return additionally needs
the B-plane crossing aimed so gravity alone bends the outbound trajectory
back through Earth's atmosphere, which is a 2-parameter aim-point targeting
problem this example doesn't attempt -- expect the example's post-flyby
trajectory to *not* re-enter on its own within the propagated window. Adding
that targeting (vary the B-plane aim point, not just distance, as the
Newton unknowns) is the natural next extension of `differential_correction.py`.

## Known limitations / extension points

- The batched Lambert solver handles 0-revolution transfers only (multi-rev
  falls back to `orbitopt.lambert.cpu.solve_lambert_single`, which wraps
  pykep's full Izzo solver).
- `verify.tudat_propagate` uses point-mass gravity only; extending to
  spherical harmonics / SRP / drag is a matter of adding
  `propagation_setup.acceleration` terms.
- Low-thrust (continuous-thrust) trajectories aren't modelled yet; pykep's
  `sims_flanagan`/shape-based modules would plug in the same way `mga.py`
  wraps `mga_1dsm`.
- `dynamics.nbody_gpu`'s Moon/Sun ephemeris is a pure two-body osculating
  orbit seeded once from a real SPICE state at J2000 -- accurate to a few
  thousand km over a several-day coast (validated), but not re-seeded for
  epochs far from J2000 or spans much longer than ~10 days.
- The free-return targeter is distance-only / fixed-time (see above); no
  B-plane aim-point control, no multi-point (outbound + return) targeting,
  no trajectory-correction-burn modelling.
