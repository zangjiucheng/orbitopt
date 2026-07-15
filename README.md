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
  bodies.py            pykep.planet ephemeris + epoch helpers
  problems/
    transfer_2body.py  single Lambert-leg transfer UDP, with a real batch_fitness()
    mga.py             wraps pykep.trajopt.mga_1dsm; GPU-screens launch windows first
  optimize/
    gpu_bfe.py         custom pygmo UDBFE that dispatches to a problem's batch_fitness()
    runner.py          drives pygmo algorithms (pso_gen/cmaes/nsga2) through the GPU bfe
  verify/
    tudat_propagate.py numerically propagate a candidate leg and diff vs. the
                        pykep patched-conic prediction
  viz/
    porkchop.py        GPU-batched porkchop grid + plotting
```

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
