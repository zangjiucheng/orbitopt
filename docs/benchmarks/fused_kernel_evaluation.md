# Fused CUDA kernel evaluation and CPU/GPU benchmark (task #143)

**Question.** `nbody_gpu.propagate_spacecraft_batch` was ~5.3x *slower* on
GPU (CuPy) than on CPU at 2,000 candidates x 300 steps (107.8 s vs 20.2 s),
because Python issued a very large number of small kernels. After the
ephemeris precompute (#139) and the closest-approach-only output (#140) have
landed, is it worth rewriting the RK4 + closest-approach loop as one fused
kernel? Candidates: CuPy `RawKernel`, Numba CUDA, JAX JIT.

**Decision: do not write a CUDA fused kernel now.** Fusing does pay off
(18-35x), but it pays off on the CPU already: a Numba-CPU build of the same
fused loop takes 0.04 s at N=2000, and half of that is host-side Moon-grid
prep that a GPU kernel would not remove. The follow-up worth doing is an
optional `numba-cpu` backend for the closest-approach path (details below).
Revisit CUDA only on the trigger listed below.

## Reproduce

```bash
pip install numba jax          # optional; missing backends are reported as "skipped"
PYTHONPATH=src python scripts/bench_nbody_backends.py --n 2000 --steps 300 --json out.json
PYTHONPATH=src python scripts/bench_nbody_backends.py --n 2000 --steps 300 --sun
PYTHONPATH=src python scripts/bench_nbody_backends.py --n 20000 --steps 300
```

Inputs are deterministic: a fixed RNG seed, and the Moon/Sun seed states are
injected into the cache so SPICE is not needed. Each timing is the median of
`--repeat` calls after one warm-up call; the warm-up, which includes JIT
compile time, is reported as `first s`. On a CUDA host the same command also
times `cupy` (the production vectorized path), `numba-cuda` and `cupy-raw`.
Correctness of every fused backend against the production path is checked
by `tests/test_nbody_fused.py`. Numba CUDA runs there on the CUDA simulator.

The prototypes are in `src/orbitopt/dynamics/nbody_fused.py`. One scalar
per-candidate routine is compiled as plain Python, `numba.njit(parallel)`
and `numba.cuda.jit`. The module also holds a CUDA C `RawKernel` of the same
algorithm and a JAX `lax.fori_loop` version. Only `numba-cpu` is wired into
a problem class (task #164, see "Production integration" at the end).

## Results (CPU-only host)

Host: Intel i5-8250U, 4C/8T, Python 3.13, numpy 2.x, numba 0.68.1, jax
0.12 (CPU). No CUDA device, so `cupy`, `numba-cuda` and `cupy-raw` were
skipped. Raw JSON: `nbody_backends_cpu_*.json` next to this file. Wall
times on this shared machine vary by about ±30% between runs.

| N x steps | Sun | numpy (prod) | numba-cpu fused | jax fused (CPU) |
|---|---|---|---|---|
| 2,000 x 300  | no  | 0.89 s | **0.044 s** (20x) | 0.21 s (4.2x) |
| 2,000 x 300  | yes | 1.50 s | **0.059 s** (25x) | 0.28 s (5.4x) |
| 20,000 x 300 | no  | 7.61 s | **0.27 s** (29x)  | 1.47 s (5.2x) |

Numba split at N=2000 (median of 7): host prep (`prepare_inputs`, which
includes the #139 Moon-grid Kepler solve in numpy) takes 0.022 s and the
fused kernel takes 0.019 s. At N=20000 they are 0.033 s and 0.146 s. JIT
compile cost on the first call is about 5 s for Numba and 0.7-2.5 s for JAX.

**Accuracy.** The median relative error against the production path is
about 1e-15. The 27-305 candidates per run that differ by more than 1e-6
are, with 1-2 exceptions per run, trajectories that pass *through* the
Earth (|r| < 6,378 km). Round-off grows without bound at that singularity,
and those candidates are infeasible anyway. A different operation order
(`dn*dn*dn` vs `dn**3`) is enough to diverge them. Among the remaining
candidates, the closest-approach step agrees everywhere except 0-6 near-ties.

## Profile: where the GPU time went, and where it would go now

The benchmark also counts the array operations the vectorized path issues.
It runs the numpy code on an op-counting `ndarray` subclass. On CuPy, each
elementwise ufunc, `where` or reduction is at least one kernel launch.

| Path | Array ops (≈ CuPy launches) per call, N=2000, 300 steps |
|---|---|
| Before #139: Kepler solve inside every RK4 stage | ~3,750 per solve x 4 x 300 ≈ **4.5 M** |
| After #139/#140 (no Sun) | **24,762** = 70 per RK4 step + 3,762 for the one grid solve |
| After #139/#140 (with Sun) | 34,514 = 90 per step + 7,514 |
| Fused kernel | **1** (plus host prep and H2D/D2H copies) |

The old measurement fits a launch-bound model: 107.8 s / 4.5 M launches ≈
24 µs per launch. At 2,000 candidates each kernel touches only about 48 KB,
so that cost is all Python and launch overhead, independent of N. Under the
same model:

- **CuPy vectorized, today:** 24.8 k launches x ~24 µs ≈ **0.6 s** at
  N=2000. That is about the same as numpy (0.8-0.9 s). The GPU stays
  launch-bound until per-kernel work dominates, somewhere around N ≈ 10^5.
  This is an estimate. Confirm it on a CUDA host with the same script before
  changing the `use_gpu=False` defaults (see `docs/engineering-notes.md`).
- **Fused GPU kernel, best case:** the arithmetic is about 300 x 4 x ~40
  FP64 flops x 2000 ≈ 0.1 GFLOP, under 1 ms on any recent GPU. The call
  would still pay the ~0.02 s host prep plus copies. Compared with
  Numba-CPU fused (0.044 s), the most a CUDA kernel could save is about
  0.02 s per 2,000-candidate batch, or about 0.12 s at N=20,000.

## Backend comparison

| | CuPy RawKernel | Numba CUDA | JAX JIT |
|---|---|---|---|
| Source | separate CUDA C string | **same Python source as the CPU/reference path** | separate vectorized version |
| Testable without a GPU | no | **yes** (`NUMBA_ENABLE_CUDASIM=1`, covered by tests) | yes (CPU XLA) |
| Measured here | n/a | n/a (sim only) | 4-5x slower than Numba-CPU fused |
| Extra dependency | CuPy + NVRTC (already optional) | numba (pip, ~4 MB) | jax + jaxlib (large) and the process-global `jax_enable_x64` flag |
| Fusion | full, manual | full, manual | XLA fuses each step's elementwise ops but keeps arrays in memory between ops |

If a GPU kernel is ever needed, **Numba CUDA** is the one to pick. It shares
its source with the CPU build, so the two cannot drift apart. It is
CI-testable through the simulator and needs no CUDA C maintenance. JAX
loses on CPU throughput, dependency weight and the global x64 switch.

## Conclusion

**Do not do now:** a CUDA fused kernel (RawKernel, Numba CUDA or JAX on
GPU).

- After #139 and #140, the remaining cost of the vectorized path is about
  25 k small array ops. That is interpreter and launch overhead, which hurts
  CPU and GPU alike.
- Fusing the loop removes it, and on the CPU that already brings the
  2,000 x 300 case down to about 0.04 s. Half of that is host-side grid prep
  that a GPU kernel would not touch.
- A CUDA fused kernel could save at most about 0.02 s per batch here. It
  would add a code path that CI cannot run on real hardware.

**Do next (separate task):** add an optional `numba-cpu` fused backend to
the closest-approach path used by `FreeReturnScreeningProblem`.

- Fall back to the current numpy path when numba is not installed.
- Cache the JIT (`cache=True`) to avoid the ~5 s first-call compile.
- Expected speedup: ~20-30x per fitness batch.
- The prototype and its equivalence tests in this task are the starting
  point.

**Revisit the CUDA kernel if** a single batch regularly exceeds about 10^5
candidates or about 2,000 steps, or if the Moon grid moves to the device.
In those cases fused-kernel compute becomes the dominant term. Port the
existing `numba-cuda` prototype and re-run this benchmark on a CUDA host.

## Production integration (task #164)

`FreeReturnScreeningProblem` now takes
`backend="auto"|"numpy"|"numba-cpu"|"cupy"` (default `auto`):

- `auto` picks `cupy` only when `use_gpu=True` is passed explicitly. Otherwise
  it picks `numba-cpu` when numba is installed, the output mode is
  `closest_approach`, and the batch has at least `AUTO_NUMBA_MIN_BATCH = 64`
  candidates (or the kernel is already loaded in the process). In every other
  case it picks `numpy`. A numba run-time failure under `auto` raises a
  warning and falls back to `numpy`.
- The numba-cpu kernel is compiled with `cache=True`. It lives at module level
  in `dynamics/_nbody_fused_numba.py` because a kernel built inside a
  closure missed the on-disk cache in every new process. Measured cold
  compile is ~9 s; loading from the warm cache takes ~1 s.
- `run_optimization()` returns `OptimizationResult.backend` (the backend that
  evaluated the most candidates) and `.backend_usage` (candidates per
  backend). The verbose line prints the backend too. pygmo deep-copies the
  UDP, so the per-backend tally (`core.problem.BackendUsage`) is shared
  across copies.

### End-to-end timing

```bash
PYTHONPATH=src python scripts/bench_free_return_backends.py --json out.json        # 256 x 25 gens
PYTHONPATH=src python scripts/bench_free_return_backends.py --pop 2000 --generations 5 \
    --repeat 1 --backends numpy,numba-cpu
```

The host has 8 logical CPUs, Python 3.13, numpy 2.5 and numba 0.68.1, and no
CUDA device. Each figure is a median of warm runs, 300 steps, 6-day coast, no
Sun. `run_optimization` is the full PSO run through the batch evaluator,
pygmo overhead included. Raw JSON is in `free_return_backends_cpu_*.json`.

| pop x gens | numpy `batch_fitness` | numba-cpu `batch_fitness` | numpy `run_optimization` | numba-cpu `run_optimization` |
|---|---|---|---|---|
| 256 x 25  | 0.18 s | 0.006-0.012 s (15-33x) | 3.6 s | 0.21-0.23 s (**~16x**) |
| 2,000 x 5 | 0.62 s | 0.027 s (23x)          | 5.2 s | 0.21 s (**~24x**) |

End to end, the whole screening run is ~16x faster at the default-like
256-candidate population and ~24x faster at 2,000 candidates. The 20-30x
figure above measures only the N-body closest-approach kernel. The pipeline
gains less at small populations because the per-call Moon-grid precompute
and pygmo's own overhead do not shrink. A new process also pays the
one-time kernel load once: ~1 s from cache, ~9 s on a cold cache.

### Fitness agreement

- `tests/test_free_return_backends.py` checks `numba-cpu`, `auto`, the
  history path and (on a GPU host) `cupy` against `numpy` at rtol 1e-9.
- In the benchmark batches, every candidate whose trajectory stays above
  Earth's surface agrees with numpy: to 0 km (bit-identical) at 256
  candidates, and to <= 0.015 km at 2,000.
- Candidates whose coarse trajectory passes *below* Earth's surface
  (perigee < 6,378 km: 61 of 256 and 477 of 2,000) diverge by up to
  0.74 km and 45 km respectively, out of fitness values of ~10^5 km. They
  pass near the 1/r^3 singularity, where step-level round-off differences
  are amplified chaotically. Neither backend is "right" for them, and they
  are far from any optimum.
- A PSO run therefore follows the same path only until such a candidate
  breaks a tie. At 256 x 25 both backends converged to the same champion
  (330.48 km). At 2,000 x 5 they diverged (81.6 km vs 88.5 km), which is an
  artifact of the swarm's path, not a fitness bias.
