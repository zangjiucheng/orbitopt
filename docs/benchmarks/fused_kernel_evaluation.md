# Fused CUDA kernel evaluation and CPU/GPU benchmark (task #143)

This study compared closest-approach screening backends after ephemeris
precomputation. It recommended a fused Numba CPU backend and deferred a fused
CUDA implementation. The measurements below are specific to the stated hosts;
no CUDA timing was measured in this study.

## Reproduce

```bash
pip install numba jax  # optional; absent backends are skipped
PYTHONPATH=src python scripts/bench_nbody_backends.py --n 2000 --steps 300 --json out.json
PYTHONPATH=src python scripts/bench_nbody_backends.py --n 2000 --steps 300 --sun
PYTHONPATH=src python scripts/bench_nbody_backends.py --n 20000 --steps 300
```

Inputs use a fixed seed and cached Moon/Sun states. Timings are medians after
warm-up; the first call includes JIT compilation. CUDA backends are measured
only when a device is present. See `tests/test_nbody_fused.py` for agreement
checks and CUDA-simulator coverage.

## Results (CPU-only host)

Intel i5-8250U, 4C/8T, Python 3.13, NumPy 2.x, Numba 0.68.1 and JAX 0.12;
run-to-run timings varied about ±30%. Raw results are in
`nbody_backends_cpu_*.json` in this directory.

| Candidates × steps | Sun | NumPy | Numba CPU | JAX CPU |
|---|---|---|---|---|
| 2,000 × 300 | no | 0.89 s | 0.044 s | 0.21 s |
| 2,000 × 300 | yes | 1.50 s | 0.059 s | 0.28 s |
| 20,000 × 300 | no | 7.61 s | 0.27 s | 1.47 s |

For 2,000 candidates, Numba spent about 0.022 s in host preparation and
0.019 s in the fused loop. The median relative error against NumPy was about
1e-15; larger differences mostly occurred on trajectories passing through
Earth, close to the singularity.

## Profile: where the GPU time went, and where it would go now

The vectorized 2,000 × 300 case issued an estimated 4.5 million array
operations before ephemeris precomputation and 24,762 afterward (34,514 with
the Sun). A fused kernel uses one launch plus preparation and transfers.
Extrapolating the earlier launch cost suggests roughly 0.6 s for the updated
CuPy path; **this is an estimate, not a GPU measurement**.

## Backend comparison

Numba CUDA shares the CPU implementation and supports simulator tests; CuPy
RawKernel requires separate CUDA C. JAX was slower than fused Numba on this CPU
and adds a larger dependency. Re-measure on the target GPU before choosing a
GPU backend.

## Conclusion

At the measured batch sizes, CPU fusion already removed most overhead and host
preparation limited potential GPU savings. Revisit CUDA if batches regularly
exceed about 100,000 candidates or 2,000 steps, or if ephemeris preparation
moves to the device.

## Production integration (task #164)

`FreeReturnScreeningProblem` supports `auto`, `numpy`, `numba-cpu` and `cupy`.
`auto` selects CuPy when GPU use is explicitly requested; otherwise it can use
Numba for closest-approach batches of at least 64, falling back to NumPy when
appropriate. Numba's cached kernel avoids recompilation in later processes.
`OptimizationResult.backend` and `backend_usage` report the backend used.

### End-to-end timing

```bash
PYTHONPATH=src python scripts/bench_free_return_backends.py --json out.json
PYTHONPATH=src python scripts/bench_free_return_backends.py --pop 2000 --generations 5 \
  --repeat 1 --backends numpy,numba-cpu
```

Warm runs on an 8-logical-CPU host, 300 steps, six-day coast, no Sun; raw JSON
is in `free_return_backends_cpu_*.json`.

| Population × generations | NumPy PSO | Numba CPU PSO | Speedup |
|---|---|---|---|
| 256 × 25 | 3.6 s | 0.21–0.23 s | ~16× |
| 2,000 × 5 | 5.2 s | 0.21 s | ~24× |

A new process paid about 1 s for a warm JIT cache or 9 s for a cold compile.

### Fitness agreement

Above-surface candidates agreed with NumPy to 0.015 km or better in these
batches. Below-surface trajectories diverged near the gravity singularity;
this can change PSO tie-breaking and the final champion. The backend regression
tests use relative tolerance 1e-9; they do not make an accuracy claim for
Earth-intersecting trajectories.
