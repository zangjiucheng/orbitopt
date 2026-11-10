# CUDA regression testing on the local GPU server

The default CI (`.github/workflows/ci.yml`) and the development platform
have no CUDA hardware. On those machines every `gpu`-marked test is
**skipped**. The CUDA path is checked by hand on the project's local GPU
server, using the procedure below. Each run produces a record in
[`runs/`](runs/), which is committed.

Acceptance (task #165): a complete record exists for commit `afa0cd9` or a
later one, and every failure in it is attributed.

## Scope

| Area | CUDA cases | CPU reference (same code, numpy) |
|---|---|---|
| Lambert | `tests/test_gpu_lambert.py::test_batch_matches_pykep_reference[True]`, `::test_cpu_and_gpu_backends_agree` | rest of `tests/test_gpu_lambert.py` |
| N-body | `tests/test_nbody_gpu_backend.py`: `test_kepler_batch_*`, `test_moon_and_sun_state_*`, `test_nbody_history_cpu_and_gpu_agree[*]` | `tests/test_nbody_ephemeris_precompute.py` |
| Closest approach | `tests/test_nbody_gpu_backend.py`: `test_closest_approach_cpu_and_gpu_agree[*]`, `test_gpu_closest_approach_matches_gpu_history`, `test_fused_cupy_raw_*`, `test_fused_numba_cuda_*` | `tests/test_nbody_closest_approach.py`, `tests/test_nbody_fused.py` |
| Device reporting | `tests/test_runner_device.py::test_cupy_backend_reports_gpu` | rest of `tests/test_runner_device.py` |

The N-body cases and the closest-approach cases compare `use_gpu=True`
against `use_gpu=False`, and compare the fused `cupy-raw`/`numba-cuda`
kernels against the numpy production path. The tolerance is about 1e-9
relative. The closest-approach step index must match exactly. The SPICE
seed states are injected into the cache, so no kernels are needed.

## Procedure

```bash
git fetch && git checkout <commit to test>        # afa0cd9 or later
micromamba create -n orbitopt-gpu -f envs/gpu.conda-lock.yml   # once; see envs/README.md
micromamba activate orbitopt-gpu
python -m pip install -e . --no-deps
nvidia-smi                                         # sanity: driver loaded, GPU visible

python scripts/gpu_regression.py                   # writes docs/gpu-regression/runs/<UTC stamp>-<sha7>.*
```

The script runs two pytest passes over the scope above:

1. `cuda`: `-m gpu` with `ORBITOPT_REQUIRE_GPU=1`. If no device is visible,
   this pass is a hard error (exit 2), not a run where every case skipped.
2. `cpu-reference`: `-m "not gpu"`, the numpy reference for the same areas.

The script writes these files:

| File | Content |
|---|---|
| `<stem>.md` | **the record**: environment, per-case result and duration, failure logs, attribution placeholders |
| `<stem>.env.json` | the same environment data, machine-readable |
| `<stem>-cuda.log`, `<stem>-cpu-reference.log` | full pytest output |
| `<stem>-*.junit.xml` | JUnit XML, the source of the per-case timings |

The record captures the following, as the task requires:

- GPU model (from `nvidia-smi` and from CuPy device 0, with compute capability)
- driver version, and CUDA versions: the driver-supported version from `nvidia-smi`, plus the CUDA driver API and runtime versions reported by CuPy
- CuPy version (also numba, numpy and Python)
- the commit hash tested, with a flag if the work tree was dirty
- every case with its result and duration
- the failure log of every failed or errored case

Extra pytest arguments go after `--`, for example
`python scripts/gpu_regression.py -- -x --maxfail=3`.

## Attributing failures

Every failed or errored case in the record has an `**Attribution:** _TODO_`
line. For CUDA cases, the line also shows the outcome of the matching CPU
reference test. Replace `_TODO_` with one of these causes and give the
reason:

| Attribution | Typical evidence |
|---|---|
| **GPU backend divergence** | the CUDA case fails, the CPU reference passes, and the mismatch is far above the documented tolerance |
| **Environment / driver** | CuPy import or `cudaErrorInsufficientDriver`, NVRTC/JIT compile errors, out of memory, or a numba-cuda toolkit mismatch. Re-run after fixing the host |
| **Test tolerance** | the mismatch is only slightly above the tolerance and comes from FMA or `libdevice` rounding. Show the observed error |
| **Not GPU-specific** | the CPU reference fails the same way. Track it as a normal bug |

Open an issue for every divergence and every regression that is not
GPU-specific, and link it from the record.

## Committing the record

Commit `<stem>.md` and `<stem>.env.json`, plus the `.log` files when there
were failures. Then add a row to the run index below.

## Run index

| Date (UTC) | Commit | GPU | CUDA / CuPy | Result | Record |
|---|---|---|---|---|---|
| _pending: first run on the local GPU server_ | | | | | |

## Automation (optional, lower priority)

`.github/workflows/gpu.yml` runs the same script on a self-hosted runner
with the labels `[self-hosted, linux, x64, gpu]`. You can start it manually
(Actions → GPU CI → Run workflow) or by adding the `gpu-ci` label to a PR.
The record directory is uploaded as the `gpu-regression-record` artifact.
Until a runner is registered, use the manual procedure above. For runner
registration, see GitHub's *Adding self-hosted runners* guide, and add the
custom label `gpu`.
