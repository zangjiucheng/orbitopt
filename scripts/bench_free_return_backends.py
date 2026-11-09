"""Task #164: end-to-end timing of FreeReturnScreeningProblem per backend.

    PYTHONPATH=src python scripts/bench_free_return_backends.py \
        [--pop 256] [--generations 25] [--steps 300] [--repeat 3] [--json out.json]

Unlike scripts/bench_nbody_backends.py (task #143, the N-body closest-
approach kernel alone), this times the whole screening pipeline the problem
class runs in production:

- ``batch_fitness``: one ``FreeReturnScreeningProblem.batch_fitness()`` call
  of ``--pop`` candidates (decision-vector reshape, Moon grid precompute,
  propagation, objective) -- median of ``--repeat`` warm calls.
- ``run_optimization``: a full ``orbitopt.optimize.runner.run_optimization``
  PSO run (``--pop`` x ``--generations``, pygmo overhead included) --
  median of ``--repeat`` warm runs.

Each backend is warmed up once first; that call (JIT compile / numba cache
load for numba-cpu) is reported separately as ``first_call_s``. Missing
backends are reported as ``skipped``. Fitness agreement against numpy is
checked on the same batch, split by whether the candidate's coarse
trajectory dips below Earth's surface (perigee < 6,378 km): those
unphysical candidates pass near the 1/r^3 singularity, where round-off is
amplified chaotically, so they are reported separately. Inputs are deterministic and the Moon seed state
is injected into nbody_gpu's cache, so no SPICE kernels are needed.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time

import numpy as np

from orbitopt.core import gpu as gpu_mod
from orbitopt.dynamics import nbody_fused as nf
from orbitopt.dynamics import nbody_gpu as nb
from orbitopt.problems.free_return import BACKENDS, FreeReturnScreeningProblem

EPOCH0 = 838_000_000.0
MOON_STATE0 = (np.array([-291_608.0, -266_716.0, 22_850.0]), np.array([0.6439, -0.6974, 0.0136]))


def make_problem(backend, n_steps):
    r0 = np.array([6_563.0, 0.0, 0.0])
    v0 = np.array([0.0, np.sqrt(nb.MU_EARTH_KM3_S2 / 6_563.0), 0.0])
    return FreeReturnScreeningProblem(
        r0, v0, EPOCH0, target_distance_km=8_000.0, coast_days=6.0, n_steps=n_steps,
        dv_bounds_km_s=([-0.6, 2.5, -0.6], [0.6, 3.7, 0.6]), backend=backend,
    )


def make_dvs(n, seed=0):
    dvs = np.random.default_rng(seed).normal(0.0, 0.6, size=(n, 3))
    dvs[:, 1] += 3.1
    return dvs.ravel()


def _median_time(fn, repeat):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


EARTH_RADIUS_KM = 6_378.137


def crashing_mask(n_steps, dvs):
    """True for candidates whose numpy history dips below Earth's surface."""
    p = make_problem("numpy", n_steps)
    v0 = p.v0_pre_burn_km_s + dvs.reshape(-1, 3)
    r0 = np.broadcast_to(p.r0_km, v0.shape)
    h = nb.propagate_spacecraft_batch(
        r0, v0, p.coast_seconds, n_steps, nb.MU_EARTH_KM3_S2, nb.MU_MOON_KM3_S2,
        epoch0=EPOCH0, use_gpu=False, output="history",
    )
    return np.linalg.norm(h.r, axis=-1).min(axis=0) < EARTH_RADIUS_KM


def bench_backend(backend, args, ref_fitness, crash):
    if backend == "numba-cpu" and not nf.numba_available():
        return {"skipped": "numba not installed"}
    if backend == "cupy" and not gpu_mod.GPU_AVAILABLE:
        return {"skipped": "CuPy/CUDA unavailable"}

    from orbitopt.optimize.runner import run_optimization

    problem = make_problem(backend, args.steps)
    dvs = make_dvs(args.pop)

    t0 = time.perf_counter()
    fitness = problem.batch_fitness(dvs)
    first = time.perf_counter() - t0

    out = {"first_call_s": first, "resolved_backend": problem.last_backend}
    if ref_fitness is not None:
        diff = np.abs(fitness - ref_fitness)
        out["max_abs_diff_vs_numpy_km"] = float(diff[~crash].max(initial=0.0))
        out["max_abs_diff_vs_numpy_km_subsurface"] = float(diff[crash].max(initial=0.0))
        out["n_subsurface_candidates"] = int(crash.sum())
    out["batch_fitness_s"] = _median_time(lambda: problem.batch_fitness(dvs), args.repeat)

    def opt():
        return run_optimization(problem, pop_size=args.pop, generations=args.generations,
                                seed=1, verbose=False)

    res = opt()
    out["run_optimization_backend"] = res.backend
    out["run_optimization_backend_usage"] = res.backend_usage
    out["run_optimization_champion_f"] = float(res.champion_f[0])
    out["run_optimization_s"] = _median_time(opt, args.repeat)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pop", type=int, default=256)
    ap.add_argument("--generations", type=int, default=25)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--backends", default=",".join(BACKENDS))
    ap.add_argument("--json", help="write full results to this JSON file")
    args = ap.parse_args(argv)

    nb._MOON_STATE0_KM[round(EPOCH0, 3)] = MOON_STATE0

    ref = make_problem("numpy", args.steps).batch_fitness(make_dvs(args.pop))
    crash = crashing_mask(args.steps, make_dvs(args.pop))
    results = {}
    for backend in args.backends.split(","):
        results[backend] = bench_backend(backend, args, ref, crash)
        print(f"{backend:10s} {json.dumps(results[backend])}", flush=True)

    base = results.get("numpy", {})
    for name, r in results.items():
        if "skipped" in r or "batch_fitness_s" not in base:
            continue
        r["batch_fitness_speedup_vs_numpy"] = base["batch_fitness_s"] / r["batch_fitness_s"]
        r["run_optimization_speedup_vs_numpy"] = base["run_optimization_s"] / r["run_optimization_s"]
        print(f"{name:10s} batch_fitness x{r['batch_fitness_speedup_vs_numpy']:.1f}, "
              f"run_optimization x{r['run_optimization_speedup_vs_numpy']:.1f} vs numpy")

    if args.json:
        meta = {
            "pop": args.pop, "generations": args.generations, "steps": args.steps,
            "repeat": args.repeat, "python": platform.python_version(),
            "numpy": np.__version__, "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "gpu": gpu_mod.device_info(),
        }
        try:
            import numba

            meta["numba"] = numba.__version__
            meta["numba_threads"] = numba.get_num_threads()
        except Exception:
            pass
        with open(args.json, "w") as fh:
            json.dump({"meta": meta, "results": results}, fh, indent=2)


if __name__ == "__main__":
    main()
