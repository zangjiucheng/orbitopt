"""Task #143: reproducible CPU/GPU benchmark + launch profile of the N-body
closest-approach propagation (``nbody_gpu.propagate_spacecraft_batch(...,
output="closest_approach")``) against the fused-kernel prototypes in
``orbitopt.dynamics.nbody_fused``.

    PYTHONPATH=src python scripts/bench_nbody_backends.py \
        [--n 2000] [--steps 300] [--sun] [--repeat 5] [--json out.json]

Inputs are deterministic (fixed RNG seed; the Moon/Sun seed states are
injected into nbody_gpu's cache, so no SPICE kernels are needed). Every
backend whose package/device is missing is reported as ``skipped`` instead
of failing, so the same command runs on a CPU-only box and on a CUDA host:

- numpy           production vectorized path, use_gpu=False (reference)
- cupy            production vectorized path, use_gpu=True
- numba-cpu       fused prototype, numba.njit(parallel=True)
- jax             fused prototype (XLA) on JAX's default device
- numba-cuda      fused prototype, numba.cuda.jit
- cupy-raw        fused prototype, hand-written CUDA C RawKernel

Each timing is the median wall time of ``--repeat`` calls *after* one
warm-up call (the warm-up, which includes JIT compilation, is reported
separately as ``first_call_s``). Every backend's result is checked against
the numpy reference.

The "profile" section counts the array operations the vectorized path issues
per call -- on CuPy each elementwise ufunc / ``where`` / reduction is (at
least) one CUDA kernel launch -- by running the numpy path on an
op-counting ndarray subclass, and splits numpy wall time into the Moon/Sun
grid precompute vs the RK4 loop.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import types

import numpy as np

from orbitopt.core import gpu as gpu_mod
from orbitopt.dynamics import nbody_fused as nf
from orbitopt.dynamics import nbody_gpu as nb

EPOCH0 = 838_000_000.0
MOON_STATE0 = (np.array([-291_608.0, -266_716.0, 22_850.0]), np.array([0.6439, -0.6974, 0.0136]))
SUN_STATE0 = (np.array([1.0e8, -1.1e8, 5.0e3]), np.array([22.0, 20.0, 0.0]))


def make_inputs(n, seed=0):
    rng = np.random.default_rng(seed)
    r_park = np.array([6_563.0, 0.0, 0.0])
    v_park = np.array([0.0, np.sqrt(nb.MU_EARTH_KM3_S2 / 6_563.0), 0.0])
    dvs = rng.normal(0.0, 0.6, size=(n, 3))
    dvs[:, 1] += 3.1
    # One shared epoch/coast time, like FreeReturnScreeningProblem.batch_fitness.
    return np.broadcast_to(r_park, (n, 3)).copy(), v_park + dvs, 6.0 * 86400.0


# ---------------------------------------------------------------------------
# Launch profile of the vectorized path
# ---------------------------------------------------------------------------

class _CountingArray(np.ndarray):
    counts = {"ufunc": 0, "where": 0}

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        _CountingArray.counts["ufunc"] += 1
        inputs = tuple(x.view(np.ndarray) if isinstance(x, _CountingArray) else x for x in inputs)
        if "out" in kwargs:
            kwargs["out"] = tuple(
                x.view(np.ndarray) if isinstance(x, _CountingArray) else x for x in kwargs["out"]
            )
        out = getattr(ufunc, method)(*inputs, **kwargs)
        if isinstance(out, np.ndarray):
            return out.view(_CountingArray)
        return out


def _counting_xp():
    """numpy stand-in whose array constructors return _CountingArray."""
    mod = types.SimpleNamespace(**{k: getattr(np, k) for k in dir(np) if not k.startswith("__")})

    def wrap(fn):
        return lambda *a, **k: np.asanyarray(fn(*a, **k)).view(_CountingArray)

    for name in ("asarray", "array", "full", "zeros", "empty", "broadcast_to", "atleast_1d",
                 "atleast_2d"):
        setattr(mod, name, wrap(getattr(np, name)))

    def where(*a):
        _CountingArray.counts["where"] += 1
        return np.where(*a).view(_CountingArray)

    mod.where = where
    return mod


def profile_vectorized(n, n_steps, include_sun):
    r0, v0, t_span = make_inputs(n)
    kw = dict(mu_earth=nb.MU_EARTH_KM3_S2, mu_moon=nb.MU_MOON_KM3_S2, include_sun=include_sun,
              epoch0=EPOCH0, output="closest_approach")

    def count(steps):
        xp = _counting_xp()
        orig = nb.get_array_module
        nb.get_array_module = lambda use_gpu=None: xp
        _CountingArray.counts = {"ufunc": 0, "where": 0}
        try:
            nb.propagate_spacecraft_batch(r0, v0, t_span, steps, **kw)
        finally:
            nb.get_array_module = orig
        return sum(_CountingArray.counts.values())

    # Two step counts -> per-step slope and fixed (precompute) intercept.
    small = 8
    c_small, c_big = count(small), count(n_steps)
    per_step = (c_big - c_small) / (n_steps - small)
    fixed = c_small - per_step * small

    # numpy wall-time split: grid precompute vs RK4 loop.
    dt = np.full(n, t_span / n_steps)
    ep = np.full(n, EPOCH0)
    t0 = time.perf_counter()
    nb._precompute_body_grid(ep, dt, n_steps, np, nb.MU_EARTH_KM3_S2, nb.MU_MOON_KM3_S2,
                             include_sun, nb.MU_SUN_KM3_S2, 20, EPOCH0)
    t_grid = time.perf_counter() - t0
    t0 = time.perf_counter()
    nb.propagate_spacecraft_batch(r0, v0, t_span, n_steps, use_gpu=False, **kw)
    t_total = time.perf_counter() - t0
    return {
        "array_ops_per_call": c_big,
        "array_ops_per_rk4_step": per_step,
        "array_ops_fixed": fixed,
        "numpy_grid_precompute_s": t_grid,
        "numpy_total_s": t_total,
        "numpy_rk4_loop_s": t_total - t_grid,
        "numpy_us_per_array_op": 1e6 * (t_total - t_grid) / max(c_big - fixed, 1),
    }


# ---------------------------------------------------------------------------
# Timed backends
# ---------------------------------------------------------------------------

def _vectorized(use_gpu):
    def run(r0, v0, t_span, n_steps, include_sun):
        out = nb.propagate_spacecraft_batch(
            r0, v0, t_span, n_steps, nb.MU_EARTH_KM3_S2, nb.MU_MOON_KM3_S2,
            include_sun=include_sun, epoch0=EPOCH0, use_gpu=use_gpu, output="closest_approach",
        )
        return out
    return run


def _fused(backend):
    def run(r0, v0, t_span, n_steps, include_sun):
        return nf.propagate_closest_approach_fused(
            r0, v0, t_span, n_steps, nb.MU_EARTH_KM3_S2, nb.MU_MOON_KM3_S2,
            include_sun=include_sun, epoch0=EPOCH0, backend=backend,
        )
    return run


def _availability():
    avail = {"numpy": None, "numba-cpu": None, "jax": None}
    avail["cupy"] = None if gpu_mod.GPU_AVAILABLE else "no CuPy/CUDA device"
    avail["cupy-raw"] = avail["cupy"]
    try:
        import numba  # noqa: F401
    except ImportError:
        avail["numba-cpu"] = "numba not installed"
    try:
        import jax
        avail["jax"] = None
        avail["_jax_device"] = str(jax.devices()[0])
    except ImportError:
        avail["jax"] = "jax not installed"
    try:
        from numba import cuda
        avail["numba-cuda"] = None if cuda.is_available() else "no CUDA device for numba"
    except ImportError:
        avail["numba-cuda"] = "numba not installed"
    return avail


RUNNERS = {
    "numpy": _vectorized(False),
    "cupy": _vectorized(True),
    "numba-cpu": _fused("numba-cpu"),
    "jax": _fused("jax"),
    "numba-cuda": _fused("numba-cuda"),
    "cupy-raw": _fused("cupy-raw"),
}


def _rel(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim == 2:
        return np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1.0)
    return np.abs(a - b) / np.maximum(np.abs(b), 1.0)


def bench(n, n_steps, include_sun, repeat, backends):
    r0, v0, t_span = make_inputs(n)
    avail = _availability()
    ref = None
    rows = {}
    for name in backends:
        reason = avail.get(name)
        if reason:
            rows[name] = {"status": "skipped", "reason": reason}
            continue
        fn = RUNNERS[name]
        t0 = time.perf_counter()
        out = fn(r0, v0, t_span, n_steps, include_sun)
        first = time.perf_counter() - t0
        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            out = fn(r0, v0, t_span, n_steps, include_sun)
            times.append(time.perf_counter() - t0)
        if name == "numpy":
            ref = out
        row = {"status": "ok", "first_call_s": first, "median_s": statistics.median(times),
               "min_s": min(times)}
        if ref is not None:
            # Candidates whose trajectory passes through the Earth (|r| -> 0,
            # unphysical) amplify round-off without bound, so report the median
            # error plus how many candidates disagree by more than 1e-6.
            e = np.maximum(_rel(out.min_distance_km, ref.min_distance_km), _rel(out.r, ref.r))
            row["median_rel_err"] = float(np.median(e))
            row["n_rel_err_gt_1e-6"] = int(np.sum(e > 1e-6))
            row["closest_step_mismatches"] = int(
                np.sum(out.min_distance_step != ref.min_distance_step))
        rows[name] = row
    if ref is not None:
        base = rows["numpy"]["median_s"]
        for row in rows.values():
            if row["status"] == "ok":
                row["speedup_vs_numpy"] = base / row["median_s"]
    return rows, avail


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--sun", action="store_true", help="include the Sun third-body term")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--backends", default=",".join(RUNNERS))
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--json", help="write full results to this JSON file")
    args = ap.parse_args(argv)

    key = round(EPOCH0, 3)
    nb._MOON_STATE0_KM[key] = MOON_STATE0
    nb._SUN_STATE0_KM[key] = SUN_STATE0

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    if "numpy" in backends:  # reference must run first
        backends.remove("numpy")
    backends.insert(0, "numpy")
    unknown = set(backends) - set(RUNNERS)
    if unknown:
        ap.error(f"unknown backends {sorted(unknown)}; choose from {list(RUNNERS)}")

    rows, avail = bench(args.n, args.steps, args.sun, args.repeat, backends)
    env = {
        "python": sys.version.split()[0], "numpy": np.__version__,
        "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "gpu": gpu_mod.device_info(), "jax_device": avail.get("_jax_device"),
    }
    for mod in ("numba", "jax", "cupy"):
        try:
            env[mod] = __import__(mod).__version__
        except ImportError:
            env[mod] = None
    result = {"config": {"n": args.n, "n_steps": args.steps, "include_sun": args.sun,
                         "repeat": args.repeat}, "env": env, "backends": rows}
    if not args.no_profile:
        result["profile"] = profile_vectorized(args.n, args.steps, args.sun)

    print(f"N={args.n} n_steps={args.steps} sun={args.sun} repeat={args.repeat}")
    print(f"{'backend':<12}{'median s':>11}{'first s':>10}{'x numpy':>9}{'med err':>10}{'>1e-6':>7}  note")
    for name, row in rows.items():
        if row["status"] != "ok":
            print(f"{name:<12}{'-':>11}{'-':>10}{'-':>9}{'-':>10}{'-':>7}  skipped: {row['reason']}")
            continue
        print(f"{name:<12}{row['median_s']:>11.4f}{row['first_call_s']:>10.3f}"
              f"{row.get('speedup_vs_numpy', float('nan')):>9.1f}{row['median_rel_err']:>10.1e}"
              f"{row['n_rel_err_gt_1e-6']:>7}  closest-step mismatches="
              f"{row['closest_step_mismatches']}")
    if "profile" in result:
        p = result["profile"]
        print(f"\nvectorized path: {p['array_ops_per_call']} array ops/call "
              f"({p['array_ops_per_rk4_step']:.0f} per RK4 step + {p['array_ops_fixed']:.0f} fixed)"
              f" -> ~1 CuPy kernel launch each")
        print(f"numpy split: grid precompute {p['numpy_grid_precompute_s']:.4f}s, "
              f"RK4 loop {p['numpy_rk4_loop_s']:.4f}s "
              f"({p['numpy_us_per_array_op']:.1f} us per array op)")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(result, fh, indent=2)
    return result


if __name__ == "__main__":
    main()
