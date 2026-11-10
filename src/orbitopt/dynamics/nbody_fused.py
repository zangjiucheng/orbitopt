"""Fused (one-kernel) prototypes of
``nbody_gpu.propagate_spacecraft_batch(..., output="closest_approach")``,
written for the task #143 evaluation (see
``docs/benchmarks/fused_kernel_evaluation.md`` and
``scripts/bench_nbody_backends.py``).

The production path in ``nbody_gpu`` is array-vectorized: every RK4 step is
a few dozen whole-batch numpy/CuPy operations, so on a GPU each step costs a
few dozen kernel launches driven from Python. The prototypes here invert
that loop nest instead: one thread (or one CPU loop iteration) owns one
candidate and runs *all* integration steps plus the running Moon
closest-approach update in registers, so a whole propagation is a single
kernel launch.

The Moon/Sun half-step grid is still built by the production
``nbody_gpu._precompute_body_grid`` (task #139) on the host; only the RK4 +
closest-approach loop is fused. All backends return the same
``ClosestApproachBatchResult`` as the production path and agree with it to
floating-point round-off (verified by ``tests/test_nbody_fused.py``).

Backends (all optional dependencies, imported lazily):

- ``"python"``    -- the scalar reference, plain Python. Slow; for tests.
- ``"numba-cpu"`` -- the same scalar code, ``numba.njit(parallel=True)``.
- ``"numba-cuda"``-- the same scalar code, ``numba.cuda.jit`` (one thread per
  candidate). Runs under ``NUMBA_ENABLE_CUDASIM=1`` without a GPU.
- ``"cupy-raw"``  -- a hand-written CUDA C ``cupy.RawKernel`` of the same
  algorithm.
- ``"jax"``       -- vectorized over candidates, ``lax.fori_loop`` over
  steps, compiled by XLA (``jax.jit``); enables ``jax_enable_x64``.

Only ``"numba-cpu"`` is used in production: task #164 wired it into
``orbitopt.problems.free_return.FreeReturnScreeningProblem`` (``backend=
"auto"|"numba-cpu"``). The other backends remain evaluation artifacts; see
the decision record in the benchmark doc before wiring them in.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from orbitopt.dynamics import nbody_gpu as nb

cuda = None  # numba.cuda, bound lazily by _numba_cuda_kernel()

BACKENDS = ("python", "numba-cpu", "numba-cuda", "cupy-raw", "jax")


# ---------------------------------------------------------------------------
# Inputs shared by every backend
# ---------------------------------------------------------------------------

def prepare_inputs(
    r0, v0, t_span, n_steps, mu_earth, mu_moon, include_sun=False, mu_sun=None,
    epoch0=0.0, kepler_maxiter=20, moon_seed_epoch_ephemeris_seconds=None,
):
    """Host-side (numpy) inputs for the fused kernels.

    Body grids are kept compact as ``(n_bodies, 2*n_steps+1, U, 3)`` over the
    ``U`` distinct (epoch0, dt) pairs plus a per-candidate column index
    ``gidx`` -- the kernels gather instead of reading an (n_grid, N, 3)
    broadcast, which keeps the device footprint O(U * n_steps), not
    O(N * n_steps).
    """
    r0 = np.ascontiguousarray(np.asarray(r0, dtype=np.float64).reshape(-1, 3))
    v0 = np.ascontiguousarray(np.asarray(v0, dtype=np.float64).reshape(-1, 3))
    n = max(r0.shape[0], v0.shape[0])
    r0 = np.ascontiguousarray(np.broadcast_to(r0, (n, 3)))
    v0 = np.ascontiguousarray(np.broadcast_to(v0, (n, 3)))
    t_span = np.broadcast_to(np.atleast_1d(np.asarray(t_span, dtype=np.float64)), (n,))
    epoch0 = np.ascontiguousarray(
        np.broadcast_to(np.atleast_1d(np.asarray(epoch0, dtype=np.float64)), (n,))
    )
    dt = np.ascontiguousarray(t_span / n_steps)

    if moon_seed_epoch_ephemeris_seconds is None:
        moon_seed_epoch_ephemeris_seconds = float(epoch0[0])
    if include_sun and mu_sun is None:
        mu_sun = nb.MU_SUN_KM3_S2

    # _precompute_body_grid returns the compact (n_grid, U, 3) grid over the
    # distinct (epoch0, dt) pairs plus each candidate's column index.
    ephem, inverse = nb._precompute_body_grid(
        epoch0, dt, n_steps, np, mu_earth, mu_moon, include_sun,
        mu_sun, kepler_maxiter, moon_seed_epoch_ephemeris_seconds,
    )
    gidx = np.ascontiguousarray(inverse.astype(np.int64))
    grid = np.ascontiguousarray(np.stack([np.asarray(e[1]) for e in ephem]))
    ind = np.ascontiguousarray(np.stack([np.asarray(e[2]) for e in ephem]))
    mus = np.array([e[0] for e in ephem], dtype=np.float64)
    return dict(
        r0=r0, v0=v0, dt=dt, epoch0=epoch0, grid=grid, ind=ind, gidx=gidx,
        mus=mus, n_steps=int(n_steps), mu_earth=float(mu_earth),
    )


def _alloc_outputs(n):
    return dict(
        min_d=np.empty(n), min_t=np.empty(n), min_s=np.empty(n, dtype=np.int64),
        r=np.empty((n, 3)), v=np.empty((n, 3)), t=np.empty(n),
    )


def _result(o):
    return nb.ClosestApproachBatchResult(
        min_distance_km=o["min_d"], min_distance_epoch_s=o["min_t"],
        min_distance_step=o["min_s"], r=o["r"], v=o["v"], t_abs=o["t"],
    )


# ---------------------------------------------------------------------------
# Scalar per-candidate kernel (plain Python; numba compiles the same source)
# ---------------------------------------------------------------------------

def _accel_py(rx, ry, rz, k, u, grid, ind, mus, mu_earth):
    rn = math.sqrt(rx * rx + ry * ry + rz * rz)
    f = -mu_earth / (rn * rn * rn)
    ax = f * rx
    ay = f * ry
    az = f * rz
    for b in range(grid.shape[0]):
        dx = grid[b, k, u, 0] - rx
        dy = grid[b, k, u, 1] - ry
        dz = grid[b, k, u, 2] - rz
        dn = math.sqrt(dx * dx + dy * dy + dz * dz)
        g = 1.0 / (dn * dn * dn)
        mb = mus[b]
        ax += mb * (dx * g - ind[b, k, u, 0])
        ay += mb * (dy * g - ind[b, k, u, 1])
        az += mb * (dz * g - ind[b, k, u, 2])
    return ax, ay, az


def _make_candidate(accel):
    """Build the per-candidate integrator around a given ``accel`` (plain
    Python or a numba-compiled device/host function)."""

    def candidate(i, r0, v0, dt_arr, epoch0, grid, ind, gidx, mus, n_steps, mu_earth,
                  min_d, min_t, min_s, r_out, v_out, t_out):
        u = gidx[i]
        h = dt_arr[i]
        rx = r0[i, 0]
        ry = r0[i, 1]
        rz = r0[i, 2]
        vx = v0[i, 0]
        vy = v0[i, 1]
        vz = v0[i, 2]
        t = epoch0[i]
        best = math.inf
        best_t = t
        best_s = 0
        for step in range(n_steps + 1):
            # Closest-approach sample at stored step ``step`` (grid index 2*step).
            k = 2 * step
            dx = rx - grid[0, k, u, 0]
            dy = ry - grid[0, k, u, 1]
            dz = rz - grid[0, k, u, 2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist < best or (math.isnan(dist) and not math.isnan(best)):
                best = dist
                best_t = t
                best_s = step
            if step == n_steps:
                break

            k1vx, k1vy, k1vz = accel(rx, ry, rz, k, u, grid, ind, mus, mu_earth)
            hh = 0.5 * h
            k2rx = vx + hh * k1vx
            k2ry = vy + hh * k1vy
            k2rz = vz + hh * k1vz
            k2vx, k2vy, k2vz = accel(rx + hh * vx, ry + hh * vy, rz + hh * vz,
                                     k + 1, u, grid, ind, mus, mu_earth)
            k3rx = vx + hh * k2vx
            k3ry = vy + hh * k2vy
            k3rz = vz + hh * k2vz
            k3vx, k3vy, k3vz = accel(rx + hh * k2rx, ry + hh * k2ry, rz + hh * k2rz,
                                     k + 1, u, grid, ind, mus, mu_earth)
            k4rx = vx + h * k3vx
            k4ry = vy + h * k3vy
            k4rz = vz + h * k3vz
            k4vx, k4vy, k4vz = accel(rx + h * k3rx, ry + h * k3ry, rz + h * k3rz,
                                     k + 2, u, grid, ind, mus, mu_earth)
            h6 = h / 6.0
            rx = rx + h6 * (vx + 2.0 * k2rx + 2.0 * k3rx + k4rx)
            ry = ry + h6 * (vy + 2.0 * k2ry + 2.0 * k3ry + k4ry)
            rz = rz + h6 * (vz + 2.0 * k2rz + 2.0 * k3rz + k4rz)
            vx = vx + h6 * (k1vx + 2.0 * k2vx + 2.0 * k3vx + k4vx)
            vy = vy + h6 * (k1vy + 2.0 * k2vy + 2.0 * k3vy + k4vy)
            vz = vz + h6 * (k1vz + 2.0 * k2vz + 2.0 * k3vz + k4vz)
            t = t + h
        min_d[i] = best
        min_t[i] = best_t
        min_s[i] = best_s
        r_out[i, 0] = rx
        r_out[i, 1] = ry
        r_out[i, 2] = rz
        v_out[i, 0] = vx
        v_out[i, 1] = vy
        v_out[i, 2] = vz
        t_out[i] = t

    return candidate


def _args(p, o):
    return (p["r0"], p["v0"], p["dt"], p["epoch0"], p["grid"], p["ind"], p["gidx"],
            p["mus"], p["n_steps"], p["mu_earth"],
            o["min_d"], o["min_t"], o["min_s"], o["r"], o["v"], o["t"])


def run_python(p):
    cand = _make_candidate(_accel_py)
    o = _alloc_outputs(p["r0"].shape[0])
    for i in range(p["r0"].shape[0]):
        cand(i, *_args(p, o))
    return _result(o)


# ---------------------------------------------------------------------------
# Numba (CPU, parallel)
# ---------------------------------------------------------------------------

def _numba_cpu_kernel():
    # Module-level jit functions in a separate module so numba's on-disk
    # cache hits across processes (see _nbody_fused_numba's docstring).
    from orbitopt.dynamics._nbody_fused_numba import kernel

    return kernel


_NUMBA_AVAILABLE = None


def numba_available() -> bool:
    """True if ``numba`` is importable (the ``numba-cpu`` backend can run)
    and hasn't been disabled by ``disable_numba()``."""
    global _NUMBA_AVAILABLE
    if _NUMBA_AVAILABLE is None:
        try:
            import numba  # noqa: F401
        except Exception:
            _NUMBA_AVAILABLE = False
        else:
            _NUMBA_AVAILABLE = True
    return _NUMBA_AVAILABLE


def disable_numba():
    """Make ``numba_available()`` report False for the rest of the process
    (used by ``backend="auto"`` callers after a numba runtime failure)."""
    global _NUMBA_AVAILABLE
    _NUMBA_AVAILABLE = False


_NUMBA_CPU_WARM = False


def numba_cpu_warm() -> bool:
    """True once the numba-cpu kernel has run in this process (so its JIT
    compile / on-disk cache load is already paid for)."""
    return _NUMBA_CPU_WARM


def run_numba_cpu(p):
    global _NUMBA_CPU_WARM
    o = _alloc_outputs(p["r0"].shape[0])
    _numba_cpu_kernel()(*_args(p, o))
    _NUMBA_CPU_WARM = True
    return _result(o)


# ---------------------------------------------------------------------------
# Numba CUDA (one thread per candidate)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _numba_cuda_kernel():
    # Module-global binding (not a closure): numba's CUDA simulator swaps the
    # ``cuda`` module only in the kernel's globals.
    global cuda
    from numba import cuda

    accel = cuda.jit(device=True)(_accel_py)
    cand = cuda.jit(device=True)(_make_candidate(accel))

    @cuda.jit
    def kernel(r0, v0, dt_arr, epoch0, grid, ind, gidx, mus, n_steps, mu_earth,
               min_d, min_t, min_s, r_out, v_out, t_out):
        i = cuda.grid(1)
        if i < r0.shape[0]:
            cand(i, r0, v0, dt_arr, epoch0, grid, ind, gidx, mus, n_steps, mu_earth,
                 min_d, min_t, min_s, r_out, v_out, t_out)

    return kernel


def run_numba_cuda(p, threads_per_block=128):
    from numba import cuda

    n = p["r0"].shape[0]
    host_out = _alloc_outputs(n)
    d = {k: (cuda.to_device(v) if isinstance(v, np.ndarray) else v) for k, v in p.items()}
    do = {k: cuda.to_device(v) for k, v in host_out.items()}
    blocks = (n + threads_per_block - 1) // threads_per_block
    _numba_cuda_kernel()[blocks, threads_per_block](*_args(d, do))
    cuda.synchronize()
    return _result({k: v.copy_to_host() for k, v in do.items()})


# ---------------------------------------------------------------------------
# CuPy RawKernel (hand-written CUDA C of the same algorithm)
# ---------------------------------------------------------------------------

CUPY_RAW_SOURCE = r"""
extern "C" __device__ __forceinline__ void accel(
    double rx, double ry, double rz, long long k, long long u,
    const double* grid, const double* ind, const double* mus,
    int n_bodies, long long n_grid, long long n_u, double mu_earth,
    double* ax, double* ay, double* az)
{
    double rn = sqrt(rx * rx + ry * ry + rz * rz);
    double f = -mu_earth / (rn * rn * rn);
    double x = f * rx, y = f * ry, z = f * rz;
    for (int b = 0; b < n_bodies; ++b) {
        long long o = (((long long)b * n_grid + k) * n_u + u) * 3;
        double dx = grid[o] - rx, dy = grid[o + 1] - ry, dz = grid[o + 2] - rz;
        double dn = sqrt(dx * dx + dy * dy + dz * dz);
        double g = 1.0 / (dn * dn * dn);
        double mb = mus[b];
        x += mb * (dx * g - ind[o]);
        y += mb * (dy * g - ind[o + 1]);
        z += mb * (dz * g - ind[o + 2]);
    }
    *ax = x; *ay = y; *az = z;
}

extern "C" __global__ void fused_rk4_closest(
    const double* r0, const double* v0, const double* dt_arr, const double* epoch0,
    const double* grid, const double* ind, const long long* gidx, const double* mus,
    int n_bodies, long long n_grid, long long n_u, long long n, int n_steps,
    double mu_earth,
    double* min_d, double* min_t, long long* min_s,
    double* r_out, double* v_out, double* t_out)
{
    long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    long long u = gidx[i];
    double h = dt_arr[i];
    double rx = r0[3 * i], ry = r0[3 * i + 1], rz = r0[3 * i + 2];
    double vx = v0[3 * i], vy = v0[3 * i + 1], vz = v0[3 * i + 2];
    double t = epoch0[i];
    double best = CUDART_INF, best_t = t;
    long long best_s = 0;
    for (int step = 0; step <= n_steps; ++step) {
        long long k = 2LL * step;
        long long o = (k * n_u + u) * 3;  /* body 0 = Moon */
        double dx = rx - grid[o], dy = ry - grid[o + 1], dz = rz - grid[o + 2];
        double dist = sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < best || (isnan(dist) && !isnan(best))) {
            best = dist; best_t = t; best_s = step;
        }
        if (step == n_steps) break;

        double k1vx, k1vy, k1vz, k2vx, k2vy, k2vz, k3vx, k3vy, k3vz, k4vx, k4vy, k4vz;
        double hh = 0.5 * h;
        accel(rx, ry, rz, k, u, grid, ind, mus, n_bodies, n_grid, n_u, mu_earth,
              &k1vx, &k1vy, &k1vz);
        double k2rx = vx + hh * k1vx, k2ry = vy + hh * k1vy, k2rz = vz + hh * k1vz;
        accel(rx + hh * vx, ry + hh * vy, rz + hh * vz, k + 1, u, grid, ind, mus,
              n_bodies, n_grid, n_u, mu_earth, &k2vx, &k2vy, &k2vz);
        double k3rx = vx + hh * k2vx, k3ry = vy + hh * k2vy, k3rz = vz + hh * k2vz;
        accel(rx + hh * k2rx, ry + hh * k2ry, rz + hh * k2rz, k + 1, u, grid, ind, mus,
              n_bodies, n_grid, n_u, mu_earth, &k3vx, &k3vy, &k3vz);
        double k4rx = vx + h * k3vx, k4ry = vy + h * k3vy, k4rz = vz + h * k3vz;
        accel(rx + h * k3rx, ry + h * k3ry, rz + h * k3rz, k + 2, u, grid, ind, mus,
              n_bodies, n_grid, n_u, mu_earth, &k4vx, &k4vy, &k4vz);
        double h6 = h / 6.0;
        rx += h6 * (vx + 2.0 * k2rx + 2.0 * k3rx + k4rx);
        ry += h6 * (vy + 2.0 * k2ry + 2.0 * k3ry + k4ry);
        rz += h6 * (vz + 2.0 * k2rz + 2.0 * k3rz + k4rz);
        vx += h6 * (k1vx + 2.0 * k2vx + 2.0 * k3vx + k4vx);
        vy += h6 * (k1vy + 2.0 * k2vy + 2.0 * k3vy + k4vy);
        vz += h6 * (k1vz + 2.0 * k2vz + 2.0 * k3vz + k4vz);
        t += h;
    }
    min_d[i] = best; min_t[i] = best_t; min_s[i] = best_s;
    r_out[3 * i] = rx; r_out[3 * i + 1] = ry; r_out[3 * i + 2] = rz;
    v_out[3 * i] = vx; v_out[3 * i + 1] = vy; v_out[3 * i + 2] = vz;
    t_out[i] = t;
}
"""


@lru_cache(maxsize=None)
def _cupy_raw_kernel():
    import cupy as cp

    src = "#include <math_constants.h>\n" + CUPY_RAW_SOURCE
    return cp.RawKernel(src, "fused_rk4_closest")


def run_cupy_raw(p, threads_per_block=128):
    import cupy as cp

    n = p["r0"].shape[0]
    d = {k: cp.asarray(v) for k, v in p.items() if isinstance(v, np.ndarray)}
    o = {k: cp.asarray(v) for k, v in _alloc_outputs(n).items()}
    n_bodies, n_grid, n_u = p["grid"].shape[:3]
    blocks = (n + threads_per_block - 1) // threads_per_block
    _cupy_raw_kernel()(
        (blocks,), (threads_per_block,),
        (d["r0"], d["v0"], d["dt"], d["epoch0"], d["grid"], d["ind"], d["gidx"], d["mus"],
         np.int32(n_bodies), np.int64(n_grid), np.int64(n_u), np.int64(n),
         np.int32(p["n_steps"]), np.float64(p["mu_earth"]),
         o["min_d"], o["min_t"], o["min_s"], o["r"], o["v"], o["t"]),
    )
    cp.cuda.Stream.null.synchronize()
    return _result({k: cp.asnumpy(v) for k, v in o.items()})


# ---------------------------------------------------------------------------
# JAX (XLA-fused, vectorized over candidates)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _jax_fn():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    def run(r0, v0, dt, epoch0, grid, ind, gidx, mus, n_steps, mu_earth):
        # (B, n_grid, N, 3) gathered once; XLA keeps it on device.
        g = grid[:, :, gidx]
        gi = ind[:, :, gidx]
        dtc = dt[:, None]

        def accel(r, k):
            rn = jnp.linalg.norm(r, axis=-1, keepdims=True)
            a = -mu_earth * r / rn**3
            rb = jax.lax.dynamic_index_in_dim(g, k, axis=1, keepdims=False)
            ib = jax.lax.dynamic_index_in_dim(gi, k, axis=1, keepdims=False)
            d = rb - r[None]
            dn = jnp.linalg.norm(d, axis=-1, keepdims=True)
            return a + jnp.sum(mus[:, None, None] * (d / dn**3 - ib), axis=0)

        def update(step, r, t, carry):
            best, best_t, best_s = carry
            moon = jax.lax.dynamic_index_in_dim(g[0], 2 * step, axis=0, keepdims=False)
            dist = jnp.linalg.norm(r - moon, axis=-1)
            better = (dist < best) | (jnp.isnan(dist) & ~jnp.isnan(best))
            return (jnp.where(better, dist, best), jnp.where(better, t, best_t),
                    jnp.where(better, step, best_s))

        def body(step, c):
            r, v, t, ca = c
            ca = update(step, r, t, ca)
            k = 2 * step
            k1v = accel(r, k)
            k2r = v + 0.5 * dtc * k1v
            k2v = accel(r + 0.5 * dtc * v, k + 1)
            k3r = v + 0.5 * dtc * k2v
            k3v = accel(r + 0.5 * dtc * k2r, k + 1)
            k4r = v + dtc * k3v
            k4v = accel(r + dtc * k3r, k + 2)
            r = r + (dtc / 6.0) * (v + 2.0 * k2r + 2.0 * k3r + k4r)
            v = v + (dtc / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
            return r, v, t + dt, ca

        n = r0.shape[0]
        ca0 = (jnp.full((n,), jnp.inf), epoch0, jnp.zeros((n,), dtype=jnp.int64))
        r, v, t, ca = jax.lax.fori_loop(0, n_steps, body, (r0, v0, epoch0, ca0))
        ca = update(n_steps, r, t, ca)
        return ca[0], ca[1], ca[2], r, v, t

    return jax.jit(run, static_argnums=(8,))


def run_jax(p):
    fn = _jax_fn()
    out = fn(p["r0"], p["v0"], p["dt"], p["epoch0"], p["grid"], p["ind"], p["gidx"],
             p["mus"], p["n_steps"], p["mu_earth"])
    out = [np.asarray(x) for x in out]  # blocks until ready
    return _result(dict(min_d=out[0], min_t=out[1], min_s=out[2].astype(np.int64),
                        r=out[3], v=out[4], t=out[5]))


_RUNNERS = {
    "python": run_python,
    "numba-cpu": run_numba_cpu,
    "numba-cuda": run_numba_cuda,
    "cupy-raw": run_cupy_raw,
    "jax": run_jax,
}


def propagate_closest_approach_fused(
    r0, v0, t_span, n_steps, mu_earth, mu_moon, include_sun=False, mu_sun=None,
    epoch0=0.0, kepler_maxiter=20, moon_seed_epoch_ephemeris_seconds=None,
    backend="numba-cpu",
):
    """Fused-kernel equivalent of
    ``nbody_gpu.propagate_spacecraft_batch(..., output="closest_approach")``.
    Same arguments (minus ``use_gpu``/``output``) plus ``backend``, one of
    ``BACKENDS``."""
    if backend not in _RUNNERS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    p = prepare_inputs(
        r0, v0, t_span, n_steps, mu_earth, mu_moon, include_sun, mu_sun, epoch0,
        kepler_maxiter, moon_seed_epoch_ephemeris_seconds,
    )
    return _RUNNERS[backend](p)
