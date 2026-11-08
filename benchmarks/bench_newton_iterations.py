"""Benchmark: analytic-derivative Newton with early exit (task #135) vs. the
legacy central-difference, fixed-iteration loops, for the batched Lambert
solver and the batched universal-Kepler propagator.

The legacy loops are reproduced inline below (verbatim algorithm of the
pre-#135 code) so the comparison runs against the same Stumpff/TOF kernels.
No speed-up is assumed: the new loops do one function evaluation per
iteration instead of three, but pay a device->host sync per iteration for
the early-exit check (and the Kepler loop a backtracking check), so on GPU
with small batches they can be slower. Report, don't assume.

    PYTHONPATH=src python benchmarks/bench_newton_iterations.py [--gpu] [--n 100000]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from orbitopt.core.gpu import get_array_module, to_numpy
from orbitopt.dynamics import nbody_gpu as nb
from orbitopt.lambert import gpu_batch as gb

MU_SUN = 1.32712440018e20  # m^3/s^2
AU = 1.495978707e11
DAY = 86400.0


# ---------------------------------------------------------------- legacy ----

def _legacy_lambert_z(xp, r1, r2, A, tof, mu, maxiter):
    """Pre-#135 Lambert loop: central-difference dTOF/dz, fixed maxiter."""
    n = r1.shape[0]

    def F(z):
        t, _, _, phys = gb._tof_of_z(z, r1, r2, A, mu, xp)
        return t - tof, phys

    lo = xp.full(n, -(4.0 * np.pi) ** 2)
    hi = xp.full(n, (2.0 * np.pi) ** 2 - 1e-2)
    z = xp.zeros(n)
    for _ in range(maxiter):
        f, phys = F(z)
        too_short = f < 0.0
        lo = xp.where(too_short, z, lo)
        hi = xp.where(too_short, hi, z)
        h = 1e-5 * xp.maximum(xp.abs(z), 1.0)
        f_plus, ph_p = F(z + h)
        f_minus, ph_m = F(z - h)
        dfdz = (f_plus - f_minus) / (2.0 * h)
        deriv_ok = phys & ph_p & ph_m & xp.isfinite(dfdz) & (dfdz > 0.0)
        z_newton = z - f / xp.where(deriv_ok, dfdz, 1.0)
        newton_ok = deriv_ok & xp.isfinite(z_newton) & (z_newton > lo) & (z_newton < hi)
        z = xp.where(newton_ok, z_newton, 0.5 * (lo + hi))
    return z


def _legacy_kepler_chi(xp, r0n, vr0, alpha, dt, mu, maxiter):
    """Pre-#135 Kepler loop: central-difference dF/dchi, fixed maxiter."""
    sqrt_mu = np.sqrt(mu)

    def res(c):
        return nb._kepler_residual_and_derivative(c, r0n, vr0, alpha, dt, sqrt_mu, xp)[0]

    chi = sqrt_mu * xp.abs(alpha) * dt
    for _ in range(maxiter):
        F = res(chi)
        h = 1e-4 * xp.maximum(xp.abs(chi), 1.0)
        dF = (res(chi + h) - res(chi - h)) / (2.0 * h)
        dF = xp.where(xp.abs(dF) < 1e-12, 1e-12, dF)
        chi = chi - F / dF
    return chi


# ----------------------------------------------------------------- cases ----

def lambert_cases(n, rng):
    r1n = rng.uniform(0.7, 1.6, n) * AU
    r2n = rng.uniform(0.7, 1.6, n) * AU
    th1 = rng.uniform(0, 2 * np.pi, n)
    th2 = th1 + rng.uniform(0.2, 2.8, n)
    r1 = np.stack([r1n * np.cos(th1), r1n * np.sin(th1), np.zeros(n)], 1)
    r2 = np.stack([r2n * np.cos(th2), r2n * np.sin(th2), np.zeros(n)], 1)
    tof = rng.uniform(20, 600, n) * DAY  # mix of hyperbolic (short) & elliptic
    return r1, r2, tof


def kepler_cases(n, rng, mu):
    r = rng.uniform(6800e3, 42000e3, n)
    th = rng.uniform(0, 2 * np.pi, n)
    r0 = np.stack([r * np.cos(th), r * np.sin(th), np.zeros(n)], 1)
    vdir = np.stack([-np.sin(th), np.cos(th), np.zeros(n)], 1)
    k = rng.integers(0, 3, n)  # 0 elliptic, 1 near-parabolic, 2 hyperbolic
    f = np.choose(k, [rng.uniform(0.5, 0.95, n), 1.0 + rng.uniform(-1e-6, 1e-6, n), rng.uniform(1.05, 3.0, n)])
    v0 = (np.sqrt(2 * mu / r) * f)[:, None] * vdir
    dt = rng.uniform(600.0, 2 * DAY, n)
    return r0, v0, dt, k


def _time(fn, xp, repeat):
    best = np.inf
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        if xp is not np:
            xp.cuda.Stream.null.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()
    xp = get_array_module(True if args.gpu else False)
    rng = np.random.default_rng(0)
    print(f"backend={xp.__name__}  N={args.n}")

    # Lambert -------------------------------------------------------------
    r1, r2, tof = lambert_cases(args.n, rng)
    r1d, r2d = xp.asarray(r1), xp.asarray(r2)
    r1n, r2n = xp.linalg.norm(r1d, axis=1), xp.linalg.norm(r2d, axis=1)
    dnu = xp.arccos(xp.clip(xp.sum(r1d * r2d, 1) / (r1n * r2n), -1, 1))
    cz = r1d[:, 0] * r2d[:, 1] - r1d[:, 1] * r2d[:, 0]
    dnu = xp.where(cz < 0, 2 * np.pi - dnu, dnu)
    A = xp.sin(dnu) * xp.sqrt(r1n * r2n / (1 - xp.cos(dnu)))
    tofd = xp.asarray(tof)
    with np.errstate(all="ignore"):
        t_old, z_old = _time(lambda: _legacy_lambert_z(xp, r1n, r2n, A, tofd, MU_SUN, 60), xp, args.repeat)
        t_new, res = _time(lambda: gb.solve_lambert_batch(r1, r2, tof, MU_SUN, use_gpu=args.gpu,
                                                          cpu_fallback=False), xp, args.repeat)
        t_old_tof = to_numpy(gb._tof_of_z(z_old, r1n, r2n, A, MU_SUN, xp)[0])
    old_rel = np.abs(t_old_tof - tof) / tof
    new_rel = res.residual[res.valid] / tof[res.valid]
    print("\nLambert (root find; new timing includes velocity/validity post-processing)")
    print(f"  legacy  FD, fixed 60 it : {t_old * 1e3:9.1f} ms   max rel TOF residual {np.nanmax(old_rel):.2e}")
    print(f"  analytic + early exit   : {t_new * 1e3:9.1f} ms   max rel TOF residual {new_rel.max():.2e}"
          f"   iterations={res.iterations}   valid={res.valid.mean():.4f}")

    # Kepler --------------------------------------------------------------
    mu = nb.MU_EARTH_KM3_S2 * 1e9
    r0, v0, dt, kind = kepler_cases(args.n, rng, mu)
    r0d, v0d, dtd = xp.asarray(r0), xp.asarray(v0), xp.asarray(dt)
    r0n = xp.linalg.norm(r0d, axis=1)
    vr0 = xp.sum(r0d * v0d, 1) / r0n
    alpha = 2 / r0n - xp.sum(v0d**2, 1) / mu
    sqrt_mu = np.sqrt(mu)
    with np.errstate(all="ignore"):
        t_old, chi_old = _time(lambda: _legacy_kepler_chi(xp, r0n, vr0, alpha, dtd, mu, 50), xp, args.repeat)
        t_new, (chi_new, conv, iters) = _time(
            lambda: nb._solve_universal_anomaly(r0d, v0d, r0n, vr0, alpha, dtd, mu, xp, 50, 1e-12), xp, args.repeat)

        def rel_res(chi):
            F, _, scale = nb._kepler_residual_and_derivative(chi, r0n, vr0, alpha, dtd, sqrt_mu, xp)
            return to_numpy(xp.abs(F) / scale)

        ro, rn_ = rel_res(chi_old), rel_res(chi_new)
    print("\nKepler universal anomaly (Newton loop only)")
    names = ["elliptic", "near-parabolic", "hyperbolic"]
    print(f"  legacy  FD, fixed 50 it : {t_old * 1e3:9.1f} ms   "
          + "  ".join(f"{names[k]} max rel res {np.nanmax(np.where(np.isfinite(ro[kind == k]), ro[kind == k], np.inf)):.1e}"
                      for k in range(3)))
    print(f"  analytic+damped+early   : {t_new * 1e3:9.1f} ms   "
          + "  ".join(f"{names[k]} max rel res {rn_[kind == k].max():.1e}" for k in range(3))
          + f"   iterations={iters}   converged={to_numpy(conv).mean():.4f}")


if __name__ == "__main__":
    main()
