"""Batched, zero-revolution Lambert solver using the universal-variable
formulation (Bate/Mueller/White; Vallado Alg. 58; Curtis Alg. 5.2), written
against the numpy/cupy array-module abstraction so the exact same code runs
on CPU or GPU.

This is the framework's primary CUDA-acceleration point: evaluating a single
Lambert problem is cheap, but trajectory optimization and porkchop-plot
generation need thousands to millions of independent solves (one per
candidate departure/arrival date pair, or one per optimizer population
member per generation). Those solves share no data dependencies, so they are
evaluated as one batched array operation instead of a Python loop -- on GPU
via CuPy when available, transparently falling back to vectorized numpy on
CPU otherwise.

The iteration count is fixed (not data-dependent) so the whole solve is a
straight-line sequence of array ops with no per-item branching, which is
exactly what GPUs execute well. Only the zero-revolution, short-way/long-way
transfer is implemented; multi-revolution solutions are out of scope for the
batched solver (use orbitopt.lambert.cpu.solve_lambert_single, which wraps
pykep's full multi-rev Izzo solver, when those are needed).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbitopt.core.gpu import get_array_module, to_numpy


@dataclass
class LambertBatchResult:
    v1: np.ndarray  # (N, 3) departure velocity
    v2: np.ndarray  # (N, 3) arrival velocity
    residual: np.ndarray  # (N,) |tof(z) - tof_target|, for convergence diagnostics
    converged: np.ndarray  # (N,) bool


def _stumpff(z, xp):
    """Stumpff functions C(z), S(z), vectorized and branchless.

    xp.where evaluates every branch eagerly, so each branch's denominator is
    pre-guarded against division by zero before selection -- this is the
    standard trick for writing branchless, GPU-friendly special functions.
    """
    pos = z > 1e-6
    neg = z < -1e-6

    z_pos = xp.where(pos, z, 1.0)
    sz_pos = xp.sqrt(z_pos)
    c_pos = (1.0 - xp.cos(sz_pos)) / z_pos
    s_pos = (sz_pos - xp.sin(sz_pos)) / z_pos**1.5

    z_neg = xp.where(neg, -z, 1.0)
    sz_neg = xp.sqrt(z_neg)
    c_neg = (1.0 - xp.cosh(sz_neg)) / (-z_neg)
    s_neg = (xp.sinh(sz_neg) - sz_neg) / z_neg**1.5

    c_small = 0.5 - z / 24.0 + z**2 / 720.0 - z**3 / 40320.0
    s_small = 1.0 / 6.0 - z / 120.0 + z**2 / 5040.0 - z**3 / 362880.0

    C = xp.where(pos, c_pos, xp.where(neg, c_neg, c_small))
    S = xp.where(pos, s_pos, xp.where(neg, s_neg, s_small))
    return C, S


def _y_of_z(z, r1, r2, A, xp):
    C, S = _stumpff(z, xp)
    y = r1 + r2 + A * (z * S - 1.0) / xp.sqrt(C)
    return y, C, S


def _tof_of_z(z, r1, r2, A, mu, xp):
    y, C, S = _y_of_z(z, r1, r2, A, xp)
    y = xp.maximum(y, 1e-9)
    chi = xp.sqrt(y / C)
    return (chi**3 * S + A * xp.sqrt(y)) / xp.sqrt(mu), y, C


def solve_lambert_batch(
    r1_vec,
    r2_vec,
    tof,
    mu,
    prograde=True,
    use_gpu=None,
    maxiter=60,
    tol=1e-6,
):
    """Solve N independent zero-revolution Lambert problems at once.

    Parameters
    ----------
    r1_vec, r2_vec : array-like, shape (N, 3)
        Departure / arrival position vectors (consistent length units).
    tof : array-like, shape (N,)
        Time of flight for each problem (consistent time units with mu).
    mu : float
        Gravitational parameter of the central body.
    prograde : bool
        Assume prograde (counter-clockwise, +z angular momentum) transfers.
    use_gpu : bool or None
        None (default) picks CuPy if available, else numpy. True forces GPU
        and raises if unavailable; False forces CPU.

    Returns
    -------
    LambertBatchResult with v1, v2 as (N, 3) numpy arrays (always brought
    back to host memory), plus per-item residual/converged diagnostics.
    """
    xp = get_array_module(use_gpu)

    r1_vec = xp.asarray(r1_vec, dtype=xp.float64)
    r2_vec = xp.asarray(r2_vec, dtype=xp.float64)
    tof = xp.asarray(tof, dtype=xp.float64)
    n = r1_vec.shape[0]

    r1 = xp.linalg.norm(r1_vec, axis=-1)
    r2 = xp.linalg.norm(r2_vec, axis=-1)

    cross_z = r1_vec[:, 0] * r2_vec[:, 1] - r1_vec[:, 1] * r2_vec[:, 0]
    cos_dnu = xp.clip(xp.sum(r1_vec * r2_vec, axis=-1) / (r1 * r2), -1.0, 1.0)
    dnu = xp.arccos(cos_dnu)
    long_way = (cross_z < 0.0) if prograde else (cross_z >= 0.0)
    dnu = xp.where(long_way, 2.0 * xp.pi - dnu, dnu)

    A = xp.sin(dnu) * xp.sqrt(r1 * r2 / xp.maximum(1.0 - xp.cos(dnu), 1e-12))

    # z is bounded in (-4*pi**2, 4*pi**2) for any physical 0-rev transfer;
    # clipping each Newton step to a wider-but-finite range keeps a
    # transiently-diverging candidate from overflowing cosh/sinh while it's
    # still being iterated on (it will simply fail the convergence check).
    z_lo, z_hi = -(4.0 * xp.pi) ** 2, (2.0 * xp.pi) ** 2
    z = xp.zeros(n, dtype=xp.float64)
    for _ in range(maxiter):
        f, _, _ = _tof_of_z(z, r1, r2, A, mu, xp)
        f = f - tof
        h = 1e-4 * xp.maximum(xp.abs(z), 1.0)
        f_plus, _, _ = _tof_of_z(z + h, r1, r2, A, mu, xp)
        f_minus, _, _ = _tof_of_z(z - h, r1, r2, A, mu, xp)
        dfdz = (f_plus - tof - (f_minus - tof)) / (2.0 * h)
        dfdz = xp.where(xp.abs(dfdz) < 1e-12, 1e-12, dfdz)
        z = xp.clip(z - f / dfdz, z_lo, z_hi)

    tof_final, y, C = _tof_of_z(z, r1, r2, A, mu, xp)
    residual = xp.abs(tof_final - tof)
    converged = residual < tol * xp.maximum(tof, 1.0)

    f_coef = 1.0 - y / r1
    g_coef = A * xp.sqrt(y / mu)
    gdot_coef = 1.0 - y / r2

    v1 = (r2_vec - f_coef[:, None] * r1_vec) / g_coef[:, None]
    v2 = (gdot_coef[:, None] * r2_vec - r1_vec) / g_coef[:, None]

    return LambertBatchResult(
        v1=to_numpy(v1),
        v2=to_numpy(v2),
        residual=to_numpy(residual),
        converged=to_numpy(converged),
    )
