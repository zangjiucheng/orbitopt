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

Each iteration is a straight-line sequence of array ops with no per-item
branching, which is exactly what GPUs execute well. Newton uses the analytic
dTOF/dz (Curtis eq. 5.43, one Stumpff evaluation per iteration instead of
the three a central difference needs); candidates whose TOF residual is
at the floating-point floor are retired and dropped from the (compacted)
working set, and the loop exits as soon as none remain (one device->host
sync per iteration), so ``maxiter`` is an upper bound, not a fixed count.
Only the zero-revolution, short-way/long-way transfer is implemented;
multi-revolution solutions are out of scope for the batched solver (use orbitopt.lambert.cpu.solve_lambert_single, which wraps
pykep's full multi-rev Izzo solver, when those are needed).

Reliability guarantees (see ``LambertBatchResult.valid``):

* The root finder is a *safeguarded* Newton iteration on z: every candidate
  carries a bracket [z_lo, z_hi] with tof(z_lo) < tof_target < tof(z_hi)
  (tof(z) is monotonic in z), Newton steps that leave the bracket, hit the
  non-physical ``y(z) <= 0`` region, or have a non-positive/non-finite
  derivative are replaced by bisection. The non-physical region is *not*
  clamped back into ``y > 0`` (the old behaviour, which let candidates keep
  iterating on a fake TOF and report a spurious "converged").
* A candidate is only marked valid if the TOF residual is within tolerance
  *and* y > 0, C(z) > 0, the Lagrange g coefficient is non-degenerate, and
  both velocity vectors are finite.
* Hard cases -- transfer angle within ``angle_tol`` of 0/180 deg (where the
  universal-variable A -> 0 / the transfer plane is ill-defined), a failed
  bracket, or a non-converged candidate -- are re-solved independently on CPU
  with pykep's Izzo solver (``cpu_fallback=True``) and accepted only if that
  answer is finite too.
* Invalid candidates never carry NaN: their v1/v2 are zero and their residual
  is +inf, so callers must gate on ``valid``/``converged``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbitopt.physics_io import check_grav_parameter
from orbitopt.core.gpu import get_array_module, to_numpy


@dataclass
class LambertBatchResult:
    v1: np.ndarray  # (N, 3) departure velocity (0 where not valid)
    v2: np.ndarray  # (N, 3) arrival velocity (0 where not valid)
    residual: np.ndarray  # (N,) |tof(z) - tof_target| (0 for CPU-fallback, inf if invalid)
    valid: np.ndarray  # (N,) bool: solution passed every validity check
    fallback: np.ndarray  # (N,) bool: solution came from the pykep CPU fallback
    iterations: int = 0  # root-finder iterations actually run (<= maxiter)

    @property
    def converged(self) -> np.ndarray:
        """Backward-compatible alias of ``valid`` (converged *and* physical)."""
        return self.valid


_STUMPFF_SERIES_ZMAX = 0.1
# C(z) = sum (-z)^k / (2k+2)!,  S(z) = sum (-z)^k / (2k+3)!,  k = 0..5
_C_SERIES = (1 / 2, -1 / 24, 1 / 720, -1 / 40320, 1 / 3628800, -1 / 479001600)
_S_SERIES = (1 / 6, -1 / 120, 1 / 5040, -1 / 362880, 1 / 39916800, -1 / 6227020800)


def _horner(z, coeffs):
    acc = coeffs[-1]
    for c in reversed(coeffs[:-1]):
        acc = c + z * acc
    return acc


def _stumpff(z, xp):
    """Stumpff functions C(z), S(z), vectorized and branchless.

    xp.where evaluates every branch eagerly, so each branch's denominator is
    pre-guarded against division by zero before selection -- this is the
    standard trick for writing branchless, GPU-friendly special functions.
    """
    # Below |z| = 0.1 the closed forms lose ~eps/|z| to cancellation
    # (1 - cos sqrt(z) etc.), which stalls Newton near parabolic orbits; the
    # degree-5 series there is accurate to < 1e-17.
    pos = z > _STUMPFF_SERIES_ZMAX
    neg = z < -_STUMPFF_SERIES_ZMAX

    z_pos = xp.where(pos, z, 1.0)
    sz_pos = xp.sqrt(z_pos)
    c_pos = (1.0 - xp.cos(sz_pos)) / z_pos
    s_pos = (sz_pos - xp.sin(sz_pos)) / z_pos**1.5

    z_neg = xp.where(neg, -z, 1.0)
    sz_neg = xp.sqrt(z_neg)
    c_neg = (1.0 - xp.cosh(sz_neg)) / (-z_neg)
    s_neg = (xp.sinh(sz_neg) - sz_neg) / z_neg**1.5

    c_small = _horner(z, _C_SERIES)
    s_small = _horner(z, _S_SERIES)

    C = xp.where(pos, c_pos, xp.where(neg, c_neg, c_small))
    S = xp.where(pos, s_pos, xp.where(neg, s_neg, s_small))
    return C, S


def _y_of_z(z, r1, r2, A, xp):
    C, S = _stumpff(z, xp)
    C_safe = xp.where(C > 0.0, C, 1.0)
    y = r1 + r2 + A * (z * S - 1.0) / xp.sqrt(C_safe)
    y = xp.where(C > 0.0, y, -1.0)  # C <= 0 is never physical
    return y, C, S


def _tof_of_z(z, r1, r2, A, mu, xp):
    """TOF(z) plus a mask of where it is physical (y > 0, C > 0, finite).

    Outside the physical region the returned TOF is 0 -- the limit of the
    true TOF as y -> 0+ -- so the bracketed root finder treats it as "too
    short" and moves z up, instead of iterating on a clamped fake value.
    """
    t, y, C, _, phys = _tof_core(z, r1, r2, A, mu, xp)
    return t, y, C, phys


def _tof_core(z, r1, r2, A, mu, xp):
    """Shared body of ``_tof_of_z``; also returns S(z) for the derivative."""
    y, C, S = _y_of_z(z, r1, r2, A, xp)
    phys = (y > 0.0) & (C > 0.0) & xp.isfinite(y) & xp.isfinite(C) & xp.isfinite(S)
    y_s = xp.where(phys, y, 1.0)
    C_s = xp.where(phys, C, 1.0)
    S_s = xp.where(phys, S, 0.0)
    chi = xp.sqrt(y_s / C_s)
    t = (chi**3 * S_s + A * xp.sqrt(y_s)) / xp.sqrt(mu)
    phys = phys & xp.isfinite(t)
    t = xp.where(phys, t, 0.0)
    return t, y, C, S, phys


# Taylor coefficients of D(z) = (C(z) - 3 S(z) / (2 C(z))) / (2 z) about z = 0.
# The closed form cancels catastrophically as z -> 0 (numerator ~ -7z/120),
# so |z| < _D_SERIES_ZMAX uses the series (truncation error < 1e-13).
_D_SERIES = (-7.0 / 240.0, 1.0 / 2520.0, -1.0 / 44800.0, -1.0 / 5702400.0)
_D_SERIES_ZMAX = 0.05
# Relative Newton step on z below which a candidate counts as converged.
_Z_STEP_TOL = 1e-13
# Relative TOF residual below which a non-improving Newton step means the
# floating-point noise floor of TOF(z) was reached.
_F_NOISE_RTOL = 1e-9


def _tof_and_dtdz(z, r1, r2, A, mu, xp):
    """TOF(z), its analytic derivative dTOF/dz and the physical mask.

    Derivative of the universal-variable TOF (Curtis eq. 5.43):

        sqrt(mu) dt/dz = (y/C)^1.5 [ D(z) + 3 S^2 / (4 C) ]
                         + A/8 [ 3 S sqrt(y) / C + A sqrt(C / y) ]

    with D(z) = (C - 3S/(2C)) / (2z), evaluated by its Taylor series near
    z = 0 (where it reduces to Curtis's z = 0 special case). Shares one
    Stumpff evaluation with TOF(z), replacing the two extra TOF evaluations
    the old central difference needed.
    """
    t, y, C, S, phys = _tof_core(z, r1, r2, A, mu, xp)
    y_s = xp.where(phys, y, 1.0)
    C_s = xp.where(phys, C, 1.0)
    S_s = xp.where(phys, S, 0.0)

    small = xp.abs(z) < _D_SERIES_ZMAX
    z_big = xp.where(small, 1.0, z)
    d_closed = (C_s - 1.5 * S_s / C_s) / (2.0 * z_big)
    c0, c1, c2, c3 = _D_SERIES
    d_series = c0 + z * (c1 + z * (c2 + z * c3))
    D = xp.where(small, d_series, d_closed)

    sqrt_y = xp.sqrt(y_s)
    dtdz = (
        (y_s / C_s) ** 1.5 * (D + 0.75 * S_s**2 / C_s)
        + 0.125 * A * (3.0 * S_s * sqrt_y / C_s + A * xp.sqrt(C_s / y_s))
    ) / xp.sqrt(mu)
    dtdz = xp.where(phys, dtdz, xp.nan)
    return t, dtdz, phys


def _pykep_fallback(r1_vec, r2_vec, tof, mu, prograde):
    """Independent CPU solve via pykep's Izzo solver; None if unusable."""
    try:
        from orbitopt.lambert.cpu import solve_lambert_single
    except Exception:  # noqa: BLE001 -- pykep not installed
        return None
    try:
        sols = solve_lambert_single(r1_vec, r2_vec, tof, mu=mu, max_revs=0, prograde=prograde)
    except Exception:  # noqa: BLE001 -- genuine geometric degeneracy in pykep
        return None
    if not sols:
        return None
    v1, v2 = (np.asarray(v, dtype=np.float64).reshape(3) for v in sols[0])
    if not (np.all(np.isfinite(v1)) and np.all(np.isfinite(v2))):
        return None
    return v1, v2


def solve_lambert_batch(
    r1_vec,
    r2_vec,
    tof,
    mu,
    prograde=True,
    use_gpu=None,
    maxiter=60,
    tol=1e-6,
    cpu_fallback=True,
    angle_tol=1e-6,
):
    """Solve N independent zero-revolution Lambert problems at once.

    Parameters
    ----------
    r1_vec, r2_vec : array-like, shape (N, 3)
        Departure / arrival position vectors (consistent length units).
    tof : array-like, shape (N,)
        Time of flight for each problem (consistent time units with mu).
    mu : float
        Gravitational parameter of the central body (must be > 0).
    prograde : bool
        Assume prograde (counter-clockwise, +z angular momentum) transfers.
    use_gpu : bool or None
        None (default) picks CuPy if available, else numpy. True forces GPU
        and raises if unavailable; False forces CPU.
    maxiter : int
        Upper bound on safeguarded Newton/bisection iterations; the loop
        exits early once every candidate has converged (or was retired as
        unsolvable). ``LambertBatchResult.iterations`` reports the count.
    tol : float
        Relative TOF tolerance: converged iff |tof(z)-tof| < tol*max(tof, 1).
    cpu_fallback : bool
        Re-solve hard/unconverged (but well-formed) candidates with pykep's
        Izzo solver on CPU. Requires pykep; silently skipped if missing.
    angle_tol : float
        Candidates with |sin(transfer angle)| below this (within ~angle_tol
        rad of 0/180 deg) are treated as degenerate for the batched solver
        and sent to the CPU fallback.

    Returns
    -------
    LambertBatchResult with v1, v2 as (N, 3) numpy arrays (always brought
    back to host memory), plus per-item residual, ``valid`` mask (also
    exposed as ``converged``) and ``fallback`` mask. Invalid candidates have
    zero velocities and infinite residual -- never NaN.
    """
    mu = float(mu)
    # The layer raises for a non-finite/non-positive mu, and (unit-agnostically)
    # reports which unit system this mu puts the endpoints in.
    check_grav_parameter(mu, where="orbitopt.lambert.gpu_batch.solve_lambert_batch: mu")

    xp = get_array_module(use_gpu)

    r1_vec = xp.asarray(r1_vec, dtype=xp.float64)
    r2_vec = xp.asarray(r2_vec, dtype=xp.float64)
    if r1_vec.ndim != 2 or r1_vec.shape[1] != 3 or r2_vec.shape != r1_vec.shape:
        raise ValueError("r1_vec and r2_vec must both have shape (N, 3)")
    n = r1_vec.shape[0]
    tof = xp.broadcast_to(xp.asarray(tof, dtype=xp.float64), (n,))

    with np.errstate(all="ignore"):
        v1, v2, residual, valid, hard, iterations = _solve_batch_device(
            xp, r1_vec, r2_vec, tof, mu, prograde, maxiter, tol, angle_tol,
        )

    v1 = np.array(to_numpy(v1), dtype=np.float64)
    v2 = np.array(to_numpy(v2), dtype=np.float64)
    residual = np.array(to_numpy(residual), dtype=np.float64)
    valid = np.array(to_numpy(valid), dtype=bool)
    hard = np.array(to_numpy(hard), dtype=bool)
    fallback = np.zeros(n, dtype=bool)

    if cpu_fallback and hard.any():
        r1_h, r2_h, tof_h = to_numpy(r1_vec), to_numpy(r2_vec), to_numpy(tof)
        for i in np.flatnonzero(hard):
            sol = _pykep_fallback(r1_h[i], r2_h[i], float(tof_h[i]), mu, prograde)
            if sol is None:
                continue
            v1[i], v2[i] = sol
            residual[i] = 0.0
            valid[i] = True
            fallback[i] = True

    v1[~valid] = 0.0
    v2[~valid] = 0.0
    residual[~valid] = np.inf

    return LambertBatchResult(
        v1=v1, v2=v2, residual=residual, valid=valid, fallback=fallback, iterations=iterations,
    )


def _solve_batch_device(xp, r1_vec, r2_vec, tof, mu, prograde, maxiter, tol, angle_tol):
    """Branchless batched solve on xp. Returns (v1, v2, residual, valid, hard)
    where ``hard`` marks well-formed inputs the batched solver could not
    reliably solve (candidates for the CPU fallback)."""
    n = r1_vec.shape[0]

    r1 = xp.linalg.norm(r1_vec, axis=-1)
    r2 = xp.linalg.norm(r2_vec, axis=-1)
    input_ok = (
        xp.all(xp.isfinite(r1_vec), axis=-1) & xp.all(xp.isfinite(r2_vec), axis=-1)
        & xp.isfinite(tof) & (tof > 0.0) & (r1 > 0.0) & (r2 > 0.0)
        & xp.isfinite(r1) & xp.isfinite(r2)
    )
    # Substitute harmless dummy geometry for malformed rows so they can't
    # poison the vectorized math; they are masked out at the end.
    r1 = xp.where(input_ok, r1, 1.0)
    r2 = xp.where(input_ok, r2, 1.0)
    tof = xp.where(input_ok, tof, 1.0)
    r1v = xp.where(input_ok[:, None], r1_vec, 0.0)
    r2v = xp.where(input_ok[:, None], r2_vec, 0.0)

    cross_z = r1v[:, 0] * r2v[:, 1] - r1v[:, 1] * r2v[:, 0]
    # atan2(|r1 x r2|, r1.r2) keeps full relative accuracy for near-0/180 deg
    # transfers, where arccos(dot/(r1 r2)) quantises the angle (cos ~ 1 - 6e-13
    # has only ~4 significant digits of 1 - cos) and corrupts A and the
    # velocities by km/s while still "converging" on the wrong geometry.
    cross = xp.cross(r1v, r2v)
    dnu = xp.arctan2(xp.linalg.norm(cross, axis=-1), xp.sum(r1v * r2v, axis=-1))
    long_way = (cross_z < 0.0) if prograde else (cross_z >= 0.0)
    dnu = xp.where(long_way, 2.0 * xp.pi - dnu, dnu)
    sin_dnu = xp.sin(dnu)
    angle_ok = xp.abs(sin_dnu) >= angle_tol

    A = sin_dnu * xp.sqrt(r1 * r2 / xp.maximum(2.0 * xp.sin(0.5 * dnu) ** 2, 1e-30))

    # z < (2*pi)**2 for any 0-rev transfer. z = (2*pi)**2 is a removable
    # singularity of y(z) (C and z*S - 1 both vanish) whose floating-point
    # evaluation is garbage within ~1e-3 of the boundary, so the upper
    # bracket end sits 1e-2 below it (negligible vs. the ~40-wide domain).
    # tof(z) -> inf as z -> (2*pi)**2, so tof(z_hi) > target for all but
    # absurdly long transfers, which simply fail the bracket check below.
    z_hi_eps = 1e-2
    z_lo = xp.full(n, -(4.0 * xp.pi) ** 2, dtype=xp.float64)
    z_hi = xp.full(n, (2.0 * xp.pi) ** 2 - z_hi_eps, dtype=xp.float64)

    def F(z):
        t, _, _, phys = _tof_of_z(z, r1, r2, A, mu, xp)
        return t - tof, phys

    # Expand the hyperbolic end of the bracket for very short TOFs (fixed
    # number of steps -> still branchless). cosh(sqrt(-z)) stays finite.
    f_lo, _ = F(z_lo)
    for _ in range(6):
        z_lo = xp.where(f_lo >= 0.0, 2.0 * z_lo, z_lo)
        f_lo, _ = F(z_lo)
    f_hi, _ = F(z_hi)
    bracket_ok = (f_lo < 0.0) & (f_hi > 0.0) & xp.isfinite(f_lo) & xp.isfinite(f_hi)

    # Candidates that cannot produce a valid answer (malformed input,
    # degenerate angle, failed bracket) are not iterated at all. The loop
    # works on a compacted copy of the still-active candidates only, so
    # converged ones cost nothing once they retire, and it exits as soon as
    # none are left (one device->host sync per iteration for the size).
    z = xp.clip(xp.zeros(n, dtype=xp.float64), z_lo, z_hi)
    idx = xp.nonzero(input_ok & angle_ok & bracket_ok)[0]
    zw, lo, hi = z[idx], z_lo[idx], z_hi[idx]
    r1w, r2w, Aw, tofw = r1[idx], r2[idx], A[idx], tof[idx]
    f_prev = xp.full(idx.shape[0], xp.inf, dtype=xp.float64)
    iterations = 0
    for _ in range(maxiter):
        if int(idx.shape[0]) == 0:
            break  # every candidate converged/retired: early exit
        iterations += 1
        t, dtdz, phys = _tof_and_dtdz(zw, r1w, r2w, Aw, mu, xp)
        f = t - tofw
        # Maintain the bracket. Non-physical (y <= 0) points count as
        # "too short" -- the true TOF tends to 0 there.
        too_short = f < 0.0
        lo = xp.where(too_short, zw, lo)
        hi = xp.where(too_short, hi, zw)

        # Analytic Newton step, damped by the bracket: any step that is
        # non-finite, has a non-positive derivative or leaves (lo, hi) is
        # replaced by bisection.
        deriv_ok = phys & xp.isfinite(dtdz) & (dtdz > 0.0)
        z_newton = zw - f / xp.where(deriv_ok, dtdz, 1.0)
        newton_ok = deriv_ok & xp.isfinite(z_newton) & (z_newton > lo) & (z_newton < hi)
        z_next = xp.where(newton_ok, z_newton, 0.5 * (lo + hi))
        # An exact root sits on the bracket edge just set (hi = z), so the
        # zero Newton step is "outside" (lo, hi); keep z rather than bisect.
        z_next = xp.where(f == 0.0, zw, z_next)

        # Early termination. ``tol`` is only the *acceptance* threshold (a
        # 1e-6 TOF residual can still be ~m/s in velocity), so a candidate
        # is retired only once it is inside ``tol`` *and* Newton has reached
        # the floating-point noise floor: exact residual, negligible step, or
        # -- with |f| already far below ``tol`` (so a slow bisection phase
        # can't trigger it) -- the last step failed to halve |f|.
        # Retired candidates keep the current z, whose residual was just
        # checked; the final residual is re-verified after the loop anyway.
        abs_f = xp.abs(f)
        tof_scale = xp.maximum(tofw, 1.0)
        resid_ok = phys & xp.isfinite(f) & (abs_f < tol * tof_scale)
        stalled = (
            (f == 0.0)
            | (newton_ok & (xp.abs(z_newton - zw) <= _Z_STEP_TOL * xp.maximum(xp.abs(zw), 1.0)))
            | ((abs_f >= 0.5 * f_prev) & (abs_f < _F_NOISE_RTOL * tof_scale))
        )
        retire = resid_ok & stalled
        # z_next == z is a fixed point of this deterministic update (e.g. a
        # bracket collapsed to one float): further iterations cannot help.
        stuck = z_next == zw
        zw = xp.where(retire, zw, z_next)
        z[idx] = zw

        keep = xp.nonzero(~(retire | stuck))[0]
        idx, zw, lo, hi, f_prev = idx[keep], zw[keep], lo[keep], hi[keep], abs_f[keep]
        r1w, r2w, Aw, tofw = r1w[keep], r2w[keep], Aw[keep], tofw[keep]

    tof_final, y, C, phys = _tof_of_z(z, r1, r2, A, mu, xp)
    residual = xp.abs(tof_final - tof)
    converged = phys & xp.isfinite(residual) & (residual < tol * xp.maximum(tof, 1.0))

    y_s = xp.where(phys, y, 1.0)
    f_coef = 1.0 - y_s / r1
    g_coef = A * xp.sqrt(y_s / mu)
    gdot_coef = 1.0 - y_s / r2
    g_ok = xp.isfinite(g_coef) & (xp.abs(g_coef) > 1e-10 * tof)
    g_safe = xp.where(g_ok, g_coef, 1.0)

    v1 = (r2v - f_coef[:, None] * r1v) / g_safe[:, None]
    v2 = (gdot_coef[:, None] * r2v - r1v) / g_safe[:, None]
    v_ok = xp.all(xp.isfinite(v1), axis=-1) & xp.all(xp.isfinite(v2), axis=-1)

    valid = input_ok & angle_ok & bracket_ok & converged & (y > 0.0) & (C > 0.0) & g_ok & v_ok
    hard = input_ok & ~valid
    return v1, v2, residual, valid, hard, iterations
