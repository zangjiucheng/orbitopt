"""Thin, clean wrapper around pykep's (Izzo, multi-revolution capable)
Lambert solver, used as the reference/ground-truth implementation and for
one-off (non-batched) solves where GPU dispatch overhead isn't worth it.
"""
from __future__ import annotations

import numpy as np
import pykep as pk


def _frame_with_z_along(normal, r1):
    """Rows of a rotation matrix whose third row is ``normal`` (unit) and
    whose first row is ``r1``'s component perpendicular to it."""
    z = np.asarray(normal, dtype=float)
    z = z / np.linalg.norm(z)
    x = np.asarray(r1, dtype=float) - np.dot(r1, z) * z
    if np.linalg.norm(x) < 1e-12 * np.linalg.norm(r1):
        raise ValueError("plane_normal is parallel to r1; it cannot define the transfer plane")
    x = x / np.linalg.norm(x)
    return np.vstack([x, np.cross(z, x), z])


def solve_lambert_single(r1, r2, tof, mu=pk.MU_SUN, max_revs=0, prograde=True, plane_normal=None):
    """Solve a single Lambert problem via pykep.lambert_problem.

    Returns a list of (v1, v2) tuples, one per admissible revolution/branch
    (index 0 is always the 0-revolution solution).

    By default pykep decides the transfer sense from the sign of the z
    component of ``r1 x r2`` ("prograde" = counterclockwise about +z). That
    is undefined -- and decided by rounding noise -- when the transfer plane
    contains the z axis (e.g. a parking orbit built in the plane spanned by
    the target direction and the ecliptic pole): a sub-millimetre change in
    the inputs can then flip between the short- and long-way solutions or
    make pykep raise. Pass ``plane_normal`` (e.g. the departure orbit's
    angular momentum ``r x v``) to make "prograde" mean counterclockwise
    about that vector instead; the problem is solved in a frame whose +z is
    ``plane_normal`` and the velocities rotated back.
    """
    if plane_normal is not None:
        rot = _frame_with_z_along(plane_normal, r1)
        sols = solve_lambert_single(
            rot @ np.asarray(r1, dtype=float), rot @ np.asarray(r2, dtype=float),
            tof, mu=mu, max_revs=max_revs, prograde=prograde,
        )
        return [(rot.T @ v1, rot.T @ v2) for v1, v2 in sols]
    # Positional args: pykep 2 names them (r1, r2, tof, mu, cw, max_revs),
    # pykep 3 (r0, r1, tof, mu, cw, multi_revs) -- same order, so positional
    # calling works on both.
    lp = pk.lambert_problem(
        list(map(float, r1)), list(map(float, r2)), float(tof), float(mu),
        not prograde, int(max_revs),
    )
    if hasattr(lp, "get_v1"):  # pykep 2.x
        v1s, v2s = lp.get_v1(), lp.get_v2()
    else:  # pykep 3.x
        v1s, v2s = lp.v0, lp.v1
    return [(np.asarray(v1), np.asarray(v2)) for v1, v2 in zip(v1s, v2s)]
