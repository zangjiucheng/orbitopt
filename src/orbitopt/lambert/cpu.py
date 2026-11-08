"""Thin, clean wrapper around pykep's (Izzo, multi-revolution capable)
Lambert solver, used as the reference/ground-truth implementation and for
one-off (non-batched) solves where GPU dispatch overhead isn't worth it.
"""
from __future__ import annotations

import numpy as np
import pykep as pk


def solve_lambert_single(r1, r2, tof, mu=pk.MU_SUN, max_revs=0, prograde=True):
    """Solve a single Lambert problem via pykep.lambert_problem.

    Returns a list of (v1, v2) tuples, one per admissible revolution/branch
    (index 0 is always the 0-revolution solution).
    """
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
