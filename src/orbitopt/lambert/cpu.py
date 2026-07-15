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
    lp = pk.lambert_problem(
        r1=list(r1), r2=list(r2), tof=float(tof), mu=mu,
        max_revs=max_revs, cw=not prograde,
    )
    v1s = lp.get_v1()
    v2s = lp.get_v2()
    return [(np.asarray(v1), np.asarray(v2)) for v1, v2 in zip(v1s, v2s)]
