"""Numba-CPU build of ``orbitopt.dynamics.nbody_fused``'s scalar
per-candidate kernel (imported lazily -- only when numba is installed).

The jitted functions live at module level on purpose: numba's on-disk
cache (``cache=True``) only hits across processes for module-level
functions; the same kernel built inside a function/closure recompiles
(~5 s) in every new interpreter. The source is still the shared
``_accel_py``/``_make_candidate`` code, so the numba build cannot drift
from the Python reference.
"""
from __future__ import annotations

import numba

from orbitopt.dynamics.nbody_fused import _accel_py, _make_candidate

_accel = numba.njit(inline="always", cache=True)(_accel_py)
_candidate = numba.njit(inline="always", cache=True)(_make_candidate(_accel))


@numba.njit(parallel=True, cache=True)
def kernel(r0, v0, dt_arr, epoch0, grid, ind, gidx, mus, n_steps, mu_earth,
           min_d, min_t, min_s, r_out, v_out, t_out):
    for i in numba.prange(r0.shape[0]):
        _candidate(i, r0, v0, dt_arr, epoch0, grid, ind, gidx, mus, n_steps, mu_earth,
                   min_d, min_t, min_s, r_out, v_out, t_out)
