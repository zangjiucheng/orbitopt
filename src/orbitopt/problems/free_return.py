"""pygmo UDP for coarse, GPU-batched global search over the translunar-
injection burn vector, screened with orbitopt.dynamics.nbody_gpu's simplified
Earth-Moon(-Sun) propagator to find a burn that brings a spacecraft close to
a specific target distance from the Moon.

This is the "cheap, wide search" half of the two-stage pipeline promised by
the framework's design: a population of candidate burns is batch-evaluated
in one GPU call per generation (same wiring as
orbitopt.problems.transfer_2body, via batch_fitness/optimize.runner), and
the best candidate found here is meant to seed
orbitopt.verify.differential_correction.target_lunar_flyby for precise,
high-fidelity refinement -- NOT to be trusted as a final answer on its own,
since it inherits nbody_gpu's two-body-osculating-ephemeris approximation
(known to be off by ~thousands of km over a several-day coast, see
orbitopt.dynamics.nbody_gpu's module docs).

Units note: orbitopt.dynamics.nbody_gpu works in km/km-s (matching its own
validation/docstrings), while orbitopt.verify.tudat_propagate and
orbitopt.verify.differential_correction work in SI meters/m-s (matching
tudatpy's convention) -- this module's decision vector and objective are in
km/km-s to match nbody_gpu, and the ``as_meters`` helper below is the
explicit conversion point when handing a result to the refinement stage.
"""
from __future__ import annotations

import numpy as np

from orbitopt.core.problem import OrbitOptProblem
from orbitopt.dynamics import nbody_gpu as nb


class FreeReturnScreeningProblem(OrbitOptProblem):
    """Decision vector: [dvx, dvy, dvz] (km/s), the TLI burn added to
    ``v0_pre_burn`` at a fixed departure state/epoch. Objective: absolute
    difference (km) between ``target_distance_km`` and the closest approach
    to the Moon reached during ``coast_days`` of coarse GPU propagation --
    zero means "this burn's coarse trajectory grazes the target distance
    from the Moon," which is what a global optimizer drives toward.
    """

    def __init__(
        self,
        r0_km,
        v0_pre_burn_km_s,
        epoch0_seconds,
        target_distance_km,
        coast_days,
        dv_bounds_km_s=(-4.5, 4.5),
        n_steps=400,
        use_gpu=None,
    ):
        super().__init__(use_gpu=use_gpu)
        self.r0_km = np.asarray(r0_km, dtype=float)
        self.v0_pre_burn_km_s = np.asarray(v0_pre_burn_km_s, dtype=float)
        self.epoch0_seconds = float(epoch0_seconds)
        self.target_distance_km = float(target_distance_km)
        self.coast_seconds = float(coast_days) * 86400.0
        self.dv_bounds_km_s = dv_bounds_km_s
        self.n_steps = n_steps

    def get_bounds(self):
        lo, hi = self.dv_bounds_km_s
        lo = np.broadcast_to(np.asarray(lo, dtype=float), (3,))
        hi = np.broadcast_to(np.asarray(hi, dtype=float), (3,))
        return lo.tolist(), hi.tolist()

    def fitness(self, dv):
        return self.batch_fitness(np.asarray(dv, dtype=float))

    def batch_fitness(self, dvs):
        dv_matrix = np.asarray(dvs, dtype=float).reshape(-1, 3)
        n = len(dv_matrix)

        r0_batch = np.broadcast_to(self.r0_km, (n, 3))
        v0_batch = self.v0_pre_burn_km_s[None, :] + dv_matrix

        result = nb.propagate_spacecraft_batch(
            r0_batch, v0_batch, self.coast_seconds, self.n_steps,
            mu_earth=nb.MU_EARTH_KM3_S2, mu_moon=nb.MU_MOON_KM3_S2,
            epoch0=self.epoch0_seconds, use_gpu=self.use_gpu,
        )

        moon_r, _ = nb.moon_state_batch(
            result.t_abs.ravel() - self.epoch0_seconds, use_gpu=self.use_gpu,
            seed_epoch_ephemeris_seconds=self.epoch0_seconds,
        )
        moon_r = moon_r.reshape(result.t_abs.shape[0], n, 3)

        distance_km = np.linalg.norm(result.r - moon_r, axis=-1)
        min_distance_km = distance_km.min(axis=0)
        min_distance_epoch_s = result.t_abs[np.argmin(distance_km, axis=0), np.arange(n)]

        self._last_min_distance_epoch_s = min_distance_epoch_s
        return np.abs(min_distance_km - self.target_distance_km)

    def time_to_closest_approach(self, dv):
        """Convenience for callers (e.g. the free_return example) that want
        a coast-duration guess for the winning candidate: runs one more
        batch_fitness of size 1 and returns the elapsed seconds (from
        epoch0) to the coarse model's closest Moon approach, since
        batch_fitness doesn't return it directly (pygmo's UDP fitness
        contract is a plain float vector, not a richer struct).
        """
        self.batch_fitness(np.asarray(dv, dtype=float))
        return float(self._last_min_distance_epoch_s[0] - self.epoch0_seconds)


def as_meters(r_or_v_km):
    """Explicit km -> m unit conversion at the handoff from this module's
    (nbody_gpu-matching) km convention to tudat_propagate/
    differential_correction's SI-meters convention.
    """
    return np.asarray(r_or_v_km, dtype=float) * 1000.0
