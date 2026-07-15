"""Single-leg, impulsive two-body transfer problem: choose departure epoch
t0 and time of flight to minimize total hyperbolic-excess delta-v (departure
+ arrival), for a Lambert arc between two ephemeris bodies.

Decision vector: [t0_mjd2000, tof_days].
Objective: |v1 - v_dep_planet(t0)| + |v2 - v_arr_planet(t0 + tof)|.

This is deliberately the simplest possible interplanetary-transfer UDP: it
exists to demonstrate the GPU batch_fitness() wiring end-to-end (see
orbitopt.optimize.runner.run_optimization). orbitopt.problems.mga builds a
more realistic multi-leg problem on top of pykep.trajopt directly.
"""
from __future__ import annotations

import numpy as np
import pykep as pk

from orbitopt.core.problem import OrbitOptProblem
from orbitopt.lambert.gpu_batch import solve_lambert_batch


class Transfer2BodyProblem(OrbitOptProblem):
    def __init__(
        self,
        departure_body,
        arrival_body,
        t0_bounds,
        tof_bounds,
        mu=pk.MU_SUN,
        use_gpu=None,
    ):
        super().__init__(use_gpu=use_gpu)
        self.departure_body = departure_body
        self.arrival_body = arrival_body
        self.t0_bounds = t0_bounds
        self.tof_bounds = tof_bounds
        self.mu = mu

    def get_bounds(self):
        lo = [self.t0_bounds[0], self.tof_bounds[0]]
        hi = [self.t0_bounds[1], self.tof_bounds[1]]
        return lo, hi

    def _ephemerides(self, t0, tof):
        r1, v1p = self.departure_body.eph(pk.epoch(float(t0)))
        r2, v2p = self.arrival_body.eph(pk.epoch(float(t0) + float(tof)))
        return np.asarray(r1), np.asarray(v1p), np.asarray(r2), np.asarray(v2p)

    def fitness(self, dv):
        t0, tof = dv
        if tof <= 0:
            return [1e6]
        r1, v1p, r2, v2p = self._ephemerides(t0, tof)
        res = solve_lambert_batch(
            r1[None, :], r2[None, :], np.array([tof * pk.DAY2SEC]),
            self.mu, use_gpu=False,
        )
        if not res.converged[0]:
            return [1e6]
        dv_total = np.linalg.norm(res.v1[0] - v1p) + np.linalg.norm(res.v2[0] - v2p)
        return [dv_total]

    def batch_fitness(self, dvs):
        n = 2
        dv_matrix = np.asarray(dvs, dtype=float).reshape(-1, n)
        t0s, tofs = dv_matrix[:, 0], dv_matrix[:, 1]

        bad = tofs <= 0
        tofs_safe = np.where(bad, 1.0, tofs)

        r1 = np.empty((len(dv_matrix), 3))
        v1p = np.empty((len(dv_matrix), 3))
        r2 = np.empty((len(dv_matrix), 3))
        v2p = np.empty((len(dv_matrix), 3))
        for i, (t0, tof) in enumerate(zip(t0s, tofs_safe)):
            r1[i], v1p[i], r2[i], v2p[i] = self._ephemerides(t0, tof)

        result = solve_lambert_batch(
            r1, r2, tofs_safe * pk.DAY2SEC, self.mu, use_gpu=self.use_gpu,
        )

        dv_total = np.linalg.norm(result.v1 - v1p, axis=1) + np.linalg.norm(
            result.v2 - v2p, axis=1
        )
        dv_total = np.where(bad | ~result.converged, 1e6, dv_total)
        return dv_total
