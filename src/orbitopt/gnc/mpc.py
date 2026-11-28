"""Constrained receding-horizon MPC on the E13-05 seam.

The constrained arm: at every feedback tick it solves a condensed
finite-horizon QP over the same CW impulse model and the same regulation
error as the LQR arm (:mod:`orbitopt.gnc.lqr`), but the constraint set is
IN the optimizer, not bolted on afterwards:

* per-pulse thrust cap ``|u_j| <= u_max`` for every step of the horizon;
* the law's remaining commanded delta-v budget (``budget_available_m_s``:
  the ledger remainder, less the reserve held for a scheduled terminal burn
  when one is configured), with the true ``sum |u_j|``
  replaced by its smooth surrogate ``sum sqrt(|u_j|^2 + eps^2)`` (eps = 1
  mm/s -- the difference is far below the execution error);
* keep-out: predicted ``|rho_k| >= keep_out_radius_m`` (nonconvex by
  nature -- a sphere constraint on an approaching vehicle; warm-starting
  from the shifted previous solution is what keeps the solver on the near
  side of the shell rather than flipping over the target);
* corridor: predicted ``|z_k| <= cross_track_max_m``, and the terminal
  cone about -V-bar inside its activation range, written with a
  range-disactivation penalty so the gradient stays smooth where the cone
  stops applying;
* velocity profile: predicted relative speed under the range-dependent
  ceiling from the mission's approach-speed family.

The terminal cost is the LQR cost-to-go ``P[steps_remaining - N]`` from the
shared Riccati sequence, so a horizon that is short compared to the
transfer (120 s of a ~2000 s one) still prices the tail of the trajectory
the way the unconstrained optimum would.

Solver: ``scipy.optimize.minimize(method="SLSQP")`` on the condensed
3N-vector of pulses with analytic cost gradient and finite-difference
constraint gradients. If a solve fails or returns non-finite pulses the law
does NOT freeze: it issues the frozen-gain LQR step (recorded as a
fallback), because a guidance law that drops a tick costs the mission more
than one that risks a single margin for a single interval. Fallback and
solve-failure counts feed the benchmark's story.

Prediction is linear-CW -- DESIGN ASSUMPTION of the research: the mission
plant carries J2/drag, but over a 120 s horizon their relative effect is
millimetre-to-centimetre, orders below the 500 m keep-out margin the
constraint cares about.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .guidance import GuidanceLawError
from .lqr import FiniteHorizonCwLaw, cw_impulse_matrices

_BUDGET_SMOOTH_EPS_M_S = 1.0e-3
_CONE_DISENGAGE_GAIN = 10.0


class CwMpcLaw(FiniteHorizonCwLaw):
    """Receding-horizon constrained guidance law; constructor arguments are
    the base law's plus ``horizon_steps``. The budget the QP sees is
    ``constraints.budget_m_s`` minus the ledger already commanded (planned
    departure burn included)."""

    def __init__(self, *, horizon_steps: int = 12, q_diag, r_diag, **kwargs):
        super().__init__(q_diag=q_diag, r_diag=r_diag, **kwargs)
        if int(horizon_steps) < 2:
            raise GuidanceLawError(f"horizon_steps must be >= 2, got {horizon_steps!r}")
        self.horizon_steps = int(horizon_steps)
        self.fallbacks: list = []
        self._solve_failures = 0
        self.skip_counts = {"keep_out": 0, "corridor": 0, "approach_speed": 0}
        self._skip_keep_out = self._skip_corridor = self._skip_speed = False
        self._warm_u = np.zeros(3 * self.horizon_steps)
        self._a, self._b = cw_impulse_matrices(self.n, self.cadence_s)
        self._psi, self._theta = _prediction_matrices(self._a, self._b, self.horizon_steps)
        self._q_diag = np.asarray(q_diag, dtype=np.float64).reshape(6)
        self._r_mat = np.diag(np.asarray(r_diag, dtype=np.float64).reshape(3))
        self._w_base = np.kron(np.eye(self.horizon_steps), np.diag(self._q_diag))
        self._h_base = self._theta.T @ (self._w_base @ self._theta) + np.kron(
            np.eye(self.horizon_steps), self._r_mat)

    # -- decision hook --------------------------------------------------------

    def _feedback_pulse(self, err_lvlh: np.ndarray, steps_remaining: int,
                        now: float):
        n_h = self.horizon_steps
        p_term = self._p[max(steps_remaining - n_h, 0)]
        w_bar = self._w_base.copy()
        w_bar[-6:, -6:] += p_term
        psi_e0 = self._psi @ err_lvlh
        lin = self._theta.T @ (w_bar @ psi_e0)
        theta_tail = self._theta[-6:]
        h = self._h_base + theta_tail.T @ (p_term @ theta_tail)

        def cost(u):
            x = psi_e0 + self._theta @ u
            return float(u @ (h @ u) + 2.0 * lin @ u + psi_e0 @ (w_bar @ psi_e0))

        def cost_jac(u):
            return 2.0 * (h @ u + lin)

        budget_left = self.budget_available_m_s
        eps2 = _BUDGET_SMOOTH_EPS_M_S ** 2
        umax = self.constraints.u_max_m_s

        def thrust(u):
            return umax - np.linalg.norm(u.reshape(n_h, 3), axis=1)

        def thrust_jac(u):
            rows = u.reshape(n_h, 3)
            j = np.zeros((n_h, 3 * n_h))
            for k in range(n_h):
                j[k, 3 * k:3 * k + 3] = -rows[k] / max(np.linalg.norm(rows[k]), 1.0e-9)
            return j

        def budget(u):
            return budget_left - self._smooth_dv(u)

        def budget_jac(u):
            rows = u.reshape(n_h, 3)
            s = np.sqrt(rows * rows + eps2).sum(axis=1)
            return (-rows / s[:, None]).ravel()[None, :]

        cons = [{"type": "ineq", "fun": thrust, "jac": thrust_jac},
                {"type": "ineq", "fun": budget, "jac": budget_jac}]
        # A trajectory family whose CURRENT sample already violates it cannot
        # be "satisfied" by any future pulses -- imposing it makes the QP
        # infeasible and the solve useless. Skip that family for this tick
        # (steering back out is the recovery) and count the skip.
        x_abs = self._abs_from_err(err_lvlh, now)
        live = self.constraints.margins(x_abs[0:3], x_abs[3:6])
        self._skip_keep_out = bool(live["keep_out"][0] < 0.0)
        self._skip_corridor = bool(live["cross_track"][0] < 0.0
                                   or live["terminal_cone"][0] < 0.0)
        self._skip_speed = bool(live["approach_speed"][0] < 0.0)
        for flag, name in ((self._skip_keep_out, "keep_out"),
                           (self._skip_corridor, "corridor"),
                           (self._skip_speed, "approach_speed")):
            if flag:
                self.skip_counts[name] += 1
        times = now + self.cadence_s * np.arange(1, n_h + 1)
        cons += self._trajectory_constraints(psi_e0, times)
        res = minimize(cost, self._warm_u, jac=cost_jac, method="SLSQP", constraints=cons,
                       options={"maxiter": 80, "ftol": 1.0e-8})
        u = np.asarray(res.x, dtype=np.float64).reshape(n_h, 3)
        if not res.success or not np.all(np.isfinite(u)):
            self._solve_failures += 1
            self.fallbacks.append(f"steps_remaining={steps_remaining}: SLSQP status "
                                  f"{res.status} ({res.message})")
            self._warm_u = np.zeros_like(self._warm_u)
            return self.unconstrained_pulse(err_lvlh, steps_remaining)
        self._warm_u = np.concatenate((u[1:].reshape(-1), u[-1], u[-1]))[:3 * n_h]
        return u[0]

    def _trajectory_constraints(self, psi_e0, times):
        con = self.constraints
        n_h = self.horizon_steps
        # regulation is about the error against the reference; the constraints
        # live on the absolute geometry: rho_k = rho_ref(t_k) + e_k and
        # rho_dot_k = rho_dot_ref(t_k) + e_dot_k, with the reference rows from
        # the uplinked table. The reference does not depend on the decisions,
        # so every Jacobian below stays the plain error-dynamics one.
        ref = self._ref_rows_at(times)
        ref_rho, ref_rhod = ref[:, :3], ref[:, 3:6]
        th = self._theta.reshape(n_h, 6, 3 * n_h)
        tp, tv = th[:, :3, :], th[:, 3:, :]

        def states(u):
            x = (psi_e0 + self._theta @ u).reshape(n_h, 6)
            return x[:, :3] + ref_rho, x[:, 3:] + ref_rhod

        def _unit(v):
            n = np.linalg.norm(v, axis=1)
            return v / np.maximum(n, 1.0e-9)[:, None], n

        def _u_unit(u):
            m = np.linalg.norm(u.reshape(n_h, 3), axis=1)
            return u.reshape(n_h, 3) / np.maximum(m, 1.0e-9)[:, None], m

        def _skip(fun_val=1.0):
            return {"type": "ineq", "fun": lambda u: np.full(n_h, fun_val),
                    "jac": lambda u: np.zeros((n_h, 3 * n_h))}

        if self._skip_keep_out:
            keep_out = _skip()
        else:
            def keep_out(u):
                rho, _ = states(u)
                return np.linalg.norm(rho, axis=1) - con.keep_out_radius_m

            def keep_out_jac(u):
                rho, _ = states(u)
                rh, _ = _unit(rho)
                return np.einsum("ki,kia->ka", rh, tp)

            keep_out = {"type": "ineq", "fun": keep_out, "jac": keep_out_jac}
        if self._skip_corridor:
            cross_track = cone = _skip()
        else:
            def cross_track(u):
                rho, _ = states(u)
                return con.cross_track_max_m - np.abs(rho[:, 2])

            def cross_track_jac(u):
                rho, _ = states(u)
                return -np.sign(rho[:, 2])[:, None] * tp[:, 2, :]

            def cone(u):
                rho, _ = states(u)
                rng = np.linalg.norm(rho, axis=1)
                margin = -rho[:, 1] - con.cone_cos_half * rng
                return margin + _CONE_DISENGAGE_GAIN * np.maximum(rng - con.cone_range_m, 0.0)

            def cone_jac(u):
                rho, _ = states(u)
                rh, rng = _unit(rho)
                j = -tp[:, 1, :] - con.cone_cos_half * np.einsum("ki,kia->ka", rh, tp)
                out = rng > con.cone_range_m
                if out.any():
                    j[out] += _CONE_DISENGAGE_GAIN * np.einsum("ki,kia->ka", rh[out], tp[out])
                return j

            cross_track = {"type": "ineq", "fun": cross_track, "jac": cross_track_jac}
            cone = {"type": "ineq", "fun": cone, "jac": cone_jac}
        if self._skip_speed:
            speed = _skip()
        else:
            knots = np.asarray(con.velocity.range_knots_m)
            caps = np.asarray(con.velocity.cap_m_s)
            seg_slope = np.diff(caps) / np.diff(knots)
            knot_slope = np.concatenate((seg_slope, seg_slope[-1:]))

            def speed(u):
                rho, rhod = states(u)
                return con.velocity.cap(np.linalg.norm(rho, axis=1)) \
                    - np.linalg.norm(rhod, axis=1)

            def speed_jac(u):
                rho, rhod = states(u)
                rh, rng = _unit(rho)
                rhd, _ = _unit(rhod)
                slope = np.interp(rng, knots, knot_slope, left=0.0, right=0.0)
                return (slope[:, None] * np.einsum("ki,kia->ka", rh, tp)
                        - np.einsum("ki,kia->ka", rhd, tv))

            speed = {"type": "ineq", "fun": speed, "jac": speed_jac}
        return [keep_out, cross_track, cone, speed]

    @staticmethod
    def _smooth_dv(u) -> float:
        mags = np.linalg.norm(np.asarray(u).reshape(-1, 3), axis=1)
        return float(np.sum(np.sqrt(mags * mags + _BUDGET_SMOOTH_EPS_M_S ** 2)))

    # -- reporting ------------------------------------------------------------

    @property
    def mpc_stats(self) -> dict:
        out = dict(self.solve_time_stats)
        out.update({"fallbacks": len(self.fallbacks), "solve_failures": self._solve_failures,
                    "horizon_steps": self.horizon_steps, "constraint_skips":
                        dict(self.skip_counts)})
        return out


def _prediction_matrices(a, b, n_steps):
    """Condensed prediction ``X = Psi e0 + Theta U`` over ``n_steps`` CW
    impulse steps: row block ``k`` is the state at step ``k`` with pulse
    coefficients ``A^(k-1-j) B`` for pulse ``j < k`` (an impulse applied at
    the start of step j+1 -- the model of :func:`cw_impulse_matrices`)."""
    psi = np.zeros((6 * n_steps, 6))
    theta = np.zeros((6 * n_steps, 3 * n_steps))
    a_pow = np.eye(6)
    for k in range(1, n_steps + 1):
        a_pow = a @ a_pow
        psi[6 * (k - 1):6 * k] = a_pow
        a_pow_b = np.eye(6)
        for j in range(k - 1, -1, -1):
            theta[6 * (k - 1):6 * k, 3 * j:3 * (j + 1)] = a_pow_b @ b
            a_pow_b = a @ a_pow_b
    return psi, theta
