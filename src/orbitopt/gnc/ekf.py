"""Relative-navigation Extended Kalman Filter (E13-02, task #419).

The discrete nonlinear state space, exactly as the task states it:

    x_{k+1} = f(x_k, u_k) + w_k        z_k = h(x_k) + v_k

with ``x`` the 6-vector ``(rho, rho_dot)`` of the chaser relative to the
target in the target's LVLH frame (E12-02's axes: x radial outward, y
along-track, z cross-track; metres and m/s, :data:`orbitopt.physics_io.
RELATIVE_STATE` band), ``u`` the commanded LVLH delta-v impulse applied at
the START of the interval (it enters the velocity block additively --
``B = [[0],[I]]``), and ``w ~ N(0, Q)``, ``v ~ N(0, R)``. The filter runs
on the LVLH state because that is where the process models live; the
contract records it publishes (:func:`make_estimate`) carry the inertial
difference state like every other E13-01 record does, converted through the
known chief state at the update instant.

Reuse, per the task ("process model reuses E12-02"): the two process models
here add no new orbital dynamics. :class:`CwProcess` wraps
:func:`orbitopt.relative.cw.cw_state_transition` (the fast planning model,
linear in its own assumptions but a legitimately nonlinear node of the
filter: CW is what a CW-based planner flies, and E13-02 must estimate
against it); :class:`NonlinearRelativeProcess` wraps
:func:`orbitopt.relative.propagation.propagate_pair` (two-body + optional
J2, RK4) and obtains the Jacobian ``F = df/dx`` by batched central
differences of that same propagator. The committed measured cost of using
either one where the other was true is the mismatch study in
``docs/benchmarks/e13_02_relative_nav_ekf.json``.

Measurement models (:class:`RelativePosition`, :class:`RelativeVelocity`,
:class:`RangeRate`, :class:`RangeAngle`, :class:`Stacked`) are the seam to
the E13-03 sensor simulator: each knows its native observables in the LVLH
convention declared below, its analytic Jacobian, and how to
:meth:`~MeasurementModel.extract` the measurement and covariance out of a
gnc-contracts ``Observation`` record -- position/velocity readings from the
contract fields themselves, native range/angle readings from the record's
``metadata`` under the keys :data:`NATIVE_MEASUREMENT_KEY`/
:data:`NATIVE_COVARIANCE_KEY`. Latency is handled the E13-01 way: the
observation's ``t_available_s`` governs when the harness may ingest it, not
this module.

Consistency bookkeeping (NEES/NIS against declared chi-square intervals,
coverage, cluster statistics) is offline-evaluator work and lives in
:mod:`orbitopt.gnc.nav_metrics`; this module never reads plant truth.

Design assumptions (declared, per the epic's rule):

* white-noise-acceleration process noise, piecewise-constant over the
  interval: ``Q = q * [[dt^3/3 I, dt^2/2 I], [dt^2/2 I, dt I]]`` -- one
  unmodelled-acceleration density ``q`` [m^2/s^3] per axis, no CW
  gravity-gradient shaping;
* LVLH angles: azimuth about the orbit normal from the radial axis
  ``az = atan2(rho_y, rho_x)``, elevation ``el = asin(rho_z/|rho|)`` --
  the camera-free "what are the LVLH angles of the LOS" convention;
  both are undefined for a purely radial LOS and raise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from orbitopt.mission.contract import Epoch
from orbitopt.mission.multispacecraft import RelativeState
from orbitopt.relative.cw import cw_state_transition
from orbitopt.relative.frames import eci_to_lvlh, lvlh_to_eci
from orbitopt.relative.propagation import EARTH_MU, propagate_pair

from .contracts import Estimate, GNCContractError, Observation

#: metadata keys under which a sensor record may carry native (non-Cartesian)
#: observables for :meth:`MeasurementModel.extract` of the non-Cartesian
#: models. E13-03 writes native readings here; the filter reads nothing else.
NATIVE_MEASUREMENT_KEY = "native_measurement"
NATIVE_COVARIANCE_KEY = "native_measurement_covariance"

_EPS_ANGULAR = 1.0e-6  # rad; LOS within this of the radial axis: az undefined


def white_noise_acceleration_q(q: float, dt: float) -> np.ndarray:
    """The 6x6 white-noise-acceleration process-noise covariance for density
    ``q`` [m^2/s^3] per axis over interval ``dt`` [s] (piecewise-constant
    acceleration, per-axis identical, ordered ``(rho, rho_dot)``)."""
    q = float(q)
    dt = float(dt)
    if not (math.isfinite(q) and q >= 0.0):
        raise ValueError(f"process noise density q must be finite and >= 0, got {q!r}")
    if not (math.isfinite(dt) and dt > 0.0):
        raise ValueError(f"predict interval dt must be finite and > 0, got {dt!r}")
    i3 = np.eye(3)
    q3 = q * dt
    top = (q * dt ** 3 / 3.0) * i3
    mid = (q3 * dt / 2.0) * i3
    return np.block([[top, mid], [mid, q3 * i3]])


# ---------------------------------------------------------------------------
# process models: x_{k+1} = f(x_k, u_k) + w_k
# ---------------------------------------------------------------------------


class RelativeProcessModel:
    """Interface of a discrete relative-dynamics process model. Concrete
    models add no new orbital physics -- they wrap E12-02."""

    def f(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """Propagate the mean state: f(x, u, dt) -> x_pred (6,)."""
        raise NotImplementedError

    def jacobian(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """F = df/dx at (x, u, dt), shape (6, 6)."""
        raise NotImplementedError

    def propagate_with_jacobian(self, x, u, dt) -> tuple:
        """(f(x,u,dt), F) in one call -- batched models do the propagations
        jointly here; :meth:`advance` then moves the reference frame."""
        return self.f(x, u, dt), self.jacobian(x, u, dt)

    def advance(self, u: np.ndarray, dt: float) -> None:
        """Move the model's own reference chief by ``(u, dt)`` AFTER the
        filter has used f and F at the current reference. Linear models have
        no evolving reference: no-op."""


class CwProcess(RelativeProcessModel):
    """Clohessy-Wiltshire process model: ``f = Phi(n, dt) (x + B u)`` with
    the closed-form STM of :mod:`orbitopt.relative.cw`. The reference frame
    is an unchanging circular orbit, so :meth:`advance` is a no-op."""

    def __init__(self, n: float):
        n = float(n)
        if not (math.isfinite(n) and n > 0.0):
            raise ValueError(f"CwProcess: chief mean motion n must be finite and > 0, got {n!r}")
        self.n = n

    def propagate_with_jacobian(self, x, u, dt) -> tuple:
        x, u = _state_inputs(x, u)
        phi = cw_state_transition(self.n, float(dt))
        xu = x + _B_u(u)
        return phi @ xu, phi.copy()

    def f(self, x, u, dt) -> np.ndarray:
        return self.propagate_with_jacobian(x, u, dt)[0]

    def jacobian(self, x, u, dt) -> np.ndarray:
        return self.propagate_with_jacobian(x, u, dt)[1]


class NonlinearRelativeProcess(RelativeProcessModel):
    """Nonlinear relative process model: the reference chief is propagated
    by :func:`orbitopt.relative.propagation.propagate_pair` (two-body,
    optional J2) and ``f`` re-expresses the propagated pair in the chief's
    new LVLH frame, exactly the E12-02 acceptance model. ``F`` is a batched
    central-difference Jacobian of that same propagator (6 perturbed pairs
    each way ride in the batch with the nominal pair -- one propagator call
    per predict).

    The reference chief is STATE of the model: :meth:`advance` moves it
    forward, and f/jacobian are always evaluated at the current reference
    (the EKF calls them in that order per predict).
    """

    def __init__(self, r0_chief, v0_chief, *, frame: str, mu: float = EARTH_MU,
                 j2: float = 0.0, sub_dt: float | None = None,
                 fd_step: float = 1.0e-5):
        fd_step = float(fd_step)
        if not (math.isfinite(fd_step) and 0.0 < fd_step < 1.0e-2):
            raise ValueError(f"NonlinearRelativeProcess: fd_step must be in (0, 1e-2), "
                             f"got {fd_step!r}")
        self.frame = frame
        self.mu = float(mu)
        self.j2 = float(j2)
        self.sub_dt = None if sub_dt is None else float(sub_dt)
        self.fd_step = fd_step
        self._chief = np.stack((np.asarray(r0_chief, dtype=np.float64).reshape(3),
                                np.asarray(v0_chief, dtype=np.float64).reshape(3)))
        self.t_s = 0.0
        self._next_chief = None

    @property
    def chief_r(self) -> np.ndarray:
        return self._chief[0].copy()

    @property
    def chief_v(self) -> np.ndarray:
        return self._chief[1].copy()

    def _deputy0(self, xs: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Inertial deputy states (K,6) for LVLH relative states ``xs`` at
        the current chief, with velocity impulse ``u`` applied."""
        chief = np.broadcast_to(self._chief.reshape(6), (xs.shape[0], 6))
        xu = xs + _B_u(u)[None, :]
        dr, dv = lvlh_to_eci(chief[:, :3], chief[:, 3:], xu[:, :3], xu[:, 3:],
                             self.frame, where="NonlinearRelativeProcess: deputy")
        return np.concatenate((chief[:, :3] + dr, chief[:, 3:] + dv), axis=1)

    def _propagate(self, xs: np.ndarray, u: np.ndarray, dt: float):
        dep = self._deputy0(xs, u)
        traj = propagate_pair(self._chief[None, 0].repeat(xs.shape[0], axis=0),
                              self._chief[None, 1].repeat(xs.shape[0], axis=0),
                              dep[:, :3], dep[:, 3:], [0.0, float(dt)],
                              frame=self.frame, mu=self.mu, j2=self.j2,
                              sub_dt=self.sub_dt, where="NonlinearRelativeProcess")
        return traj.rel_lvlh[:, -1, :], traj.chief_r[0, -1, :], traj.chief_v[0, -1, :]

    def propagate_with_jacobian(self, x, u, dt) -> tuple:
        x, u = _state_inputs(x, u)
        rows = [x]
        for i in range(6):
            h = self.fd_step * max(abs(float(x[i])), 1.0 if i < 3 else 1.0e-3)
            xp = x.copy()
            xm = x.copy()
            xp[i] += h
            xm[i] -= h
            rows += [xp, xm]
        out, chief_r, chief_v = self._propagate(np.stack(rows), u, dt)
        f = out[0].copy()
        phi = np.empty((6, 6))
        for i in range(6):
            h = self.fd_step * max(abs(float(x[i])), 1.0 if i < 3 else 1.0e-3)
            phi[:, i] = (out[1 + 2 * i] - out[2 + 2 * i]) / (2.0 * h)
        self._next_chief = (chief_r, chief_v)
        return f, phi

    def f(self, x, u, dt) -> np.ndarray:
        x, u = _state_inputs(x, u)
        return self._propagate(x[None, :], u, dt)[0][0].copy()

    def jacobian(self, x, u, dt) -> np.ndarray:
        return self.propagate_with_jacobian(x, u, dt)[1]

    def advance(self, u: np.ndarray, dt: float) -> None:
        """Move the reference chief forward by ``dt``. Takes the chief final
        state from the last :meth:`propagate_with_jacobian` (the nominal row
        of that batch, so chief motion costs no extra propagation); falls
        back to a chief-only propagation."""
        pair = self._next_chief
        if pair is None:
            _, rc, vc = self._propagate(np.zeros((1, 6)), u, dt)
            pair = (rc, vc)
        self._chief = np.stack(pair)
        self.t_s += float(dt)
        self._next_chief = None


def _state_inputs(x, u) -> tuple:
    x = np.asarray(x, dtype=np.float64).reshape(6)
    u = np.zeros(3) if u is None else np.asarray(u, dtype=np.float64).reshape(3)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(u))):
        raise GNCContractError("filter state and control must be finite")
    return x, u


def _B_u(u: np.ndarray) -> np.ndarray:
    out = np.zeros(6)
    out[3:] = u
    return out


def frame_covariance(cov_lvlh, chief_r, chief_v, frame: str) -> np.ndarray:
    """Map a 6x6 covariance from chief-centred LVLH difference coordinates
    into ``frame`` by the chief-fixed linear change of frame.
    :func:`~orbitopt.relative.frames.lvlh_to_eci` is exactly linear in the
    relative state for a fixed chief, so differencing the columns recovers
    the Jacobian to floating-point rounding and the step sizes below are
    not a tuning parameter."""
    cr = np.asarray(chief_r, dtype=np.float64).reshape(3)
    cv = np.asarray(chief_v, dtype=np.float64).reshape(3)
    jac = np.zeros((6, 6))
    for j in range(3):
        for k, (step, is_pos) in enumerate(((1.0, True), (1.0e-6, False))):
            e = np.zeros(3)
            e[j] = step
            dr, dv = lvlh_to_eci(cr, cv, e if is_pos else np.zeros(3),
                                 np.zeros(3) if is_pos else e, frame,
                                 where="frame_covariance")
            col = j if is_pos else 3 + j
            jac[:3, col] = dr / step
            jac[3:, col] = dv / step
    p = np.asarray(cov_lvlh, dtype=np.float64).reshape(6, 6)
    return jac @ p @ jac.T


# ---------------------------------------------------------------------------
# measurement models: z_k = h(x_k) + v_k
# ---------------------------------------------------------------------------


class MeasurementModel:
    """Native LVLH measurement ``z = h(x)``, its analytic Jacobian, its
    covariance extractor from a gnc-contracts ``Observation``, and its dof.

    ``extract(obs, chief_r=None, v_chief=None)`` returns ``(z, R)`` with
    ``z`` shape ``(dof,)`` and ``R`` shape ``(dof, dof)``. Cartesian models
    read the contract's inertial ``relative_state`` (chief state required,
    since the filter state is LVLH); non-Cartesian models read the native
    observables from ``metadata`` (chief state not required -- angles and
    range are already filter-frame quantities by the declared convention).
    """

    #: components of z in native order, for reporting
    components: tuple = ()
    #: True when extract() needs the chief's inertial state at t_sample
    needs_chief_state = False

    @property
    def dof(self) -> int:
        return len(self.components)

    def h(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def extract(self, obs: Observation, chief_r=None, chief_v=None) -> tuple:
        raise NotImplementedError


class RelativePosition(MeasurementModel):
    """Full Cartesian relative position measured in LVLH (the inertial
    contract reading is transformed with the chief state at sample time)."""

    components = ("rho_x", "rho_y", "rho_z")
    needs_chief_state = True

    def h(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(6)
        return x[:3].copy()

    def jacobian(self, x):
        return np.concatenate((np.eye(3), np.zeros((3, 3))), axis=1)

    def extract(self, obs, chief_r=None, chief_v=None):
        rs, cov = _inertial_reading(obs, chief_r, chief_v, position=True)
        return rs[:3], cov


class RelativeVelocity(MeasurementModel):
    """Full Cartesian relative velocity measured in LVLH."""

    components = ("rho_dot_x", "rho_dot_y", "rho_dot_z")
    needs_chief_state = True

    def h(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(6)
        return x[3:].copy()

    def jacobian(self, x):
        return np.concatenate((np.zeros((3, 3)), np.eye(3)), axis=1)

    def extract(self, obs, chief_r=None, chief_v=None):
        rs, cov = _inertial_reading(obs, chief_r, chief_v, position=False)
        return rs[3:6], cov


class RangeRate(MeasurementModel):
    """Range ``|r|`` and/or range rate ``(r . v)/|r|``, in the order given by
    ``components`` (non-empty subset of ``("range", "rate")``; the Doppler
    sensor the benchmark models reports both)."""

    _ALL = ("range", "rate")

    def __init__(self, components: Sequence[str] = ("range", "rate")):
        comps = tuple(components)
        if not comps or not set(comps) <= set(self._ALL):
            raise ValueError(f"RangeRate components must be a non-empty subset of "
                             f"{self._ALL}, got {components!r}")
        self.components = comps

    def _values(self, x):
        r = float(np.linalg.norm(x[:3]))
        if r == 0.0:
            raise ValueError("RangeRate undefined at zero range")
        vals = {"range": r, "rate": float(np.dot(x[:3], x[3:])) / r}
        return np.array([vals[c] for c in self.components])

    def h(self, x):
        return self._values(np.asarray(x, dtype=np.float64).reshape(6))

    def jacobian(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(6)
        r = float(np.linalg.norm(x[:3]))
        if r == 0.0:
            raise ValueError("RangeRate Jacobian undefined at zero range")
        rv = float(np.dot(x[:3], x[3:]))
        g_rng = np.zeros(6)
        g_rng[:3] = x[:3] / r
        g_rate = np.zeros(6)
        g_rate[:3] = x[3:] / r - rv * x[:3] / r ** 3
        g_rate[3:] = x[:3] / r
        rows = {"range": g_rng, "rate": g_rate}
        return np.vstack([rows[c][None, :] for c in self.components])

    def extract(self, obs, chief_r=None, chief_v=None):
        return _native_reading(obs, self.dof)


class RangeAngle(MeasurementModel):
    """Range and/or LVLH LOS angles, in the order given by ``components``
    (subset/any order of ``("range", "azimuth", "elevation")``)."""

    _ALL = ("range", "azimuth", "elevation")
    _ANGULAR = frozenset({"azimuth", "elevation"})

    def __init__(self, components: Sequence[str] = ("range", "azimuth", "elevation")):
        comps = tuple(components)
        if not comps or not set(comps) <= set(self._ALL):
            raise ValueError(f"RangeAngle components must be a non-empty subset of "
                             f"{self._ALL}, got {components!r}")
        self.components = comps

    def h(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(6)
        r = float(np.linalg.norm(x[:3]))
        p = float(math.hypot(x[0], x[1]))
        if r == 0.0:
            raise ValueError("RangeAngle undefined at zero range")
        if self._ANGULAR & set(self.components) and p <= _EPS_ANGULAR * r:
            raise ValueError("RangeAngle azimuth/elevation undefined for a purely radial "
                             f"LOS (separation {r:.3g} m, cross-range {p:.3g} m)")
        values = {"range": r, "azimuth": math.atan2(x[1], x[0]), "elevation": math.asin(x[2] / r)}
        return np.array([values[c] for c in self.components])

    def jacobian(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(6)
        r = float(np.linalg.norm(x[:3]))
        p = float(math.hypot(x[0], x[1]))
        if r == 0.0 or (self._ANGULAR & set(self.components) and p <= _EPS_ANGULAR * r):
            raise ValueError("RangeAngle Jacobian undefined at zero range / radial LOS")
        rows = []
        for c in self.components:
            row = np.zeros(6)
            if c == "range":
                row[:3] = x[:3] / r
            elif c == "azimuth":
                row[0], row[1] = -x[1] / p ** 2, x[0] / p ** 2
            else:
                row[0], row[1], row[2] = (-x[2] * x[0], -x[2] * x[1], p * p)
                row[:3] /= (r * r * p)
            rows.append(row)
        return np.stack(rows)

    def extract(self, obs, chief_r=None, chief_v=None):
        return _native_reading(obs, self.dof)


class Stacked(MeasurementModel):
    """One update from several component models at one instant: z and R are
    the block stacks of the components' (component order fixed at
    construction; each component keeps its own extract path)."""

    def __init__(self, models: Sequence[MeasurementModel]):
        self.models = tuple(models)
        if not self.models:
            raise ValueError("Stacked needs at least one measurement model")
        self.components = tuple(c for m in self.models for c in m.components)
        self.needs_chief_state = any(m.needs_chief_state for m in self.models)

    def h(self, x):
        return np.concatenate([m.h(x) for m in self.models])

    def jacobian(self, x):
        return np.concatenate([m.jacobian(x) for m in self.models], axis=0)

    def extract(self, obs, chief_r=None, chief_v=None):
        raise GNCContractError(
            "Stacked.extract: a stacked update combines one Observation per component -- "
            "call update_from_observations() with one observation per component model")

    def extract_each(self, obs_by_model: Mapping[str, Observation], chief_r=None,
                     chief_v=None) -> tuple:
        """Per-component extraction keyed by component model class name;
        returns the stacked ``(z, R)``."""
        zs, rs = [], []
        for m in self.models:
            o = obs_by_model[type(m).__name__]
            z, r = m.extract(o, chief_r, chief_v)
            if z.shape != (m.dof,) or r.shape != (m.dof, m.dof):
                raise GNCContractError(f"Stacked.extract_each: {type(m).__name__} returned "
                                       f"z{z.shape}/R{r.shape}, expected ({m.dof},)/({m.dof},{m.dof})")
            zs.append(z)
            rs.append(r)
        return np.concatenate(zs), _block_diag(rs)


def _block_diag(mats):
    sizes = [m.shape[0] for m in mats]
    n = sum(sizes)
    out = np.zeros((n, n))
    off = 0
    for m, s in zip(mats, sizes):
        out[off:off + s, off:off + s] = m
        off += s
    return out


def _inertial_reading(obs: Observation, chief_r, chief_v, *, position: bool):
    if chief_r is None or chief_v is None:
        raise GNCContractError(
            f"{obs.sensor_id}: the contract carries the relative state in the observation's "
            f"inertial frame {obs.relative_state.frame!r}; converting to the filter's LVLH "
            "frame needs the chief's inertial state at t_sample_s (pass chief_r/chief_v)")
    dr = np.asarray(obs.relative_state.r_rel_m, dtype=np.float64)
    dv = np.asarray(obs.relative_state.v_rel_m_s, dtype=np.float64)
    cr = np.asarray(chief_r, dtype=np.float64).reshape(3)
    cv = np.asarray(chief_v, dtype=np.float64).reshape(3)
    rho, rho_dot = eci_to_lvlh(cr, cv, dr, dv, obs.relative_state.frame,
                               where=f"extract {obs.sensor_id}")
    state = np.concatenate((rho, rho_dot))
    cp = obs.covariance_position_m2
    cv2 = obs.covariance_velocity_m2_s2
    if position:
        if not cp:
            raise GNCContractError(f"{obs.sensor_id}: position reading but no position covariance")
        return state, np.asarray(cp, dtype=np.float64).reshape(3, 3)
    if not cv2:
        raise GNCContractError(f"{obs.sensor_id}: velocity reading but no velocity covariance "
                               "-- the filter cannot weight an unbounded variance")
    return state, np.asarray(cv2, dtype=np.float64).reshape(3, 3)


def _native_reading(obs: Observation, dof: int) -> tuple:
    z = obs.metadata.get(NATIVE_MEASUREMENT_KEY)
    r = obs.metadata.get(NATIVE_COVARIANCE_KEY)
    if z is None or r is None:
        raise GNCContractError(
            f"{obs.sensor_id}: native measurement not in metadata under {NATIVE_MEASUREMENT_KEY!r}"
            f"/{NATIVE_COVARIANCE_KEY!r} -- a non-Cartesian sensor model reads its observables "
            "from the record the sensor wrote, it does not re-derive them from the "
            "contract's Cartesian field (that would double-count the noise)")
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    r = np.asarray(r, dtype=np.float64).reshape(dof, dof)
    if z.shape != (dof,):
        raise GNCContractError(f"{obs.sensor_id}: native measurement has {z.size} components, "
                               f"model expects {dof}")
    return z, r


# ---------------------------------------------------------------------------
# the filter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UpdateReport:
    """One measurement update, exactly what the consistency study consumes:
    innovation, innovation covariance, NIS (= y' S^-1 y, dof = len(z)), and
    the observation ids that produced it."""

    innovation: tuple
    innovation_cov: tuple
    nis: float
    dof: int
    observation_ids: tuple
    t_update_s: float


class RelativeNavEKF:
    """The EKF itself: predict propagates ``(x, P)`` with a process model
    and white-noise-acceleration ``Q``; update fuses one measurement (or a
    stacked set) with the Joseph form for numerical hygiene. The filter
    tracks its own elapsed time (the clock's estimator cadence drives when
    the harness calls predict/update -- this class is time-agnostic, it just
    advances by what it is told)."""

    def __init__(self, process: RelativeProcessModel, x0, P0, *, q_accel_m2_s3: float):
        x0 = np.asarray(x0, dtype=np.float64).reshape(6)
        p0 = np.asarray(P0, dtype=np.float64).reshape(6, 6)
        if not np.all(np.isfinite(x0)):
            raise GNCContractError("EKF x0 must be finite")
        if not np.all(np.isfinite(p0)):
            raise GNCContractError("EKF P0 must be finite")
        if np.any(np.diag(p0) <= 0.0):
            raise GNCContractError("EKF P0 needs positive variances -- a zero-variance axis "
                                   "claims exact knowledge and never recovers")
        q = float(q_accel_m2_s3)
        if not (math.isfinite(q) and q >= 0.0):
            raise GNCContractError(f"EKF q_accel_m2_s3 must be finite and >= 0, got {q!r}")
        self.process = process
        self.q = q
        self.x = x0
        self.P = 0.5 * (p0 + p0.T)
        self.t_s = 0.0
        self.n_predict = 0
        self.n_update = 0
        self._source_ids: list = []
        self._pending_ids: list = []
        self._last_report: UpdateReport | None = None

    def predict(self, dt: float, u=None) -> tuple:
        """Time update over ``dt`` [s] with control impulse ``u`` (LVLH
        delta-v, or None): x <- f(x, u, dt), P <- F P F' + Q, reference
        model advanced."""
        x_new, phi = self.process.propagate_with_jacobian(self.x, _u_or_zero(u), dt)
        self.process.advance(_u_or_zero(u), dt)
        self.P = phi @ self.P @ phi.T + white_noise_acceleration_q(self.q, dt)
        self.P = 0.5 * (self.P + self.P.T)
        self.x = x_new
        self.t_s += float(dt)
        self.n_predict += 1
        return self.x.copy(), self.P.copy()

    def update(self, z, R, model: MeasurementModel) -> UpdateReport:
        """Measurement update with a native (filter-frame) ``(z, R)`` pair;
        prefer :meth:`update_from_observation` on live records."""
        z = np.asarray(z, dtype=np.float64).reshape(model.dof)
        r = np.asarray(R, dtype=np.float64).reshape(model.dof, model.dof)
        if not (np.all(np.isfinite(z)) and np.all(np.isfinite(r))):
            raise GNCContractError("measurement and covariance must be finite")
        y = z - model.h(self.x)
        hh = model.jacobian(self.x)
        s = hh @ self.P @ hh.T + r
        s = 0.5 * (s + s.T)
        try:
            s_inv_y = np.linalg.solve(s, y)
        except np.linalg.LinAlgError as exc:
            raise GNCContractError("innovation covariance is singular -- check the "
                                   "measurement covariance is positive definite") from exc
        nis = float(y @ s_inv_y)
        k = self.P @ hh.T @ np.linalg.inv(s)
        ikh = np.eye(6) - k @ hh
        self.P = ikh @ self.P @ ikh.T + k @ r @ k.T
        self.P = 0.5 * (self.P + self.P.T)
        self.x = self.x + k @ y
        self.n_update += 1
        report = UpdateReport(tuple(float(v) for v in y),
                              tuple(float(v) for v in s.reshape(-1)),
                              nis, model.dof, tuple(self._pending_ids), self.t_s)
        self._source_ids.extend(self._pending_ids)
        self._pending_ids = []
        self._last_report = report
        return report

    def update_from_observation(self, obs: Observation, model: MeasurementModel,
                                chief_r=None, chief_v=None) -> UpdateReport:
        """Contract seam: consume one gnc-contracts ``Observation`` through
        its model's :meth:`~MeasurementModel.extract`."""
        z, r = model.extract(obs, chief_r, chief_v)
        self._pending_ids = [obs.record_id]
        return self.update(z, r, model)

    def update_from_observations(self, obs_by_model: Mapping[str, Observation],
                                 stacked: Stacked, chief_r=None, chief_v=None) -> UpdateReport:
        """One fused update from a :class:`Stacked` model and one observation
        per component (keyed by component class name)."""
        z, r = stacked.extract_each(obs_by_model, chief_r, chief_v)
        self._pending_ids = [obs_by_model[type(m).__name__].record_id for m in stacked.models]
        return self.update(z, r, stacked)

    @property
    def source_observation_ids(self) -> tuple:
        """Observation ids consumed since the last :meth:`make_estimate`."""
        return tuple(self._source_ids)

    def make_estimate(self, *, record_id: str, estimator_id: str, epoch: Epoch,
                      target_id: str, chaser_id: str, inertial_frame: str,
                      chief_r, chief_v) -> Estimate:
        """Publish the current filter state as an E13-01 ``Estimate``: the
        LVLH state and covariance converted to the inertial frame of the
        contract vocabulary through the chief's inertial state at this
        instant (the covariance carried on the estimate is the LVLH
        covariance mapped by the linear chief-fixed change-of-frame Jacobian,
        so it describes the inertial vectors as published), with the
        observations behind it as mandatory provenance. Resets the
        source-id ledger."""
        if not self.source_observation_ids:
            raise GNCContractError(
                f"EKF.make_estimate({record_id}): no observations consumed since the last "
                "estimate -- the contract forbids a provenance-less update; propagate the "
                "time update in metadata instead or do not publish")
        cr = np.asarray(chief_r, dtype=np.float64).reshape(3)
        cv = np.asarray(chief_v, dtype=np.float64).reshape(3)
        dr, dv = lvlh_to_eci(cr, cv, self.x[:3], self.x[3:], inertial_frame,
                             where=f"make_estimate({record_id})")
        p_i = frame_covariance(self.P, cr, cv, inertial_frame)
        est = Estimate(
            record_id=record_id, estimator_id=estimator_id,
            t_update_s=self.t_s,
            relative_state=RelativeState(
                target_id=target_id, chaser_id=chaser_id, epoch=epoch, frame=inertial_frame,
                r_rel_m=tuple(float(v) for v in dr), v_rel_m_s=tuple(float(v) for v in dv)),
            covariance_rel_m2=tuple(float(v) for v in p_i.reshape(36)),
            source_observation_ids=self.source_observation_ids)
        self._source_ids = []
        return est

    @property
    def last_update(self) -> UpdateReport | None:
        return self._last_report


def _u_or_zero(u):
    return np.zeros(3) if u is None else np.asarray(u, dtype=np.float64).reshape(3)
