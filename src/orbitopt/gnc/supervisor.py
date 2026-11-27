"""E13-06 safety supervisor: covariance-aware monitoring and a
nominal -> hold -> abort state machine on the guidance side.

The supervisor wraps any :class:`~orbitopt.gnc.feed.GuidanceLaw` and answers
one question every tick -- *from the estimate and its covariance only* -- is
the commanded flight still inside the E12-04 safety envelope, and if not,
cut thrust (hold) or end the mission (abort). Guidance-side discipline is the
same as the feed's: the law and the supervisor read an ``Estimate`` and
static uplinked data (the envelope values, the reference ephemeris), never
the plant. The offline half of the task (score the supervisor afterwards
against truth) lives in :func:`score_supervision`, which takes the sealed
event list and truth-side
:class:`orbitopt.rendezvous.safety.SafetyCheck`-shaped margins -- the
supervisor itself never sees them.

Monitors (every margin is *positive = safe*, the E12-04 sign convention):

``keep_out``            ``|rho| - keep_out_radius``
``cross_track``         ``cross_track_max_m - |z|``
``terminal_cone``       ``cone_half_angle_deg - LOS angle from -V-bar``,
                        active within ``cone_range_m`` (exactly the E12-04
                        applicability rule)
``approach_speed``      range-dependent cap ``- |rho_dot|``, the E12-04
                        piecewise-linear profile interpolated on the envelope
``passive_coast``       keep-out margin along the zero-thrust CW coast
                        predicted ``passive_contingency_s`` ahead, covariance
                        propagated with the CW state transition and the
                        E13-02 white-noise-acceleration process noise

Each instantaneous margin is inflated to a lower bound ``margin -
k_sigma * sigma`` with ``sigma`` from the first-order delta method on the
estimate covariance (rotated into LVLH at the estimate epoch, where the
chief-fixed frame map is linear in the relative state, so the numerical
Jacobian is exact); the passive monitor gets a covariance-grown sigma along
the coast. All five use the same rule, and covariance inflation -- not the
raw estimate -- is what trips them: a 2 m sensor at 3 sigma is 6 m of
"might be there", and the supervisor acts on the might.

State machine:

* ``nominal`` -- the inner law commands.
* ``hold`` -- entered after ``trip_ticks`` consecutive ticks with any
  instantaneous lower bound under its warning threshold and a coast that
  still clears the ``warn_passive_margin_m`` reserve; all thrust suppressed.
  Returns to ``nominal`` after ``clear_ticks`` quiet ticks.
* ``abort`` -- terminal, zero thrust forever. Reached from ``nominal`` when
  any instantaneous lower bound goes negative (a violation is inside the
  k-sigma bubble *now*), from ``hold`` when an instantaneous lower bound goes
  negative, or after ``hold_max_s`` (an undecided hold is an unsafe hold).

The passive-coast monitor gates the zero-thrust states rather than tripping
them (mirroring the E12-04 design, which claims ballistic safety only at its
declared release epochs): it never trips a hold on its own, and HOLD entry
requires the coast lower bound to clear ``warn_passive_margin_m`` -- a hold
is itself a coast, and committing to one predicted at (or inside) the warning
reserve of the keep-out sphere would make the supervisor's own response the
false alarm (or the violation), so a warning trip with a coast below that
margin logs a ``nominal``-to-``nominal`` ``hold_suppressed_passive_unsafe``
event and keeps the law in command; a hold whose coast prediction turns red
returns to ``nominal`` (``passive_unsafe``) because braking, not drifting,
is the only safe action left. This is what makes the post-abort trajectory a
*passive safe* one: a hold-timeout abort always leaves a coast predicted
outside the sphere, and the campaign's independent post-abort check measures
the residual sigma optimism honestly.

DESIGN ASSUMPTION (E13-06): the warning thresholds, debounce counts and the
passive-coast process noise are a supervisor tuning set, not a re-declaration
of the mission envelope; ``warn_*`` values must stay below the geometry that
a coast can bleed off between two warning ticks and one hold decision, or
the machine under-reacts by construction. ``q_accel_m2_s3`` should equal the
filter's own process noise so prediction and monitor agree about how fast
uncertainty grows.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from orbitopt.relative.cw import cw_state_transition
from orbitopt.relative.frames import eci_to_lvlh

from .contracts import Estimate, GNCContractError
from .ekf import white_noise_acceleration_q
from .guidance import ReferenceEphemeris

STATE_NOMINAL = "nominal"
STATE_HOLD = "hold"
STATE_ABORT = "abort"
SUPERVISOR_STATES = (STATE_NOMINAL, STATE_HOLD, STATE_ABORT)

#: Monitors whose *instantaneous* lower bound gates the state machine; the
#: passive coast gates hold-entry and hold-exit-to-abort separately because a
#: predicted future violation is a reason to stop thrusting, not (yet) a
#: violation.
INSTANTANEOUS_MONITORS = ("keep_out", "cross_track", "terminal_cone", "approach_speed")

_EPS_M = 1e-6
_STEP_LVLH = np.array([0.05, 0.05, 0.05, 5e-4, 5e-4, 5e-4])


@dataclass(frozen=True)
class SafetyEnvelope:
    """The uplinked envelope: the E12-04 constraint values the supervisor
    monitors, as a plain frozen record so ``gnc`` never imports the mission
    config layer. :meth:`from_safety_cfg` copies it from a ``SafetyCfg`` (or
    any record carrying the same fields)."""

    keep_out_radius_m: float
    cross_track_max_m: float
    cone_half_angle_deg: float
    cone_range_m: float
    speed_range_knots_m: tuple
    speed_cap_m_s: tuple
    passive_contingency_s: float

    def __post_init__(self):
        for name in ("keep_out_radius_m", "cross_track_max_m", "cone_range_m",
                     "passive_contingency_s"):
            v = float(getattr(self, name))
            if not (math.isfinite(v) and v > 0.0):
                raise GNCContractError(f"SafetyEnvelope.{name} must be > 0, got {v!r}")
        cone = float(self.cone_half_angle_deg)
        if not (math.isfinite(cone) and 0.0 < cone <= 90.0):
            raise GNCContractError(f"SafetyEnvelope.cone_half_angle_deg in (0, 90] expected, "
                                   f"got {cone!r}")
        knots = tuple(float(x) for x in self.speed_range_knots_m)
        caps = tuple(float(x) for x in self.speed_cap_m_s)
        if len(knots) < 2 or len(caps) != len(knots) or any(b <= a for a, b in zip(knots, knots[1:])):
            raise GNCContractError("SafetyEnvelope speed profile needs >= 2 strictly increasing "
                                   f"range knots and one cap per knot, got {knots} / {caps}")
        if any(not math.isfinite(c) or c <= 0.0 for c in caps):
            raise GNCContractError(f"SafetyEnvelope speed caps must be > 0, got {caps}")

    @classmethod
    def from_safety_cfg(cls, safety: Any) -> "SafetyEnvelope":
        """Copy the envelope out of an E12-04 ``SafetyCfg`` -- duck-typed on
        the field names so this module stays above the mission-config layer;
        the values themselves are the mission's, unmodified."""
        try:
            return cls(
                keep_out_radius_m=float(safety.keep_out_radius_m),
                cross_track_max_m=float(safety.cross_track_max_m),
                cone_half_angle_deg=float(safety.cone_half_angle_deg),
                cone_range_m=float(safety.cone_range_m),
                speed_range_knots_m=tuple(float(x) for x in safety.speed_range_knots_m),
                speed_cap_m_s=tuple(float(x) for x in safety.speed_cap_m_s),
                passive_contingency_s=float(safety.passive_contingency_s))
        except AttributeError as exc:
            raise GNCContractError(f"SafetyEnvelope.from_safety_cfg: record is not SafetyCfg-"
                                   f"shaped ({exc})") from exc

    def approach_speed_cap_m_s(self, ranges_m) -> np.ndarray:
        """The piecewise-linear range/speed ceiling, the same ``np.interp``
        semantics as :func:`orbitopt.rendezvous.safety.approach_speed_cap_m_s`
        (constant below the first knot and above the last)."""
        return np.interp(np.asarray(ranges_m, dtype=float),
                         np.asarray(self.speed_range_knots_m, dtype=float),
                         np.asarray(self.speed_cap_m_s, dtype=float))


@dataclass(frozen=True)
class SupervisorConfig:
    """Supervisor tuning: inflation, warning bands, debounce and hold timing.
    Every value is a DESIGN ASSUMPTION for E13-06 tuning; the envelope values
    live in :class:`SafetyEnvelope`, not here. ``warn_passive_margin_m``
    doubles as the coast margin a hold entry must clear."""

    k_sigma: float = 3.0
    warn_keep_out_m: float = 200.0
    warn_cross_track_m: float = 15.0
    warn_cone_deg: float = 15.0
    warn_approach_speed_m_s: float = 1.5
    warn_passive_margin_m: float = 200.0
    trip_ticks: int = 2
    clear_ticks: int = 4
    hold_max_s: float = 180.0
    passive_step_s: float = 60.0
    q_accel_m2_s3: float = 3.0e-8

    def __post_init__(self):
        if not (math.isfinite(self.k_sigma) and self.k_sigma >= 1.0):
            raise GNCContractError(f"SupervisorConfig.k_sigma >= 1 required, got {self.k_sigma}")
        for name in ("warn_keep_out_m", "warn_cross_track_m", "warn_cone_deg",
                     "warn_approach_speed_m_s", "warn_passive_margin_m",
                     "hold_max_s", "passive_step_s"):
            v = float(getattr(self, name))
            if not (math.isfinite(v) and v > 0.0):
                raise GNCContractError(f"SupervisorConfig.{name} must be > 0, got {v!r}")
        if not (isinstance(self.trip_ticks, int) and isinstance(self.clear_ticks, int)
                and self.trip_ticks >= 1 and self.clear_ticks >= 1):
            raise GNCContractError("SupervisorConfig trip_ticks/clear_ticks must be integers "
                                   ">= 1")
        if not (math.isfinite(self.q_accel_m2_s3) and self.q_accel_m2_s3 >= 0.0):
            raise GNCContractError(f"SupervisorConfig.q_accel_m2_s3 >= 0 required, "
                                   f"got {self.q_accel_m2_s3!r}")


@dataclass(frozen=True)
class MonitorReading:
    """One monitor at one tick: raw margin, first-order covariance sigma, the
    inflated lower bound that gates the state machine, and the warning
    threshold. Inactive monitors report ``active=False`` with ``sigma=0`` and
    ``lower=inf`` (the constraint does not apply at this range, E12-04 rule)."""

    name: str
    unit: str
    active: bool
    margin: float
    sigma: float
    lower: float
    warn: float

    @property
    def warned(self) -> bool:
        return bool(self.active and self.lower < self.warn)

    @property
    def tripped(self) -> bool:
        return bool(self.active and self.lower < 0.0)

    def to_dict(self) -> dict:
        return {"name": self.name, "unit": self.unit, "active": self.active,
                "margin": self.margin, "sigma": self.sigma, "lower": self.lower,
                "warn": self.warn}


@dataclass(frozen=True)
class SupervisionEvent:
    """One state transition, emitted at the estimate epoch that caused it."""

    t_s: float
    state_from: str
    state_to: str
    trigger: str
    margin: float
    detail: str

    def to_dict(self) -> dict:
        return {"t_s": self.t_s, "state_from": self.state_from, "state_to": self.state_to,
                "trigger": self.trigger, "margin": self.margin, "detail": self.detail}


class SafetySupervisor:
    """A ``GuidanceLaw`` that watches its own wingman: pass the inner law and
    the envelope in, hand it the estimates, and it commands the inner law
    while safe and zero thrust when not. ``step``/``monitor`` are public for
    campaign wiring and tests; ``delta_v``/``phase_for`` satisfy the E13-01
    law seam so it drops into ``simulate_closed_loop`` unchanged."""

    def __init__(self, *, law, reference: ReferenceEphemeris, envelope: SafetyEnvelope,
                 config: SupervisorConfig | None = None,
                 target_id: str = "tgt", chaser_id: str = "chaser"):
        if not callable(getattr(law, "delta_v", None)):
            raise GNCContractError(f"SafetySupervisor: inner law has no delta_v, got {law!r}")
        if not isinstance(reference, ReferenceEphemeris):
            raise GNCContractError(f"SafetySupervisor: reference must be a ReferenceEphemeris, "
                                   f"got {reference!r}")
        if not isinstance(envelope, SafetyEnvelope):
            raise GNCContractError("SafetySupervisor: envelope must be a SafetyEnvelope "
                                   "(use SafetyEnvelope.from_safety_cfg)")
        self._law = law
        self._reference = reference
        self._envelope = envelope
        self._cfg = config if config is not None else SupervisorConfig()
        if not (isinstance(target_id, str) and target_id) or not (isinstance(chaser_id, str)
                                                                  and chaser_id):
            raise GNCContractError("SafetySupervisor: target_id/chaser_id must be non-empty str")
        self._target_id, self._chaser_id = target_id, chaser_id
        self._n = float(reference.mean_motion)
        self._state = STATE_NOMINAL
        self._t_state = 0.0
        self._trip = 0
        self._clear = 0
        self._suppress_open = False
        self._events: list = []
        self._snapshot: dict | None = None

    # -- guidance-side interface ------------------------------------------------
    def delta_v(self, estimate: Estimate) -> tuple:
        self.step(estimate)
        if self._state != STATE_NOMINAL:
            return (0.0, 0.0, 0.0)
        return self._law.delta_v(estimate)

    def phase_for(self, estimate: Estimate) -> str:
        """Command metadata phase: the supervision state while supervising,
        otherwise the inner law's own phase."""
        if self._state != STATE_NOMINAL:
            return self._state
        inner = getattr(self._law, "phase_for", None)
        return str(inner(estimate)) if callable(inner) else "coast"

    # -- observation --------------------------------------------------------------
    @property
    def state(self) -> str:
        return self._state

    @property
    def events(self) -> tuple:
        return tuple(self._events)

    @property
    def snapshot(self) -> dict | None:
        """The last tick's monitor readings, keyed by monitor name."""
        return self._snapshot

    # -- monitors ------------------------------------------------------------------
    def lvlh_state_cov(self, estimate: Estimate) -> tuple:
        """Estimate state and covariance in the reference LVLH frame at the
        estimate epoch. The chief-fixed frame map is linear in the relative
        state for a frozen chief, so the central-difference Jacobian is exact
        to round-off, not an approximation."""
        if not isinstance(estimate, Estimate):
            raise GNCContractError(f"SafetySupervisor: expected an Estimate, got {estimate!r}")
        rs = estimate.relative_state
        if rs.target_id != self._target_id or rs.chaser_id != self._chaser_id:
            raise GNCContractError(f"SafetySupervisor: estimate {rs.target_id}/{rs.chaser_id} "
                                   f"is not the supervised {self._target_id}/{self._chaser_id}")
        cr, cv = self._reference.state_at(float(estimate.t_update_s))
        frame = rs.frame

        def to_lvlh(state6: np.ndarray) -> np.ndarray:
            rho, rhod = eci_to_lvlh(cr, cv, state6[:3], state6[3:], frame,
                                     where="SafetySupervisor frame transform")
            return np.concatenate((rho, rhod))

        x_in = np.concatenate((np.asarray(rs.r_rel_m, dtype=float),
                               np.asarray(rs.v_rel_m_s, dtype=float)))
        x = to_lvlh(x_in)
        jac = np.empty((6, 6))
        for i in range(6):
            xp, xm = x_in.copy(), x_in.copy()
            xp[i] += _STEP_LVLH[i]
            xm[i] -= _STEP_LVLH[i]
            jac[:, i] = (to_lvlh(xp) - to_lvlh(xm)) / (2.0 * _STEP_LVLH[i])
        p_in = np.asarray(estimate.covariance_rel_m2, dtype=float).reshape(6, 6)
        p_lvlh = jac @ p_in @ jac.T
        return x, 0.5 * (p_lvlh + p_lvlh.T)

    def monitor(self, estimate: Estimate) -> dict:
        """All five readings at this estimate, keyed by monitor name (no
        state change)."""
        x, p = self.lvlh_state_cov(estimate)
        rho, rhod = x[:3], x[3:]
        rng = float(np.linalg.norm(rho))
        spd = float(np.linalg.norm(rhod))
        env, cfg = self._envelope, self._cfg

        def _delta(name: str, unit: str, f, warn: float, active: bool,
                   margin: float | None = None, sigma: float | None = None) -> MonitorReading:
            m = float(f(x)) if margin is None else float(margin)
            if not active:
                return MonitorReading(name, unit, False, m, 0.0, math.inf, warn)
            s = self._delta_sigma(f, x, p) if sigma is None else float(sigma)
            return MonitorReading(name, unit, True, m, s, m - cfg.k_sigma * s, warn)

        readings: dict = {}
        readings["keep_out"] = _delta("keep_out", "m",
                                      lambda s: float(np.linalg.norm(s[:3])) - env.keep_out_radius_m,
                                      cfg.warn_keep_out_m, True)
        readings["cross_track"] = _delta("cross_track", "m",
                                         lambda s: env.cross_track_max_m - abs(s[2]),
                                         cfg.warn_cross_track_m, True)
        los_deg = self._los_deg(rho, rng)
        cone_active = rng <= env.cone_range_m + cfg.k_sigma * self._range_sigma(p, rho, rng)
        cone_sigma = 0.0 if abs(math.sin(math.radians(los_deg))) < 1e-3 else None
        readings["terminal_cone"] = _delta(
            "terminal_cone", "deg",
            lambda s: env.cone_half_angle_deg - self._los_deg(s[:3],
                                                             max(float(np.linalg.norm(s[:3])),
                                                                 _EPS_M)),
            cfg.warn_cone_deg, cone_active, margin=env.cone_half_angle_deg - los_deg,
            sigma=cone_sigma)
        readings["approach_speed"] = _delta(
            "approach_speed", "m/s",
            lambda s: float(env.approach_speed_cap_m_s(float(np.linalg.norm(s[:3]))))
            - float(np.linalg.norm(s[3:])),
            cfg.warn_approach_speed_m_s, True, sigma=0.0 if spd < _EPS_M else None)
        readings["passive_coast"] = self._passive_reading(x, p)
        return readings

    def step(self, estimate: Estimate) -> dict:
        """Monitor, advance the state machine, return the readings."""
        readings = self.monitor(estimate)
        t = float(estimate.t_update_s)
        inst = [readings[name] for name in INSTANTANEOUS_MONITORS]
        hard = [r for r in inst if r.tripped]
        warned = [r for r in inst if r.warned]
        pas = readings["passive_coast"]
        if self._state == STATE_NOMINAL:
            if hard:
                self._transition(STATE_ABORT, t, self._worst(hard),
                                 "instantaneous lower bound below zero")
            elif warned:
                self._trip += 1
                self._clear = 0
                if self._trip >= self._cfg.trip_ticks:
                    if pas.lower < self._cfg.warn_passive_margin_m:
                        self._suppress_hold(t, pas)
                    else:
                        self._transition(
                            STATE_HOLD, t, self._worst(warned),
                            "warning band trip (debounced)")
            else:
                self._trip = 0
                self._suppress_open = False
        elif self._state == STATE_HOLD:
            if hard:
                self._transition(STATE_ABORT, t, self._worst(hard), "tripped while holding")
            elif pas.lower < 0.0:
                self._resume_guidance(t, pas)
            elif t - self._t_state >= self._cfg.hold_max_s:
                self._transition(STATE_ABORT, t, self._worst([pas] + inst),
                                 f"hold exceeded hold_max_s={self._cfg.hold_max_s:.0f} s")
            elif not (warned or pas.warned):
                self._clear += 1
                if self._clear >= self._cfg.clear_ticks:
                    self._transition(STATE_NOMINAL, t, self._worst([pas] + inst), "recovery")
            else:
                self._clear = 0
        self._snapshot = readings
        return readings

    # -- internals -------------------------------------------------------------
    def _suppress_hold(self, t_s: float, pas: MonitorReading) -> None:
        """A warning trip whose zero-thrust coast does not clear the holdable
        margin cannot become a hold (a hold IS that coast): the law stays in
        command and the suppression is logged once per warning episode as a
        nominal-to-nominal event (never an alert interval)."""
        if not self._suppress_open:
            self._events.append(SupervisionEvent(
                t_s=float(t_s), state_from=STATE_NOMINAL, state_to=STATE_NOMINAL,
                trigger="hold_suppressed_passive_unsafe", margin=float(pas.margin),
                detail=f"hold entry suppressed; passive coast lower bound {pas.lower:+.3g} m "
                       f"below holdable margin"))
            self._suppress_open = True
        self._trip = 0

    def _resume_guidance(self, t_s: float, pas: MonitorReading) -> None:
        """A hold whose coast prediction turns red ends the hold: drifting is
        no longer the safe option, so guidance takes back command."""
        self._events.append(SupervisionEvent(
            t_s=float(t_s), state_from=STATE_HOLD, state_to=STATE_NOMINAL,
            trigger="passive_unsafe", margin=float(pas.margin),
            detail=f"hold ended, passive coast lower bound {pas.lower:+.3g} m; guidance resumed"))
        self._state = STATE_NOMINAL
        self._t_state = float(t_s)
        self._trip = 0
        self._clear = 0
        self._suppress_open = True

    def _transition(self, to: str, t_s: float, reading: MonitorReading, detail: str) -> None:
        trigger = reading.name if to in (STATE_HOLD, STATE_ABORT) and reading.active else \
            ("recovery" if to == STATE_NOMINAL else "hold_timeout")
        if to == STATE_ABORT and "hold exceeded" in detail:
            trigger = "hold_timeout"
        self._events.append(SupervisionEvent(
            t_s=float(t_s), state_from=self._state, state_to=to, trigger=trigger,
            margin=float(reading.margin),
            detail=f"{detail}; {reading.name} margin={reading.margin:+.3g} {reading.unit}, "
                   f"lower={('inf' if not math.isfinite(reading.lower) else format(reading.lower, '+.3g'))}"))
        self._state = to
        self._t_state = float(t_s)
        self._trip = 0
        self._clear = 0
        self._suppress_open = False

    @staticmethod
    def _worst(readings) -> MonitorReading:
        return min(readings, key=lambda r: r.lower)

    @staticmethod
    def _los_deg(rho: np.ndarray, rng: float) -> float:
        cos = -float(rho[1]) / max(rng, _EPS_M)
        return math.degrees(math.acos(max(-1.0, min(1.0, cos))))

    @staticmethod
    def _range_sigma(p_lvlh: np.ndarray, rho: np.ndarray, rng: float) -> float:
        if rng < _EPS_M:
            return 0.0
        g = np.asarray(rho, dtype=float) / rng
        return math.sqrt(max(float(g @ p_lvlh[:3, :3] @ g), 0.0))

    def _delta_sigma(self, f, x: np.ndarray, p: np.ndarray) -> float:
        grad = np.zeros(6)
        for i in range(6):
            xp, xm = x.copy(), x.copy()
            xp[i] += _STEP_LVLH[i]
            xm[i] -= _STEP_LVLH[i]
            grad[i] = (float(f(xp)) - float(f(xm))) / (2.0 * _STEP_LVLH[i])
        return math.sqrt(max(float(grad @ p @ grad), 0.0))

    def _passive_reading(self, x: np.ndarray, p: np.ndarray) -> MonitorReading:
        """Worst keep-out lower bound along the zero-thrust CW coast over the
        envelope's contingency horizon, covariance propagated sequentially
        with the CW transition and the E13-02 process noise. The coast is
        *no thrust from now* -- exactly what HOLD/ABORT command; executed
        burns are already in the state."""
        env, cfg = self._envelope, self._cfg
        horizon = env.passive_contingency_s
        steps = max(1, int(math.ceil(horizon / cfg.passive_step_s)))
        dt = horizon / steps
        phi = cw_state_transition(self._n, dt)
        q = white_noise_acceleration_q(cfg.q_accel_m2_s3, dt)
        xk, pk, best = np.array(x, dtype=float), np.array(p, dtype=float), None
        for _ in range(steps):
            xk = phi @ xk
            pk = phi @ pk @ phi.T + q
            rng = float(np.linalg.norm(xk[:3]))
            margin = rng - env.keep_out_radius_m
            sigma = self._range_sigma(pk, xk[:3], rng)
            lower = margin - cfg.k_sigma * sigma
            if best is None or lower < best[0]:
                best = (lower, margin, sigma)
        lower, margin, sigma = best
        return MonitorReading("passive_coast", "m", True, float(margin), float(sigma),
                              float(lower), cfg.warn_passive_margin_m)


# ---------------------------------------------------------------------------
# offline scoring: event stream vs truth margins (the guidance side never
# runs this; it consumes the run log and E12-04-shaped truth checks)
# ---------------------------------------------------------------------------

def _check_series(check: Any) -> tuple:
    if isinstance(check, Mapping):
        times, margins = check["times_s"], check["margins"]
    else:
        times, margins = check.times_s, check.margins
    return np.asarray(times, dtype=float), np.asarray(margins, dtype=float)


def violation_intervals(times: np.ndarray, margins: np.ndarray,
                        *, merge_gap_s: float = 120.0) -> list:
    """Contiguous truth-violation windows (negative-margin samples) as
    ``[t_start, t_end]`` pairs; two runs merge into one window when separated
    by less than ``merge_gap_s`` (a margin flickering across zero at the
    sample rate is one event, not five)."""
    t = np.asarray(times, dtype=float)
    neg = np.asarray(margins, dtype=float) < 0.0
    n = int(neg.size)
    runs: list = []
    i = 0
    while i < n:
        if not neg[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and neg[j + 1]:
            j += 1
        runs.append([i, j])
        i = j + 1
    out: list = []
    for i0, i1 in runs:
        if out and (t[i0] - t[out[-1][1]]) <= merge_gap_s:
            out[-1][1] = i1
        else:
            out.append([i0, i1])
    return [[float(t[a]), float(t[b])] for a, b in out]


def alert_intervals(events, t_end_s: float) -> list:
    """Maximal non-``nominal`` intervals ``{start_s, end_s, trigger}`` from
    the event stream; an alert still open at the end of the run closes at
    ``t_end_s``."""
    out: list = []
    open_alert: dict | None = None
    for ev in sorted(events, key=lambda e: float(_ev_get(e, "t_s"))):
        state_to = _ev_get(ev, "state_to")
        t = float(_ev_get(ev, "t_s"))
        if state_to != STATE_NOMINAL and open_alert is None:
            open_alert = {"start_s": t, "end_s": t, "trigger": str(_ev_get(ev, "trigger"))}
        elif state_to != STATE_NOMINAL and open_alert is not None:
            open_alert["end_s"] = t
            if open_alert["trigger"] != _ev_get(ev, "trigger"):
                open_alert["trigger"] = str(_ev_get(ev, "trigger"))
        elif state_to == STATE_NOMINAL and open_alert is not None:
            open_alert["end_s"] = t
            out.append(open_alert)
            open_alert = None
    if open_alert is not None:
        open_alert["end_s"] = max(open_alert["end_s"], float(t_end_s))
        out.append(open_alert)
    return out


def _ev_get(event: Any, key: str):
    return event[key] if isinstance(event, Mapping) else getattr(event, key)


def score_supervision(events, checks: Mapping[str, Any], *, t_end_s: float,
                      detect_grace_s: float = 30.0, late_grace_s: float = 60.0,
                      justify_lead_s: float = 900.0,
                      merge_gap_s: float = 120.0) -> dict:
    """Post-run truth scoring of a supervisor (the acceptance statistics:
    missed alarms and false alarms, counted separately).

    ``events`` is the supervisor's event stream (objects or dicts); ``checks``
    maps constraint name -> truth evaluation with ``times_s``/``margins``
    (``SafetyCheck`` objects from
    :func:`orbitopt.rendezvous.safety.evaluate_trajectory_safety` fit
    directly, which is the E12-04 reuse the task asks for).

    Definitions (the knobs are parameters, the semantics are fixed):

    * a truth violation window is *missed* unless some alert interval starts
      within ``detect_grace_s`` after the violation onset and is still open
      (or closed only ``late_grace_s`` before) at the onset -- an alarm that
      arrives after the fact is a miss;
    * an alert interval is a *false alarm* unless some violation window
      starts within ``justify_lead_s`` after the alert onset (or overlaps the
      alert, or ends no more than ``late_grace_s`` before it) -- a warning
      hours ahead of a far-future violation was over-cautious, and is
      counted as such, not hidden."""
    alerts = alert_intervals(events, float(t_end_s))
    per_check: dict = {}
    violations: list = []
    missed = 0
    for name in sorted(checks):
        t, m = _check_series(checks[name])
        intervals = violation_intervals(t, m, merge_gap_s=merge_gap_s) if t.size else []
        detected = []
        for v0, v1 in intervals:
            hit = any(a["start_s"] <= v0 + detect_grace_s and a["end_s"] >= v0 - late_grace_s
                      for a in alerts)
            detected.append(hit)
            violations.append((name, v0, v1, hit))
            missed += 0 if hit else 1
        per_check[name] = {
            "violated": bool(intervals), "n_violation_intervals": len(intervals),
            "violation_intervals": [[v[0], v[1]] for v in intervals],
            "detected": detected, "n_missed": sum(1 for d in detected if not d),
        }
    false_alarms = [a for a in alerts if not any(
        v1 >= a["start_s"] - late_grace_s and v0 <= a["end_s"] + justify_lead_s
        for _name, v0, v1, _h in violations)]
    return {
        "alert_intervals": alerts,
        "n_alert_intervals": len(alerts),
        "checks": per_check,
        "violations": [{"check": n_, "t_start_s": v0, "t_end_s": v1, "detected": hit}
                       for n_, v0, v1, hit in violations],
        "false_alarms": false_alarms,
        "counts": {"alerts": len(alerts), "violation_intervals": len(violations),
                   "missed": missed, "false_alarms": len(false_alarms)},
    }
