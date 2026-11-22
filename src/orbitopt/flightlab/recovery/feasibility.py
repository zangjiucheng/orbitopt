"""E11-02 (task #309): the recovery-feasibility envelope at the separation handoff.

E10-03 closed on a hard fact: the DC8 unpowered-return hold-out lost 168/168
samples with the *planning-infeasible* class (all ``CORRIDOR_MISSED``), and
the robust arm trained to 0/11 -- the planner was searching scripts inside a
region of separation states from which no script in its family flies. A
search is the wrong instrument for an empty feasible set: it pays its whole
budget to prove a negative. E11-01 (task #308) then measured the grid-fin
authority the vehicle actually has (q cured: 460 -> 118-161 kPa; miss < 320 m;
arrival speed 361.5 m/s and corridor tilt floor 25-30 deg against a 15 deg
limit). This module turns those measurements into the object the task asks
for: a *feasibility envelope*, decided from the separation state before any
search runs.

Mathematical definition
-----------------------
A separation state ``s`` (position, velocity, mass at the hot-staging
handoff) is **inside the recovery envelope of a reference mission** ``R``
-- the authored script + planning limits carried by :class:`EnvelopeSpec` --
iff every condition below holds. Two of them are closed-form in ``s``; the
rest are measured on a single full-physics rollout of ``R`` from ``s`` (the
*reference probe*, the same integrator, atmosphere and corridor referee the
planner uses; nothing is kinematic):

======================  =======================================================  ========
condition               statement                                                reason
======================  =======================================================  ========
``fuel_floor``          ``m_p(s) >= m(s) (1 - exp(-v_h /(Isp g0)))``              ``INSUFFICIENT_PROPELLANT_BOOSTBACK``
``corridor_reached``    the probe crosses the corridor plane ``h0``              ``CORRIDOR_NOT_REACHED``
``corridor_miss``       miss at ``h0`` ``<=`` corridor radius                    ``CORRIDOR_MISSED``
``corridor_tilt``       tilt at ``h0`` ``<=`` ``max_tilt_deg``                   ``CORRIDOR_MISSED``
``arrival_speed``       speed at ``h0`` ``<=`` corridor ``v_max``                ``CORRIDOR_MISSED``
``landing_forecast``    the closed-loop landing forecast (when modelled)         ``CORRIDOR_MISSED``
                        closes inside the landing limits
``handoff_overlap``     the entry burn is over at ``h0`` (engine-off handoff)    ``CORRIDOR_MISSED``
``entry_q``             peak entry-window q ``<=`` ``entry_q_max_pa``            ``ENTRY_Q_LIMIT_EXCEEDED``
``entry_g``             peak entry-window load ``<=`` ``entry_g_max``            ``ENTRY_G_LIMIT_EXCEEDED``
``fuel_margin``         propellant at ``h0`` ``>=`` landing need + reserve       ``INSUFFICIENT_PROPELLANT_RESERVE``
                        + planner margin
======================  =======================================================  ========

``m_p(s)`` is the separation propellant load, ``v_h`` the horizontal speed
(the analytic floor is the planner's own :func:`_propellant_floor_kg`, so
screen and planner can never disagree about it). Each condition reports a
normalized severity ``value/limit`` (``limit/value`` for the two fuel
conditions); the *governing* condition is the violated one with the largest
severity, and its reason code is what the envelope predicts the flight
result will be. The envelope is thus a **screen, not a proof**: it is sound
for the reference mission ``R`` -- an inside verdict means "``R`` flies the
corridor physics from here", an outside verdict means "``R`` does not, and
this is the reason" -- and its agreement with actual flown outcomes over the
E10-03 hold-out is measured, not assumed (bench
``benchmarks/bench_e11_02_recovery_feasibility_envelope.py``, report
``docs/benchmarks/e11_02_recovery_feasibility_envelope.json``). The DC8
unpowered family's envelope is empty over the whole swept separation
manifold -- that is the E10-03 finding restated as geometry, and exactly the
case the planner should decide without spending a search (wired as
``plan_boostback_entry(feasibility_envelope=True)``, off by default so every
existing caller is bit-for-bit unchanged).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from orbitopt.flightlab.recovery.boostback import (
    CORRIDOR_MISSED,
    CORRIDOR_NOT_REACHED,
    DEFAULT_ENTRY_G_MAX,
    DEFAULT_ENTRY_Q_MAX_PA,
    DEFAULT_LANDING_DIVERT_BUDGET_M,
    DEFAULT_LANDING_RESERVE_MARGIN_KG,
    ENTRY_CUT_SPEED_MAX_M_S,
    ENTRY_G_LIMIT_EXCEEDED,
    ENTRY_Q_LIMIT_EXCEEDED,
    INSUFFICIENT_PROPELLANT_BOOSTBACK,
    INSUFFICIENT_PROPELLANT_ENTRY_BURN,
    INSUFFICIENT_PROPELLANT_RESERVE,
    RecoveryCorridor,
    RecoveryTarget,
    _enu_basis,
    _geodetic_fast,
    _propellant_floor_kg,
    plan_boostback_entry,
)
from orbitopt.flightlab.recovery.propagator import BoosterModel, BoosterState

#: The condition names, in report order.
ENVELOPE_CONDITIONS = (
    "fuel_floor", "boostback_exhausted", "entry_exhausted", "corridor_reached",
    "corridor_miss", "corridor_tilt", "arrival_speed", "landing_forecast",
    "handoff_overlap", "entry_q", "entry_g", "fuel_margin",
)

#: condition -> the flight reason code the envelope predicts for it
CONDITION_REASON = {
    "fuel_floor": INSUFFICIENT_PROPELLANT_BOOSTBACK,
    "boostback_exhausted": INSUFFICIENT_PROPELLANT_BOOSTBACK,
    "entry_exhausted": INSUFFICIENT_PROPELLANT_ENTRY_BURN,
    "corridor_reached": CORRIDOR_NOT_REACHED,
    "corridor_miss": CORRIDOR_MISSED,
    "corridor_tilt": CORRIDOR_MISSED,
    "arrival_speed": CORRIDOR_MISSED,
    "landing_forecast": CORRIDOR_MISSED,
    "handoff_overlap": CORRIDOR_MISSED,
    "entry_q": ENTRY_Q_LIMIT_EXCEEDED,
    "entry_g": ENTRY_G_LIMIT_EXCEEDED,
    "fuel_margin": INSUFFICIENT_PROPELLANT_RESERVE,
}

#: severity used when a condition is violated but exposes no finite
#: normalized ratio (never crossed / no measurable window / forecast reject)
_NO_RATIO_SEVERITY = {"corridor_reached": 1.0, "landing_forecast": 1.05,
                      "handoff_overlap": 1.02,
                      "boostback_exhausted": 1.5, "entry_exhausted": 1.5}


@dataclass(frozen=True)
class EnvelopeCondition:
    """One graded condition of the envelope: measured ``value`` against the
    spec ``limit``, with ``severity = value/limit`` (``limit/value`` for the
    two fuel conditions, where larger-is-worse inverts). ``severity`` of a
    passed condition is its *margin* (how far inside); ``detail`` says what
    the number is."""

    name: str
    violated: bool
    severity: float
    value: float | None
    limit: float | None
    unit: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "violated": bool(self.violated),
                "severity": round(float(self.severity), 4),
                "value": None if self.value is None else float(self.value),
                "limit": None if self.limit is None else float(self.limit),
                "unit": self.unit, "detail": self.detail,
                "predicted_reason": CONDITION_REASON[self.name]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EnvelopeCondition":
        return cls(name=str(d["name"]), violated=bool(d["violated"]),
                   severity=float(d["severity"]),
                   value=None if d.get("value") is None else float(d["value"]),
                   limit=None if d.get("limit") is None else float(d["limit"]),
                   unit=str(d.get("unit", "")), detail=str(d.get("detail", "")))


@dataclass(frozen=True)
class EnvelopeVerdict:
    """The envelope's answer for one separation state: ``in_envelope`` plus
    every condition, the governing violated condition and its predicted
    flight reason, a one-line ``explanation``, and the probe evidence
    (``probe``: the reference rollout's corridor numbers)."""

    in_envelope: bool
    conditions: tuple[EnvelopeCondition, ...]
    governing_condition: str | None
    governing_reason: str | None
    explanation: str
    probe: Mapping[str, Any] = field(default_factory=dict)
    violated_codes: tuple[str, ...] = ()

    @property
    def worst_severity(self) -> float:
        """Largest normalized violation (>1 out of the envelope), or, for an
        inside verdict, the severity of the tightest condition."""
        return max((c.severity for c in self.conditions), default=0.0)

    def to_dict(self) -> dict[str, Any]:
        return {"in_envelope": bool(self.in_envelope),
                "conditions": [c.to_dict() for c in self.conditions],
                "governing_condition": self.governing_condition,
                "governing_reason": self.governing_reason,
                "worst_severity": round(self.worst_severity, 4),
                "violated_codes": list(self.violated_codes),
                "explanation": self.explanation, "probe": dict(self.probe)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EnvelopeVerdict":
        return cls(in_envelope=bool(d["in_envelope"]),
                   conditions=tuple(EnvelopeCondition.from_dict(c)
                                    for c in d.get("conditions", ())),
                   governing_condition=d.get("governing_condition"),
                   governing_reason=d.get("governing_reason"),
                   explanation=str(d.get("explanation", "")),
                   probe=dict(d.get("probe", {})),
                   violated_codes=tuple(str(c) for c in d.get("violated_codes", ())))


@dataclass(frozen=True)
class EnvelopeSpec:
    """The reference mission that defines the envelope: the authored script
    (``reference_script``, the planner parameter vector as ``initial_guess``
    takes it) plus the planning limits the corridor physics is graded
    against. Built from a sealed recovery recipe (:meth:`from_recipe`), from
    live planner keyword arguments (:meth:`from_planner_kwargs`), or from a
    plain mapping (:meth:`from_mapping`)."""

    name: str = "reference"
    reference_script: tuple[float, ...] = ()
    entry_burns: int = 1
    entry_throttle: bool = False
    boostback_cutoff: str = "duration"
    entry_cutoff: str = "duration"
    entry_cut_speed_max_m_s: float = ENTRY_CUT_SPEED_MAX_M_S
    corridor_aim_margin_m: float = 0.0
    entry_q_max_pa: float = DEFAULT_ENTRY_Q_MAX_PA
    entry_g_max: float = DEFAULT_ENTRY_G_MAX
    landing_reserve_margin_kg: float = DEFAULT_LANDING_RESERVE_MARGIN_KG
    landing_need_model: str = "momentum"
    #: the FL-303/FL-305 grading the closed-loop landing forecast must match
    #: the flight's: ``landing_limits`` (``LandingLimits`` or a mapping),
    #: ``landing_guidance`` (``TerminalGuidance`` / recipe ``landing`` dict)
    #: and the R-01 divert budget (``None`` = divert off).
    landing_limits: Any = None
    landing_guidance: Any = None
    landing_divert_budget_m: float | None = DEFAULT_LANDING_DIVERT_BUDGET_M
    entry_aero: Mapping[str, Any] | None = None
    horizon_s: float = 900.0
    dt_opt: float = 0.5
    #: integration step of the probe's final segment: a sealed recipe pins
    #: its own ``dt_final`` and the closed-loop landing forecast is
    #: step-sensitive, so the probe must integrate the way the flight does
    dt_final: float = 0.5

    #: the planner kwargs the spec carries (everything but the script)
    _PLAN_KWARGS = ("entry_burns", "entry_throttle", "boostback_cutoff",
                    "entry_cutoff", "entry_cut_speed_max_m_s",
                    "corridor_aim_margin_m", "entry_q_max_pa", "entry_g_max",
                    "landing_reserve_margin_kg", "landing_need_model",
                    "entry_aero", "horizon_s", "landing_limits",
                    "landing_guidance", "landing_divert_budget_m")

    def planner_kwargs(self) -> dict[str, Any]:
        """Keyword arguments reproducing the reference mission on
        ``plan_boostback_entry`` (combine with ``initial_guess=
        spec.reference_script, search="keep"``; the probe step is
        ``dt_opt``). Mapping-valued landing grading is rebuilt as the
        planner's own dataclasses so a spec that crossed a serialisation
        boundary flies identically."""
        out = {k: getattr(self, k) for k in self._PLAN_KWARGS}
        lim = out.get("landing_limits")
        if isinstance(lim, Mapping):
            from orbitopt.flightlab.recovery.landing import LandingLimits
            out["landing_limits"] = LandingLimits(**dict(lim))
        gui = out.get("landing_guidance")
        if isinstance(gui, Mapping):
            from orbitopt.flightlab.recovery.landing import TerminalGuidance
            out["landing_guidance"] = TerminalGuidance(
                **{k: v for k, v in dict(gui).items()
                   if k not in ("mode", "mismatch")})
        return out

    def to_dict(self) -> dict[str, Any]:
        import dataclasses
        d = {"name": self.name, "reference_script": list(self.reference_script),
             **{k: getattr(self, k) for k in self._PLAN_KWARGS},
             "dt_opt": self.dt_opt, "dt_final": self.dt_final}
        for k in ("landing_limits", "landing_guidance"):
            v = d.get(k)
            if dataclasses.is_dataclass(v) and not isinstance(v, type):
                d[k] = dataclasses.asdict(v)
        return d

    @classmethod
    def from_mapping(cls, d: Mapping[str, Any]) -> "EnvelopeSpec":
        """A spec written as a plain dict (``to_dict`` output, or a recipe
        fragment; unknown keys such as ``initial_guess``/``search`` are
        ignored, ``initial_guess`` becomes the reference script)."""
        kwargs = {k: d[k] for k in cls._PLAN_KWARGS if k in d}
        script = d.get("reference_script", d.get("initial_guess"))
        return cls(name=str(d.get("name", "reference")),
                   reference_script=tuple(float(x) for x in (script or ())),
                   dt_opt=float(d.get("dt_opt", 0.5)),
                   dt_final=float(d.get("dt_final", d.get("dt_opt", 0.5))),
                   **kwargs)

    @classmethod
    def from_recipe(cls, recipe: Mapping[str, Any], *, name: str = "recipe",
                    script: Sequence[float] | None = None) -> "EnvelopeSpec":
        """The spec of a sealed recovery recipe file (the E5-08
        ``recovery-script.json`` shape): the recipe's plan kwargs and its
        pinned ``initial_guess``, overridden by ``script`` when given (e.g.
        a trained selection). The recipe's ``landing`` block becomes the
        closed-loop forecast's guidance (exactly as ``fly_recovery`` does
        it); the reserve *string* is MC-layer plumbing and is not envelope
        physics."""
        d = {**dict(recipe), "name": name}
        landing = d.get("landing")
        if isinstance(landing, Mapping):
            d["landing_guidance"] = dict(landing)
            d.pop("landing", None)
        if script is not None:
            d["reference_script"] = [float(x) for x in script]
        return cls.from_mapping(d)

    @classmethod
    def from_planner_kwargs(cls, script: Sequence[float], *,
                            name: str = "live", **kwargs: Any) -> "EnvelopeSpec":
        """The spec of a live ``plan_boostback_entry`` call: the mission and
        limit keyword arguments as the planner received them, with the
        handed ``initial_guess`` as the reference script."""
        return cls.from_mapping({**kwargs, "name": name,
                                 "reference_script": [float(x) for x in script]})


def fuel_floor_kg(sep_state: BoosterState, booster: BoosterModel) -> tuple[float, float]:
    """``(separation propellant [kg], analytic boostback floor [kg])`` -- the
    same rocket-equation floor the planner gates its search on
    (:func:`plan_boostback_entry`, ``_propellant_floor_kg``): the propellant
    to null the horizontal separation velocity at vacuum Isp."""
    sep_r = np.asarray(sep_state.position_m, dtype=np.float64)
    sep_v = np.asarray(sep_state.velocity_m_s, dtype=np.float64)
    lat, lon, _ = _geodetic_fast(sep_r)
    e_, n_, _u = _enu_basis(lat, lon)
    horiz_speed = math.hypot(float(sep_v @ np.array(e_)), float(sep_v @ np.array(n_)))
    sep_prop = float(sep_state.mass_kg) - float(booster.dry_mass_kg)
    return sep_prop, _propellant_floor_kg(float(sep_state.mass_kg), horiz_speed,
                                          float(booster.engine.isp_vacuum_s))


def grade_probe(probe: Mapping[str, Any] | None, spec: EnvelopeSpec, *,
                corridor: RecoveryCorridor, landing_reserve_kg: float,
                sep_propellant_kg: float, prop_floor_kg: float) -> EnvelopeVerdict:
    """Grade one reference probe against the envelope and return the verdict.

    ``probe`` carries the rollout's measured corridor numbers under the keys
    ``reached, miss_m, tilt_deg, speed_m_s, prop_kg, landing_need_kg,
    entry_q_pa, entry_g, exhausted, landing_forecast_ok`` (``None`` = not
    measured). ``None``/``{}`` grades the analytic conditions only -- the
    state fails no matter which script is flown from it. The grading mirrors
    ``plan_boostback_entry``'s own feasibility order so the predicted reason
    is the reason the flight verdict would carry; the joint-chance check is
    planner policy, not corridor physics, and is deliberately not an
    envelope condition."""
    conds: list[EnvelopeCondition] = []

    def add(name: str, violated: bool, severity: float, value: float | None,
            limit: float | None, unit: str, detail: str = "") -> None:
        conds.append(EnvelopeCondition(name=name, violated=violated,
                                       severity=float(severity), value=value,
                                       limit=limit, unit=unit, detail=detail))

    floor_bad = sep_propellant_kg < prop_floor_kg
    add("fuel_floor", floor_bad,
        (prop_floor_kg / max(sep_propellant_kg, 1e-9)) if prop_floor_kg > 0.0 else 0.0,
        sep_propellant_kg, prop_floor_kg, "kg",
        "analytic boostback floor (rocket equation on the horizontal "
        "separation velocity at vacuum Isp)")
    if probe is None:
        return _finalize(conds, spec)

    exhausted = probe.get("exhausted")
    add("boostback_exhausted", exhausted == "boostback",
        _NO_RATIO_SEVERITY["boostback_exhausted"] if exhausted == "boostback" else 0.0,
        None, None, "", "boostback burn ran the stage dry")
    add("entry_exhausted", exhausted == "entry",
        _NO_RATIO_SEVERITY["entry_exhausted"] if exhausted == "entry" else 0.0,
        None, None, "", "entry burn ran the stage dry")
    reached = bool(probe.get("reached"))
    add("corridor_reached", not reached,
        _NO_RATIO_SEVERITY["corridor_reached"] if not reached else 0.0,
        None, None, "", f"probe crossed the corridor plane h0={corridor.h0_m} m")
    if reached:
        miss = probe.get("miss_m")
        if miss is not None:
            add("corridor_miss", miss > corridor.radius_m,
                float(miss) / corridor.radius_m, float(miss),
                corridor.radius_m, "m", "crossing offset from the pad")
        tilt = probe.get("tilt_deg")
        if tilt is not None:
            add("corridor_tilt", tilt > corridor.max_tilt_deg,
                float(tilt) / corridor.max_tilt_deg, float(tilt),
                corridor.max_tilt_deg, "deg", "descent angle at the crossing")
        speed = probe.get("speed_m_s")
        ok = probe.get("landing_forecast_ok")
        bad = False
        if speed is not None:
            bad = (float(speed) > corridor.v_max_m_s) if ok is None else (not ok)
            add("arrival_speed", bad, float(speed) / corridor.v_max_m_s,
                float(speed), corridor.v_max_m_s, "m/s",
                "speed at the crossing"
                + ("" if ok is None else " (closed-loop landing forecast rejects it)"))
            if ok is False:
                add("landing_forecast", True,
                    _NO_RATIO_SEVERITY["landing_forecast"], None, None, "",
                    "closed-loop landing forecast from this handoff does not "
                    "close inside the landing limits")
        # the planner grades the handoff as an elif after miss/tilt/speed,
        # and the envelope mirrors that order (boostback.py _assemble): the
        # engine-on-at-crossing violation only governs when the corridor
        # conditions above it are clean.
        ov = probe.get("handoff_overlap_s")
        if (ov is not None and float(ov) > 0.0 and not bad and ok is not False
                and (miss is None or float(miss) <= corridor.radius_m)
                and (tilt is None or float(tilt) <= corridor.max_tilt_deg)):
            add("handoff_overlap", True, _NO_RATIO_SEVERITY["handoff_overlap"],
                float(ov), 0.0, "s",
                "entry burn still burning at the corridor crossing -- the "
                "handoff state is not an engine-off landing-burn start")
        prop = probe.get("prop_kg")
        need = probe.get("landing_need_kg")
        if prop is not None and need is not None:
            lim = float(need) + landing_reserve_kg + spec.landing_reserve_margin_kg
            add("fuel_margin", float(prop) < lim,
                lim / max(float(prop), 1e-9),
                float(prop), lim, "kg",
                "propellant at the crossing vs landing need + reserve + margin")
    q = probe.get("entry_q_pa")
    if q:
        add("entry_q", float(q) > spec.entry_q_max_pa,
            float(q) / spec.entry_q_max_pa, float(q), spec.entry_q_max_pa,
            "Pa", "peak dynamic pressure over the entry window")
    g = probe.get("entry_g")
    if g:
        add("entry_g", float(g) > spec.entry_g_max,
            float(g) / spec.entry_g_max, float(g), spec.entry_g_max, "g",
            "peak proper load over the entry window")
    return _finalize(conds, spec)


def _finalize(conds: list[EnvelopeCondition], spec: EnvelopeSpec) -> EnvelopeVerdict:
    # ``conds`` arrive in the order plan_boostback_entry grades them; the
    # predicted flight reason is the *first* violated condition's reason, the
    # way ``_assemble`` returns it -- severity then only picks which corridor
    # condition (miss / tilt / speed / forecast) drives a shared
    # CORRIDOR_MISSED, and worst_severity still says how far out it is.
    first = next((c for c in conds if c.violated), None)
    violated = [c for c in conds if c.violated]
    predicted = None if first is None else CONDITION_REASON[first.name]
    gov = first
    if predicted == CORRIDOR_MISSED:
        gov = max((c for c in violated if CONDITION_REASON[c.name] == CORRIDOR_MISSED),
                  key=lambda c: c.severity)
    codes: list[str] = []
    for c in violated:
        code = CONDITION_REASON[c.name]
        if code not in codes:
            codes.append(code)
    if predicted is None:
        tight = min(conds, key=lambda c: -c.severity)
        why = (f"E11-02 envelope {spec.name}: IN -- reference script flies the "
               f"corridor physics from this separation state "
               f"(tightest condition {tight.name} at {tight.severity:.2f}x its "
               f"limit); a search from here starts inside the feasible set")
    else:
        detail = (f"{gov.severity:.2f}x its limit" if gov.value is not None
                  else gov.detail or "not measured")
        others = ", ".join(f"{c.name} {c.severity:.2f}x" for c in violated
                           if c is not gov) or "no other violation"
        why = (f"E11-02 envelope {spec.name}: OUT -- governing condition "
               f"{gov.name} at {detail}; predicted flight reason "
               f"{predicted} for the reference script "
               f"(other violations: {others}); the flight verdict needs no "
               f"search to reach")
    return EnvelopeVerdict(in_envelope=predicted is None, conditions=tuple(conds),
                           governing_condition=None if gov is None else gov.name,
                           governing_reason=predicted,
                           explanation=why,
                           violated_codes=tuple(codes))


def probe_from_plan(plan: Any) -> dict[str, Any]:
    """The ``grade_probe`` mapping of a ``ReturnPlan`` (its corridor report
    and verdict codes) -- what the reference rollout measured. The
    ``exhausted`` and ``landing_forecast_ok`` flags are inferred from the
    plan's reason codes (the planner's screen path passes the rollout's own
    ``landing_ok``; through a returned plan a corridor miss that hides a
    closed-loop forecast reject is graded as a forecast rejection -- same
    predicted reason either way, ``CORRIDOR_MISSED``)."""
    rep = plan.corridor_report
    codes = set(plan.reason_codes)
    exhausted = ("boostback"
                 if INSUFFICIENT_PROPELLANT_BOOSTBACK in codes else
                 "entry" if INSUFFICIENT_PROPELLANT_ENTRY_BURN in codes
                 else None)
    overlap = float(getattr(plan, "handoff_burn_overlap_s", 0.0) or 0.0)
    corridor = plan.corridor
    # a CORRIDOR_MISSED that the handoff overlap alone explains (the
    # corridor miss/tilt/speed numbers are inside their limits) is not a
    # forecast rejection:
    explained = (overlap > 0.0
                 and rep.miss_m is not None and rep.miss_m <= corridor.radius_m
                 and rep.tilt_deg is not None and rep.tilt_deg <= corridor.max_tilt_deg
                 and rep.speed_m_s is not None
                 and rep.speed_m_s <= corridor.v_max_m_s)
    return {"reached": bool(rep.reached), "miss_m": rep.miss_m,
            "tilt_deg": rep.tilt_deg, "speed_m_s": rep.speed_m_s,
            "prop_kg": rep.propellant_available_kg,
            "landing_need_kg": rep.landing_need_kg,
            "entry_q_pa": rep.entry_max_q_pa or 0.0,
            "entry_g": rep.entry_max_proper_g or 0.0,
            "exhausted": exhausted,
            "handoff_overlap_s": overlap,
            "landing_forecast_ok": (None if rep.landing_touchdown_miss_m is None
                                    else (CORRIDOR_MISSED not in codes
                                          or explained))}


def envelope_verdict(sep_state: BoosterState, booster: BoosterModel,
                     recovery_target: RecoveryTarget, spec: EnvelopeSpec, *,
                     physics: Any = None) -> EnvelopeVerdict:
    """The envelope verdict for one separation state: fly the spec's
    reference script once (``search="keep"``, planner dt/atmosphere), grade
    the corridor physics, and report which condition (if any) the state
    violates and what flight reason that predicts. One rollout -- no search
    budget is ever spent on the answer."""
    plan = plan_boostback_entry(sep_state, booster, recovery_target,
                                initial_guess=list(spec.reference_script),
                                search="keep", dt_opt=spec.dt_opt,
                                dt_final=spec.dt_final,
                                verdict_physics=physics,
                                **spec.planner_kwargs())
    sep_prop, floor = fuel_floor_kg(sep_state, booster)
    verdict = grade_probe(probe_from_plan(plan), spec,
                          corridor=recovery_target.corridor,
                          landing_reserve_kg=float(
                              booster.landing_propellant_reserve_kg),
                          sep_propellant_kg=sep_prop, prop_floor_kg=floor)
    probe = dict(verdict.probe or {})
    probe.update({"plan_feasible": bool(plan.feasible),
                  "plan_reason_codes": list(plan.reason_codes),
                  "reference_script": list(spec.reference_script),
                  "spec_name": spec.name})
    return EnvelopeVerdict(in_envelope=verdict.in_envelope,
                           conditions=verdict.conditions,
                           governing_condition=verdict.governing_condition,
                           governing_reason=verdict.governing_reason,
                           explanation=verdict.explanation, probe=probe,
                           violated_codes=verdict.violated_codes)
