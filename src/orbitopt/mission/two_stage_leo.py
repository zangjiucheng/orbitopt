"""M-07 (task #225): the frozen two-stage LEO reference mission.

Epic #215's Definition of Done is one sentence -- ``Launch Site -> Liftoff ->
MECO -> Separation -> SECO -> Parking Orbit -> Orbit Propagation`` -- with a
continuity + mass-conservation gate at every handoff and one task report for
the whole thing. Every link of that chain already exists as its own verified
segment:

* **Launch Site / window** -- :func:`orbitopt.mission.launch_pipeline.run_launch_pipeline`
  (M-04): site + vehicle + orbit + date become a frozen FlightTarget and the
  liftoff epoch.
* **Liftoff -> MECO -> Separation -> SECO** -- the OpenTsiolkovsky 3-DOF ascent
  that pipeline optimizes, whose output is a
  :class:`~orbitopt.flightlab.contract.FlightRun`.
* **SECO -> Parking Orbit** -- :func:`~orbitopt.mission.ascent.ascent_handoff_state`
  plus :func:`~orbitopt.mission.ascent.check_ascent_to_parking_orbit` (M-02).
* **Parking Orbit -> Orbit Propagation** --
  :class:`orbitopt.mission.propagation.OrbitPropagationSegment` (M-05).

What was missing is the *reference scenario*: one frozen, packaged description
of the mission (inputs, expected results, tolerances), the runner that drives
the chain, and the one report that grades it. That is this module:

* :class:`TwoStageLeoScenario` / :func:`scenario_from_dict` /
  :func:`load_scenario` / :func:`reference_scenario` -- the frozen input; the
  packaged reference is ``data/mission_two_stage_leo.yaml``.
* :func:`run_two_stage_leo` -- the chain: station events, staging handoffs
  (M-01 gates across each staging boundary; since E2-02 the first stage's own
  branch of the separation fork is graded too, with the cross-branch
  ``mass.separation_fork`` conservation gate), the
  SECO-to-parking-orbit handoff (M-02), the vehicle mass bookkeeping, and the
  M-05 propagation of the parking orbit.
* :func:`mission_report` -- the unified task report (kind
  ``two-stage-leo-mission``, schema ``verification-report-1.0``): every
  station, every handoff check, the expected-versus-actual table and the child
  reports (launch pipeline, orbit propagation) under ``inputs``.
* :func:`build_from_config` -- the ``orbitopt run`` adapter for mission kind
  ``two-stage-leo``, so the same scenario runs through the unified pipeline,
  its Verification Gate and its scene export.

Optional first-stage recovery (E2-01, task #254)
------------------------------------------------
A ``recovery:`` block makes the same mission one-stage-recoverable: after the
ascent is acquired, :func:`orbitopt.flightlab.recovery.pipeline.fly_recovery`
(FL-304) flies the FL-302/FL-303/FL-305 chain from the run's separation state
and merges the booster track back into the ``FlightRun``, so one call reports
both the upper-stage insertion (the M-07 chain above) and the first-stage
recovery (``recovery.landed`` etc.) over the same combined run. The recovery
gate is its own link of the chain: after separation and mass bookkeeping,
before the upper-stage expected-orbit verdict (:data:`STATUS_RECOVERY_FAILED`).

Branched mission timeline (E2-02, task #255)
--------------------------------------------
From separation the mission is two continuous trajectories, not one: the
upper stage and the first stage each inherit the stack's state at the split.
:mod:`orbitopt.mission.branches` finds that fork in a run and grades both
branch boundaries under the same M-01 contract; the mission report carries
the fork under ``inputs.handoffs.separation_fork`` and its cross-branch mass
gate as check ``mass.separation_fork``.

Full mission verification gate (E2-03, task #256)
-------------------------------------------------
The chain status answers "which link broke?"; the full-mission gate answers
"did injection, the parking-orbit propagation and the recovery touchdown each
succeed, and is the whole mission therefore a success?"
:func:`orbitopt.mission.phases.full_mission_gate` grades the chain's checks per
phase (tri-state verdict + reason codes per phase, overall rule
``orbitopt.mission.phases.OVERALL_RULE``) and the mission report carries the
document under ``inputs.full_mission``. A partial success is never an overall
pass: an injected upper stage with a lost booster is overall ``failed`` with a
green ``orbit_injection`` phase. When the recovery stage runs over the merged
run it additionally records ``ascent.insertion_verified`` -- the Gate-B
conjunction with the touchdown checks excluded -- so a red touchdown cannot
red-grade the injection phase.

Verification coverage gate (E8-03, task #283)
---------------------------------------------
A scenario can declare ``full_mission: true``: the run *is* the complete
physical mission, so the gate's three phases are required physical phases --
an ``expected.propagation_result: not_checked`` or a missing ``recovery:``
block cannot take one out of scope (the declaration is refused, the phase
recorded ``not_checked``, the verdict capped). Every gate document carries
the required-phase coverage (``checked/required``, e.g. ``2/3``) and a
``qualification`` block: the overall description may claim *independently
verified* only when the run declares the full mission, the gate verdict is
``passed`` with complete coverage, and the report's own independent-evidence
rollup passed. See :mod:`orbitopt.mission.phases`.

Two ascent sources, one grading rule
------------------------------------
``ascent.mode: optimize`` flies the ascent now (needs the pinned
OpenTsiolkovsky binary and pygmo, like ``orbitopt launch``). ``ascent.mode:
run_dir`` replays a previously flown run directory -- what the fast test suite
uses, and what any machine without the compute stack uses. The two differ in
what can be *graded*, never in what is claimed: an unassessed stage is recorded
``not_checked`` with the reason, never as a pass.

The expected parking orbit is never invented here either: unless the config
declares ``expected.parking_orbit`` explicitly it is derived from the mission
target itself (:func:`~orbitopt.mission.launch_pipeline.target_orbit_elements`)
with tolerances derived from the target's own constraint block
(:func:`~orbitopt.mission.launch_pipeline._injection_tolerances`) -- the same
20 km altitude band the Gate-B insertion checks enforce.

    from orbitopt.mission.two_stage_leo import reference_scenario, run_two_stage_leo

    mission = run_two_stage_leo(reference_scenario(), "runs/m07")
    print(mission.status, mission.report_path)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Sequence

from orbitopt.mission.ascent import (
    ElementTolerances,
    ascent_handoff_state,
    check_ascent_to_parking_orbit,
    epoch_at_ascent_time,
)
from orbitopt.mission.contract import (
    HandoffCheck,
    HandoffState,
    HandoffTolerances,
    check_handoff,
)
from orbitopt.mission.phases import full_mission_gate

__all__ = [
    "ASCENT_MODES",
    "DEFAULT_STAGING_VELOCITY_M_S",
    "ELEMENT_KEYS",
    "EXPECTED_PROPAGATION_RESULTS",
    "MISSION_KIND",
    "MISSION_REPORT_KIND",
    "REQUIRED_EVENT_CHAIN",
    "STATUSES",
    "STATUS_ASCENT_NOT_CONVERGED",
    "STATUS_HANDOFF_FAILED",
    "STATUS_MASS_BOOKKEEPING_FAILED",
    "STATUS_OK",
    "STATUS_PARKING_ORBIT_OUT_OF_TOLERANCE",
    "STATUS_PROPAGATION_FAILED",
    "STATUS_RECOVERY_FAILED",
    "STATUS_STATION_MISSING",
    "ExpectedResults",
    "PropagationPlan",
    "RecoveryPlan",
    "TwoStageLeoMission",
    "TwoStageLeoScenario",
    "build_from_config",
    "event_chain_checks",
    "load_scenario",
    "mission_report",
    "reference_scenario",
    "reference_scenario_path",
    "run_two_stage_leo",
    "scenario_from_config",
    "scenario_from_dict",
    "scenario_to_dict",
    "scenario_warnings",
    "staging_handoffs",
]

#: Mission-config ``kind`` this module implements (registered in
#: ``orbitopt.mission_config``) and the ``kind`` of its unified report.
MISSION_KIND = "two-stage-leo"
MISSION_REPORT_KIND = "two-stage-leo-mission"
#: Unified-report filename inside the mission's output directory.
_REPORT_FILE = "mission.verification.json"
#: Packaged reference scenario (top-level ``data/`` so the ``data/*.yaml``
#: package-data glob ships it).
_REFERENCE_FILE = "mission_two_stage_leo.yaml"

#: The station chain the DoD names, in the order it must appear in the run's
#: event log. ``ignition`` precedes liftoff and ``impact`` follows in a
#: two-stage flight; neither is part of the graded chain.
REQUIRED_EVENT_CHAIN: tuple[str, ...] = ("liftoff", "meco", "separation", "seco")

#: Which link of the chain failed, in chain precedence order (the earliest
#: failure names the run; the report lists every failure either way).
STATUS_OK = "OK"
STATUS_ASCENT_NOT_CONVERGED = "ASCENT_NOT_CONVERGED"
STATUS_STATION_MISSING = "EVENT_CHAIN_INCOMPLETE"
STATUS_HANDOFF_FAILED = "HANDOFF_CHECK_FAILED"
STATUS_PARKING_ORBIT_OUT_OF_TOLERANCE = "PARKING_ORBIT_OUT_OF_TOLERANCE"
STATUS_MASS_BOOKKEEPING_FAILED = "MASS_BOOKKEEPING_FAILED"
STATUS_RECOVERY_FAILED = "RECOVERY_FAILED"
STATUS_PROPAGATION_FAILED = "PROPAGATION_FAILED"
STATUS_NOT_CHECKED = "NOT_CHECKED"
STATUSES = (STATUS_OK, STATUS_ASCENT_NOT_CONVERGED, STATUS_STATION_MISSING,
            STATUS_HANDOFF_FAILED, STATUS_PARKING_ORBIT_OUT_OF_TOLERANCE,
            STATUS_MASS_BOOKKEEPING_FAILED, STATUS_RECOVERY_FAILED,
            STATUS_PROPAGATION_FAILED, STATUS_NOT_CHECKED)

#: ``ascent.mode`` values: fly it now, or replay a flown run directory.
ASCENT_MODES = ("optimize", "run_dir")

#: Results ``expected.propagation_result`` may demand of the M-05 report.
#: ``not_checked`` is a legal expectation: a propagation that declined to run
#: must never be reported as having passed.
EXPECTED_PROPAGATION_RESULTS = ("passed", "failed", "not_checked")

#: Default velocity gate across a staging boundary. Position/epoch/mass keep
#: the M-01 contract defaults; velocity gets the declared separation impulse as
#: headroom (a spring or pyro push IS a delta-v -- the same rule the M-06 scene
#: exporter applies).
DEFAULT_STAGING_VELOCITY_M_S = 0.01
#: How the staging gap is bridged (:func:`_bridge_across_gap`).
BRIDGE_METHOD = "j2_coast"

#: Element vocabulary of ``expected.parking_orbit`` / ``expected.tolerances``
#: (same names ``orbitopt.mission.launch_pipeline`` grades).
ELEMENT_KEYS = ("semi_major_axis_m", "eccentricity", "inclination_deg", "raan_deg",
                "periapsis_altitude_m", "apoapsis_altitude_m")


# ---------------------------------------------------------------------------
# Small check helpers (the report's vocabulary: id, verdict, measurement)
# ---------------------------------------------------------------------------


def _check(check_id: str, passed: bool, value: float | None = None,
           limit: float | None = None, unit: str = "", reason_code: str = "") -> HandoffCheck:
    """A :class:`~orbitopt.mission.contract.HandoffCheck` for a gate this
    module owns (station presence, expected orbit, mass bookkeeping) -- the
    same record the M-01 gates produce, so one report can list them alike."""
    return HandoffCheck(id=check_id, passed=bool(passed), value=value, limit=limit, unit=unit,
                        reason_code="" if passed else (reason_code or "failed"))


def _checks_doc(checks: Sequence[HandoffCheck]) -> list[dict]:
    return [{"id": c.id, "passed": bool(c.passed), "value": c.value, "limit": c.limit,
             "unit": c.unit, "reason_code": c.reason_code} for c in checks]


def _state_doc(state: HandoffState) -> dict:
    return {"frame": state.frame, "epoch_mjd2000": float(state.epoch.mjd2000),
            "timescale": state.epoch.timescale,
            "position_m": [float(x) for x in state.r_m],
            "velocity_m_s": [float(x) for x in state.v_m_s],
            "mass_kg": float(state.mass_kg)}


def _finite_positive(value: Any, where: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{where} must be a finite number > 0, got {value!r}")
    return value


def _reject_unknown(doc: Mapping[str, Any], allowed: Sequence[str], where: str) -> None:
    unknown = sorted(set(doc) - set(allowed))
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {', '.join(unknown)} "
                         f"(known: {', '.join(sorted(allowed))})")


# ---------------------------------------------------------------------------
# Propagation plan (frozen data; tudatpy only on demand)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PropagationPlan:
    """The frozen M-05 force model and arc length of the reference mission, as
    plain data. :meth:`force_model` builds the
    :class:`~orbitopt.mission.propagation.PropagationForceModel` lazily, so
    reading a scenario never imports numpy or tudatpy."""

    j2: bool = True
    third_bodies: tuple[str, ...] = ()
    #: orbitopt.mission.propagation.DragConfig fields as a plain mapping
    #: (exponential atmosphere, constant Cd), or None for no drag.
    drag: dict | None = None
    revolutions: float = 1.0
    max_step_s: float = 300.0

    def __post_init__(self):
        object.__setattr__(self, "j2", bool(self.j2))
        object.__setattr__(self, "third_bodies", tuple(str(b) for b in self.third_bodies))
        if self.drag is not None:
            if not isinstance(self.drag, Mapping):
                raise ValueError(f"PropagationPlan.drag must be a mapping or None, got {self.drag!r}")
            object.__setattr__(self, "drag", dict(self.drag))
        for name in ("revolutions", "max_step_s"):
            object.__setattr__(self, name, _finite_positive(getattr(self, name),
                                                            f"PropagationPlan.{name}"))

    def force_model(self):
        """The M-05 force model this plan describes (lazy import: the force
        model needs numpy, reading a scenario does not)."""
        from orbitopt.mission.propagation import DragConfig, PropagationForceModel

        return PropagationForceModel(
            j2=self.j2,
            drag=None if self.drag is None else DragConfig(**dict(self.drag)),
            third_bodies=tuple(self.third_bodies))

    def to_dict(self) -> dict:
        return {"j2": bool(self.j2), "third_bodies": list(self.third_bodies),
                "drag": None if self.drag is None else dict(self.drag),
                "revolutions": float(self.revolutions),
                "max_step_s": float(self.max_step_s)}

    @staticmethod
    def from_dict(doc: Mapping[str, Any] | None) -> "PropagationPlan":
        doc = doc or {}
        _reject_unknown(doc, ("j2", "third_bodies", "drag", "revolutions", "max_step_s"),
                        "propagation")
        return PropagationPlan(
            j2=bool(doc.get("j2", True)),
            third_bodies=tuple(doc.get("third_bodies") or ()),
            drag=None if doc.get("drag") is None else dict(doc["drag"]),
            revolutions=float(doc.get("revolutions", 1.0)),
            max_step_s=float(doc.get("max_step_s", 300.0)))


# ---------------------------------------------------------------------------
# Expected results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpectedResults:
    """What the reference mission is expected to produce, and how far off it
    may be. ``events`` is the required station chain; elements left out of
    ``parking_orbit`` / ``tolerances`` fall back to the mission target and its
    constraint block (never invented here -- see the module docstring);
    ``propagation_result`` is the expected tri-state of the M-05 report."""

    events: tuple[str, ...] = REQUIRED_EVENT_CHAIN
    parking_orbit: Mapping[str, float] = field(default_factory=dict)
    tolerances: Mapping[str, float] = field(default_factory=dict)
    propagation_result: str = "passed"

    def __post_init__(self):
        events = tuple(str(e) for e in self.events)
        if not events or any(not e for e in events):
            raise ValueError(f"ExpectedResults.events must be non-empty names, got {self.events!r}")
        object.__setattr__(self, "events", events)
        for name in ("parking_orbit", "tolerances"):
            doc = dict(getattr(self, name) or {})
            _reject_unknown(doc, ELEMENT_KEYS, f"expected.{name}")
            for key, value in list(doc.items()):
                if value is None:
                    continue
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(f"expected.{name}.{key} must be finite, got {doc[key]!r}")
                doc[key] = value
            object.__setattr__(self, name, doc)
        if self.propagation_result not in EXPECTED_PROPAGATION_RESULTS:
            raise ValueError(f"expected.propagation_result {self.propagation_result!r} not in "
                             f"{EXPECTED_PROPAGATION_RESULTS}")

    def to_dict(self) -> dict:
        return {"events": list(self.events), "parking_orbit": dict(self.parking_orbit),
                "tolerances": dict(self.tolerances),
                "propagation_result": self.propagation_result}

    @staticmethod
    def from_dict(doc: Mapping[str, Any] | None) -> "ExpectedResults":
        doc = doc or {}
        _reject_unknown(doc, ("events", "parking_orbit", "tolerances", "propagation_result"),
                        "expected")
        return ExpectedResults(
            events=tuple(doc.get("events") or REQUIRED_EVENT_CHAIN),
            parking_orbit=dict(doc.get("parking_orbit") or {}),
            tolerances=dict(doc.get("tolerances") or {}),
            propagation_result=str(doc.get("propagation_result", "passed")))


# ---------------------------------------------------------------------------
# The recovery stage (E2-01: the same mission, first stage coming home)
# ---------------------------------------------------------------------------


#: Keys of ``recovery.spec`` -- the flight-target ``recovery`` block the M-04
#: target builder accepts (``kind: none`` is not a recovery stage).
RECOVERY_SPEC_KEYS = ("kind", "target", "max_touchdown_speed_m_s",
                      "max_touchdown_lateral_speed_m_s", "min_propellant_reserve_kg")
RECOVERY_SPEC_KINDS = ("downrange", "rtls")


@dataclass(frozen=True)
class RecoveryPlan:
    """The optional first-stage recovery stage of the mission (task #254):
    the same flight, with the booster flown back through FL-302/FL-303/FL-305
    and merged into the ``FlightRun`` before the chain is graded.

    ``spec`` is the optimize-mode knob: the flight-target recovery block
    forwarded verbatim to ``run_launch_pipeline(recovery=...)``, so the flown
    target itself declares the recovery. ``target_file`` is the run_dir-mode
    knob: the FlightTarget the recorded flight is graded against (defaults to
    ``ascent.target_file``); its declared recovery is what
    :func:`~orbitopt.flightlab.recovery.pipeline.fly_recovery` flies.
    ``vehicle_file`` is the vehicle the booster model is read from (default:
    ``launch.vehicle``); ``plan_options`` and ``landing_sites`` pass straight
    through to the recovery pipeline. ``drive_from_separation`` (E5-02, task
    #269) couples the run: the booster branch is flown from the canonical
    separation ``HandoffState`` (:func:`~orbitopt.mission.coupled_branches.
    separation_handoff` -- the stack's state at separation, digest-identified)
    instead of a separately read-out debris-track sample, and the coupled
    branch gates (``handoff.coupled.*``, ``mass.coupled.*``) are graded into
    the chain. Default False: replayed/frozen scenarios keep the E2-01
    read-out path byte-for-byte."""

    spec: Mapping[str, Any] | None = None
    target_file: str | None = None
    vehicle_file: str | None = None
    plan_options: Mapping[str, Any] = field(default_factory=dict)
    landing_sites: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    drive_from_separation: bool = False

    def __post_init__(self):
        if self.spec is not None:
            if not isinstance(self.spec, Mapping):
                raise ValueError(f"recovery.spec must be a mapping or None, got {self.spec!r}")
            spec = dict(self.spec)
            _reject_unknown(spec, RECOVERY_SPEC_KEYS, "recovery.spec")
            kind = str(spec.get("kind") or "")
            if kind not in RECOVERY_SPEC_KINDS:
                raise ValueError(f"recovery.spec.kind {kind!r} not in {RECOVERY_SPEC_KINDS}")
            object.__setattr__(self, "spec", spec)
        for name in ("target_file", "vehicle_file"):
            value = str(getattr(self, name) or "").strip() or None
            object.__setattr__(self, name, value)
        object.__setattr__(self, "plan_options", dict(self.plan_options or {}))
        sites = dict(self.landing_sites or {})
        object.__setattr__(self, "landing_sites",
                           {str(k): dict(v) for k, v in sites.items()})
        object.__setattr__(self, "drive_from_separation", bool(self.drive_from_separation))

    def to_dict(self) -> dict:
        doc = {"spec": None if self.spec is None else dict(self.spec),
               "target_file": self.target_file, "vehicle_file": self.vehicle_file,
               "plan_options": dict(self.plan_options),
               "landing_sites": {k: dict(v) for k, v in self.landing_sites.items()}}
        if self.drive_from_separation:  # keep the default document byte-identical
            doc["drive_from_separation"] = True
        return doc

    @staticmethod
    def from_dict(doc: Mapping[str, Any] | None) -> "RecoveryPlan":
        doc = doc or {}
        _reject_unknown(doc, ("spec", "target_file", "vehicle_file", "plan_options",
                              "landing_sites", "drive_from_separation"), "recovery")
        return RecoveryPlan(
            spec=None if doc.get("spec") is None else dict(doc["spec"]),
            target_file=doc.get("target_file"), vehicle_file=doc.get("vehicle_file"),
            plan_options=dict(doc.get("plan_options") or {}),
            landing_sites=dict(doc.get("landing_sites") or {}),
            drive_from_separation=bool(doc.get("drive_from_separation", False)))


# ---------------------------------------------------------------------------
# The scenario
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TwoStageLeoScenario:
    """The frozen input of the reference mission: what to fly, how to fly it,
    and what the flight has to come back with. Every field is YAML/JSON
    friendly, so :meth:`to_dict` round-trips through :meth:`from_dict`.

    ``ascent`` holds the M-04 optimizer pass-through (``objective``,
    ``payload_kg``, ``seed``, ``algorithm``, ``generations``, ``population``,
    ``workers``, ``local``, ``refly``, ``timeout_s``,
    ``sensitivity_offsets_s``); ``ascent_run_dir`` / ``ascent_target_file`` are
    the ``run_dir`` mode's inputs. ``handoff`` is the M-01 tolerance set used
    for the staging gates and the SECO-to-parking gate, ``elements`` the M-02
    element-agreement tolerances, ``staging_velocity_m_s`` the base velocity
    gate across a staging boundary (before separation-impulse headroom).
    ``full_mission`` (E8-03, task #283) declares the run the complete physical
    mission: every required physical phase is then forced into the
    full-mission gate, out-of-scope declarations cannot bypass one, and the
    report may claim independent verification only with complete coverage and
    passing independent evidence."""

    mission_id: str
    title: str
    site: str
    vehicle: str
    orbit: str
    date_utc: str
    raan_deg: float | None = None
    window_days: float = 14.0
    ascent_mode: str = "optimize"
    ascent_run_dir: str | None = None
    ascent_target_file: str | None = None
    ascent: Mapping[str, Any] = field(default_factory=dict)
    handoff: HandoffTolerances = field(default_factory=HandoffTolerances)
    staging_velocity_m_s: float = DEFAULT_STAGING_VELOCITY_M_S
    elements: ElementTolerances = field(default_factory=ElementTolerances)
    propagation: PropagationPlan = field(default_factory=PropagationPlan)
    expected: ExpectedResults = field(default_factory=ExpectedResults)
    recovery: RecoveryPlan | None = None
    full_mission: bool = False
    coast_orbits: float = 2.0
    earth_texture: bool = True

    def __post_init__(self):
        for name in ("mission_id", "title", "site", "vehicle", "orbit", "date_utc"):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise ValueError(f"scenario.{name} must be a non-empty string")
            object.__setattr__(self, name, value)
        if self.ascent_mode not in ASCENT_MODES:
            raise ValueError(f"ascent.mode {self.ascent_mode!r} not in {ASCENT_MODES}")
        if self.ascent_mode == "run_dir" and not str(self.ascent_run_dir or "").strip():
            raise ValueError("ascent.mode 'run_dir' needs ascent.run_dir: the FlightResult "
                             "directory to replay")
        if self.raan_deg is not None:
            raan = float(self.raan_deg) % 360.0
            object.__setattr__(self, "raan_deg", 0.0 if raan >= 360.0 else raan)
        object.__setattr__(self, "window_days", _finite_positive(
            self.window_days, "scenario.window_days"))
        object.__setattr__(self, "staging_velocity_m_s", _finite_positive(
            self.staging_velocity_m_s, "scenario.staging_velocity_m_s"))
        coast = float(self.coast_orbits)
        if not math.isfinite(coast) or coast < 0.0:
            raise ValueError(f"scenario.coast_orbits must be finite and >= 0, got {coast!r}")
        object.__setattr__(self, "coast_orbits", coast)
        object.__setattr__(self, "earth_texture", bool(self.earth_texture))
        object.__setattr__(self, "full_mission", bool(self.full_mission))
        object.__setattr__(self, "ascent", dict(self.ascent or {}))
        if self.recovery is not None:
            if self.ascent_mode == "optimize" and self.recovery.spec is None:
                raise ValueError("ascent.mode 'optimize' + recovery needs recovery.spec: the "
                                 "flight-target recovery block the ascent is flown against")
            if self.ascent_mode == "run_dir" and self.recovery.spec is not None:
                raise ValueError("recovery.spec is the optimize-mode knob: a replayed run was "
                                 "flown against its recorded target -- use recovery.target_file "
                                 "(or ascent.target_file) to grade it")
            if self.ascent_mode == "run_dir" and not (self.recovery.target_file
                                                      or self.ascent_target_file):
                raise ValueError("ascent.mode 'run_dir' + recovery needs recovery.target_file (or "
                                 "ascent.target_file): the FlightTarget the replayed flight is "
                                 "graded against")

    # -- (de)serialisation --------------------------------------------------

    def to_dict(self) -> dict:
        """The canonical config document -- what :meth:`from_dict` reads back
        and what a ``two-stage-leo`` mission config's ``scenario`` carries.
        The optional ``recovery`` block is written only when the scenario has
        one, so an expendable scenario round-trips unchanged."""
        ascent: dict[str, Any] = {"mode": self.ascent_mode, **dict(self.ascent)}
        if self.ascent_run_dir:
            ascent["run_dir"] = str(self.ascent_run_dir)
        if self.ascent_target_file:
            ascent["target_file"] = str(self.ascent_target_file)
        doc = {
            "mission": {"id": self.mission_id, "title": self.title, "kind": MISSION_KIND},
            "launch": {"site": self.site, "vehicle": self.vehicle, "orbit": self.orbit,
                       "date_utc": self.date_utc, "raan_deg": self.raan_deg,
                       "window_days": float(self.window_days)},
            "ascent": ascent,
            "handoff": {"position_m": self.handoff.position_m,
                        "velocity_m_s": self.handoff.velocity_m_s,
                        "epoch_s": self.handoff.epoch_s,
                        "max_epoch_gap_s": self.handoff.max_epoch_gap_s,
                        "mass_kg": self.handoff.mass_kg,
                        "staging_velocity_m_s": float(self.staging_velocity_m_s),
                        "elements": {"semi_major_axis_m": self.elements.semi_major_axis_m,
                                     "eccentricity": self.elements.eccentricity,
                                     "inclination_deg": self.elements.inclination_deg,
                                     "raan_deg": self.elements.raan_deg,
                                     "apsides_m": self.elements.apsides_m}},
            "propagation": self.propagation.to_dict(),
            "expected": self.expected.to_dict(),
            "scene": {"coast_orbits": float(self.coast_orbits),
                      "earth_texture": bool(self.earth_texture)},
        }
        if self.recovery is not None:
            doc["recovery"] = self.recovery.to_dict()
        if self.full_mission:  # written only when declared, like recovery (E8-03)
            doc["full_mission"] = True
        return doc

    @staticmethod
    def from_dict(doc: Mapping[str, Any]) -> "TwoStageLeoScenario":
        """Read a scenario document (the :meth:`to_dict` shape, i.e. also the
        packaged reference YAML). Raises ``ValueError`` naming the offending
        field; the config schema is the first line of defence, this is the one
        Python callers hit."""
        if not isinstance(doc, Mapping):
            raise ValueError(f"scenario document must be a mapping, got {type(doc).__name__}")
        mission = dict(doc.get("mission") or {})
        launch = dict(doc.get("launch") or {})
        ascent = dict(doc.get("ascent") or {})
        handoff = dict(doc.get("handoff") or {})
        elements = dict(handoff.get("elements") or {})
        scene = dict(doc.get("scene") or {})
        _reject_unknown(mission, ("id", "title", "kind"), "scenario.mission")
        _reject_unknown(launch, ("site", "vehicle", "orbit", "date_utc", "raan_deg",
                                 "window_days"), "scenario.launch")
        _reject_unknown(handoff, ("position_m", "velocity_m_s", "epoch_s", "max_epoch_gap_s",
                                  "mass_kg", "staging_velocity_m_s", "elements"),
                        "scenario.handoff")
        _reject_unknown(scene, ("coast_orbits", "earth_texture"), "scenario.scene")
        kind = mission.get("kind")
        if kind is not None and kind != MISSION_KIND:
            raise ValueError(f"scenario mission.kind {kind!r}, expected {MISSION_KIND!r}")
        for key in ("id", "title"):
            if not str(mission.get(key) or "").strip():
                raise ValueError(f"scenario: mission.{key} is required")
        for key in ("site", "vehicle", "orbit", "date_utc"):
            if not str(launch.get(key) or "").strip():
                raise ValueError(f"scenario: launch.{key} is required")
        gap = handoff.get("max_epoch_gap_s")
        return TwoStageLeoScenario(
            mission_id=str(mission["id"]).strip(), title=str(mission["title"]).strip(),
            site=str(launch["site"]).strip(), vehicle=str(launch["vehicle"]).strip(),
            orbit=str(launch["orbit"]).strip(), date_utc=str(launch["date_utc"]).strip(),
            raan_deg=None if launch.get("raan_deg") is None else float(launch["raan_deg"]),
            window_days=float(launch.get("window_days", 14.0)),
            ascent_mode=str(ascent.get("mode", "optimize")),
            ascent_run_dir=ascent.get("run_dir"),
            ascent_target_file=ascent.get("target_file"),
            ascent={k: v for k, v in ascent.items()
                    if k not in ("mode", "run_dir", "target_file")},
            handoff=HandoffTolerances(
                position_m=float(handoff.get("position_m", 1.0)),
                velocity_m_s=float(handoff.get("velocity_m_s", 0.01)),
                epoch_s=float(handoff.get("epoch_s", 1.0)),
                max_epoch_gap_s=None if gap is None else float(gap),
                mass_kg=float(handoff.get("mass_kg", 1e-3))),
            staging_velocity_m_s=float(handoff.get("staging_velocity_m_s",
                                                   DEFAULT_STAGING_VELOCITY_M_S)),
            elements=ElementTolerances(
                semi_major_axis_m=float(elements.get("semi_major_axis_m", 10.0)),
                eccentricity=float(elements.get("eccentricity", 1e-6)),
                inclination_deg=float(elements.get("inclination_deg", 1e-3)),
                raan_deg=float(elements.get("raan_deg", 1e-2)),
                apsides_m=float(elements.get("apsides_m", 10.0))),
            propagation=PropagationPlan.from_dict(doc.get("propagation")),
            expected=ExpectedResults.from_dict(doc.get("expected")),
            recovery=None if doc.get("recovery") is None else RecoveryPlan.from_dict(
                doc["recovery"]),
            full_mission=bool(doc.get("full_mission", False)),
            coast_orbits=float(scene.get("coast_orbits", 2.0)),
            earth_texture=bool(scene.get("earth_texture", True)))


def scenario_to_dict(scenario: TwoStageLeoScenario) -> dict:
    """Module-level alias of :meth:`TwoStageLeoScenario.to_dict`."""
    return scenario.to_dict()


def scenario_from_dict(doc: Mapping[str, Any]) -> TwoStageLeoScenario:
    """Module-level alias of :meth:`TwoStageLeoScenario.from_dict`."""
    return TwoStageLeoScenario.from_dict(doc)


def reference_scenario_path() -> Path:
    """Path of the packaged reference scenario
    (``orbitopt/data/mission_two_stage_leo.yaml``)."""
    return Path(str(resources.files("orbitopt").joinpath("data", _REFERENCE_FILE)))


def load_scenario(path) -> TwoStageLeoScenario:
    """Read and validate a scenario YAML/JSON document. Accepts both shapes:
    a bare scenario document (:meth:`TwoStageLeoScenario.to_dict`'s shape) or
    a full ``two-stage-leo`` mission config (schema-validated, i.e. the
    packaged reference and anything ``orbitopt run`` takes)."""
    import yaml

    path = Path(path)
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: expected a mapping at the top level, got {type(doc).__name__}")
    if "scenario" in doc or "schemaVersion" in doc:
        return scenario_from_config(doc)
    return TwoStageLeoScenario.from_dict(doc)


def reference_scenario() -> TwoStageLeoScenario:
    """The packaged, frozen two-stage LEO reference scenario of task #225."""
    return load_scenario(reference_scenario_path())


def scenario_warnings(scenario: TwoStageLeoScenario) -> list[str]:
    """Advisory notes about a scenario -- what a caller should know before it
    spends compute on it. Never raises: the scenario is already valid."""
    notes: list[str] = []
    if scenario.ascent_mode == "run_dir":
        notes.append("ascent is replayed from a flown run directory: that flight's launch window "
                     "and ascent optimization are not re-graded here (recorded not_checked, "
                     "never passed)")
    if scenario.raan_deg is None:
        notes.append("RAAN unconstrained: the liftoff epoch is the configured date, not an "
                     "in-plane node pass")
    if scenario.expected.propagation_result != "passed":
        notes.append(f"expected.propagation_result is {scenario.expected.propagation_result!r}: "
                     "a passing M-05 report is not required for the mission to pass, but the "
                     "E5-04 physical gate keeps parking_propagation required "
                     "(a declaration is not evidence)")
    if scenario.recovery is not None:
        notes.append("recovery: the first stage flies the FL-302/FL-303(/FL-305) chain after "
                     "separation and is merged into the run; recovery.landed must be green for "
                     "the mission to pass (an unassessed landing -- solver stack missing -- is "
                     "reported failed with its reason code, never as a pass)")
    else:
        notes.append("full-mission gate: no recovery stage declared -- the recovery_touchdown "
                     "phase is not applicable and the full-mission verdict grades only the "
                     "phases this mission flies (E2-03); the E5-04 physical gate does cap at "
                     "not_checked: a mission that flies no landing is not a full physical "
                     "mission")
    if scenario.full_mission:
        if scenario.expected.propagation_result == "not_checked":
            notes.append("full_mission declared with expected.propagation_result "
                         "'not_checked': parking_propagation is a required physical phase -- "
                         "the out-of-scope declaration is refused and the phase stays "
                         "not_checked (E8-03, task #283)")
        if scenario.recovery is None:
            notes.append("full_mission declared with no recovery stage: recovery_touchdown is "
                         "a required physical phase -- it stays not_checked, caps the "
                         "full-mission verdict and blocks independent verification (E8-03)")
    return notes


# ---------------------------------------------------------------------------
# Expected-orbit resolution (never invented: scenario spec + Gate-B defaults)
# ---------------------------------------------------------------------------


def resolve_expected(scenario: TwoStageLeoScenario) -> tuple[dict, dict, dict]:
    """The graded expected parking orbit, its tolerances and the basis
    document saying where each came from.

    Baseline = the mission's own orbit spec (``launch.orbit``, parsed exactly
    like the M-04 pipeline parses it) plus the Gate-B insertion tolerance
    defaults -- the same derivation
    :func:`orbitopt.mission.launch_pipeline._injection_tolerances` applies to a
    target's constraint block. Anything the scenario declares under
    ``expected.parking_orbit`` / ``expected.tolerances`` overrides the baseline
    key by key (that is how a frozen reference scenario pins its numbers), and
    the basis records which keys were declared and which were derived.
    """
    from orbitopt.flightlab import checks
    from orbitopt.flightlab.target_export import parse_orbit_spec

    if isinstance(scenario.orbit, str):
        peri, apo, inc = parse_orbit_spec(scenario.orbit)
    else:
        peri, apo, inc = (float(v) for v in scenario.orbit)
    a = checks.EARTH_RADIUS_M + 0.5 * (peri + apo)
    e = (apo - peri) / (2.0 * a)
    expected = {"semi_major_axis_m": a, "eccentricity": e, "inclination_deg": float(inc),
                "raan_deg": scenario.raan_deg, "periapsis_altitude_m": float(peri),
                "apoapsis_altitude_m": float(apo)}
    alt_tol = float(checks.DEFAULT_ORBIT_ALTITUDE_TOLERANCE_M)
    inc_tol = float(checks.DEFAULT_INCLINATION_TOLERANCE_DEG)
    tolerances = {"semi_major_axis_m": alt_tol, "eccentricity": alt_tol * (1.0 + e) / a,
                  "inclination_deg": inc_tol, "raan_deg": float(checks.RAAN_TOLERANCE_DEG),
                  "periapsis_altitude_m": alt_tol, "apoapsis_altitude_m": alt_tol}
    declared_elements = {k: v for k, v in scenario.expected.parking_orbit.items()
                         if v is not None}
    declared_tolerances = {k: v for k, v in scenario.expected.tolerances.items()
                           if v is not None}
    expected.update(declared_elements)
    tolerances.update(declared_tolerances)
    basis = {
        "orbit_spec": str(scenario.orbit),
        "constraint_defaults": {
            "orbit_altitude_tolerance_m": alt_tol, "inclination_tolerance_deg": inc_tol,
            "raan_tolerance_deg": float(checks.RAAN_TOLERANCE_DEG)},
        "derived_from": {k: "launch.orbit spec (Gate-B tolerance defaults)"
                         for k in tolerances if k not in declared_tolerances},
        "declared": sorted(declared_elements) + [f"tolerances.{k}" for k in declared_tolerances],
    }
    return expected, tolerances, basis


# ---------------------------------------------------------------------------
# The chain, link by link
# ---------------------------------------------------------------------------


def event_chain_checks(events: Sequence[Mapping[str, Any]],
                       chain: Sequence[str] = REQUIRED_EVENT_CHAIN) -> list[HandoffCheck]:
    """One presence check per required station plus the chain-order check.
    ``events`` are event documents (``{"kind", "t_s", "body_id"}`` or
    :class:`~orbitopt.flightlab.contract.Event`); a station counts at its
    first occurrence."""
    docs = [{"kind": getattr(e, "kind", None) or e["kind"],
             "t_s": float(getattr(e, "t_s", None) if not isinstance(e, Mapping) else e["t_s"]),
             "body_id": getattr(e, "body_id", None) or (e.get("body_id") if isinstance(e, Mapping) else "")}
            for e in events]
    first: dict[str, dict] = {}
    for ev in docs:
        first.setdefault(ev["kind"], ev)
    checks: list[HandoffCheck] = []
    present: list[tuple[str, float]] = []
    for kind in chain:
        ev = first.get(kind)
        if ev is None:
            checks.append(_check(f"station.{kind}", False, reason_code="event_missing"))
        else:
            checks.append(_check(f"station.{kind}", True, value=ev["t_s"], unit="s"))
            present.append((kind, ev["t_s"]))
    ordered = [k for k, _ in present]
    expected_order = [k for k in chain if k in first]
    gaps = [b[1] - a[1] for a, b in zip(present, present[1:]) if b[1] < a[1]]
    if ordered != expected_order or gaps:
        checks.append(_check("station.event_chain_order", False,
                             value=-(min(gaps) if gaps else 0.0) or None,
                             unit="s", reason_code="chain_order"))
    else:
        checks.append(_check("station.event_chain_order", True, unit="s"))
    return checks


def _sample_state(track, index: int, stage_config: Mapping[str, Any]) -> HandoffState:
    """One track sample as a contract state in GCRS (frame bridge on demand,
    same rule as :mod:`orbitopt.mission.ascent` -- a handoff gate always
    compares GCRS with GCRS)."""
    from orbitopt.mission.ascent import _bridge_to_gcrs, epoch_at_ascent_time

    r_m, v_m_s = _bridge_to_gcrs(track, track.position_m[index],
                                 track.velocity_m_s[index], track.t_s[index])
    return HandoffState(
        epoch=epoch_at_ascent_time(track.epoch_utc, track.t_s[index], "UTC"),
        frame="GCRS", r_m=r_m, v_m_s=v_m_s, mass_kg=float(track.mass_kg[index]),
        stage_config=dict(stage_config),
        metadata={"source": "flightlab-track-sample", "body_id": track.body_id,
                  "t_s": float(track.t_s[index]), "ascent_frame": track.frame})


def _ascent_lineage(run) -> list[str]:
    """The mission's continuing stack, upper stage first: the insertion body
    and its ancestors up to the first stage. Staging handoffs are graded along
    this lineage and the separation fork (E2-02: the first stage is a mission
    branch, not a lost handoff) -- a jettisoned fairing diverges off both, on
    purpose, and grading its divergence as a continuity failure would be
    grading the wrong physics (it is recorded, not graded)."""
    from orbitopt.flightlab import checks as gate_checks

    by_id = {t.body_id: t for t in run.tracks}
    try:
        body = gate_checks.insertion_state(run)["body_id"]
    except Exception:
        body = max(run.tracks, key=lambda t: t.t_end_s).body_id
    lineage: list[str] = []
    seen: set[str] = set()
    while body in by_id and body not in seen:
        lineage.append(body)
        seen.add(body)
        body = by_id[body].parent_body_id
    return lineage


def _earth_pole(state: HandoffState) -> tuple[tuple[float, float, float], str]:
    """Earth's rotation axis in ``state``'s frame at its epoch, plus a label
    saying where it came from. GCRS needs the ITRS->GCRS rotation (the pole
    has precessed ~0.4 deg from GCRS z since J2000); without astropy the
    frame's z axis is the documented fallback."""
    if state.frame == "GCRS":
        try:
            from datetime import datetime, timedelta
            from orbitopt.flightlab.frames import itrs_to_gcrs_matrix

            when = datetime(2000, 1, 1) + timedelta(days=state.epoch.to_utc_mjd2000())
            m = itrs_to_gcrs_matrix(when)
            return (m[0][2], m[1][2], m[2][2]), "itrs_z_in_gcrs"
        except Exception:  # astropy missing or epoch outside its tables
            return (0.0, 0.0, 1.0), "gcrs_z_approximation"
    return (0.0, 0.0, 1.0), "frame_z"


def _bridge_across_gap(before: HandoffState, after: HandoffState) -> dict | None:
    """Coast ``before`` to ``after``'s epoch when the two samples are not
    simultaneous: returns ``{state, gap_s, method, pole}``, or ``None`` when
    no bridge is needed (same epoch). The coast is two-body + J2 gravity
    (:func:`orbitopt.flightlab.coast.j2_coast`), the gravity model the ascent
    backend itself flies across the gap; a Kepler bridge left ~0.03 m/s of
    J2 in the residual, above the 0.01 m/s staging gate. Drag over the gap
    is not modelled. A state the coast cannot take (``ValueError``) is
    returned unbridged with ``method: unavailable: ...``, and the caller
    gates the raw pair and records why."""
    gap = before.epoch.delta_seconds(after.epoch)
    if gap < 1e-9:
        return None
    from orbitopt.flightlab.coast import j2_coast

    pole, pole_source = _earth_pole(before)
    try:
        r, v = j2_coast(before.r_m, before.v_m_s, gap, pole=pole)
    except ValueError as exc:
        return {"state": before, "gap_s": gap, "pole": pole_source,
                "method": f"unavailable: J2 coast bridge impossible ({exc})"}
    return {"state": HandoffState(
                epoch=after.epoch, frame=before.frame,
                r_m=tuple(float(x) for x in r),
                v_m_s=tuple(float(x) for x in v),
                mass_kg=before.mass_kg, stage_config=before.stage_config,
                metadata={**before.metadata, "staging_bridge": BRIDGE_METHOD,
                          "bridge_gap_s": gap, "bridge_pole": pole_source}),
            "gap_s": gap, "method": BRIDGE_METHOD, "pole": pole_source}


def staging_handoffs(run, tolerances: HandoffTolerances,
                     staging_velocity_m_s: float = DEFAULT_STAGING_VELOCITY_M_S
                     ) -> list[dict]:
    """Every staging boundary of a multi-body run, as the M-01 handoff it is:
    the parent body's final sample against its child's first sample, gated by
    the four contract checks with the declared separation-impulse headroom on
    velocity and the stage-config change licensing the mass drop.

    Like-for-like epochs: a backend samples each body on its own clock, so the
    parent's track typically ends a second or two *before* the child's begins
    (the FL-004 two-stage output does exactly that, 206 s vs 208 s). Comparing
    those two samples raw measures the coast in between -- thousands of metres,
    physics, not a lost handoff. The parent state is therefore bridged across
    the gap by a two-body + J2 coast (:func:`orbitopt.flightlab.coast.j2_coast`,
    the gravity the ascent backend flies over the gap) before
    the gate runs, and the bridge is recorded on the handoff document
    (``bridge``: gap, method and the pole source). A state the coast cannot
    take is gated raw and the document says why (``method:
    unavailable:``-reason) -- the failure is reported, never hidden.

    Boundaries on the :func:`_ascent_lineage` (the insertion body's ancestor
    chain) **and on the E2-02 separation fork** (the first stage's own branch,
    :func:`orbitopt.mission.branches.separation_fork` -- two continuous
    missions start at separation, not one) are graded. A boundary off both --
    a fairing jettisoned off a branch -- is still recorded (``graded: false``,
    with the reason), because the run did stage there; it is simply not the
    state a mission segment inherits, so its divergence is physics, not a lost
    handoff. Returns one document per boundary, mission order first; empty for
    a single-body run."""
    from orbitopt.mission.branches import separation_fork

    boundary_tol = HandoffTolerances(
        position_m=tolerances.position_m, velocity_m_s=float(staging_velocity_m_s),
        epoch_s=tolerances.epoch_s,
        max_epoch_gap_s=tolerances.max_epoch_gap_s, mass_kg=tolerances.mass_kg)
    lineage = _ascent_lineage(run)
    graded_pairs = set(zip(lineage[1:], lineage[:-1]))  # (parent, child) up the chain
    fork = separation_fork(run)
    if fork is not None:
        graded_pairs |= {(fork["parent_body"], b["body_id"]) for b in fork["branches"]}
    docs: list[dict] = []
    for track in run.tracks:
        if track.parent_body_id is None:
            continue
        parent = next((t for t in run.tracks if t.body_id == track.parent_body_id), None)
        if parent is None:
            continue
        # A fork boundary's separation event may name any member of the split:
        # the parent, the child, a sibling branch (the event is recorded on the
        # continuing stack, with ``separated_from`` naming the shed body), or
        # -- in an E2-01 combined run -- the debris the booster track replaced.
        family = {track.body_id, track.parent_body_id}
        if fork is not None and track.parent_body_id == fork["parent_body"]:
            family |= {b["body_id"] for b in fork["branches"]}
        separation = next((float(e.t_s) for e in run.events
                           if e.kind == "separation"
                           and (e.body_id in family
                                or (e.data or {}).get("separated_from") in family)),
                          None)
        graded = (parent.body_id, track.body_id) in graded_pairs
        before = _sample_state(parent, len(parent.t_s) - 1,
                               {"body_id": parent.body_id,
                                "parent_body_id": parent.parent_body_id or ""})
        after = _sample_state(track, 0, {"body_id": track.body_id,
                                         "parent_body_id": track.parent_body_id or ""})
        bridge = _bridge_across_gap(before, after)
        if bridge is not None:
            before = bridge["state"]
        doc: dict = {"parent_body": parent.body_id, "child_body": track.body_id,
                     "separation_t_s": separation, "graded": graded,
                     "bridge": (None if bridge is None else
                                {k: bridge[k] for k in ("gap_s", "method", "pole")}),
                     "before": _state_doc(before), "after": _state_doc(after)}
        if not graded:
            doc["checks"] = []
            doc["passed"] = None
            doc["reason"] = ("off the ascent lineage and off the separation fork "
                             "(a fairing jettisoned off a branch): recorded, "
                             "not graded -- its divergence is physics, not a lost handoff")
            docs.append(doc)
            continue
        report = check_handoff(before, after, boundary_tol)
        checks = [HandoffCheck(id=f"handoff.{track.body_id}.{c.id.split('.', 1)[1]}",
                               passed=c.passed, value=c.value, limit=c.limit, unit=c.unit,
                               reason_code=c.reason_code) for c in report.checks]
        checks.append(_check(
            f"handoff.{track.body_id}.separation_event", separation is not None,
            value=separation, unit="s",
            reason_code="" if separation is not None else "event_missing"))
        doc["checks"] = _checks_doc(checks)
        doc["passed"] = all(c.passed for c in checks)
        docs.append(doc)
    return docs


def mass_checks(run, tolerances: HandoffTolerances) -> list[HandoffCheck]:
    """Vehicle-level mass bookkeeping for the whole run: no body's tracked
    mass may rise (and a fall must stay within the declared bookkeeping
    tolerance -- staging drops are graded at the handoff itself, where the
    stage-config change is visible), and the SECO insertion mass the handoff
    carries is the upper stage's own tracked final mass."""
    tol = float(tolerances.mass_kg)
    checks: list[HandoffCheck] = []
    for track in run.tracks:
        rises = [b - a for a, b in zip(track.mass_kg, track.mass_kg[1:]) if b - a > tol]
        checks.append(_check(
            f"mass.{track.body_id}.non_increasing", not rises,
            value=max(rises) if rises else 0.0, limit=tol, unit="kg",
            reason_code="mass_gain" if rises else ""))
    return checks


# ---------------------------------------------------------------------------
# The mission
# ---------------------------------------------------------------------------


@dataclass
class TwoStageLeoMission:
    """One run of the reference mission: the chain's evidence and its verdict.
    ``status`` is the chain diagnosis (:data:`STATUSES` precedence, earliest
    failing link first); ``checks`` is every graded check, and ``report`` /
    ``report_path`` carry the unified task report."""

    scenario: TwoStageLeoScenario
    status: str
    message: str
    reason_codes: tuple[str, ...] = ()
    out_dir: Path = field(default_factory=lambda: Path("."))
    ascent_run_dir: Path | None = None
    run_id: str | None = None
    launch: dict | None = None
    stations: list = field(default_factory=list)
    staging: list = field(default_factory=list)
    branches: dict | None = None
    parking: dict | None = None
    parking_elements: dict = field(default_factory=dict)
    expected_elements: dict = field(default_factory=dict)
    expected_tolerances: dict = field(default_factory=dict)
    expected_basis: dict = field(default_factory=dict)
    propagation: dict = field(default_factory=dict)
    recovery: dict | None = None
    recovery_run_dir: Path | None = None
    coupled_branches: dict | None = None
    checks: tuple[HandoffCheck, ...] = ()
    not_assessed: list = field(default_factory=list)
    report: dict = field(default_factory=dict)
    full_mission: dict = field(default_factory=dict)
    report_path: Path | None = None

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    def to_dict(self) -> dict:
        return {
            "mission_id": self.scenario.mission_id, "kind": MISSION_KIND,
            "status": self.status, "message": self.message,
            "reason_codes": list(self.reason_codes),
            "out_dir": str(self.out_dir),
            "ascent_run_dir": str(self.ascent_run_dir) if self.ascent_run_dir else None,
            "run_id": self.run_id, "launch": self.launch,
            "stations": self.stations, "staging": self.staging,
            "branches": self.branches, "parking": self.parking,
            "parking_elements": self.parking_elements,
            "expected_elements": self.expected_elements,
            "propagation": self.propagation,
            "recovery": self.recovery,
            "recovery_run_dir": (str(self.recovery_run_dir) if self.recovery_run_dir
                                 else None),
            "coupled_branches": self.coupled_branches,
            "checks": _checks_doc(self.checks), "not_assessed": self.not_assessed,
            "full_mission": self.full_mission,
            "report_file": str(self.report_path) if self.report_path else None,
        }


def _relabel(checks: Sequence[HandoffCheck], prefix: str) -> list[HandoffCheck]:
    return [HandoffCheck(id=f"{prefix}.{c.id.split('.', 1)[1] if '.' in c.id else c.id}",
                         passed=c.passed, value=c.value, limit=c.limit, unit=c.unit,
                         reason_code=c.reason_code) for c in checks]


def _acquire_ascent(scenario: TwoStageLeoScenario, out: Path):
    """Fly (optimize) or load (run_dir) the ascent; returns
    ``(run, launch_result_or_None, failure_message_or_None)``. Raises for
    input trouble (bad mode kwargs, unreadable run dir) -- a mission-level
    failure comes back as the third element, never an exception."""
    from orbitopt.flightlab.contract import read_run

    if scenario.ascent_mode == "run_dir":
        run = read_run(scenario.ascent_run_dir)
        return run, None, None
    from orbitopt.mission.launch_pipeline import run_launch_pipeline

    _reject_unknown(scenario.ascent, ("objective", "payload_kg", "seed", "algorithm",
                                      "generations", "population", "workers", "local",
                                      "refly", "timeout_s", "sensitivity_offsets_s"),
                    "ascent")
    pipeline = run_launch_pipeline(
        site=scenario.site, orbit=scenario.orbit, vehicle=scenario.vehicle,
        date_utc=scenario.date_utc, raan_deg=scenario.raan_deg,
        window_days=scenario.window_days, out_dir=out / "launch",
        recovery=None if scenario.recovery is None else dict(scenario.recovery.spec or {}),
        # E5-05 (task #272): with a recovery declared, the pipeline's own
        # winner re-flight must stay ascent-only -- the recovery this mission
        # grades is the one _fly_recovery_stage flies below (with the
        # scenario's plan options and the E5-02 separation drive), over this
        # very ascent, exactly once.
        run_recovery=scenario.recovery is None,
        **dict(scenario.ascent))
    if pipeline.ascent_run_dir is None:
        # No flight at all (infeasible window, optimizer abort): there is no
        # chain to grade -- the diagnosis is the pipeline's own status.
        return None, pipeline.to_dict(), pipeline.message
    return read_run(pipeline.ascent_run_dir), pipeline.to_dict(), None


def _fly_recovery_stage(scenario: TwoStageLeoScenario, run, out: Path, *,
                        handoff_report_file: Path | None = None,
                        separation=None,
                        ) -> tuple[object, dict, list[HandoffCheck], list[dict]]:
    """Fly the FL-304 recovery stage over the acquired ascent (task #254): the
    booster is planned (FL-302), flown and, when the landing stack is present,
    landed (FL-303/FL-305); the combined run (module docstring) is written to
    ``<out>/recovery/run`` and re-evaluated against the recovery target, so
    ``ascent.flight_verified`` downstream means insertion *and* touchdown --
    the Gate-B semantics of :func:`orbitopt.flightlab.simulate.simulate`.

    Returns ``(combined_run, block, checks, not_assessed)``. Input trouble
    (target without a recovery declaration, unreadable vehicle) raises; a
    mission-level failure -- infeasible plan, missed touchdown, a landing the
    solver stack could not assess -- comes back as a red ``recovery.*`` check,
    never as an exception. ``handoff_report_file`` (E5-03, task #270)
    additionally writes the R-02 ``RecoveryHandoffReport`` document there, so
    a cross-environment orchestrator can seal the M-01 boundary states without
    re-flying anything; it does not change the returned block. ``separation``
    (E5-02, task #269) is the canonical ``SeparationHandoff`` computed from
    the ascent run before the merge: when given, the booster is flown from
    that shared state (``fly_recovery(sep_state=...)`` at the branch's own
    separation mass) instead of a separately read-out debris sample, and the
    block records ``driven_from_separation`` with the handoff's digest.
    """
    from orbitopt.flightlab import checks as gate_checks
    from orbitopt.flightlab.contract import load_target, write_run
    from orbitopt.flightlab.recovery.pipeline import fly_recovery
    from orbitopt.flightlab.vehicle_source import data_root

    plan = scenario.recovery
    if scenario.ascent_mode == "run_dir":
        target = load_target(plan.target_file or scenario.ascent_target_file)
    else:
        target = load_target(out / "launch" / "target.yaml")
    if target.recovery.kind == "none":
        raise ValueError(
            "recovery needs a target that declares one: the flight target has "
            "recovery.kind 'none' (set recovery.spec in optimize mode, or point "
            "recovery.target_file at the flown target in run_dir mode)")
    vehicle = plan.vehicle_file or scenario.vehicle
    vehicle_path = Path(vehicle)
    if not vehicle_path.is_file():
        vehicle_path = data_root() / "vehicles" / f"{vehicle_path.stem}.yaml"
        if not vehicle_path.is_file():
            raise FileNotFoundError(f"recovery vehicle {vehicle!r}: not a file and not in the "
                                    "packaged vehicle catalog")

    fly_kwargs: dict = {"plan_options": dict(plan.plan_options) or None,
                        "landing_sites": dict(plan.landing_sites) or None}
    if separation is not None:  # E5-02: fly the booster from the shared handoff
        from orbitopt.mission.coupled_branches import booster_state_from_run

        fly_kwargs["sep_state"] = booster_state_from_run(run, separation)
    outcome = fly_recovery(run, target, vehicle_path, **fly_kwargs)
    # Gate-B re-assessment over the merged run: the checks the pipeline
    # contributed (landing gate, touchdown limits, reserve) are kept, the
    # target-declared limits are re-measured against the flown booster track --
    # the same post-merge semantics as orbitopt.flightlab.simulate.
    combined = replace(outcome.run,
                       verification=gate_checks.evaluate(outcome.run, target))
    run_dir = out / "recovery" / "run"
    write_run(run_dir, combined)
    if handoff_report_file is not None and outcome.handoff is not None:
        import json

        report_path = Path(handoff_report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(outcome.handoff.to_dict(), indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8")
    target_file = (str(Path(plan.target_file or scenario.ascent_target_file))
                   if scenario.ascent_mode == "run_dir"
                   else str(out / "launch" / "target.yaml"))

    checks: list[HandoffCheck] = []
    landing_gate = next((c for c in outcome.checks if c.id == gate_checks.RECOVERY_GATE_CHECK_ID),
                        None)
    checks.append(_check(
        "recovery.landed", bool(outcome.success),
        reason_code=(landing_gate.reason_code
                     if landing_gate is not None and landing_gate.reason_code
                     else (outcome.reason_codes[0] if outcome.reason_codes else "failed"))))
    # E2-03 (task #256): the merged run's Gate-B conjunction grades touchdown
    # inside ``ascent.flight_verified``; record the insertion-only half
    # separately so a red recovery cannot red-grade the injection phase.
    insertion = [c for c in combined.verification.checks
                 if c.id in gate_checks.CONSTRAINT_CHECK_IDS
                 and c.id not in gate_checks.LANDING_CONSTRAINT_CHECK_IDS
                 and c.id not in (gate_checks.RECOVERY_GATE_CHECK_ID,
                                  "recovery-propellant-reserve")]
    insertion_failures = [c for c in insertion if not c.passed]
    insertion_notes: list[dict] = []
    if insertion:
        checks.append(_check(
            "ascent.insertion_verified", not insertion_failures,
            reason_code=(insertion_failures[0].reason_code or "insertion_constraints_failed")
            if insertion_failures else ""))
    else:
        insertion_notes = [{"id": "ascent.insertion_verified",
                            "reason": "the combined run records no insertion constraint checks "
                                      "-- never treated as passed"}]
    reserve = next((c for c in outcome.checks if c.id == "recovery-propellant-reserve"), None)
    if reserve is not None:
        checks.append(_check("recovery.propellant_reserve", bool(reserve.passed),
                             value=reserve.value, limit=reserve.limit, unit=reserve.unit,
                             reason_code=reserve.reason_code))
    for c in (outcome.landing.checks if outcome.landing is not None else ()):
        checks.append(_check(f"recovery.{c.id}", bool(c.passed), value=c.value, limit=c.limit,
                             unit=c.unit, reason_code=c.reason_code))
    handoff = outcome.handoff
    for c in (handoff.checks if handoff is not None else ()):
        checks.append(_check(f"recovery.{c.id}", bool(c.passed), value=c.value, limit=c.limit,
                             unit=c.unit, reason_code=c.reason_code))
    notes = insertion_notes + [{"id": f"recovery.{u.id}", "reason": f"phase {u.phase}: {u.note}"}
                               for u in (handoff.unassessed if handoff is not None else ())]

    block: dict = {
        "status": "flown",
        "target_file": target_file,
        "kind": target.recovery.kind,
        "target": target.recovery.target,
        "vehicle": str(vehicle_path),
        "run_dir": str(run_dir),
        "success": bool(outcome.success),
        "reason_codes": list(outcome.reason_codes),
        "plan_feasible": None if outcome.plan is None else bool(outcome.plan.feasible),
        "ignitions_used": int(outcome.plan.ignitions_used) if outcome.plan is not None else 0,
        "landing_feasible": None if outcome.landing is None else bool(outcome.landing.feasible),
        "landed": bool(outcome.landing.landed) if outcome.landing is not None else False,
        "touchdown_enu_m": (list(outcome.landing.touchdown_enu_m)
                            if outcome.landing is not None
                            and outcome.landing.touchdown_enu_m is not None else None),
        "vertical_speed_m_s": (outcome.landing.vertical_speed_m_s
                               if outcome.landing is not None else None),
        "position_error_m": (outcome.landing.position_error_m
                             if outcome.landing is not None else None),
        "max_tilt_deg": (outcome.landing.max_tilt_deg if outcome.landing is not None else None),
        "handoff_passed": None if handoff is None else bool(handoff.passed),
        "assumptions": [str(a) for a in outcome.assumptions],
    }
    if separation is not None:
        block["driven_from_separation"] = True
        block["separation_handoff_id"] = separation.handoff_id
    return combined, block, checks, notes


def _propagate(scenario: TwoStageLeoScenario, parking_state, out: Path,
               tolerances: HandoffTolerances) -> tuple[dict, list[HandoffCheck], list[dict]]:
    """Run (or honestly decline to run) the M-05 segment over the parking
    orbit; returns ``(block, checks, not_assessed)``."""
    block: dict = {"status": "not_run"}
    checks: list[HandoffCheck] = []
    notes: list[dict] = []
    report_path = out / "orbit_propagation.verification.json"
    try:
        from orbitopt.mission.propagation import OrbitPropagationSegment

        segment = OrbitPropagationSegment(
            force_model=scenario.propagation.force_model(),
            revolutions=scenario.propagation.revolutions,
            max_step_s=scenario.propagation.max_step_s,
            report_path=report_path,
            label=f"{scenario.mission_id} parking orbit")
        result = segment.run(parking_state)
    except ImportError as exc:
        block = {"status": "not_run", "error": f"{type(exc).__name__}: {exc}"}
        if scenario.expected.propagation_result == "passed":
            checks.append(_check("propagation.ran", False, reason_code="propagation_unavailable"))
            block["status"] = "failed"
        else:
            notes.append({"id": "propagation", "reason": f"compute stack missing: {exc}"})
        return block, checks, notes
    except Exception as exc:  # a failed propagation is a verdict, not a crash
        block = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        checks.append(_check("propagation.ran", False, reason_code="propagation_error"))
        return block, checks, notes

    report = segment_report = result.diagnostics.get("report_status") or {}
    result_state = result.final_state
    events = result.events[0] if result.events else {}
    block = {
        "status": "ran", "segment": result.segment,
        "result": str(report.get("result", "not_checked")),
        "digest": str(result.diagnostics.get("report_digest", "")),
        "report_file": str(report_path),
        "duration_s": float(events.get("duration_s", 0.0)),
        "revolutions": float(events.get("revolutions", scenario.propagation.revolutions)),
        "final_elements": dict((result.diagnostics.get("analytic_comparison") or {})),
    }
    checks.append(_check("propagation.ran", True))
    checks.append(_check("propagation.report_passed",
                         block["result"] == "passed" or (
                             block["result"] == "failed"
                             and scenario.expected.propagation_result == "failed"),
                         reason_code=f"report_{block['result']}"))
    delta = parking_state.epoch.delta_seconds(result_state.epoch)
    checks.append(_check("propagation.epoch_forward", delta > -tolerances.epoch_s,
                         value=delta, limit=-tolerances.epoch_s, unit="s",
                         reason_code="epoch_backwards"))
    checks.append(_check("propagation.mass_conservation",
                         abs(result_state.mass_kg - parking_state.mass_kg) <= tolerances.mass_kg,
                         value=result_state.mass_kg - parking_state.mass_kg,
                         limit=tolerances.mass_kg, unit="kg",
                         reason_code="mass_drift"))
    return block, checks, notes


def run_two_stage_leo(scenario: TwoStageLeoScenario, out_dir, *,
                      propagate: bool | None = None) -> TwoStageLeoMission:
    """Drive the frozen scenario end to end and grade it: ascent (flown now or
    replayed) -> station chain -> staging handoffs -> SECO-to-parking handoff
    -> mass bookkeeping -> [first-stage recovery, when the scenario declares
    one: FL-304 ``fly_recovery`` over the acquired ascent, its gates graded as
    :data:`STATUS_RECOVERY_FAILED`] -> expected-orbit agreement -> orbit
    propagation. The unified report lands at ``<out_dir>/mission.verification
    .json``; the M-04 launch-pipeline report (``optimize`` mode), the recovery
    run directory and the M-05 propagation report stay next to it and are
    referenced from it by path and digest."""
    from orbitopt.mission.ascent import ascent_handoff_state, check_ascent_to_parking_orbit
    from orbitopt.mission.branches import branch_handoffs

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    want_propagation = (scenario.expected.propagation_result != "not_checked"
                        if propagate is None else bool(propagate))
    expected_elements, expected_tolerances, basis = resolve_expected(scenario)

    run, launch, ascent_failure = _acquire_ascent(scenario, out)
    checks: list[HandoffCheck] = []
    not_assessed: list[dict] = []
    stations: list[dict] = []
    staging: list[dict] = []
    branches: dict | None = None
    parking: dict | None = None
    parking_elements: dict = {}
    propagation: dict = {"status": "not_run"}
    recovery_block: dict | None = None
    recovery_run_dir: Path | None = None
    separation = None
    coupled_doc: dict | None = None

    if run is None:
        checks.append(_check("ascent.flight_verified", False,
                             reason_code=(launch or {}).get("reason_codes", ["no_flight"])[0]
                             if launch else "no_flight"))
        message = f"ascent did not produce a flown run: {ascent_failure}"
        if scenario.recovery is not None:
            not_assessed.append({"id": "recovery",
                                 "reason": "the first stage never separated: the ascent "
                                           "produced no flown run"})
    else:
        if scenario.recovery is not None:
            if scenario.recovery.drive_from_separation:  # E5-02 (task #269)
                from orbitopt.mission.coupled_branches import separation_handoff

                separation = separation_handoff(run)
            run, recovery_block, recovery_checks, recovery_notes = _fly_recovery_stage(
                scenario, run, out, separation=separation)
            recovery_run_dir = Path(recovery_block["run_dir"])
            checks.extend(recovery_checks)
            not_assessed.extend(recovery_notes)
        elif scenario.ascent_mode == "run_dir" and scenario.ascent_target_file:
            # E5-05 (task #272): a replayed flight carries the Gate-B verdict
            # recorded against its flown target -- which for the RTLS design
            # campaign declares the recovery gates the landing branch grades.
            # A mission without a recovery leg (including the cross-env ascent
            # stage, where the landing leg runs in another environment) cannot
            # assess those gates, so re-evaluate the replayed run against the
            # target with the recovery declaration stripped.
            from orbitopt.flightlab import checks as gate_checks
            from orbitopt.flightlab.contract import Recovery, load_target

            flown_target = load_target(scenario.ascent_target_file)
            if flown_target.recovery.kind != "none":
                run = replace(run, verification=gate_checks.evaluate(
                    run, replace(flown_target, recovery=Recovery())))
        checks.append(_check("ascent.flight_verified",
                             bool(run.verification.mission_constraints_passed),
                             reason_code="mission_constraints_failed"))
        if run.verification.optimizer_converged is None:
            not_assessed.append({"id": "ascent.optimizer_converged",
                                 "reason": "run records no optimizer verdict"})
        else:
            checks.append(_check("ascent.optimizer_converged",
                                 bool(run.verification.optimizer_converged),
                                 reason_code="optimizer_not_converged"))
        if run.verification.physics_verified is None:
            not_assessed.append({"id": "ascent.physics_verified",
                                 "reason": "Gate-B re-fly skipped or unavailable -- never "
                                           "treated as passed"})
        else:
            checks.append(_check("ascent.physics_verified",
                                 bool(run.verification.physics_verified),
                                 reason_code="physics_verification_failed"))
        if scenario.ascent_mode == "run_dir":
            not_assessed.append({"id": "launch.window_and_optimization",
                                 "reason": "run_dir mode replays a flown run; its launch window "
                                           "and ascent optimization are recorded, not re-graded"})

        chain = event_chain_checks(run.events, scenario.expected.events)
        checks.extend(chain)
        stations = [{"kind": c.id.split(".", 1)[1], "t_s": c.value, "passed": c.passed}
                    for c in chain if c.id.startswith("station.") and c.id != "station.event_chain_order"]

        staging = staging_handoffs(run, scenario.handoff, scenario.staging_velocity_m_s)
        for doc in staging:
            checks.extend(HandoffCheck(id=c["id"], passed=c["passed"], value=c["value"],
                                       limit=c["limit"], unit=c["unit"],
                                       reason_code=c["reason_code"]) for c in doc["checks"])
        branches = branch_handoffs(run, scenario.handoff, scenario.staging_velocity_m_s,
                                   staging)
        if branches is not None:
            checks.extend(HandoffCheck(id=c["id"], passed=c["passed"], value=c["value"],
                                       limit=c["limit"], unit=c["unit"],
                                       reason_code=c["reason_code"])
                          for c in branches["checks"])

        from orbitopt.mission.coupled_branches import coupled_branches

        coupled_doc = coupled_branches(run, scenario.handoff,
                                       scenario.staging_velocity_m_s, staging,
                                       handoff=separation)
        if coupled_doc is not None and separation is not None:
            coupled_doc["driven_from_separation"] = True
            checks.extend(HandoffCheck(id=c["id"], passed=c["passed"], value=c["value"],
                                       limit=c["limit"], unit=c["unit"],
                                       reason_code=c["reason_code"])
                          for c in coupled_doc["checks"])
        if (scenario.recovery is not None and scenario.recovery.drive_from_separation
                and separation is None):
            checks.append(_check("handoff.coupled.separation_found", False,
                                 reason_code="no_separation_fork_to_drive_from"))

        state = ascent_handoff_state(run)
        report = check_ascent_to_parking_orbit(run, state, scenario.handoff, scenario.elements)
        parking_checks = _relabel(report.checks, "parking")
        checks.extend(parking_checks)
        parking_elements = dict(report.parking_elements) or parking_orbit_elements(state)
        parking = {
            "state": _state_doc(state), "passed": bool(report.passed),
            "checks": _checks_doc(parking_checks),
            "ascent_elements": dict(report.ascent_elements),
            "parking_elements": dict(report.parking_elements),
        }

        checks.extend(mass_checks(run, scenario.handoff))

        for key, label, unit in (("semi_major_axis_m", "semi_major_axis", "m"),
                                 ("eccentricity", "eccentricity", ""),
                                 ("inclination_deg", "inclination", "deg"),
                                 ("periapsis_altitude_m", "periapsis", "m"),
                                 ("apoapsis_altitude_m", "apoapsis", "m")):
            got, want, tol = parking_elements.get(key), expected_elements.get(key), \
                expected_tolerances.get(key)
            if got is None or want is None or tol is None:
                not_assessed.append({"id": f"expected.{label}",
                                     "reason": "parking-orbit element not available"})
                continue
            deviation = abs(float(got) - float(want))
            checks.append(_check(f"expected.{label}", deviation <= float(tol),
                                 value=deviation, limit=float(tol), unit=unit,
                                 reason_code="out_of_tolerance"))
        if expected_elements.get("raan_deg") is None:
            not_assessed.append({"id": "expected.raan", "reason": "RAAN unconstrained"})
        else:
            got = parking_elements.get("raan_deg")
            from orbitopt.mission.ascent import _angle_delta

            dev = _angle_delta(float(got), float(expected_elements["raan_deg"]))
            tol = float(expected_tolerances["raan_deg"])
            checks.append(_check("expected.raan", dev <= tol, value=dev, limit=tol, unit="deg",
                                 reason_code="out_of_tolerance"))

        if want_propagation:
            propagation, prop_checks, prop_notes = _propagate(scenario, state, out,
                                                              scenario.handoff)
            checks.extend(prop_checks)
            not_assessed.extend(prop_notes)
        else:
            propagation = {"status": "not_run",
                           "reason": "propagation not requested (expected result: "
                                     f"{scenario.expected.propagation_result!r})"}
            not_assessed.append({"id": "propagation", "reason": "skipped by caller"})
        if coupled_doc is not None:  # both branch outcomes in one run (E5-02)
            from orbitopt.mission.coupled_branches import record_branch_results

            record_branch_results(
                coupled_doc,
                upper_stage={"insertion_passed": bool(parking and parking["passed"]),
                             "parking_elements": dict(parking_elements),
                             "propagation_status": propagation.get("status")},
                booster=None if recovery_block is None else dict(recovery_block))
        message = ""

    status, message, reason_codes = _grade(checks, run)
    if run is None:
        ascent_run_dir = None
    elif scenario.ascent_mode == "run_dir":
        ascent_run_dir = Path(scenario.ascent_run_dir)
    else:
        ascent_run_dir = Path(launch["ascent_run_dir"]) if launch and launch.get("ascent_run_dir") \
            else None
    mission = TwoStageLeoMission(
        scenario=scenario, status=status, message=message, reason_codes=reason_codes,
        out_dir=out, ascent_run_dir=ascent_run_dir,
        run_id=getattr(run, "run_id", None), launch=launch,
        stations=stations, staging=staging, branches=branches, parking=parking,
        parking_elements=parking_elements, expected_elements=expected_elements,
        expected_tolerances=expected_tolerances, expected_basis=basis,
        propagation=propagation, recovery=recovery_block,
        recovery_run_dir=recovery_run_dir, coupled_branches=coupled_doc,
        checks=tuple(checks), not_assessed=not_assessed)
    mission.report = mission_report(mission)
    mission.full_mission = mission.report["inputs"]["full_mission"]
    mission.report_path = _write_report(mission)
    return mission


def _grade(checks: Sequence[HandoffCheck], run) -> tuple[str, str, tuple[str, ...]]:
    """Chain diagnosis: the earliest failing link names the run (the report
    lists every failure either way). Order = chain order, so a run that never
    flew is never mislabelled as an out-of-tolerance parking orbit."""
    failed = [c for c in checks if not c.passed]

    def code(check_id: str) -> str:
        return next((c.reason_code for c in failed if c.id == check_id), "failed")

    def any_failed(*prefixes) -> bool:
        return any(c.id.startswith(p) for c in failed for p in prefixes)

    if run is None:
        return (STATUS_ASCENT_NOT_CONVERGED, "the ascent produced no flown run",
                tuple(sorted({c.reason_code or "failed" for c in failed})))
    if any_failed("ascent.optimizer_converged"):
        return (STATUS_ASCENT_NOT_CONVERGED, "the ascent optimizer reported no converged flight",
                ("optimizer_not_converged",))
    if any_failed("station."):
        return (STATUS_STATION_MISSING,
                "station chain incomplete or out of order: "
                + ", ".join(c.id.split(".", 1)[1] for c in failed if c.id.startswith("station.")),
                tuple(sorted({c.reason_code or "failed" for c in failed
                              if c.id.startswith("station.")})))
    if any_failed("handoff.", "parking."):
        return (STATUS_HANDOFF_FAILED,
                "handoff continuity/mass gate failed: " + ", ".join(c.id for c in failed
                                                                    if c.id.startswith(("handoff.", "parking."))),
                tuple(sorted({c.reason_code or "failed" for c in failed
                              if c.id.startswith(("handoff.", "parking."))})))
    if any_failed("mass."):
        return (STATUS_MASS_BOOKKEEPING_FAILED,
                "mass bookkeeping failed: " + ", ".join(c.id for c in failed if c.id.startswith("mass.")),
                ("mass_bookkeeping",))
    if any_failed("recovery."):
        return (STATUS_RECOVERY_FAILED,
                "first-stage recovery failed: "
                + ", ".join(f"{c.id.split('.', 1)[1]} ({c.reason_code})" for c in failed
                            if c.id.startswith("recovery.")),
                tuple(sorted({c.reason_code or "failed" for c in failed
                              if c.id.startswith("recovery.")})))
    if any_failed("expected."):
        return (STATUS_PARKING_ORBIT_OUT_OF_TOLERANCE,
                "achieved parking orbit outside the expected results: "
                + ", ".join(c.id for c in failed if c.id.startswith("expected.")),
                ("parking_orbit_out_of_tolerance",))
    if any_failed("propagation."):
        return (STATUS_PROPAGATION_FAILED,
                "orbit propagation failed: "
                + ", ".join(c.reason_code or c.id for c in failed if c.id.startswith("propagation.")),
                ("propagation_failed",))
    if any_failed("ascent."):
        return (STATUS_HANDOFF_FAILED, "ascent run verification failed: "
                + ", ".join(c.id for c in failed if c.id.startswith("ascent.")),
                ("ascent_verification_failed",))
    return (STATUS_OK, "station chain complete, every handoff gate green, parking orbit "
            "inside the expected results", ())


def _check_category(name: str) -> str:
    """V-04 (task #243) evidence category of one mission-chain check. The
    chain's own integrity -- stations present and ordered, handoff
    continuity, mass bookkeeping, the parking-orbit transform of the same
    state, the propagation link running forward and conserving mass, the
    optimizer converging -- is numerical consistency; grading against the
    target orbit, the Gate-B verdict, tudatpy's re-fly and the M-05 report
    (a residual against an analytic coast) is model agreement."""
    from orbitopt.verify.report import CHECK_MODEL, CHECK_NUMERICAL

    if name.startswith(("station.", "handoff.", "mass.", "parking.")):
        return CHECK_NUMERICAL
    if name.startswith("recovery.handoff"):
        return CHECK_NUMERICAL
    if name.startswith("recovery.") or name == "recovery.landed":
        return CHECK_MODEL
    if name in ("ascent.optimizer_converged", "propagation.ran",
                "propagation.epoch_forward", "propagation.mass_conservation"):
        return CHECK_NUMERICAL
    if name.startswith("expected.") or name in (
            "ascent.flight_verified", "ascent.insertion_verified",
            "ascent.physics_verified", "propagation.report_passed"):
        return CHECK_MODEL
    raise ValueError(f"mission check {name!r} has no verification category")


def _check_evidence(name: str) -> str:
    """Task #263 (E4-03) evidence type of one mission-chain check. Chain
    integrity (stations, handoffs, mass bookkeeping, parking-orbit transform,
    propagation running forward, optimizer converged) is bare numerical
    consistency. ``ascent.physics_verified`` is the *independent Gate-B
    re-flight* verdict (task #261: only set when an independent propagator
    replays the flown run, never by the producing backend) -- the sole
    validation-grade evidence in this report. ``propagation.report_passed``
    grades the M-05 propagation against an *analytic* coast: an
    ``analytical`` reference, not real-world data. Every other ``model_agreement``
    check (``expected.*`` vs the design orbit; ``ascent.flight_verified`` and
    its insertion-only half ``ascent.insertion_verified`` = the Gate-B
    mission-constraints verdict; the ``recovery.*`` touchdown/reserve checks vs
    the flight target's own limits) is graded against the **design input**
    itself -- internal consistency, not independent validation. Recovery
    handoff continuity is chain integrity, i.e. numerical consistency."""
    from orbitopt.verify.report import (
        EVIDENCE_ANALYTICAL,
        EVIDENCE_DESIGN_INPUT,
        EVIDENCE_INDEPENDENT_MODEL,
        EVIDENCE_NUMERICAL,
    )

    if name.startswith(("station.", "handoff.", "mass.", "parking.")):
        return EVIDENCE_NUMERICAL
    if name.startswith("recovery.handoff"):
        return EVIDENCE_NUMERICAL
    if name in ("ascent.optimizer_converged", "propagation.ran",
                "propagation.epoch_forward", "propagation.mass_conservation"):
        return EVIDENCE_NUMERICAL
    if name == "ascent.physics_verified":
        return EVIDENCE_INDEPENDENT_MODEL
    if name == "propagation.report_passed":
        return EVIDENCE_ANALYTICAL
    if name.startswith("expected.") or name.startswith("recovery.") or name in (
            "ascent.flight_verified", "ascent.insertion_verified"):
        return EVIDENCE_DESIGN_INPUT
    raise ValueError(f"mission check {name!r} has no evidence type")


def mission_report(mission: TwoStageLeoMission) -> dict:
    """The unified task report of the reference mission (kind
    ``two-stage-leo-mission``, schema ``verification-report-1.0``): one status
    block over every graded check of the chain, the expected-versus-achieved
    table under ``targets``, and the chain's evidence (stations, staging and
    parking handoff documents, propagation summary, not-assessed list, child
    report paths + digests) under ``inputs``. Not written here -- see
    :func:`run_two_stage_leo`."""
    from orbitopt.verify.report import provenance, report_digest, status_from_checks, to_jsonable

    checks = {c.id: bool(c.passed) for c in mission.checks}
    status = status_from_checks(checks, {name: _check_category(name) for name in checks},
                                {name: _check_evidence(name) for name in checks})
    child_reports = []
    for kind, block in (("launch-pipeline", mission.launch),
                        ("orbit-propagation", mission.propagation)):
        if not block:
            continue
        entry = {"kind": kind}
        for key in ("report_file", "digest", "result"):
            if block.get(key):
                entry[key.replace("report_file", "path")] = block[key]
        if kind == "launch-pipeline" and block.get("report_file"):
            entry["path"] = block["report_file"]
            entry["status"] = block.get("status")
        child_reports.append(entry)
    achieved = {k: mission.parking_elements.get(k) for k in ELEMENT_KEYS}
    deviations = {}
    for key, got in achieved.items():
        want = mission.expected_elements.get(key)
        tol = mission.expected_tolerances.get(key)
        if got is None or want is None or tol is None or want is None:
            continue
        deviations[key] = {"deviation": abs(float(got) - float(want)), "tolerance": float(tol)}
    chain = " -> ".join(mission.scenario.expected.events)
    if mission.recovery is not None:
        chain = (chain.replace("separation", "separation -> booster recovery", 1)
                 if "separation" in chain else chain + " -> booster recovery")
    if mission.branches is not None:
        chain += " (branched after separation: upper stage + first stage)"
    report = {
        "schemaVersion": "1.0",
        "kind": MISSION_REPORT_KIND,
        "label": f"{mission.scenario.mission_id}: " + chain
                 + " -> parking orbit -> orbit propagation",
        "status": status,
        "inputs": {
            "scenario": mission.scenario.to_dict(),
            "expected_basis": mission.expected_basis,
            "warnings": scenario_warnings(mission.scenario),
            "ascent": {"mode": mission.scenario.ascent_mode, "run_id": mission.run_id,
                       "run_dir": str(mission.ascent_run_dir) if mission.ascent_run_dir else None,
                       "launch": mission.launch},
            "stations": mission.stations,
            "handoffs": {"staging": mission.staging, "separation_fork": mission.branches,
                         "parking_orbit": mission.parking,
                         "coupled_branches": mission.coupled_branches},
            "propagation": mission.propagation,
            "recovery": mission.recovery,
            "full_mission": full_mission_gate(
                mission, independent_validation=status["independent_validation"]["result"]),
            "not_assessed": mission.not_assessed,
            "child_reports": child_reports,
        },
        "propagations": [],
        "residuals": [],
        "targets": {
            "expected_parking_orbit": dict(mission.expected_elements),
            "achieved_parking_orbit": achieved,
            "deviations": deviations,
            "converged": mission.status == STATUS_OK,
        },
        "provenance": provenance(),
    }
    report = to_jsonable(report)
    report["digest"] = report_digest(report)
    return report


def _write_report(mission: TwoStageLeoMission) -> Path:
    from orbitopt.verify.report import write_report

    return write_report(mission.report, mission.out_dir / _REPORT_FILE)


def build_from_config(config: dict, recorder=None) -> dict:
    """Pipeline builder for kind ``two-stage-leo``: run the frozen scenario in
    the mission's output directory, record the chain on the ``search`` and
    ``verify`` stages, export the ascent run's scene (M-06) with the unified
    mission report attached -- so ``run_pipeline``, the Verification Gate and
    ``orbitopt run``'s outputs work exactly as for the other kinds, and an
    honest red verdict flows through as a degraded export, never a silent
    pass."""
    from orbitopt.flightlab.contract import read_run
    from orbitopt.flightlab.scene_export import run_to_scene
    from orbitopt.pipeline import REFINE, SEARCH, SELECT, VERIFY, StageRecorder, default_output_path
    from orbitopt.verify.report import attach_to_scene

    mission_doc = config["mission"]
    out_dir = config.get("out_dir") or default_output_path(
        (config.get("output") or {}).get("scene_file") or f"{mission_doc['id']}.scene.json",
        ".m07")
    recorder = StageRecorder() if recorder is None else recorder
    scenario = scenario_from_config(config)

    with recorder.stage(SEARCH) as st:
        mission = run_two_stage_leo(scenario, out_dir)
        st.candidates = [mission.to_dict()]
        st.physical_model = {"name": "OpenTsiolkovsky 3-DOF two-stage ascent -> tudatpy "
                                     "high-fidelity parking-orbit propagation",
                             "fidelity": "high-fidelity"}
        st.convergence.update({"converged": mission.status == STATUS_OK,
                               "chain_status": mission.status, "run_id": mission.run_id})
        st.constraint_violations.update({c.id: 1.0 for c in mission.checks if not c.passed})
        st.numerical_error.update({"handoff_position_discontinuity_m": max(
            (c.value for c in mission.checks if c.id.endswith("position_continuity")
             and c.value is not None), default=0.0)})
        st.notes.extend(scenario_warnings(scenario))
        st.notes.append(f"mission report: {mission.report_path}")

    with recorder.stage(VERIFY) as st:
        st.physical_model = {"name": "M-07 chain gate (station chain, M-01 handoffs, "
                                     "M-02 element agreement, M-05 propagation report)",
                             "fidelity": "high-fidelity"}
        physical = dict(mission.full_mission.get("physical") or {})
        st.convergence.update({"converged": mission.status == STATUS_OK,
                               "mission_status": mission.status,
                               "full_mission_verdict": mission.full_mission.get("verdict"),
                               "full_mission_coverage": (mission.full_mission.get("coverage")
                                                         or {}).get("summary"),
                               "independently_verified": bool((mission.full_mission.get(
                                   "qualification") or {}).get("independently_verified")),
                               "physical_mission_verdict": physical.get("verdict"),
                               "checks_passed": sum(1 for c in mission.checks if c.passed),
                               "checks_total": len(mission.checks)})
        st.notes.append("full-mission gate: " + mission.full_mission.get("verdict", "not_checked")
                        + " (" + ", ".join(f"{p['phase']}={p['verdict']}"
                                           for p in mission.full_mission.get("phases", []))
                        + "; coverage " + str((mission.full_mission.get("coverage") or {})
                                             .get("summary", "?"))
                        + "; independently_verified="
                        + str(bool((mission.full_mission.get("qualification") or {})
                                   .get("independently_verified"))) + ")")
        st.notes.append("full physical mission gate (E5-04): "
                        + physical.get("verdict", "not_checked")
                        + " (" + ", ".join(f"{c['check']}={c['verdict']}"
                                           for c in physical.get("checks", [])) + ")")
        st.notes.extend(f"not assessed: {n['id']} ({n['reason']})" for n in mission.not_assessed)

    recorder.skip(SELECT, "one frozen reference mission -- nothing to select")
    recorder.skip(REFINE, "the reference scenario is frozen -- no refinement walk")

    if mission.ascent_run_dir is None:
        raise RuntimeError(f"{mission.status}: {mission.message} -- no scene to export")
    # With a recovery stage the scene carries the flown booster, not the
    # ballistic debris: export the combined run the recovery stage wrote.
    scene_run_dir = mission.recovery_run_dir or mission.ascent_run_dir
    scene = run_to_scene(read_run(scene_run_dir),
                         earth_texture=scenario.earth_texture,
                         coast_orbits=scenario.coast_orbits,
                         mission_scenario=scenario)
    scene["id"] = mission_doc["id"]
    scene["title"] = mission_doc["title"]
    attach_to_scene(scene, mission.report)
    return scene


def scenario_from_config(config: Mapping[str, Any]) -> TwoStageLeoScenario:
    """Read a full ``two-stage-leo`` mission config document (schema
    ``mission-two-stage-leo-1.0.json``, i.e. what ``orbitopt run`` and
    :func:`load_mission_config` load) and return its scenario. Validated
    against the schema first, so a config that would be rejected by
    ``orbitopt run`` is rejected here too."""
    from orbitopt.mission_config import validate_mission_config

    validate_mission_config(dict(config))
    scenario = dict(config.get("scenario") or {})
    mission = dict(config.get("mission") or {})
    # The mission block is the config's identity; the scenario's own mission
    # block must agree with it (never silently two ids for one mission).
    for key in ("id", "title"):
        if scenario.get("mission", {}).get(key) is None and mission.get(key):
            scenario.setdefault("mission", {})[key] = mission[key]
    return TwoStageLeoScenario.from_dict(scenario)
