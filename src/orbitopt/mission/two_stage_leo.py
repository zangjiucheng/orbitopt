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
  (M-01 gates between a spent stage's last sample and its child's first), the
  SECO-to-parking-orbit handoff (M-02), the vehicle mass bookkeeping, and the
  M-05 propagation of the parking orbit.
* :func:`mission_report` -- the unified task report (kind
  ``two-stage-leo-mission``, schema ``verification-report-1.0``): every
  station, every handoff check, the expected-versus-actual table and the child
  reports (launch pipeline, orbit propagation) under ``inputs``.
* :func:`build_from_config` -- the ``orbitopt run`` adapter for mission kind
  ``two-stage-leo``, so the same scenario runs through the unified pipeline,
  its Verification Gate and its scene export.

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
from dataclasses import dataclass, field
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
    "STATUS_STATION_MISSING",
    "ExpectedResults",
    "PropagationPlan",
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
STATUS_PROPAGATION_FAILED = "PROPAGATION_FAILED"
STATUS_NOT_CHECKED = "NOT_CHECKED"
STATUSES = (STATUS_OK, STATUS_ASCENT_NOT_CONVERGED, STATUS_STATION_MISSING,
            STATUS_HANDOFF_FAILED, STATUS_PARKING_ORBIT_OUT_OF_TOLERANCE,
            STATUS_MASS_BOOKKEEPING_FAILED, STATUS_PROPAGATION_FAILED, STATUS_NOT_CHECKED)

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
    gate across a staging boundary (before separation-impulse headroom)."""

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
        object.__setattr__(self, "ascent", dict(self.ascent or {}))

    # -- (de)serialisation --------------------------------------------------

    def to_dict(self) -> dict:
        """The canonical config document -- what :meth:`from_dict` reads back
        and what a ``two-stage-leo`` mission config's ``scenario`` carries."""
        ascent: dict[str, Any] = {"mode": self.ascent_mode, **dict(self.ascent)}
        if self.ascent_run_dir:
            ascent["run_dir"] = str(self.ascent_run_dir)
        if self.ascent_target_file:
            ascent["target_file"] = str(self.ascent_target_file)
        return {
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
                     "a passing M-05 report is not required for the mission to pass")
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
    this lineage only -- a spent-stage debris track or a jettisoned fairing
    diverges on purpose, and grading its divergence as a continuity failure
    would be grading the wrong physics (they are recorded, not graded)."""
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


def _bridge_across_gap(before: HandoffState, after: HandoffState) -> dict | None:
    """Coast ``before`` to ``after``'s epoch when the two samples are not
    simultaneous: returns ``{state, gap_s, method}``, or ``None`` when no
    bridge is needed (same epoch) or none is possible (suborbital parent --
    then the caller gates the raw pair and records why)."""
    gap = before.epoch.delta_seconds(after.epoch)
    if gap < 1e-9:
        return None
    from orbitopt.flightlab.coast import coast_arc

    try:
        _, positions, velocities = coast_arc(before.r_m, before.v_m_s, gap, samples=2)
    except ValueError as exc:
        return {"state": before, "gap_s": gap,
                "method": f"unavailable: two-body bridge impossible ({exc})"}
    return {"state": HandoffState(
                epoch=after.epoch, frame=before.frame,
                r_m=tuple(float(x) for x in positions[-1]),
                v_m_s=tuple(float(x) for x in velocities[-1]),
                mass_kg=before.mass_kg, stage_config=before.stage_config,
                metadata={**before.metadata, "staging_bridge": "two_body_coast",
                          "bridge_gap_s": gap}),
            "gap_s": gap, "method": "two_body_coast"}


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
    the gap by an analytic two-body coast (:func:`orbitopt.flightlab.coast.
    coast_arc`, the same coast the propagation segment compares against) before
    the gate runs, and the bridge is recorded on the handoff document
    (``bridge``). A suborbital parent state cannot be coasted; then the gate
    runs on the raw pair and the document says why (``method:
    unavailable:``-reason) -- the failure is reported, never hidden.

    Only boundaries on the :func:`_ascent_lineage` (the insertion body's
    ancestor chain) are graded. A boundary off that lineage -- stage-1 debris,
    a fairing -- is still recorded (``graded: false``, with the reason),
    because the run did stage there; it is simply not the state the next
    segment inherits, so its divergence is physics, not a lost handoff.
    Returns one document per boundary, mission order first; empty for a
    single-body run."""
    boundary_tol = HandoffTolerances(
        position_m=tolerances.position_m, velocity_m_s=float(staging_velocity_m_s),
        epoch_s=tolerances.epoch_s,
        max_epoch_gap_s=tolerances.max_epoch_gap_s, mass_kg=tolerances.mass_kg)
    lineage = _ascent_lineage(run)
    graded_pairs = set(zip(lineage[1:], lineage[:-1]))  # (parent, child) up the chain
    docs: list[dict] = []
    for track in run.tracks:
        if track.parent_body_id is None:
            continue
        parent = next((t for t in run.tracks if t.body_id == track.parent_body_id), None)
        if parent is None:
            continue
        separation = next((float(e.t_s) for e in run.events
                           if e.kind == "separation"
                           and e.body_id in (track.body_id, track.parent_body_id)), None)
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
                                {k: bridge[k] for k in ("gap_s", "method")}),
                     "before": _state_doc(before), "after": _state_doc(after)}
        if not graded:
            doc["checks"] = []
            doc["passed"] = None
            doc["reason"] = ("off the ascent lineage (debris / fairing track): recorded, "
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
    parking: dict | None = None
    parking_elements: dict = field(default_factory=dict)
    expected_elements: dict = field(default_factory=dict)
    expected_tolerances: dict = field(default_factory=dict)
    expected_basis: dict = field(default_factory=dict)
    propagation: dict = field(default_factory=dict)
    checks: tuple[HandoffCheck, ...] = ()
    not_assessed: list = field(default_factory=list)
    report: dict = field(default_factory=dict)
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
            "stations": self.stations, "staging": self.staging, "parking": self.parking,
            "parking_elements": self.parking_elements,
            "expected_elements": self.expected_elements,
            "propagation": self.propagation,
            "checks": _checks_doc(self.checks), "not_assessed": self.not_assessed,
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
        **dict(scenario.ascent))
    if pipeline.ascent_run_dir is None:
        # No flight at all (infeasible window, optimizer abort): there is no
        # chain to grade -- the diagnosis is the pipeline's own status.
        return None, pipeline.to_dict(), pipeline.message
    return read_run(pipeline.ascent_run_dir), pipeline.to_dict(), None


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
    -> mass bookkeeping -> expected-orbit agreement -> orbit propagation. The
    unified report lands at ``<out_dir>/mission.verification.json``; the M-04
    launch-pipeline report (``optimize`` mode) and the M-05 propagation report
    stay next to it and are referenced from it by path and digest."""
    from orbitopt.mission.ascent import ascent_handoff_state, check_ascent_to_parking_orbit

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
    parking: dict | None = None
    parking_elements: dict = {}
    propagation: dict = {"status": "not_run"}

    if run is None:
        checks.append(_check("ascent.flight_verified", False,
                             reason_code=(launch or {}).get("reason_codes", ["no_flight"])[0]
                             if launch else "no_flight"))
        message = f"ascent did not produce a flown run: {ascent_failure}"
    else:
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
        stations=stations, staging=staging, parking=parking,
        parking_elements=parking_elements, expected_elements=expected_elements,
        expected_tolerances=expected_tolerances, expected_basis=basis,
        propagation=propagation, checks=tuple(checks), not_assessed=not_assessed)
    mission.report = mission_report(mission)
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
    status = status_from_checks(checks)
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
    report = {
        "schemaVersion": "1.0",
        "kind": MISSION_REPORT_KIND,
        "label": f"{mission.scenario.mission_id}: " + " -> ".join(mission.scenario.expected.events)
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
            "handoffs": {"staging": mission.staging, "parking_orbit": mission.parking},
            "propagation": mission.propagation,
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
        st.convergence.update({"converged": mission.status == STATUS_OK,
                               "mission_status": mission.status,
                               "checks_passed": sum(1 for c in mission.checks if c.passed),
                               "checks_total": len(mission.checks)})
        st.notes.extend(f"not assessed: {n['id']} ({n['reason']})" for n in mission.not_assessed)

    recorder.skip(SELECT, "one frozen reference mission -- nothing to select")
    recorder.skip(REFINE, "the reference scenario is frozen -- no refinement walk")

    if mission.ascent_run_dir is None:
        raise RuntimeError(f"{mission.status}: {mission.message} -- no scene to export")
    scene = run_to_scene(read_run(mission.ascent_run_dir),
                         earth_texture=scenario.earth_texture,
                         coast_orbits=scenario.coast_orbits)
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
