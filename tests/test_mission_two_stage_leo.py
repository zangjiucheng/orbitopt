"""M-07 (task #225): the frozen two-stage LEO reference mission.

The fast tests need no compute stack: they grade the **recorded** Gate-B
two-stage ascent (``tests/data/flightlab/reference/gateb-leo-two-stage``, the
same fixture the FL-003 reference tests use) in ``run_dir`` mode, and run the
propagation link against the fake tudatpy of ``test_mission_propagation``.
That recorded flight is deliberately off-target (its apogee is 139 km short --
the single-burn cutoff the vehicle file documents), so the reference mission
must come back *red with a diagnosis*, never green: the station chain and
every handoff gate are green, the expected parking orbit is not. The one
``slow`` test is the acceptance run itself -- optimize the ascent now with the
real binary, propagate with real tudatpy, and require a green chain; that is
also what the Flight Lab Gate B CI job runs.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import math
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from orbitopt.flightlab.contract import read_run
from orbitopt.flightlab.coast import coast_arc
from orbitopt.mission.contract import HandoffCheck
from orbitopt.mission.propagation import MU_EARTH
from orbitopt.mission.two_stage_leo import (
    MISSION_KIND, MISSION_REPORT_KIND, REQUIRED_EVENT_CHAIN, STATUS_ASCENT_NOT_CONVERGED,
    STATUS_HANDOFF_FAILED, STATUS_OK, STATUS_PARKING_ORBIT_OUT_OF_TOLERANCE,
    STATUS_PROPAGATION_FAILED, STATUS_STATION_MISSING,
    TwoStageLeoScenario, event_chain_checks, reference_scenario, reference_scenario_path,
    resolve_expected, run_two_stage_leo, scenario_from_config, scenario_warnings,
    staging_handoffs)
from orbitopt.mission_config import known_kinds, load_mission_config
from orbitopt.verify.report import read_report, validate_report

REPO = Path(__file__).resolve().parents[1]
GATEB = REPO / "tests" / "data" / "flightlab" / "reference" / "gateb-leo-two-stage"
PYGENO = importlib.util.find_spec("pygmo") is not None


def _has_stack() -> bool:
    from orbitopt.flightlab.adapters.opentsiolkovsky import find_binary

    return (PYGENO and importlib.util.find_spec("tudat") is not None
            and find_binary() is not None)


def _replay(tmp_path, run_dir=GATEB, **changes) -> TwoStageLeoScenario:
    """The packaged reference scenario, replaying a flown run directory
    instead of optimizing (the machine-without-a-compute-stack mode)."""
    scenario = reference_scenario()
    return dataclasses.replace(
        scenario, ascent_mode="run_dir", ascent_run_dir=str(run_dir), ascent={}, **changes)


# --- fake tudatpy (same shape as tests/test_mission_propagation.py) -----------


def kepler_history(state6, et0, duration, n=1201):
    times, positions, velocities = coast_arc(state6[:3], state6[3:], duration,
                                             samples=n, mu=MU_EARTH)
    epochs = np.asarray(times) + et0
    states = np.column_stack([np.asarray(positions), np.asarray(velocities)])
    return epochs, states


@pytest.fixture
def fake_tudat(monkeypatch):
    """Install fake ``tudatpy`` modules and a Keplerian propagator so the
    M-05 link of the chain runs without the compute stack."""
    modules = {}

    def mod(name, **attrs):
        m = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(m, key, value)
        modules[name] = m
        return m

    identity = np.eye(3)
    spice = mod("tudatpy.interface.spice",
                load_standard_kernels=lambda: None,
                compute_rotation_matrix_between_frames=MagicMock(return_value=identity))
    mod("tudatpy.interface", spice=spice)
    env = MagicMock(name="environment_setup")
    prop = MagicMock(name="propagation_setup")
    mod("tudatpy.numerical_simulation", environment_setup=env, propagation_setup=prop)
    mod("tudatpy", constants=MagicMock(),
        numerical_simulation=modules["tudatpy.numerical_simulation"],
        interface=modules["tudatpy.interface"])
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    import orbitopt.verify.tudat_propagate as tp

    def fake_propagate(bodies, accel, central, state6, et0, duration, step,
                       integrator="dop853", rtol=1e-11, atol=1e-6):
        return kepler_history(np.asarray(state6, dtype=float), float(et0), float(duration))

    def fake_override(body_settings, central_body, gravity):
        return {"gravitational_parameter": gravity["gravitational_parameter"],
                "reference_radius": gravity["reference_radius"], "j2": gravity.get("j2", 0.0),
                "normalized_c20": -gravity.get("j2", 0.0) / math.sqrt(5.0),
                "associated_reference_frame": f"IAU_{central_body}"}

    monkeypatch.setattr(tp, "_SPICE_LOADED", True)
    monkeypatch.setattr(tp, "_propagate_single_coast", fake_propagate)
    monkeypatch.setattr(tp, "_override_central_gravity", fake_override)
    yield {"env": env, "prop": prop}


# --- the packaged reference scenario -----------------------------------------

def test_kind_registered_and_reference_scenario_packaged():
    assert MISSION_KIND in known_kinds()
    config = load_mission_config(reference_scenario_path())      # schema-validates
    scenario = scenario_from_config(config)
    assert scenario == reference_scenario()
    assert scenario.expected.events == REQUIRED_EVENT_CHAIN
    assert scenario.expected.parking_orbit["semi_major_axis_m"] == 6778137.0
    assert scenario.expected.tolerances["semi_major_axis_m"] == 20000.0  # the Gate-B band


def test_expected_orbit_is_derived_from_the_mission_or_never_invented():
    scenario = reference_scenario()
    expected, tolerances, basis = resolve_expected(scenario)
    assert expected["semi_major_axis_m"] == pytest.approx(6378137.0 + 400000.0)
    assert expected["raan_deg"] is None
    assert tolerances["inclination_deg"] == 0.2
    assert "semi_major_axis_m" in basis["declared"]  # declared in the YAML, and recorded as such

    free = dataclasses.replace(reference_scenario(),
                               orbit="500x500@53.0",
                               expected=scenario.expected.__class__(
                                   parking_orbit={}, tolerances={}))
    derived, tol, basis = resolve_expected(free)
    assert derived["semi_major_axis_m"] == pytest.approx(6378137.0 + 500000.0)
    assert derived["inclination_deg"] == 53.0
    assert tol["semi_major_axis_m"] == 20000.0       # the documented default, in the basis
    assert basis["derived_from"]["semi_major_axis_m"].startswith("launch.orbit spec")


def test_scenario_round_trips_through_its_own_document():
    scenario = reference_scenario()
    assert TwoStageLeoScenario.from_dict(scenario.to_dict()) == scenario


@pytest.mark.parametrize("bad, match", [
    ({"mission": {"id": "x", "title": "y"}, "launch": {"site": "kourou", "vehicle": "v",
                                                       "orbit": "400x400@51.6",
                                                       "date_utc": "2026-11-10",
                                                       "colour": "blue"}}, "launch: unknown key"),
    ({"mission": {"id": "x", "title": "y"}, "launch": {"site": "kourou", "vehicle": "v",
                                                       "orbit": "400x400@51.6",
                                                       "date_utc": "2026-11-10"},
       "ascent": {"mode": "teleport"}}, "ascent.mode"),
    ({"mission": {"id": "x", "title": "y"}, "launch": {"site": "kourou", "vehicle": "v",
                                                       "orbit": "400x400@51.6",
                                                       "date_utc": "2026-11-10"},
       "ascent": {"mode": "run_dir"}}, "needs ascent.run_dir"),
])
def test_scenario_loader_rejects_what_the_schema_misses(bad, match):
    with pytest.raises(ValueError, match=match):
        TwoStageLeoScenario.from_dict(bad)


def test_warnings_tell_a_caller_what_it_should_know():
    notes = " | ".join(scenario_warnings(reference_scenario()))
    assert "RAAN unconstrained" in notes
    replay = _replay(Path("."))
    assert "replayed" in " | ".join(scenario_warnings(replay))


# --- the chain, replayed over the recorded Gate-B ascent ----------------------

def test_replayed_reference_flight_is_red_where_the_flight_is_off_target(tmp_path):
    mission = run_two_stage_leo(_replay(tmp_path), tmp_path / "m07", propagate=False)

    # every station of the DoD chain, in order
    assert [s["kind"] for s in mission.stations] == list(REQUIRED_EVENT_CHAIN)
    assert all(s["passed"] for s in mission.stations)
    # every graded handoff of the chain is green (the recorded flight stages cleanly)
    graded = [d for d in mission.staging if d["graded"]]
    assert [d["child_body"] for d in graded] == ["stage2"]
    assert all(d["passed"] for d in graded)
    assert mission.staging[0]["bridge"]["method"] == "j2_coast"  # epochs compared like-for-like
    assert mission.parking["passed"] is True
    assert mission.status == STATUS_PARKING_ORBIT_OUT_OF_TOLERANCE
    failed = {c.id for c in mission.checks if not c.passed}
    assert {"expected.semi_major_axis", "expected.apoapsis"} <= failed
    assert "expected.inclination" not in failed          # 51.76 vs 51.6: inside 0.2 deg
    # the off-target insertion is what the flight's own Gate-B verdict said
    assert "ascent.flight_verified" in failed
    # honest about what it did not assess, and a written, digested report
    assert {n["id"] for n in mission.not_assessed} >= {"expected.raan", "propagation"}
    report = read_report(mission.report_path)
    validate_report(report)
    assert report["kind"] == MISSION_REPORT_KIND
    assert report["status"]["result"] == "failed"        # never a defaulted pass
    assert report["status"]["checks"]["parking.mass_conservation"] is True
    assert report["targets"]["achieved_parking_orbit"]["apoapsis_altitude_m"] < 550000.0


def test_debris_and_fairing_boundaries_are_recorded_not_graded(tmp_path):
    docs = {d["child_body"]: d for d in staging_handoffs(
        read_run(GATEB), reference_scenario().handoff,
        reference_scenario().staging_velocity_m_s)}
    assert docs["stage2"]["graded"] is True
    for off_lineage in ("stage1_debris", "stage2_fairing"):
        assert docs[off_lineage]["graded"] is False
        assert "lineage" in docs[off_lineage]["reason"]


def test_event_chain_checks_report_a_broken_chain():
    events = [{"kind": "liftoff", "t_s": 0.0, "body_id": "stage1"},
              {"kind": "meco", "t_s": 100.0, "body_id": "stage1"},
              {"kind": "separation", "t_s": 110.0, "body_id": "stage2"}]
    checks = {c.id: c for c in event_chain_checks(events)}
    assert checks["station.seco"].passed is False and checks["station.seco"].value is None
    assert checks["station.event_chain_order"].passed is True   # order of what IS there

    backwards = [{"kind": "seco", "t_s": 10.0, "body_id": "s"},
                 {"kind": "liftoff", "t_s": 20.0, "body_id": "s"},
                 {"kind": "meco", "t_s": 30.0, "body_id": "s"},
                 {"kind": "separation", "t_s": 40.0, "body_id": "s"}]
    order = {c.id: c for c in event_chain_checks(backwards)}["station.event_chain_order"]
    assert order.passed is False and order.reason_code == "chain_order"


def test_status_precedence_names_the_earliest_failing_link():
    from orbitopt.mission.two_stage_leo import _grade

    run = object()
    ok = HandoffCheck(id="station.liftoff", passed=True, value=0.0)
    cases = [
        (None, [ok], STATUS_ASCENT_NOT_CONVERGED),
        (run, [HandoffCheck(id="ascent.optimizer_converged", passed=False,
                            reason_code="optimizer_not_converged")],
         STATUS_ASCENT_NOT_CONVERGED),
        (run, [ok, HandoffCheck(id="station.seco", passed=False, reason_code="event_missing")],
         STATUS_STATION_MISSING),
        (run, [ok, HandoffCheck(id="parking.mass_conservation", passed=False,
                                reason_code="mass_loss_without_staging")],
         STATUS_HANDOFF_FAILED),
        (run, [ok, HandoffCheck(id="expected.apoapsis", passed=False,
                                reason_code="out_of_tolerance")],
         STATUS_PARKING_ORBIT_OUT_OF_TOLERANCE),
        (run, [ok, HandoffCheck(id="propagation.ran", passed=False,
                                reason_code="propagation_error")],
         STATUS_PROPAGATION_FAILED),
        (run, [ok], STATUS_OK),
    ]
    for given_run, checks, want in cases:
        status, message, _ = _grade(checks, given_run)
        assert status == want, (status, want)
        assert message


@pytest.mark.skipif(PYGENO, reason="with pygmo installed the optimize mode would fly, not raise")
def test_missing_ascent_stack_raises_out_of_the_mission_like_it_does_out_of_m04(tmp_path):
    # An unrunnable optimize mode is an input error (the M-04 contract), not a
    # mission verdict -- the scenario says the ascent is flown, and it cannot be.
    scenario = reference_scenario()
    with pytest.raises(RuntimeError, match="pygmo"):
        run_two_stage_leo(scenario, tmp_path / "m07")


def test_missing_propagation_stack_is_recorded_as_a_failure_of_the_link(tmp_path, monkeypatch):
    import orbitopt.mission.propagation as propagation

    class ExplodingSegment:
        def __init__(self, *a, **kw):
            pass

        def run(self, state):
            raise ImportError("No module named 'tudat'")

    monkeypatch.setattr(propagation, "OrbitPropagationSegment", ExplodingSegment)
    mission = run_two_stage_leo(_replay(tmp_path), tmp_path / "m07")
    assert mission.propagation["status"] == "failed"
    assert {c.id: c.reason_code for c in mission.checks
            if not c.passed and c.id.startswith("propagation.")} == {
                "propagation.ran": "propagation_unavailable"}


def test_propagation_link_gates_on_the_m05_report(tmp_path, fake_tudat):
    mission = run_two_stage_leo(_replay(tmp_path), tmp_path / "m07")
    assert mission.propagation["status"] == "ran"
    assert mission.propagation["result"] == "passed"
    prop = {c.id: c for c in mission.checks if c.id.startswith("propagation.")}
    assert prop["propagation.report_passed"].passed
    assert prop["propagation.epoch_forward"].value > 0.0
    assert prop["propagation.mass_conservation"].value == pytest.approx(0.0, abs=1e-6)
    written = Path(mission.propagation["report_file"])
    assert written.is_file() and written.parent == mission.out_dir
    child = {c["kind"]: c for c in mission.report["inputs"]["child_reports"]}
    assert child["orbit-propagation"]["result"] == "passed"
    # the ascent was still off-target: the propagation passing does not turn
    # the mission green
    assert mission.status == STATUS_PARKING_ORBIT_OUT_OF_TOLERANCE


def test_unwitnessed_propagation_cannot_be_expected_to_pass(tmp_path, monkeypatch):
    import orbitopt.mission.propagation as propagation

    class ExplodingSegment:
        def __init__(self, *a, **kw):
            pass

        def run(self, state):
            raise ImportError("nope")

    monkeypatch.setattr(propagation, "OrbitPropagationSegment", ExplodingSegment)
    scenario = dataclasses.replace(
        _replay(tmp_path),
        expected=reference_scenario().expected.__class__(
            events=REQUIRED_EVENT_CHAIN, propagation_result="not_checked"))
    mission = run_two_stage_leo(scenario, tmp_path / "m07")
    assert not [c for c in mission.checks if c.id.startswith("propagation.")]
    assert mission.status == STATUS_PARKING_ORBIT_OUT_OF_TOLERANCE   # not PROPAGATION_FAILED


# --- the pipeline adapter ----------------------------------------------------

def _config(scenario: TwoStageLeoScenario, out_dir: Path) -> dict:
    return {"schemaVersion": "1.0",
            "mission": {"id": "m07-two-stage-leo", "title": scenario.title,
                        "kind": MISSION_KIND},
            "scenario": scenario.to_dict(),
            "out_dir": str(out_dir)}


def test_pipeline_builder_exports_the_scene_with_the_unified_report(tmp_path, fake_tudat):
    from orbitopt.pipeline import run_pipeline
    from orbitopt.verify.report import SCENE_REPORTS_KEY

    config = _config(_replay(tmp_path), tmp_path / "m07")
    result = run_pipeline(config)
    assert [s.stage for s in result.stages if s.status == "completed"][:3] == [
        "mission_config", "search", "verify"]
    assert result.stages[2].convergence["chain_status"] == STATUS_PARKING_ORBIT_OUT_OF_TOLERANCE
    assert result.success is False                      # an honest red is not swallowed
    reports = result.scene.pop(SCENE_REPORTS_KEY, None) or result.verification_reports
    kinds = [r["kind"] for r in reports]
    assert MISSION_REPORT_KIND in kinds
    assert result.scene["id"] == "m07-two-stage-leo"
    assert any(p["id"] == "Separation \u2014 stage1 \u2192 stage2" for p in
               result.scene.get("mission", {}).get("phases", [])) or "handoffs" in result.scene


def test_pipeline_config_validation_happens_before_anything_flies(tmp_path):
    from orbitopt.pipeline import run_pipeline

    config = _config(_replay(tmp_path), tmp_path / "m07")
    config["scenario"]["launch"]["orbit"] = "400x400"     # not an M-04 orbit spec
    with pytest.raises(Exception):
        run_pipeline(config)


# --- the acceptance run (compute stack; Flight Lab Gate B CI) ----------------

@pytest.mark.slow
@pytest.mark.skipif(not _has_stack(),
                    reason="needs the OpenTsiolkovsky binary + pygmo + tudatpy")
def test_reference_scenario_end_to_end_is_green(tmp_path):
    """The task #225 acceptance run: optimize the ascent now, propagate the
    parking orbit with real tudatpy, and the whole chain is green -- one
    report, every handoff gate passed."""
    mission = run_two_stage_leo(reference_scenario(), tmp_path / "m07")
    assert mission.status == STATUS_OK, mission.message
    assert mission.launch["status"] == "OK"
    assert [s["kind"] for s in mission.stations] == list(REQUIRED_EVENT_CHAIN)
    assert all(d["passed"] for d in mission.staging if d["graded"])
    assert mission.parking["passed"] and mission.propagation["result"] == "passed"
    assert all(c.passed for c in mission.checks)
    report = read_report(mission.report_path)
    validate_report(report)
    assert report["status"]["result"] == "passed"
    assert report["targets"]["converged"] is True
