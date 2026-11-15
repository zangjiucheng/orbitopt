"""E5-01 (task #268): the full mission chain's ascent input is REAL.

E2-04 graded mission integration over a synthetic, two-sample ascent handoff
with a prescribed SECO state (``docs/acceptance-e1-e4.md``, Evidence
boundaries): the numbers the chain graded were constants injected by
``scripts/e204_reference.py``, not a flown trajectory. This module closes
that gap for the ascent leg of the chain:

* the SLOW acceptance run drives the packaged M-07 scenario end to end in
  ``ascent.mode: optimize`` -- the pinned OpenTsiolkovsky v1.0.0 binary flies
  the ascent now (pygmo optimizer, real event log, full time history) -- and
  then grades WHERE the chain's evidence came from: the station times are the
  engine's events, the MECO/SECO states are computed track samples, and the
  raw engine outputs sit next to the run directory. Where tudatpy is absent
  the M-05 leg stays honestly ``not_checked`` (never passed).
* the FAST guards pin the exception: the only preset-SECO code path left is
  the E2-04 fixture, kept as a fast integration test and annotated as such
  (E5-01 exception), and the mission scenario surface cannot carry an
  injected ascent/SECO state at all.

The fast ``test_recorded_ot_flight_is_a_full_time_history`` check applies the
same real-input contract to the committed recorded OpenTsiolkovsky flight
(``tests/data/flightlab/reference/gateb-leo-two-stage``), so a machine
without the compute stack still proves what a real ascent looks like.
"""
from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import pytest

from orbitopt.flightlab import checks as gate_checks
from orbitopt.flightlab.contract import read_run
from orbitopt.mission.two_stage_leo import (
    REQUIRED_EVENT_CHAIN, STATUS_OK, TwoStageLeoScenario, reference_scenario,
    run_two_stage_leo)
from orbitopt.verify.report import read_report, validate_report

REPO = Path(__file__).resolve().parents[1]
GATEB = REPO / "tests" / "data" / "flightlab" / "reference" / "gateb-leo-two-stage"
PYGENO = importlib.util.find_spec("pygmo") is not None

#: The synthetic E2-04 ascent fixture, spelled in pieces so the consumer scan
#: below matches its CONSUMERS, not this guard.
_E204_FIXTURE = "e2-04-frozen-" "mission"
#: Files allowed to consume the frozen synthetic ascent (the E2-04 fast tests
#: and the builder that produced the fixture -- the exception E5-01 grants).
_E204_EXCEPTION_FILES = {
    "scripts/e204_reference.py",
    "tests/test_e204_reference.py",
    "tests/test_mission_phases.py",
}


def _has_stack() -> bool:
    from orbitopt.flightlab.adapters.opentsiolkovsky import find_binary

    return PYGENO and find_binary() is not None


def _has_tudat() -> bool:
    return importlib.util.find_spec("tudat") is not None


def _e5_scenario() -> TwoStageLeoScenario:
    """The packaged M-07 reference scenario, flown now (``optimize``): the
    same frozen mission, with the M-05 propagation leg left ``passed`` where
    tudatpy can grade it and honestly ``not_checked`` where it cannot (the
    chain records the decline, never a pass)."""
    scenario = reference_scenario()
    assert scenario.ascent_mode == "optimize"
    if _has_tudat():
        return scenario
    return dataclasses.replace(
        scenario,
        expected=dataclasses.replace(scenario.expected, propagation_result="not_checked"))


def _track(run, body_id: str):
    return next(t for t in run.tracks if t.body_id == body_id)


# --- fast: the synthetic-SECO exception is explicit and contained ------------


def test_scenario_surface_cannot_carry_a_preset_ascent_state(tmp_path):
    """The mission scenario surface has no knob for a preset SECO/injection
    state: the ascent is either flown now (``optimize``) or replayed from a
    flown run directory (``run_dir``), and any attempt to smuggle state into
    the ``ascent`` block is rejected before a single flight is flown."""
    from orbitopt.mission import two_stage_leo as chain

    doc = reference_scenario().to_dict()
    assert doc["ascent"]["mode"] == "optimize"
    for forbidden in ("seco_state", "injection_state", "ascent_state", "states"):
        scenario = dataclasses.replace(
            reference_scenario(), ascent={**dict(reference_scenario().ascent),
                                         forbidden: {"position_m": [0.0, 0.0, 0.0]}})
        with pytest.raises(ValueError, match="unknown key"):
            chain._acquire_ascent(scenario, tmp_path / forbidden)


def test_e204_synthetic_fixture_is_annotated_as_the_fast_test_exception():
    """E5-01's single exception: the E2-04 frozen mission keeps its synthetic
    two-sample ascent (prescribed separation and SECO states), and must SAY
    so -- in the packaged scenario, in its provenance record and in its
    docs. An un-annotated preset-SECO mission path is the regression this
    guard fails on."""
    scenario_text = (REPO / "src" / "orbitopt" / "data" /
                     "mission_two_stage_leo_recovery.yaml").read_text(encoding="utf-8")
    assert "E5-01" in scenario_text, (
        "mission_two_stage_leo_recovery.yaml must annotate its synthetic "
        "ascent as the E5-01 fast-test exception")
    assert "synthetic" in scenario_text.lower()

    provenance = REPO / "tests" / "data" / "mission" / "reference" / _E204_FIXTURE / \
        "PROVENANCE.json"
    prov = __import__("json").loads(provenance.read_text(encoding="utf-8"))
    assert prov["toolchain"]["opentsiolkovsky"].startswith("not used"), (
        "the frozen E2-04 ascent is synthetic: its provenance must keep saying "
        "OpenTsiolkovsky was not used to compute it")
    assert prov["frozen_inputs"]["ascent_constants"]["t_seco_s"] == 622.0


def test_no_other_consumer_of_the_preset_seco_fixture():
    """The synthetic ascent fixture is consumed ONLY by the E2-04 fast tests
    and its builder -- no other full-mission code path or test may feed its
    preset SECO state to the mission chain (E5-01 exception, tripwire)."""
    offenders = set()
    for path in sorted(REPO.rglob("*.py")):
        if ".git" in path.parts or path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8")
        if "e2-04-frozen-" in text and _E204_FIXTURE.replace("e2-04-frozen-", "") in text:
            offenders.add(path.relative_to(REPO).as_posix())
    assert offenders == _E204_EXCEPTION_FILES, (
        f"unexpected consumers of the E2-04 synthetic ascent fixture: "
        f"{sorted(offenders - _E204_EXCEPTION_FILES)} -- a preset SECO state "
        "may only live in the annotated E2-04 fast-test exception")


def test_recorded_ot_flight_is_a_full_time_history():
    """The real-input contract on the committed RECORDED OpenTsiolkovsky
    flight (no compute stack needed): engine-stamped provenance, a dense
    strictly-increasing time history per body, and an event log containing
    solver-determined events the mission chain never asks for (so it cannot
    be an injected station list)."""
    run = read_run(GATEB)
    assert run.engines == {"opentsiolkovsky-cli": "1.0.0"}
    assert (GATEB / "opentsiolkovsky" / "trajectory.csv").is_file()
    total = 0
    for track in run.tracks:
        ts = list(track.t_s)
        assert all(b > a for a, b in zip(ts, ts[1:])), \
            f"track {track.body_id}: t_s not strictly increasing"
        total += len(ts)
        if track.body_id in ("stage1", "stage2"):
            assert len(ts) >= 20, f"track {track.body_id}: not a full time history"
    assert total >= 200
    kinds = {e.kind for e in run.events}
    assert {"liftoff", "meco", "separation", "seco", "max_q", "impact"} <= kinds
    impact_times = [e.t_s for e in run.events if e.kind == "impact"]
    assert impact_times and all(not float(t).is_integer() for t in impact_times), \
        "impact times must be solver-determined"


# --- slow: the acceptance run (pinned OT binary; Flight Lab Gate B CI) --------


@pytest.mark.slow
@pytest.mark.skipif(not _has_stack(), reason="needs the OpenTsiolkovsky binary + pygmo")
def test_full_chain_ascent_is_computed_by_opentsiolkovsky(tmp_path):
    """E5-01 acceptance run: the packaged full mission chain flies its ascent
    NOW with the pinned OpenTsiolkovsky binary, and every station the chain
    grades is traced back into the computed trajectory -- events from the
    engine's event log, handoff states from track samples, no preset SECO
    state anywhere on the path."""
    mission = run_two_stage_leo(_e5_scenario(), tmp_path / "e5-01")
    assert mission.status == STATUS_OK, mission.message
    assert mission.launch["status"] == "OK"
    assert [s["kind"] for s in mission.stations] == list(REQUIRED_EVENT_CHAIN)
    assert all(c.passed for c in mission.checks)
    assert mission.parking["passed"]

    run_dir = Path(mission.ascent_run_dir)
    run = read_run(run_dir)

    # -- computed by the pinned engine, raw outputs shipped with the run
    assert run.engines == {"opentsiolkovsky-cli": "1.0.0"}
    raw = run_dir / "opentsiolkovsky"
    assert (raw / "trajectory.csv").is_file() and (raw / "summary.json").is_file()

    # -- full time history: dense strictly-increasing tracks, on disk
    total = 0
    for track in run.tracks:
        ts = list(track.t_s)
        assert all(b > a for a, b in zip(ts, ts[1:])), \
            f"track {track.body_id}: t_s not strictly increasing"
        rises = [b - a for a, b in zip(track.mass_kg, track.mass_kg[1:]) if b - a > 1e-3]
        assert not rises, f"track {track.body_id}: mass gained along the history"
        csv_path = run_dir / "trajectories" / f"{track.body_id}.csv"
        assert csv_path.is_file()
        assert sum(1 for _ in csv_path.open(encoding="utf-8")) - 1 == len(ts)
        total += len(ts)
        if track.body_id in ("stage1", "stage2"):
            assert len(ts) >= 20, f"track {track.body_id}: not a full time history"
    assert total >= 200

    # -- events from the simulation, not constants: the engine log carries
    # events the chain never asks for, at solver-determined times
    kinds = {e.kind for e in run.events}
    assert {"liftoff", "meco", "separation", "seco", "max_q", "impact"} <= kinds
    for kind in ("liftoff", "meco", "separation", "seco"):
        station = next(s for s in mission.stations if s["kind"] == kind)
        event = next(e for e in run.events if e.kind == kind)
        assert abs(station["t_s"] - event.t_s) <= 1e-9
    impact_times = sorted(e.t_s for e in run.events if e.kind == "impact")
    assert impact_times and all(not float(t).is_integer() for t in impact_times), \
        "impact times must be solver-determined, not injected"
    meco = next(e for e in run.events if e.kind == "meco")
    stage1 = _track(run, "stage1")
    assert 0.0 < meco.t_s < stage1.t_s[-1], "MECO must fall inside the flown stage-1 arc"

    # -- SECO state: the chain's parking-orbit handoff IS the computed track
    # sample at the engine's seco event (identity frame bridge for GCRS)
    seco = next(e for e in run.events if e.kind == "seco")
    ins = gate_checks.insertion_state(run)
    stage2 = _track(run, "stage2")
    assert ins["body_id"] == "stage2"
    assert abs(ins["t_s"] - seco.t_s) <= 1.0
    nearest = min(range(len(stage2.t_s)), key=lambda i: abs(stage2.t_s[i] - seco.t_s))
    assert ins["t_s"] == stage2.t_s[nearest]
    parked = mission.parking["state"]
    assert all(abs(a - b) <= 1e-6 for a, b in zip(parked["position_m"], ins["position_m"]))
    assert all(abs(a - b) <= 1e-6 for a, b in zip(parked["velocity_m_s"], ins["velocity_m_s"]))
    assert abs(parked["mass_kg"] - ins["mass_kg"]) <= 1e-6

    # -- staging states: graded from the flown track samples themselves
    fork_boundary = next(d for d in mission.staging
                         if d["graded"] and d["child_body"] == "stage2")
    assert fork_boundary["passed"]
    assert all(abs(a - b) <= 1e-6 for a, b
               in zip(fork_boundary["after"]["position_m"], stage2.position_m[0]))

    # -- one report, graded green; the M-05 leg is honest either way
    report = read_report(mission.report_path)
    validate_report(report)
    assert report["status"]["result"] == "passed"
    if _has_tudat():
        assert mission.propagation["result"] == "passed"
    else:
        assert mission.propagation["status"] == "not_run"
        assert any(n["id"] == "propagation" for n in mission.not_assessed)

