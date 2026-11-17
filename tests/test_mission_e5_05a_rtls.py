"""E5-05a (task #285): the RTLS-capable vehicle redesign and its mission.

The E5-05 blocker (ContextDB ``e5-05-rtls-infeasibility-evidence``): the
sealed catalog vehicle burns stage 1 to depletion, so separation mass ==
dry mass and the recovery chain has nothing to plan from -- RTLS is
infeasible *by construction*, and #272 could not record a computed
launch-to-orbit/landing reference. The redesign ships a new catalog vehicle
(``synthetic_two_stage_rtls``) whose declared stage-1 cutoff leaves a real,
COMPUTED propellant reserve at separation, and a new flight-target scenario.

Fast tests (no compute stack): the packaged documents load, the reserve is a
computed structural property (burn-off bookkeeping, not a constant), and the
sealed E2-04 reference files are untouched.

Slow tests (pinned OpenTsiolkovsky + gfold): the vehicle FLIES LIVE -- MECO
carries the computed reserve, the RTLS chain plans from THAT separation
state and grades the reserve gate green, and the packaged mission chain
replays the recorded flight end to end. The insertion band (F9) CLOSED in
E5-05d (#288): the redesigned stage 2 + coast/re-ignition schedule inserts
inside the band at SECO and the gate is a hard assertion. The landing (F10)
CLOSED in E5-05c (#287): closed-loop return guidance plus a +48 t stage-1
reserve (resized stage 1, still inside the F9 band) land the booster from
the computed separation state, and both landing tests are hard assertions.
E2-04 keeps ownership of the fast synthetic-integration test (E5-01 exception);
this file is the fully-computed E5-05a reference.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from orbitopt.flightlab.contract import load_target, read_run
from orbitopt.flightlab.recovery.propagator import BoosterModel
from orbitopt.mission.two_stage_leo import scenario_from_config
from orbitopt.mission_config import load_mission_config

REPO = Path(__file__).resolve().parents[1]
VEHICLE = REPO / "src" / "orbitopt" / "data" / "flightlab" / "vehicles" / \
    "synthetic_two_stage_rtls.yaml"
SCENARIO = REPO / "src" / "orbitopt" / "data" / "flightlab" / "scenarios" / \
    "leo_two_stage_rtls.yaml"
MISSION = REPO / "src" / "orbitopt" / "data" / "mission_two_stage_leo_rtls.yaml"
FIXTURE = REPO / "tests" / "data" / "mission" / "reference" / \
    "e5-05a-rtls-mission"

#: the sealed references E5-05a must never touch (task: keep E2-04 as the
#: fast integration test; digest pinning lives in the sealed-manifest suite)
SEALED = [
    "src/orbitopt/data/flightlab/vehicles/synthetic_two_stage.yaml",
    "src/orbitopt/data/flightlab/scenarios/leo_two_stage.yaml",
    "src/orbitopt/data/mission_two_stage_leo_recovery.yaml",
]

G0 = 9.80665
MU_EARTH = 398600.4418e9
RE_EARTH = 6371000.0


def _has_stack() -> bool:
    from orbitopt.flightlab.adapters.opentsiolkovsky import find_binary

    return (importlib.util.find_spec("pygmo") is not None
            and find_binary() is not None)


def _has_gfold() -> bool:
    return importlib.util.find_spec("gfold") is not None


# --- fast: the packaged documents and the computed-reserve property ---------


def test_packaged_documents_validate():
    """Vehicle, scenario and mission are packaged and schema-valid, and the
    mission wires run_dir + drive_from_separation + the ADR plan recipe."""
    booster = BoosterModel.load(VEHICLE)
    assert booster.landing_propellant_reserve_kg > 0

    target = load_target(SCENARIO)
    assert target.recovery.kind == "rtls"
    assert target.recovery.min_propellant_reserve_kg == 10000.0

    scenario = scenario_from_config(load_mission_config(MISSION))
    assert scenario.ascent_mode == "run_dir"
    assert scenario.ascent_run_dir.endswith("e5-05a-rtls-mission/ascent")
    assert scenario.recovery.drive_from_separation is True
    opts = scenario.recovery.plan_options
    # ADR flightlab-0008 (F10, E5-05c #287): the Monte Carlo-verified fixed
    # script with the closed-loop return options it was verified under.
    assert len(opts["initial_guess"]) == 8 and opts["search"] == "keep"
    assert (opts["boostback_cutoff"], opts["entry_cutoff"],
            opts["landing_need_model"]) == ("iip", "speed", "closed_loop")
    assert opts["entry_throttle"] is True and opts["entry_burns"] == 1
    assert opts["entry_q_max_pa"] == 250000.0 and opts["entry_g_max"] == 12.0
    assert (opts["dt_opt"], opts["dt_final"]) == (0.25, 0.1)
    assert opts["landing_reserve"] == "target"
    assert opts["landing"]["max_thrust_accel_g"] == 3.6


def test_stage_one_reserve_is_a_computed_structural_property():
    """The reserve mechanism: the vehicle's OWN fields (not a constant and
    not the sealed run-to-depletion pattern) imply a real stage-1 reserve at
    separation -- mass_initial - mass_dry - massflow * (burn_end - burn_start)
    -- large enough for the RTLS chain, while insertion is still sized."""
    import yaml

    doc = yaml.safe_load(VEHICLE.read_text(encoding="utf-8"))
    s1 = next(s for s in doc["stages"] if s.get("role", "first") != "second"
              and s["engine"]["burn_start_s"] == 0.0)
    prop = s1["mass_initial_kg"] - s1["mass_dry_kg"]
    eng = s1["engine"]
    thrust = eng["thrust_vac_n"]
    thrust_n = thrust[0][1] if isinstance(thrust, list) else thrust
    flow = thrust_n / (eng["isp_vac_s"] * G0)
    reserve = prop - flow * (eng["burn_end_s"] - eng["burn_start_s"])
    assert eng["burn_end_s"] < prop / flow, \
        "stage 1 must not run to depletion (the E5-05 blocker)"
    assert reserve >= 90_000.0, \
        f"computed stage-1 reserve {reserve:.0f} kg below the RTLS sizing (ADR-0008)"
    booster = BoosterModel.load(VEHICLE)
    assert booster.landing_propellant_reserve_kg <= reserve


@pytest.mark.parametrize("rel", SEALED)
def test_sealed_e204_references_untouched(rel):
    """E5-05a adds files; the sealed E2-04 baseline and its fast test stay
    byte-identical (task scope). Digests: the sealed-manifest suite pins the
    content; here: existence + the E5-01 exception annotation."""
    path = REPO / rel
    assert path.is_file()
    if "recovery" in rel:
        assert "E5-01" in path.read_text(encoding="utf-8")


# --- slow: the computed flight, the computed plan, the chained mission ------


@pytest.fixture(scope="module")
def live_flight(tmp_path_factory):
    """One LIVE OpenTsiolkovsky flight of the shipped vehicle (no optimizer,
    no fixture): the acceptance evidence that the ascent itself is computed."""
    from orbitopt.flightlab.adapters.opentsiolkovsky import run_ascent

    out = tmp_path_factory.mktemp("e5-05a-live")
    return run_ascent(load_target(SCENARIO), VEHICLE, None, out, timeout_s=900.0)


def _track_array(run, body: str):
    track = next(t for t in run.tracks if t.body_id == body)
    return (np.asarray(track.t_s), np.asarray(track.position_m),
            np.asarray(track.velocity_m_s), track.frame)


def _elements_at(run, body: str, epoch_utc):
    """Classical elements of the last dense samples of a track, in GCRS
    (the OT tracks are ECEF; frames.to_gcrs bridges them inertially)."""
    from orbitopt.flightlab import frames

    t, r, v, frame = _track_array(run, body)
    take = slice(-200, None) if len(t) >= 200 else slice(None)
    rg, vg = frames.to_gcrs(r[take], v[take], frame, epoch_utc, t[take])
    out = {}
    for rr, vv in zip(rg[-50:], vg[-50:]):
        rm, vm = float(np.linalg.norm(rr)), float(np.linalg.norm(vv))
        h = np.cross(rr, vv)
        energy = vm * vm / 2.0 - MU_EARTH / rm
        a = -MU_EARTH / (2.0 * energy)
        evec = np.cross(vv, h) / MU_EARTH - rr / rm
        ecc = float(np.linalg.norm(evec))
        inc = math.degrees(math.acos(max(-1.0, min(1.0,
              float(h[2]) / float(np.linalg.norm(h))))))
        out = {"a": a, "rp": a * (1.0 - ecc) - RE_EARTH,
               "ra": a * (1.0 + ecc) - RE_EARTH, "inc": inc}
    return out


#: the two walls ADR flightlab-0008 documents (findings F9/F10), both closed:
#: F9 by the E5-05d upper-stage redesign, F10 by the E5-05c closed-loop
#: return + stage-1 reserve resize (campaign and Monte Carlo in the ADR).
F9 = ("F9 (CLOSED, E5-05d #288): the OT commanded-attitude backend cannot "
      "close the insertion band on the FIXED E5-05a vehicle (perigee floor "
      "~-0.7 km single-burn, ~-0.4 km with the best coast/restart schedule); "
      "the E5-05d upper-stage redesign (128 t wet / 8.5 t dry stage-2, Isp "
      "380 s, 14 MN boost leg + coast + re-ignition circularisation) closes "
      "the band on the same computed reserve-carrying separation state "
      "(rp 281.8 km / ra 296.1 km, docs/adr/flightlab-0008)")
F10 = ("F10 (CLOSED, E5-05c #287): the FL-302 timed 7-parameter script could "
       "not reach the FL-301 corridor from a computed MECO; the IIP-cut "
       "boostback, speed-cut throttled entry burn and closed-loop landing "
       "forecast, with a +48 t stage-1 reserve, land it (n=64 Monte Carlo "
        "64/64, docs/adr/flightlab-0008)")



@pytest.mark.slow
@pytest.mark.skipif(not _has_stack(), reason="needs the OpenTsiolkovsky binary + pygmo")
def test_live_ascent_computes_reserve(live_flight):
    """MECO/separation carry a real COMPUTED reserve: the stage-1 debris
    track is heavier than the dry mass by the RTLS sizing. This is the
    anti-E5-05 property (sealed pattern: SEP mass == dry mass)."""
    kinds = {e.kind for e in live_flight.events}
    assert {"liftoff", "meco", "separation", "seco"} <= kinds

    track = next(t for t in live_flight.tracks if t.body_id == "stage1_debris")
    sep_mass = float(np.asarray(track.mass_kg)[0])
    assert sep_mass > 120_000.0, \
        f"separation mass {sep_mass:.0f} kg: computed reserve below the RTLS sizing"

    import yaml

    doc = yaml.safe_load(VEHICLE.read_text(encoding="utf-8"))
    dry = next(s for s in doc["stages"] if s.get("role", "first") != "second"
               and s["engine"]["burn_start_s"] == 0.0)["mass_dry_kg"]
    assert sep_mass - dry >= 90_000.0, \
        f"computed stage-1 reserve {sep_mass - dry:.0f} kg below the RTLS sizing"


@pytest.mark.slow
@pytest.mark.skipif(not _has_stack(), reason="needs the OpenTsiolkovsky binary + pygmo")
def test_seco_insertion_band(live_flight):
    """Insertion closes the 250 km band -- HARD assertion (F9 closed in
    E5-05d #288): the redesigned upper stage + restart/coast schedule reaches
    rp 281.8 km / ra 296.1 km at SECO on the SAME computed run that carries
    the stage-1 reserve (test_live_ascent_computes_reserve). The strict-xfail
    of E5-05b/E5-05c is promoted here per the ADR flightlab-0008 F9
    resolution."""
    target = load_target(SCENARIO)
    elems = _elements_at(live_flight, "stage2", target.mission.epoch_utc)
    assert elems["rp"] > 230_000.0, elems
    assert abs(elems["ra"] - 250_000.0) < 60_000.0, elems


@pytest.mark.slow
@pytest.mark.skipif(not (_has_stack() and _has_gfold()),
                    reason="needs the OpenTsiolkovsky binary and gfold")
def test_recovery_plans_from_the_computed_separation_state(live_flight):
    """FL-302/303 plans on the LIVE computed separation state (the E5-05
    unblocking claim, budget half): the propellant-reserve gate passes on
    the computed (not depleted, not textbook) state with the pinned ADR
    recipe."""
    from orbitopt.flightlab.recovery.pipeline import fly_recovery, \
        separation_state_ecef
    from orbitopt.mission.two_stage_leo import scenario_from_config

    sep = separation_state_ecef(live_flight)
    assert sep.mass_kg > 120_000.0
    scenario = scenario_from_config(load_mission_config(MISSION))
    outcome = fly_recovery(live_flight, load_target(SCENARIO), VEHICLE,
                           plan_options=dict(scenario.recovery.plan_options))
    gate = {c.id: c.passed for c in outcome.checks}
    assert gate.get("recovery-propellant-reserve") is True, outcome.reason_codes
    assert outcome.plan is not None


@pytest.mark.slow
@pytest.mark.skipif(not (_has_stack() and _has_gfold()),
                    reason="needs the OpenTsiolkovsky binary and gfold")
def test_recovery_landing_gate_on_the_computed_state(live_flight):
    """The landing gate closes on the LIVE computed separation state -- HARD
    assertion (F10 closed in E5-05c #287) with the mission's pinned recipe:
    every landing check and the 10 t reserve gate pass."""
    from orbitopt.flightlab.recovery.pipeline import fly_recovery
    from orbitopt.mission.two_stage_leo import scenario_from_config

    scenario = scenario_from_config(load_mission_config(MISSION))
    outcome = fly_recovery(live_flight, load_target(SCENARIO), VEHICLE,
                           plan_options=dict(scenario.recovery.plan_options))
    gate = {c.id: c.passed for c in outcome.checks}
    assert gate.get("recovery-landing") is True, outcome.reason_codes
    assert gate.get("recovery-propellant-reserve") is True, outcome.reason_codes
    assert outcome.landing is not None and outcome.landing.landed
    assert outcome.landing.max_proper_g <= 5.0


@pytest.mark.slow
@pytest.mark.skipif(not FIXTURE.is_dir(),
                    reason="recorded-flight fixture baked by scripts/e505a_reference.py")
@pytest.mark.skipif(not _has_gfold(), reason="needs gfold for the landing leg")
def test_packaged_mission_chain_replays_the_recorded_flight(tmp_path):
    """The packaged mission (run_dir replay of the recorded OT flight,
    drive_from_separation) chains the full dual-branch mission to OK. The
    chain gates that kept this xfail(strict) through E5-05b/c are closed by
    #272: the booster handoff hands over the separation-epoch state, the
    expected parking orbit is the delivered SECO state read AFTER the restart
    burn (checks.insertion_state), and the scenario bands hold the computed
    flight (docs/adr/flightlab-0008 "Mission chain" closure). Propagation
    stays not_checked (E2-04 environment reason)."""
    from orbitopt.mission.two_stage_leo import (
        STATUS_OK, run_two_stage_leo, scenario_from_config)

    scenario = scenario_from_config(load_mission_config(MISSION))
    mission = run_two_stage_leo(scenario, tmp_path / "e5-05a")
    gate = [c for c in mission.checks if getattr(c, "id", "") == "recovery.landed"]
    assert gate, "recovery landing gate missing from the mission chain"
    assert mission.status == STATUS_OK, mission.status


# --- E5-05b: the restart/coast (multi-burn) backend capability --------------

#: a coast + re-ignition schedule that cuts the stage-2 main burn, coasts
#: ballistically, and re-ignites at low thrust -- the mechanism E5-05b adds so
#: stage 2 can circularise (and the enabler for higher bands). Marked
#: illustrative: it is a capability probe, not the shipped reference schedule.
RESTART_PROBE = {"stage": 2, "coast_start_s": 224.0, "coast_duration_s": 120.0,
                 "restart_thrust_n": 1.0e6, "restart_burn_s": 60.0,
                 "restart_elevation_deg": 0.0, "restart_azimuth_deg": 45.0}


def _vehicle_with_restart(tmp_path):
    import yaml

    doc = yaml.safe_load(VEHICLE.read_text(encoding="utf-8"))
    doc["restart"] = dict(RESTART_PROBE)
    doc["data_provenance"]["restart"] = {
        "provenance": "illustrative",
        "source": "task #286 restart/coast capability probe",
        "as_of": "2026-11-16"}
    path = tmp_path / "veh_restart.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


def test_restart_block_is_schema_valid(tmp_path):
    """The vehicle document now accepts an optional `restart` block whose every
    number is covered by a single subtree provenance entry (E5-05b)."""
    from orbitopt.flightlab.vehicle_catalog import load_validated

    doc = load_validated(_vehicle_with_restart(tmp_path))
    assert doc["restart"]["coast_start_s"] == 224.0


def test_restart_synthesizes_a_coast_gap(tmp_path):
    """build_config folds `restart` into the stage thrust curve: the main burn
    drops to zero (coast), stays there, then returns for the re-ignition, and
    the simulation window extends to cover it (E5-05b)."""
    from orbitopt.flightlab.adapters.opentsiolkovsky import Vehicle, build_config

    path = _vehicle_with_restart(tmp_path)
    cfg = build_config(load_target(SCENARIO), Vehicle.load(path), None)
    curve = [[float(t), float(f)] for t, f in cfg["stages"][1]["engine"]["thrust_vac_n"]]
    zero = [i for i, (_, f) in enumerate(curve) if f == 0.0]
    assert zero, "coast: thrust must reach zero through the coast window"
    assert any(f > 0.0 for _, f in curve[zero[-1] + 1:]), \
        "restart: thrust must return after the coast"
    assert cfg["stages"][1]["engine"]["burn_end_s"] > curve[zero[-1]][0], \
        "burn must resume and end after the coast"
    assert cfg["simulation"]["end_time_s"] > RESTART_PROBE["coast_start_s"] + \
        RESTART_PROBE["coast_duration_s"], "sim window must cover the restart"


@pytest.mark.slow
@pytest.mark.skipif(not _has_stack(), reason="needs the OpenTsiolkovsky binary + pygmo")
def test_restart_reignites_a_single_seco(tmp_path):
    """A restart/coast run flies through OT and maps to exactly ONE `seco`:
    the coast-start burnout is suppressed (not insertion), and `seco` is the
    post-coast re-ignition burnout. The raw stage-2 thrust shows the
    powered -> coast -> powered signature. This is the capability gate for
    E5-05b; band closure is a separate (vehicle) question (F9)."""
    import csv

    from orbitopt.flightlab.adapters.opentsiolkovsky import run_ascent

    run = run_ascent(load_target(SCENARIO), _vehicle_with_restart(tmp_path), None,
                     tmp_path / "out", timeout_s=900.0)
    secoes = [e for e in run.events if e.kind == "seco"]
    assert len(secoes) == 1, f"expected one seco, got {[e.kind for e in run.events]}"
    assert secoes[0].t_s > RESTART_PROBE["coast_start_s"] + \
        RESTART_PROBE["coast_duration_s"], secoes[0].t_s
    traj = tmp_path / "out" / "opentsiolkovsky" / "trajectory.csv"
    th = [float(r["thrust_n"]) for r in csv.DictReader(traj.open(encoding="utf-8"))
          if int(float(r["stage"])) == 2]
    assert min(th) < 1.0, "expected a zero-thrust coast interval"
    assert any(t > 1e4 for t in th[len(th) // 2:]), \
        "expected the re-ignition to produce thrust after the coast"
