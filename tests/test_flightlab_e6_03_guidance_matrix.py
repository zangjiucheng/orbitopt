"""E6-03 (#275): the upper-stage guidance law is general, or its envelope is.

Two halves.

The matrix itself is checked from its declared definition (fast): the
acceptance's coverage clause (>=3 target altitudes, >=2 inclinations, >=2
payload/vehicle configs, >=2 actuator constraint sets), the invariant that the
guided configs all keep the parity 8 t wet mass so only the tank split moves,
that a cell's actuator set is applied to BOTH the plant's engine travel and the
servo model, that a stage with no TVC travel gets no law at all, and that every
failure reason code is derivable from the burn bookkeeping a record carries (the
attribution is a rule over observable quantities, not a label).

The physics is checked from the committed study
``docs/benchmarks/e6_03_guidance_matrix.json`` (regenerate with
``benchmarks/bench_e6_03_guidance_matrix.py``): every cell carries its own
insertion grading against its DECLARED target and its own cross-backend
difference, the flown verdicts match the declared expectations, every miss names
a cause its own numbers support, and the E1-03 scenario point still lands inside
the band with a 6-DOF/3-DOF difference of the same small order E1-03 reported --
generalisation must not cost the point that was already proven. One slow test
re-flies a cell, to show the committed numbers are the physics and not the JSON.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from orbitopt.flightlab.sixdof.guidance_matrix import (
    ACTUATOR_SETS, BACKEND_PAIR, BURN_ENDED_BY, DEPLETION_MASS_KG,
    FAILURE_REASONS, GUIDANCE_MATRIX, INCLINATION_TOLERANCE_DEG,
    INSERTION_TOLERANCE_M, REASON_ACTUATOR_CONSTRAINT, REASON_GUIDANCE,
    REASON_VEHICLE_CAPABILITY, VEHICLE_CONFIGS, failure_reasons, matrix_coverage,
    matrix_scenario, run_matrix_cell,
)

REPO = Path(__file__).resolve().parents[1]
BENCH_JSON = REPO / "docs" / "benchmarks" / "e6_03_guidance_matrix.json"

PARITY_WET_MASS_KG = 8_000.0
BASELINE = "a220-i28.5-V1-A1"


def _cell(cell_id: str):
    return next(c for c in GUIDANCE_MATRIX if c.cell_id == cell_id)


# ---------------------------------------------------------------------------
# the declared matrix
# ---------------------------------------------------------------------------

def test_matrix_covers_every_axis_the_acceptance_asks_for():
    cov = matrix_coverage()
    assert len(cov["target_altitudes_m"]) >= 3
    assert len(cov["inclinations_deg"]) >= 2
    assert len(cov["vehicle_ids"]) >= 2
    assert len(cov["actuator_ids"]) >= 2
    # and the axes are actually crossed, not just listed side by side: the
    # heavy-payload config and the stretched tank both appear against more than
    # one declared target altitude.
    for vehicle_id in ("V3-heavy-payload", "V2-stretched-tank"):
        altitudes = {c.target_altitude_m for c in GUIDANCE_MATRIX
                     if c.vehicle_id == vehicle_id}
        assert len(altitudes) >= 2, f"{vehicle_id} only flown at one altitude"
    for inclination in cov["inclinations_deg"]:
        altitudes = {c.target_altitude_m for c in GUIDANCE_MATRIX
                     if c.inclination_deg == inclination}
        assert len(altitudes) >= 2, f"inclination {inclination} only at one target"


def test_the_original_f06_stage_is_in_the_matrix():
    original = [c for c in GUIDANCE_MATRIX if not c.vehicle.guided]
    assert original, "the ORIGINAL F-06 stage must be one of the configs"
    for cell in original:
        assert cell.vehicle_id == "V0-f06-original"
        assert cell.actuator_id == "A0-no-tvc"
        assert (cell.vehicle.dry_mass_kg, cell.vehicle.propellant_mass_kg) \
            == (2_000.0, 6_000.0)


def test_cell_ids_and_axis_references_are_clean():
    ids = [c.cell_id for c in GUIDANCE_MATRIX]
    assert len(set(ids)) == len(ids)
    for cell in GUIDANCE_MATRIX:
        assert cell.vehicle_id in VEHICLE_CONFIGS
        assert cell.actuator_id in ACTUATOR_SETS
        assert cell.expected in ("insert", "miss")
        assert cell.note
        # a guided stage cannot name the no-TVC set, and a stage with no TVC
        # axis cannot name a servo (the plant and the law must agree).
        assert cell.vehicle.guided == (cell.actuator.travel_deg > 0.0)


def test_guided_configs_move_only_the_tank_split():
    guided = [v for v in VEHICLE_CONFIGS.values() if v.guided]
    assert len(guided) >= 2
    for cfg in guided:
        assert cfg.wet_mass_kg == PARITY_WET_MASS_KG
    assert len({(v.dry_mass_kg, v.propellant_mass_kg) for v in guided}) == len(guided)


def test_actuator_constraint_sets_are_ordered_and_distinct():
    guided = [a for a in ACTUATOR_SETS.values() if a.travel_deg > 0.0]
    assert len(guided) >= 2
    assert len({(a.travel_deg, a.slew_rate_deg_s, a.lag_tau_s)
                for a in guided}) == len(guided)
    assert all(a.lag_tau_s > 0.0 and a.slew_rate_deg_s > 0.0 for a in guided)
    assert ACTUATOR_SETS["A0-no-tvc"].travel_deg == 0.0
    assert ACTUATOR_SETS["A0-no-tvc"].tvc_actuator() is None
    standard = ACTUATOR_SETS["A1-f04-standard"]
    assert math.isclose(standard.travel_deg, 3.0)  # the E1-03/E1-02 servo
    assert math.isclose(standard.slew_rate_deg_s, math.degrees(0.10), rel_tol=1e-9)


def test_scenario_carries_its_own_target_and_servo():
    for cell_id in (BASELINE, "a220-i45.0-V1-A1", "a260-i28.5-V1-A1",
                    "a220-i28.5-V1-A3", "a220-i28.5-V3-A1"):
        cell = _cell(cell_id)
        scenario = matrix_scenario(cell)
        target = scenario.upper_guidance.target
        assert target.altitude_m == cell.target_altitude_m
        assert target.inclination_deg == cell.inclination_deg
        assert target.periapsis_tolerance_m == INSERTION_TOLERANCE_M
        assert target.inclination_tolerance_deg == INCLINATION_TOLERANCE_DEG
        # the cell's actuator set reaches BOTH the servo model and the plant's
        # engine travel -- one actuator axis, not two.
        assert scenario.upper_actuator is not None
        assert scenario.upper_actuator.limit_rad == pytest.approx(
            math.radians(cell.actuator.travel_deg))
        assert scenario.upper_actuator.tau_s == cell.actuator.lag_tau_s
        assert scenario.pair.upper.engine.gimbal_limit_rad == pytest.approx(
            math.radians(cell.actuator.travel_deg))
        assert scenario.pair.upper.propellant_mass_kg == cell.vehicle.propellant_mass_kg
        assert scenario.state0.mass_kg == pytest.approx(78_000.0)


def test_pad_latitude_is_the_declared_inclination_of_the_cell():
    # due-east launch: the achievable inclination is the pad latitude, read as
    # the geocentric latitude of a geodetic pad (the residual the grading
    # reports is that conversion plus the Earth-rotation drift, not a dog-leg
    # the native plant cannot fly).
    for cell_id in ("a220-i28.5-V1-A1", "a220-i45.0-V1-A1", "a220-i51.6-V1-A1"):
        cell = _cell(cell_id)
        r = matrix_scenario(cell).state0.position_r_m if hasattr(
            matrix_scenario(cell), "position_r_m") else None
        scenario = matrix_scenario(cell)
        pos = getattr(scenario.state0, "position_r_m", None)
        if pos is None:
            pos = scenario.state0.position_m
        lat = math.degrees(math.atan2(float(pos[2]),
                                      math.hypot(float(pos[0]), float(pos[1]))))
        assert abs(lat - cell.inclination_deg) < 0.35, f"{cell_id}: {lat}"


def test_a_stage_without_tvc_travel_gets_no_law():
    cell = _cell("a220-i28.5-V0-A0")
    scenario = matrix_scenario(cell)
    assert scenario.upper_guidance is None
    assert scenario.pair.upper.engine.gimbal_limit_rad == 0.0
    assert scenario.pair.upper.dry_mass_kg == 2_000.0
    assert scenario.name == "f07-two-stage"  # the unguided F-06 mission


# ---------------------------------------------------------------------------
# the attribution rule, on synthetic bookkeeping
# ---------------------------------------------------------------------------

def _record(inserted: bool, *, prop_left: float, ended_by: str = "law-cutoff",
            energy_margin_m: float = 5_000.0, burn_off: float | None = 202.0) -> dict:
    return {
        "inserted": inserted,
        "propellant_remaining_kg": prop_left,
        "burn_off_t_s": burn_off,
        "burn_ended_by": ended_by,
        "energy_margin_vs_target_orbit_m": energy_margin_m,
        "propellant_at_ignition_kg": 7_000.0,
        "insertion": {"d_periapsis_m": -40_000.0},
        "speed_surplus_m_s": -40.0,
        "terminal_flight_path_angle_deg": -12.0,
        "dv_ideal_m_s": 2_600.0,
    }


def _pair(three: dict, six: dict) -> dict:
    return {BACKEND_PAIR[0]: three, BACKEND_PAIR[1]: six}


def test_an_unguided_stage_is_a_vehicle_limit_not_a_guidance_result():
    cell = _cell("a220-i28.5-V0-A0")
    backends = _pair(_record(False, prop_left=0.0, ended_by="no-law", burn_off=None),
                     _record(False, prop_left=0.0, ended_by="no-law", burn_off=None))
    codes = [r["code"] for r in failure_reasons(cell, backends)]
    assert codes == [REASON_VEHICLE_CAPABILITY]


def test_running_dry_with_a_tighter_orbit_than_the_target_is_a_vehicle_limit():
    # the tank is what stopped the stage: the terminal orbit does not even have
    # the energy of the target circle.
    cell = _cell("a220-i28.5-V3-A1")
    backends = _pair(_record(False, prop_left=0.0, ended_by="depletion",
                             energy_margin_m=-95_000.0),
                     _record(False, prop_left=0.0, ended_by="depletion",
                             energy_margin_m=-77_000.0))
    codes = [r["code"] for r in failure_reasons(cell, backends)]
    assert codes == [REASON_VEHICLE_CAPABILITY]


def test_running_dry_on_an_ORBIT_WIDER_THAN_THE_TARGET_IS_A_GUIDANCE_FAILURE():
    # the tank emptied because the cutoff never closed -- and the stage passed
    # the target's own energy on the way, so the budget was not the limit.
    cell = _cell("a300-i28.5-V2-A1")
    backends = _pair(_record(False, prop_left=0.0, ended_by="depletion",
                             energy_margin_m=3_400_000.0),
                     _record(False, prop_left=0.0, ended_by="depletion",
                             energy_margin_m=float("inf")))
    codes = [r["code"] for r in failure_reasons(cell, backends)]
    assert codes == [REASON_GUIDANCE]


def test_one_backend_inserting_is_an_actuator_constraint():
    # precedence matters: the mis-tracked backend here ran dry, and the reason
    # is still the servo, because the actuator-free twin inserted.
    cell = _cell("a220-i28.5-V1-A3")
    backends = _pair(_record(True, prop_left=52.0),
                     _record(False, prop_left=0.0, ended_by="depletion",
                             energy_margin_m=-6_000_000.0))
    codes = [r["code"] for r in failure_reasons(cell, backends)]
    assert codes == [REASON_ACTUATOR_CONSTRAINT]


def test_a_closed_burn_that_still_misses_is_the_laws_own_error():
    cell = _cell("a180-i28.5-V1-A1")
    backends = _pair(_record(False, prop_left=478.0, energy_margin_m=24_000.0),
                     _record(False, prop_left=484.0, energy_margin_m=20_000.0))
    codes = [r["code"] for r in failure_reasons(cell, backends)]
    assert codes == [REASON_GUIDANCE]


def test_every_reason_code_is_one_of_the_three_the_acceptance_names():
    cell = _cell("a300-i28.5-V1-A1")
    for backends in (
            _pair(_record(False, prop_left=0.0, ended_by="depletion",
                          energy_margin_m=2e6),
                  _record(False, prop_left=0.0, ended_by="depletion",
                          energy_margin_m=2e6)),
            _pair(_record(True, prop_left=10.0),
                  _record(False, prop_left=0.0, ended_by="depletion",
                          energy_margin_m=-4e5)),
            _pair(_record(False, prop_left=0.0, ended_by="depletion",
                          energy_margin_m=-3e5),
                  _record(False, prop_left=0.0, ended_by="depletion",
                          energy_margin_m=-3.1e5)),
            _pair(_record(False, prop_left=80.0, energy_margin_m=4e3),
                  _record(False, prop_left=80.0, energy_margin_m=4.5e3)),
            _pair(_record(False, prop_left=0.0, ended_by="still-burning",
                          energy_margin_m=1e6),
                  _record(False, prop_left=0.0, ended_by="still-burning",
                          energy_margin_m=1e6))):
        for reason in failure_reasons(cell, backends):
            assert reason["code"] in FAILURE_REASONS
            assert len(reason["detail"]) > 40


# ---------------------------------------------------------------------------
# the committed matrix study
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bench() -> dict:
    if not BENCH_JSON.exists():
        pytest.skip("run benchmarks/bench_e6_03_guidance_matrix.py first")
    return json.loads(BENCH_JSON.read_text())


def _records(bench: dict) -> dict[str, dict]:
    return {rec["cell_id"]: rec for rec in bench["cells"]}


def test_committed_study_declares_its_schema_and_coverage(bench):
    assert bench["schema"] == "orbitopt-e603-1"
    assert "#275" in bench["task"]
    assert bench["coverage"] == matrix_coverage()
    cov = bench["coverage"]
    assert len(cov["target_altitudes_m"]) >= 3
    assert len(cov["inclinations_deg"]) >= 2
    assert len(cov["vehicle_ids"]) >= 2
    assert len(cov["actuator_ids"]) >= 2


def test_every_declared_cell_was_flown_exactly_once(bench):
    recs = _records(bench)
    assert set(recs) == {c.cell_id for c in GUIDANCE_MATRIX}
    for rec in recs.values():
        assert set(rec["backends"]) == set(BACKEND_PAIR)


def test_every_cell_grades_itself_against_its_own_declared_target(bench):
    for rec in bench["cells"]:
        cell = _cell(rec["cell_id"])
        assert rec["target"]["altitude_m"] == cell.target_altitude_m
        assert rec["target"]["inclination_deg"] == cell.inclination_deg
        assert rec["vehicle"]["vehicle_id"] == cell.vehicle_id
        assert rec["actuator"]["actuator_id"] == cell.actuator_id
        assert rec["expected"] == cell.expected
        for backend, data in rec["backends"].items():
            ins = data["insertion"]
            assert ins["target_altitude_m"] == rec["target"]["altitude_m"]
            assert math.isclose(ins["d_periapsis_m"],
                                ins["periapsis_altitude_m"]
                                - rec["target"]["altitude_m"], abs_tol=1e-3)
            assert math.isclose(ins["d_apoapsis_m"],
                                ins["apoapsis_altitude_m"]
                                - rec["target"]["altitude_m"], abs_tol=1e-3)
            assert math.isclose(ins["d_inclination_deg"],
                                ins["inclination_deg"] - cell.inclination_deg,
                                abs_tol=1e-6)
            assert ins["inserted"] is data["inserted"]
            assert ins["inserted"] is bool(ins["periapsis_within_tolerance"]
                                           and ins["apoapsis_within_tolerance"]
                                           and ins["inclination_within_tolerance"])
        assert rec["inserted_both"] is all(b["inserted"] for b in rec["backends"].values())
        assert rec["status"] == ("pass" if rec["inserted_both"] else "miss")


def test_every_cell_records_its_cross_backend_difference(bench):
    for rec in bench["cells"]:
        cross = rec["cross_backend"]
        for key in ("d_periapsis_6dof_minus_3dof_m", "d_apoapsis_6dof_minus_3dof_m",
                    "d_inclination_6dof_minus_3dof_deg"):
            assert math.isfinite(cross[key]), f"{rec['cell_id']}: {key}"
        three, six = (rec["backends"][b]["insertion"] for b in BACKEND_PAIR)
        assert math.isclose(cross["d_periapsis_6dof_minus_3dof_m"],
                            six["periapsis_altitude_m"] - three["periapsis_altitude_m"],
                            abs_tol=1e-3)
        # both sides are the study's own values rounded to 6 decimals, so the
        # difference of the pair can only be checked at that precision
        assert math.isclose(cross["d_inclination_6dof_minus_3dof_deg"],
                            six["inclination_deg"] - three["inclination_deg"],
                            abs_tol=1e-5)
        diff = cross["ascent_diff"]
        assert not diff.get("error"), f"{rec['cell_id']}: {diff.get('error')}"
        assert diff["t_span_s"] > 20.0
        assert math.isfinite(diff["d_altitude_m_max_abs_m"])
        assert math.isfinite(diff["d_mass_kg_terminal_kg"])


def test_the_flown_verdicts_are_the_declared_ones(bench):
    recs = _records(bench)
    unexpected = [cid for cid, rec in recs.items() if rec["unexpected"]]
    assert not unexpected, f"matrix cells that flipped verdict: {unexpected}"
    assert recs[BASELINE]["status"] == "pass"
    assert recs["a180-i28.5-V1-A1"]["status"] == "miss"
    assert recs["a220-i28.5-V1-A3"]["status"] == "miss"
    assert recs["a220-i28.5-V0-A0"]["status"] == "miss"
    # the pass set is exactly the declared one, so no cell can drift across the
    # band without this test (and the unexpected check above) noticing.
    assert {cid for cid, rec in recs.items() if rec["status"] == "pass"} \
        == {c.cell_id for c in GUIDANCE_MATRIX if c.expected == "insert"}


def test_every_miss_names_a_cause_its_own_numbers_support(bench):
    for rec in bench["cells"]:
        if rec["status"] == "pass":
            assert rec["reasons"] == []
            continue
        codes = [r["code"] for r in rec["reasons"]]
        assert codes and set(codes) <= set(FAILURE_REASONS), rec["cell_id"]
        backends = rec["backends"]
        for reason in rec["reasons"]:
            assert len(reason["detail"]) > 40
            ended = [b["burn_ended_by"] for b in backends.values()]
            margin = max(b["energy_margin_vs_target_orbit_m"] or 0.0
                         for b in backends.values())
            if reason["code"] == REASON_ACTUATOR_CONSTRAINT:
                assert sum(1 for b in backends.values() if b["inserted"]) == 1
            elif reason["code"] == REASON_VEHICLE_CAPABILITY:
                if rec["vehicle"]["guided"]:  # else: no TVC axis at all
                    assert set(ended) == {"depletion"}
                    assert margin < 0.0 or margin is None  # the orbit is too tight
            else:  # guidance: energy to spare, or the law shut the burn and missed
                assert set(ended) <= {"depletion", "law-cutoff", "still-burning"}
                assert (set(ended) == {"law-cutoff"}
                        or any(b["burn_ended_by"] != "depletion"
                               for b in backends.values())
                        or margin > 0.0)


def test_the_actuator_axis_moves_the_6dof_result_but_not_the_command(bench):
    recs = _records(bench)
    standard, rate_limited = recs[BASELINE], recs["a220-i28.5-V1-A3"]
    assert standard["status"] == "pass" and rate_limited["status"] == "miss"
    # the point-mass twin is actuator-free: the same law on the same command is
    # still inside the band when the servo cannot follow it.
    assert rate_limited["backends"][BACKEND_PAIR[0]]["inserted"]
    assert not rate_limited["backends"][BACKEND_PAIR[1]]["inserted"]
    # and the wide-travel set does not hurt: it inserts too.
    assert recs["a220-i28.5-V1-A4"]["status"] == "pass"


def test_the_altitude_axis_is_bounded_at_both_ends_and_neither_limit_is_the_tank(bench):
    recs = _records(bench)
    assert recs[BASELINE]["status"] == "pass"
    assert recs["a180-i28.5-V1-A1"]["status"] == "miss"
    for cell_id in ("a260-i28.5-V1-A1", "a300-i28.5-V1-A1", "a400-i28.5-V1-A1"):
        assert recs[cell_id]["status"] == "miss"
    # low end: the burn terminated by the law, periapsis inside the band, and it
    # is APOAPSIS that sits outside it -- the criterion grades one element.
    low = recs["a180-i28.5-V1-A1"]
    assert [r["code"] for r in low["reasons"]] == [REASON_GUIDANCE]
    for data in low["backends"].values():
        assert data["burn_ended_by"] == "law-cutoff"
        assert data["propellant_remaining_kg"] > 100.0  # nothing ran dry here
        assert abs(data["insertion"]["d_periapsis_m"]) < INSERTION_TOLERANCE_M
        assert data["insertion"]["d_apoapsis_m"] > INSERTION_TOLERANCE_M
    # high end: 200 kg MORE propellant makes the same missed target WORSE, so the
    # tank was never the binding limit -- the cutoff criterion is.
    lean, standard = recs["a300-i28.5-V2-A1"], recs["a300-i28.5-V1-A1"]
    assert lean["vehicle"]["upper_propellant_mass_kg"] \
        > standard["vehicle"]["upper_propellant_mass_kg"]
    for backend in BACKEND_PAIR:
        assert lean["backends"][backend]["insertion"]["d_apoapsis_m"] \
            > standard["backends"][backend]["insertion"]["d_apoapsis_m"]
        assert lean["backends"][backend]["burn_ended_by"] == "depletion"


def test_the_inclination_axis_is_inserted_at_every_declared_angle(bench):
    recs = _records(bench)
    for cell_id in ("a220-i28.5-V1-A1", "a220-i45.0-V1-A1", "a220-i51.6-V1-A1"):
        rec = recs[cell_id]
        assert rec["status"] == "pass", rec["cell_id"]
        for data in rec["backends"].values():
            assert abs(data["insertion"]["d_inclination_deg"]) \
                <= INCLINATION_TOLERANCE_DEG
            # the same law, the same pad longitude, a different pad: the
            # insertion errors stay of the same order as the baseline cell
            assert abs(data["insertion"]["d_periapsis_m"]) < 25e3
            assert abs(data["insertion"]["d_apoapsis_m"]) < 25e3


def test_e103_scenario_point_still_inserts_with_a_small_backend_difference(bench):
    # the generalisation study must not cost the point E1-03 already proved:
    # inside the 25 km band on both backends, and a 6-DOF/3-DOF terminal
    # difference of the same 1.65 km order E1-03 reported (never the 235 km the
    # open-loop F-06 stage produced).
    rec = _records(bench)[BASELINE]
    cross = rec["cross_backend"]
    assert rec["status"] == "pass"
    assert abs(cross["d_periapsis_6dof_minus_3dof_m"]) < 15e3
    assert abs(cross["d_apoapsis_6dof_minus_3dof_m"]) < 15e3
    assert abs(cross["d_inclination_6dof_minus_3dof_deg"]) < 0.1
    original = _records(bench)["a220-i28.5-V0-A0"]
    assert abs(original["cross_backend"]["d_periapsis_6dof_minus_3dof_m"]) \
        > 100e3  # open loop, no law: the pre-E1-03 residual is huge, as F-06 showed


def test_guided_cells_hold_their_burn_bookkeeping(bench):
    for rec in bench["cells"]:
        for data in rec["backends"].values():
            assert data["burn_ended_by"] in BURN_ENDED_BY
            assert 0.0 <= data["propellant_remaining_kg"] \
                <= data["propellant_at_ignition_kg"] + 1e-6
            assert math.isclose(
                data["propellant_burned_kg"],
                data["propellant_at_ignition_kg"] - data["propellant_remaining_kg"],
                abs_tol=1e-3)
            assert data["dv_ideal_m_s"] > 1e3
            assert data["target_circular_speed_m_s"] > 7e3
            if not rec["vehicle"]["guided"]:  # no law: nothing could command a shut
                assert data["burn_ended_by"] == "no-law"
                assert data["burn_off_t_s"] is None
                assert data["propellant_remaining_kg"] == pytest.approx(0.0, abs=1e-6)
                continue
            assert data["burn_ended_by"] != "no-law"
            if data["burn_ended_by"] == "depletion":
                assert data["propellant_remaining_kg"] <= DEPLETION_MASS_KG
            elif data["burn_ended_by"] == "law-cutoff":
                assert data["propellant_remaining_kg"] > DEPLETION_MASS_KG
                assert data["ignition_t_s"] < data["burn_off_t_s"] <= rec["t_end_s"]
            if data["inserted"]:
                # an inserted cell was terminated by the law's own cutoff (or
                # emptied exactly on the target), never by coasting past it
                assert data["burn_ended_by"] in ("law-cutoff", "depletion")


@pytest.mark.slow
def test_committed_baseline_cell_reproduces():
    if not BENCH_JSON.exists():
        pytest.skip("run benchmarks/bench_e6_03_guidance_matrix.py first")
    committed = _records(json.loads(BENCH_JSON.read_text()))[BASELINE]
    rec = run_matrix_cell(_cell(BASELINE))
    assert rec["status"] == committed["status"]
    for backend in BACKEND_PAIR:
        for key in ("d_periapsis_m", "d_apoapsis_m"):
            assert abs(rec["backends"][backend]["insertion"][key]
                       - committed["backends"][backend]["insertion"][key]) < 1.0
