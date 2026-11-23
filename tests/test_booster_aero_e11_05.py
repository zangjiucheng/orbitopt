"""E11-05 Starbase aero model qualification: module, evidence and sync tests.

The qualified power-off drag model lives in orbitopt.flightlab.booster_aero;
its evidence is the committed bench JSON.  These tests re-derive the module
tables, check the evidence is self-consistent with E10-04 and the 12 % band,
and prove the sealed vehicle YAMLs were NOT edited (the qualified table lands
as a module + evidence, applied at the next vehicle revision).
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from orbitopt.flightlab import booster_aero as ba  # noqa: E402
from orbitopt.flightlab.recovery.propagator import (BoosterModel,  # noqa: E402
                                                    BoosterState, propagate)

EVIDENCE = REPO / "docs/benchmarks/e11_05_starbase_aero_qualification.json"
E1004 = (REPO / "tests/data/mission/reference"
         "/e5-08-new-vehicle-starbase-reports/debris-model-difference.json")
DC8_YAML = (REPO / "src/orbitopt/data/flightlab/vehicles"
            "/starship_class_rtls_dc8.yaml")

# the declared DC8 booster polar, byte-checked here so a silent YAML edit of
# the sealed table fails loudly (digest tests cover the rest of the document)
DC8_DECLARED_CD = [(0.0, 0.8), (0.8, 0.9), (1.1, 1.3), (2.0, 1.25),
                   (4.0, 1.2), (8.0, 1.15)]


@pytest.fixture(scope="module")
def evidence():
    return json.loads(EVIDENCE.read_text())


@pytest.fixture(scope="module")
def veh():
    return BoosterModel.load(DC8_YAML)


def test_design_assumptions_flagged():
    assert ba.DESIGN_ASSUMPTIONS
    for a in ba.DESIGN_ASSUMPTIONS:
        assert "DESIGN ASSUMPTION" in a
    assert ba.QUALIFICATION_SOURCES


def test_engines_first_table_reproduces_committed_evidence(evidence):
    got = ba.cd_engines_first_table()
    want = evidence["part_a_component_tables"]["engines_first_stowed_qualified"]
    assert [r[0] for r in got] == [r[0] for r in want]
    assert all(abs(g[1] - w[1]) <= 1e-3 for g, w in zip(got, want))


def test_engines_first_physics_bounds():
    tbl = dict(ba.cd_engines_first_table())
    # subsonic blockage-dominated: inside the declared +/-30 % at M0/M0.8
    assert abs(tbl[0.0] / 0.80 - 1.0) <= 0.30
    assert abs(tbl[0.8] / 0.90 - 1.0) <= 0.30
    # transonic hump then monotone hypersonic rise (wave + base plateau)
    assert tbl[1.0] < tbl[1.1] and tbl[1.1] < tbl[1.5] < tbl[2.0]
    assert tbl[2.0] < tbl[3.0] < tbl[4.0] < tbl[6.0] < tbl[12.0]
    # bounded above by pitot Cp + full base suction at every anchor
    for m, c in tbl.items():
        assert 0.4 < c < 2.0
        assert c <= ba._pitot_cp(m) + 0.62
    # aft-face opening fraction moves the hypersonic plateau, subsonic less
    lo = dict(ba.cd_engines_first_table(aft_face_fraction=0.75))
    assert lo[6.0] < tbl[6.0] and abs(lo[0.0] - tbl[0.0]) < 0.10


def test_stowed_and_deployed_fin_state(veh):
    assert ba.STARBASE_V1_STOWED.cd_increment(2.0, 63.617) == 0.0
    stow = ba.DC8_STOWED.cd_increment(2.0, veh.reference_area_m2)
    assert 0.0 < stow < 0.15
    delta = ba.fin_state_delta_cd(veh.grid_fins, ref_area_m2=veh.reference_area_m2)
    # deployed minus stowed stays positive across the Mach range
    assert all(row[1] > 0.0 for row in delta)
    # stowed fairing is subtracted, not added: delta < bare cd_add
    assert delta[3][1] < veh.grid_fins.cd_add(1.0)


def test_interp_matches_propagator_semantics(veh):
    for m in (0.0, 0.5, 0.9, 1.05, 1.3, 3.3, 7.7, 12.0, 15.0):
        assert ba.interp_cd(veh.cd_mach, m) == pytest.approx(veh.cd(m))


def test_evidence_band_and_e1004_control(evidence):
    v = evidence["verdict"]
    assert v["band_impact_speed_pct_UNCHANGED"] == 12.0
    assert v["control_reproduces_e1004"] is True
    e1004 = json.loads(E1004.read_text())
    ctrl = evidence["part_c_same_state_replays"]["control_declared_starbase"]
    want = e1004["replays"]["starbase"]["twin_from_recorded_state"]["1.0"][
        "speed_pct"]
    assert abs(ctrl["speed_pct"] - want) < 0.15
    dc7 = evidence["part_c_same_state_replays"]["control_declared_dc7"]
    want7 = e1004["replays"]["dc7"]["twin_from_recorded_state"]["1.0"][
        "speed_pct"]
    assert abs(dc7["speed_pct"] - want7) < 0.15
    # Starbase v1 has no fins/legs: qualified axial == declared == control
    q = evidence["part_c_same_state_replays"]["qualified_axial_starbase_v1"]
    assert q["speed_pct"] == ctrl["speed_pct"]
    assert v["inside_band"] is False
    assert q["speed_pct"] > 12.0
    assert "genuine" in v["conclusion"]


def test_evidence_inversion_and_sync(evidence):
    inv = evidence["part_b_recorded_inversion"]
    # recorded flights invert BELOW the declared polar: no power-off build-up
    # correction closes the gap (the E11-05 finding)
    assert 0.85 < inv["starbase"]["cd_eff_mean_ratio"] < 1.0
    assert 0.85 < inv["dc7"]["cd_eff_mean_ratio"] < 1.0
    sync = evidence["part_e_sync_main_twin"]
    assert sync["interp_agreement_abs"] == 0.0
    assert abs(sync["impact_speed_delta_pct"]) < 12.0


def test_qualified_polar_main_twin_sync_coast(veh, evidence):
    """The same generated table drives both models: main propagator coast and
    the twin debris integrator from the sealed E10-07 handoff state, inside
    the 12 % model-agreement band."""
    from orbitopt.verify import independent_dynamics as idyn
    import dataclasses
    import numpy as np

    qualified = [list(r) for r in
                 evidence["part_a_component_tables"]
                 ["engines_first_stowed_qualified"]]
    veh_q = dataclasses.replace(veh, cd_mach=tuple(
        (float(m), float(c)) for m, c in qualified))
    hs = json.loads((REPO / "docs/benchmarks/e10_07_full_recovery_reflight"
                     ".json").read_text())["handoff"]["state"]
    st0 = BoosterState(float(hs["t_s"]),
                       datetime.fromisoformat(hs["epoch_utc"]), hs["frame"],
                       np.array(hs["position_m"]), np.array(hs["velocity_m_s"]),
                       float(hs["mass_kg"]))
    tr = propagate(st0, veh_q, None, st0.t_s + 900.0, events=("ground",),
                   dt_s=0.05)
    assert tr.impact is not None
    imp = idyn._debris_to_impact(st0.position_m, st0.velocity_m_s,
                                 st0.mass_kg, qualified, 0.02,
                                 diameter_m=9.0, t0_s=st0.t_s)
    delta = (tr.impact.speed_m_s / float(imp["speed_earth_m_s"]) - 1.0) * 100
    assert abs(delta) < 12.0
    assert abs(delta - evidence["part_e_sync_main_twin"]
               ["impact_speed_delta_pct"]) < 0.1


def test_sealed_dc8_booster_polar_unchanged():
    import yaml
    doc = yaml.safe_load(DC8_YAML.read_text())
    cd = doc["booster"]["cd_mach"]
    cd = cd["value"] if isinstance(cd, dict) else cd
    assert [[float(m), float(c)] for m, c in cd] == [list(r)
                                                     for r in DC8_DECLARED_CD]
