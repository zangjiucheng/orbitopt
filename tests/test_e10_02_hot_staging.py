"""E10-02 (task #299): hot staging & throttled MECO.

Gates the DC8 hot-staging timeline contract (orbitopt.flightlab.hot_staging),
its fold into the OpenTsiolkovsky config, the native rigid-body plan
validation, the JSBSim staging-config plumbing, and the committed benchmark
JSON (docs/benchmarks/e10_02_hot_staging.json).
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from orbitopt.flightlab import hot_staging
from orbitopt.flightlab.hot_staging import HotStaging, HotStagingError, load_hot_staging

REPO = Path(__file__).resolve().parents[1]
DC7 = REPO / "src/orbitopt/data/flightlab/vehicles/starship_class_rtls_dc7.yaml"
DC8 = REPO / "src/orbitopt/data/flightlab/vehicles/starship_class_rtls_dc8.yaml"
SCENARIO = REPO / "src/orbitopt/data/flightlab/scenarios/leo_starship_class_rtls_e508_dc7.yaml"
BENCH = REPO / "docs/benchmarks/e10_02_hot_staging.json"


def _doc(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class TestTimelineContract:
    def test_dc8_declares_the_four_events_in_order(self):
        hs = load_hot_staging(_doc(DC8))
        assert hs is not None
        names = [e[0] for e in hs.events()]
        assert names == ["throttle_down", "ship_ignition", "separation", "booster_meco"]
        assert hs.throttle_down_start_s == 100.0
        assert hs.ship_ignition_s == 124.0
        assert hs.separation_s == 125.0
        assert hs.booster_meco_s == 125.5
        assert hs.overlap_s == pytest.approx(1.0)
        assert hs.meco_after_separation_s == pytest.approx(0.5)

    def test_dc7_has_no_staging_block(self):
        assert load_hot_staging(_doc(DC7)) is None

    @pytest.mark.parametrize("field,value", [
        ("ship_ignition_s", 126.0),   # ignites after separation
        ("ship_ignition_s", 99.0),    # before throttle-down started
        ("booster_meco_s", 124.5),    # MECO before separation
        ("plume_axial_loss_fraction", 1.0),
        ("throttle_slew_max_fraction_per_s", 0.0),
    ])
    def test_illegal_timelines_are_rejected(self, field, value):
        doc = _doc(DC8)
        doc["staging"][field] = value
        with pytest.raises(HotStagingError):
            HotStaging.load(doc["staging"])

    def test_no_coast_window_dc8_passes_and_a_cold_gap_fails(self):
        d8 = _doc(DC8)
        hs = load_hot_staging(d8)
        s1 = d8["stages"][0]["engine"]["thrust_vac_n"]
        s2 = d8["stages"][1]["engine"]["thrust_vac_n"]
        assert hs.no_coast_window(s1, s2)
        # the DC7 cold timeline -- ship lights AT separation, after the 123 s
        # booster cutoff: the 3-s unpowered coast must fail this check
        d7 = _doc(DC7)
        cold = replace(hs, ship_ignition_s=126.0, separation_s=126.001,
                       booster_meco_s=126.002)
        gap = [list(p) for p in d7["stages"][0]["engine"]["thrust_vac_n"]] + [[123.0, 0.0]]
        assert not cold.no_coast_window(gap, d7["stages"][1]["engine"]["thrust_vac_n"])


class TestThrottleShape:
    def test_dc8_curve_throttles_without_a_step(self):
        doc = _doc(DC8)
        hs = load_hot_staging(doc)
        eng = doc["stages"][0]["engine"]
        rep = hs.check_stage1_curve(eng["thrust_vac_n"], eng["burn_end_s"])
        assert rep.max_step_fraction == pytest.approx(0.0)
        assert rep.max_slew_fraction_per_s <= hs.throttle_slew_max_fraction_per_s + 1e-9
        assert rep.floor_fraction == pytest.approx(0.40, abs=1e-3)
        assert rep.cut_level_n == pytest.approx(0.40 * 74926500.0, rel=1e-3)
        assert rep.violations(hs.thrust_step_max_fraction,
                              hs.throttle_slew_max_fraction_per_s,
                              hs.throttle_floor_fraction) == ()

    def test_dc7_curve_violates_the_declared_limits(self):
        hs = load_hot_staging(_doc(DC8))
        doc7 = _doc(DC7)
        eng = doc7["stages"][0]["engine"]
        rep = hs.check_stage1_curve(eng["thrust_vac_n"], eng["burn_end_s"])
        # 74.93 -> 62 MN in one 1-s jump at 101 s and a 77 %-of-peak cutoff:
        # the acceptance's premise is that DC7 fails exactly these gates
        assert rep.violations(hs.thrust_step_max_fraction,
                              hs.throttle_slew_max_fraction_per_s) != ()

    def test_composite_curve_and_overlap_burn(self):
        doc = _doc(DC8)
        hs = load_hot_staging(doc)
        s1 = doc["stages"][0]["engine"]["thrust_vac_n"]
        s2 = doc["stages"][1]["engine"]["thrust_vac_n"]
        isp2 = doc["stages"][1]["engine"]["isp_vac_s"]
        curve, burn = hs.composite_stack_curve(s1, s2, isp2)
        assert curve[0][0] == 0.0
        assert curve[-1] == [hs.separation_s, 0.0]
        mid = hot_staging.interp_thrust(curve, 124.5)
        want = (hot_staging.interp_thrust(s1, 124.5)
                + (1.0 - hs.plume_axial_loss_fraction)
                * hot_staging.interp_thrust(s2, 124.5 - hs.ship_ignition_s))
        assert mid == pytest.approx(want, rel=1e-9)
        assert burn == pytest.approx(4123.6, abs=5.0)
        assert hs.ship_burn_kg(s2, isp2) == pytest.approx(burn, rel=1e-9)


class TestOpenTsiolkovskyFold:
    def test_dc8_config_fold(self):
        from orbitopt.flightlab.adapters.opentsiolkovsky import Vehicle, build_config
        from orbitopt.flightlab.contract import load_target
        v = Vehicle.load(DC8)
        assert v.staging is not None
        cfg = build_config(load_target(SCENARIO), v)
        s1, s2 = cfg["stages"]
        assert s1["separation_time_s"] == 125.0
        assert s1["engine"]["burn_end_s"] == 125.0
        assert s1["engine"]["thrust_vac_n"][-1] == [125.0, 0.0]
        # the ship's own clock runs one second early in the OT model
        assert s2["engine"]["burn_end_s"] == pytest.approx(387.663)
        assert s2["mass_initial_kg"] == pytest.approx(
            _doc(DC8)["stages"][1]["mass_initial_kg"] - 4123.6, abs=5.0)

    def test_dc7_config_is_untouched(self):
        from orbitopt.flightlab.adapters.opentsiolkovsky import Vehicle, build_config
        from orbitopt.flightlab.contract import load_target
        v = Vehicle.load(DC7)
        assert v.staging is None
        cfg = build_config(load_target(SCENARIO), v)
        doc7 = _doc(DC7)
        assert cfg["stages"][0]["separation_time_s"] == \
            doc7["stages"][0]["separation_time_s"]
        assert cfg["stages"][0]["engine"]["thrust_vac_n"] == \
            doc7["stages"][0]["engine"]["thrust_vac_n"]


class TestNativeRigidBodyPlan:
    def test_plan_rejects_cold_orderings(self):
        from orbitopt.flightlab.sixdof.two_stage import HotStagingPlan
        with pytest.raises(ValueError):
            HotStagingPlan(ship_ignition_t_s=126.0, separation_t_s=125.0)
        with pytest.raises(ValueError):
            HotStagingPlan(ship_ignition_t_s=0.0, separation_t_s=125.0)
        with pytest.raises(ValueError):
            HotStagingPlan(ship_ignition_t_s=124.0, separation_t_s=125.0, throttle=1.5)
        with pytest.raises(ValueError):
            HotStagingPlan(ship_ignition_t_s=124.0, separation_t_s=125.0,
                           plume_axial_loss_fraction=1.0)

    def test_scenario_requires_zero_ignition_delay(self):
        from orbitopt.flightlab.sixdof.two_stage import (
            HotStagingPlan, f06_two_stage_scenario)
        base = f06_two_stage_scenario()
        hot = HotStagingPlan(ship_ignition_t_s=base.separation_t_s - 1.0,
                             separation_t_s=base.separation_t_s)
        with pytest.raises(ValueError, match="already alight"):
            replace(base, hot=hot)
        ok = replace(base, hot=hot,
                     ignition=replace(base.ignition, ignition_delay_s=0.0))
        assert ok.hot.overlap_s == pytest.approx(1.0)


class TestJsbsimStagingPlumbing:
    def test_hot_staging_config_validation(self):
        pytest.importorskip("jsbsim")
        from orbitopt.flightlab.adapters.sixdof import StagingConfig
        cfg = StagingConfig(at_t_s=125.0, upper_ignition_advance_s=1.0,
                            upper_mated_burn_kg=4123.6)
        assert cfg.upper_ignition_advance_s == 1.0
        with pytest.raises(ValueError):
            StagingConfig(at_t_s=125.0, upper_ignition_advance_s=0.0,
                          upper_mated_burn_kg=4123.6)
        with pytest.raises(ValueError):
            StagingConfig(at_t_s=125.0, upper_ignition_advance_s=125.0)
        with pytest.raises(ValueError):
            StagingConfig(at_t_s=125.0, upper_mated_burn_kg=-1.0)


@pytest.fixture(scope="module")
def bench():
    if not BENCH.exists():
        pytest.skip("regenerate with benchmarks/bench_e10_02_hot_staging.py")
    return json.loads(BENCH.read_text(encoding="utf-8"))


class TestBenchmarkArtifact:

    def test_acceptance_flags(self, bench):
        assert bench["schema"] == "e10-02-hot-staging-1.0"
        assert all(bench["acceptance"].values()), bench["acceptance"]
        assert bench["event_order_flown_ok"]

    def test_throttle_shape_cold_vs_hot(self, bench):
        a7 = bench["stage1_thrust_steps"]["dc7"]
        a8 = bench["stage1_thrust_steps"]["dc8"]
        assert a8["max_step_fraction"] == pytest.approx(0.0)
        assert a8["max_slew_fraction_of_peak_per_s"] <= 0.05
        assert a8["cutoff_step_fraction_of_peak"] < a7["cutoff_step_fraction_of_peak"]
        assert a8["cutoff_step_fraction_of_peak"] == pytest.approx(0.40, abs=1e-3)

    def test_cold_vs_hot_comparison(self, bench):
        c = bench["comparison_cold_vs_hot"]
        assert c["ascent_checks_pass"] == {"dc7": True, "dc8": True}
        # a hotter stack leaves separation faster, having spent more of the
        # booster's propellant (the ship burns 1 s of mated stack time)
        assert c["separation_speed_m_s"]["dc8"] > c["separation_speed_m_s"]["dc7"]
        assert (c["stage1_propellant_at_separation_kg"]["dc8"]
                < c["stage1_propellant_at_separation_kg"]["dc7"])
        for k in ("dc7", "dc8"):
            assert c["ship_propellant_margin_kg"][k] > 100000.0
            assert c["insertion_periapsis_altitude_m"][k] >= 230000.0
            assert abs(c["insertion_apoapsis_altitude_m"][k] - 250000.0) <= 60000.0

    def test_e805_block_covers_the_hot_timeline(self, bench):
        e = bench["e805_dc8"]
        assert e is not None, "regenerate with the E8-05 block (no --skip-e805)"
        ids = {c["id"] for c in e["checks"]}
        assert "ascent.separation.state_position" in ids
        assert "ascent.meco.state_position" in ids
        assert e["n_checks"] >= 28
        # the point-mass twin and the OT fold agree on position/attitude
        # bookkeeping; every open band is a consequence of the two declared
        # representation splits (OT charges the mated ship thrust against the
        # stage-1 tank at stage-1 Isp and drops the booster's post-separation
        # floor burn; the twin burns it from the ship tank at flow-blended Isp)
        # accumulating over the 390-s stage-2 arc, or of E8-05's budget-residual
        # discipline. E10-04's event-aware raw interpolation removed the
        # chord-across-cutoff artifact here too (SECO velocity 45.2 -> 37.0 m/s,
        # SMA 50.7 -> 34.5 km, separation position back inside its band, SECO
        # mass -2.9 -> -2.3 t); what remains is the hot-timeline split itself,
        # larger on this timeline because the stack flies powered through
        # MECO/separation. DC7's own sealed report closes fully after E10-04.
        assert set(e["failing"]) <= {
            "ascent.meco.residual_position", "ascent.meco.residual_velocity",
            "ascent.meco.state_velocity",
            "ascent.separation.state_position", "ascent.separation.state_velocity",
            "ascent.separation.debris_mass",
            "ascent.seco.residual_position", "ascent.seco.state_position",
            "ascent.seco.state_velocity", "ascent.seco.mass",
            "ascent.seco.orbit_semi_major_axis",
            "ascent.seco.orbit_eccentricity"}, e["failing"]
        assert "blended" in e["note"]
