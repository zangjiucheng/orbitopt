"""Checks for the optional ``launch`` section of mars-transfer mission configs
and orbitopt.launch_injection (site + rocket -> parking orbit -> verified
TMI), the launch-to-injection step shared by the config pipeline and
examples/14_mars2020_replay.py. Needs numpy/scipy only (no pykep/tudatpy),
except the adapter test, which skips without pykep."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest

from orbitopt.injection import (InjectionTarget, asymptote_unit_vector, equatorial_to_ecliptic,
                                plan_departure, verify_injection)
from orbitopt.launch_injection import LaunchSpec, launch_to_injection
from orbitopt.launch_windows import as_mjd2000, asymptote_launch_windows
from orbitopt.mission_config import is_valid_mission_config, load_mission_config

REPLAY = Path(__file__).parents[1] / "examples" / "mars2020_replay.yaml"
MINIMAL = {
    "schemaVersion": "1.0",
    "mission": {"id": "m", "title": "M", "kind": "mars-transfer"},
    "output": {"scene_file": "out/m.json"},
}
# A Mars-2020-like departure asymptote (C3, RLA, DLA) as an ecliptic v_inf [m/s].
C3, RLA, DLA = 13.0, 160.0, 24.0
V_INF_ECL = equatorial_to_ecliptic(asymptote_unit_vector(RLA, DLA) * math.sqrt(C3) * 1000.0)


def _with_launch(launch):
    cfg = copy.deepcopy(MINIMAL)
    cfg["launch"] = launch
    return cfg


def test_replay_config_has_a_valid_launch_section():
    config = load_mission_config(REPLAY)
    spec = LaunchSpec.from_config(config["launch"])
    assert (spec.site, spec.vehicle, spec.payload_kg) == ("ksc", "atlas-v-551", 4037.0)
    assert spec.pinned and spec.tof_days == pytest.approx(203.4)
    # Epochs go through the unified time API (MJD2000 UTC).
    assert spec.departure_mjd2000 == as_mjd2000("2020-07-30T00:00:00Z")
    assert spec.preferred_liftoff_mjd2000 == as_mjd2000("2020-07-30T11:50:00Z")


@pytest.mark.parametrize("launch, ok", [
    ({"site": "ksc"}, True),
    ({"site": "ksc", "vehicle": "falcon-heavy", "payload_kg": 2000, "constraint_mode": "penalty"}, True),
    ({"vehicle": "falcon-heavy"}, False),  # site required
    ({"site": "ksc", "rocket": "falcon-heavy"}, False),  # typo'd key
    ({"site": "ksc", "departure": {"utc": "2020-07-30"}}, False),  # tof_days required
    ({"site": "ksc", "parking_orbit": {"altitude_km": -1}}, False),
    ({"site": "ksc", "constraint_mode": "soft"}, False),
])
def test_launch_section_schema(launch, ok):
    assert is_valid_mission_config(_with_launch(launch)) is ok


def test_absent_or_disabled_launch_is_none():
    assert LaunchSpec.from_config(None) is None
    assert LaunchSpec.from_config({}) is None
    assert LaunchSpec.from_config({"site": "ksc", "enabled": False}) is None


def test_parking_altitude_falls_back_to_top_level():
    assert LaunchSpec.from_config({"site": "ksc"}, default_altitude_km=300.0).parking_altitude_km == 300.0
    spec = LaunchSpec.from_config({"site": "ksc", "parking_orbit": {"altitude_km": 200.0}}, default_altitude_km=300.0)
    assert spec.parking_altitude_km == 200.0
    assert not spec.pinned


def test_constraints_carry_vehicle_payload_site_and_mode():
    spec = LaunchSpec("ksc", "atlas-v-551", 4037.0, constraint_mode="penalty")
    lc = spec.constraints(frame="equatorial")
    d = lc.describe()
    assert (d["vehicle"], d["payload_kg"], d["mode"], lc.frame) == ("atlas-v-551", 4037.0, "penalty", "equatorial")
    assert d["site"] is not None


def test_launch_to_injection_matches_the_manual_chain(tmp_path):
    """The shared helper is exactly example 14's steps 4-5: windows on the
    departure day, the one nearest the preferred lift-off, plan_departure,
    verify_injection with the vehicle/site constraints."""
    spec = LaunchSpec.from_config(load_mission_config(REPLAY)["launch"])
    t0 = spec.departure_mjd2000
    out = launch_to_injection(spec, V_INF_ECL, t0 + 0.3, report_path=str(tmp_path / "inj.json"))

    windows = asymptote_launch_windows(spec.site, out.rla_deg, out.dla_deg, t0, 1.0, None, 30.0)
    nearest = min(windows, key=lambda w: abs(w.optimal_mjd2000 - spec.preferred_liftoff_mjd2000))
    plan = plan_departure(spec.site, InjectionTarget.from_c3_rla_dla(C3, RLA, DLA, name="TMI"),
                          liftoff=nearest.optimal_mjd2000, altitude_km=185.0, min_coast_s=600.0)
    v = verify_injection(plan, launch_constraints=spec.constraints(frame="equatorial"))

    assert (out.rla_deg, out.dla_deg, out.c3_km2_s2) == pytest.approx((RLA, DLA, C3))
    assert out.liftoff_mjd2000 == pytest.approx(nearest.optimal_mjd2000, abs=1e-12)
    np.testing.assert_allclose(out.verification.delta_v_m_s, v.delta_v_m_s, atol=1e-9)
    assert out.verification.passed == v.passed
    assert out.verification.passed
    assert out.report["kind"] == "launch-injection"
    assert (tmp_path / "inj.json").exists()
    s = out.summary()
    assert s["liftoff_utc"].startswith("2020-07-30T") and s["passed"]


def test_preferred_liftoff_selects_nearest_window():
    first = launch_to_injection(LaunchSpec("ksc", None, window_days=2.0), V_INF_ECL, 7516.0)
    assert len(first.windows) >= 2
    assert first.window is first.windows[0]
    later = first.windows[-1].optimal_mjd2000
    pref = launch_to_injection(LaunchSpec("ksc", None, window_days=2.0, preferred_liftoff_mjd2000=later),
                               V_INF_ECL, 7516.0)
    assert pref.liftoff_mjd2000 == pytest.approx(later, abs=1e-12)


def test_build_from_config_only_enables_launch_when_configured(monkeypatch):
    pytest.importorskip("pykep")
    import orbitopt.viz.mars_transfer as mt

    seen = []

    def fake_build(recorder=None, **kwargs):  # the pipeline always hands a recorder through
        seen.append(kwargs)
        return {"id": "", "title": ""}

    monkeypatch.setattr(mt, "_build_mars_mission", fake_build)
    mt.build_from_config(copy.deepcopy(MINIMAL))
    mt.build_from_config(_with_launch({"site": "ksc", "enabled": False}))
    mt.build_from_config(load_mission_config(REPLAY))
    assert seen[0] == {} and seen[1] == {}
    assert isinstance(seen[2]["launch"], LaunchSpec) and seen[2]["launch"].pinned
