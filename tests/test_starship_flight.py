"""Starship Flight 15 return-pass prediction."""
from __future__ import annotations

import pytest

pytest.importorskip("tudatpy")

from orbitopt.launch_sites import get_launch_site  # noqa: E402
from orbitopt.viz.starship_flight import (  # noqa: E402
    ANNOUNCED_ENTRY_S,
    build_flight15_scene,
    predict_return,
    southeast_azimuth,
)


def test_azimuth_for_flight14_inclination_is_inside_the_starbase_corridor():
    site = get_launch_site("starbase")
    az = southeast_azimuth(site, 30.5)
    assert az == pytest.approx(106.54, abs=0.01)  # cos i = cos(phi) sin(Az), south-east branch
    lo, hi = site.azimuth_corridors_deg[0]
    assert lo <= az <= hi
    with pytest.raises(ValueError, match="not reachable"):
        southeast_azimuth(site, 20.0)  # below the site's latitude


@pytest.fixture(scope="module")
def prediction():
    return predict_return()


@pytest.mark.slow
def test_return_pass_follows_the_announced_entry(prediction):
    ret = prediction.return_pass
    assert ret is not None
    # The pass that can bring the ship home comes shortly after SpaceX's
    # announced entry -- a number this model never sees.
    assert 0.0 < ret.time_s - ANNOUNCED_ENTRY_S < 30 * 60
    assert 14.0 < prediction.orbits_by_announced_entry < 15.0  # SpaceX: "approximately 14 orbits"
    assert ret.north_of_site and 200.0 < ret.distance_km < 500.0
    assert 250.0 < prediction.alt_km.min() and prediction.alt_km.max() < 300.0


@pytest.mark.slow
def test_ground_track_does_not_depend_on_the_launch_time(prediction):
    other = predict_return("2026-11-13T22:45:00Z")  # the other published launch time
    assert other.return_pass.time_s == pytest.approx(prediction.return_pass.time_s, abs=1.0)
    assert other.return_pass.distance_km == pytest.approx(prediction.return_pass.distance_km, abs=1.0)


@pytest.mark.slow
def test_scene_validates_and_labels_what_is_not_modelled(prediction):
    from orbitopt.scene_format import validate_scene

    scene = build_flight15_scene(prediction)
    validate_scene(scene)
    kinds = {b["id"]: b["kind"] for b in scene["bodies"]}
    assert kinds == {"earth": "planet", "starbase": "site", "spacecraft": "spacecraft"}
    labels = [p["label"] for p in scene["timeline"]["phases"]]
    assert labels == ["Ascent (not modelled)", "Orbital coast", "Orbit continued (hypothetical)"]
    events = {e["label"] for e in scene["timeline"]["events"]}
    assert {"SpaceX-announced entry", "Return pass (predicted)"} <= events
    assert "prediction, not a replay" in scene["subtitle"]
    assert scene["initialView"]["focusBodyId"] == "spacecraft"


@pytest.fixture(scope="module")
def solution(prediction):
    from orbitopt.viz.starship_flight import solve_return

    return solve_return(prediction)


@pytest.mark.slow
def test_return_is_solved_to_the_tower(prediction, solution):
    assert solution.converged and solution.miss_m < 100.0
    # The order of a return: burn, entry interface, landing burn, catch.
    assert solution.burn_time_s < solution.entry_interface_s < solution.landing_start_s < solution.landing_end_s
    # The predicted entry interface lands near SpaceX's announced entry -- a
    # number the return solve never sees.
    assert abs(solution.entry_interface_s - ANNOUNCED_ENTRY_S) < 10 * 60
    assert 10.0 < solution.landing_end_s - solution.landing_start_s < 60.0
    assert 1.0 < solution.entry.aero_g.max() < 4.0
    # With Flight 14's inclination the entry cross-range falls short, so part
    # of the deorbit burn steers out of plane.
    assert abs(solution.dv_normal_m_s) > 1.0


@pytest.mark.slow
def test_return_bank_is_the_best_reach(prediction, solution):
    """Flight 14's inclination leaves Starbase out of constant-bank reach, so
    the solved bank is the one of maximum cross-range reach: a degree either
    way (burn time re-solved for downrange) lands farther off to the side.
    The search is over the bank itself -- no starting guess to stall on (the
    old 2-D solve stopped short on x86-64 macOS, 48 s off in burn time)."""
    from orbitopt.viz import starship_flight as sf

    assert solution.dv_normal_m_s != 0.0  # stage 2 ran: the bank alone can't reach

    def cross_miss_km(bank):
        def downrange(dt_min):
            burn_t = prediction.return_pass.time_s - 2500.0 + 60.0 * dt_min
            traj, _ = sf._fly_from(prediction, burn_t, bank, 0.0)
            return sf._miss_vector_km(prediction, burn_t, traj)

        from scipy.optimize import brentq

        dt = brentq(lambda x: downrange(x)[0], -15.0, 15.0, xtol=1e-5)
        return abs(float(downrange(dt)[1]))

    at = cross_miss_km(solution.bank_deg)
    assert cross_miss_km(solution.bank_deg - 1.0) > at
    assert cross_miss_km(solution.bank_deg + 1.0) > at


@pytest.mark.slow
def test_full_scene_shows_the_raptor_burns(prediction, solution):
    from orbitopt.scene_format import validate_scene

    scene = build_flight15_scene(prediction, solution=solution)
    validate_scene(scene)
    events = {e["label"]: e for e in scene["timeline"]["events"]}
    assert "vector" in events["Deorbit burn (Raptor)"] and "vector" in events["Flip + landing burn (Raptor)"]
    assert "Tower catch (predicted)" in events
    labels = [p["label"] for p in scene["timeline"]["phases"]]
    assert "Entry (3-DOF, assumed aero)" in labels and "Landing burn (Raptor)" in labels


@pytest.mark.slow
def test_full_scene_has_the_ascent_and_the_booster(prediction, solution):
    import numpy as np

    from orbitopt.launch_sites import get_launch_site
    from orbitopt.scene_format import validate_scene
    from orbitopt.viz import starship_ascent as sa
    from orbitopt.viz.starship_flight import SECO_S

    ascent = sa.build_ascent(get_launch_site("starbase"), prediction.azimuth_deg, prediction.liftoff_mjd2000,
                             SECO_S, prediction.states_eq[0])
    scene = build_flight15_scene(prediction, solution=solution, ascent=ascent)
    validate_scene(scene)
    ids = [b["id"] for b in scene["bodies"]]
    assert ids.index("spacecraft") < ids.index("booster")  # the ship stays "the" spacecraft
    ship = next(b for b in scene["bodies"] if b["id"] == "spacecraft")
    assert ship["trail"]["times"][0] == 0.0
    assert np.linalg.norm(ship["trail"]["positions"][0]) < 6380.0  # on the pad at lift-off, not in orbit
    events = {e["label"]: e for e in scene["timeline"]["events"]}
    assert {"MECO", "Hot staging", "Booster catch (predicted)"} <= set(events)
    assert "vector" in events["Boostback burn (Raptor)"] and "vector" in events["Booster landing burn (Raptor)"]
    earth = next(b for b in scene["bodies"] if b["id"] == "earth")
    assert earth["flattening"] == pytest.approx(1.0 / 298.257223563)


@pytest.mark.slow
def test_return_verification_reports(prediction, solution):
    """The four reports validate, and say what they found: the numerics hold,
    the loads and timeline are fine, and only a 10% lift shortfall cannot be
    recovered -- the nominal already flies at the entry's full cross-range."""
    from orbitopt.injection import DEFAULT_ATOL, DEFAULT_RTOL
    from orbitopt.verify.report import publish_to_scene, validate_report
    from orbitopt.viz.starship_verify import verify_return

    tight = predict_return(rtol=DEFAULT_RTOL * 0.01, atol=DEFAULT_ATOL * 0.01)
    reports = verify_return(prediction, tight, solution, inclinations={"nominal": 29.97, "lift_low": None})
    for r in reports:
        validate_report(r)
    result = {r["kind"]: r["status"]["result"] for r in reports}
    assert result == {"starship_return_refly": "passed", "starship_return_dispersions": "failed",
                      "starship_entry_limits": "passed", "starship_timeline_consistency": "passed"}
    failed = [k for k, ok in reports[1]["status"]["checks"].items() if not ok]
    assert failed == ["lift_low.recoverable"]
    lift_high = next(m for m in reports[1]["targets"]["metrics"] if m["name"] == "Lift-to-drag +10%")
    assert lift_high["ok"] and "reversal" in lift_high["value"]  # overshoot fixed by one bank reversal

    info = {m["name"]: m for m in reports[1]["targets"]["metrics"] if m["ok"] is None}
    assert info["Entry alone reaches Starbase"]["value"] == "inclination <= 29.97 deg"
    assert "not in the searched range" in info["... also with lift-to-drag -10%"]["value"]
    assert reports[1]["status"]["checks"].keys() == {f"{c}.recoverable" for c in (
        "density_low", "density_high", "burn_late", "burn_early", "dv_high", "dv_low", "lift_low", "lift_high")}

    scene = publish_to_scene(build_flight15_scene(prediction, solution=solution), reports)
    assert scene["verification"]["result"] == "failed" and len(scene["verification"]["reports"]) == 4


@pytest.mark.slow
def test_entry_alone_reaches_starbase_below_about_30_deg(prediction):
    """Flight 14's 30.5 deg needs an out-of-plane burn; somewhat lower the bank
    alone suffices (29.5 deg: bank ~34 deg, no burn; 30.0: still ~3 m/s)."""
    from orbitopt.viz.starship_verify import entry_reach_inclination

    inc = entry_reach_inclination(prediction, xtol_deg=0.1)
    assert inc is not None and 29.5 < inc < 30.1
