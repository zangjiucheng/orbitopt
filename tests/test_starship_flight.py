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
def test_return_solution_does_not_depend_on_the_bank_guess(prediction, solution):
    from orbitopt.viz.starship_flight import solve_return

    other = solve_return(prediction, bank_guess_deg=20.0)
    assert other.burn_time_s == pytest.approx(solution.burn_time_s, abs=2.0)
    assert other.bank_deg == pytest.approx(solution.bank_deg, abs=0.2)
    assert other.dv_normal_m_s == pytest.approx(solution.dv_normal_m_s, abs=1.0)


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
