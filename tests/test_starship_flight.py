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
    assert labels[0] == "Ascent (not modelled)" and "Entry (not modelled)" in labels
    events = {e["label"] for e in scene["timeline"]["events"]}
    assert {"SpaceX-announced entry", "Return pass (predicted)"} <= events
    assert "prediction, not a replay" in scene["subtitle"]
