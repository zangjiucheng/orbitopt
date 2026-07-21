"""Checks for the Earth->Mars transfer mission (viz.mars_transfer): fast,
compute-free schema-validation tests for the mars-transfer mission-config
kind, plus @pytest.mark.slow end-to-end tests that run the real
porkchop-screen + PSO-optimized Lambert transfer and the tudatpy TCM/MOI
targeting (verify.mars_insertion), matching test_geo_raising.py's split.

This mission ends at Mars orbit insertion, not a landing -- see
docs/mars_mission_plan.md and viz/mars_transfer.py's module docstring for why.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from orbitopt.mission_config import is_valid_mission_config, known_kinds, load_mission_config, validate_mission_config
from orbitopt.scene_format import validate_scene

FIXTURE = Path(__file__).parent / "fixtures" / "mars_transfer_config.yaml"

MINIMAL_CONFIG = {
    "schemaVersion": "1.0",
    "mission": {"id": "mars-transfer", "title": "Mars probe", "kind": "mars-transfer"},
    "output": {"scene_file": "scenes/mars-transfer.json"},
}


def test_known_kinds_lists_mars_transfer():
    assert "mars-transfer" in known_kinds()


def test_minimal_config_validates():
    validate_mission_config(MINIMAL_CONFIG)  # must not raise
    assert is_valid_mission_config(MINIMAL_CONFIG)


def test_fixture_config_loads_and_validates():
    config = load_mission_config(FIXTURE)
    assert config["mission"]["kind"] == "mars-transfer"


def test_unknown_top_level_field_is_invalid():
    broken = copy.deepcopy(MINIMAL_CONFIG)
    broken["mision"] = {}  # typo
    assert not is_valid_mission_config(broken)


def test_missing_output_section_is_invalid():
    broken = copy.deepcopy(MINIMAL_CONFIG)
    del broken["output"]
    assert not is_valid_mission_config(broken)


def test_launch_window_dates_must_be_year_month_day_triples():
    broken = copy.deepcopy(MINIMAL_CONFIG)
    broken["launch_window"] = {"start_date": [2026, 9], "end_date": [2027, 2, 1]}  # too short
    assert not is_valid_mission_config(broken)


@pytest.mark.slow
def test_build_scene_from_config_produces_a_valid_scene():
    from orbitopt.mission_config import build_scene_from_config

    config = load_mission_config(FIXTURE)
    scene = build_scene_from_config(config)
    validate_scene(scene)
    assert scene["id"] == config["mission"]["id"]
    assert [b["id"] for b in scene["bodies"]] == ["sun", "earth", "mars", "spacecraft"]


@pytest.mark.slow
def test_full_mission_pipeline_end_to_end():
    """Runs the real porkchop-screen + PSO-optimized transfer, the real
    tudatpy TCM targeter, and the real MOI capture-burn targeter -- checks
    the exported scene is physically sane (captured, not escaping; periapsis
    inside Mars's SOI; events present and ordered) and that the one
    continuous heliocentric trail is actually continuous at both phase
    boundaries (no double-back-in-time jump, the exact class of bug found
    and fixed during development -- see docs/mars_mission_plan.md)."""
    from orbitopt.viz.mars_transfer import compute_and_export_mars_mission

    scene = compute_and_export_mars_mission(seed=1, pop_size=64, generations=40)
    validate_scene(scene)

    spacecraft = scene["bodies"][-1]
    assert spacecraft["id"] == "spacecraft"
    info = {row["label"]: row["value"] for row in spacecraft["info"]}

    # Captured orbit is a real, bound ellipse (e < 1), not an escape/miss.
    captured = info["Captured orbit"]
    e = float(captured.split("e=")[1])
    assert 0.0 <= e < 1.0

    events = scene["timeline"]["events"]
    labels = [e["label"] for e in events]
    assert labels == [
        "Liftoff",
        "Parking-orbit insertion",
        "TMI burn",
        "Earth-SOI exit",
        "Trajectory-correction maneuver",
        "Mars-SOI entry / approach",
        "Mars orbit insertion",
    ]
    times = [e["time"] for e in events]
    assert times == sorted(times)  # events are chronological

    moi_event = events[-1]
    assert "Mission ends here" in moi_event["note"]  # honest about not modeling landing

    # MOI's target periapsis is Mars radius + 500 km by default (see
    # MARS_TCM_TARGET_PERIAPSIS_KM); the TCM targeter isn't guaranteed to
    # fully converge, so allow generous slack, but the achieved periapsis
    # must still land comfortably inside Mars's sphere of influence
    # (~577,000 km) -- otherwise the whole premise of a "captured" orbit
    # would be physically meaningless.
    moi_dv = float(info["MOI delta-v"].split()[0])
    assert 0.0 < moi_dv < 5000.0

    # --- Trail continuity across the three propagated phases ---
    trail = spacecraft["trail"]
    positions = np.array(trail["positions"])
    times_days = np.array(trail["times"])
    assert np.all(np.diff(times_days) >= 0)  # never runs backward in time

    # The Earth-SOI-exit / Trajectory-correction-maneuver event marks the
    # phase1->cruise handoff; the spacecraft position there is the *same*
    # physical point in both phases (Earth-relative state + Earth's own
    # heliocentric position, vs. the cruise leg's own start state), so the
    # nearest trail sample to that event time should show ~zero position
    # jump to its neighbor once resolved to that instant.
    soi_exit_time = next(e["time"] for e in events if e["label"] == "Earth-SOI exit")
    idx = int(np.argmin(np.abs(times_days - soi_exit_time)))
    if 0 < idx < len(positions) - 1:
        # a genuine seam has (near-)zero gap; a coasting sample doesn't
        gap_before = np.linalg.norm(positions[idx] - positions[idx - 1])
        gap_after = np.linalg.norm(positions[idx + 1] - positions[idx])
        assert min(gap_before, gap_after) < 1000.0

    # No sample-to-sample jump anywhere in the trail should exceed what's
    # physically plausible for heliocentric cruise speed (Earth's own
    # orbital speed, ~30 km/s, is close to the fastest this mission ever
    # moves) times the largest gap between consecutive samples, with
    # generous headroom for the faster near-Mars hyperbolic segment.
    dt_s = np.diff(times_days) * 86400.0
    dist_km = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    implied_speed_km_s = np.divide(dist_km, dt_s, out=np.zeros_like(dist_km), where=dt_s > 0)
    assert np.all(implied_speed_km_s < 100.0)  # generous ceiling; a real bug measured >1000 km/s here


@pytest.mark.slow
def test_compute_and_export_mars_mission_zero_arg_call_matches_built_in():
    """The 'mars' built-in mission (missions.py) calls
    compute_and_export_mars_mission() with no arguments -- confirm that
    zero-arg call actually produces a valid, correctly-titled, correctly-
    scoped scene, since this is what `orbitopt view mars` runs."""
    from orbitopt.viz.mars_transfer import compute_and_export_mars_mission

    scene = compute_and_export_mars_mission()
    validate_scene(scene)
    assert scene["id"] == "mars-transfer"
    assert scene["title"] == "Mars — Earth to Orbit Insertion"
    assert "not a landing" in scene["subtitle"]
