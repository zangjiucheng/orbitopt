"""KSP-style flight-data labels: height, speed, delta-v."""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("PySide6")

from orbitopt.viz.flight_labels import label_text, labelled_bodies, velocity_km_s  # noqa: E402

MU = 398_600.4418  # km^3/s^2
R_EQ = 6378.137
R_ORBIT = R_EQ + 275.0
V_CIRC = math.sqrt(MU / R_ORBIT)


def _scene():
    days = np.linspace(0.0, 0.1, 2001)
    w = V_CIRC / R_ORBIT * 86400.0  # rad/day
    ship = np.column_stack([R_ORBIT * np.cos(w * days), R_ORBIT * np.sin(w * days), np.zeros_like(days)])
    # A body standing on the equator turns with the Earth.
    spin = 2.0 * math.pi / (23.9345 / 24.0) * days
    ground = np.column_stack([R_EQ * np.cos(spin), R_EQ * np.sin(spin), np.zeros_like(days)])
    return {
        "id": "l", "title": "L", "distanceUnit": "km", "centralBodyId": "earth",
        "bodies": [
            {"id": "earth", "name": "Earth", "color": "#4c8eff", "kind": "planet", "radiusDisplay": 4.6,
             "position": [0, 0, 0], "radius": R_EQ, "flattening": 1.0 / 298.257223563, "rotationPeriodHours": 23.9345},
            {"id": "ship", "name": "Ship", "color": "#e8a23e", "kind": "spacecraft", "radiusDisplay": 3.0,
             "trail": {"times": days.tolist(), "positions": ship.tolist()}},
            {"id": "booster", "name": "Booster", "color": "#c4c9d4", "kind": "spacecraft", "radiusDisplay": 3.0,
             "trail": {"times": days.tolist(), "positions": ground.tolist()}},
        ],
        "timeline": {"min": 0.0, "max": 0.1, "unitLabel": "days", "events": [
            {"label": "Burn A", "time": 0.02, "vector": {"position": [0, 0, 0], "deltaV": [30.0, 40.0, 0.0]}},
            {"label": "Boostback", "time": 0.03, "bodyId": "booster",
             "vector": {"position": [0, 0, 0], "deltaV": [0.0, 1000.0, 0.0]}},
            {"label": "Burn B", "time": 0.06, "vector": {"position": [0, 0, 0], "deltaV": [0.0, 0.0, 20.0]}},
        ]},
    }


def _lookup(scene, body_id):
    body = next(b for b in scene["bodies"] if b["id"] == body_id)
    times = np.asarray(body["trail"]["times"])
    pos = np.asarray(body["trail"]["positions"])
    return lambda t: np.array([np.interp(t, times, pos[:, k]) for k in range(3)])


def test_height_speed_and_delta_v_tally():
    scene = _scene()
    ship = _lookup(scene, "ship")
    assert np.linalg.norm(velocity_km_s(scene, ship, 0.05)) == pytest.approx(V_CIRC, rel=1e-4)
    text = label_text(scene, "ship", 0.05, ship)
    lines = text.split("\n")
    assert lines[0] == "Ship"
    assert lines[1] == "H 275.0 km"                  # on the equator: no flattening correction
    assert lines[2] == f"V {V_CIRC:.2f} km/s"         # no surface speed up at 275 km
    assert lines[3] == "Δv 50 m/s"                   # Burn A only -- not B (later), not the booster's
    events = scene["timeline"]["events"]
    near_b = label_text(scene, "ship", 0.059, ship, {id(events[2]): 0.005})
    assert near_b.split("\n")[-1] == "» Burn B 20 m/s"


def test_burns_count_for_their_owner_and_surface_speed_near_the_ground():
    scene = _scene()
    booster = _lookup(scene, "booster")
    text = label_text(scene, "booster", 0.05, booster)
    assert "Δv 1,000 m/s" in text                     # the boostback is the booster's
    # Standing on the equator: inertial speed is the Earth's rotation
    # (~465 m/s), the surface-relative speed ~0.
    assert "V 465 m/s · surf 0 m/s" in text
    assert labelled_bodies(scene) == ["ship", "booster"]


def test_no_delta_v_line_before_the_first_burn():
    scene = _scene()
    text = label_text(scene, "ship", 0.01, _lookup(scene, "ship"))  # before Burn A
    assert "Δv" not in text and len(text.split("\n")) == 3


def test_distance_instead_of_height_without_a_planet_center():
    scene = _scene()
    scene["centralBodyId"] = None
    assert "\nR 6,653 km\n" in label_text(scene, "ship", 0.05, _lookup(scene, "ship"))
