"""E11-06: the SpaceX-style engine HUD -- the engine map layout and lighting order, the throttle of the lit
engines, the flight-path orientation, the state lookup on a flown track, and the painted widget."""
from __future__ import annotations

import math

import pytest

from orbitopt.flightlab.ui.engine_hud import (RINGS, engine_layout, engine_state_at, flight_path_orientation,
                                              lit_engines, lit_order)


def test_the_layout_is_the_super_heavy_rings():
    lay = engine_layout(33)
    assert len(lay) == 33
    assert [sum(1 for e in lay if e[2] == r) for r in range(3)] == list(RINGS) == [3, 10, 20]
    assert all(math.hypot(x, y) <= 1.0 for x, y, _ in lay)
    centre = [e for e in lay if e[2] == 0]
    assert all(math.hypot(x, y) < 0.2 for x, y, _ in centre)
    assert [sum(1 for e in engine_layout(13) if e[2] == r) for r in range(3)] == [3, 10, 0]      # the landing set
    assert engine_layout(0) == ()


def test_engines_light_centre_first_then_inner_then_outer_and_shut_down_in_reverse():
    order = lit_order(33)
    lay = engine_layout(33)
    assert sorted(order) == list(range(33))
    assert {lay[i][2] for i in order[:3]} == {0}
    assert {lay[i][2] for i in order[3:13]} == {1}
    assert {lay[i][2] for i in order[13:]} == {2}
    assert lit_engines(33, 13) >= lit_engines(33, 5) >= lit_engines(33, 3)       # shutdown only removes engines
    assert lit_engines(33, 3) == frozenset(order[:3])
    assert len(lit_engines(33, 13)) == 13 and len(lit_engines(33, 99)) == 33 and lit_engines(33, 0) == frozenset()
    five = lit_engines(33, 5) - lit_engines(33, 3)                                 # two inner-ring engines, spread out
    assert len(five) == 2 and {lay[i][2] for i in five} == {1}


def test_flight_path_orientation_against_the_local_horizon():
    r = (6.4e6, 0.0, 0.0)                              # on the equator at the prime meridian: up = +x, east = +y
    up = flight_path_orientation(r, (100.0, 0.0, 0.0))
    assert up.elevation_deg == pytest.approx(90.0)
    east = flight_path_orientation(r, (0.0, 300.0, 0.0))
    assert east.heading_deg == pytest.approx(90.0) and east.elevation_deg == pytest.approx(0.0, abs=1e-9)
    north_down = flight_path_orientation(r, (-100.0, 0.0, 100.0))
    assert north_down.heading_deg == pytest.approx(0.0) and north_down.elevation_deg == pytest.approx(-45.0)
    assert north_down.nose_elevation_deg == pytest.approx(45.0)                    # tail first while falling
    assert flight_path_orientation((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)) is None


def _track():
    t = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    extra = {"throttle": [0.0, 0.9, 0.9, 0.0, 0.3, 0.2], "engines_lit": [0, 13, 13, 0, 5, 3]}
    pos = [(6.4e6, 0.0, 0.0)] * 6
    vel = [(0.0, 300.0, 0.0)] * 6
    return t, extra, pos, vel


def test_state_at_a_time_reads_the_flown_engines_and_the_throttle_of_the_lit_ones():
    t, extra, pos, vel = _track()
    s = engine_state_at(t, extra, pos, vel, 1.5, total=33)
    assert s.lit == 13 and s.on and s.count_known and s.throttle == pytest.approx(0.9)
    coast = engine_state_at(t, extra, pos, vel, 3.5, total=33)
    assert coast.lit == 0 and not coast.on and coast.throttle == 0.0
    staged = engine_state_at(t, extra, pos, vel, 4.5, total=33)                  # 5 engines at 0.3 of the FULL set
    assert staged.lit == 5 and staged.throttle == pytest.approx(0.3 / (5 / 13))
    assert engine_state_at(t, extra, pos, vel, 9.0, total=33) is None
    assert s.orientation.heading_deg == pytest.approx(90.0)


def test_a_track_without_an_engine_column_still_shows_whether_an_engine_fired():
    t, extra, pos, vel = _track()
    s = engine_state_at(t, {"throttle": extra["throttle"]}, pos, vel, 1.5, total=33)
    assert s.on and s.lit is None and not s.count_known and s.throttle == pytest.approx(0.9)
    assert not engine_state_at(t, {"throttle": extra["throttle"]}, pos, vel, 3.5, total=33).on


class _Track:
    """A duck-typed BodyTrack: only what the HUD reads."""

    def __init__(self, body_id, t, extra, pos, vel, mass=None):
        self.body_id, self.t_s, self.extra_columns = body_id, tuple(t), extra
        self.position_m, self.velocity_m_s = pos, vel
        self.mass_kg = tuple(mass if mass is not None else [1000.0 - i for i in range(len(t))])


def test_the_throttle_pointer_sits_at_the_throttle_on_a_270_degree_arc_open_at_the_bottom():
    from orbitopt.flightlab.ui.engine_hud import gauge_angle_deg

    assert gauge_angle_deg(0.0) == 225.0                      # lower left
    assert gauge_angle_deg(0.5) == 90.0                       # straight up
    assert gauge_angle_deg(1.0) == -45.0                      # lower right
    assert gauge_angle_deg(-3.0) == 225.0 and gauge_angle_deg(9.0) == -45.0
    assert gauge_angle_deg(0.8) < gauge_angle_deg(0.4)        # more throttle moves the pointer clockwise


def test_the_starship_has_six_engines_three_big_and_three_small():
    from orbitopt.flightlab.ui.engine_hud import starship_layout

    lay = starship_layout()
    assert len(lay) == 6 and [k for *_xy, k in lay].count("rvac") == 3 and [k for *_xy, k in lay].count("sl") == 3
    big = [e for e in lay if e[3] == "rvac"]
    small = [e for e in lay if e[3] == "sl"]
    assert all(b[2] > s_[2] * 2 for b in big for s_ in small)
    assert all(math.hypot(x, y) + rad <= 1.0 for x, y, rad, _ in lay)


def test_the_super_heavy_row_follows_stage_one_then_the_booster_and_the_ship_its_own_track():
    from orbitopt.flightlab.ui.engine_hud import SUPER_HEAVY, STARSHIP, pick_track

    t1 = _Track("stage1", [0, 1, 2], {}, [(6.4e6, 0, 0)] * 3, [(0, 100, 0)] * 3)
    boo = _Track("booster", [2, 3, 4, 5], {}, [(6.4e6, 0, 0)] * 4, [(0, 100, 0)] * 4)
    ship = _Track("stage2", [0, 1, 2, 3, 4, 5], {}, [(6.4e6, 0, 0)] * 6, [(0, 100, 0)] * 6)
    tracks = {"stage1": t1, "booster": boo, "stage2": ship}
    assert pick_track(tracks, SUPER_HEAVY.bodies, 1.0).body_id == "stage1"
    assert pick_track(tracks, SUPER_HEAVY.bodies, 4.0).body_id == "booster"
    assert pick_track(tracks, STARSHIP.bodies, 1.0).body_id == "stage2"
    assert pick_track({"stage1": t1}, STARSHIP.bodies, 1.0) is None


def test_the_readout_carries_speed_and_the_real_propellant_of_each_vehicle():
    from orbitopt.flightlab.ui.engine_hud import SUPER_HEAVY, STARSHIP, readout_at

    pos = [(6.4e6, 0, 0)] * 4
    vel = [(0, 100, 0)] * 4
    sh = _Track("stage1", [0, 1, 2, 3], {"throttle": [0.0, 1.0, 1.0, 0.0]}, pos, vel)
    ship = _Track("stage2", [0, 1, 2, 3], {"throttle": [1.0, 1.0, 1.0, 1.0]}, pos, vel)
    speed = {"stage1": [(0.0, 0.0), (3.0, 3.0)], "stage2": [(0.0, 0.0), (3.0, 6.0)]}
    prop = {"stage1": ("recorded", [(0.0, 3400.0), (3.0, 2400.0)]), "stage2": ("est.", [(0.0, 1200.0), (3.0, 1000.0)])}
    a = readout_at(SUPER_HEAVY, {"stage1": sh, "stage2": ship}, speed, prop, 1.5)
    b = readout_at(STARSHIP, {"stage1": sh, "stage2": ship}, speed, prop, 1.5)
    assert a.name == "SUPER HEAVY" and b.name == "STARSHIP"
    assert a.speed_kmh == pytest.approx(5400.0) and b.speed_kmh == pytest.approx(10800.0)   # 1.5 and 3 km/s
    assert a.propellant_t == pytest.approx(2900.0) and a.propellant_label == "recorded"
    assert a.propellant_frac == pytest.approx(2900 / 3400)
    assert b.propellant_t == pytest.approx(1100.0) and b.propellant_label == "est."
    assert a.speed_frac == pytest.approx(0.5) and b.speed_frac == pytest.approx(0.5)
    assert a.engines.on and b.engines.on


def test_under_power_the_rocket_points_where_the_thrust_points_so_a_landing_burn_is_vertical():
    from orbitopt.flightlab.ui.engine_hud import SUPER_HEAVY, readout_at

    # falling and drifting east at 80 m/s: the FLIGHT PATH is shallow, the thrust is vertical
    pos = [(6.4e6, 0, 0)] * 3
    vel = [(-30.0, 80.0, 0.0)] * 3
    extra = {"throttle": [0.0, 0.8, 0.8], "engines_lit": [0, 5, 5],
             "thrust_elev_deg": [0.0, 89.0, 88.0], "thrust_heading_deg": [0.0, 90.0, 90.0]}
    tr = _Track("booster", [0, 1, 2], extra, pos, vel)
    on = readout_at(SUPER_HEAVY, {"booster": tr}, {}, {}, 1.5)
    assert on.pointing == "thrust" and on.engines.orientation.nose_elevation_deg == pytest.approx(88.5, abs=0.6)
    assert on.engines.orientation.elevation_deg < 0.0                     # the path really is shallow and descending
    coast = readout_at(SUPER_HEAVY, {"booster": _Track("booster", [0, 1, 2], {**extra, "throttle": [0, 0, 0],
                                                                                 "engines_lit": [0, 0, 0]}, pos, vel)}, {}, {}, 1.5)
    assert coast.pointing == "flight path" and coast.engines.orientation.nose_elevation_deg < 30.0


def test_before_separation_the_starship_rides_the_stack_with_its_propellant_and_no_engines_lit():
    from orbitopt.flightlab.ui.engine_hud import STARSHIP, readout_at

    pos = [(6.4e6, 0, 0)] * 6
    vel = [(0, 100, 0)] * 6
    stack = _Track("stage1", [0, 1, 2, 3, 4, 5], {"throttle": [1.0] * 6}, pos, vel)
    ship = _Track("stage2", [4, 5], {"throttle": [1.0, 1.0]}, pos[:2], vel[:2])
    speed = {"stage1": [(0.0, 0.0), (5.0, 2.0)], "stage2": [(4.0, 1.6), (5.0, 2.0)]}
    prop = {"stage2": ("est.", [(0.0, 1200.0), (5.0, 1190.0)])}
    r = readout_at(STARSHIP, {"stage1": stack, "stage2": ship}, speed, prop, 2.0)
    assert r.attached and not r.engines.on and r.engines.lit == 0
    assert r.speed_kmh == pytest.approx(0.8 * 3600) and r.propellant_t == pytest.approx(1196.0)
    later = readout_at(STARSHIP, {"stage1": stack, "stage2": ship}, speed, prop, 4.5)
    assert not later.attached and later.engines.on                      # its own track, its own engines


def test_a_run_without_throttle_or_engine_columns_still_shows_when_the_engines_burn():
    from orbitopt.flightlab.ui.engine_hud import SUPER_HEAVY, readout_at

    pos = [(6.4e6, 0, 0)] * 4
    vel = [(0, 100, 0)] * 4
    burning = _Track("stage1", [0, 1, 2, 3], {}, pos, vel, mass=[1000, 900, 800, 800])
    r = readout_at(SUPER_HEAVY, {"stage1": burning}, {}, {}, 1.0)
    assert r.engines.on and r.engines.lit is None and not r.engines.count_known
    assert not readout_at(SUPER_HEAVY, {"stage1": burning}, {}, {}, 3.0).engines.on       # mass no longer falling


def test_the_widgets_light_the_engines_each_vehicle_reports_and_paint():
    pytest.importorskip("PySide6")
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication

    from orbitopt.flightlab.ui.engine_hud import EngineHud

    QApplication.instance() or QApplication([])
    pos = [(6.4e6, 0, 0)] * 6
    vel = [(0, 100, 0)] * 6
    booster = _Track("booster", [0, 1, 2, 3, 4, 5], {"throttle": [0.0, 0.9, 0.9, 0.0, 0.3, 0.2],
                                                      "engines_lit": [0, 13, 13, 0, 5, 3]}, pos, vel)
    ship = _Track("stage2", [0, 1, 2, 3, 4, 5], {"throttle": [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]}, pos, vel)
    from orbitopt.flightlab.ui.telemetry import TelemetryChannel, TelemetrySeries

    ch = (TelemetryChannel("speed", "Speed", "km/s", "", (TelemetrySeries("speed", "Booster", "km/s", "#fff", "booster",
                                                                       ((0.0, 0.0), (5.0, 2.0))),
                                                          TelemetrySeries("speed", "Stage2", "km/s", "#fff", "stage2",
                                                                          ((0.0, 0.0), (5.0, 7.0))))),
          TelemetryChannel("propellant", "Propellant", "t", "", (
              TelemetrySeries("propellant", "Booster (est.)", "t", "#fff", "booster", ((0.0, 300.0), (5.0, 100.0))),
              TelemetrySeries("propellant", "Stage2 (recorded)", "t", "#fff", "stage2", ((0.0, 1200.0), (5.0, 800.0))))))
    hud = EngineHud()
    hud.resize(600, 420)
    hud.set_run((booster, ship), ch)
    hud.set_time(1.5)
    assert len(hud.superheavy.lit_indices) == 13
    assert not hud.starship.lit_indices and hud.starship.readout.engines.on
    img13 = QImage(hud.size(), QImage.Format_ARGB32); img13.fill(0); hud.render(img13)
    hud.set_time(4.5)
    assert len(hud.superheavy.lit_indices) == 5 and hud.starship.lit_indices == frozenset()
    img5 = QImage(hud.size(), QImage.Format_ARGB32); img5.fill(0); hud.render(img5)
    assert img13 != img5
    assert hud.superheavy.readout.propellant_label == "est." and hud.starship.readout.propellant_t is not None
    # the outer ring is the PROPELLANT (fraction of the start), the inner ring the throttle of the lit engines
    prop, thr = hud.superheavy.gauge_fractions
    assert prop == pytest.approx((300.0 - 40.0 * 4.5) / 300.0, rel=1e-6) and thr == pytest.approx(0.3 / (5 / 13))
    ship_prop, ship_thr = hud.starship.gauge_fractions
    assert ship_prop == pytest.approx((1200.0 - 80.0 * 4.5) / 1200.0, rel=1e-6)
    assert hud.starship.gauge_fractions[1] == 0.0                         # engines off: inner ring empty
    hud.clear()
    assert hud.superheavy.readout is None and hud.state is None and hud.superheavy.gauge_fractions == (0.0, 0.0)


def test_the_flight_tab_has_an_engines_tab_that_follows_the_replay_time():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from orbitopt.flightlab.ui.flight_panel import FlightRunPanel

    QApplication.instance() or QApplication([])
    panel = FlightRunPanel()
    tabs = [panel.detail_tabs.tabText(i) for i in range(panel.detail_tabs.count())]
    assert tabs[-1] == "Engines" and panel.detail_tabs.widget(tabs.index("Engines")) is panel.engine_hud
    panel.set_time(12.0)                                  # no run loaded: nothing to show, nothing breaks
    assert panel.engine_hud.state is None


def test_engine_rows_follow_a_multistage_run_and_do_not_guess_unknown_counts():
    from orbitopt.flightlab.ui.engine_hud import stage_specs

    pos, vel = [(6.4e6, 0, 0)] * 2, [(0, 100, 0)] * 2
    tracks = {
        "stage1": _Track("stage1", [0, 1], {"engines_lit": [7, 4]}, pos, vel),
        "stage3": _Track("stage3", [8, 9], {"throttle": [0, 1]}, pos, vel),
    }
    specs = stage_specs(tracks, {"name": "test-launcher", "stages": [{}, {}, {"engine": {"count": 2}}]})
    assert list(specs) == [1, 3]
    assert (specs[1].total, specs[1].observed_only) == (7, True)
    assert (specs[3].total, specs[3].observed_only, specs[3].kind) == (2, False, "generic")
    assert stage_specs(tracks, None)[3].total == 0


def test_a_rocket_standing_on_the_pad_points_straight_up_even_in_an_inertial_frame():
    from orbitopt.flightlab.ui.engine_hud import EARTH_RATE_RAD_S, flight_path_orientation

    r = (6.4e6, 0.0, 0.0)                                       # on the equator, prime meridian
    at_rest_inertial = (0.0, EARTH_RATE_RAD_S * 6.4e6, 0.0)     # omega x r: 467 m/s east, but it is not moving
    lying = flight_path_orientation(r, at_rest_inertial)                    # without the correction: 'flying east'
    assert lying.elevation_deg == pytest.approx(0.0, abs=1e-6)
    standing = flight_path_orientation(r, at_rest_inertial, inertial=True)  # Earth rotation removed
    assert standing.nose_elevation_deg == 90.0 and standing.elevation_deg == 90.0
    # climbing at 200 m/s straight up while the Earth turns underneath: still up, now with a real velocity
    climbing = flight_path_orientation(r, (200.0, EARTH_RATE_RAD_S * 6.4e6, 0.0), inertial=True)
    assert climbing.elevation_deg == pytest.approx(90.0, abs=1e-6)
    # ... and once it has real downrange speed the path tilts over
    pitched = flight_path_orientation(r, (200.0, EARTH_RATE_RAD_S * 6.4e6 + 200.0, 0.0), inertial=True)
    assert pitched.elevation_deg == pytest.approx(45.0, abs=0.5) and pitched.heading_deg == pytest.approx(90.0)


def test_the_hud_starts_a_run_pointing_up():
    from orbitopt.flightlab.ui.engine_hud import EARTH_RATE_RAD_S, SUPER_HEAVY, readout_at

    r = (6.4e6, 0.0, 0.0)
    v_pad = (0.0, EARTH_RATE_RAD_S * 6.4e6, 0.0)
    tr = _Track("stage1", [0, 1, 2], {"throttle": [1.0, 1.0, 1.0]}, [r] * 3, [v_pad] * 3)   # a GCRS track, on the pad
    t0 = readout_at(SUPER_HEAVY, {"stage1": tr}, {}, {}, 0.0)
    assert t0.engines.orientation.nose_elevation_deg == 90.0                          # liftoff: vertical


def test_the_speed_gauge_reads_the_speed_against_the_ground_so_liftoff_is_zero():
    from orbitopt.flightlab.ui.engine_hud import EARTH_RATE_RAD_S, earth_relative_speed_series

    r = (6.4e6, 0.0, 0.0)
    pad = _Track("stage1", [0, 1], {}, [r, r], [(0.0, EARTH_RATE_RAD_S * 6.4e6, 0.0)] * 2)
    assert [round(v * 1000.0, 3) for _t, v in earth_relative_speed_series(pad)] == [0.0, 0.0]     # inertial 467 m/s
    moving = _Track("stage1", [0, 1], {}, [r, r], [(0.0, EARTH_RATE_RAD_S * 6.4e6 + 300.0, 0.0)] * 2)
    assert earth_relative_speed_series(moving)[0][1] == pytest.approx(0.3, abs=1e-6)
    ecef = _Track("booster", [0, 1], {}, [r, r], [(0.0, 300.0, 0.0)] * 2)
    ecef.frame = "ECEF"
    assert earth_relative_speed_series(ecef)[0][1] == pytest.approx(0.3, abs=1e-6)          # already Earth-fixed


def test_neighbouring_rings_never_touch_however_narrow_the_panel_is():
    from orbitopt.flightlab.ui.engine_hud import ROW_HALF_EXTENTS, ROW_MIN_GAP, row_layout

    for width in (300, 360, 440, 520, 700, 1000, 1600):
        for height in (150, 200, 260, 400):
            r, xs, gap = row_layout(float(width), float(height))
            assert r > 0 and len(xs) == 4 and list(xs) == sorted(xs)
            for i in range(3):                                   # clear space between neighbouring things
                right_edge = xs[i] + ROW_HALF_EXTENTS[i] * r
                left_edge = xs[i + 1] - ROW_HALF_EXTENTS[i + 1] * r
                assert left_edge - right_edge >= ROW_MIN_GAP - 1e-6, (width, height, i)
            assert xs[0] - ROW_HALF_EXTENTS[0] * r >= 0.0
            assert xs[3] + ROW_HALF_EXTENTS[3] * r <= width + 1e-6, (width, height)
            assert gap >= ROW_MIN_GAP


def test_a_wide_panel_keeps_the_rings_as_large_as_the_height_allows():
    from orbitopt.flightlab.ui.engine_hud import row_layout

    r_tall, _, gap_wide = row_layout(1600.0, 260.0)
    r_narrow, _, _ = row_layout(440.0, 260.0)
    assert r_tall > r_narrow                                      # the width, not the height, limited the narrow one
    assert gap_wide > 16.0                                        # spare width goes into the gaps


def _angle_deg(a, b):
    dot = max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))
    return math.degrees(math.acos(dot))


def test_the_attitude_is_continuous_across_every_ignition_and_shutdown():
    from orbitopt.flightlab.ui.engine_hud import SUPER_HEAVY, readout_at

    # a burn (thrust level, nose horizontal), then a long coast falling tail first (nose up), then a burn pointing up
    t = [float(i) for i in range(0, 121)]
    lit = [13 if (10 <= x < 30 or 90 <= x < 110) else 0 for x in t]
    elev = [0.0 if x < 60 else 88.0 for x in t]
    pos = [(6.4e6, 0.0, 0.0)] * len(t)
    vel = [(-150.0, 120.0, 0.0)] * len(t)                      # falling and drifting east: nose up while coasting
    extra = {"throttle": [0.6 if v else 0.0 for v in lit], "engines_lit": lit,
             "thrust_elev_deg": elev, "thrust_heading_deg": [90.0] * len(t)}
    tr = _Track("booster", t, extra, pos, vel)
    tr.frame = "ECEF"
    noses = []
    for k in range(0, 1201):                                   # every 0.1 s
        r = readout_at(SUPER_HEAVY, {"booster": tr}, {}, {}, k * 0.1)
        noses.append(r.engines.orientation.nose_enu)
    steps = [_angle_deg(a, b) for a, b in zip(noses, noses[1:])]
    assert max(steps) < 6.0                                     # no jump: under 6 degrees per 0.1 s anywhere
    # ... and the two attitudes really are far apart, so a switch WOULD have jumped
    on = readout_at(SUPER_HEAVY, {"booster": tr}, {}, {}, 15.0).engines.orientation
    coast = readout_at(SUPER_HEAVY, {"booster": tr}, {}, {}, 55.0).engines.orientation
    assert _angle_deg(on.nose_enu, coast.nose_enu) > 60.0
    # at ignition and at shutdown the picture sits on the thrust attitude; mid-coast on the path attitude
    assert readout_at(SUPER_HEAVY, {"booster": tr}, {}, {}, 10.0).engines.orientation.nose_elevation_deg == pytest.approx(0.0, abs=1e-6)
    assert readout_at(SUPER_HEAVY, {"booster": tr}, {}, {}, 60.0).pointing == "flight path"


def test_a_track_that_records_no_throttle_gets_one_from_the_mass_flow_so_the_pointer_shows_at_liftoff():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from orbitopt.flightlab.ui.engine_hud import EngineHud, mass_flow_series

    QApplication.instance() or QApplication([])
    t = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    mass = [1000.0, 800.0, 600.0, 500.0, 400.0, 400.0]           # 100 kg/s full, then 50 kg/s, then off
    pos = [(6.4e6, 0.0, 0.0)] * 6
    vel = [(0.0, 0.0, 0.0)] * 6
    flow = mass_flow_series(_Track("stage1", t, {}, pos, vel, mass=mass))
    assert [round(v) for _t, v in flow] == [0, 100, 100, 50, 50, 0]
    hud = EngineHud()
    hud.set_run((_Track("stage1", t, {}, pos, vel, mass=mass),), ())
    hud.set_time(3.0)
    st = hud.superheavy.readout.engines
    assert st.on and st.throttle == pytest.approx(1.0) and st.throttle_est and not st.count_known
    assert hud.superheavy.gauge_fractions[1] == pytest.approx(1.0)            # the pointer is at full throttle
    hud.set_time(7.0)
    assert hud.superheavy.gauge_fractions[1] == pytest.approx(0.5)            # throttled back
    hud.set_time(9.5)
    assert not hud.superheavy.readout.engines.on and hud.superheavy.gauge_fractions[1] == 0.0


# ------------------------------------------------------------------ the grid-fin demonstration

def test_the_fin_layout_is_the_real_one():
    from orbitopt.flightlab.ui.grid_fin_hud import fin_azimuths_deg

    three = fin_azimuths_deg(3)                                  # Super Heavy V3: 90 / 90 / 180, the catch side bare
    gaps = sorted(((b - a) % 360 for a, b in zip(three, three[1:] + three[:1])))
    assert gaps == [90.0, 90.0, 180.0]
    assert 180.0 not in [a % 360 for a in three]                 # nothing at the bottom, the catch side
    four = fin_azimuths_deg(4)
    assert [round((b - a) % 360) for a, b in zip(four, four[1:] + four[:1])] == [90] * 4
    assert fin_azimuths_deg(0) == () and len(fin_azimuths_deg(5)) == 5


def test_the_fin_readout_follows_the_flown_telemetry_and_the_models_deploy_rule():
    from orbitopt.flightlab.ui.grid_fin_hud import NO_FIN_DATA, gridfin_readout_at, visible_chord_fraction

    t = [0.0, 1.0, 2.0, 3.0, 4.0]
    extra = {"fin_alpha_deg": [0, 0, 30, 30, 10], "fin_delta_deg": [0, 0, 17, 17, 5], "fin_bank_deg": [0, 0, 0, 0, 0],
             "fin_telemetry_valid": [0, 1, 1, 1, 1], "altitude_m": [90000, 90000, 40000, 30000, 1000],
             "dynamic_pressure_pa": [1, 5, 4000, 30000, 5000], "mach": [9, 9, 6, 4, 0.5]}
    assert gridfin_readout_at(t, extra, 0.5) == NO_FIN_DATA                      # no fin telemetry on this arc
    stowed = gridfin_readout_at(t, extra, 1.5)
    assert stowed.has_data and not stowed.deployed                               # above 70 km, no angle
    out = gridfin_readout_at(t, extra, 2.5)
    assert out.deployed and out.alpha_deg == 30 and out.delta_deg == 17 and out.q_kpa == pytest.approx(4.0)
    assert gridfin_readout_at(t, extra, 4.5) == NO_FIN_DATA                      # past the track
    assert gridfin_readout_at(t, {"throttle": [0] * 5}, 2.0) == NO_FIN_DATA      # a track with no fin columns
    assert visible_chord_fraction(0.0) == 1.0 and visible_chord_fraction(60.0) == pytest.approx(0.5)
    assert visible_chord_fraction(89.0) >= 0.12                                  # a fin edge-on never vanishes


def test_the_grid_fin_view_paints_and_changes_with_the_fins():
    pytest.importorskip("PySide6")
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication

    from orbitopt.flightlab.ui.engine_hud import EngineHud
    from orbitopt.flightlab.ui.grid_fin_hud import GridFinHud, GridFinReadout, row_layout_ok

    QApplication.instance() or QApplication([])
    hud = GridFinHud()
    hud.resize(560, 220)
    shots = []
    for rd in (None, GridFinReadout(1.0, True, False, 0, 0, 0, 9.0, 0.0, 90.0),
               GridFinReadout(2.0, True, True, 30.0, 17.0, 0.0, 4.0, 30.0, 30.0),
               GridFinReadout(3.0, True, True, 30.0, 17.0, 20.0, 4.0, 30.0, 30.0)):
        hud.set_readout(rd)
        img = QImage(hud.size(), QImage.Format_ARGB32); img.fill(0); hud.render(img)
        shots.append(img)
    assert all(not s.isNull() for s in shots)
    assert shots[0] != shots[1] != shots[2] != shots[3]                           # none / stowed / deployed / banked
    hud.set_fin_count(4)
    assert hud.fin_count == 4
    # fixed fins (the default: Super Heavy's do not fold) are drawn out even with no data; fold-away ones are not
    assert hud.fixed is True
    hud.set_readout(None)
    fixed_none = QImage(hud.size(), QImage.Format_ARGB32); fixed_none.fill(0); hud.render(fixed_none)
    hud.set_fixed(False)
    folding_none = QImage(hud.size(), QImage.Format_ARGB32); folding_none.fill(0); hud.render(folding_none)
    assert fixed_none != folding_none
    hud.set_fixed(True)
    assert row_layout_ok(280, 84) and row_layout_ok(1600, 400)                    # the three parts never overlap
    page = EngineHud()
    assert page.gridfins is not None
    page.clear()
    assert page.gridfins.readout.has_data is False


def test_fixed_fins_are_never_shown_stowed_and_the_flight_tab_decides_it_from_the_vehicle():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from orbitopt.flightlab.ui.engine_hud import EngineHud
    from orbitopt.flightlab.ui.grid_fin_hud import GridFinHud, GridFinReadout

    QApplication.instance() or QApplication([])
    hud = GridFinHud()
    assert hud._out(GridFinReadout(1.0, True, False, 0, 0, 0, 9.0, 0.0, 90.0))          # fixed: out although "not deployed"
    hud.set_fixed(False)
    assert not hud._out(GridFinReadout(1.0, True, False, 0, 0, 0, 9.0, 0.0, 90.0))      # a folding fin is folded
    assert hud._out(GridFinReadout(1.0, True, True, 5, 3, 0, 4.0, 5.0, 30.0))
    page = EngineHud()
    page.set_run((), (), (), fins_fixed=False)
    assert page.gridfins.fixed is False
    page.set_run((), (), ())                                                              # no vehicle info: Super Heavy's
    assert page.gridfins.fixed is True
