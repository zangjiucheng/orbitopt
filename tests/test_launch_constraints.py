"""Checks for orbitopt.launch_constraints (C3 / DLA launch constraints in the
optimizer fitness). The constraint math is compute-free; the problem-level
tests exercise Transfer2BodyProblem's fitness() and batch_fitness() paths.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from orbitopt.launch_constraints import (
    OBLIQUITY_J2000_DEG,
    LaunchConstraints,
    max_abs_dla_for_site,
)
from orbitopt.launch_sites import LaunchSite, SourceRef, get_launch_site
from orbitopt.launch_vehicles import OutOfPerformanceRange, get_launch_vehicle


def _site(lat, corridors):
    return LaunchSite(
        id="toy", name="Toy", latitude_deg=lat, longitude_deg=0.0, altitude_m=0.0,
        rotation_speed_gain_m_s=0.0, azimuth_corridors_deg=tuple(corridors),
        sources=(SourceRef(("location",), "test"),),
    )


def _vinf(c3, dla_deg):
    """Equatorial-frame v_inf [m/s] with the given C3 [km^2/s^2] and DLA [deg]."""
    v = math.sqrt(c3) * 1000.0
    d = math.radians(dla_deg)
    return np.array([[v * math.cos(d), 0.0, v * math.sin(d)]])


def test_max_abs_dla_from_site_inclinations():
    # Cape: direct ascent i in [28.6, 59.8] -> |DLA| <= 59.8
    cape = get_launch_site("cape-canaveral")
    assert max_abs_dla_for_site(cape) == pytest.approx(cape.reachable_inclination_range()[1])
    # Sites that reach polar orbits can point the asymptote anywhere.
    assert max_abs_dla_for_site(get_launch_site("vandenberg")) == 90.0
    # Purely eastward corridor at 28.5 deg latitude -> only i ~ 28.5 -> |DLA| <= 28.5
    assert max_abs_dla_for_site(_site(28.5, [(90.0, 90.0)])) == pytest.approx(28.5)


def test_c3_limit_matches_vehicle_curve():
    lc = LaunchConstraints.from_names("falcon-9", 4000.0, None, frame="equatorial")
    assert lc.max_c3 == pytest.approx(get_launch_vehicle("falcon-9").max_c3(4000.0))
    ev = lc.evaluate(np.vstack([_vinf(lc.max_c3 - 0.5, 0.0), _vinf(lc.max_c3 + 0.5, 0.0)]))
    assert ev.feasible.tolist() == [True, False]
    assert ev.c3_excess[1] == pytest.approx(0.5)


def test_payload_outside_curve_is_rejected():
    with pytest.raises(OutOfPerformanceRange):
        LaunchConstraints.from_names("falcon-9", 1e6, None)


def test_dla_limit_and_ecliptic_frame():
    lc = LaunchConstraints(site=_site(28.5, [(90.0, 90.0)]), frame="equatorial")
    ev = lc.evaluate(np.vstack([_vinf(10.0, 20.0), _vinf(10.0, -35.0)]))
    assert ev.dla_deg.tolist() == pytest.approx([20.0, -35.0])
    assert ev.feasible.tolist() == [True, False]
    assert ev.dla_excess_deg[1] == pytest.approx(35.0 - 28.5)
    # A vector along ecliptic +y sits at DLA = +obliquity in the equatorial frame.
    ecl = LaunchConstraints(site=_site(28.5, [(90.0, 90.0)]))
    assert ecl.evaluate(np.array([[0.0, 3000.0, 0.0]])).dla_deg[0] == pytest.approx(OBLIQUITY_J2000_DEG)


def test_apply_modes_and_toggle():
    base = dict(vehicle=get_launch_vehicle("falcon-9"), payload_kg=4000.0, frame="equatorial")
    lc_inv = LaunchConstraints(mode="invalid", **base)
    lc_pen = LaunchConstraints(mode="penalty", penalty_weight=100.0, **base)
    v = np.vstack([_vinf(5.0, 0.0), _vinf(lc_inv.max_c3 + 2.0, 0.0), _vinf(5.0, 0.0)])
    obj = np.array([5000.0, 5000.0, 1e6])  # last one: failed Lambert solve

    inv = lc_inv.apply(obj, v)
    assert inv[0] == 5000.0 and inv[1] >= lc_inv.invalid_fitness and inv[2] == 1e6
    # graded: a larger violation scores worse
    worse = lc_inv.apply(obj, np.vstack([v[0], _vinf(lc_inv.max_c3 + 4.0, 0.0), v[2]]))
    assert worse[1] > inv[1]

    pen = lc_pen.apply(obj, v)
    assert pen.tolist() == pytest.approx([5000.0, 5000.0 + 100.0 * 2.0, 1e6])

    off = lc_inv.with_enabled(False)
    assert off.apply(obj, v) is obj


def test_invalid_mode_rejects_bad_config():
    with pytest.raises(ValueError):
        LaunchConstraints(mode="soft")
    with pytest.raises(ValueError):
        LaunchConstraints(frame="icrf")


# --- problem integration (needs pykep) -------------------------------------

def _problem(lc):
    from orbitopt.bodies import mjd2000_from_date, planet
    from orbitopt.problems.transfer_2body import Transfer2BodyProblem

    return Transfer2BodyProblem(
        departure_body=planet("earth"),
        arrival_body=planet("mars"),
        t0_bounds=(mjd2000_from_date(2028, 1, 1), mjd2000_from_date(2029, 1, 1)),
        tof_bounds=(100.0, 400.0),
        use_gpu=False,
        launch_constraints=lc,
    )


@pytest.mark.parametrize("mode", ["invalid", "penalty"])
def test_problem_fitness_and_batch_agree_with_constraints(mode):
    lc = LaunchConstraints.from_names("falcon-9", 4000.0, "cape-canaveral", mode=mode)
    problem = _problem(lc)
    rng = np.random.default_rng(1)
    lo, hi = problem.get_bounds()
    dvs = rng.uniform(lo, hi, size=(64, 2))

    looped = np.array([problem.fitness(dv)[0] for dv in dvs])
    batched = problem.batch_fitness(dvs.ravel())
    plain = _problem(None).batch_fitness(dvs.ravel())

    assert np.allclose(looped, batched, rtol=1e-6, atol=1.0)
    ok = plain < 1e6
    # Constraints only ever make a candidate worse, and some candidates in a
    # random population exceed C3 = 9 km^2/s^2.
    assert np.all(batched[ok] >= plain[ok] - 1e-6)
    assert np.any(batched[ok] > plain[ok] + 1.0)


def test_problem_toggle_off_matches_unconstrained():
    lc = LaunchConstraints.from_names("falcon-9", 4000.0, "cape-canaveral", enabled=False)
    rng = np.random.default_rng(2)
    p_off, p_none = _problem(lc), _problem(None)
    dvs = rng.uniform(*p_off.get_bounds(), size=(32, 2)).ravel()
    assert np.array_equal(p_off.batch_fitness(dvs), p_none.batch_fitness(dvs))


def test_describe_reports_data_sources_and_confidence():
    d = LaunchConstraints.from_names("falcon-9", 4000.0, "kourou").describe()
    ds = d["data_sources"]
    items = [r["item"] for r in ds["items"]]
    assert items == ["falcon-9 c3_curve", "kourou location", "kourou azimuth_corridors"]
    assert all(r["source"] for r in ds["items"])
    # the digitized C3 curve is an estimate, so the whole set is not "official"
    assert ds["confidence"] == "public_estimate"
    assert LaunchConstraints.from_names(None, 0.0, "kourou").describe()["data_sources"]["confidence"] == "official"
