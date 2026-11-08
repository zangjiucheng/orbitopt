"""Checks for orbitopt.launch_windows: geometry of in-plane / dogleg windows,
corridor filtering, the porkchop link, and plausibility against flown
missions. Everything except the porkchop tests needs only numpy."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
import pytest

from orbitopt.cli import main as cli_main
from orbitopt.launch_sites import get_launch_site
from orbitopt.launch_windows import (
    OrbitPlane,
    _site_vectors,
    as_mjd2000,
    asymptote_from_v_inf,
    asymptote_launch_windows,
    azimuth_allowed,
    datetime_to_mjd2000,
    geocentric_latitude_deg,
    gmst_deg,
    j2_raan_rate_deg_day,
    mjd2000_to_datetime,
    plane_launch_windows,
    sun_synchronous_plane,
)

ISS_EPOCH = "2026-11-08T00:00:00Z"
ISS = OrbitPlane.from_altitude(51.64, 100.0, ISS_EPOCH, 420.0, name="ISS-like")


def _offplane(site, plane, t):
    s, _, _ = _site_vectors(site, t)
    return float(np.sum(plane.normal_at(t) * s, -1)[0])


def _utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


def test_time_conversions_and_gmst():
    assert datetime_to_mjd2000(_utc(2000, 1, 1, 12)) == pytest.approx(0.5)
    assert mjd2000_to_datetime(0.5) == _utc(2000, 1, 1, 12)
    assert as_mjd2000("2000-01-02T00:00:00Z") == pytest.approx(1.0)
    assert float(gmst_deg(0.5)) == pytest.approx(280.46061837)
    # One solar day advances GMST by ~0.9856 deg.
    assert float(gmst_deg(1.5) - gmst_deg(0.5)) % 360 == pytest.approx(0.98564736, abs=1e-6)


def test_j2_regression_iss():
    # ISS-like orbit regresses ~5 deg/day westward.
    assert j2_raan_rate_deg_day(6378.137 + 420, 51.64) == pytest.approx(-4.95, abs=0.1)
    assert j2_raan_rate_deg_day(7000, 90.0) == pytest.approx(0.0, abs=1e-12)


def test_azimuth_corridor_wraps_north():
    assert azimuth_allowed(355.0, [(349.5, 93.5)])
    assert azimuth_allowed(10.0, [(349.5, 93.5)])
    assert not azimuth_allowed(180.0, [(349.5, 93.5)])


@pytest.mark.parametrize("sid", ["cape-canaveral", "kourou", "baikonur", "wenchang", "vandenberg"])
def test_in_plane_instants_and_azimuths(sid):
    site = get_launch_site(sid)
    wins = plane_launch_windows(site, ISS, ISS_EPOCH, 1.0, respect_corridors=False)
    assert len(wins) == 2  # |lat| < i: one northbound + one southbound pass per day
    phi = math.radians(geocentric_latitude_deg(site.latitude_deg))
    az_ne = math.degrees(math.asin(math.cos(math.radians(51.64)) / math.cos(phi)))
    got = sorted(w.azimuth_optimal_deg for w in wins)
    assert got[0] == pytest.approx(az_ne, abs=1e-3)
    assert got[1] == pytest.approx(180.0 - az_ne, abs=1e-3)
    for w in wins:
        assert w.kind == "in-plane"
        assert w.open_mjd2000 == pytest.approx(w.close_mjd2000, abs=2e-5)  # instantaneous
        assert abs(_offplane(site, ISS, w.optimal_mjd2000)) < 1e-8
        assert w.inclination_deg == pytest.approx(51.64, abs=1e-3)


def test_corridors_keep_only_ksc_northeast_iss_launch():
    # KSC ISS launches fly the ~45 deg north-east trajectory; the south-east
    # pass (Az ~135) is outside the [35, 120] range-safety corridor.
    wins = plane_launch_windows("cape-canaveral", ISS, ISS_EPOCH, 3.0)
    assert len(wins) == 3
    for w in wins:
        assert w.direction == "northbound"
        assert w.azimuth_optimal_deg == pytest.approx(44.9, abs=0.3)


def test_iss_window_moves_earlier_each_day():
    # Operational rule of thumb: ISS launch times move ~20-25 min earlier a
    # day (sidereal day ~3m56s + ~5 deg/day nodal regression ~20 min).
    wins = plane_launch_windows("baikonur", ISS, ISS_EPOCH, 5.0)
    shifts = np.diff([w.optimal_mjd2000 for w in wins]) * 1440 - 1440
    assert np.all((shifts > -26) & (shifts < -20)), shifts


def test_dogleg_budget_bounds_window():
    budget = 0.5
    site = get_launch_site("cape-canaveral")
    (w,) = plane_launch_windows(site, ISS, ISS_EPOCH, 1.0, max_dogleg_deg=budget)
    edge = math.sin(math.radians(budget))
    assert abs(_offplane(site, ISS, w.open_mjd2000)) == pytest.approx(edge, abs=1e-6)
    assert abs(_offplane(site, ISS, w.close_mjd2000)) == pytest.approx(edge, abs=1e-6)
    assert w.open_mjd2000 < w.optimal_mjd2000 < w.close_mjd2000
    assert 5 * 60 < w.duration_s < 8 * 60  # ~0.5 deg / (15 deg/h * geometry)
    assert w.azimuth_open_deg < w.azimuth_optimal_deg < w.azimuth_close_deg


def test_dogleg_only_window_when_latitude_exceeds_inclination():
    site = get_launch_site("baikonur")
    phi = geocentric_latitude_deg(site.latitude_deg)
    plane = OrbitPlane(41.5, 30.0)
    assert plane_launch_windows(site, plane, 0.0, 1.0, max_dogleg_deg=0.0, respect_corridors=False) == []
    assert plane_launch_windows(site, plane, 0.0, 1.0, max_dogleg_deg=phi - 41.5 - 0.1,
                                respect_corridors=False) == []
    (w,) = plane_launch_windows(site, plane, 0.0, 1.0, max_dogleg_deg=5.0, respect_corridors=False)
    assert w.kind == "dogleg"
    assert w.dogleg_deg == pytest.approx(phi - 41.5, abs=1e-3)
    assert w.azimuth_optimal_deg == pytest.approx(90.0, abs=0.01)


def test_multi_day_scan_matches_daily_scans_without_duplicates():
    plane = OrbitPlane.from_altitude(51.64, 100.0, ISS_EPOCH, 420.0)
    whole = plane_launch_windows("wenchang", plane, ISS_EPOCH, 3.0, max_dogleg_deg=2.0)
    start = as_mjd2000(ISS_EPOCH)
    # Daily queries also report a window already open at their start (one
    # straddling midnight); keep only those opening in the day to de-duplicate.
    daily = [w for d in range(3)
             for w in plane_launch_windows("wenchang", plane, start + d, 1.0, max_dogleg_deg=2.0)
             if w.open_mjd2000 >= start + d]
    assert [round(w.open_mjd2000, 6) for w in whole] == [round(w.open_mjd2000, 6) for w in daily]


@pytest.mark.parametrize("launch", [
    _utc(2015, 6, 23, 1, 51, 58),  # Sentinel-2A, Vega VV05, Kourou
    _utc(2017, 3, 7, 1, 49, 24),   # Sentinel-2B, Vega VV09, Kourou
])
def test_sentinel2_sso_launch_time_plausible(launch):
    # Sentinel-2: sun-synchronous, i = 98.57 deg, 10:30 descending node
    # (22:30 ascending node). Vega flies north from Kourou.
    day = launch.replace(hour=0, minute=0, second=0)
    plane = sun_synchronous_plane(22.5, day, 98.57)
    wins = plane_launch_windows("kourou", plane, day, 1.0)
    assert len(wins) == 1
    (w,) = wins
    assert w.direction == "northbound"
    assert 345.0 < w.azimuth_optimal_deg < 355.0
    # Geometric in-plane instant vs. actual lift-off: within 15 min (real
    # launches lead the in-plane time slightly to null cross-range in ascent).
    assert abs((w.optimal_utc - launch).total_seconds()) < 15 * 60


def test_asymptote_from_v_inf_frames():
    rla, dla, c3 = asymptote_from_v_inf([0.0, 0.0, 3.0], frame="equatorial")
    assert (dla, c3) == (pytest.approx(90.0), pytest.approx(9.0))
    # Ecliptic +Y (summer-solstice direction) has declination = obliquity.
    rla, dla, _ = asymptote_from_v_inf([0.0, 1.0, 0.0])
    assert rla == pytest.approx(90.0) and dla == pytest.approx(23.4392911)


def test_asymptote_window_azimuth_range():
    wins = asymptote_launch_windows("cape-canaveral", 10.0, 23.0, "2020-07-30", 1.0, [(90.0, 115.0)])
    assert wins
    for w in wins:
        for az in (w.azimuth_open_deg, w.azimuth_close_deg, w.azimuth_optimal_deg):
            assert 90.0 - 1e-3 <= az <= 115.0 + 1e-3
        assert w.close_mjd2000 > w.open_mjd2000


def test_window_already_open_at_start_is_reported():
    wins = asymptote_launch_windows("cape-canaveral", 10.0, 23.0, "2020-07-30", 1.0, [(90.0, 115.0)])
    first = wins[0]
    start = 0.5 * (first.open_mjd2000 + first.close_mjd2000)
    later = asymptote_launch_windows("cape-canaveral", 10.0, 23.0, start, 0.1, [(90.0, 115.0)])
    assert later, "window open at the query start was reported as no window"
    (w,) = later
    assert w.open_before(start)
    assert w.open_mjd2000 == pytest.approx(first.open_mjd2000, abs=1e-6)
    assert w.close_mjd2000 == pytest.approx(first.close_mjd2000, abs=1e-6)
    assert start <= w.optimal_mjd2000 <= w.close_mjd2000  # best *remaining* instant
    assert w.open_bounded and w.close_bounded


def test_window_open_past_query_end_reports_true_close():
    wins = asymptote_launch_windows("cape-canaveral", 10.0, 23.0, "2020-07-30", 1.0, [(90.0, 115.0)])
    w0 = wins[-1]
    # Query a span that ends inside the window and whose scan margin ends
    # inside it too: the true close must still be found.
    start = w0.open_mjd2000 - 0.01
    (w,) = [x for x in asymptote_launch_windows("cape-canaveral", 10.0, 23.0, start, 0.02,
                                                  [(90.0, 115.0)]) if x.open_mjd2000 >= start]
    assert w.close_mjd2000 == pytest.approx(w0.close_mjd2000, abs=1e-6)


def test_optimum_never_after_search_end():
    # Window 02:07-08:42 UTC, optimum (due east) at 06:55. A search that ends
    # at 04:31 must not pick a lift-off after its cutoff.
    corr = [(60.0, 115.0)]
    (w0, _) = asymptote_launch_windows("cape-canaveral", 10.0, 23.0, "2020-07-30", 1.0, corr)
    start = w0.open_mjd2000 - 0.001
    end = 0.5 * (w0.open_mjd2000 + w0.optimal_mjd2000)
    (w,) = asymptote_launch_windows("cape-canaveral", 10.0, 23.0, start, end - start, corr)
    assert w.close_mjd2000 == pytest.approx(w0.close_mjd2000, abs=1e-6)  # true bounds kept
    assert start <= w.optimal_mjd2000 <= end
    assert w.optimal_mjd2000 == pytest.approx(end, abs=1e-6)  # closest to due east before the cutoff
    assert 60.0 - 1e-3 <= w.azimuth_optimal_deg <= 115.0 + 1e-3


def test_continuously_open_window_is_reported():
    wins = asymptote_launch_windows("cape-canaveral", 10.0, 23.0, "2020-07-30", 1.0, [(0.0, 360.0)])
    assert len(wins) == 1
    (w,) = wins
    assert w.continuous and not w.open_bounded and not w.close_bounded
    start = as_mjd2000("2020-07-30")
    assert (w.open_mjd2000, w.close_mjd2000) == (pytest.approx(start), pytest.approx(start + 1.0))
    assert w.azimuth_optimal_deg == pytest.approx(90.0, abs=0.01)
    assert w.to_dict()["open_bounded"] is False
    # Plane target with a dogleg budget covering every geometry: same.
    (p,) = plane_launch_windows("cape-canaveral", OrbitPlane(51.6, 0.0), start, 1.0,
                                max_dogleg_deg=90.0, respect_corridors=False)
    assert p.continuous


def test_cli_launch_window_json(capsys):
    cli_main(["launch-window", "kourou", "--inclination", "98.57", "--ltan", "22.5",
              "--epoch", "2015-06-23", "--json"])
    (row,) = json.loads(capsys.readouterr().out)
    assert row["site"] == "kourou" and row["kind"] == "in-plane"
    assert row["open_utc"].startswith("2015-06-23T01:")


def test_cli_launch_window_table_lists_data_sources(capsys):
    cli_main(["launch-window", "jiuquan", "--inclination", "98.0", "--epoch", "2020-01-01"])
    out = capsys.readouterr().out
    assert "Data sources (overall confidence: public_estimate)" in out
    assert "jiuquan location" in out and "jiuquan azimuth_corridors" in out
    assert "not official" in out


# ---- porkchop link (needs pykep) -------------------------------------------

def _planet(name):
    pk = pytest.importorskip("pykep")
    if hasattr(pk.planet, "jpl_lp"):
        return pk.planet.jpl_lp(name)
    return pk.planet(pk.udpla.jpl_lp(name))


@pytest.fixture(scope="module")
def mars2020_grid():
    pytest.importorskip("pykep")
    from orbitopt.viz.porkchop import compute_porkchop

    t0 = datetime_to_mjd2000(_utc(2020, 7, 30))
    return compute_porkchop(_planet("earth"), _planet("mars"), (t0 - 2, t0 + 2), (190.0, 215.0),
                            n_t0=5, n_tof=26, use_gpu=False)


def test_mars2020_launch_time_plausible(mars2020_grid):
    # Mars 2020 / Perseverance: Atlas V 541, SLC-41, lift-off 2020-07-30
    # 11:50 UTC (window 11:50-13:50), arrival 2021-02-18 (TOF 203 d).
    g = mars2020_grid
    i = int(np.argmin(np.abs(g.tof_grid - 203.0)))
    rla, dla, c3 = asymptote_from_v_inf(g.v_inf_departure[i, 2])
    assert 13.0 < c3 < 16.0          # published max C3 ~14.5 km^2/s^2
    assert 20.0 < dla < 27.0
    wins = asymptote_launch_windows("cape-canaveral", rla, dla, "2020-07-30", 1.0, [(90.0, 115.0)])
    liftoff = _utc(2020, 7, 30, 11, 50)
    # The window opening at the due-east azimuth falls within 10 min of lift-off.
    hit = [w for w in wins if abs((w.open_utc - liftoff).total_seconds()) < 10 * 60]
    assert len(hit) == 1
    assert hit[0].azimuth_open_deg == pytest.approx(90.0, abs=0.01)


def test_porkchop_launch_windows_marks_dates(mars2020_grid, tmp_path):
    from orbitopt.launch_windows import porkchop_launch_windows
    from orbitopt.viz.porkchop import plot_porkchop

    g = mars2020_grid
    assert g.v_inf_departure.shape == g.c3_departure.shape + (3,)
    np.testing.assert_allclose(np.sum(g.v_inf_departure**2, -1), g.c3_departure, rtol=1e-12)
    days = porkchop_launch_windows(g, "cape-canaveral", azimuth_range_deg=[(90.0, 115.0)])
    assert len(days) == len(g.t0_grid)
    assert all(d.launchable for d in days)
    for d in days:
        for w in d.windows:
            assert d.t0_mjd2000 <= w.open_mjd2000 < d.t0_mjd2000 + 1
    # A C3 cap below every column's minimum leaves no launchable date.
    capped = porkchop_launch_windows(g, "cape-canaveral", max_c3=1.0)
    assert not any(d.launchable for d in capped)

    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    ax = plot_porkchop(g, launch_days=capped)
    ax.figure.savefig(tmp_path / "pc.png")
