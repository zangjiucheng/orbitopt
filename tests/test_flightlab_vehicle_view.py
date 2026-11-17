"""Flight Lab's Vehicle view model (Qt-free): what a vehicle file declares,
and declared-vs-flown against a recorded run.

The flown side is the frozen E5-05a RTLS ascent run, so its facts (the event
times, the masses) are pinned here; the declared side is read from the
packaged vehicle catalog. The time-basis tests pin the convention found by
checking the model against that run's mass track: a thrust table is measured
from the stage's frame start (the previous separation), not from its ignition.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from orbitopt.flightlab import vehicle_catalog as vc
from orbitopt.flightlab.contract import read_run
from orbitopt.flightlab.vehicle_view import (
    G0,
    _truncate_impulse,
    _window,
    build_vehicle_view,
    flown_overlay,
    load_vehicle_view,
)

REPO = Path(__file__).resolve().parents[1]
RTLS_RUN = REPO / "tests" / "data" / "mission" / "reference" / "e5-05a-rtls-mission" / "ascent"
SYNTH_RUN = REPO / "tests" / "data" / "flightlab" / "synthetic_run"


@pytest.fixture(scope="module")
def rtls():
    return load_vehicle_view("synthetic_two_stage_rtls")


@pytest.fixture(scope="module")
def rtls_run():
    return read_run(RTLS_RUN)


@pytest.mark.parametrize("name", sorted(vc.catalog_paths()))
def test_every_catalog_vehicle_builds_a_view(name):
    view = load_vehicle_view(name)
    assert view.stages and view.liftoff_mass_kg > 0 and view.liftoff_twr_vac > 1.0
    assert view.total_delta_v_m_s == pytest.approx(sum(s.delta_v_m_s for s in view.stages))
    for s in view.stages:
        assert s.ignition_s < s.burnout_s
        assert s.thrust and s.thrust[0][0] == pytest.approx(s.ignition_s)
        assert s.burned_kg <= s.propellant_kg * (1 + 1e-9)       # never burns what is not loaded
        assert 0.0 < s.delta_v_m_s < 15_000.0
    assert view.provenance_for("stages[0].mass_initial_kg") is not None


def test_rtls_stage_one_is_cut_off_with_its_declared_reserve(rtls):
    s1 = rtls.stages[0]
    assert s1.burnout_s == pytest.approx(134.0)
    # 13.75 MN at Isp 330 s for 134 s
    assert s1.burned_kg == pytest.approx(13_750_573.8 * 134.0 / (330.0 * G0), rel=1e-3)
    assert s1.left_at_cutoff_kg == pytest.approx(153_400.0, rel=0.01)   # the +48 t reserve
    assert any("cut off before depletion" in n and "Stage 1" in n for n in rtls.notes)
    assert rtls.liftoff_mass_kg == pytest.approx(880_770.0)


def test_thrust_table_is_measured_from_the_frame_start_not_from_ignition(rtls):
    """Stage 2 ignites 7 s after separation (148 s) but its table starts at the
    separation (141 s): the first thrust is the table's value at +7 s, not its
    peak. A flown stage 2 burns exactly that (mass track, 148 -> 150 s)."""
    s2 = rtls.stages[1]
    assert s2.ignition_s == pytest.approx(148.0)
    first_t, first_f = s2.thrust[0]
    assert first_t == pytest.approx(148.0)
    assert first_f == pytest.approx(14e6 - (14e6 - 1_641_233.6) * 7.0 / 17.0, rel=1e-3)
    assert first_f < 0.7 * 14e6


def test_restart_schedule_is_drawn_as_burn_coast_restart(rtls):
    s2 = rtls.stages[1]
    assert [g.kind for g in s2.segments] == ["burn", "coast", "restart"]
    assert (s2.segments[1].t0_s, s2.segments[1].t1_s) == pytest.approx((224.5, 379.5))
    assert s2.burnout_s == pytest.approx(403.17, abs=0.01)
    assert rtls.restart is not None


def test_a_stage_that_runs_dry_before_burn_end_is_cut_where_it_runs_dry():
    """synthetic_two_stage declares 200 s of thrust for 350 t of propellant;
    it burns out at 193.8 s (a flown run of that file: MECO 193.77 s)."""
    view = load_vehicle_view("synthetic_two_stage")
    s1 = view.stages[0]
    assert s1.burnout_s == pytest.approx(193.8, abs=0.2)
    assert s1.burned_kg == pytest.approx(s1.propellant_kg)
    assert any("runs dry" in n for n in view.notes)


def test_window_and_truncate_helpers():
    pts = [(0.0, 10.0), (10.0, 20.0), (20.0, 20.0)]
    assert _window(pts, 5.0, 15.0) == [(5.0, 15.0), (10.0, 20.0), (15.0, 20.0)]
    cut, t = _truncate_impulse(pts, 100.0)            # ramp 10->20 over 10 s is 150 N s
    assert t is not None and cut[-1][0] == pytest.approx(t)
    imp = sum(0.5 * (f0 + f1) * (t1 - t0) for (t0, f0), (t1, f1) in zip(cut, cut[1:]))
    assert imp == pytest.approx(100.0)
    assert _truncate_impulse(pts, 1e9)[1] is None     # never reached
    flat, tf = _truncate_impulse([(0.0, 5.0), (10.0, 5.0)], 20.0)
    assert tf == pytest.approx(4.0) and math.isfinite(tf)


def test_geometry_is_drawn_to_scale_only_when_the_file_declares_it():
    f9 = load_vehicle_view("f9_class_public_estimate")
    assert f9.geometry.declared and f9.geometry.stage_length[0] == pytest.approx(42.6)
    assert f9.geometry.fairing_length == pytest.approx(13.2)
    prov = f9.provenance_for("geometry.stages[0].length_m")
    assert prov is not None and prov.provenance == "public_estimate"
    schematic = load_vehicle_view("synthetic_two_stage_rtls").geometry
    assert not schematic.declared and schematic.fairing_length is None
    assert max(schematic.stage_length) <= 11.0 * max(schematic.stage_diameter) + 1e-9


def test_geometry_block_validates_and_is_not_flown():
    doc = vc.load_validated(vc.catalog_paths()["f9_class_public_estimate"])
    assert "geometry" in doc
    bad = {**doc, "geometry": {"stages": [{"length_m": -1.0}]}}
    with pytest.raises(vc.VehicleCatalogError):
        vc.validate_vehicle_document(bad)
    extra = {**doc, "geometry": {"stages": [{"length_m": 10.0, "tank": 1}]}}
    with pytest.raises(vc.VehicleCatalogError):
        vc.validate_vehicle_document(extra)


def test_view_needs_a_stage():
    with pytest.raises(ValueError, match="stage"):
        build_vehicle_view({"name": "empty", "stages": []})


# ------------------------------------------------------------ declared vs flown

def _by_label(rows):
    return {r.label: r for r in rows}


def test_overlay_matches_the_flown_staging_events(rtls, rtls_run):
    rows = _by_label(flown_overlay(rtls, rtls_run))
    for label, t in (("Stage 1 cutoff (MECO)", 134.0), ("Stage 1 separation", 141.0),
                     ("Stage 2 ignition", 148.0), ("Fairing jettison", 201.0)):
        assert rows[label].declared == pytest.approx(t)
        assert rows[label].flown == pytest.approx(t, abs=0.01), label
    assert rows["Liftoff mass"].declared == pytest.approx(rows["Liftoff mass"].flown)
    assert rows["Booster mass after separation"].flown == pytest.approx(183_405.0, rel=1e-3)
    assert rows["Booster mass after separation"].declared is None      # flown-only row
    assert rows["Max-Q"].flown == pytest.approx(56.15, abs=0.05)


def test_overlay_reads_the_flown_cutoff_and_coast_from_the_run(rtls, rtls_run):
    """The recorded run's SECO is a fact of the run; the coast is read from the
    stage's mass track, whose edges are only as sharp as its 2 s output step."""
    rows = _by_label(flown_overlay(rtls, rtls_run))
    assert rows["Final cutoff (SECO)"].flown == pytest.approx(396.17, abs=0.05)
    coast, restart = rows["Coast start (mass track)"], rows["Restart ignition (mass track)"]
    assert coast.declared == pytest.approx(224.5) and restart.declared == pytest.approx(379.5)
    assert 150.0 < coast.flown < restart.flown < 396.0
    assert coast.flown == pytest.approx(217.5, abs=2.5)
    assert restart.flown == pytest.approx(372.5, abs=2.5)


def test_overlay_row_delta_is_none_when_one_side_is_missing():
    from orbitopt.flightlab.vehicle_view import OverlayRow

    assert OverlayRow("x", "s", None, 5.0).delta is None
    assert OverlayRow("x", "s", 5.0, None).delta is None
    assert OverlayRow("x", "s", 5.0, 7.5).delta == pytest.approx(2.5)


def test_overlay_survives_a_run_with_missing_events(rtls):
    """The tiny synthetic fixture has no per-stage ignition / jettison / restart
    data: the overlay keeps its rows with ``None`` flown values, never raises."""
    rows = _by_label(flown_overlay(rtls, read_run(SYNTH_RUN)))
    assert rows["Stage 2 ignition"].flown is None
    assert rows["Coast start (mass track)"].flown is None or rows["Coast start (mass track)"].flown > 0
