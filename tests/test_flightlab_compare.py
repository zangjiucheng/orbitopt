"""FL-503 Compare Runs + recovery panel tests.

The diff/recovery logic is Qt-free and tested on the small synthetic run
fixture; the Qt pages (compare page, replay page recovery/why tabs,
workbench wiring) are ``slow`` and use a stub-flown Gate-B run, like the
FL-501 replay tests. Ascent+recovery runs are built in the FL-304 output
format by ``flightlab_compare_fixture`` (FL-304 is not on this branch).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from flightlab_compare_fixture import fly_stub_run, make_ascent_only, make_ascent_recovery
from orbitopt.flightlab.compare import (
    check_margin,
    compare_runs,
    discover_runs,
    ellipse_polygon,
    find_montecarlo,
    load_bundle,
    overlay_channels,
    recovery_kind,
    recovery_summary,
    trajectory_profiles,
)
from orbitopt.flightlab.contract import Check, read_run
from orbitopt.flightlab.ui.telemetry import build_telemetry_channels

RUN_FIXTURE = Path(__file__).parent / "data" / "flightlab" / "synthetic_run"


@pytest.fixture
def pair(tmp_path):
    a = make_ascent_only(RUN_FIXTURE, tmp_path / "ascent", payload_kg=15000.0)
    b = make_ascent_recovery(RUN_FIXTURE, tmp_path / "rtls", payload_kg=12600.0, prop_td_kg=1850.0)
    return load_bundle(a), load_bundle(b)


def _row(rows, key):
    return next(r for r in rows if r.key == key)


def test_compare_ascent_only_vs_recovery_reports_payload_cost(pair):
    a, b = pair
    assert recovery_kind(a) == "none" and recovery_kind(b) == "rtls"
    result = compare_runs(a, b)
    assert any("Recovery reserve cost 2,400 kg payload" in h for h in result.headlines), result.headlines
    assert any("Booster propellant at touchdown: 1,850 kg (run B only)" in h for h in result.headlines)

    inputs = {r.key: r for r in result.inputs}
    assert inputs["vehicle.payload_kg"].delta == pytest.approx(-2400.0)
    assert inputs["mission.recovery"].a == "none" and inputs["mission.recovery"].b == "rtls"
    assert not inputs["vehicle.s1.isp_s"].changed  # same vehicle otherwise

    out = {r.key: r for r in result.outputs}
    assert out["outcome.payload_kg"].trend == "worse"
    assert out["recovery.booster_prop_kg"].a is None
    assert out["recovery.booster_prop_kg"].b == pytest.approx(1850.0)
    assert out["recovery.gate"].b == "LANDED"
    assert out["value.landing-position"].b == pytest.approx(7.8)
    assert out["margin.landing-vertical-speed"].b == pytest.approx(1.6)
    assert out["recovery.mc_success"].b == pytest.approx(20 / 24 * 100.0)
    # failure reasons (synthetic fixture code is uncatalogued but still listed)
    assert result.reasons_a and not result.reasons_a[0].known


def test_compare_two_ascent_only_runs(tmp_path):
    a = load_bundle(make_ascent_only(RUN_FIXTURE, tmp_path / "a", payload_kg=15000.0))
    b = load_bundle(make_ascent_only(RUN_FIXTURE, tmp_path / "b", payload_kg=14200.0))
    result = compare_runs(a, b)
    assert result.headlines == ("Payload down 800 kg from A to B (15,000 → 14,200 kg).",)
    out = {r.key: r for r in result.outputs}
    assert out["recovery.booster_prop_kg"].a is None and out["recovery.booster_prop_kg"].b is None
    assert [r.key for r in result.changed_inputs] == ["vehicle.payload_kg"]


def test_recovery_summary_with_montecarlo(pair):
    a, b = pair
    sa = recovery_summary(a)
    assert sa.state == "none" and sa.montecarlo is None

    sb = recovery_summary(b)
    assert sb.state == "flown" and sb.gate is not None and sb.gate.passed
    assert [c.id for c in sb.checks][:7] == ["recovery-landing", "landing-position", "landing-vertical-speed",
                                             "landing-lateral-speed", "landing-tilt",
                                             "landing-propellant-reserve", "landing-max-g"]
    assert sb.touchdown is not None and sb.touchdown["body_id"] == "booster"
    assert sb.success_rate == pytest.approx(20 / 24)
    codes = {code: count for code, count, _e in sb.mc_reasons}
    assert codes == {"LANDED": 20, "LANDING_VERTICAL_SPEED_EXCEEDED": 2, "INSUFFICIENT_PROPELLANT_BOOSTBACK": 2}
    assert all(entry.known for _c, _n, entry in sb.mc_reasons)
    assert len(sb.touchdown_points) == 22  # 20 landed + 2 hard touchdowns
    poly = ellipse_polygon(sb.ellipse, k=1.0)
    assert poly and poly[0] == pytest.approx(poly[-1])
    ell = sb.ellipse
    radii = [math.hypot(e - ell["mean_east_m"], n - ell["mean_north_m"]) for e, n in poly]
    assert max(radii) == pytest.approx(ell["sigma_along_m"], rel=1e-6)
    assert min(radii) == pytest.approx(ell["sigma_cross_m"], rel=1e-2)


def test_failed_recovery_is_explained(tmp_path):
    b = load_bundle(make_ascent_recovery(RUN_FIXTURE, tmp_path / "bad", landed=False, montecarlo=False))
    s = recovery_summary(b)
    assert s.gate is not None and not s.gate.passed
    assert [e.code for e in s.explanations] == ["LANDING_VERTICAL_SPEED_EXCEEDED",
                                                "LANDING_PROPELLANT_RESERVE_LOW"]
    assert s.montecarlo is None and not s.touchdown_points


def test_recovery_requested_but_not_flown_is_not_assessed(tmp_path):
    run_dir = make_ascent_only(RUN_FIXTURE, tmp_path / "req")
    form = json.loads((run_dir / "designer-form.json").read_text())
    form["mission_form"]["recovery"] = "rtls"
    (run_dir / "designer-form.json").write_text(json.dumps(form))
    assert recovery_summary(load_bundle(run_dir)).state == "not_assessed"


def test_check_margin_directions():
    assert check_margin(Check("max-q-within-limit", True, 36000.0, 60000.0, "Pa")) == 24000.0
    assert check_margin(Check("recovery-propellant-reserve", False, 0.0, 1000.0, "kg")) == -1000.0
    assert check_margin(Check("insertion-apoapsis-within-tolerance", False, 539e3, 400e3, "m")) == 139e3
    assert check_margin(Check("insertion-raan-within-tolerance", True, 359.0, 1.0, "deg")) == pytest.approx(-2.0)
    assert check_margin(Check("mass-non-negative", True, 1.0, 0.0, "kg")) is None
    assert check_margin(Check("max-q-within-limit", False, None, 60000.0, "Pa")) is None


def test_trajectory_profiles_and_overlay(pair):
    a, b = pair
    profiles = dict(trajectory_profiles(a.run))
    assert set(profiles) == {t.body_id for t in a.run.tracks}
    assert profiles["stack"][0][0] == pytest.approx(0.0, abs=1e-9)
    assert all(alt > -10 for _d, alt in profiles["upper_stage"])

    merged = overlay_channels(build_telemetry_channels(a.run), build_telemetry_channels(b.run))
    alt = next(c for c in merged if c.id == "altitude")
    styles = {s.label.split()[0]: s.style for s in alt.series}
    assert styles == {"A": "solid", "B": "dash"}
    assert len(alt.series) == len(a.run.tracks) + len(b.run.tracks)
    only_b = overlay_channels((), build_telemetry_channels(b.run))
    assert {s.style for c in only_b for s in c.series} == {"dash"}


def test_discover_runs_and_montecarlo_layouts(tmp_path, pair):
    runs = tmp_path / "cache" / "runs"
    done = make_ascent_only(RUN_FIXTURE, runs / "abc123")
    busy = make_ascent_only(RUN_FIXTURE, runs / "def456")
    (busy / "_designer-in-progress").write_text("1")
    found = discover_runs(tmp_path / "cache")
    assert [p for _l, p in found] == [done]
    assert "recovery=none" in found[0][0]
    assert discover_runs(tmp_path / "nowhere") == []

    # FL-304 CLI layout: -o X.json next to the X.ascent/ run directory
    ascent = make_ascent_only(RUN_FIXTURE, tmp_path / "mc.ascent")
    (tmp_path / "mc.json").write_text(json.dumps({"n": 1, "success_rate": 1.0}))
    assert find_montecarlo(ascent) == tmp_path / "mc.json"
    assert load_bundle(ascent).montecarlo == {"n": 1, "success_rate": 1.0}
    assert find_montecarlo(pair[1].run_dir) == pair[1].run_dir / "montecarlo.json"


# --------------------------------------------------------------------------- Qt


@pytest.fixture
def stub_pair(tmp_path, monkeypatch):
    monkeypatch.setenv("ORBITOPT_CACHE_DIR", str(tmp_path / "cache"))
    flown = fly_stub_run(tmp_path / "flown", tmp_path, monkeypatch)
    a = make_ascent_only(flown, tmp_path / "expendable", payload_kg=15000.0)
    b = make_ascent_recovery(flown, tmp_path / "rtls", payload_kg=12600.0)
    return a, b


@pytest.mark.slow
def test_stub_recovery_variant_is_a_valid_run(stub_pair):
    _a, b = stub_pair
    run = read_run(b)
    assert {t.body_id for t in run.tracks} >= {"stage1", "stage2", "booster"}
    assert "stage1_debris" not in {t.body_id for t in run.tracks}
    booster = next(t for t in run.tracks if t.body_id == "booster")
    assert booster.extra_columns["altitude_m"][-1] == pytest.approx(0.0, abs=1.0)


@pytest.mark.slow
def test_compare_page_ascent_only_vs_recovery(stub_pair):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from orbitopt.flightlab.ui.compare_page import FlightLabComparePage

    app = QApplication.instance() or QApplication([])
    a, b = stub_pair
    page = FlightLabComparePage()
    page.resize(1400, 900)
    page.show()
    app.processEvents()
    try:
        assert page.set_runs(a, b)
        app.processEvents()
        assert page.result is not None
        assert "Recovery reserve cost 2,400 kg payload" in page.headline_label.text()
        keys = page.diff_keys()
        assert "vehicle.payload_kg" in keys and "recovery.booster_prop_kg" in keys
        assert "vehicle.s1.isp_s" not in keys  # unchanged rows hidden by default
        page.changed_only.setChecked(False)
        assert "vehicle.s1.isp_s" in page.diff_keys()
        # overlaid trajectories: A solid + B dashed, booster only in B
        labels = [s[0] for s in page.profile_chart.series]
        assert "A stage2" in labels and "B stage2" in labels and "B booster" in labels
        assert {s[2] for s in page.profile_chart.series if s[0].startswith("B")} == {"dash"}
        alt = page.charts["altitude"]._channel
        assert {s.style for s in alt.series} == {"solid", "dash"}
        # plain-language reasons + recovery panels
        assert page.reasons_a.entries and all(e.known for e in page.reasons_a.entries)
        assert "Apoapsis off target" in page.reasons_a.toPlainText()
        assert "远地点偏离目标" in page.reasons_a.toPlainText()
        assert page.recovery_b.summary is not None and page.recovery_b.summary.state == "flown"
        assert not page.recovery_b.mc_label.isHidden()
        assert "83%" in page.recovery_b.mc_label.text()
        assert page.recovery_a.summary.state == "none"
        page.swap()
        assert "run A recovers the booster" in page.headline_label.text()
        # ascent-only vs ascent-only also works
        assert page.set_run("B", a)
        assert page.result is not None and page.recovery_b.summary.state == "none"
        assert not page.set_run("A", Path(a).parent / "missing")
        assert "Could not open" in page.status_label.text()
    finally:
        page.close()
        app.processEvents()


@pytest.mark.slow
def test_replay_page_shows_reasons_and_recovery(stub_pair, monkeypatch):
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvista")
    pytest.importorskip("pyvistaqt")
    from PySide6.QtWidgets import QApplication

    from orbitopt.flightlab.ui import FlightLabReplayPage

    app = QApplication.instance() or QApplication([])
    _a, b = stub_pair
    page = FlightLabReplayPage()
    page.show()
    app.processEvents()
    try:
        page.load_run(b)
        app.processEvents()
        assert page.run_dir == Path(b)
        codes = [e.code for e in page.reason_panel.entries]
        assert "INSERTION_APOAPSIS_OUT_OF_TOLERANCE" in codes
        reason_cells = [page.check_table.item(r, 4).text() for r in range(page.check_table.rowCount())
                        if page.check_table.item(r, 1).text() == "FAIL"]
        assert any(cell.startswith("Apoapsis off target") for cell in reason_cells)
        tip = next(page.check_table.item(r, 4).toolTip() for r in range(page.check_table.rowCount())
                   if page.check_table.item(r, 1).text() == "FAIL"
                   and "APOAPSIS" in page.check_table.item(r, 4).text())
        assert "远地点偏离目标" in tip and "Fix:" in tip
        summary = page.recovery_panel.summary
        assert summary is not None and summary.state == "flown"
        assert page.recovery_panel.check_table.rowCount() == len(summary.checks)
        assert "LANDED" in page.recovery_panel.verdict_label.text()
    finally:
        page.hide()
        app.processEvents()


@pytest.mark.slow
def test_workbench_has_compare_tab(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvista")
    from PySide6.QtWidgets import QApplication

    from orbitopt.flightlab.ui import FlightLabWorkbench

    monkeypatch.setenv("ORBITOPT_CACHE_DIR", str(tmp_path / "cache"))
    app = QApplication.instance() or QApplication([])
    wb = FlightLabWorkbench()
    try:
        assert [wb.tabs.tabText(i) for i in range(wb.tabs.count())] == ["Designer", "Replay", "Compare"]
        run = make_ascent_only(RUN_FIXTURE, tmp_path / "r")
        wb.replay_page.load_run(run)
        app.processEvents()
        assert wb.compare_page.picker_a.combo.findData(str(run)) >= 0
    finally:
        wb.terminate_compute()
        wb.close()
        app.processEvents()
