"""E12-06 mission studio integration: the LEO-RDV-001 replay.

The studio bridge turns the reference rendezvous plan into a FlightResult v1
run dir (plus mission/replay/sidecar artefacts) that Flight Lab can replay
end to end. These tests pin the three acceptance criteria: the run replays,
the timeline carries the maneuvers and safety verdicts, and the relative
curves end on exactly the verification report's arrival numbers.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from orbitopt.flightlab.contract import read_run
from orbitopt.flightlab.ui.telemetry import build_telemetry_channels

RUN_FIXTURE = Path(__file__).parent / "data" / "flightlab" / "synthetic_run"


@pytest.fixture(scope="module")
def studio_dir(tmp_path_factory):
    from orbitopt.rendezvous.config import load_reference_config
    from orbitopt.rendezvous.studio import build_studio_run, write_studio_run

    root = tmp_path_factory.mktemp("e12-06") / "leo-rdv-001-studio"
    write_studio_run(root, build_studio_run(load_reference_config()))
    return root


def _sidecar(studio_dir: Path) -> dict:
    return json.loads((studio_dir / "rendezvous.json").read_text(encoding="utf-8"))


def test_studio_run_roundtrips_and_all_checks_pass(studio_dir):
    run = read_run(studio_dir)
    assert run.run_id == "leo-rdv-001-studio"
    assert [track.body_id for track in run.tracks] == ["target", "chaser"]
    assert all(track.t_end_s > track.t_s[0] for track in run.tracks)
    assert all(check.passed for check in run.verification.checks)
    assert any(check.id == "keepout-respected" for check in run.verification.checks)
    # flight-result-1.0 events are ascent-kind only: maneuvers live sidecar-only
    assert run.events == ()
    # E12-09 (task #261 rule): no independent re-fly evidence was carried --
    # physics_verified is NOT a hard-coded True; it records "not assessed".
    assert run.verification.physics_verified is None
    # the landing residual (4.9e-7 m) within the config's Newton tolerance is
    # what optimizer_converged records; not a literal True
    assert run.verification.optimizer_converged is True


def test_studio_plan_matches_the_reference_verification(studio_dir):
    from orbitopt.rendezvous.canonical import canonical_plan
    from orbitopt.rendezvous.config import load_reference_config

    plan = canonical_plan(load_reference_config()).plan
    sidecar = _sidecar(studio_dir)
    doc = sidecar["plan"]
    assert doc["drift_end_s"] == plan.drift_end_s
    assert doc["transfer_tof_s"] == plan.tof_s
    assert doc["arrival_epoch_s"] == plan.arrival_epoch_s
    assert doc["total_delta_v_m_s"] == pytest.approx(plan.total_delta_v_m_s, abs=1e-12)
    assert sidecar["verification"]["passed"] is True
    assert [m["magnitude_m_s"] for m in doc["maneuvers"]] == \
           [pytest.approx(m.magnitude_m_s, abs=1e-12) for m in plan.maneuvers]
    # E12-08: the replay carries the sealed plan's identity and is the E12-04
    # safety-constrained canonical plan: the safety summary is present, it
    # passed, and it records the same plan digest the sidecar does.
    assert sidecar["plan_digest"] == canonical_plan(load_reference_config()).plan_digest
    safety = sidecar["safety"]
    assert safety is not None and safety["passed"]
    assert safety["plan_digest"] == sidecar["plan_digest"]


def test_timeline_events_cover_both_maneuvers(studio_dir):
    sidecar = _sidecar(studio_dir)
    events = sidecar["events"]
    assert [event["kind"] for event in events] == ["maneuver", "hold_point_arrival"]
    assert events[0]["t_s"] == pytest.approx(sidecar["plan"]["drift_end_s"])
    assert events[1]["t_s"] == pytest.approx(sidecar["plan"]["arrival_epoch_s"])
    assert "transfer injection" in events[0]["title"].lower()
    assert "hold-point insertion" in events[1]["title"].lower()
    keepout = sidecar["keepout"]
    assert keepout["respected"] and keepout["min_range_m"] > keepout["radius_m"]


def test_relative_curves_end_on_report_numbers_to_the_bit(studio_dir):
    run = read_run(studio_dir)
    channels = {channel.id: channel for channel in build_telemetry_channels(run)}
    ver = _sidecar(studio_dir)["verification"]
    expected = {
        "rel_range": abs(ver["arrival_rel_lvlh_m"][1]),
        "rel_speed": ver["relative_speed_m_s"],
        "rel_x_lvlh": ver["arrival_rel_lvlh_m"][0],
        "rel_y_lvlh": ver["arrival_rel_lvlh_m"][1],
        "rel_z_lvlh": ver["arrival_rel_lvlh_m"][2],
    }
    for channel_id, value in expected.items():
        series = channels[channel_id].series
        assert series, channels[channel_id].missing_reason
        assert series[0].points[-1][1] == value, channel_id
    # the approach curve starts at the designed separation and closes to the hold point
    first = channels["rel_range"].series[0].points[0][1]
    assert first > expected["rel_range"] * 2


def test_launch_runs_keep_the_launch_channel_set(studio_dir):
    launch = read_run(RUN_FIXTURE)
    ids = {channel.id for channel in build_telemetry_channels(launch)}
    assert not any(cid.startswith("rel_") for cid in ids)


def test_mission_and_replay_documents_validate(studio_dir):
    from orbitopt.mission.multispacecraft import (
        is_valid_mission_document,
        is_valid_replay_document,
        mission_from_dict,
    )

    mission_doc = json.loads((studio_dir / "mission.json").read_text(encoding="utf-8"))
    replay_doc = json.loads((studio_dir / "replay.json").read_text(encoding="utf-8"))
    assert is_valid_mission_document(mission_doc)
    assert is_valid_replay_document(replay_doc)
    mission = mission_from_dict(mission_doc)
    assert {zone.name for zone in mission.zones} == {"keepout", "hold-point"}
    assert [event.kind for event in mission.events] == ["maneuver", "hold_point_arrival"]
    assert len(mission.spacecraft) == 2


def test_scene_renders_markers_and_maneuver_events(studio_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("ORBITOPT_CACHE_DIR", str(tmp_path / "cache"))
    from orbitopt.flightlab.ui.telemetry import load_run_scene

    run, scene = load_run_scene(studio_dir, earth_texture=False)
    markers = {body["id"]: body for body in scene["bodies"]
               if body.get("kind") == "site"}
    assert {"keepout_center", "hold_point"} <= set(markers)
    assert all(body["hideTrail"] for body in markers.values())
    labels = [event["label"] for event in scene["timeline"]["events"]]
    assert any("transfer injection" in label.lower() for label in labels)
    assert any("hold-point insertion" in label.lower() for label in labels)
    assert run.run_id == "leo-rdv-001-studio"


def _qt_app():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_flightlab_panels_replay_the_rendezvous(studio_dir, tmp_path, monkeypatch):
    """Acceptance: the run opens in Flight Lab -- key-event rows for both
    maneuvers, the Rendezvous tab's PASS verdict, and enabled rel-state plots;
    a launch run's panel shows neither."""
    app = _qt_app()
    monkeypatch.setenv("ORBITOPT_CACHE_DIR", str(tmp_path / "cache"))
    from orbitopt.flightlab.ui.flight_panel import (
        FlightRunPanel,
        TelemetryPanel,
        load_flight_bundle,
    )
    from orbitopt.flightlab.ui.telemetry import load_run_scene

    run, scene = load_run_scene(studio_dir, earth_texture=False)
    bundle, target, channels = load_flight_bundle(studio_dir, run=run, scene=scene)
    assert bundle.rendezvous is not None
    panel = FlightRunPanel()
    panel.set_run(bundle, target, channels)
    app.processEvents()
    titles = [panel.event_table.item(row, 1).text()
              for row in range(panel.event_table.rowCount())]
    assert any("transfer injection" in title.lower() for title in titles)
    assert any("hold-point insertion" in title.lower() for title in titles)
    assert [event.t_s for event in panel.events] == sorted(event.t_s for event in panel.events)
    verdict = panel.rendezvous_panel.verdict_label.text()
    assert "PASS" in verdict and "leo-rdv-001" in verdict
    assert panel.rendezvous_panel.table.rowCount() > 0

    telemetry = TelemetryPanel(settings=_scratch_settings(tmp_path))
    telemetry.set_channels(channels, 0.0, run.tracks[0].t_end_s)
    for channel_id in ("rel_range", "rel_speed", "rel_x_lvlh", "rel_y_lvlh", "rel_z_lvlh"):
        assert telemetry._plot_actions[channel_id].isEnabled(), channel_id

    panel.clear()
    assert panel.rendezvous_panel.table.rowCount() == 0
    assert "No rendezvous run loaded." in panel.rendezvous_panel.verdict_label.text()

    launch_run, launch_scene = load_run_scene(RUN_FIXTURE, earth_texture=False)
    launch_bundle, _, launch_channels = load_flight_bundle(RUN_FIXTURE, run=launch_run,
                                                           scene=launch_scene)
    assert launch_bundle.rendezvous is None
    panel.set_run(launch_bundle, None, launch_channels)
    assert panel.rendezvous_panel.table.rowCount() == 0
    launch_telemetry = TelemetryPanel(settings=_scratch_settings(tmp_path / "launch"))
    launch_telemetry.set_channels(launch_channels, 0.0, launch_run.tracks[0].t_end_s)
    assert not launch_telemetry._plot_actions["rel_range"].isEnabled()


def _scratch_settings(tmp_path):
    from PySide6.QtCore import QSettings

    return QSettings(str(tmp_path / "plots.ini"), QSettings.Format.IniFormat)


def test_cli_rendezvous_studio_writes_a_replayable_run(tmp_path):
    from orbitopt.cli import main

    out = tmp_path / "cli-run"
    main(["rendezvous", "studio", "-o", str(out), "--dt", "20", "--json"])
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == "leo-rdv-001-studio"
    run = read_run(out)
    assert all(check.passed for check in run.verification.checks)
    assert (out / "rendezvous.json").is_file()


def test_safety_gates_are_graded_as_run_checks(studio_dir):
    run = read_run(studio_dir)
    ids = {c.id for c in run.verification.checks}
    assert {"safety-keep-out", "safety-terminal-cone", "safety-cross-track",
            "safety-approach-speed", "safety-fuel-budget", "safety-time-window",
            "safety-passive-releases"} <= ids
    assert all(c.passed for c in run.verification.checks)
    from orbitopt.rendezvous.config import load_reference_config
    sidecar = _sidecar(studio_dir)
    assert sidecar["keepout"]["radius_m"] == load_reference_config().safety.keep_out_radius_m


def test_unsafe_custom_config_produces_no_green_run(tmp_path):
    """E12-08: a config whose fixed safety constraints no plan can meet must
    fail at planning, not render a run -- Mission Studio cannot show green
    for a mission the safety gate refuses."""
    import re

    import pytest

    from importlib import resources as imp_resources

    from orbitopt.cli import main
    from orbitopt.rendezvous.planner import RendezvousPlanError

    raw = imp_resources.files("orbitopt").joinpath("data", "leo_rdv_001.yaml").read_text()
    bad = re.sub(r"transfer_tof_min_s: \S+", "transfer_tof_min_s: 1600.0", raw)
    bad = re.sub(r"transfer_tof_max_s: \S+", "transfer_tof_max_s: 1900.0", bad)
    assert bad != raw
    path = tmp_path / "unsafe-leo-rdv-001.yaml"
    path.write_text(bad, encoding="utf-8")
    out = tmp_path / "unsafe-run"
    with pytest.raises(RendezvousPlanError):
        main(["rendezvous", "studio", "--config", str(path), "-o", str(out)])
    assert not (out / "rendezvous.json").exists()


def test_cli_can_replay_a_sealed_plan_document(tmp_path):
    from orbitopt.cli import main
    from orbitopt.rendezvous import canonical_plan, load_reference_config, write_sealed_plan

    sealed = canonical_plan(load_reference_config())
    doc = tmp_path / "leo-rdv-001.sealed.json"
    write_sealed_plan(sealed, doc)
    out = tmp_path / "cli-run"
    main(["rendezvous", "studio", "--sealed-plan", str(doc), "-o", str(out), "--json"])
    sidecar = json.loads((out / "rendezvous.json").read_text(encoding="utf-8"))
    assert sidecar["plan_digest"] == sealed.plan_digest
    assert sidecar["safety"]["passed"] is True


class TestPhysicsVerifiedFromEvidence:
    """E12-09: ``physics_verified`` is exported from evidence, never hard-coded.
    True only when a rendezvous-refly report for THIS plan (matching maneuver
    digest) says the independent-model line passed; a failed line is False;
    missing, foreign or digest-mismatched evidence is not assessed (None).
    The report keys on the plan the studio actually flies -- the canonical
    sealed plan (E12-08)."""

    @staticmethod
    def _reference_plan():
        from orbitopt.rendezvous import canonical_plan, load_reference_config

        return canonical_plan(load_reference_config()).plan

    @staticmethod
    def _report(plan, result="passed"):
        from orbitopt.verify.rendezvous_refly import REPORT_KIND, maneuvers_digest

        return {"schemaVersion": "1.0", "kind": REPORT_KIND,
                "inputs": {"mission_id": plan.mission_id,
                           "main_model": {"maneuvers_sha256": maneuvers_digest(plan)},
                           "error_sources": {
                               "independent_model_agreement": {"result": result}}}}

    @staticmethod
    def _build(refly_report):
        from orbitopt.rendezvous.config import load_reference_config
        from orbitopt.rendezvous.studio import build_studio_run

        return build_studio_run(load_reference_config(), sample_dt_s=60.0,
                                refly_report=refly_report)

    def test_no_evidence_is_not_assessed(self):
        assert self._build(None).run.verification.physics_verified is None

    def test_matching_passed_report_verifies(self):
        assert self._build(self._report(self._reference_plan(), "passed")
                           ).run.verification.physics_verified is True

    def test_matching_failed_report_falsifies(self):
        assert self._build(self._report(self._reference_plan(), "failed")
                           ).run.verification.physics_verified is False

    def test_not_checked_report_is_not_assessed(self):
        assert self._build(self._report(self._reference_plan(), "not_checked")
                           ).run.verification.physics_verified is None

    def test_foreign_plan_digest_is_not_assessed(self):
        doc = self._report(self._reference_plan())
        doc["inputs"]["main_model"]["maneuvers_sha256"] = "0" * 64
        assert self._build(doc).run.verification.physics_verified is None

    def test_foreign_mission_is_not_assessed(self):
        doc = self._report(self._reference_plan())
        doc["inputs"]["mission_id"] = "not-my-mission"
        assert self._build(doc).run.verification.physics_verified is None

    def test_report_path_is_read(self, tmp_path):
        path = tmp_path / "rendezvous-refly-report.json"
        path.write_text(json.dumps(self._report(self._reference_plan())), encoding="utf-8")
        assert self._build(path).run.verification.physics_verified is True
