"""E13-08 acceptance gate -- GNC Mission Studio replay.

A sealed closed-loop GNC run replays as a Flight Lab run: truth vs estimate
tracks, report-consistent error/covariance/NEES curves, dropped-frame events,
covariance ellipsoid scenes, and two evidence lines that are never blended --
the local replay digest and the E13-07 independent report stand on their own,
and a not-checked error class is shown as not-checked, never as a pass."""
import importlib.util
import json
import math

import pytest

import orbitopt.verify.gnc_closed_loop as gl
from orbitopt.gnc import (ClockSpec, ControllerSpec, EstimatorSpec, ProportionalLaw, SensorNoise,
                          SensorSpec, simulate, write_run_log)
from orbitopt.gnc.studio import build_gnc_studio_run, write_gnc_studio_run
from orbitopt.flightlab.compare import load_side_files
from orbitopt.flightlab.contract import read_run
from orbitopt.mission.contract import Epoch

T0 = Epoch(9300.0, "TDB")


HAS_TUDATPY = importlib.util.find_spec("tudatpy") is not None


def make_clock():
    return ClockSpec(
        t0=T0, dt_dynamics_s=5.0,
        sensors=(SensorSpec("navcam", period_s=2.0, phase_s=0.5, latency_s=0.3,
                            dropped_ticks=(3, 9)),),
        estimator=EstimatorSpec(period_s=1.0, phase_s=0.0),
        controller=ControllerSpec(period_s=4.0, phase_s=0.0, latency_s=1.0))


def mission_doc():
    from orbitopt.mission.multispacecraft import MultiSpacecraftMission, SpacecraftSpec
    m = MultiSpacecraftMission(
        mission_id="e13-08-sim", title="closed-loop studio replay", frame="GCRS",
        epoch=T0, spacecraft=(SpacecraftSpec("tgt", "TARGET"), SpacecraftSpec("chaser", "CHASER")))
    return m.to_dict()


def sim_log(run_id="e13-08-run", seed=7):
    return simulate(
        run_id=run_id, clock=make_clock(), mission=mission_doc(),
        frame="GCRS", target_id="tgt", chaser_id="chaser",
        chief_r_m=(7000000.0, 0.0, 0.0), chief_v_m_s=(0.0, 7546.1, 0.0),
        deputy_r_m=(7000040.0, 0.0, 15.0), deputy_v_m_s=(0.0, 7546.1, 0.02),
        laws={"chaser": ProportionalLaw(kp_per_s=0.01, kd=0.2)},
        noise={"navcam": SensorNoise(std_position_m=1.0, std_velocity_m_s=0.02, seed=seed)},
        t_end_s=80.0, execution_scale=0.95, execution_rtol=0.05, execution_seed=3)


@pytest.fixture(scope="module")
def log():
    return sim_log()


@pytest.fixture(scope="module")
def report(log):
    return gl.verify_gnc_closed_loop(log)


@pytest.fixture(scope="module")
def studio_dir(log, report, tmp_path_factory):
    studio = build_gnc_studio_run(log, verification_report=report)
    return write_gnc_studio_run(tmp_path_factory.mktemp("e13-08") / "studio", studio)


def _scored_rows(log):
    nav = gl.navigation_consistency(log)
    return [r for r in nav["estimates"]["rows"] if r.get("gate_passed") is not None]


def test_studio_run_round_trips_as_flight_run(studio_dir):
    run = read_run(studio_dir)
    assert {t.body_id for t in run.tracks} == {"target", "chaser", "chaser_est"}
    assert (studio_dir / "gnc.json").is_file()
    ids = {check.id for check in run.verification.checks}
    assert {"gnc-numerical-error", "gnc-navigation-inconsistency",
            "gnc-model-error", "gnc-optimizer-control-nonconvergence"} <= ids


def test_replay_digest_and_report_digest_are_separate_evidence_lines(studio_dir, log, report):
    sc = json.loads((studio_dir / "gnc.json").read_text(encoding="utf-8"))
    assert sc["schema_version"] == "gnc-studio-1.0"
    assert sc["local_replay"]["run_digest"] == log.digest()
    assert sc["independent_verification"]["report_digest"] == report["digest"]
    assert "NOT independent" in sc["local_replay"]["source"]
    assert sc["local_replay"]["run_id"] == log.run_id


def test_missing_independent_backend_never_reads_as_pass(studio_dir):
    sc = json.loads((studio_dir / "gnc.json").read_text(encoding="utf-8"))
    model = sc["independent_verification"]["error_classes"]["model_error"]
    run = read_run(studio_dir)
    check = next(c for c in run.verification.checks if c.id == "gnc-model-error")
    if HAS_TUDATPY:
        # the independent re-flight ran: the line is a real verdict, never a placeholder
        assert model["result"] == "passed"
        assert check.passed is True
        assert run.verification.physics_verified is True
    else:
        assert model["result"] == "not_checked"
        assert model.get("reason")
        assert check.passed is False
        assert check.reason_code == "not_checked"
        assert run.verification.physics_verified is None


def test_curves_are_the_report_rows_to_the_bit(studio_dir, log):
    """Acceptance: the replay curves carry the same numbers the report scored."""
    run = read_run(studio_dir)
    est = next(t for t in run.tracks if t.body_id == "chaser_est")
    cols = est.extra_columns
    rows = _scored_rows(log)
    assert len(rows) == len(est.t_s)
    for r, t, pe, ve, nis in zip(rows, est.t_s, cols["gnc_est_pos_err_m"],
                                 cols["gnc_est_vel_err_m_s"], cols["gnc_nees_per_dof"]):
        assert abs(t - r["t_s"]) < 1e-9
        assert math.isclose(pe, r["position_error_m"], rel_tol=1e-12)
        assert math.isclose(ve, r["velocity_error_m_s"], rel_tol=1e-12)
        assert math.isclose(nis, r["nis"] / r["dof"], rel_tol=1e-12)


def test_telemetry_channels_carry_the_gnc_curves(studio_dir):
    from orbitopt.flightlab.ui.telemetry import build_telemetry_channels
    chans = {c.id: c for c in build_telemetry_channels(read_run(studio_dir))
             if c.id.startswith("gnc_")}
    assert {"gnc_est_pos_err", "gnc_est_vel_err", "gnc_nees_per_dof",
            "gnc_cov_trace", "gnc_cov_1sigma"} == set(chans)
    for chan in chans.values():
        assert len(chan.series) == 1
        assert len(chan.series[0].points) == 80


def test_dropped_frames_surface_as_events_and_accounting(studio_dir):
    sc = json.loads((studio_dir / "gnc.json").read_text(encoding="utf-8"))
    drops = [e for e in sc["events"] if e.get("kind") == "gnc_dropout"]
    assert [round(e["t_s"], 3) for e in drops] == [6.5, 18.5]
    assert sum(int(s["dropped"]) for s in sc["measurement"]["sensors"].values()) == 2
    nav_cam = sc["measurement"]["sensors"]["navcam"]
    assert nav_cam["dropped"] == len(nav_cam["declared_dropped_ticks"]) == 2
    assert nav_cam["expected_ticks"] - nav_cam["received"] == nav_cam["dropped"]
    burn_events = [e for e in sc["events"] if e.get("kind") == "gnc_burn"]
    assert burn_events and all(e["t_s"] > 0 for e in burn_events)


def test_sidecar_binds_into_the_compare_bundle(studio_dir):
    side = load_side_files(studio_dir)
    assert side["gnc"] is not None
    assert side["gnc"]["local_replay"]["run_digest"] \
        == json.loads((studio_dir / "gnc.json").read_text(encoding="utf-8"))["local_replay"]["run_digest"]


def test_scene_replays_ellipsoids_and_gnc_events(studio_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("ORBITOPT_CACHE_DIR", str(tmp_path / "cache"))
    from orbitopt.flightlab.ui.telemetry import load_run_scene
    run, scene = load_run_scene(studio_dir, earth_texture=False)
    covs = [b for b in scene["bodies"] if b["id"].startswith("gnc_cov_")]
    assert covs and all(b.get("semiAxesKm") and len(b["semiAxesKm"]) == 3 for b in covs)
    assert all(b.get("tStart") is not None and b.get("tEnd") > b["tStart"] for b in covs)
    labels = [event["label"] for event in scene["timeline"]["events"]]
    assert any("dropped" in label.lower() for label in labels)
    assert any("burn" in label.lower() for label in labels)


def test_panel_rows_keep_the_two_evidence_lines(studio_dir):
    pytest.importorskip("PySide6")
    from orbitopt.flightlab.ui.gnc_panel import gnc_rows
    sc = json.loads((studio_dir / "gnc.json").read_text(encoding="utf-8"))
    rows = dict(((label, value) for label, value, _ in gnc_rows(sc)))
    assert rows["Run log (local replay)"].startswith("e13-08-run")
    model_row = rows["Model error (independent re-flight)"]
    assert ("PASS" in model_row) if HAS_TUDATPY else ("NOT CHECKED" in model_row)
    assert "none consumed" not in rows["Verification report (independent)"]


def test_panel_rows_mark_a_missing_report_as_a_gap_not_a_pass(studio_dir):
    pytest.importorskip("PySide6")
    from orbitopt.flightlab.ui.gnc_panel import gnc_rows
    sc = json.loads((studio_dir / "gnc.json").read_text(encoding="utf-8"))
    sc["independent_verification"] = None
    rows = dict(((label, value) for label, value, _ in gnc_rows(sc)))
    assert rows["Verification report (independent)"] == "none consumed"


def test_cli_gnc_studio_replays_a_sealed_log(log, tmp_path):
    from orbitopt.cli import main
    log_path = tmp_path / "run.json"
    write_run_log(log, log_path)
    out = tmp_path / "studio"
    main(["gnc", "studio", "--run-log", str(log_path), "-o", str(out), "--json"])
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == "e13-08-run-studio"
    assert read_run(out) is not None
    assert (out / "gnc.json").is_file()
    side = json.loads((out / "gnc.json").read_text(encoding="utf-8"))
    assert side["independent_verification"] is None  # no --report: honest gap, no fake green


def test_cli_gnc_studio_binds_a_report_and_rejects_a_mismatched_one(log, tmp_path):
    from orbitopt.cli import main
    log_path = tmp_path / "run.json"
    write_run_log(log, log_path)
    rep_path = tmp_path / "report.json"
    gl.verify_gnc_closed_loop(log, out_path=rep_path)
    out = tmp_path / "studio"
    main(["gnc", "studio", "--run-log", str(log_path), "--report", str(rep_path),
          "-o", str(out), "--json"])
    side = json.loads((out / "gnc.json").read_text(encoding="utf-8"))
    assert side["independent_verification"]["report_digest"]

    other = sim_log(run_id="e13-08-other", seed=8)
    other_path = tmp_path / "other.json"
    write_run_log(other, other_path)
    with pytest.raises(ValueError, match="evidence binding mismatch"):
        main(["gnc", "studio", "--run-log", str(other_path), "--report", str(rep_path),
              "-o", str(tmp_path / "nope"), "--json"])
