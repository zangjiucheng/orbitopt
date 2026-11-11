"""OpenTsiolkovsky v1.0.0 adapter (FL-004).

Fast tests need no OpenTsiolkovsky binary: they run the whole pipeline
(config build + schema validation, subprocess, CSV/JSON parsing, frame
conversion, body split, event mapping, verification) against the recorded
v1.0.0 outputs in tests/fixtures/flightlab/opentsiolkovsky/synthetic_out/
through a CLI stub pointed at by ORBITOPT_OPENTSIOLKOVSKY. The @slow test
re-flights the pinned example falcon9.json with the real binary and
reproduces its own summary.json key values.
"""
from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import pytest

from orbitopt.flightlab.adapters.opentsiolkovsky import (
    OTBinaryMissingError,
    AttitudePoint,
    Vehicle,
    build_config,
    find_binary,
    probe_version,
    run_ascent,
)
from orbitopt.flightlab.contract import load_target, read_run, validate_path

FIX = Path(__file__).parent / "fixtures" / "flightlab" / "opentsiolkovsky"
REPO = Path(__file__).resolve().parents[1]
VEHICLE = REPO / "src" / "orbitopt" / "data" / "flightlab" / "vehicles" / "synthetic_two_stage.yaml"
TARGET = REPO / "src" / "orbitopt" / "data" / "flightlab" / "scenarios" / "leo_two_stage.yaml"
EPOCH = "2026-11-10T12:00:00Z"


@pytest.fixture
def stub(monkeypatch, tmp_path):
    path = tmp_path / "otcli"
    shutil.copy(FIX / "stub_cli.py", path)
    path.chmod(0o755)
    monkeypatch.setenv("ORBITOPT_OPENTSIOLKOVSKY", str(path))
    monkeypatch.setenv("OT_STUB_FIXTURE", str(FIX / "synthetic_out"))
    monkeypatch.setenv("OT_STUB_SRC", str(REPO / "src" / "orbitopt"))
    return path


@pytest.fixture
def target():
    return load_target(TARGET)


def test_find_binary_honours_env_override(stub):
    assert find_binary() == stub


def test_missing_binary_is_actionable(monkeypatch, tmp_path):
    monkeypatch.setenv("ORBITOPT_OPENTSIOLKOVSKY", str(tmp_path / "definitely-not-here"))
    with pytest.raises(OTBinaryMissingError) as exc:
        find_binary()
    assert "fetch_opentsiolkovsky.py" in str(exc.value)


def test_version_mismatch_rejected(tmp_path, monkeypatch):
    bogus = tmp_path / "otcli"
    bogus.write_text("#!/bin/sh\necho 'openTsiolkovsky-cli 0.9.0'\n")
    bogus.chmod(0o755)
    monkeypatch.setenv("ORBITOPT_OPENTSIOLKOVSKY", str(bogus))
    with pytest.raises(OTBinaryMissingError, match="0.9.0"):
        probe_version(find_binary())


def test_adapter_git_sha_falls_back_to_schema_valid_hex_without_git(monkeypatch):
    from orbitopt.flightlab.adapters import opentsiolkovsky as adapter

    def _fail(*args, **kwargs):
        raise OSError("git missing")

    monkeypatch.setattr(adapter.subprocess, "run", _fail)
    adapter._git_sha.cache_clear()
    try:
        sha = adapter._git_sha()
        adapter._git_sha.cache_clear()
        assert re.fullmatch(r"[0-9a-f]{7,40}", sha)
        assert adapter._git_sha() == sha
    finally:
        adapter._git_sha.cache_clear()


def test_build_config_validates_and_wires_target(target):
    cfg = build_config(target, Vehicle.load(VEHICLE))
    assert cfg["launch"]["latitude_deg"] == pytest.approx(28.5619)
    assert cfg["launch"]["time_utc"] == "2026-11-10T12:00:00Z"
    assert cfg["stages"][0]["separation_time_s"] == pytest.approx(207.0)
    assert cfg["attitude"]["azimuth_deg"][-1][0] == pytest.approx(620.0)


def test_build_config_rejects_unknown_stage_key(target, tmp_path):
    text = VEHICLE.read_text(encoding="utf-8").replace(
        "separation_time_s: 207.0",
        "separation_time_s: 207.0\n    bogus_key: 1", 1)
    bad = tmp_path / "bad_vehicle.yaml"
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(Exception, match="bogus_key"):
        build_config(target, Vehicle.load(bad))


def test_vehicle_provenance_and_keys_validated(tmp_path):
    good = Vehicle.load(VEHICLE)
    assert good.provenance == "illustrative"
    bad = tmp_path / "v.yaml"
    bad.write_text(VEHICLE.read_text(encoding="utf-8").replace(
        "provenance: illustrative", "provenance: from-a-forum"), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        Vehicle.load(bad)
    bad2 = tmp_path / "v2.yaml"
    bad2.write_text("provenance: illustrative\nstages: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="name"):
        Vehicle.load(bad2)


def test_run_ascent_recorded_outputs(stub, target, tmp_path):
    out = tmp_path / "run"
    run = run_ascent(target, VEHICLE, None, out)
    validate_path(out)
    rr = read_run(out)
    assert rr.run_id == run.run_id and len(rr.tracks) == len(run.tracks)
    assert {tr.body_id for tr in rr.tracks} == {
        "stage1", "stage2", "stage1_debris", "stage2_fairing"}
    parents = {tr.body_id: tr.parent_body_id for tr in rr.tracks}
    assert parents == {"stage1": None, "stage2": "stage1",
                       "stage1_debris": "stage1", "stage2_fairing": "stage2"}
    assert all(tr.frame == "GCRS" and tr.epoch_utc == EPOCH for tr in rr.tracks)
    kinds = {e.kind for e in rr.events}
    assert {"liftoff", "max_q", "meco", "separation", "seco", "impact"} <= kinds
    ev = {(e.kind, e.body_id): e for e in rr.events}
    assert ev[("meco", "stage1")].t_s == pytest.approx(193.77, abs=0.01)
    assert ev[("separation", "stage2")].data["separated_from"] == "stage1_debris"
    assert ev[("separation", "stage2_fairing")].data["ot_event"] == "jettison"
    assert ev[("impact", "stage1_debris")].t_s == pytest.approx(739.94, abs=0.01)
    checks = {c.id: c for c in rr.verification.checks}
    assert checks["mass-non-negative"].passed
    assert checks["mass-continuity-stage2"].passed
    assert checks["max-q-within-limit"].value == pytest.approx(36690.2, rel=1e-3)
    assert rr.verification.mission_constraints_passed is True
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["engines"] == {"opentsiolkovsky-cli": "1.0.0"}
    assert any("sha256" in a for a in manifest["assumptions"])
    assert manifest["created_utc"] == EPOCH
    assert (out / "opentsiolkovsky" / "summary.json").exists()
    st2 = next(tr for tr in rr.tracks if tr.body_id == "stage2")
    assert st2.mass_kg[0] == pytest.approx(55000.0)
    assert min(min(tr.mass_kg) for tr in rr.tracks) >= 0.0


def test_run_is_pure_and_scratch_cleaned(stub, target, tmp_path):
    run_ascent(target, VEHICLE, None, tmp_path / "a")
    run_ascent(target, VEHICLE, None, tmp_path / "b")
    file_a = sorted(p for p in (tmp_path / "a").rglob("*") if p.is_file())
    file_b = sorted(p for p in (tmp_path / "b").rglob("*") if p.is_file())
    assert [p.relative_to(tmp_path / "a") for p in file_a] == \
        [p.relative_to(tmp_path / "b") for p in file_b]
    assert all(pa.read_bytes() == pb.read_bytes() for pa, pb in zip(file_a, file_b))
    assert not any(p.name.startswith(".ot-scratch-") for p in (tmp_path / "a").iterdir())


def test_explicit_attitude_program_matches_vehicle(stub, target, tmp_path):
    veh = Vehicle.load(VEHICLE)
    ra = run_ascent(target, VEHICLE, None, tmp_path / "v")
    rb = run_ascent(target, VEHICLE, list(veh.attitude_program), tmp_path / "p")
    assert ra.run_id == rb.run_id


@pytest.mark.slow
def test_falcon9_example_reproduction(tmp_path, monkeypatch):
    monkeypatch.delenv("ORBITOPT_OPENTSIOLKOVSKY", raising=False)
    try:
        binary = find_binary()
    except OTBinaryMissingError:
        pytest.skip("pinned OpenTsiolkovsky v1.0.0 binary not installed")
    probe_version(binary)
    out = tmp_path / "f9"
    run = run_ascent(load_target(FIX / "falcon9_target.yaml"),
                     FIX / "falcon9_vehicle.yaml", None, out, timeout_s=600)
    summary = json.loads((out / "opentsiolkovsky" / "summary.json").read_text(encoding="utf-8"))
    assert summary["maxima"]["dynamic_pressure_pa"]["value"] == pytest.approx(33660.33639, rel=1e-6)
    assert summary["maxima"]["dynamic_pressure_pa"]["time_s"] == pytest.approx(59.9427064, rel=1e-6)
    assert summary["final_state"]["mass_kg"] == pytest.approx(9618.651172, rel=1e-6)
    ev = {(e.kind, e.body_id): e.t_s for e in run.events}
    assert ev[("meco", "stage1")] == pytest.approx(139.0, abs=0.01)
    assert ev[("separation", "stage2")] == pytest.approx(143.0, abs=0.01)
    assert ev[("separation", "stage2_fairing")] == pytest.approx(255.0, abs=0.01)
    assert ev[("seco", "stage2")] == pytest.approx(510.34, abs=0.01)
    st2 = next(tr for tr in run.tracks if tr.body_id == "stage2")
    assert st2.mass_kg[-1] == pytest.approx(9618.651172, rel=1e-4)
    eci = math.sqrt(sum(x * x for x in summary["final_state"]["position_eci_m"]))
    gcr = math.sqrt(sum(x * x for x in st2.position_m[-1]))
    assert gcr == pytest.approx(eci, rel=1e-9)  # GCRS rotation preserves |r|
    checks = {c.id: c.passed for c in run.verification.checks}
    assert checks["mass-non-negative"] and checks["mass-continuity-stage2"]
