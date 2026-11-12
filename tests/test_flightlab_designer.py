"""FL-502 Mission Designer / Vehicle Builder core (Qt-free half of the task).

Everything here must run on a base install with no display and no PySide6:
the forms produce plain dicts, and the engine around them (validation,
derived preview numbers, input-hash cache, worker entry) stays Qt-free so it
is testable exactly like the contract/simulate layers."""
from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path

import pytest

import orbitopt.flightlab.designer as D
from orbitopt.flightlab.contract import FLIGHT_PROVENANCE_TAGS

REPO = Path(__file__).resolve().parents[1]
OT_FIX = Path(__file__).parent / "fixtures" / "flightlab" / "opentsiolkovsky"

GOOD_MISSION = {
    "site": "kourou",
    "periapsis_km": 400.0,
    "apoapsis_km": 400.0,
    "inclination_deg": 51.6,
    "raan_deg": None,
    "date": (dt.date.today() + dt.timedelta(days=30)).isoformat(),
    "window_days": 2.0,
    "recovery": "rtls",
    "recovery_reserve_kg": 1000.0,
    "max_q_kpa": 60.0,
    "max_g": 5.0,
    "min_periapsis_km": 380.0,
}


def _site(site_id: str):
    return next(s for s in D.mission_sites() if s.id == site_id)


def _stub(tmp_path: Path) -> Path:
    cli = tmp_path / "otcli"
    shutil.copy(OT_FIX / "stub_cli.py", cli)
    cli.chmod(0o755)
    return cli


def _stub_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ORBITOPT_OPENTSIOLKOVSKY", str(_stub(tmp_path)))
    monkeypatch.setenv("OT_STUB_FIXTURE", str(OT_FIX / "synthetic_out"))
    monkeypatch.setenv("OT_STUB_SRC", str(REPO / "src" / "orbitopt"))
    monkeypatch.setenv("ORBITOPT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("ORBITOPT_FL_DESIGNER_NO_PGRP", "1")


def test_target_from_form_builds_generator_provenance_target():
    target, errors = D.target_from_form(GOOD_MISSION)
    assert errors == {}
    assert target is not None
    doc = target.to_dict()
    assert doc["target_orbit"]["apoapsis_altitude_m"] == pytest.approx(400_000.0)
    assert doc["target_orbit"]["inclination_deg"] == pytest.approx(51.6)
    assert doc["recovery"]["kind"] == "rtls"
    assert any(a.startswith("generator:") for a in doc["assumptions"])


def test_target_from_form_reports_field_level_errors():
    cases = {
        "apoapsis_km": {"apoapsis_km": 100.0},
        "site": {"site": "not-a-spaceport"},
        "date": {"date": "not-a-date"},
        "inclination_deg": {"inclination_deg": 3.0, "site": "kourou"},
    }
    for field, patch in cases.items():
        form = dict(GOOD_MISSION, **patch)
        target, errors = D.target_from_form(form)
        assert target is None, field
        assert field in errors, (field, errors)
        assert errors[field]


def test_launch_azimuth_matches_site_inclination():
    site = _site("kourou")
    az = D.launch_azimuth(site, 51.6)
    assert az is not None
    assert abs(site.inclination_for_azimuth(az) - 51.6) < 0.2


def test_vehicle_presets_round_trip_through_the_form():
    presets = D.vehicle_presets()
    assert "synthetic_two_stage" in presets
    doc = D.load_vehicle_preset("synthetic_two_stage")
    assert doc["provenance"] in FLIGHT_PROVENANCE_TAGS
    form = D.vehicle_form_from_document(doc)
    rebuilt = D.vehicle_document(form)
    assert D.validate_vehicle_document(rebuilt) == {}
    # what the form owns must survive the trip exactly:
    assert rebuilt["name"] == doc["name"]
    assert rebuilt["provenance"] == doc["provenance"]
    assert rebuilt["simulation"] == doc["simulation"]
    assert rebuilt["attitude_program"] == json.loads(json.dumps(doc["attitude_program"]))
    assert len(form["stages"]) == len(doc["stages"]) == 2
    for a, b in zip(rebuilt["stages"], doc["stages"]):
        assert a["mass_initial_kg"] == pytest.approx(b["mass_initial_kg"])
        assert a["mass_dry_kg"] == pytest.approx(b["mass_dry_kg"])
        assert a.get("separation_time_s") == b.get("separation_time_s")
    # and re-editing the rebuilt doc must not churn the form:
    assert D._canonical(D.vehicle_form_from_document(rebuilt)) == D._canonical(form)


def test_validate_vehicle_document_maps_errors_to_fields():
    doc = D.load_vehicle_preset("synthetic_two_stage")
    assert D.validate_vehicle_document(doc) == {}

    bad = json.loads(json.dumps(doc))
    bad["stages"][0]["engine"]["thrust_vac_n"] = -5.0
    assert D.validate_vehicle_document(bad) == {
        "stages[0].engine.thrust_vac_n": "thrust must be positive"}

    missing_sep = json.loads(json.dumps(doc))
    missing_sep["stages"][0].pop("separation_time_s")
    assert "stages[0].separation_time_s" in D.validate_vehicle_document(missing_sep)

    swapped = json.loads(json.dumps(doc))
    swapped["stages"][1]["engine"]["burn_start_s"] = 100.0
    swapped["stages"][1]["engine"]["burn_end_s"] = 1.0
    assert "stages[1].engine.burn_end_s" in D.validate_vehicle_document(swapped)

    assert "stages" in D.validate_vehicle_document({"stages": []})


def test_derived_metrics_are_live_previews():
    doc = D.load_vehicle_preset("synthetic_two_stage")
    m = D.derived_metrics(doc)
    assert m["liftoff_thrust_to_weight"] == pytest.approx(1.83, abs=0.05)
    assert m["liftoff_tw_ok"]
    assert len(m["stages"]) == 2
    for stage in m["stages"]:
        assert stage["ideal_delta_v_m_s"] > 0
        assert stage["propellant_kg"] > 0
        assert stage["reference_area_m2"] > 0
    assert m["stages"][0]["sea_level_thrust_n"] > 0


def test_build_job_digest_is_input_hash_and_cache_tracks_completeness(tmp_path, monkeypatch):
    monkeypatch.setenv("ORBITOPT_CACHE_DIR", str(tmp_path / "cache"))
    target, errors = D.target_from_form(GOOD_MISSION)
    assert errors == {}
    doc = D.load_vehicle_preset("synthetic_two_stage")

    digest_a, job_a = D.build_job(target, doc, {"timeout_s": 60.0})
    digest_b, _ = D.build_job(target, doc, {"timeout_s": 60.0})
    assert digest_a == digest_b
    digest_c, _ = D.build_job(target, doc, {"timeout_s": 60.0, "refly": False})
    assert digest_a != digest_c  # options are part of the identity
    assert job_a["run_dir"] == str(D.cache_root() / "runs" / digest_a)
    assert D.cached_run_dir(digest_a) is None  # not even created yet

    run_dir = Path(job_a["run_dir"])
    run_dir.mkdir(parents=True)
    assert D.cached_run_dir(digest_a) is None  # incomplete run is not a cache hit
    (run_dir / D.IN_PROGRESS).write_text("1", encoding="ascii")
    (run_dir / "manifest.json").write_text("{}", encoding="ascii")
    assert D.cached_run_dir(digest_a) is None  # marker means "in flight", never cached
    (run_dir / D.IN_PROGRESS).unlink()
    assert D.cached_run_dir(digest_a) == run_dir


def test_run_form_job_flies_caches_and_leaves_form_audit(tmp_path, monkeypatch):
    _stub_env(tmp_path, monkeypatch)
    target, _ = D.target_from_form(GOOD_MISSION)
    doc = D.load_vehicle_preset("synthetic_two_stage")
    digest, job = D.build_job(target, doc, {"timeout_s": 60.0},
                              mission_form=GOOD_MISSION,
                              vehicle_form=D.vehicle_form_from_document(doc))
    result = D.run_form_job(job)
    run_dir = Path(result["run_dir"])
    assert result["run_dir"] == str(D.cached_run_dir(digest))  # immediately cacheable
    assert not (run_dir / D.IN_PROGRESS).exists()
    assert (run_dir / "vehicle.yaml").exists()
    audit = (run_dir / "designer-form.json").read_text(encoding="utf-8")
    assert "kourou" in audit and digest[:8] in audit
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == result["run_id"]
