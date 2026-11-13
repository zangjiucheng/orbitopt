"""M-04: the unified launch pipeline (window -> ascent optimization ->
injection verification) and its ``orbitopt launch`` entry point.

The window-infeasible and input-error paths need no compute stack and run
anywhere. The stages that fly use the recorded FL-004 stub (same fixture as
the FL-001/FL-201/FL-202 tests); because the recorded flight misses the
apoapsis band, the stub pipeline must come back NOT converged with a written
report -- the "no fake success" criterion. The end-to-end tests that need
pygmo (and would converge only with the real binary) are marked ``slow``."""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from orbitopt.cli import main
from orbitopt.mission import launch_pipeline as lp
from orbitopt.verify.report import read_report, validate_report

FIX = Path(__file__).parent / "fixtures" / "flightlab" / "opentsiolkovsky"
REPO = Path(__file__).resolve().parents[1]
VEHICLE = REPO / "src" / "orbitopt" / "data" / "flightlab" / "vehicles" / "synthetic_two_stage.yaml"
HAS_PYGENO = importlib.util.find_spec("pygmo") is not None


@pytest.fixture
def stub(monkeypatch, tmp_path):
    path = tmp_path / "otcli"
    shutil.copy(FIX / "stub_cli.py", path)
    path.chmod(0o755)
    monkeypatch.setenv("ORBITOPT_OPENTSIOLKOVSKY", str(path))
    monkeypatch.setenv("OT_STUB_FIXTURE", str(FIX / "synthetic_out"))
    monkeypatch.setenv("OT_STUB_SRC", str(REPO / "src" / "orbitopt"))
    return path


def _run(**kw):
    kw.setdefault("site", "kourou")
    kw.setdefault("vehicle", "synthetic_two_stage")
    kw.setdefault("orbit", "400x400@51.6")
    kw.setdefault("date_utc", "2026-11-10")
    return lp.run_launch_pipeline(**kw)


# --- inputs and window selection (no compute stack) ---------------------------

def test_unknown_site_is_an_input_error(tmp_path):
    with pytest.raises(KeyError):
        _run(out_dir=tmp_path / "o", site="nowhere-ville")


def test_missing_vehicle_file_is_an_input_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        _run(out_dir=tmp_path / "o", vehicle="/nope/two_stage.yaml")


def test_vehicle_name_resolves_against_the_packaged_catalog():
    assert lp._resolve_vehicle("synthetic_two_stage") == VEHICLE


def test_inclination_below_site_latitude_is_window_infeasible(tmp_path):
    result = _run(out_dir=tmp_path / "o", orbit="400x400@3.0")
    assert result.status == lp.STATUS_WINDOW_INFEASIBLE
    assert not result.ok
    assert result.reason_codes == ("WINDOW_INFEASIBLE",)
    assert "latitude" in result.message or "infeasible" in result.message
    report = json.loads((tmp_path / "o" / "launch.verification.json").read_text())
    validate_report(report)
    assert report["kind"] == "launch-pipeline"
    assert report["status"]["result"] == "failed"
    assert report["status"]["checks"]["window.selected"] is False
    assert report["targets"]["pipeline_status"] == lp.STATUS_WINDOW_INFEASIBLE
    assert report["targets"]["window"] is None and report["targets"]["injection"] is None


def test_target_orbit_elements_are_the_gate_b_altitude_checks_in_element_form():
    from orbitopt.flightlab.checks import EARTH_RADIUS_M
    from orbitopt.flightlab.target_export import build_target

    target = build_target("kourou", 400e3, 500e3, 51.6, date_utc="2026-11-10")
    el = lp.target_orbit_elements(target)
    assert el["semi_major_axis_m"] == pytest.approx(EARTH_RADIUS_M + 450e3)
    assert el["eccentricity"] == pytest.approx(100e3 / (2 * el["semi_major_axis_m"]))
    assert el["inclination_deg"] == pytest.approx(51.6)
    assert el["raan_deg"] is None


# --- pipeline with the recorded stub flight -----------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not HAS_PYGENO, reason="ascent optimization needs pygmo (conda-forge)")
def test_stub_pipeline_is_honest_and_writes_the_full_record(tmp_path, stub):
    out = tmp_path / "o"
    result = _run(out_dir=out, generations=4, population=8, local=False,
                  refly=False, timeout_s=120.0)
    # the recorded flight misses the apoapsis band: a red verdict is required
    assert result.status in (lp.STATUS_OPTIMIZATION_NOT_CONVERGED,
                             lp.STATUS_INJECTION_OUT_OF_TOLERANCE)
    assert not result.ok and result.message
    assert (out / "target.yaml").is_file() and (out / "ascent" / "manifest.json").is_file()
    report = read_report(result.report_path)
    validate_report(report)
    assert report["status"]["result"] == "failed"
    targets = report["targets"]
    assert targets["pipeline_status"] == result.status
    assert targets["window"]["liftoff_epoch_utc"]
    inj = targets["injection"]
    for key in ("semi_major_axis", "eccentricity", "inclination", "raan"):
        assert key in inj["checks"] and key in inj["tolerance"]
    assert all(err is not None for err in (inj["error"]["delta_semi_major_axis_m"],
                                           inj["error"]["delta_eccentricity"],
                                           inj["error"]["delta_inclination_deg"]))
    assert report["status"]["checks"]["injection.semi_major_axis_within_tolerance"] \
        == inj["checks"]["semi_major_axis"]


@pytest.mark.slow
@pytest.mark.skipif(not HAS_PYGENO, reason="ascent optimization needs pygmo (conda-forge)")
def test_cli_launch_reports_the_failure_reason(tmp_path, stub, capsys):
    out = tmp_path / "o"
    with pytest.raises(SystemExit) as exc:
        main(["launch", "--site", "kourou", "--vehicle", "synthetic_two_stage",
              "--orbit", "400x400@51.6", "--date", "2026-11-10",
              "-o", str(out), "--generations", "4", "--population", "8",
              "--no-local", "--no-refly"])
    assert exc.value.code == 1
    printed = capsys.readouterr().out
    assert "FAILED [OPTIMIZATION_NOT_CONVERGED]" in printed or \
        "FAILED [INJECTION_OUT_OF_TOLERANCE]" in printed
    assert "reason codes:" in printed and str(out / "launch.verification.json") in printed


def test_cli_launch_window_infeasible_exits_one_with_reason(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["launch", "--site", "kourou", "--vehicle", "synthetic_two_stage",
              "--orbit", "400x400@3.0", "--date", "2026-11-10", "-o", str(tmp_path)])
    assert exc.value.code == 1
    printed = capsys.readouterr().out
    assert "FAILED [WINDOW_INFEASIBLE]" in printed
    assert (tmp_path / "launch.verification.json").is_file()
