"""FL-202: `orbitopt flight optimize` -- resolver, solve loop, honest verdict.

The recorded FL-004 stub (same fixture as the FL-001/FL-201 tests) makes the
loop run without the real binary; because the recorded flight misses the
apoapsis band, a stub solve must end ``optimizer_converged=false`` with
reason codes -- the "no fake success" acceptance criterion, testable in
seconds. The end-to-end convergence and over-heavy runs need pygmo AND the
OpenTsiolkovsky binary and are marked ``slow``."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from orbitopt.flightlab.contract import load_target
from orbitopt.flightlab.vehicle_source import data_root, resolve_vehicle_path

FIX = Path(__file__).parent / "fixtures" / "flightlab" / "opentsiolkovsky"
REPO = Path(__file__).resolve().parents[1]
VEHICLE = REPO / "src" / "orbitopt" / "data" / "flightlab" / "vehicles" / "synthetic_two_stage.yaml"
TARGET = REPO / "src" / "orbitopt" / "data" / "flightlab" / "scenarios" / "leo_two_stage.yaml"


@pytest.fixture
def target():
    return load_target(TARGET)


@pytest.fixture
def stub(monkeypatch, tmp_path):
    path = tmp_path / "otcli"
    shutil.copy(FIX / "stub_cli.py", path)
    path.chmod(0o755)
    monkeypatch.setenv("ORBITOPT_OPENTSIOLKOVSKY", str(path))
    monkeypatch.setenv("OT_STUB_FIXTURE", str(FIX / "synthetic_out"))
    monkeypatch.setenv("OT_STUB_SRC", str(REPO / "src" / "orbitopt"))
    return path


# --- vehicle resolution (shared helper, FL-202 decision) ---------------------

def test_resolver_target_relative_hit(tmp_path, target):
    scene = tmp_path / "scenarios" / "vehicles"
    scene.mkdir(parents=True)
    shutil.copy(VEHICLE, scene / "synthetic_two_stage.yaml")
    assert resolve_vehicle_path(None, target, tmp_path / "leo_two_stage.yaml") == \
        tmp_path / "scenarios" / "vehicles" / "synthetic_two_stage.yaml"


def test_resolver_package_data_basename_fallback(target, tmp_path):
    # the shipped fixture declares a path that exists nowhere relative to a
    # copied target dir -- the basename lookup in package data must catch it
    assert resolve_vehicle_path(None, target, tmp_path / "t.yaml") == \
        data_root() / "vehicles" / "synthetic_two_stage.yaml"


def test_resolver_explicit_wins(target, tmp_path):
    assert resolve_vehicle_path(VEHICLE, target, tmp_path / "t.yaml") == VEHICLE


def test_resolver_explicit_missing_is_direct_error(target, tmp_path):
    with pytest.raises(FileNotFoundError, match="not found: /nope.yaml"):
        resolve_vehicle_path("/nope.yaml", target, tmp_path / "t.yaml")


def test_resolver_miss_lists_every_path_tried(tmp_path):
    from types import SimpleNamespace

    fake = SimpleNamespace(vehicle=SimpleNamespace(
        source="vehicles/nonexistent_two_stage.yaml"))
    broken = tmp_path / "t.yaml"
    broken.write_text("x", encoding="utf-8")
    with pytest.raises(FileNotFoundError) as exc:
        resolve_vehicle_path(None, fake, broken)
    message = str(exc.value)
    assert "vehicles/nonexistent_two_stage.yaml" in message
    assert str(tmp_path / "vehicles" / "nonexistent_two_stage.yaml") in message
    assert "  " in message                       # every candidate listed


def test_flight_optimize_no_vehicle_resolves_for_shipped_scenario(target, tmp_path):
    """Epic exit gate: `flight optimize leo_two_stage.yaml` works with no
    --vehicle (the declared path only resolves via the package-data rule)."""
    assert resolve_vehicle_path(None, target, TARGET) == VEHICLE


# --- the pygmo gate: actionable message, never a traceback -------------------

def test_missing_pygmo_raises_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "pygmo", None)
    from orbitopt.flightlab.optimize.run import _import_pygmo

    with pytest.raises(RuntimeError, match="conda"):
        _import_pygmo()


def test_cli_without_pygmo_exits_1_with_message(monkeypatch, stub, tmp_path, capsys):
    monkeypatch.setitem(sys.modules, "pygmo", None)
    from orbitopt import cli

    with pytest.raises(SystemExit) as exc:
        cli.main(["flight", "optimize", str(TARGET), "-o", str(tmp_path / "r")])
    assert exc.value.code == 1
    assert "ERROR:" in capsys.readouterr().out


# --- the solve loop against the recorded stub ---------------------------------

def _solve(target, out, **kw):
    pytest.importorskip("pygmo")
    pytest.importorskip("scipy")
    from orbitopt.flightlab.optimize.run import optimize_ascent

    kw.setdefault("generations", 2)
    kw.setdefault("population", 8)
    kw.setdefault("workers", 1)
    kw.setdefault("local", False)
    kw.setdefault("seed", 7)
    return optimize_ascent(target, VEHICLE, out, refly=False, target_path=str(TARGET), **kw)


def _has_pygmo_and_scipy():
    import importlib.util

    return all(importlib.util.find_spec(m) for m in ("pygmo", "scipy"))


@pytest.mark.skipif(not _has_pygmo_and_scipy(), reason="pygmo/scipy missing")
def test_stub_solve_is_honest_infeasible_and_complete(target, stub, tmp_path):
    out = tmp_path / "run"
    result = _solve(target, out)
    assert result["optimizer_converged"] is False
    assert result["status"] == "INFEASIBLE"
    assert "INFEASIBLE_APOAPSIS_OUT_OF_TOLERANCE" in result["reason_codes"]
    assert result["run"].verification.optimizer_converged is False
    for name in ("manifest.json", "verification.json", "optimization.json",
                 "attitude_program.csv"):
        assert (out / name).is_file(), name
    assert (out / "vehicle" / "vehicle.yaml").is_file()


@pytest.mark.skipif(not _has_pygmo_and_scipy(), reason="pygmo/scipy missing")
def test_fixed_seed_is_byte_identical(target, stub, tmp_path):
    a, b = tmp_path / "r1" / "run", tmp_path / "r2" / "run"
    _solve(target, a)
    _solve(target, b)
    assert (a / "optimization.json").read_bytes() == \
        (b / "optimization.json").read_bytes()
    assert (a / "manifest.json").read_bytes() == (b / "manifest.json").read_bytes()


@pytest.mark.skipif(not _has_pygmo_and_scipy(), reason="pygmo/scipy missing")
def test_overheavy_reserve_objective_never_fakes_success(target, stub, tmp_path):
    result = _solve(target, tmp_path / "over", objective="reserve",
                    payload_kg=30000.0)
    assert result["optimizer_converged"] is False
    assert result["reason_codes"]                  # an honest "why"


@pytest.mark.skipif(not _has_pygmo_and_scipy(), reason="pygmo/scipy missing")
def test_cli_optimize_stub_writes_run_and_report(stub, tmp_path, capsys):
    from orbitopt import cli

    out = tmp_path / "cli-run"
    assert cli.main(["flight", "optimize", str(TARGET), "-o", str(out),
                     "--seed", "3", "--generations", "1", "--population", "6",
                     "--workers", "1", "--no-local", "--no-refly"]) is None
    printed = capsys.readouterr().out
    assert "optimizer_converged:        FAILED" in printed
    assert "INFEASIBLE" in printed
    assert (out / "optimization.json").is_file()


# --- real binary, pygmo (epic acceptance criteria; slow) -----------------------

def _real_binary():
    from orbitopt.flightlab.adapters.opentsiolkovsky import find_binary

    try:
        return find_binary()
    except Exception:
        return None


@pytest.mark.slow
@pytest.mark.skipif(not _has_pygmo_and_scipy() or _real_binary() is None,
                    reason="pygmo/scipy/OpenTsiolkovsky binary missing")
def test_leo_two_stage_real_converges(tmp_path):
    result = _solve(load_target(TARGET), tmp_path / "real",
                    generations=40, population=40, workers=4, seed=11)
    assert result["optimizer_converged"] is True
    assert result["status"] == "CONVERGED"


@pytest.mark.slow
@pytest.mark.skipif(not _has_pygmo_and_scipy() or _real_binary() is None,
                    reason="pygmo/scipy/OpenTsiolkovsky binary missing")
def test_overheavy_real_never_fakes_success(tmp_path):
    from orbitopt.flightlab.optimize.run import OptimizeAbort

    try:
        result = _solve(load_target(TARGET), tmp_path / "real-oh",
                        objective="reserve", payload_kg=20000.0,
                        generations=10, population=24, workers=4, seed=5)
    except OptimizeAbort as exc:
        assert exc.reason_codes                   # honest abort, no fake run
    else:
        assert result["optimizer_converged"] is False
        assert result["reason_codes"]
