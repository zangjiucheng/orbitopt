"""Checks for orbitopt.flightlab.target_export (FL-003, #190): an exported
FlightTarget is schema-valid, deterministic (byte-exact golden file for the
RAAN-free path) and never invents what it cannot derive -- a missing epoch or
an infeasible inclination (below |latitude|, or outside the site's corridor-
reachable bands) is a readable error, not a guessed number.

The RAAN-constrained path is checked for determinism and internal consistency
(derived epoch inside the requested span, regressed RAAN) rather than byte
equality, because its epoch carries the window scan's second-level result.
Like test_flightlab_contract.py, deliberately compute-free where possible:
only numpy arrives through orbitopt.launch_windows."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from orbitopt.cli import main as cli_main
from orbitopt.flightlab.contract import load_target, validate_path, validate_target
from orbitopt.flightlab.target_export import (
    DEFAULT_WINDOW_DAYS,
    build_target,
    feasibility_report,
    generator_assumption,
    parse_orbit_spec,
    parse_window_spec,
    save_target,
)

GOLDEN = Path(__file__).parent / "data" / "flightlab" / "export_kourou_400x400_i51-6.yaml"
GOLDEN_CALL = dict(site_id="kourou", periapsis_alt_m=400e3, apoapsis_alt_m=400e3,
                   inclination_deg=51.6, date_utc="2026-11-10T06:30:00Z", record_generator=False)


# --------------------------------------------------------------------------
# parsers
# --------------------------------------------------------------------------

def test_parse_orbit_spec():
    assert parse_orbit_spec("400x400@51.6") == (400000.0, 400000.0, 51.6)
    assert parse_orbit_spec(" 200x35786.0@0 ") == (200000.0, 35786000.0, 0.0)
    for bad in ("400x400", "400@51.6", "51.6@400x400", "a x b @ c", "", "400x200@51.6", "400x400@200"):
        with pytest.raises(ValueError):
            parse_orbit_spec(bad)


def test_parse_window_spec():
    assert parse_window_spec("2026-11-10") == ("2026-11-10", DEFAULT_WINDOW_DAYS)
    assert parse_window_spec("2026-11-10/3") == ("2026-11-10", 3.0)
    for bad in ("", "/3", "2026-11-10/0", "2026-11-10/-2"):
        with pytest.raises(ValueError):
            parse_window_spec(bad)


# --------------------------------------------------------------------------
# golden file (acceptance: deterministic, validates)
# --------------------------------------------------------------------------

def test_export_golden_is_byte_exact(tmp_path):
    """The RAAN-free export must reproduce the packaged golden YAML byte for
    byte -- the contract E2/E5 will diff exported targets against."""
    out = save_target(build_target(**GOLDEN_CALL), tmp_path / "target.yaml")
    assert out.read_bytes() == GOLDEN.read_bytes()


def test_export_golden_file_validates():
    """The golden file itself passes the contract, standalone (someone edits
    it by hand -> orbitopt flight validate catches it)."""
    validate_target(load_target(GOLDEN).to_dict())
    summary = validate_path(GOLDEN)
    assert "kourou-400x400km-i51.6" in str(summary)


def test_export_records_generator_provenance():
    t = build_target("kourou", 400e3, 400e3, 51.6, date_utc="2026-11-10")
    assert t.assumptions[0].startswith("generator: orbitopt ")
    assert "git " in t.assumptions[0]
    assert t.assumptions[0] == generator_assumption()


# --------------------------------------------------------------------------
# rejections (acceptance: infeasible inclination rejected with a reason)
# --------------------------------------------------------------------------

def test_inclination_below_latitude_rejected_with_reason():
    with pytest.raises(ValueError, match=r"inclination 20 deg is infeasible from launch site 'cape-canaveral'"):
        build_target("cape-canaveral", 400e3, 400e3, 20.0, date_utc="2026-11-10")
    try:
        build_target("cape-canaveral", 400e3, 400e3, 20.0, date_utc="2026-11-10")
    except ValueError as exc:
        assert "site latitude" in str(exc)  # the number to fix, not a bare "infeasible"


def test_inclination_outside_corridor_bands_rejected():
    """Above |latitude| but beyond what the range-safety corridors admit."""
    with pytest.raises(ValueError, match="azimuth corridors"):
        build_target("cape-canaveral", 400e3, 400e3, 70.0, date_utc="2026-11-10")


def test_unknown_site_rejected():
    with pytest.raises(KeyError):
        build_target("somewhere-else", 400e3, 400e3, 51.6, date_utc="2026-11-10")


def test_epoch_never_invented():
    with pytest.raises(ValueError, match="date_utc"):
        build_target("kourou", 400e3, 400e3, 51.6)


def test_apoapsis_below_periapsis_rejected():
    with pytest.raises(ValueError, match="below periapsis"):
        build_target("kourou", 500e3, 400e3, 51.6, date_utc="2026-11-10")


# --------------------------------------------------------------------------
# RAAN-constrained epoch derivation
# --------------------------------------------------------------------------

def test_raan_constrained_epoch_is_derived_and_deterministic():
    a = build_target("kourou", 400e3, 400e3, 51.6, raan_deg=40.0, date_utc="2026-11-10")
    b = build_target("kourou", 400e3, 400e3, 51.6, raan_deg=40.0, date_utc="2026-11-10")
    assert a.to_dict() == b.to_dict()  # same inputs, same document, second for second
    assert a.target_orbit.raan_deg is not None
    assert a.target_orbit.epoch_utc == a.mission.epoch_utc
    assert "node pass" in " ".join(a.assumptions)


def test_raan_constrained_window_moves_the_epoch():
    """The requested anchor is only *earliest liftoff*: the exported epoch is
    the first node pass, and the exported RAAN is the regressed plane RAAN
    there (prograde Kourou plane at 40 deg => RAAN decreases)."""
    t = build_target("kourou", 400e3, 400e3, 51.6, raan_deg=40.0, date_utc="2026-11-10T00:00:00Z")
    assert t.mission.epoch_utc > "2026-11-10T00:00:00Z"
    assert 0.0 <= t.target_orbit.raan_deg < 40.0


def test_raan_free_path_keeps_requested_epoch():
    t = build_target("kourou", 400e3, 400e3, 51.6, date_utc="2026-11-10T06:30:00Z")
    assert t.mission.epoch_utc == "2026-11-10T06:30:00Z"
    assert t.target_orbit.raan_deg is None
    assert t.target_orbit.epoch_utc is None
    doc = t.to_dict()
    assert doc["target_orbit"]["raan_deg"] is None


def test_constraints_and_recovery_pass_through():
    t = build_target("kourou", 400e3, 400e3, 51.6, date_utc="2026-11-10",
                     constraints={"max_dynamic_pressure_pa": 35000.0},
                     recovery={"kind": "rtls", "target": "LCES-1"})
    doc = t.to_dict()
    assert doc["constraints"]["max_dynamic_pressure_pa"] == 35000.0
    assert doc["recovery"]["kind"] == "rtls"
    joined = " ".join(t.assumptions)
    assert "no mission constraints specified" not in joined
    assert "no recovery plan specified" not in joined


def test_feasibility_report():
    r = feasibility_report("cape-canaveral", 45.0)
    assert r["feasible"] is True and r["site"] == "cape-canaveral"
    assert r["max_abs_dla_deg"] > 0
    assert any(lo <= 45.0 <= hi for lo, hi in r["reachable_inclination_intervals"])
    assert feasibility_report("cape-canaveral", 20.0)["feasible"] is False


# --------------------------------------------------------------------------
# CLI end-to-end (acceptance: output passes `orbitopt flight validate`)
# --------------------------------------------------------------------------

def test_cli_export_target_then_validate(tmp_path, capsys):
    out = tmp_path / "target.yaml"
    cli_main(["flight", "export-target", "--site", "kourou", "--orbit", "400x400@51.6",
              "--date", "2026-11-10", "-o", str(out)])
    printed = capsys.readouterr().out
    assert f"wrote {out}" in printed and "validated against flight-target-1.0.json" in printed
    assert out.exists()
    summary = validate_path(out)
    assert "kourou-400x400km-i51.6" in str(summary)


def test_cli_export_target_raan_window_form(tmp_path, capsys):
    out = tmp_path / "t.yaml"
    cli_main(["flight", "export-target", "--site", "wenchang", "--orbit", "400x400@45",
              "--raan", "180", "--window", "2026-11-10/4", "-o", str(out)])
    assert "RAAN" in capsys.readouterr().out  # the resolved plane RAAN, echoed back
    t = load_target(out)
    assert t.target_orbit.raan_deg is not None and t.mission.epoch_utc >= "2026-11-10T00:00:00Z"


def test_cli_export_target_infeasible_exits_1(tmp_path, capsys):
    out = tmp_path / "t.yaml"
    with pytest.raises(SystemExit) as exc:
        cli_main(["flight", "export-target", "--site", "cape-canaveral", "--orbit", "400x400@20",
                  "--date", "2026-11-10", "-o", str(out)])
    assert exc.value.code == 1
    err = capsys.readouterr().out
    assert err.startswith("ERROR:") and "infeasible" in err
    assert not out.exists()  # never writes a file it could not validate


def test_cli_export_target_rejects_bad_orbit_and_missing_epoch(tmp_path, capsys):
    out = tmp_path / "t.yaml"
    with pytest.raises(SystemExit) as exc:
        cli_main(["flight", "export-target", "--site", "kourou", "--orbit", "400x400",
                  "--date", "2026-11-10", "-o", str(out)])
    assert exc.value.code == 1 and "expected <peri_km>x<apo_km>@<inc_deg>" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:  # --date/--window group is required
        cli_main(["flight", "export-target", "--site", "kourou", "--orbit", "400x400@51.6",
                  "-o", str(out)])
    assert exc.value.code == 2