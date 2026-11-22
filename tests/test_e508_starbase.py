"""E5-08c: the E5-08b (DC7) design flown from Starbase.

The claim is narrow and checkable: the flight target is the Kourou DC7 one with only the
launch site and the target inclination changed (every graded constraint is identical), and
the vehicle differs from DC7 only in the declared fields (launch azimuth, stage-1 cutoff /
separation, the ship's burn window, the post-MECO pitch nodes). The recorded evidence is then
asserted as recorded -- including what did NOT pass and how little robustness there is.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "src/orbitopt/data"
REF = REPO / "tests/data/mission/reference/e5-08-new-vehicle-starbase"
REPORTS = REPO / "tests/data/mission/reference/e5-08-new-vehicle-starbase-reports"

DC7 = DATA / "flightlab/vehicles/starship_class_rtls_dc7.yaml"
SB = DATA / "flightlab/vehicles/starship_class_rtls_starbase.yaml"
SCEN_DC7 = DATA / "flightlab/scenarios/leo_starship_class_rtls_e508_dc7.yaml"
SCEN_SB = DATA / "flightlab/scenarios/leo_starship_class_rtls_e508_starbase.yaml"


def _load(p: Path) -> dict:
    return yaml.safe_load(p.read_text())


# --------------------------------------------------------- what Starbase may change


def test_the_flight_target_differs_from_dc7_only_in_site_and_inclination():
    a, b = _load(SCEN_DC7), _load(SCEN_SB)
    for doc in (a, b):
        doc["mission"].pop("id")
        doc.pop("assumptions")
        doc["vehicle"].pop("source")
        doc.pop("launch_site")
        doc["target_orbit"].pop("inclination_deg")
    assert a == b                    # constraints, recovery limits, reserve, simulation: all identical


def test_the_site_is_the_one_in_the_launch_site_database():
    sites = {s["id"]: s for s in _load(DATA / "launch_sites.yaml")["sites"]}
    site = _load(SCEN_SB)["launch_site"]
    sb = sites["starbase"]
    assert (site["latitude_deg"], site["longitude_deg"]) == (sb["latitude_deg"], sb["longitude_deg"])
    assert site["altitude_wgs84_m"] == sb["altitude_m"]
    lo, hi = sb["azimuth_corridors_deg"][0]
    assert lo <= _load(SB)["attitude_program"][0]["azimuth_deg"] <= hi       # inside the public corridor
    assert _load(SCEN_SB)["target_orbit"]["inclination_deg"] == 30.5


def test_the_vehicle_differs_from_dc7_only_in_the_declared_fields():
    a, b = _load(DC7), _load(SB)

    def strip(doc):
        d = json.loads(json.dumps(doc))
        for key in ("name", "assumptions", "data_provenance", "attitude_program"):
            d.pop(key, None)
        s1, s2 = d["stages"]
        s1["engine"].pop("thrust_vac_n"); s1["engine"].pop("burn_end_s"); s1.pop("separation_time_s")
        s2["engine"].pop("thrust_vac_n"); s2["engine"].pop("burn_end_s")
        return d

    assert strip(a) == strip(b)
    pa, pb = a["attitude_program"], b["attitude_program"]
    assert [r["t_s"] for r in pa] == [r["t_s"] for r in pb]
    assert len({r["azimuth_deg"] for r in pb}) == 1                          # one constant launch azimuth
    pre = [(x["t_s"], x["elevation_deg"]) for x in pa if x["t_s"] <= 141.0]
    assert pre == [(x["t_s"], x["elevation_deg"]) for x in pb if x["t_s"] <= 141.0]   # pre-MECO pitch inherited


def test_the_separation_sits_on_the_two_second_trajectory_grid():
    """A separation time off the grid leaves the stack's last sample a second short, and the
    mission's handoff-continuity checks fail (position / velocity / mass across the fork)."""
    s1 = _load(SB)["stages"][0]
    assert float(s1["separation_time_s"]) % 2.0 == 0.0
    assert 0.0 < float(s1["separation_time_s"]) - float(s1["engine"]["burn_end_s"]) <= 3.0


def test_the_post_meco_pitch_program_never_pitches_back_up():
    elev = [r["elevation_deg"] for r in _load(SB)["attitude_program"] if r["t_s"] >= 141.0]
    assert all(b <= a + 1e-9 for a, b in zip(elev, elev[1:]))


def test_the_vehicle_loads_through_the_catalog():
    from orbitopt.flightlab.vehicle_view import load_vehicle_view

    assert load_vehicle_view("starship_class_rtls_starbase").name == "starship-class-rtls-starbase-v1"


# ------------------------------------------------------- the recorded evidence


def test_the_packaged_mission_flies_the_sealed_script():
    doc = (DATA / "mission_two_stage_leo_starship_e508_starbase.yaml").read_text()
    x_doc = json.loads(re.search(r"initial_guess: (\[[^\]]*\])", doc).group(1))
    assert x_doc == json.loads((REF / "recovery-script.json").read_text())["recipe"]["initial_guess"]


def test_a_stale_crossenv_report_is_never_regraded(tmp_path):
    import importlib.util
    import os
    import sys
    import time

    spec = importlib.util.spec_from_file_location("e508_script_sb", REPO / "scripts/e508_new_vehicle.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["e508_script_sb"] = mod
    spec.loader.exec_module(mod)
    report, inp = tmp_path / "report.json", tmp_path / "script.json"
    inp.write_text("{}")
    report.write_text("{}")
    os.utime(inp, (time.time() + 100, time.time() + 100))
    assert mod._report_is_stale(report, [inp])


def test_the_sealed_tree_matches_its_provenance_digests():
    prov = json.loads((REF / "PROVENANCE.json").read_text())
    for rel, digest in prov["files"].items():
        assert hashlib.sha256((REF / rel).read_bytes()).hexdigest() == digest, rel
    frozen = prov["frozen_inputs"]
    assert frozen["vehicle_sha256"] == hashlib.sha256(SB.read_bytes()).hexdigest()
    assert frozen["scenario_sha256"] == hashlib.sha256(SCEN_SB.read_bytes()).hexdigest()


def test_the_recorded_recovery_closes_inside_the_inherited_gates():
    rec = json.loads((REF / "recovery-script.json").read_text())
    assert rec.get("status") != "not_closed"
    assert rec["checks"]["recovery-landing"] and rec["checks"]["recovery-propellant-reserve"]
    td = rec["touchdown"]
    assert td["landed"] and not td["reason_codes"]
    assert abs(td["vertical_speed_m_s"]) <= 3.0 and abs(td["lateral_speed_m_s"]) <= 1.0
    assert td["propellant_remaining_kg"] >= 100_000.0 and td["max_proper_g"] <= 5.0
    plan = rec["recipe"]
    assert plan["entry_q_max_pa"] == 250_000.0 and plan["entry_g_max"] == 12.0
    assert plan["landing_reserve"] == "target"


def test_the_ascent_meets_every_inherited_limit():
    ver = json.loads((REF / "mission-optimized/launch/ascent/verification.json").read_text())
    by = {c["id"]: c for c in ver["checks"]}
    assert by["max-accel-within-limit"]["passed"] and by["max-accel-within-limit"]["value"] <= 3.0
    assert by["max-q-within-limit"]["passed"] and by["max-q-within-limit"]["value"] <= 55_000.0
    assert by["insertion-periapsis-above-minimum"]["value"] >= 230_000.0
    assert by["insertion-apoapsis-within-tolerance"]["passed"]
    assert by["insertion-inclination-within-tolerance"]["passed"]
    assert abs(by["insertion-inclination-within-tolerance"]["value"] - 30.5) <= 1.0


def test_the_mission_replays_green_from_the_sealed_tree(tmp_path):
    pytest.importorskip("gfold")
    import os
    import subprocess
    import sys

    out = subprocess.run([sys.executable, str(REPO / "scripts/e508_new_vehicle.py"), "run", str(tmp_path / "replay")],
                         env={**os.environ, "ORBITOPT_E508_VARIANT": "starbase", "PYTHONPATH": str(REPO / "src")},
                         capture_output=True, text=True, cwd=REPO)
    assert "mission status: OK" in out.stdout, out.stdout[-400:] + out.stderr[-400:]


# ---------------------------------------------- what did NOT pass, as recorded


def test_the_independent_dynamics_result_is_recorded_as_it_is():
    """E8-05 on this design: 27 of 28. The debris-impact speed misses its band by 0.8 points
    (12.8 % against 12 %); the band is NOT loosened. E10-04 evidenced this as a genuine
    aero-model difference, not an implementation artifact: replaying the twin's own debris
    integrator from the RECORDED separation state and debris mass reproduces the gap
    (+12.7 %, debris-model-difference.json), the descent agrees to within 0.5 %
    until dynamic pressure passes ~100 kPa and then the recorded trajectory sheds 393 m/s
    below 22 km against the twin's 190 (~2x the deceleration), and scaling the
    declared illustrative blunt-body CA polar by 1.1 closes it (+10.8 %). Same-state replay
    rules out timing, frame, interpolation and engine lines; the two models disagree on the
    power-off drag coefficient in high-q transonic flow."""
    rep = json.loads((REPORTS / "independent-dynamics.json").read_text())
    details = rep["inputs"]["check_details"]
    assert rep["status"]["passed"] is False and len(details) == 28
    assert sorted(c["id"] for c in details if not c["passed"]) == ["ascent.debris_impact.speed"]
    by = {c["id"]: c for c in details}
    assert by["ascent.seco.orbit_semi_major_axis"]["passed"] and by["ascent.seco.state_velocity"]["passed"]


def test_the_monte_carlo_robustness_is_recorded_honestly():
    """The nominal mission closes; it is not robust. Pinned, the recorded script lands in 4 of
    36 FL-304 dispersed samples (seeds 8, 9, 10 -- never used to tune it)."""
    pinned = json.loads((REPORTS / "mc-summary-pinned.json").read_text())
    assert pinned["mode"].startswith("pinned") and pinned["n"] == 36 and pinned["seeds"] == [8, 9, 10]
    assert pinned["successes"] == 4
    refine = json.loads((REPORTS / "mc-summary.json").read_text())
    assert refine["mode"].startswith("per-sample") and refine["n"] == 36
    assert refine["successes"] <= 4
