"""E5-08b (DC7): the E5-08 design with an earlier stage-1 cutoff.

The claim is narrow and checkable: the FLIGHT TARGET (every graded constraint)
is exactly the E5-08 / E5-07 one, and the VEHICLE differs from the sealed E5-08
vehicle only in the declared DC7 fields (stage-1 cutoff and separation, the
ship's burn window, the post-MECO pitch nodes). The recorded evidence is then
asserted as recorded -- including what did NOT pass.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "src/orbitopt/data"
REF = REPO / "tests/data/mission/reference/e5-08-new-vehicle-dc7"
REPORTS = REPO / "tests/data/mission/reference/e5-08-new-vehicle-dc7-reports"

DERIVED = DATA / "flightlab/vehicles/starship_class_rtls_derived.yaml"
DC7 = DATA / "flightlab/vehicles/starship_class_rtls_dc7.yaml"
SCEN = DATA / "flightlab/scenarios/leo_starship_class_rtls_e508.yaml"
SCEN_DC7 = DATA / "flightlab/scenarios/leo_starship_class_rtls_e508_dc7.yaml"


def _load(p: Path) -> dict:
    return yaml.safe_load(p.read_text())


# --------------------------------------------------------- what DC7 may change


def test_the_flight_target_is_the_e508_one_with_only_the_vehicle_swapped():
    """Every constraint (max-q, g, insertion band, touchdown, reserve) is byte-equal."""
    a, b = _load(SCEN), _load(SCEN_DC7)
    for doc in (a, b):
        doc["mission"].pop("id")
        doc.pop("assumptions")
        doc["vehicle"].pop("source")
    assert a == b


def test_the_vehicle_differs_only_in_the_declared_dc7_fields():
    a, b = _load(DERIVED), _load(DC7)

    def strip(doc):
        d = json.loads(json.dumps(doc))
        for key in ("name", "assumptions", "data_provenance", "attitude_program"):
            d.pop(key, None)
        s1, s2 = d["stages"]
        s1["engine"].pop("thrust_vac_n"); s1["engine"].pop("burn_end_s"); s1.pop("separation_time_s")
        s2["engine"].pop("thrust_vac_n"); s2["engine"].pop("burn_end_s")
        return d

    assert strip(a) == strip(b)                                   # masses, Isp, areas, aero, geometry, booster
    s1a, s1b = a["stages"][0], b["stages"][0]
    assert s1a["engine"]["burn_end_s"] - s1b["engine"]["burn_end_s"] == pytest.approx(12.0)
    assert s1a["separation_time_s"] - s1b["separation_time_s"] == pytest.approx(12.0)
    # the pre-MECO pitch program is inherited; only the post-MECO nodes move
    pa = {r["t_s"]: r for r in a["attitude_program"]}
    pb = {r["t_s"]: r for r in b["attitude_program"]}
    assert pa.keys() == pb.keys()
    moved = {t for t in pa if pa[t] != pb[t]}
    assert moved and min(moved) >= 155.0
    assert all(pa[t] == pb[t] for t in pa if t <= 141.0)


def test_the_thrust_tables_are_the_inherited_ones_cut_at_the_new_times():
    a, b = _load(DERIVED), _load(DC7)
    t1a = a["stages"][0]["engine"]["thrust_vac_n"]
    t1b = b["stages"][0]["engine"]["thrust_vac_n"]
    assert t1b[:3] == t1a[:3] and t1b[-1][0] == b["stages"][0]["engine"]["burn_end_s"]
    assert t1a[2][1] >= t1b[-1][1] >= t1a[-1][1]                   # on the inherited throttle schedule
    t2a = a["stages"][1]["engine"]["thrust_vac_n"]
    t2b = b["stages"][1]["engine"]["thrust_vac_n"]
    assert t2b[: len(t2a)] == t2a and t2b[-1][1] == t2a[-1][1]     # held at the floor to the new cutoff


def test_the_vehicle_loads_through_the_catalog_with_its_dc7_provenance():
    from orbitopt.flightlab.vehicle_view import load_vehicle_view

    view = load_vehicle_view("starship_class_rtls_dc7")
    assert view.name == "starship-class-rtls-dc7-v1"
    text = DC7.read_text()
    assert "DC7" in text and "388.66" in text and "~100 t" in text   # the cost is written down, not hidden


# ------------------------------------------------------- the recorded evidence


def test_the_packaged_mission_flies_the_sealed_script():
    """The cross-environment plan reads its booster script from the packaged mission
    document, the Monte Carlo and the touchdown evidence from the sealed recovery-script.json:
    they must be the same script. (They once were not: the record step only filled a
    placeholder, so a re-record left the first script in the document and the joint gate
    graded a script other than the one sealed.)"""
    import re

    doc = (DATA / "mission_two_stage_leo_starship_e508_dc7.yaml").read_text()
    x_doc = json.loads(re.search(r"initial_guess: (\[[^\]]*\])", doc).group(1))
    assert x_doc == json.loads((REF / "recovery-script.json").read_text())["recipe"]["initial_guess"]


def test_a_stale_crossenv_report_is_never_regraded(tmp_path):
    """crossenv-check re-graded an old report after the script it was run for had been
    replaced; a report older than any of its inputs must be re-run."""
    import importlib.util
    import os
    import sys
    import time

    spec = importlib.util.spec_from_file_location("e508_script", REPO / "scripts/e508_new_vehicle.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["e508_script"] = mod
    spec.loader.exec_module(mod)
    report, inp = tmp_path / "report.json", tmp_path / "script.json"
    assert mod._report_is_stale(report, [inp])                      # no report: run it
    inp.write_text("{}")
    report.write_text("{}")
    os.utime(inp, (time.time() - 100, time.time() - 100))
    assert not mod._report_is_stale(report, [inp])                  # the report is newer than its input
    os.utime(inp, (time.time() + 100, time.time() + 100))
    assert mod._report_is_stale(report, [inp, tmp_path / "missing.yaml"])   # an input changed after it


def test_the_sealed_tree_matches_its_provenance_digests():
    prov = json.loads((REF / "PROVENANCE.json").read_text())
    assert prov["reference_class"] == "new-vehicle-qualification"
    for rel, digest in prov["files"].items():
        assert hashlib.sha256((REF / rel).read_bytes()).hexdigest() == digest, rel
    frozen = prov["frozen_inputs"]
    assert frozen["vehicle_sha256"] == hashlib.sha256(DC7.read_bytes()).hexdigest()
    assert frozen["scenario_sha256"] == hashlib.sha256(SCEN_DC7.read_bytes()).hexdigest()


def test_the_recorded_recovery_closes_inside_the_inherited_gates():
    rec = json.loads((REF / "recovery-script.json").read_text())
    assert rec.get("status") != "not_closed"
    checks = rec["checks"]
    assert checks["recovery-landing"] and checks["recovery-propellant-reserve"]
    td = rec["touchdown"]
    assert td["landed"] and not td["reason_codes"]
    assert abs(td["vertical_speed_m_s"]) <= 3.0 and abs(td["lateral_speed_m_s"]) <= 1.0     # graded, unchanged
    assert td["propellant_remaining_kg"] >= 100_000.0                                          # graded, unchanged
    assert td["max_proper_g"] <= 5.0                                                           # LandingLimits
    plan = rec["recipe"]
    assert plan["entry_q_max_pa"] == 250_000.0 and plan["entry_g_max"] == 12.0                 # the DECLARED entry limits
    assert plan["landing_reserve"] == "target"                                                  # planned against the graded reserve


def test_the_ascent_still_meets_every_inherited_limit_and_the_costs_are_measured():
    ver = json.loads((REF / "mission-optimized/launch/ascent/verification.json").read_text())
    by = {c["id"]: c for c in ver["checks"]}
    assert by["max-accel-within-limit"]["passed"] and by["max-accel-within-limit"]["value"] <= 3.0
    assert by["max-q-within-limit"]["passed"] and by["max-q-within-limit"]["value"] <= 55_000.0
    assert by["insertion-periapsis-above-minimum"]["value"] >= 230_000.0
    assert by["insertion-apoapsis-within-tolerance"]["passed"]
    assert by["insertion-inclination-within-tolerance"]["passed"]
    cmp = json.loads((REF / "comparison.json").read_text())
    kept = cmp["payload_capability"]["ship_propellant_margin"]
    assert kept["delta_kg"] < -10_000.0                  # the ship pays for the booster's reserve; recorded, not hidden
    assert "measured" in cmp["ascent"]["note"] and cmp["ascent"]["peak_g"]["delta"].startswith("+")


def test_the_mission_replays_green_from_the_sealed_tree(tmp_path):
    """The single-interpreter replay of the whole mission over the sealed ascent
    flies the recorded script and lands (the cross-environment joint gate that
    grades the five physical checks is the CI step / ``crossenv-check``)."""
    pytest.importorskip("gfold")
    import os
    import subprocess
    import sys

    out = subprocess.run([sys.executable, str(REPO / "scripts/e508_new_vehicle.py"), "run",
                          str(tmp_path / "replay")],
                         env={**os.environ, "ORBITOPT_E508_VARIANT": "dc7", "PYTHONPATH": str(REPO / "src")},
                         capture_output=True, text=True, cwd=REPO)
    assert "mission status: OK" in out.stdout, out.stdout[-400:] + out.stderr[-400:]


# ---------------------------------------------- what did NOT pass, as recorded


def test_the_independent_dynamics_result_is_recorded_as_it_is():
    """E8-05 on this design: all 28 checks pass. E10-04 closed the SECO
    semi-major-axis gate (it read 27.7 km against the unchanged 15 km band) by
    fixing the comparison side, not the band: the raw 2-s history was chorded
    linearly across the MECO (123.0 s) and SECO (514.663 s) mid-segment engine
    discontinuities, which is itself a ~13 m/s / ~22 km artifact at both events
    (see orbitopt.verify.independent_dynamics._raw_interp and the E10-04
    regression tests in tests/test_e805_independent_dynamics.py). Reconstructed
    against the left-segment recorded accelerations the SECO discrepancy is
    ~5.9 km and the SECO velocity discrepancy ~5.2 m/s, both inside their
    original bands. The band is NOT loosened."""
    rep = json.loads((REPORTS / "independent-dynamics.json").read_text())
    details = rep["inputs"]["check_details"]
    failing = sorted(c["id"] for c in details if not c["passed"])
    assert rep["status"]["passed"] is True
    assert failing == [] and len(details) == 28
    by = {c["id"]: c for c in details}
    # the closed gate sits near a third of its band, and the post-MECO
    # program no longer carries the unphysical bump that the first DC7 search found
    assert by["ascent.seco.orbit_semi_major_axis"]["value"] < 15000.0 / 2
    assert by["ascent.seco.state_velocity"]["passed"] and by["ascent.seco.orbit_eccentricity"]["passed"]
    prog = _load(DC7)["attitude_program"]
    elev = [r["elevation_deg"] for r in prog if r["t_s"] >= 141.0]
    assert all(b <= a + 1e-9 for a, b in zip(elev, elev[1:]))      # never pitches back up after MECO


def test_the_monte_carlo_robustness_is_recorded_honestly():
    """The recorded script closes the NOMINAL mission with margin, but it is not robust:
    flown through the FL-304 standard dispersions (seeds 8, 9, 10 -- never used to pick
    or tune it) it lands in 11 of 36 samples when pinned, and in 3 of 36 when every
    sample first re-plans with the planner's own local refine (which cannot even keep
    the nominal case closed). Both are recorded, neither is hidden, and the success-rate
    CI is the whole-seed cluster bootstrap."""
    pinned = json.loads((REPORTS / "mc-summary-pinned.json").read_text())
    assert pinned["mode"].startswith("pinned") and pinned["n"] == 36 and pinned["seeds"] == [8, 9, 10]
    assert pinned["successes"] == 11
    refine = json.loads((REPORTS / "mc-summary.json").read_text())
    assert refine["mode"].startswith("per-sample") and refine["n"] == 36 and refine["successes"] == 3
    # every failure of either protocol is a corridor condition, not a propellant exhaustion
    for doc in (pinned, refine):
        reasons = {k for r in doc["per_seed"].values() for k in r["reasons"]}
        assert reasons <= {"CORRIDOR_MISSED", "INSUFFICIENT_PROPELLANT_RESERVE"}


def test_the_recorded_touchdown_has_margin_over_the_graded_reserve():
    td = json.loads((REF / "recovery-script.json").read_text())["touchdown"]
    assert td["propellant_remaining_kg"] >= 110_000.0            # chosen with 10 t of room, not on the gate
    assert abs(td["vertical_speed_m_s"]) <= 2.0 and td["max_proper_g"] <= 4.8
