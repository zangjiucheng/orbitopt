"""E12-00 (E11-06 release gate): structured records for the archived continuous-burn evidence.

    python scripts/e11_06_gate_record.py fly CAND.json OUT.json
        Fly the continuous-burn recovery script of a search result through the full recovery chain
        (FL-302 plan + FL-303 landing burn) and dump the graded checks, touchdown and ignition count.

    python scripts/e11_06_gate_record.py checks RUN_DIR OUT.json
        Dump the checks of a recorded run -- a mission chain directory (mission.verification.json),
        a Flight Lab run directory (verification.json), or an ascent directory -- as {id: passed} plus
        the values/limits, for the gate archive.

Everything else in the gate record is produced by the shipped E11-06 runners:
  e11_continuous_burn_search.py (champion), e11_continuous_burn_twin.py (independent twin),
  e11_continuous_burn_mission.py (full mission), e11_run_orbit_refly.py (SECO orbit re-fly),
  e11_continuous_burn_mc.py (FL-304 Monte Carlo with cluster-bootstrap CI).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "benchmarks"))


def fly(cand: Path, out: Path) -> int:
    import tempfile
    from dataclasses import replace

    import e11_continuous_burn_fly as F
    import e11_unpowered_search as U
    from orbitopt.flightlab.contract import load_target, read_run
    from orbitopt.flightlab.recovery.pipeline import _to_ecef_state, fly_recovery
    from orbitopt.mission.coupled_branches import booster_state_from_run, separation_handoff
    from orbitopt.reference_eop import freeze_reference, load_snapshot

    res = json.loads(cand.read_text())
    x = res["x"][:13]
    target = load_target(U.B.SCEN)
    run = read_run(U.B.ASCENT)
    sep = booster_state_from_run(run, separation_handoff(run))
    if sep.frame != "ECEF":
        r, v = _to_ecef_state(sep)
        sep = replace(sep, frame="ECEF", position_m=tuple(float(c) for c in r),
                      velocity_m_s=tuple(float(c) for c in v))

    with tempfile.TemporaryDirectory() as tmp, freeze_reference(load_snapshot(U.B.REF / "earth_orientation.json")):
        veh = U.B._vehicle_yaml(res["fins"], Path(tmp))
        rep = fly_recovery(run, target, veh, sep_state=sep, plan_options=F.plan_options(x, F.LANDING))
    td = getattr(rep, "landing", None)
    row = {
        "plan_feasible": bool(rep.plan.feasible),
        "plan_reason_codes": list(rep.plan.reason_codes),
        "ignitions_used": int(getattr(rep.plan, "ignitions_used", 0)),
        "checks": {c.id: bool(c.passed) for c in rep.checks},
        "touchdown": None if td is None else {
            k: getattr(td, k, None) for k in
            ("landed", "reason_codes", "vertical_speed_m_s", "lateral_speed_m_s", "miss_m",
             "propellant_remaining_kg", "max_proper_g", "t_burn_s", "ignition_altitude_m")},
        "ascent": str(U.B.ASCENT), "vehicle_fins": res["fins"],
    }
    out.write_text(json.dumps(row, indent=1, default=float) + "\n")
    print(json.dumps({"feasible": row["plan_feasible"], "ignitions": row["ignitions_used"],
                      "checks_passed": sum(row["checks"].values()), "n_checks": len(row["checks"])}))
    return 0


def checks(run_dir: Path, out: Path) -> int:
    mv = run_dir / "mission.verification.json"
    if mv.is_file():                                   # mission chain report
        doc = json.loads(mv.read_text())

        def verdicts(o):                               # every {check-id: bool} node, deep
            if isinstance(o, dict):
                if o and all(isinstance(v, bool) for v in o.values()) and len(o) > 5:
                    yield o
                for v in o.values():
                    yield from verdicts(v)
        fm = doc.get("inputs", {}).get("full_mission", {})
        flat: dict = {}
        for node in verdicts(doc):
            flat.update({k: bool(v) for k, v in node.items()})
        row = {"kind": "mission", "status": doc.get("status"), "verdict": fm.get("verdict"),
               "coverage": fm.get("coverage"), "checks": flat}
    else:                                              # a Flight Lab / ascent run directory
        from orbitopt.flightlab.contract import read_run

        run = read_run(run_dir)
        row = {"kind": "run", "physics_verified": run.verification.physics_verified,
               "mission_constraints_passed": run.verification.mission_constraints_passed,
               "checks": {c.id: bool(c.passed) for c in run.verification.checks},
               "detail": {c.id: {"value": c.value, "limit": c.limit, "unit": c.unit}
                          for c in run.verification.checks}}
    out.write_text(json.dumps(row, indent=1, default=str) + "\n")
    bad = [k for k, v in row["checks"].items() if not v]
    print(json.dumps({"checks": len(row["checks"]), "failed": bad}))
    return 0


if __name__ == "__main__":
    raise SystemExit({"fly": fly, "checks": checks}[sys.argv[1]](Path(sys.argv[2]), Path(sys.argv[3])))
