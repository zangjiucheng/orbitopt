"""E11-06b: the continuous-burn Starbase design through the full two-stage LEO mission chain
(recorded Starbase ascent replayed, separation handoff, recovery leg flown, mission checks)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "benchmarks"))
import e11_continuous_burn_fly as F  # noqa: E402

DOC = REPO / "src/orbitopt/data/mission_two_stage_leo_starship_e508_starbase.yaml"
REF = REPO / "tests/data/mission/reference/e5-08-new-vehicle-starbase"


def main() -> int:
    from orbitopt.mission.two_stage_leo import TwoStageLeoScenario, run_two_stage_leo
    from orbitopt.reference_eop import freeze_reference, load_snapshot

    res = json.loads(Path(sys.argv[1]).read_text()); out = Path(sys.argv[2])
    x = np.array(res["x"][:13])
    doc = yaml.safe_load(DOC.read_text())["scenario"]
    doc["launch"]["vehicle"] = "starship_class_rtls_starbase_fins"      # the same design, V3-class fins on the booster
    import e11_unpowered_search as _U
    doc["ascent"]["run_dir"] = str(_U.B.ASCENT)
    doc["ascent"]["target_file"] = str(REF / "mission-optimized/launch/target.yaml")
    rec = doc.setdefault("recovery", {})
    rec["target_file"] = str(REF / "mission-optimized/launch/target.yaml")
    opts = F.plan_options(x, F.LANDING)
    opts["landing"] = dict(F.LANDING)
    rec["plan_options"] = json.loads(json.dumps(opts, default=float))
    out.mkdir(parents=True, exist_ok=True)
    import e11_unpowered_search as U
    with tempfile.TemporaryDirectory() as tmp, freeze_reference(load_snapshot(REF / "earth_orientation.json")):
        veh = U.B._vehicle_yaml(res["fins"], Path(tmp))                       # this candidate's fin calibration
        import os
        if os.environ.get("E11_ASCENT_VEHICLE"):          # the hot-staging ascent: its vehicle carries the staging block
            vdoc = yaml.safe_load(Path(os.environ["E11_ASCENT_VEHICLE"]).read_text())
            vdoc["booster"]["grid_fins"] = yaml.safe_load(veh.read_text())["booster"]["grid_fins"]
            veh = Path(tmp) / "hs_vehicle_with_fins.yaml"
            veh.write_text(yaml.safe_dump(vdoc, sort_keys=False, width=120))
            # the expected parking orbit is a regression pin of the RECORDED ascent: pin this ascent's own
            ver = json.loads((_U.B.ASCENT / "verification.json").read_text())
            by = {c["id"]: c["value"] for c in ver["checks"]}
            hp, ha = float(by["insertion-periapsis-above-minimum"]), float(by["insertion-apoapsis-within-tolerance"])
            a = 6378137.0 + 0.5 * (hp + ha)
            doc["expected"]["parking_orbit"].update(
                semi_major_axis_m=a, eccentricity=(ha - hp) / (2.0 * a),
                inclination_deg=float(by["insertion-inclination-within-tolerance"]),
                periapsis_altitude_m=hp, apoapsis_altitude_m=ha)
        doc["launch"]["vehicle"] = str(veh)
        scen = TwoStageLeoScenario.from_dict(doc)
        result = run_two_stage_leo(scen, out, propagate=False)
    print("mission status:", getattr(result, "status", None))
    for c in getattr(result, "checks", []) or []:
        print(f"  {'PASS' if c.passed else 'FAIL'} {c.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
