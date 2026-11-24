"""E11-06: does a Starship-class grid-fin set close the unpowered RTLS entry from STARBASE?

E11-01 flew the closure trial on the Kourou DC8 handoff with a 4 x 4 m^2 DESIGN-ASSUMPTION
fin set and found the corridor tilt (25-30 deg against 15) unreachable; the report put the
needed authority at ~2.4x that fin set. This benchmark re-runs the same acceptance question

  * from the STARBASE design (25.997 N, -97.157 E, azimuth inside the FAA corridor; the sealed
    E5-08c Starbase ascent and its canonical separation handoff), not Kourou, and
  * with fin sets sized from the public Super Heavy photos / SpaceX statements instead of the
    4 x 4 m^2 guess:
      - ``assumed``  : 4 x  4.0 m^2  (the E11-01 DESIGN ASSUMPTION, control)
      - ``block12``  : 4 x 12.5 m^2  (about 5.0 m x 2.5 m each, estimate for the four-fin booster)
      - ``v3_area``  : 3 x 18.75 m^2 (V3 read as +50 % AREA per fin: the conservative reading)
      - ``v3``       : 3 x 28.1 m^2  (about 7.5 m x 3.75 m each; SpaceX: "three fins ... 50 % larger")

Everything else (hinge limits, rate limits, Mach tables) is the E11-01 DC8 block, unchanged, so
the only thing that moves is the fin planform and count. The graded limits are NOT relaxed:
3 km / <= 350 m/s / <= 15 deg / 8 km corridor, q <= 250 kPa, 12 g, 100 t reserve.

Planning only (the sealed ascent is read, not re-flown): no OpenTsiolkovsky needed.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

REF = REPO / "tests/data/mission/reference/e5-08-new-vehicle-starbase"
import os as _os
#: the recorded ascent whose separation state the recovery flies from; E11_ASCENT_DIR swaps in another one
#: (the hot-staging ascent) without touching the sealed Starbase tree
ASCENT = Path(_os.environ.get("E11_ASCENT_DIR", str(REF / "mission-optimized/launch/ascent")))
SB = REPO / "src/orbitopt/data/flightlab/vehicles/starship_class_rtls_starbase.yaml"
DC8 = REPO / "src/orbitopt/data/flightlab/vehicles/starship_class_rtls_dc8.yaml"
SCEN = REPO / "src/orbitopt/data/flightlab/scenarios/leo_starship_class_rtls_e508_starbase.yaml"

PP_CD = [[0.3, 0.55], [1.0, 1.8], [1.2, 1.9], [2.0, 1.0], [5.0, 0.6], [12.0, 0.5]]

# "_rr": fin drag as a FRACTION of the vehicle's own drag, after the RETALT1 CFD figures supplied by
# the user (four grid fins ~25 % of the whole-vehicle drag at subsonic / low supersonic Mach, ~5-8 % at the
# highest Mach studied; UNVERIFIED at the source -- the paper is behind a login). f(M): 0.25 to M1.5, 0.065
# at M3.5, 0.05 from M8; the table is dCd = f/(1-f) * Cd_body(M) for the Starbase booster Cd table, derived
# for the 4 x 12.5 m^2 set (50 m^2 total) and scaled by total fin area for the others.
RR_CD = [[0.3, 0.30], [0.8, 0.30], [1.1, 0.43], [1.5, 0.42], [2.0, 0.23], [3.5, 0.09],
         [4.0, 0.077], [8.0, 0.07], [12.0, 0.07]]

FIN_SETS = {
    "assumed": dict(count=4, fin_area_m2=4.0),
    "block12": dict(count=4, fin_area_m2=12.5),
    "v3_area": dict(count=3, fin_area_m2=18.75),
    "v3": dict(count=3, fin_area_m2=28.1),
    # drag now scales with the fin planform. "_l": the E11-01 lumped table (derived for the 4 x 4 m^2
    # set) scaled by total fin area / 16 m^2. "_pp": a per-planform table (DESIGN ASSUMPTION, shape
    # after the lattice-fin transonic-choking literature: ~3.5x rise from subsonic to Mach ~1.2)
    # referred to the vehicle S_ref through the fin/vehicle area ratio.
    "block12_l": dict(count=4, fin_area_m2=12.5, cd_add_reference_fin_area_m2=16.0),
    "v3_area_l": dict(count=3, fin_area_m2=18.75, cd_add_reference_fin_area_m2=16.0),
    "v3_l": dict(count=3, fin_area_m2=28.1, cd_add_reference_fin_area_m2=16.0),
    "block12_pp": dict(count=4, fin_area_m2=12.5, cd_add_reference_fin_area_m2=63.617,
                       cd_add_mach=PP_CD),
    "v3_area_pp": dict(count=3, fin_area_m2=18.75, cd_add_reference_fin_area_m2=63.617,
                       cd_add_mach=PP_CD),
    "v3_pp": dict(count=3, fin_area_m2=28.1, cd_add_reference_fin_area_m2=63.617,
                  cd_add_mach=PP_CD),
    "block12_rr": dict(count=4, fin_area_m2=12.5, cd_add_reference_fin_area_m2=50.0, cd_add_mach=RR_CD),
    "v3_area_rr": dict(count=3, fin_area_m2=18.75, cd_add_reference_fin_area_m2=50.0, cd_add_mach=RR_CD),
    "v3_rr": dict(count=3, fin_area_m2=28.1, cd_add_reference_fin_area_m2=50.0, cd_add_mach=RR_CD),
}
SOURCES = {
    "block12": "~5.0 m x 2.5 m per fin, estimated for the four-fin booster (Tesla Oracle 2025-08-16)",
    "v3_area": "3 fins, each 50 % larger in AREA than the 12.5 m^2 block-1/2 estimate (Digital Trends "
               "reads the SpaceX statement as '50% more surface area') -- the conservative V3 reading",
    "v3": "SpaceX: 'moving from four fins to three', 'redesigned grid fins are 50% larger' "
          "(linear), i.e. ~7.5 m x 3.75 m (estimate, Tesla Oracle 2025-08-16)",
}

# FL-302 recipe, E11-01 limits; the script is searched (Starbase is not the DC8 state).
RECIPE = dict(entry_burns=0, boostback_cutoff="iip", landing_need_model="closed_loop",
              corridor_aim_margin_m=7000.0, entry_q_max_pa=250_000.0, entry_g_max=12.0,
              landing_reserve_margin_kg=2500.0, horizon_s=900.0, dt_opt=0.5, dt_final=0.25)

SCHEDULES = {
    "a60": {"alpha": 60.0, "gates": [[70000.0, 0.0, 60.0]]},
    "a45_a60_12k": {"alpha": 45.0, "gates": [[70000.0, 0.0, 45.0], [12000.0, 0.0, 60.0]]},
    "a40_a60_12k_a0_6k": {"alpha": 40.0, "gates": [[70000.0, 0.0, 40.0], [12000.0, 0.0, 60.0],
                                                    [6000.0, 0.0, 0.0]]},
}


def _vehicle_yaml(name: str, out_dir: Path) -> Path:
    doc = yaml.safe_load(SB.read_text())
    fins = copy.deepcopy(yaml.safe_load(DC8.read_text())["booster"]["grid_fins"])
    fins.update(FIN_SETS[name])
    fins["provenance"] = "public_estimate" if name != "assumed" else "illustrative"
    if name in SOURCES:
        fins["source"] = SOURCES[name]
    fins["assumptions"] = [f"E11-06 fin set {name}: {FIN_SETS[name]}; all other fin parameters "
                           "inherited from the E11-01 DC8 block"]
    if _os.environ.get("E11_FINS_FIXED"):         # Super Heavy's fins do not fold: always out, deflection only
        fins["stowable"] = False
    doc["booster"]["grid_fins"] = fins
    path = out_dir / f"starbase_fins_{name}.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def _row(label, plan) -> dict:
    cr = plan.corridor_report
    f = lambda v, n=1: None if v is None else round(float(v), n)  # noqa: E731
    return {"label": label, "feasible": bool(plan.feasible), "reason_codes": list(plan.reason_codes),
            "speed_m_s": f(cr.speed_m_s), "tilt_deg": f(cr.tilt_deg), "miss_m": f(cr.miss_m, 0),
            "entry_q_kpa": f(None if cr.entry_max_q_pa is None else cr.entry_max_q_pa / 1e3, 0),
            "landing_need_t": f(None if cr.landing_need_kg is None else cr.landing_need_kg / 1e3),
            "propellant_avail_t": f(cr.propellant_available_kg / 1e3)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sets", default="assumed,block12,v3")
    ap.add_argument("--schedules", default=",".join(SCHEDULES))
    ap.add_argument("--max-fevals", type=int, default=120)
    args = ap.parse_args(argv)
    t0 = time.time()

    from orbitopt.flightlab.contract import load_target, read_run
    from orbitopt.flightlab.recovery.boostback import RecoveryTarget, plan_boostback_entry
    from orbitopt.flightlab.recovery.pipeline import _to_ecef_state, landing_reserve_booster
    from orbitopt.flightlab.recovery.propagator import BoosterModel
    from orbitopt.mission.coupled_branches import booster_state_from_run, separation_handoff
    from orbitopt.reference_eop import freeze_reference, load_snapshot
    from dataclasses import replace

    target = load_target(SCEN)
    run = read_run(ASCENT)
    sep = booster_state_from_run(run, separation_handoff(run))
    if sep.frame != "ECEF":
        r, v = _to_ecef_state(sep)
        sep = replace(sep, frame="ECEF", position_m=tuple(float(c) for c in r),
                      velocity_m_s=tuple(float(c) for c in v))
    rt = RecoveryTarget.from_scenario(target.to_dict())
    site = target.to_dict()["launch_site"]
    rows = []
    with tempfile.TemporaryDirectory() as tmp, \
            freeze_reference(load_snapshot(REF / "earth_orientation.json")):
        for name in args.sets.split(","):
            veh = landing_reserve_booster(BoosterModel.load(_vehicle_yaml(name, Path(tmp))),
                                          target, "target")
            cases = [("ballistic", None)] + [(s, SCHEDULES[s]) for s in args.schedules.split(",")]
            for sched_name, spec in cases:
                if name != "assumed" and spec is None:
                    continue                       # the ballistic row does not depend on the fins
                ea = None if spec is None else {"mode": "schedule", "alpha_deg": spec["alpha"],
                                                "schedule": spec["gates"]}
                plan = plan_boostback_entry(sep, veh, rt, entry_aero=ea,
                                            max_fevals=args.max_fevals, **RECIPE)
                row = {"fins": name, **FIN_SETS[name], **_row(sched_name, plan)}
                rows.append(row)
                print(json.dumps(row), flush=True)
    closed = [r for r in rows if r["feasible"]]
    rep = {"task": "E11-06 Starbase grid-fin scaling", "launch_site": site,
           "scenario": str(SCEN.relative_to(REPO)), "separation_state_from": str(REF.relative_to(REPO)),
           "fin_sets": FIN_SETS, "fin_sources": SOURCES, "recipe": RECIPE, "trials": rows,
           "verdict": {"closed": bool(closed), "closing_cases": [(r["fins"], r["label"]) for r in closed]},
           "runtime_s": round(time.time() - t0, 1)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep["verdict"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
