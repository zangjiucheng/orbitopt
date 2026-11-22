"""E11-01 (task #308): grid-fin aerodynamic authority benchmark.

Two parts, both anchored on the COMMITTED E10-07 separation handoff
(``docs/benchmarks/e10_07_full_recovery_reflight.json``) so no ascent
re-flight (OpenTsiolkovsky) is needed:

A) *controllability envelope* -- coast from the handoff with a fixed
   grid-fin schedule over a Mach x bank matrix and measure what the fins
   actually buy: peak lateral acceleration, peak side force, cross-range
   and down-range shift at the 3 km corridor altitude vs the same-state
   finless coast, plus peak entry q.

B) *two-burn closure trial* -- the DC8 E10-07 script through
   ``plan_boostback_entry(entry_burns=0, entry_aero=...)`` for a battery
   of schedules, ballistic baseline included, each graded by the stock
   E5-06 checks (3000 m / 350 m/s / 15 deg corridor, 250 kPa / 12 g
   entry, 2.5 t landing reserve over the closed-loop landing need).

Writes the full report to ``--out``
(``docs/benchmarks/e11_01_grid_fin_authority.json`` when generating the
committed copy). Runtime ~8 min, no external solver needed.
"""
from __future__ import annotations

import argparse
import datetime as dtm
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from orbitopt.flightlab.contract import load_target  # noqa: E402
from orbitopt.flightlab.recovery.boostback import (  # noqa: E402
    RecoveryTarget, _enu_basis, _geodetic_fast, plan_boostback_entry,
)
from orbitopt.flightlab.recovery.grid_fins import GridFinGuidance  # noqa: E402
from orbitopt.flightlab.recovery.propagator import (  # noqa: E402
    BoosterModel, BoosterState, propagate,
)
from orbitopt.flightlab.recovery.pipeline import landing_reserve_booster  # noqa: E402

SCENARIO = REPO / "src/orbitopt/data/flightlab/scenarios/leo_starship_class_rtls_e508_dc7.yaml"
DC8 = REPO / "src/orbitopt/data/flightlab/vehicles/starship_class_rtls_dc8.yaml"
HANDOFF = REPO / "docs/benchmarks/e10_07_full_recovery_reflight.json"

CORRIDOR_ALT_M = 3000.0
RECIPE = dict(entry_burns=0, search="keep", boostback_cutoff="iip",
              landing_need_model="closed_loop", corridor_aim_margin_m=7000.0,
              entry_q_max_pa=250_000.0, entry_g_max=12.0,
              landing_reserve_margin_kg=2500.0, horizon_s=900.0,
              dt_opt=0.5, dt_final=0.25)
SCRIPT = [21.2, -2.0, 1.0, 0.0, 0.0]


def _handoff_state() -> BoosterState:
    hs = json.loads(HANDOFF.read_text())["handoff"]["state"]
    return BoosterState(t_s=hs["t_s"],
                        epoch_utc=dtm.datetime.fromisoformat(hs["epoch_utc"]),
                        frame="ECEF", position_m=np.array(hs["position_m"]),
                        velocity_m_s=np.array(hs["velocity_m_s"]),
                        mass_kg=hs["mass_kg"])


def _enu(r, rt: RecoveryTarget) -> np.ndarray:
    e, n, u = _enu_basis(rt.latitude_deg, rt.longitude_deg)
    d = np.asarray(r, dtype=float) - np.asarray(rt.position_ecef_m, dtype=float)
    return np.array([e[0] * d[0] + e[1] * d[1] + e[2] * d[2],
                     n[0] * d[0] + n[1] * d[1] + n[2] * d[2],
                     u[0] * d[0] + u[1] * d[1] + u[2] * d[2]])


def _state_at(track, h_target: float):
    """First track sample whose height is at or below ``h_target``."""
    h = track.height_m_arr
    for k in range(1, len(h)):
        if h[k] <= h_target:
            st = BoosterState(t_s=track.times_s[k],
                              epoch_utc=track.epoch_utc
                              + dtm.timedelta(seconds=float(track.times_s[k])),
                              frame="ECEF",
                              position_m=np.asarray(track.position_ecef_m[k]),
                              velocity_m_s=np.asarray(track.velocity_ecef_m_s[k]),
                              mass_kg=track.mass_kg[k])
            return st, k
    return None, None


def _coast_case(veh, sep, rt, label, sched) -> dict:
    guidance = None
    if sched is not None:
        guidance = GridFinGuidance(model=veh.grid_fins, mode="schedule",
                                   alpha_deg=float(sched[0][2]),
                                   schedule=tuple(tuple(g) for g in sched))
        cmd = lambda t, s, g=guidance: g.command(  # noqa: E731
            _geodetic_fast(s.position_m)[2], s.position_m, s.velocity_m_s)
    else:
        cmd = None
    track = propagate(sep, veh, None, sep.t_s + 600.0, dt_s=0.25,
                      sample_s=0.25, aero_cmd=cmd)
    st, k = _state_at(track, CORRIDOR_ALT_M)
    if st is None:
        return {"label": label, "reached_3km": False}
    enu = _enu(st.position_m, rt)
    lat_n = 0.0
    if track.aero_force_n is not None:
        v = np.asarray(track.velocity_ecef_m_s[:k + 1], dtype=float)
        f = np.asarray(track.aero_force_n[:k + 1], dtype=float)
        unit = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        along = np.sum(f * unit, axis=1)
        side = np.linalg.norm(f - along[:, None] * unit, axis=1)
        lat_n = float(np.max(side))
    return {"label": label, "reached_3km": True,
            "t_s": round(float(track.times_s[k]), 1),
            "speed_m_s": round(float(np.linalg.norm(st.velocity_m_s)), 1),
            "mach": round(float(track.mach[k]), 2),
            "q_at_3km_kpa": round(float(track.dynamic_pressure_pa[k]) / 1e3, 1),
            "peak_q_kpa": round(float(track.max_dynamic_pressure_pa) / 1e3, 1),
            "peak_lateral_force_n": round(lat_n, 0),
            "peak_lateral_accel_m_s2": round(lat_n / float(st.mass_kg), 3),
            "e_m": round(float(enu[0]), 0), "n_m": round(float(enu[1]), 0),
            "up_m": round(float(enu[2]), 0)}



def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-twin", action="store_true",
                    help="skip part C (main vs independent twin fin re-flight)")
    args = ap.parse_args(argv)
    t0 = time.time()

    sep = _handoff_state()
    target = load_target(SCENARIO)
    veh = landing_reserve_booster(BoosterModel.load(DC8), target, "target")
    rt = RecoveryTarget.from_scenario(target.to_dict())

    # ---- Part A: controllability envelope -------------------------------
    coasts = [
        ("finless_coast", None),
        ("a30_b0", [[70000.0, 0.0, 30.0]]),
        ("a45_b0", [[70000.0, 0.0, 45.0]]),
        ("a60_b0", [[70000.0, 0.0, 60.0]]),
        ("a30_b45", [[70000.0, 45.0, 30.0]]),
        ("a30_bm45", [[70000.0, -45.0, 30.0]]),
        ("a45_b45", [[70000.0, 45.0, 45.0]]),
        ("a45_bm45", [[70000.0, -45.0, 45.0]]),
        ("a60_b45", [[70000.0, 45.0, 60.0]]),
        ("a60_bm45", [[70000.0, -45.0, 60.0]]),
    ]
    part_a = [_coast_case(veh, sep, rt, label, sched)
              for label, sched in coasts]
    base = next(c for c in part_a if c["label"] == "finless_coast")
    for c in part_a:
        if c.get("reached_3km"):
            c["cross_shift_m"] = round(c["n_m"] - base["n_m"], 0)
            c["down_shift_m"] = round(c["e_m"] - base["e_m"], 0)

    # ---- Part B: two-burn closure trial ---------------------------------
    trials = []
    cases = [("ballistic", None)] + [
        ("sched_a40", {"alpha": 40.0, "gates": [[70000.0, 0.0, 40.0]]}),
        ("sched_a50", {"alpha": 50.0, "gates": [[70000.0, 0.0, 50.0]]}),
        ("sched_a60", {"alpha": 60.0, "gates": [[70000.0, 0.0, 60.0]]}),
        ("ladder_a40_a20_18k",
         {"alpha": 40.0, "gates": [[70000.0, 0.0, 40.0], [18000.0, 0.0, 20.0]]}),
        ("flare_a40_a60_12k",
         {"alpha": 40.0, "gates": [[70000.0, 0.0, 40.0], [12000.0, 0.0, 60.0]]}),
        ("flare_a50_a60_10k",
         {"alpha": 50.0, "gates": [[70000.0, 0.0, 50.0], [10000.0, 0.0, 60.0]]}),
        ("dive_a40_a0_20k",
         {"alpha": 40.0, "gates": [[70000.0, 0.0, 40.0], [20000.0, 0.0, 0.0]]}),
        ("dive_a40_a60_12k_a0_6k",
         {"alpha": 40.0, "gates": [[70000.0, 0.0, 40.0], [12000.0, 0.0, 60.0],
                                   [6000.0, 0.0, 0.0]]}),
    ]
    for label, spec in cases:
        ea = None if spec is None else {
            "mode": "schedule", "alpha_deg": spec["alpha"],
            "schedule": spec["gates"]}
        plan = plan_boostback_entry(sep, veh, rt, initial_guess=list(SCRIPT),
                                    entry_aero=ea, **RECIPE)
        cr = plan.corridor_report
        trials.append({"label": label, "feasible": bool(plan.feasible),
                       "reason_codes": list(plan.reason_codes),
                       "arrival_speed_m_s": (None if cr.speed_m_s is None
                                             else round(cr.speed_m_s, 1)),
                       "tilt_deg": (None if cr.tilt_deg is None
                                    else round(cr.tilt_deg, 1)),
                       "miss_m": (None if cr.miss_m is None
                                  else round(cr.miss_m)),
                       "entry_q_kpa": (None if cr.entry_max_q_pa is None
                                       else round(cr.entry_max_q_pa / 1e3)),
                       "landing_need_t": (None if cr.landing_need_kg is None
                                          else round(cr.landing_need_kg / 1e3, 1)),
                       "propellant_avail_t": round(cr.propellant_available_kg / 1e3, 1)})

    twin_pairs = []
    if not args.no_twin:
        from orbitopt.verify.full_recovery_reflight import fin_pair_reflight
        for sched, a0 in (((70000.0, 0.0, 40.0),), 40.0), (
                ((70000.0, 0.0, 40.0), (12000.0, 0.0, 60.0)), 40.0):
            twin_pairs.append(fin_pair_reflight(
                handoff_json=HANDOFF, scenario_yaml=SCENARIO,
                vehicle_yaml=DC8,
                entry_aero={"mode": "schedule", "alpha_deg": a0,
                            "schedule": [list(g) for g in sched]}))

    from orbitopt.flightlab.recovery.grid_fins import DESIGN_ASSUMPTIONS
    rep = {"task": "E11-01 grid-fin aerodynamic authority (task #308)",
           "base_commit": "2e20641", "handoff_source": str(HANDOFF.relative_to(REPO)),
           "vehicle": str(DC8.relative_to(REPO)),
           "scenario": str(SCENARIO.relative_to(REPO)),
           "corridor": {"alt_m": 3000.0, "v_max_m_s": 350.0, "max_tilt_deg": 15.0,
                        "radius_m": 8000.0, "entry_q_max_kpa": 250.0,
                        "entry_g_max": 12.0, "landing_reserve_kg": 2500.0},
           "envelope": part_a, "closure_trial": trials,
           "twin_comparison": twin_pairs,
           "design_assumptions": list(DESIGN_ASSUMPTIONS),
           "verdict": None, "runtime_s": None}

    feasible = [t for t in trials if t["feasible"]]
    best_v = min((t["arrival_speed_m_s"] for t in trials
                  if t["arrival_speed_m_s"] is not None), default=None)
    best_tilt = min((t["tilt_deg"] for t in trials if t["tilt_deg"] is not None),
                    default=None)
    rep["verdict"] = {
        "closed": bool(feasible),
        "best_arrival_speed_m_s": best_v,
        "best_tilt_deg": best_tilt,
        "ballistic_reference_speed_m_s":
            trials[0]["arrival_speed_m_s"],
        "twin_checks_ok": all(c["ok"] for r in twin_pairs for c in r["checks"])
            if twin_pairs else None,
    }
    rep["runtime_s"] = round(time.time() - t0, 1)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=1, default=float) + "\n",
                            encoding="utf-8")
        print(f"wrote {args.out}")
    print(json.dumps({"verdict": rep["verdict"], "runtime_s": rep["runtime_s"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
