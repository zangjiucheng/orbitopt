"""E11-06b: independent point-mass twin (E10-07 module: own WGS-84 geodesy, US76, J2, RK4, own grid-fin
actuator/aero) re-flies the authored continuous-burn script from the same separation state and is
compared with the main FL-301/FL-302 model on the corridor-crossing quantities."""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "benchmarks"))
import e11_continuous_burn_fly as F  # noqa: E402
import e11_continuous_burn_search as C  # noqa: E402
import e11_unpowered_search as U  # noqa: E402


def main() -> int:
    from orbitopt.flightlab.contract import load_target, read_run
    from orbitopt.flightlab.recovery import boostback as bb_mod
    from orbitopt.flightlab.recovery.boostback import RecoveryTarget, _Mission, _bearing_of_target, _unpack_script, plan_boostback_entry
    from orbitopt.flightlab.recovery.grid_fins import GridFinGuidance
    from orbitopt.flightlab.recovery.pipeline import _to_ecef_state, landing_reserve_booster
    from orbitopt.flightlab.recovery.propagator import BoosterModel
    from orbitopt.mission.coupled_branches import booster_state_from_run, separation_handoff
    from orbitopt.reference_eop import freeze_reference, load_snapshot
    from orbitopt.verify import full_recovery_reflight as T
    from orbitopt.flightlab.recovery.landing import TerminalGuidance
    from orbitopt.flightlab.recovery.pipeline import _landing_limits_for_target

    res = json.loads(Path(sys.argv[1]).read_text()); x = np.array(res["x"][:13])
    target = load_target(U.B.SCEN)
    run = read_run(U.B.ASCENT)
    sep = booster_state_from_run(run, separation_handoff(run))
    if sep.frame != "ECEF":
        r, v = _to_ecef_state(sep)
        sep = replace(sep, frame="ECEF", position_m=tuple(float(c) for c in r), velocity_m_s=tuple(float(c) for c in v))
    rt = RecoveryTarget.from_scenario(target.to_dict())
    with tempfile.TemporaryDirectory() as tmp, freeze_reference(load_snapshot(U.B.REF / "earth_orientation.json")):
        veh_path = U.B._vehicle_yaml(res["fins"], Path(tmp))
        booster = landing_reserve_booster(BoosterModel.load(veh_path), target, "target")
        ea = U._schedule(np.concatenate([np.zeros(5), x[8:13]]))
        recipe = {**C.RECIPE, "dt_opt": 0.5, "dt_final": 0.25}
        plan = plan_boostback_entry(sep, booster, rt, entry_aero=ea, entry_aero_with_burn=True,
                                    landing_guidance=TerminalGuidance(**C.LANDING),
                                    landing_limits=_landing_limits_for_target(target),
                                    initial_guess=[float(v) for v in x[:8]], search="keep", **recipe)
        cr = plan.corridor_report
        pad = np.asarray(rt.position_ecef_m, dtype=float)
        lat, lon, _ = T._twin_geodetic(pad)
        e_ = np.array([-np.sin(lon), np.cos(lon), 0.0])
        n_ = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
        d = np.asarray(sep.position_m, dtype=float) - pad
        horiz = float(d @ e_) * e_ + float(d @ n_) * n_
        return_dir = -horiz / max(float(np.linalg.norm(horiz)), 1e-9)
        reserve = float(getattr(booster, "landing_propellant_reserve_kg", 0.0))

        def factory():
            bb, eb, eb2 = _unpack_script(x[:8], 1, True)
            bb = replace(bb, duration_s=bb_mod.IIP_BOOSTBACK_CAP_S)
            eb = replace(eb, duration_s=bb_mod.ENTRY_CUTOFF_CAP_S)
            return _Mission(float(sep.t_s), bb, eb, _bearing_of_target(pad), eb2,
                            iip_aim_m=1000.0 * float(x[1]), pad_ecef=pad, return_dir=return_dir,
                            vernier_throttle=booster.engine.throttle_min,
                            bb_floor_mass_kg=booster.dry_mass_kg + reserve,
                            entry_cut_speed_m_s=float(x[6]))

        law = GridFinGuidance(model=booster.grid_fins, mode="schedule", alpha_deg=float(ea["alpha_deg"]),
                              bank_max_deg=float(booster.grid_fins.bank_max_deg),
                              schedule=tuple(tuple(float(v) for v in g) for g in ea["schedule"]))
        h0 = float(rt.corridor.h0_m); hor = float(recipe["horizon_s"])
        out = {}
        for label, dt, kw in (("twin", 0.1, {}), ("twin_fine", 0.05, {}), ("twin_main_atmosphere", 0.1, {"density_fn": __import__("orbitopt.entry", fromlist=["density_kg_m3"]).density_kg_m3})):
            tw = T._twin_flight(factory, sep, booster, dt_s=dt, corridor_h0_m=h0, horizon_s=hor,
                                fins_pair=(T._TwinFins(booster.grid_fins), law), **kw)
            out[label] = tw
    main = {"speed_m_s": cr.speed_m_s, "tilt_deg": cr.tilt_deg, "peak_q_kpa": cr.entry_max_q_pa / 1e3,
            "peak_g": cr.entry_max_proper_g, "cross_t_s": cr.t_s}
    print("main :", {k: round(float(v), 2) for k, v in main.items()}, "feasible", plan.feasible)
    for label, tw in out.items():
        pos = np.asarray(tw["h0_position_ecef_m"], dtype=float)
        miss = float(np.linalg.norm((pos - pad) - ((pos - pad) @ (pad / np.linalg.norm(pad))) * (pad / np.linalg.norm(pad))))
        print(f"{label:22s} speed={tw['h0_speed_m_s']:.1f} tilt={tw['h0_tilt_deg']:.2f} q={tw['entry_max_q_pa']/1e3:.1f}kPa "
              f"g={tw['entry_max_proper_g']:.2f} cross_t={tw['corridor_h0_cross_t_s']:.2f} miss~{miss:.0f} m")
    t = out["twin"]
    delta = {"speed_pct": 100 * abs(t["h0_speed_m_s"] - main["speed_m_s"]) / main["speed_m_s"],
             "tilt_deg": abs(t["h0_tilt_deg"] - main["tilt_deg"]),
             "peak_q_pct": 100 * abs(t["entry_max_q_pa"] / 1e3 - main["peak_q_kpa"]) / main["peak_q_kpa"],
             "peak_g": abs(t["entry_max_proper_g"] - main["peak_g"])}
    print("twin vs main delta:", {k: round(v, 2) for k, v in delta.items()})
    print("E10-07 bands:", {k: v[1] for k, v in T.BANDS.items() if "entry" in k or "corridor" in k})
    Path(sys.argv[2]).write_text(json.dumps({"main": main, "twin": {k: {kk: vv for kk, vv in v.items() if kk != "impact"} for k, v in out.items()}, "delta": delta}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
