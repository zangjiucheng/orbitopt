"""E11-06b validation: fly a continuous-burn search result through the full recovery chain
(FL-302 plan + FL-303 landing burn) and print the graded checks and touchdown."""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "benchmarks"))
import e11_continuous_burn_search as C  # noqa: E402
import e11_unpowered_search as U  # noqa: E402

LANDING = C.LANDING


def plan_options(x, landing=None, dt_final=0.25):
    x = np.asarray(x, dtype=float)
    opts = {**C.RECIPE, "dt_opt": 0.5, "dt_final": dt_final, "landing_reserve": "target",
            "initial_guess": [float(v) for v in x[:8]], "search": "keep",
            "entry_aero": U._schedule(np.concatenate([np.zeros(5), np.asarray(x[8:13])])),
            "entry_aero_with_burn": True}
    if landing is not None:
        opts["landing"] = landing
    import os
    if os.environ.get("E11_FIN_GAIN"):          # E11-06: closed-loop speed feedback on the fin schedule
        opts["entry_aero"] = {**opts["entry_aero"], "speed_gain_deg_per_mps": float(os.environ["E11_FIN_GAIN"]),
                              "speed_ref": json.loads(Path(os.environ["E11_FIN_REF"]).read_text())}
    return opts


def main() -> int:
    from orbitopt.flightlab.contract import load_target, read_run
    from orbitopt.flightlab.recovery.pipeline import _to_ecef_state, fly_recovery
    from orbitopt.mission.coupled_branches import booster_state_from_run, separation_handoff
    from orbitopt.reference_eop import freeze_reference, load_snapshot

    res = json.loads(Path(sys.argv[1]).read_text())
    fins = res["fins"]; x = np.array(res["x"]); vs = float(res.get("vscale", 1.0))
    target = load_target(U.B.SCEN)
    run = read_run(U.B.ASCENT)
    sep = booster_state_from_run(run, separation_handoff(run))
    if sep.frame != "ECEF":
        r, v = _to_ecef_state(sep)
        sep = replace(sep, frame="ECEF", position_m=tuple(float(c) for c in r), velocity_m_s=tuple(float(c) for c in v))
    if vs != 1.0:
        sep = replace(sep, velocity_m_s=tuple(float(c) * vs for c in sep.velocity_m_s))
    with tempfile.TemporaryDirectory() as tmp, freeze_reference(load_snapshot(U.B.REF / "earth_orientation.json")):
        veh = U.B._vehicle_yaml(fins, Path(tmp))
        cases = [("continuous burn: landing law lit at hand-over (4.0 g cap)", LANDING)]
        for name, landing in cases:
            out = fly_recovery(run, target, veh, sep_state=sep, plan_options=plan_options(x, landing))
            print("==", name)
            print("plan feasible:", out.plan.feasible, list(out.plan.reason_codes), "ignitions(boostback+entry):", out.plan.ignitions_used)
            print({c.id: bool(c.passed) for c in out.checks})
            td = getattr(out, "landing", None)
            keys = ("landed", "reason_codes", "vertical_speed_m_s", "lateral_speed_m_s", "miss_m",
                    "propellant_remaining_kg", "max_proper_g", "t_burn_s", "ignition_altitude_m")
            print("touchdown:", {k: getattr(td, k, None) for k in keys} if td is not None else None)
            if td is not None:
                print("  landing fields:", [k for k in dir(td) if not k.startswith("_")][:40])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
