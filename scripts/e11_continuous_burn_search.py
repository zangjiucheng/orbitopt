"""E11-06b: Starbase recovery with a SHORT entry burn flown CONTINUOUSLY into the landing burn.

Ignitions: (1) the boostback, which starts at separation with the engines still lit (hot staging:
no shutdown / coast after separation, so ``delay`` is pinned to ~0); (2) ONE burn that is the
entry burn and the landing burn together -- the entry burn must end at the corridor (3000 m ..
3250 m) so the landing burn takes over without an engine shutdown. Two ignitions in all. Grid fins
fly the coast between the two burns. Every graded limit is the planner's own and unchanged.

Decision vector (13): boostback [delay, aim_km, throttle, pitch, yaw], entry [trigger_alt_m,
cut_speed_m_s, throttle], fin schedule [a1, h1, a2, h2, a3].
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import e11_unpowered_search as U  # noqa: E402

BOUNDS = [(0.0, 0.5), (-40.0, 120.0), (0.4, 1.0), (-40.0, 40.0), (-45.0, 45.0),
          (3200.0, 30000.0), (150.0, 350.0), (0.4, 1.0),
          (0.0, 60.0), (12000.0, 60000.0), (0.0, 60.0), (8000.0, 25000.0), (0.0, 60.0)]
#: landing-law options of the sealed Starbase recipe plus the continuous-burn hand-over: the engine
#: is already lit when the law takes over (E11-06 ``lit_at_start``), so the entry burn and the landing
#: burn are one burn and the booster is ignited twice in all (boostback; entry+landing)
LANDING = {"max_thrust_accel_g": 4.0, "tvc_rate_limit_deg_s": 20.0, "tvc_lag_tau_s": 0.2,
           "tvc_lag_compensation": True, "lit_at_start": True}
import os as _os
if _os.environ.get("E11_ENGINE_STAGES"):         # E11-06: 13 -> 5 -> 3 engine shutdown in the landing burn
    LANDING["engine_stage_fractions"] = (1.0, 5.0 / 13.0, 3.0 / 13.0)
CORRIDOR_H_M = 3011.6      # the planner's corridor-insertion altitude: the entry burn must not have ended above it
CONT_GAP_M = 40.0          # the landing burn must light within this height of the entry cutoff
RECIPE = {**U.B.RECIPE, "entry_burns": 1, "entry_throttle": True, "entry_cutoff": "speed"}


def _plan(x):
    from orbitopt.flightlab.recovery.boostback import plan_boostback_entry

    from orbitopt.flightlab.recovery.landing import TerminalGuidance
    from orbitopt.flightlab.recovery.pipeline import _landing_limits_for_target

    c = U._CTX
    sched = U._schedule(np.concatenate([np.zeros(5), x[8:13]]))
    guid = TerminalGuidance(**LANDING)
    return plan_boostback_entry(c["sep"], c["veh"], c["rt"], entry_aero=sched,
                                entry_aero_with_burn=True, landing_guidance=guid,
                                landing_limits=_landing_limits_for_target(c["target"]),
                                initial_guess=[float(v) for v in x[:8]], search="keep", **c["recipe"])


def _gap_m(cr) -> float:
    """Height by which the entry burn ended ABOVE the corridor insertion: 0 = it was still burning
    when the landing law took over (one continuous burn); > 0 = an engine-off hop, then a relight."""
    if cr.entry_cutoff_altitude_m is None:
        return 0.0
    return max(0.0, cr.entry_cutoff_altitude_m - CORRIDOR_H_M)


def _corridor_excess(cr) -> float:
    """The DEFAULT_CORRIDOR limits the planner skips once it grades by the closed-loop landing forecast
    (speed <= 350 m/s, tilt <= 15 deg at the corridor): they stay requirements here, never relaxed."""
    if cr.speed_m_s is None or cr.tilt_deg is None:
        return 5.0
    return max(0.0, cr.speed_m_s - 350.0) / 350.0 + max(0.0, cr.tilt_deg - 15.0) / 15.0


def _penalty(plan) -> float:
    cr = plan.corridor_report
    exc = _corridor_excess(cr)
    if plan.feasible and _gap_m(cr) <= CONT_GAP_M and exc == 0.0:
        return -1.0
    p = 1.0 + U._penalty(plan) + (1.0 if plan.feasible else 0.0) + 2.0 * exc
    if cr.landing_touchdown_vertical_m_s is not None:
        p += (max(0.0, abs(cr.landing_touchdown_vertical_m_s) - 3.0) / 3.0
              + max(0.0, (cr.landing_touchdown_lateral_m_s or 0.0) - 1.5) / 50.0
              + max(0.0, (cr.landing_touchdown_miss_m or 0.0) - 25.0) / 200.0)
    return p + min(_gap_m(cr), 3000.0) / 300.0


def _twin_ctx():
    """One-time setup of the independent twin's pieces (own geodesy / US76 / J2 / RK4 / fin model)."""
    from dataclasses import replace  # noqa: F401
    from orbitopt.flightlab.recovery.grid_fins import GridFinGuidance  # noqa: F401
    from orbitopt.verify import full_recovery_reflight as T

    c = U._CTX
    rt = c["rt"]
    pad = np.asarray(rt.position_ecef_m, dtype=float)
    lat, lon, _ = T._twin_geodetic(pad)
    e_ = np.array([-np.sin(lon), np.cos(lon), 0.0])
    n_ = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    d = np.asarray(c["sep"].position_m, dtype=float) - pad
    horiz = float(d @ e_) * e_ + float(d @ n_) * n_
    c["twin"] = {"T": T, "pad": pad, "return_dir": -horiz / max(float(np.linalg.norm(horiz)), 1e-9)}


def twin_terms(x) -> dict:
    """Corridor quantities the independent twin sees for the authored script (None if it never crossed)."""
    from dataclasses import replace

    from orbitopt.flightlab.recovery import boostback as bb_mod
    from orbitopt.flightlab.recovery.boostback import _Mission, _bearing_of_target, _unpack_script
    from orbitopt.flightlab.recovery.grid_fins import GridFinGuidance

    c = U._CTX
    if "twin" not in c:
        _twin_ctx()
    T, pad, rd = c["twin"]["T"], c["twin"]["pad"], c["twin"]["return_dir"]
    booster, sep = c["veh"], c["sep"]
    reserve = float(getattr(booster, "landing_propellant_reserve_kg", 0.0))
    ea = U._schedule(np.concatenate([np.zeros(5), x[8:13]]))

    def factory():
        bb, eb, eb2 = _unpack_script(x[:8], 1, True)
        bb = replace(bb, duration_s=bb_mod.IIP_BOOSTBACK_CAP_S)
        eb = replace(eb, duration_s=bb_mod.ENTRY_CUTOFF_CAP_S)
        return _Mission(float(sep.t_s), bb, eb, _bearing_of_target(pad), eb2, iip_aim_m=1000.0 * float(x[1]),
                        pad_ecef=pad, return_dir=rd, vernier_throttle=booster.engine.throttle_min,
                        bb_floor_mass_kg=booster.dry_mass_kg + reserve, entry_cut_speed_m_s=float(x[6]))

    law = GridFinGuidance(model=booster.grid_fins, mode="schedule", alpha_deg=float(ea["alpha_deg"]),
                          bank_max_deg=float(booster.grid_fins.bank_max_deg),
                          schedule=tuple(tuple(float(v) for v in g) for g in ea["schedule"]))
    tw = T._twin_flight(factory, sep, booster, dt_s=0.2, corridor_h0_m=float(c["rt"].corridor.h0_m),
                        horizon_s=900.0, fins_pair=(T._TwinFins(booster.grid_fins), law))
    pos = np.asarray(tw["h0_position_ecef_m"], dtype=float) - pad
    up = pad / np.linalg.norm(pad)
    return {"speed": tw["h0_speed_m_s"], "tilt": tw["h0_tilt_deg"],
            "miss": float(np.linalg.norm(pos - (pos @ up) * up)), "q": tw["entry_max_q_pa"],
            "g": tw["entry_max_proper_g"]}


def twin_penalty(x) -> float:
    """0 when the twin meets the same corridor limits (speed <= 350, tilt <= 15, q <= 250 kPa, 12 g)."""
    try:
        t = twin_terms(x)
    except Exception:
        return 20.0
    if t["speed"] is None or t["tilt"] is None:
        return 20.0
    return (max(0.0, t["speed"] - 350.0) / 350.0 + max(0.0, t["tilt"] - 15.0) / 15.0
            + max(0.0, t["miss"] - 7000.0) / 8000.0 + max(0.0, t["q"] - 250e3) / 250e3
            + max(0.0, t["g"] - 12.0) / 12.0)


USE_TWIN = False


def objective(x) -> float:
    try:
        xa = np.asarray(x, dtype=float)
        p = float(_penalty(_plan(xa)))
        if USE_TWIN:                     # the twin must agree: -1 only if BOTH models close
            tp = twin_penalty(xa)
            if tp > 0.0:
                p = max(p, 0.0) + 1.0 + tp
        return p
    except Exception:
        return 100.0


def _ckpt(args, hist, xk, t0) -> bool:
    """Per-generation checkpoint: the server kills long jobs, so the best-so-far survives."""
    pen = float(objective(xk))
    hist.append(pen)
    print(f"gen {len(hist)} best {pen:.4f} t={time.time() - t0:.0f}s", flush=True)
    ck = args.out.with_suffix(".ckpt.json")
    ck.parent.mkdir(parents=True, exist_ok=True)
    ck.write_text(json.dumps({"gen": len(hist), "penalty": pen, "x": [float(v) for v in xk],
                              "fins": args.fins, "vscale": args.vscale}))
    return pen <= -1.0


def _warm(path):
    if path is None:
        return None
    x = list(json.loads(Path(path).read_text())["x"])
    x = x[:len(BOUNDS)]
    return np.clip(x, [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])


def main(argv=None) -> int:
    from scipy.optimize import differential_evolution

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fins", default="v3_pp")
    ap.add_argument("--vscale", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--maxiter", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--warm", type=Path, default=None)
    ap.add_argument("--twin", action="store_true", help="twin-in-the-loop: require the independent twin to close too")
    ap.add_argument("--time-one", action="store_true")
    args = ap.parse_args(argv)
    U._init(args.fins, args.vscale)
    global USE_TWIN
    USE_TWIN = bool(args.twin)
    from orbitopt.flightlab.contract import load_target

    U._CTX["recipe"] = {**RECIPE, "dt_opt": 0.5, "dt_final": 0.25}
    U._CTX["target"] = load_target(U.B.SCEN)
    if args.time_one:
        x = np.array([0.1, 3.4, 0.84, 11.0, 7.1, 6000.0, 300.0, 1.0, 40.0, 40000.0, 40.0, 12000.0, 0.0])
        t0 = time.time()
        pl = _plan(x)
        print(round(time.time() - t0, 2), "s", pl.feasible, _penalty(pl), pl.reason_codes,
              pl.corridor_report.entry_cutoff_altitude_m, pl.ignitions_used)
        return 0
    import multiprocessing as mp

    hist: list = []
    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers) as pool:
        res = differential_evolution(
            objective, BOUNDS, seed=args.seed, popsize=args.popsize, maxiter=args.maxiter, tol=0,
            mutation=(0.5, 1.0), recombination=0.8, polish=False, updating="deferred",
            workers=pool.map, init="sobol", x0=_warm(args.warm),
            callback=lambda xk, convergence=None: _ckpt(args, hist, xk, t0))
    x = np.asarray(res.x)
    plan = _plan(x)
    cr = plan.corridor_report
    row = {**U.B._row("continuous", plan), "fins": args.fins, "vscale": args.vscale, "x": [float(v) for v in x],
           "entry_cutoff_alt_m": cr.entry_cutoff_altitude_m, "ignitions_used": plan.ignitions_used,
           "penalty": float(res.fun), "nfev": int(res.nfev), "runtime_s": round(time.time() - t0, 1)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(row, indent=1) + "\n")
    print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
