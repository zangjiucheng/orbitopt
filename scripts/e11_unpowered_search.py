"""E11-06 search: jointly optimise the boostback script AND the grid-fin entry schedule so the
two-ignition, UNPOWERED-entry RTLS from Starbase closes every graded limit.

Decision vector (10): boostback [delay_s, iip_aim_km, throttle, pitch_deg, yaw_deg] and the fin
schedule [alpha1, h1, alpha2, h2, alpha3] with gates (70 km, alpha1) (h1, alpha2) (h2, alpha3).
The graded limits are the planner's own (``plan.feasible``): 3 km / <=350 m/s / <=15 deg / 8 km
corridor, q <= 250 kPa, 12 g, landing reserve. Nothing is relaxed; the penalty only shapes the
search, the verdict is ``plan.feasible``.

Usage: e11_unpowered_search.py --fins v3 --out out.json [--workers 40 --popsize 12 --maxiter 40]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks"))

import bench_e11_06_starbase_grid_fin_scaling as B  # noqa: E402

_CTX: dict = {}
BOUNDS = [(0.0, 120.0), (-40.0, 120.0), (0.4, 1.0), (-40.0, 40.0), (-45.0, 45.0),
          (0.0, 60.0), (12000.0, 60000.0), (0.0, 60.0), (3500.0, 20000.0), (0.0, 60.0)]


def _setup(fins: str, tmp: Path, vscale: float = 1.0) -> dict:
    from dataclasses import replace

    from orbitopt.flightlab.contract import load_target, read_run
    from orbitopt.flightlab.recovery.boostback import RecoveryTarget
    from orbitopt.flightlab.recovery.pipeline import _to_ecef_state, landing_reserve_booster
    from orbitopt.flightlab.recovery.propagator import BoosterModel
    from orbitopt.mission.coupled_branches import booster_state_from_run, separation_handoff

    target = load_target(B.SCEN)
    run = read_run(B.ASCENT)
    sep = booster_state_from_run(run, separation_handoff(run))
    if sep.frame != "ECEF":
        r, v = _to_ecef_state(sep)
        sep = replace(sep, frame="ECEF", position_m=tuple(float(c) for c in r),
                      velocity_m_s=tuple(float(c) for c in v))
    if vscale != 1.0:                      # screening: a slower separation state (not a flown ascent)
        sep = replace(sep, velocity_m_s=tuple(float(c) * vscale for c in sep.velocity_m_s))
    veh = landing_reserve_booster(BoosterModel.load(B._vehicle_yaml(fins, tmp)), target, "target")
    return {"sep": sep, "veh": veh, "rt": RecoveryTarget.from_scenario(target.to_dict())}


def _schedule(x) -> dict:
    a1, h1, a2, h2, a3 = (float(v) for v in x[5:10])
    h2 = min(h2, h1 - 1000.0)
    return {"mode": "schedule", "alpha_deg": a1,
            "schedule": [[70000.0, 0.0, a1], [h1, 0.0, a2], [h2, 0.0, a3]]}


def _plan(x):
    from orbitopt.flightlab.recovery.boostback import plan_boostback_entry

    c = _CTX
    return plan_boostback_entry(c["sep"], c["veh"], c["rt"], entry_aero=_schedule(x),
                                initial_guess=[float(v) for v in x[:5]], search="keep", **c["recipe"])


def _penalty(plan) -> float:
    cr = plan.corridor_report
    if plan.feasible:
        return -1.0
    p = 1.0                                              # any miss of the verdict costs at least 1
    if cr.speed_m_s is None:
        return 50.0
    p += max(0.0, cr.speed_m_s - 350.0) / 350.0 + max(0.0, (cr.tilt_deg or 0.0) - 15.0) / 15.0
    p += max(0.0, (cr.miss_m or 0.0) - 7000.0) / 8000.0
    if cr.entry_max_q_pa:
        p += max(0.0, cr.entry_max_q_pa - 250e3) / 250e3
    if cr.landing_need_kg is not None:
        # the verdict's own test: prop >= landing need + the graded reserve (100 t for these targets)
        p += max(0.0, cr.landing_need_kg + (cr.reserve_kg or 0.0) + 2500.0
                 - cr.propellant_available_kg) / 1e5
    return p


def objective(x) -> float:
    try:
        return float(_penalty(_plan(x)))
    except Exception:                                    # a script the planner refuses is just bad
        return 100.0


def _init(fins: str, vscale: float = 1.0):
    tmp = Path(tempfile.mkdtemp(prefix="e11fins_"))
    from orbitopt.reference_eop import freeze_reference, load_snapshot

    _CTX["eop"] = freeze_reference(load_snapshot(B.REF / "earth_orientation.json"))
    _CTX["eop"].__enter__()
    _CTX.update(_setup(fins, tmp, vscale))
    _CTX["recipe"] = {**B.RECIPE, "dt_opt": 0.5, "dt_final": 0.25}


def _warm(paths):
    xs = [np.clip(json.loads(Path(p).read_text())["x"], [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])
          for p in paths]
    return xs[0] if len(xs) == 1 else None      # scipy takes one seed vector; extras go in via bounds


def main(argv=None) -> int:
    from scipy.optimize import differential_evolution

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fins", default="v3")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--maxiter", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--warm", type=Path, nargs="*", default=[], help="result jsons whose x seeds the population")
    ap.add_argument("--vscale", type=float, default=1.0)
    ap.add_argument("--time-one", action="store_true")
    args = ap.parse_args(argv)
    _init(args.fins, args.vscale)
    if args.time_one:
        x = np.array([54.0, 3.4, 0.84, 11.0, 7.1, 60.0, 20000.0, 60.0, 9000.0, 0.0])
        t0 = time.time()
        pl = _plan(x)
        print(round(time.time() - t0, 2), "s", pl.feasible, _penalty(pl), pl.reason_codes)
        return 0

    t0 = time.time()
    hist: list = []
    import multiprocessing as mp

    with mp.get_context("fork").Pool(args.workers) as pool:
        res = differential_evolution(
            objective, BOUNDS, seed=args.seed, popsize=args.popsize, maxiter=args.maxiter,
            tol=0, mutation=(0.5, 1.0), recombination=0.8, polish=False, updating="deferred",
            workers=pool.map, init="sobol", x0=_warm(args.warm),
            callback=lambda xk, convergence=None: hist.append(float(objective(xk))) or
            print(f"gen {len(hist)} best {hist[-1]:.4f} t={time.time() - t0:.0f}s", flush=True)
            or (hist[-1] <= -1.0))
    x = res.x
    plan = _plan(x)
    row = {**B._row("search", plan), "fins": args.fins, "x": [float(v) for v in x],
           "schedule": _schedule(x), "penalty": float(res.fun), "nfev": int(res.nfev),
           "runtime_s": round(time.time() - t0, 1)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(row, indent=1) + "\n")
    print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
