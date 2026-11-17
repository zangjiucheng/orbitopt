"""E5-05c (task #287) offline global search for the RTLS recovery script.

The stock FL-302 planner (Nelder-Mead) stalls on the boundary of the F10
landing corridor once the stage-1 reserve is raised (ADR flightlab-0008).
This script swaps in a parallel differential-evolution search through
``plan_boostback_entry(search=...)`` -- same objective, same clamps, same
verdict path (re-integration at ``dt_final``, ``_assemble``, FL-303/305
landing) -- and records what it finds for each separation-mass increment.

``--dm`` adds propellant to the COMPUTED separation state of the recorded
E5-05a/E5-05d fixture at fixed position/velocity: a sizing proxy for the
stage-1 reserve (a heavier stage would also reach a slightly slower MECO,
which helps the return). The shipped answer still needs the vehicle
redesign + a re-flown ascent + a re-baked fixture.

Step size matters: the FL-301 fixed-step propagator samples the scripted
guidance (entry triggers, burn cutoffs) once per step, and the corridor
verdict is NOT step-converged at the planner defaults (dt_opt 2 s,
dt_final 0.5 s): the same +30 t script reads 805 m/s at 2 s, 386 m/s at
0.5 s, 331 m/s at 0.25 s and 276 m/s at 0.1 s. The search therefore runs at
``--dt-opt`` (default 0.25 s) and grades at ``--dt-final`` (default 0.1 s);
the recipe it writes pins both.

Run from the repository root (Linux; uses fork for the worker pool)::

    python scripts/e505c_recovery_search.py --dm 27 30 33 --entry-burns 1 \\
        --g-max 12 --workers 32 --out search-out

Each run writes ``<out>/dm<T>_eb<N>.json`` with the best script, the plan
and mission-gate verdicts, the corridor report, a paste-ready
``plan_options`` recipe and the DE convergence history. Needs the same
stack as the slow tests (scipy; gfold for the landing leg).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from scipy.optimize import differential_evolution  # noqa: E402

from orbitopt.flightlab.contract import load_target, read_run  # noqa: E402
from orbitopt.flightlab.recovery.pipeline import (  # noqa: E402
    fly_recovery, separation_state_ecef)
from orbitopt.flightlab.recovery.propagator import BoosterModel  # noqa: E402

VEHICLE = REPO / "src" / "orbitopt" / "data" / "flightlab" / "vehicles" / \
    "synthetic_two_stage_rtls.yaml"
SCENARIO = REPO / "src" / "orbitopt" / "data" / "flightlab" / "scenarios" / \
    "leo_two_stage_rtls.yaml"
ASCENT = REPO / "tests" / "data" / "mission" / "reference" / \
    "e5-05a-rtls-mission" / "ascent"

#: best local-search scripts found at +30 t / 12 g (session scans, #287):
#: one-burn [delay, bb_dur, bb_thr, pitch, yaw, h_trig, eb_dur, eb_thr] and
#: two-burn [..., h_trig2, eb2_dur, eb_thr, eb2_thr]
SEEDS = {
    1: [22.37, 49.76, 1.0, 0.0, -0.127, 20492.6, 18.54, 1.0],
    2: [23.22, 49.91, 1.0, 0.0, -0.083, 23359.7, 14.04, 5311.6, 3.0, 1.0, 0.1],
}

_OBJ = None  # the planner objective; set before the worker pool forks


def _call(x):
    return _OBJ(np.asarray(x, dtype=np.float64))


def bounds_for(entry_burns: int, throttle_min: float, h0: float,
               horizon_s: float) -> list[tuple[float, float]]:
    """Box bounds mirroring the planner's ``clamp`` (the objective clamps
    again, so the dependent trigger-2 ceiling is enforced there)."""
    b = [(0.0, min(0.4 * horizon_s, 150.0)), (4.0, 60.0), (throttle_min, 1.0),
         (0.0, 30.0), (-45.0, 45.0), (h0 + 2000.0, 35_000.0), (3.0, 45.0)]
    if entry_burns == 2:
        b += [(h0 + 300.0, 34_000.0), (3.0, 45.0)]
    b += [(throttle_min, 1.0)] * entry_burns
    return b


def make_search(bounds, args, history: list):
    """Build the ``search(objective, x_start)`` callable for the planner."""
    def search(obj, x_start):
        global _OBJ
        _OBJ = obj
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        t0 = time.time()

        def cb(intermediate_result):
            history.append({"gen": len(history) + 1,
                            "best_penalty": float(intermediate_result.fun),
                            "wall_s": round(time.time() - t0, 1)})
            print(f"  gen {len(history):3d}  best {intermediate_result.fun:.4g}  "
                  f"{time.time() - t0:.0f}s", flush=True)
            return (args.target_penalty is not None
                    and intermediate_result.fun <= args.target_penalty)

        with mp.get_context("fork").Pool(args.workers) as pool:
            res = differential_evolution(
                _call, bounds, x0=np.clip(x_start, lo, hi), seed=args.seed,
                popsize=args.popsize, maxiter=args.maxiter, mutation=(0.5, 1.0),
                recombination=0.7, init="latinhypercube", tol=0.0,
                polish=False, updating="deferred", workers=pool.map,
                callback=cb)
        history.append({"final": True, "nfev": int(res.nfev),
                        "best_penalty": float(res.fun), "message": res.message})
        return res.x
    return search


def _jsonable(v):
    if dataclasses.is_dataclass(v):
        return {k: _jsonable(x) for k, x in dataclasses.asdict(v).items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def run_one(dm_t: float, args, run, target, booster) -> dict:
    sep = separation_state_ecef(run)
    sep = dataclasses.replace(sep, mass_kg=sep.mass_kg + dm_t * 1000.0)
    eb = args.entry_burns
    x0 = json.loads(args.x0) if args.x0 else SEEDS[eb]
    bounds = bounds_for(eb, booster.engine.throttle_min, 3000.0, args.horizon)
    history: list = []
    recipe = dict(entry_burns=eb, entry_throttle=True, initial_guess=x0,
                  entry_q_max_pa=args.q_max, entry_g_max=args.g_max,
                  horizon_s=args.horizon, dt_opt=args.dt_opt,
                  dt_final=args.dt_final)
    t0 = time.time()
    print(f"== dm +{dm_t:g} t (sep {sep.mass_kg / 1e3:.1f} t), entry_burns={eb}, "
          f"{len(bounds)} params", flush=True)
    out = fly_recovery(run, target, VEHICLE, sep_state=sep,
                       plan_options={**recipe,
                                     "search": make_search(bounds, args, history)})
    p = out.plan
    x_best = [p.boostback.start_delay_s, p.boostback.duration_s, p.boostback.throttle,
              p.boostback.pitch_deg, p.boostback.yaw_deg,
              p.entry_burn.trigger_altitude_m, p.entry_burn.duration_s]
    if p.entry_burn_2 is not None:
        x_best += [p.entry_burn_2.trigger_altitude_m, p.entry_burn_2.duration_s]
    x_best += [p.entry_burn.throttle] + (
        [p.entry_burn_2.throttle] if p.entry_burn_2 is not None else [])
    checks = {c.id: c.passed for c in out.checks}
    result = {
        "dm_t": dm_t, "sep_mass_kg": sep.mass_kg, "entry_burns": eb,
        "wall_s": round(time.time() - t0, 1),
        "closed": bool(checks.get("recovery-landing")
                       and checks.get("recovery-propellant-reserve")),
        "checks": checks,
        "plan_feasible": p.feasible, "plan_reason_codes": list(p.reason_codes),
        "landing_reason_codes": list(out.landing.reason_codes) if out.landing else None,
        "corridor_report": _jsonable(p.corridor_report),
        "x_best": x_best,
        "plan_options_recipe": {**recipe, "initial_guess": x_best},
        "search": {"popsize": args.popsize, "maxiter": args.maxiter,
                   "seed": args.seed, "workers": args.workers,
                   "history": history},
    }
    r = p.corridor_report
    print(f"   closed={result['closed']} checks={checks} plan={p.reason_codes} "
          f"miss={r.miss_m} v={r.speed_m_s} prop={r.propellant_available_kg} "
          f"need={r.landing_need_kg} g={r.entry_max_proper_g}", flush=True)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dm", type=float, nargs="+", default=[30.0],
                    help="separation-mass increments [t] to search (default 30)")
    ap.add_argument("--entry-burns", type=int, choices=(1, 2), default=1)
    ap.add_argument("--g-max", type=float, default=12.0,
                    help="entry proper-load ceiling [g] (DESIGN ASSUMPTION; planner default 8)")
    ap.add_argument("--q-max", type=float, default=250_000.0,
                    help="entry dynamic-pressure ceiling [Pa] (ADR-0008 recipe)")
    ap.add_argument("--horizon", type=float, default=900.0)
    ap.add_argument("--dt-opt", type=float, default=0.25,
                    help="propagator step [s] inside the search (planner default 2.0 "
                         "is not step-converged; see module docstring)")
    ap.add_argument("--dt-final", type=float, default=0.1,
                    help="propagator step [s] for the graded re-integration")
    ap.add_argument("--popsize", type=int, default=12,
                    help="DE population multiplier (population = popsize * n_params)")
    ap.add_argument("--maxiter", type=int, default=60, help="DE generations")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seed", type=int, default=287)
    ap.add_argument("--target-penalty", type=float, default=None,
                    help="stop early once the best penalty drops to this value")
    ap.add_argument("--x0", default=None,
                    help="JSON seed vector (default: the session's +30 t best)")
    ap.add_argument("--out", type=Path, default=Path("search-out"))
    args = ap.parse_args(argv)

    run = read_run(ASCENT)
    target = load_target(SCENARIO)
    booster = BoosterModel.load(VEHICLE)
    args.out.mkdir(parents=True, exist_ok=True)
    any_closed = False
    for dm_t in args.dm:
        res = run_one(dm_t, args, run, target, booster)
        path = args.out / f"dm{dm_t:g}_eb{args.entry_burns}.json"
        path.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
        print(f"   -> {path}", flush=True)
        any_closed |= res["closed"]
    return 0 if any_closed else 1


if __name__ == "__main__":
    raise SystemExit(main())
