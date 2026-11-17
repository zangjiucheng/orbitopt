"""E5-05c (task #287) offline global search for the RTLS recovery script.

The stock FL-302 planner (Nelder-Mead) stalls on the boundary of the F10
landing corridor once the stage-1 reserve is raised (ADR flightlab-0008).
This script swaps in a parallel differential-evolution search through
``plan_boostback_entry(search=...)`` -- same objective, same clamps, same
verdict path (re-integration at ``dt_final``, ``_assemble``, FL-305
landing) -- and records what it finds for each separation-mass increment.

``--dm`` adds propellant to the COMPUTED separation state of the recorded
E5-05a/E5-05d fixture at fixed position/velocity: a sizing proxy for the
stage-1 reserve (a heavier stage would also reach a slightly slower MECO,
which helps the return). The shipped answer still needs the vehicle
redesign + a re-flown ascent + a re-baked fixture.

Default recipe (every knob a CLI flag, recorded in the output):

* closed-loop return: IIP boostback cutoff (``x[1]`` = aim km past the
  pad), speed-cut entry burn (``x[6]`` = cutoff speed m/s; the planner
  rejects an engine-on handoff and asks for 1500 m of cutoff clearance above
  h0), throttled entry burn, closed-loop landing forecast;
* landing law at ``--landing-max-accel-g`` 3.6 g with the declared TVC
  actuator set, and the planner forecasts the same law (and its peak
  proper load). FL-305's 4.5 g plus the drag at ignition broke the 5 g
  landing-max-g limit; 4.0 g still flew 5.1 g on hot-engine Monte Carlo
  samples (the cap is a nominal-thrust command, a +5 % engine and the
  ignition drag ride on top), 3.6 g leaves that margin;
* planning aim radius 1 km (``corridor_aim_margin_m`` 7000 on the 8 km
  referee, which stays the acceptance radius);
* entry limits 12 g / 250 kPa (DESIGN ASSUMPTIONs, #287);
* step sizes ``--dt-opt`` 0.25 s / ``--dt-final`` 0.1 s: the FL-301
  fixed-step propagator samples triggers and cutoffs once per step and the
  verdict is NOT step-converged at the planner defaults (2 s / 0.5 s: the
  same +30 t script read 805 / 386 / 331 / 276 m/s at 2 / 0.5 / 0.25 /
  0.1 s).

Run from the repository root (Linux; uses fork for the worker pool)::

    python scripts/e505c_recovery_search.py --dm 27 30 33 --workers 32 \\
        --out search-out

Each run writes ``<out>/dm<T>_eb<N>.json`` with the accepted parameter
vector, the plan and mission-gate verdicts, the corridor report, the
touchdown read-out, a paste-ready ``plan_options`` recipe (input of
``scripts/e505c_recovery_montecarlo.py``) and the DE history. Needs the
same stack as the slow tests (scipy; gfold for the landing leg).
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
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from scipy.optimize import differential_evolution  # noqa: E402

from orbitopt.flightlab.contract import load_target, read_run  # noqa: E402
from orbitopt.flightlab.recovery.landing import DECLARED_LANDING_ACTUATOR  # noqa: E402
from orbitopt.flightlab.recovery.pipeline import (  # noqa: E402
    fly_recovery, separation_state_ecef)
from orbitopt.flightlab.recovery.propagator import BoosterModel  # noqa: E402

VEHICLE = REPO / "src" / "orbitopt" / "data" / "flightlab" / "vehicles" / \
    "synthetic_two_stage_rtls.yaml"
SCENARIO = REPO / "src" / "orbitopt" / "data" / "flightlab" / "scenarios" / \
    "leo_two_stage_rtls.yaml"
ASCENT = REPO / "tests" / "data" / "mission" / "reference" / \
    "e5-05a-rtls-mission" / "ascent"
H0_M = 3000.0  # FL-301 corridor height

#: best local-search scripts found at +30 t / 12 g (session scans, #287), in
#: the timed layout: one-burn [delay, bb_dur, bb_thr, pitch, yaw, h_trig,
#: eb_dur, eb_thr], two-burn [..., h_trig2, eb2_dur, eb_thr, eb2_thr].
#: seed_for() swaps in the closed-loop parameters (IIP aim, cutoff speed).
SEEDS = {
    1: [22.37, 49.76, 1.0, 0.0, -0.127, 20492.6, 18.54, 1.0],
    2: [23.22, 49.91, 1.0, 0.0, -0.083, 23359.7, 14.04, 5311.6, 3.0, 1.0, 0.1],
}
SEED_IIP_AIM_KM = 2.5
SEED_CUT_SPEED_M_S = 162.0
SEED_SPEED_CUT_TRIGGER_M = 22_000.0  # raised: the cut must fire above h0

_OBJ = None  # the planner objective; set before the worker pool forks


def _call(x):
    return _OBJ(np.asarray(x, dtype=np.float64))


def add_recipe_args(ap: argparse.ArgumentParser) -> None:
    """The recipe flags shared with the Monte Carlo script."""
    ap.add_argument("--entry-burns", type=int, choices=(1, 2), default=1)
    ap.add_argument("--boostback-cutoff", choices=("duration", "iip"), default="iip")
    ap.add_argument("--entry-cutoff", choices=("duration", "speed"), default="speed",
                    help="speed cutoff applies to the one-burn script only")
    ap.add_argument("--landing-need-model", choices=("momentum", "closed_loop"),
                    default="closed_loop")
    ap.add_argument("--landing-max-accel-g", type=float, default=3.6,
                    help="closed-loop landing law thrust ceiling [g] (FL-305 default 4.5; see docstring)")
    ap.add_argument("--no-actuator", action="store_true",
                    help="fly the landing without the declared TVC actuator set")
    ap.add_argument("--landing-reserve", default="target",
                    help="graded landing reserve: 'target' (the scenario's "
                         "min_propellant_reserve_kg, 10 t), 'vehicle' (the vehicle "
                         "file's 15 t planning number) or kg")
    ap.add_argument("--reserve-margin-kg", type=float, default=2500.0,
                    help="planning margin above the graded reserve "
                         "(landing_reserve_margin_kg; 2500 = the +48 t Monte Carlo nominal-to-p01 "
                         "touchdown spread, Isp 3 %%)")
    ap.add_argument("--aim-margin", type=float, default=7000.0,
                    help="corridor_aim_margin_m: planning aim radius = 8 km - this "
                         "(default 7000 -> 1 km; the 8 km referee is unchanged)")
    ap.add_argument("--g-max", type=float, default=12.0,
                    help="entry proper-load ceiling [g] (DESIGN ASSUMPTION; planner default 8)")
    ap.add_argument("--q-max", type=float, default=250_000.0,
                    help="entry dynamic-pressure ceiling [Pa] (ADR-0008 recipe)")
    ap.add_argument("--horizon", type=float, default=900.0)
    ap.add_argument("--dt-opt", type=float, default=0.25,
                    help="propagator step [s] inside the search (module docstring)")
    ap.add_argument("--dt-final", type=float, default=0.1,
                    help="propagator step [s] for the graded re-integration")


def build_recipe(args) -> dict[str, Any]:
    """``plan_options`` for ``fly_recovery`` (without ``initial_guess``)."""
    landing = {} if args.no_actuator else dict(DECLARED_LANDING_ACTUATOR)
    landing["max_thrust_accel_g"] = float(args.landing_max_accel_g)
    return dict(
        entry_burns=args.entry_burns, entry_throttle=True,
        boostback_cutoff=args.boostback_cutoff,
        entry_cutoff=args.entry_cutoff if args.entry_burns == 1 else "duration",
        landing_need_model=args.landing_need_model,
        corridor_aim_margin_m=float(args.aim_margin),
        entry_q_max_pa=float(args.q_max), entry_g_max=float(args.g_max),
        horizon_s=float(args.horizon), dt_opt=float(args.dt_opt),
        dt_final=float(args.dt_final), landing=landing,
        landing_reserve=(args.landing_reserve if args.landing_reserve in ("target", "vehicle")
                         else float(args.landing_reserve)),
        landing_reserve_margin_kg=float(args.reserve_margin_kg))


def seed_for(recipe: dict[str, Any]) -> list[float]:
    x = list(SEEDS[recipe["entry_burns"]])
    if recipe["boostback_cutoff"] == "iip":
        x[1] = SEED_IIP_AIM_KM
    if recipe["entry_cutoff"] == "speed":
        x[5] = SEED_SPEED_CUT_TRIGGER_M
        x[6] = SEED_CUT_SPEED_M_S
    return x


def bounds_for(recipe: dict[str, Any], throttle_min: float) -> list[tuple[float, float]]:
    """Box bounds mirroring the planner's ``clamp`` for this layout (the
    objective clamps again, so the dependent trigger-2 ceiling is enforced
    there)."""
    eb = recipe["entry_burns"]
    b = [(0.0, min(0.4 * recipe["horizon_s"], 150.0)),
         (-20.0, 80.0) if recipe["boostback_cutoff"] == "iip" else (4.0, 60.0),
         (throttle_min, 1.0), (0.0, 30.0), (-45.0, 45.0),
         (H0_M + 2000.0, 35_000.0),
         (60.0, 1500.0) if recipe["entry_cutoff"] == "speed" else (3.0, 45.0)]
    if eb == 2:
        b += [(H0_M + 300.0, 34_000.0), (3.0, 45.0)]
    b += [(throttle_min, 1.0)] * eb
    return b


def make_search(bounds, args, history: list, accepted: dict):
    """Build the ``search(objective, x_start)`` callable for the planner;
    ``accepted["x"]`` receives the vector the planner will keep (the DE best
    when it beats the start, else the start -- the planner's own rule)."""
    def search(obj, x_start):
        global _OBJ
        _OBJ = obj
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        f_start = float(obj(x_start))
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
        history.append({"final": True, "nfev": int(res.nfev), "start_penalty": f_start,
                        "best_penalty": float(res.fun), "message": res.message})
        accepted["x"] = [float(v) for v in (res.x if res.fun < f_start else x_start)]
        return res.x
    return search


def jsonable(v):
    if dataclasses.is_dataclass(v):
        return {k: jsonable(x) for k, x in dataclasses.asdict(v).items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def touchdown_readout(outcome) -> dict[str, Any] | None:
    """Landing-leg numbers the gates and the reserve sizing read."""
    land = outcome.landing
    if land is None:
        return None
    return {
        "landed": bool(land.landed), "reason_codes": list(land.reason_codes),
        "position_error_m": land.position_error_m,
        "vertical_speed_m_s": land.vertical_speed_m_s,
        "lateral_speed_m_s": land.lateral_speed_m_s,
        "max_proper_g": float(land.max_proper_g),
        "propellant_remaining_kg": (float(land.track.propellant_remaining_kg)
                                    if land.track is not None else None),
    }


def run_one(dm_t: float, args, run, target, booster) -> dict:
    sep = separation_state_ecef(run)
    sep = dataclasses.replace(sep, mass_kg=sep.mass_kg + dm_t * 1000.0)
    recipe = build_recipe(args)
    x0 = json.loads(args.x0) if args.x0 else seed_for(recipe)
    bounds = bounds_for(recipe, booster.engine.throttle_min)
    if len(x0) != len(bounds):
        raise SystemExit(f"--x0 has {len(x0)} entries; this layout has {len(bounds)}")
    history: list = []
    accepted: dict = {"x": list(x0)}
    t0 = time.time()
    print(f"== dm +{dm_t:g} t (sep {sep.mass_kg / 1e3:.1f} t), {len(bounds)} params, "
          f"recipe {json.dumps({k: v for k, v in recipe.items() if k != 'landing'})}",
          flush=True)
    out = fly_recovery(run, target, VEHICLE, sep_state=sep,
                       plan_options={**recipe, "initial_guess": x0,
                                     "search": make_search(bounds, args, history,
                                                           accepted)})
    p = out.plan
    checks = {c.id: c.passed for c in out.checks}
    result = {
        "dm_t": dm_t, "sep_mass_kg": sep.mass_kg,
        "wall_s": round(time.time() - t0, 1),
        "closed": bool(checks.get("recovery-landing")
                       and checks.get("recovery-propellant-reserve")),
        "checks": checks,
        "plan_feasible": p.feasible, "plan_reason_codes": list(p.reason_codes),
        "corridor_report": jsonable(p.corridor_report),
        "touchdown": touchdown_readout(out),
        "x_best": accepted["x"],
        "plan_options_recipe": {**recipe, "initial_guess": accepted["x"]},
        "search": {"popsize": args.popsize, "maxiter": args.maxiter,
                   "seed": args.seed, "workers": args.workers,
                   "history": history},
    }
    r = p.corridor_report
    td = result["touchdown"] or {}
    print(f"   closed={result['closed']} checks={checks} plan={p.reason_codes}\n"
          f"   miss={r.miss_m} v={r.speed_m_s} cut_alt={r.entry_cutoff_altitude_m} "
          f"prop={r.propellant_available_kg} need={r.landing_need_kg} "
          f"entry_g={r.entry_max_proper_g} | touchdown prop="
          f"{td.get('propellant_remaining_kg')} max_g={td.get('max_proper_g')}",
          flush=True)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dm", type=float, nargs="+", default=[30.0],
                    help="separation-mass increments [t] to search (default 30)")
    add_recipe_args(ap)
    ap.add_argument("--popsize", type=int, default=12,
                    help="DE population multiplier (population = popsize * n_params)")
    ap.add_argument("--maxiter", type=int, default=60, help="DE generations")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seed", type=int, default=287)
    ap.add_argument("--target-penalty", type=float, default=None,
                    help="stop early once the best penalty drops to this value")
    ap.add_argument("--x0", default=None,
                    help="JSON seed vector in this layout (default: seed_for(recipe))")
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
