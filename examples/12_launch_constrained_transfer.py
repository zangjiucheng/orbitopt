"""Earth->Mars Lambert transfer optimized with and without real-world launch
constraints (rocket C3 capability + launch-site DLA reachability), printing
how the optimum moves when the constraints are switched on.

Run: python examples/12_launch_constrained_transfer.py [--vehicle falcon-9]
        [--payload 4000] [--site cape-canaveral] [--mode invalid|penalty]
        [--penalty-weight 1000] [--start-year 2028]

Results for the defaults are recorded in docs/engineering-notes.md.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pygmo as pg
import pykep as pk

from orbitopt.bodies import mjd2000_from_date, planet
from orbitopt.lambert.gpu_batch import solve_lambert_batch
from orbitopt.launch_constraints import LaunchConstraints
from orbitopt.optimize.runner import run_optimization
from orbitopt.problems.transfer_2body import Transfer2BodyProblem


def _describe_solution(problem, x, f):
    t0, tof = float(x[0]), float(x[1])
    r1, v1p, r2, _ = problem._ephemerides(t0, tof)
    res = solve_lambert_batch(r1[None, :], r2[None, :], np.array([tof * pk.DAY2SEC]), problem.mu, use_gpu=False)
    v_inf = res.v1[0] - v1p
    ev = problem.launch_constraints.evaluate(v_inf[None, :])
    return {
        "t0_mjd2000": round(t0, 2),
        "departure_date": str(pk.epoch(t0)),
        "tof_days": round(tof, 2),
        "fitness_m_s": round(float(f[0]), 1),
        "c3_km2_s2": round(float(ev.c3[0]), 3),
        "dla_deg": round(float(ev.dla_deg[0]), 2),
        "feasible": bool(ev.feasible[0]),
    }


def run(vehicle, payload, site, mode, t0_years=(2028, 2029), penalty_weight=1000.0, seed=7, generations=150, pop_size=512):
    earth, mars = planet("earth"), planet("mars")
    t0_bounds = (mjd2000_from_date(t0_years[0], 1, 1), mjd2000_from_date(t0_years[1], 1, 1))
    lc = LaunchConstraints.from_names(vehicle, payload, site, mode=mode, penalty_weight=penalty_weight)
    out = {"constraints": lc.describe(), "runs": {}}
    for label, enabled in (("unconstrained", False), ("constrained", True)):
        problem = Transfer2BodyProblem(
            earth, mars, t0_bounds, (100.0, 400.0), launch_constraints=lc.with_enabled(enabled),
        )
        x, f, elapsed = run_optimization(
            problem, algorithm=pg.pso_gen(gen=generations, seed=seed),
            pop_size=pop_size, generations=generations, seed=seed, verbose=False,
        )
        sol = _describe_solution(problem, x, f)
        sol["elapsed_s"] = round(elapsed, 2)
        out["runs"][label] = sol
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vehicle", default="falcon-9")
    ap.add_argument("--payload", type=float, default=4000.0)
    ap.add_argument("--site", default="cape-canaveral")
    ap.add_argument("--mode", default="invalid", choices=("invalid", "penalty"))
    ap.add_argument("--penalty-weight", type=float, default=1000.0)
    ap.add_argument("--start-year", type=int, default=2028)
    args = ap.parse_args()
    out = run(args.vehicle, args.payload, args.site, args.mode,
              (args.start_year, args.start_year + 1), penalty_weight=args.penalty_weight)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
