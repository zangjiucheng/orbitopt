"""E5-05c (task #287) Monte Carlo of the RTLS recovery recipe: reserve sizing.

Flies ``--n`` dispersed recoveries of ONE recipe from the computed
separation state of the recorded E5-05a/E5-05d fixture (plus ``--dm``
tonnes of stage-1 reserve, the same sizing proxy as
``scripts/e505c_recovery_search.py``) and reports what the reserve has to
cover: the touchdown-propellant distribution against the scenario's
``min_propellant_reserve_kg`` gate, the success rate and the failure modes.

Dispersions are the FL-304 ones (thrust / Isp / Cd from the vehicle file's
uncertainty tags, density scale, separation state; ``--uncertainty`` takes
an R-04 YAML/JSON declaration instead), drawn per ``[seed, idx]`` exactly
like :func:`orbitopt.flightlab.recovery.montecarlo.run_monte_carlo`, so a
seed reproduces the population regardless of ``--workers``. Two
differences from that runner, both deliberate for this question:

* ``--mode fixed`` (default) flies the recipe's script AS IS in every
  sample (no re-plan): the closed-loop pieces -- IIP boostback cutoff,
  speed-cut entry burn, FL-305 landing law -- are what absorb the
  dispersion, which is the property being sized. ``--mode replan`` refines
  the script per sample with the stock search (``--max-fevals``, FL-304
  style: below 40 stays local).
* the density/wind draw also flies the corridor leg (planner
  ``verdict_physics``), not only the landing burn: a speed-cut entry burn
  is precisely what a density error stresses.

Planner reserve probe (``--probe-landing``, default on): the FL-302 planner
refuses a handoff whose propellant is below landing need + the vehicle's
``landing_propellant_reserve_kg``, and then no landing is flown -- exactly
the regime being sized. When that is the ONLY planning failure, the sample
flies the landing burn anyway (same FL-305 call ``fly_recovery`` makes) so
its touchdown propellant is measured; the sample still counts as failed
(its reason stays the planner's), the record carries ``probe: true``.

Reserve reading: with ``q = --quantile`` (default 0.05), the q-quantile of
touchdown propellant over the samples that reached touchdown is compared
with the gate; a shortfall is translated to an extra separation-mass
estimate by the rocket-equation ratio sep mass / touchdown mass (first
order: a heavier stage also flies a slightly different script) -- iterate
with ``--dm`` to confirm.

Run from the repository root::

    python scripts/e505c_recovery_montecarlo.py --recipe search-out/dm30_eb1.json \\
        --dm 30 --n 64 --workers 8 --out mc-out/dm30.json

``--recipe`` takes a search output (its ``plan_options_recipe``) or a plain
``plan_options`` JSON; without it the default recipe flags of the search
script and its seed vector are used. Needs gfold for the landing leg.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from e505c_recovery_search import (  # noqa: E402
    ASCENT, H0_M, SCENARIO, VEHICLE, add_recipe_args, build_recipe, jsonable,
    seed_for, touchdown_readout)

from orbitopt.flightlab.contract import load_target, read_run  # noqa: E402
from orbitopt.flightlab.recovery.boostback import (  # noqa: E402
    INSUFFICIENT_PROPELLANT_RESERVE, RecoveryTarget)
from orbitopt.flightlab.recovery.pipeline import (  # noqa: E402
    _landing_flight, _landing_limits_for_target, dispersed_separation_state,
    landing_reserve_booster,
    fly_recovery, separation_state_ecef)
from orbitopt.flightlab.recovery.propagator import (  # noqa: E402
    BoosterModel, BoosterPhysics)
from orbitopt.flightlab.recovery.stats import success_rate_interval  # noqa: E402
from orbitopt.flightlab.recovery.uncertainty import UncertaintyModel  # noqa: E402


def _keep_script(obj, x):  # --mode fixed: the planner keeps the recipe's script
    return x


def run_sample(job: dict[str, Any]) -> dict[str, Any]:
    """One dispersed recovery (pool worker; plain data in, plain data out)."""
    idx = int(job["idx"])
    model = UncertaintyModel.from_dict(job["uncertainty"])
    rng = np.random.default_rng([int(job["seed"]), idx])
    draw = model.draw(rng)
    rec: dict[str, Any] = {"idx": idx, "draw": {k: float(v) for k, v in draw.items()}}
    try:
        run = read_run(Path(job["run_dir"]))
        vehicle = BoosterModel.load(job["vehicle"])
        eng = vehicle.engine
        vehicle = dataclasses.replace(
            vehicle,
            engine=dataclasses.replace(
                eng, thrust_vacuum_n=eng.thrust_vacuum_n * draw["thrust_factor"],
                isp_sea_level_s=eng.isp_sea_level_s * draw["isp_factor"],
                isp_vacuum_s=eng.isp_vacuum_s * draw["isp_factor"]),
            cd_mach=tuple((m, c * draw["cd_factor"]) for m, c in vehicle.cd_mach))
        physics = BoosterPhysics(density_scale=draw["density_scale"],
                                 wind_east_m_s=draw["wind_east_m_s"],
                                 wind_north_m_s=draw["wind_north_m_s"],
                                 wind_up_m_s=draw["wind_up_m_s"])
        sep = separation_state_ecef(run)
        sep = dataclasses.replace(sep, mass_kg=sep.mass_kg + float(job["dm_kg"]))
        sep = dispersed_separation_state(
            sep, altitude_m=draw["sep_altitude_m"], downrange_m=draw["sep_downrange_m"],
            crossrange_m=draw["sep_crossrange_m"], speed_m_s=draw["sep_speed_m_s"],
            heading_deg=draw["sep_heading_deg"])
        opts = dict(job["plan_options"])
        opts["verdict_physics"] = physics
        if job["mode"] == "fixed":
            opts["search"] = _keep_script
        else:
            opts["max_fevals"] = int(job["max_fevals"])
        target = load_target(job["scenario"])
        out = fly_recovery(run, target, job["vehicle"],
                           sep_state=sep, booster=vehicle, physics=physics,
                           plan_options=opts)
        p = out.plan
        r = p.corridor_report
        td = touchdown_readout(out)
        if (td is None and job["probe_landing"]
                and tuple(p.reason_codes) == (INSUFFICIENT_PROPELLANT_RESERVE,)
                and p.landing_burn_initial_state is not None):
            land = _landing_flight(opts.get("landing"))(
                p.landing_burn_initial_state_enu, p.landing_burn_initial_state,
                RecoveryTarget.from_scenario(target.to_dict()),
                landing_reserve_booster(vehicle, target, opts.get("landing_reserve")),
                limits=_landing_limits_for_target(target), physics=physics)
            td = touchdown_readout(dataclasses.replace(out, landing=land))
            rec["probe"] = True
        rec.update(
            outcome="landed" if out.success else "failed",
            reason="LANDED" if out.success else out.reason_codes[0],
            checks={c.id: bool(c.passed) for c in out.checks},
            plan_reason_codes=list(p.reason_codes),
            corridor={"miss_m": r.miss_m, "speed_m_s": r.speed_m_s,
                      "entry_cutoff_altitude_m": r.entry_cutoff_altitude_m,
                      "propellant_kg": r.propellant_available_kg,
                      "landing_need_kg": r.landing_need_kg,
                      "entry_max_g": r.entry_max_proper_g,
                      "entry_max_q_pa": r.entry_max_q_pa},
            touchdown=td)
    except Exception as exc:  # noqa: BLE001 -- one broken sample must not erase the run
        rec.update(outcome="error", reason=f"SAMPLE_ERROR:{type(exc).__name__}",
                   error=str(exc))
    return rec


def _pct(values: list[float]) -> dict[str, float] | None:
    v = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not v:
        return None
    q = np.percentile(v, [1, 5, 50, 95, 99])
    return {"n": len(v), "min": min(v), "p01": q[0], "p05": q[1], "p50": q[2],
            "p95": q[3], "p99": q[4], "max": max(v)}


def summarise(records: list[dict], *, gate_kg: float, quantile: float,
              sep_mass_kg: float, dry_kg: float) -> dict[str, Any]:
    n = len(records)
    landed = sum(1 for r in records if r["outcome"] == "landed")
    corr = [r.get("corridor") or {} for r in records]
    td = [r.get("touchdown") or {} for r in records]
    td_prop = [t.get("propellant_remaining_kg") for t in td
               if t.get("propellant_remaining_kg") is not None]
    cut = [c.get("entry_cutoff_altitude_m") for c in corr]
    summary: dict[str, Any] = {
        "n": n, "landed": landed,
        "probe_landings": sum(1 for r in records if r.get("probe")),
        "probe_touchdowns_in_limits": sum(1 for r in records if r.get("probe")
                                          and (r.get("touchdown") or {}).get("landed")), "success_rate": landed / n,
        "success_ci95": success_rate_interval(landed, n),
        "reasons": dict(Counter(r["reason"] for r in records).most_common()),
        "engine_on_handoffs": sum(1 for h in cut if h is not None and h < H0_M),
        "percentiles": {
            "touchdown_propellant_kg": _pct(td_prop),
            "entry_cutoff_altitude_m": _pct(cut),
            "corridor_speed_m_s": _pct([c.get("speed_m_s") for c in corr]),
            "corridor_miss_m": _pct([c.get("miss_m") for c in corr]),
            "entry_max_g": _pct([c.get("entry_max_g") for c in corr]),
            "landing_max_g": _pct([t.get("max_proper_g") for t in td]),
            "touchdown_position_error_m": _pct([t.get("position_error_m") for t in td]),
        },
    }
    reserve: dict[str, Any] = {"gate_kg": gate_kg, "quantile": quantile,
                               "touchdowns": len(td_prop)}
    if td_prop:
        q_val = float(np.quantile(td_prop, quantile))
        shortfall = max(0.0, gate_kg - q_val)
        td_mass = dry_kg + float(np.median(td_prop))
        reserve.update(
            touchdown_propellant_at_quantile_kg=q_val,
            shortfall_kg=shortfall,
            extra_separation_mass_estimate_kg=shortfall * sep_mass_kg / td_mass,
            note=("first order: shortfall scaled by sep mass / median touchdown "
                  "mass (carried through boostback + entry); confirm with --dm"))
    else:
        reserve["note"] = "no sample reached touchdown: nothing to size from"
    summary["reserve"] = reserve
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--recipe", type=Path, default=None,
                    help="search output JSON (plan_options_recipe) or plan_options JSON")
    add_recipe_args(ap)
    ap.add_argument("--dm", type=float, default=30.0, help="stage-1 reserve increment [t]")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--mode", choices=("fixed", "replan"), default="fixed")
    ap.add_argument("--max-fevals", type=int, default=30,
                    help="--mode replan: per-sample search budget (< 40 stays local)")
    ap.add_argument("--uncertainty", type=Path, default=None,
                    help="R-04 YAML/JSON declaration (default: vehicle-tag dispersions)")
    ap.add_argument("--quantile", type=float, default=0.05)
    ap.add_argument("--no-probe-landing", action="store_true",
                    help="do not fly the landing of planner-reserve-only failures")
    ap.add_argument("--out", type=Path, default=Path("mc-out") / "montecarlo.json")
    args = ap.parse_args(argv)

    if args.recipe is not None:
        doc = json.loads(args.recipe.read_text(encoding="utf-8"))
        recipe = dict(doc.get("plan_options_recipe", doc))
    else:
        recipe = build_recipe(args)
        recipe["initial_guess"] = seed_for(recipe)
    if "initial_guess" not in recipe:
        raise SystemExit("the recipe carries no initial_guess (the script to fly)")

    if args.uncertainty is not None:
        model = UncertaintyModel.from_file(args.uncertainty)
        unc_note = str(args.uncertainty)
    else:
        from orbitopt.flightlab.vehicle_catalog import vehicle_dispersions
        disp, notes = vehicle_dispersions(VEHICLE)
        model = UncertaintyModel.from_dispersions(disp)
        unc_note = "vehicle-tag dispersions: " + "; ".join(notes)

    target = load_target(SCENARIO)
    booster = BoosterModel.load(VEHICLE)
    sep_mass = separation_state_ecef(read_run(ASCENT)).mass_kg + args.dm * 1000.0
    jobs = [{"idx": i, "seed": args.seed, "run_dir": str(ASCENT),
             "vehicle": str(VEHICLE), "scenario": str(SCENARIO),
             "uncertainty": model.to_dict(), "plan_options": recipe,
             "dm_kg": args.dm * 1000.0, "mode": args.mode,
             "max_fevals": args.max_fevals,
             "probe_landing": not args.no_probe_landing} for i in range(args.n)]
    print(f"== {args.n} samples, dm +{args.dm:g} t (sep {sep_mass / 1e3:.1f} t), "
          f"mode {args.mode}, seed {args.seed}, {args.workers} workers", flush=True)
    t0 = time.time()
    records: list[dict] = []
    if args.workers > 1 and args.n > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for rec in pool.map(run_sample, jobs):
                records.append(rec)
                print(f"  #{rec['idx']:3d} {rec['reason']}", flush=True)
    else:
        for job in jobs:
            records.append(run_sample(job))
            print(f"  #{records[-1]['idx']:3d} {records[-1]['reason']}", flush=True)

    summary = summarise(records, gate_kg=float(target.recovery.min_propellant_reserve_kg),
                        quantile=args.quantile, sep_mass_kg=sep_mass,
                        dry_kg=float(booster.dry_mass_kg))
    doc = {"schema": "e505c-recovery-montecarlo-1", "dm_t": args.dm,
           "sep_mass_kg": sep_mass, "mode": args.mode, "seed": args.seed,
           "n": args.n, "uncertainty": unc_note, "wall_s": round(time.time() - t0, 1),
           "plan_options_recipe": jsonable(recipe), "summary": summary,
           "samples": records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    s, res = summary, summary["reserve"]
    print(f"   landed {s['landed']}/{s['n']} (CI95 {s['success_ci95']}), "
          f"engine-on handoffs {s['engine_on_handoffs']}\n   reasons {s['reasons']}\n"
          f"   reserve: gate {res['gate_kg']:.0f} kg, q{args.quantile:g} touchdown "
          f"{res.get('touchdown_propellant_at_quantile_kg')}, shortfall "
          f"{res.get('shortfall_kg')}, extra sep mass ~{res.get('extra_separation_mass_estimate_kg')}"
          f"\n   -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
