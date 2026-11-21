"""E5-08 landing-gate design search (follow-up to RECOVERY_GAP_E508.md).

The recorded E5-08 recovery leg reaches the landing corridor but arrives with
96.1 t of reserve against the inherited 100 t gate. RECOVERY_GAP_E508.md names
two declared design iterations that do NOT relax a graded constraint:

* **cut**   raise the planner's entry cut-off speed ceiling above 1500 m/s
            (``plan_options["entry_cut_speed_max_m_s"]``, a guidance-domain
            bound of the search), so the entry burn can stop earlier;
* **site**  a DC6 downrange landing site (a shorter boostback) via the
            ``landing_sites`` mapping of ``fly_recovery`` (``recovery.kind:
            downrange`` in the scenario).

Every graded number (touchdown 3 / 1 m/s, reserve 100 t, entry limits) is
left exactly as recorded. This script only SEARCHES and REPORTS; it seals
nothing and writes no reference data.

    python scripts/e508_landing_search.py probe
    python scripts/e508_landing_search.py search --cut-max 2500 --workers 24 \
        [--site-km 0] [--popsize 12 --maxiter 40 --seed 11] --out out.json
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _context(args):
    e508 = _load("e508_new_vehicle", REPO / "scripts/e508_new_vehicle.py")
    e505c = _load("e505c_recovery_search", REPO / "scripts/e505c_recovery_search.py")
    from orbitopt.flightlab.contract import load_target, read_run
    from orbitopt.flightlab.recovery.pipeline import separation_state_ecef

    asc = Path(getattr(args, "ascent_dir", None) or e508.REF_ROOT / "mission-optimized/launch/ascent")
    vehicle = Path(getattr(args, "vehicle", None) or e508.VEHICLE)
    run = read_run(asc)
    target = load_target(e508.SCENARIO)
    rs = e508.REF_ROOT / "recovery-script.json"       # absent for a design not yet recorded: --x0 is then required
    recorded = json.loads(rs.read_text()) if rs.is_file() else {"recipe": {"initial_guess": json.loads(args.x0)}}
    if getattr(args, "recipe", "gap") == "gap":          # exactly as recorded
        gap = dict(e508.GAP_RECIPE)
        gap["entry_q_max_pa"] = 1000.0 * float(args.entry_q_kpa)   # recorded: 400 kPa
        recipe = e508._recipe_defaults(**gap)
    else:                                                  # the search recipe: the
        # planner aims at the GRADED reserve (landing_reserve="target", 100 t + margin),
        # not the recorded 135 t planning reserve, so the objective rewards closing it
        recipe = e508._recipe_defaults(entry_q_max_pa=1000.0 * float(args.entry_q_kpa),
                                       entry_g_max=float(args.entry_g))
    recipe["landing"] = {**recipe["landing"], **e508.GAP_LANDING}
    if getattr(args, "landing_g", None):
        # the graded landing limit is LandingLimits.max_proper_g (5.0 g): the guidance cap
        # must sit below it or the planner rejects the corridor (landing_ok False)
        recipe["landing"]["max_thrust_accel_g"] = float(args.landing_g)
    if getattr(args, "sep_source", "mission") == "mission":
        # the state the full mission and the cross-environment recovery leg fly the
        # booster from: the canonical separation handoff (the stack's last sample
        # before separation, the booster branch's own mass)
        from orbitopt.mission.coupled_branches import booster_state_from_run, separation_handoff

        sep = booster_state_from_run(run, separation_handoff(run))
    else:
        sep = separation_state_ecef(run)
    return dict(e508=e508, e505c=e505c, run=run, target=target, vehicle=vehicle,
                sep=sep, recorded=recorded, recipe=recipe)


def _site_target(c, km: float):
    """DC6: a downrange landing pad ``km`` from the launch site along the
    great circle toward the booster's separation ground point (the direction
    the stage is already travelling), as ``(target, landing_sites)`` for
    ``fly_recovery`` -- ``recovery.kind`` becomes ``downrange``; every graded
    number is untouched. ``km == 0`` is the recorded RTLS pad."""
    import math

    from orbitopt.flightlab.contract import Recovery
    from orbitopt.flightlab.recovery.boostback import _geodetic_fast

    if not km:
        return c["target"], None
    ls = c["target"].launch_site
    lat0, lon0 = math.radians(ls.latitude_deg), math.radians(ls.longitude_deg)
    import numpy as np

    lat1, lon1, _h = (math.radians(v) for v in _geodetic_fast(
        np.asarray(c["sep"].position_m, dtype=float)[:3]))           # ECEF (separation_state_ecef)
    brg = math.atan2(math.sin(lon1 - lon0) * math.cos(lat1),
                     math.cos(lat0) * math.sin(lat1) - math.sin(lat0) * math.cos(lat1) * math.cos(lon1 - lon0))
    d = km / 6371.0
    lat2 = math.asin(math.sin(lat0) * math.cos(d) + math.cos(lat0) * math.sin(d) * math.cos(brg))
    lon2 = lon0 + math.atan2(math.sin(brg) * math.sin(d) * math.cos(lat0),
                             math.cos(d) - math.sin(lat0) * math.sin(lat2))
    sites = {"dc6-pad": {"latitude_deg": math.degrees(lat2), "longitude_deg": math.degrees(lon2),
                         "altitude_m": float(ls.altitude_wgs84_m)}}
    import dataclasses as dc
    tgt = dc.replace(c["target"], recovery=dc.replace(c["target"].recovery, kind="downrange",
                                                      target="dc6-pad"))
    return tgt, sites


def _readout(outcome) -> dict:
    plan = outcome.plan
    checks = {c.id: bool(c.passed) for c in outcome.checks}
    out = {"checks": checks,
           "landing_ok": checks.get("recovery-landing"),
           "reserve_ok": checks.get("recovery-propellant-reserve"),
           "reason_codes": list(getattr(plan, "reason_codes", ()) or ())}
    rep = getattr(plan, "corridor_report", None)
    if rep is not None:
        d = json.loads(json.dumps(dataclasses.asdict(rep), default=str))
        out["corridor"] = d
        avail, need = d.get("propellant_available_kg"), d.get("landing_need_kg")
        if avail is not None and need is not None:
            out["reserve_kg"] = avail - need
        out["miss_m"] = d.get("miss_m")
    return out


def cmd_probe(args) -> int:
    from orbitopt.flightlab.recovery.pipeline import fly_recovery

    c = _context(args)
    x = c["recorded"]["recipe"]["initial_guess"]
    opts = {**c["recipe"], "initial_guess": x, "search": "keep"}
    t0 = time.time()
    outcome = fly_recovery(c["run"], c["target"], c["vehicle"],
                           sep_state=c["sep"], plan_options=opts)
    wall = time.time() - t0
    r = _readout(outcome)
    print(json.dumps({"x": x, "wall_s": round(wall, 1), **{k: r[k] for k in
                      ("checks", "reserve_kg", "miss_m", "reason_codes") if k in r}}, indent=2))
    return 0


_CTX: dict = {}
#: search margins: "base" holds the gates (100 t reserve, corridor v <= 350 m/s, entry cutoff
#: above h0, landing g < 5.0); "robust" asks for room on every one of them, because a script
#: that sits ON a gate fails under any dispersion (pinned-script Monte Carlo: 7/36).
_MARGINS = {
    "base": dict(reserve_t=100.0, v_target=350.0, cut_hard=300.0, cut_soft=1500.0, g_clear=0.25),
    "mid": dict(reserve_t=112.0, v_target=320.0, cut_hard=500.0, cut_soft=1600.0, g_clear=0.3),
    "robust": dict(reserve_t=125.0, v_target=300.0, cut_hard=800.0, cut_soft=1800.0, g_clear=0.4),
}


def _landing_g_penalty(outcome, c) -> float:
    """Margin on the planner's own forecast of the landing (``landing_ok`` needs the
    peak proper g <= LandingLimits.max_proper_g = 5.0 and an exact touchdown): the
    corridor report does not carry them, so re-forecast from the corridor state."""
    import numpy as np

    from orbitopt.flightlab.recovery.boostback import RecoveryTarget, _geodetic_fast
    from orbitopt.flightlab.recovery.landing import DEFAULT_LANDING_LIMITS, TerminalGuidance, \
        predict_closed_loop_landing
    from orbitopt.flightlab.recovery.pipeline import landing_reserve_booster
    from orbitopt.flightlab.recovery.propagator import BoosterModel

    enu = outcome.plan.landing_burn_initial_state_enu
    if enu is None:
        return 1.0e3
    recipe = c["recipe"]
    booster = landing_reserve_booster(BoosterModel.load(c["vehicle"]), c["target"],
                                      recipe["landing_reserve"])
    pad = np.asarray(RecoveryTarget.from_scenario(c["target"].to_dict()).position_ecef_m)
    g_opts = {k: v for k, v in recipe["landing"].items() if k not in ("mode", "mismatch")}
    pred = predict_closed_loop_landing(enu.position_enu_m, enu.velocity_enu_m, float(enu.mass_kg), booster,
                                       pad_altitude_m=float(_geodetic_fast(pad)[2]),
                                       guidance=TerminalGuidance(**g_opts))
    lim = DEFAULT_LANDING_LIMITS
    pen = 0.0 if pred.feasible else 500.0
    pen += 200.0 * max(0.0, float(pred.max_proper_g)
                       - (lim.max_proper_g - _MARGINS[_CTX.get("margin", "base")]["g_clear"]))
    pen += max(0.0, float(pred.touchdown_miss_m) - 0.5 * lim.position_tol_m)
    return pen


def _ensemble_penalty(x) -> float:
    """Robustness term: fly the PINNED script ``x`` through ``ens_k`` FL-304 dispersed
    samples (the same dispersion model and protocol as ``e508_new_vehicle.py mc --pinned``,
    around the canonical separation state, common random numbers from ``ens_seed``) and
    charge the share that does not land. The seed is disjoint from the hold-out seeds the
    robustness is reported on (5, 6, 7), so the objective is not tuned to its own exam."""
    import orbitopt.flightlab.recovery.montecarlo as mc

    c = _CTX
    k = int(c["ens_k"])
    res = mc.run_monte_carlo(c["target"], c["vehicle"], c["run"], seed=int(c["ens_seed"]), n=k, workers=1,
                             plan_options={**c["recipe"], "initial_guess": list(map(float, x)), "search": "keep"})
    landed = sum(1 for smp in res.samples if smp.get("outcome") == "landed")
    return float(c["ens_weight"]) * (k - landed) / k


def _score(x) -> float:
    """Penalty of one booster script ``x``: fly the recovery chain with the
    script pinned (``search="keep"``) and score what the landing gate grades.

    The corridor report carries the planner's closed-loop PREDICTION of the
    landing from the corridor state (``landing_touchdown_vertical_m_s`` /
    ``_lateral_m_s``): the gate wants <= 3 / 1 m/s. So the score is
    a steep (100 per m/s) penalty beyond a margined 2.5 / 0.8 m/s, the corridor's own limits (speed <= v_max at h0, tilt <= max_tilt), plus the shortfall of the corridor reserve
    against the 100 t gate [t] and any overshoot of the entry-q planning limit
    [kPa], plus the corridor miss [m]/100; a script that never reaches it costs ~1000 + miss/100. Truth is
    graded afterwards by :func:`_final_flight` (both recovery checks)."""
    from orbitopt.flightlab.recovery.pipeline import fly_recovery

    c = _CTX
    try:
        outcome = fly_recovery(c["run"], c["target"], c["vehicle"], sep_state=c["sep"],
                               landing_sites=c.get("sites"),
                               plan_options={**c["recipe"], "initial_guess": list(map(float, x)),
                                             "search": "keep"})
    except Exception:  # noqa: BLE001 -- an infeasible script is just a bad point
        return 1.0e7
    rep = getattr(outcome.plan, "corridor_report", None)
    if rep is None or not rep.reached:
        miss = getattr(rep, "miss_m", None)
        return 1.0e6 + (float(miss) / 100.0 if miss is not None else 9.0e5)   # always worse than any reached script
    vert = rep.landing_touchdown_vertical_m_s
    lat = rep.landing_touchdown_lateral_m_s
    # soft landing is the gate (3 / 1 m/s): 100 per m/s over a margined 2.5 / 0.8
    pen = (100.0 * max(0.0, abs(float(vert)) - 2.5) + 100.0 * max(0.0, abs(float(lat)) - 0.8)
           if vert is not None and lat is not None else 5.0e5)
    reserve_t = (float(rep.propellant_available_kg) - float(rep.landing_need_kg)) / 1e3
    M = _MARGINS[_CTX.get("margin", "base")]
    pen += max(0.0, M["reserve_t"] - reserve_t)
    # the entry burn must be OFF before the corridor crossing, or the handoff is engine-on and
    # the planner refuses the landing (handoff_burn_overlap_s > 0 -> CORRIDOR_MISSED): demand
    # the cutoff above h0 with room, and prefer the planner's own clearance
    cut_alt = rep.entry_cutoff_altitude_m
    h0 = float(c["target_corridor"].h0_m)
    pen += (0.5 * max(0.0, h0 + M["cut_hard"] - float(cut_alt))
            + 0.002 * max(0.0, h0 + M["cut_soft"] - float(cut_alt))
            if cut_alt is not None else 1.0e3)
    # the CORRIDOR itself (boostback.DEFAULT_CORRIDOR): <= v_max at h0 and within
    # max_tilt of vertical -- outside it the planner refuses to fly the landing
    # burn (CORRIDOR_MISSED), whatever the closed-loop prediction says
    corr = c["target_corridor"]
    pen += 10.0 * max(0.0, float(rep.speed_m_s) - min(corr.v_max_m_s, M["v_target"]))
    pen += 100.0 * max(0.0, float(rep.tilt_deg) - corr.max_tilt_deg)
    if (float(rep.speed_m_s) <= c["target_corridor"].v_max_m_s and cut_alt is not None
            and float(cut_alt) >= h0 + 300.0):
        pen += _landing_g_penalty(outcome, c)
    qlim = float(c["recipe"]["entry_q_max_pa"])
    pen += max(0.0, float(rep.entry_max_q_pa) - qlim) / 1e3
    pen += float(rep.miss_m) * float(_CTX.get("miss_w", 0.01))   # default: 424 m -> 4, 8 km (the referee) -> 80
    if _CTX.get("ens_k") and pen < _CTX["ens_gate"]:
        pen += _ensemble_penalty(x)      # only worth its cost once the nominal case is (nearly) closed
    return pen


def _final_flight(x) -> dict:
    from orbitopt.flightlab.recovery.pipeline import fly_recovery

    c = _CTX
    outcome = fly_recovery(c["run"], c["target"], c["vehicle"], sep_state=c["sep"],
                           landing_sites=c.get("sites"),
                           plan_options={**c["recipe"], "initial_guess": list(map(float, x)),
                                         "search": "keep"})
    r = _readout(outcome)
    try:
        r["touchdown"] = c["e505c"].touchdown_readout(outcome)
    except Exception as exc:  # noqa: BLE001
        r["touchdown"] = f"unavailable: {exc}"
    return r


def _enter_eop(args):
    """``--eop e507``: run under the frozen Earth-orientation snapshot of the E5-07 epoch --
    the one a cross-environment plan hands to every stage -- so what the search closes is what
    the sealed cross-environment evidence flies. Entered for the life of the process (forked
    workers inherit it)."""
    if getattr(args, "eop", "live") != "e507":
        return None
    from contextlib import ExitStack

    from orbitopt.reference_eop import freeze_reference, load_snapshot

    stack = ExitStack()
    stack.enter_context(freeze_reference(load_snapshot(
        REPO / "tests/data/mission/reference/e5-07-physics-qualification/earth_orientation.json")))
    return stack


def cmd_search(args) -> int:
    import multiprocessing as mp
    _eop_stack = _enter_eop(args)

    import numpy as np
    from scipy.optimize import differential_evolution

    from orbitopt.flightlab.recovery.propagator import BoosterModel

    c = _context(args)
    recipe = dict(c["recipe"])
    recipe["entry_cut_speed_max_m_s"] = float(args.cut_max)
    target, sites = _site_target(c, args.site_km)
    from orbitopt.flightlab.recovery.boostback import DEFAULT_CORRIDOR

    _CTX.update(c, recipe=recipe, target=target, sites=sites, target_corridor=DEFAULT_CORRIDOR,
                margin=getattr(args, "margin", "base"), miss_w=float(getattr(args, "miss_weight", 0.01)),
                ens_k=int(getattr(args, "ensemble", 0) or 0),
                ens_seed=int(getattr(args, "ens_seed", 101)), ens_weight=float(getattr(args, "ens_weight", 40.0)),
                ens_gate=float(getattr(args, "ens_gate", 3.0)))
    if _CTX["ens_k"]:
        # the MC disperses in the ECEF ENU basis of the state it is given: hand it the canonical
        # separation handoff converted to ECEF (the state the full mission flies from)
        import orbitopt.flightlab.recovery.montecarlo as _mc
        from dataclasses import replace as _replace

        from orbitopt.flightlab.recovery.pipeline import _to_ecef_state

        def _sep_ecef(_run, _sep=c["sep"]):
            if _sep.frame == "ECEF":
                return _sep
            r_e, v_e = _to_ecef_state(_sep)
            return _replace(_sep, frame="ECEF", position_m=tuple(float(v) for v in r_e),
                            velocity_m_s=tuple(float(v) for v in v_e))

        _mc.separation_state_ecef = _sep_ecef
    booster = BoosterModel.load(c["vehicle"])
    bounds = c["e505c"].bounds_for(recipe, booster.engine.throttle_min)
    bounds[6] = (60.0, float(args.cut_max))           # the cut-speed ceiling is the lever
    lo, hi = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
    x0 = np.clip(np.array(json.loads(args.x0) if args.x0 else c["recorded"]["recipe"]["initial_guess"],
                          dtype=float), lo, hi)
    if args.x0 and args.radius:      # warm start: search a box around the given script
        span = (hi - lo) * float(args.radius)
        lo, hi = np.maximum(lo, x0 - span), np.minimum(hi, x0 + span)
        bounds = [(float(a), float(b)) for a, b in zip(lo, hi)]
    history: list[dict] = []
    t0 = time.time()

    def cb(intermediate_result):  # the parameter NAME selects scipy's OptimizeResult form
        fun = float(intermediate_result.fun)
        history.append({"gen": len(history) + 1, "best_penalty": fun,
                        "wall_s": round(time.time() - t0, 1)})
        print(f"  gen {len(history):3d}  best {fun:.4g}  {time.time() - t0:.0f}s", flush=True)

    print(f"start script penalty {_score(x0):.4g}", flush=True)
    with mp.get_context("fork").Pool(args.workers) as pool:
        res = differential_evolution(_score, bounds, x0=x0, seed=args.seed, popsize=args.popsize,
                                     maxiter=args.maxiter, mutation=(0.5, 1.0), recombination=0.7,
                                     init="latinhypercube", tol=0.0, polish=False,
                                     updating="deferred", workers=pool.map, callback=cb)
    best = [float(v) for v in res.x]
    r = _final_flight(best)
    doc = {"lever": {"cut_max_m_s": args.cut_max, "site_km": args.site_km,
                     "recipe": args.recipe, "entry_q_kpa": args.entry_q_kpa},
           "search": {"seed": args.seed, "popsize": args.popsize, "maxiter": args.maxiter,
                      "workers": args.workers, "nfev": int(res.nfev),
                      "wall_s": round(time.time() - t0, 1)},
           "best_penalty": float(res.fun), "best_x": best, "history": history, **r}
    Path(args.out).write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({k: doc[k] for k in ("lever", "best_penalty", "best_x", "checks",
                                          "reserve_kg", "miss_m") if k in doc}, indent=2))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("probe", help="re-fly the recorded best script (search=keep)")
    pr.add_argument("--recipe", choices=("search", "gap"), default="gap")
    pr.add_argument("--entry-q-kpa", type=float, default=400.0)
    pr.add_argument("--entry-g", type=float, default=12.0)
    s = sub.add_parser("search", help="global DE search with a lever applied")
    s.add_argument("--cut-max", type=float, default=1500.0)
    s.add_argument("--margin", choices=tuple(_MARGINS), default="base")
    s.add_argument("--eop", choices=("live", "e507"), default="live",
                   help="e507: freeze the Earth orientation of the E5-07 epoch (as a cross-env plan does)")
    s.add_argument("--miss-weight", type=float, default=0.01,
                   help="penalty per metre of corridor miss (0.01: 8 km = 80; raise it when the landing "
                        "is only infeasible because the aim is off)")
    s.add_argument("--ensemble", type=int, default=0, help="dispersed samples per candidate (0 = off)")
    s.add_argument("--ens-seed", type=int, default=101)
    s.add_argument("--ens-weight", type=float, default=40.0)
    s.add_argument("--ens-gate", type=float, default=3.0, help="nominal penalty below which the ensemble runs")
    s.add_argument("--sep-source", choices=("mission", "ecef"), default="mission",
                   help="mission: the canonical separation handoff the full mission flies from (default)")
    s.add_argument("--landing-g", type=float, default=None,
                   help="landing guidance thrust-accel cap [g] (recorded gap recipe: 5.5, above the 5.0 g limit)")
    s.add_argument("--x0", default=None, help="JSON script to start from (warm start)")
    s.add_argument("--radius", type=float, default=0.0,
                   help="with --x0: search +/- radius x (global bound width) around it")
    s.add_argument("--ascent-dir", default=None, help="ascent run (default: the sealed E5-08 one)")
    s.add_argument("--vehicle", default=None, help="vehicle YAML (default: starship_class_rtls_derived)")
    s.add_argument("--recipe", choices=("search", "gap"), default="gap")
    s.add_argument("--entry-q-kpa", type=float, default=400.0,
                   help="entry-burn planning limit; the vehicle file declares 250, the "
                        "recorded gap flew 400")
    s.add_argument("--entry-g", type=float, default=12.0)
    s.add_argument("--site-km", type=float, default=0.0)
    s.add_argument("--workers", type=int, default=16)
    s.add_argument("--popsize", type=int, default=12)
    s.add_argument("--maxiter", type=int, default=40)
    s.add_argument("--seed", type=int, default=11)
    s.add_argument("--out", required=True)
    a = p.parse_args(argv)
    return cmd_probe(a) if a.cmd == "probe" else cmd_search(a)


if __name__ == "__main__":
    sys.exit(main())
