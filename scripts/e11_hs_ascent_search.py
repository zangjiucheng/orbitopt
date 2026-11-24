"""E11-06: restore the Starbase insertion band with the hot-staging timeline held fixed.

Decision vector: the seven post-staging pitch nodes, the ship burn end and the (constant) azimuth. The
pre-staging program, the staging timeline and every graded ascent limit (3.0 g, 55 kPa, periapsis >= 230 km,
apoapsis band, 30.5 deg +/- 1) are unchanged -- only what the hot staging made different is re-tuned.

    ORBITOPT_E508_VARIANT=starbase python scripts/e11_hs_ascent_search.py --out DIR --workers 40
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
import e508_ascent_variants as V  # noqa: E402
import e11_starbase_hot_staging as H  # noqa: E402

_CTX: dict = {}


def _with_s2_burn(doc: dict, burn2: float) -> dict:
    d = copy.deepcopy(doc)
    s2 = d["stages"][1]
    t2 = [[a, b] for a, b in s2["engine"]["thrust_vac_n"] if a < burn2]
    t2.append([burn2, s2["engine"]["thrust_vac_n"][-1][1]])
    s2["engine"]["thrust_vac_n"] = t2
    s2["engine"]["burn_end_s"] = burn2
    return d


def build(base: dict, x) -> dict:
    n = len(V.PROGRAM_NODES)
    return V.with_program(_with_s2_burn(base, float(x[n])), list(x[:n]), float(x[n + 1]))


def _obj(x) -> float:
    r = V.evaluate(build(_CTX["base"], x), _CTX["tmp"], f"w{os.getpid()}")
    return V.penalty(r) + V.monotone_penalty(list(x[:len(V.PROGRAM_NODES)]))


def main() -> int:
    from scipy.optimize import differential_evolution

    from orbitopt.flightlab.contract import load_target

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--maxiter", type=int, default=40)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--azimuth-range", type=float, default=6.0)
    a = ap.parse_args()
    e508 = V.m._load("e508_new_vehicle", REPO / "scripts/e508_new_vehicle.py")
    inc = float(load_target(e508.SCENARIO).target_orbit.inclination_deg)
    V._LIM["incl"] = (inc - 0.9, inc + 0.9)
    base = H.hot_staging_doc()
    tmp = Path(a.out); tmp.mkdir(parents=True, exist_ok=True)
    _CTX.update(base=base, tmp=tmp)
    base_el = [next(float(r["elevation_deg"]) for r in base["attitude_program"] if abs(float(r["t_s"]) - t) < 1e-6)
               for t in V.PROGRAM_NODES]
    az0 = float(base["attitude_program"][0]["azimuth_deg"])
    burn0 = float(base["stages"][1]["engine"]["burn_end_s"])
    bounds = [(max(0.5, e - 9.0), min(89.0, e + 9.0)) for e in base_el] + [(335.0, 400.0)] \
        + [(az0 - a.azimuth_range, az0 + a.azimuth_range)]
    x0 = np.array(base_el + [burn0, az0])
    hist: list = []

    def cb(intermediate_result):                       # new-style callback (scipy >= 1.12)
        ir = intermediate_result
        hist.append(float(ir.fun))
        print(f"  gen {len(hist):3d} best penalty {ir.fun:.4g}", flush=True)
        (tmp / "ckpt.json").write_text(json.dumps({"gen": len(hist), "penalty": float(ir.fun), "x": [float(v) for v in ir.x]}))
        if float(ir.fun) <= 1e-6:
            raise StopIteration

    with mp.get_context("fork").Pool(a.workers) as pool:
        res = differential_evolution(_obj, bounds, x0=x0, seed=a.seed, popsize=a.popsize, maxiter=a.maxiter,
                                     mutation=(0.5, 1.0), recombination=0.7, init="latinhypercube", tol=0.0,
                                     polish=False, updating="deferred", workers=pool.map, callback=cb)
    x = [float(v) for v in res.x]
    doc = build(base, x)
    r = V.evaluate(doc, tmp, "best")
    (tmp / "best_vehicle.yaml").write_text(yaml.safe_dump(doc, sort_keys=False, width=120))
    out = {"penalty": float(res.fun), "ok": r.get("ok"), "x": x, "checks": {k: v for k, v in r.get("checks", {}).items()}}
    (tmp / "best.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("penalty", "ok")}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
