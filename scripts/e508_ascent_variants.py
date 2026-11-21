"""E5-08 design adjustment: an earlier stage-1 cutoff (MECO) with a longer
stage-2 burn.

An earlier MECO leaves more propellant in the booster and a lower separation
energy for the recovery leg; the ship's spare propellant pays for it. Every
graded constraint (max-q, 3.0 g, insertion band, inclination, periapsis) is
unchanged: a variant is only useful if the ascent still passes ALL of them.
Variants are written to a scratch directory; no repository file is edited.

    python scripts/e508_ascent_variants.py scan --dt1 0 3 6 --out /tmp/e508-var
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import e508_landing_search as m  # noqa: E402  (loads the e508 module helpers)


def _interp(table, t):
    pts = [(float(a), float(b)) for a, b in table]
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if t0 <= t <= t1:
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return pts[-1][1]


def variant_doc(base: dict, dt1: float, burn2: float, sep_gap: float = 3.0) -> dict:
    """Stage-1 cutoff ``dt1`` s earlier, separation ``sep_gap`` s after it
    (the base's hot-staging gap), stage-2 burn to ``burn2`` s."""
    d = copy.deepcopy(base)
    s1, s2 = d["stages"][0], d["stages"][1]
    end1 = float(s1["engine"]["burn_end_s"]) - dt1
    tab = [[a, b] for a, b in s1["engine"]["thrust_vac_n"] if a < end1]
    tab.append([end1, _interp(s1["engine"]["thrust_vac_n"], end1)])
    s1["engine"]["thrust_vac_n"] = tab
    s1["engine"]["burn_end_s"] = end1
    s1["separation_time_s"] = end1 + sep_gap
    t2 = [[a, b] for a, b in s2["engine"]["thrust_vac_n"] if a < burn2]
    t2.append([burn2, s2["engine"]["thrust_vac_n"][-1][1]])
    s2["engine"]["thrust_vac_n"] = t2
    s2["engine"]["burn_end_s"] = burn2
    return d


PROGRAM_NODES = (155.0, 200.0, 260.0, 320.0, 380.0, 430.0, 445.0)   # post-MECO elevation nodes [s]


def with_program(doc: dict, elev: list[float]) -> dict:
    """``doc`` with the post-MECO elevation nodes replaced (azimuth, pre-MECO
    program and the final hold untouched)."""
    d = copy.deepcopy(doc)
    prog = d["attitude_program"]
    for t, el in zip(PROGRAM_NODES, elev):
        for row in prog:
            if abs(float(row["t_s"]) - t) < 1e-6:
                row["elevation_deg"] = float(el)
    return d


_LIM = dict(peri_km=232.0, apo_km=300.0, incl=(50.7, 52.5), g=2.95, q_pa=54000.0)


def monotone_penalty(elev: list[float], before: float = 44.0, after: float = 2.7) -> float:
    """A gravity-turn-style post-MECO program never pitches back UP: the elevation
    nodes (between the inherited 44 deg at 141 s and the inherited 2.7 deg hold) must
    be non-increasing. A search with freedom to bump the program (it once put 11.8 deg
    between 5.8 and 2.5 deg) reaches the insertion window by a trick that two
    independent models then disagree about; this keeps the program physical."""
    seq = [before] + list(elev) + [after]
    return 50.0 * sum(max(0.0, b - a) for a, b in zip(seq, seq[1:]))


def penalty(r: dict) -> float:
    """Hinge penalty of an evaluated variant against the INHERITED limits
    (periapsis >= 230 km, apoapsis within 250 +/- 60 km, inclination 51.6 +/- 1,
    <= 3.0 g, max-q <= 55 kPa) with a small margin; 0 means every graded
    ascent constraint holds."""
    ck = r.get("checks")
    if not ck:
        return 1.0e6
    peri = ck["insertion-periapsis-above-minimum"][1] / 1e3
    apo = ck["insertion-apoapsis-within-tolerance"][1] / 1e3
    incl = ck["insertion-inclination-within-tolerance"][1]
    g = ck["max-accel-within-limit"][1]
    q = ck["max-q-within-limit"][1]
    pen = max(0.0, _LIM["peri_km"] - peri) + max(0.0, apo - _LIM["apo_km"]) * 0.5
    pen += 20.0 * (max(0.0, _LIM["incl"][0] - incl) + max(0.0, incl - _LIM["incl"][1]))
    pen += 100.0 * max(0.0, g - _LIM["g"]) + max(0.0, q - _LIM["q_pa"]) / 1e3
    pen += 50.0 * sum(1 for k in ("mass-non-negative", "mass-continuity-stage2") if not ck[k][0])
    return pen


def _obj(x):
    c = _OBJ
    elev, burn2 = list(x[:-1]), float(x[-1])
    doc = with_program(variant_doc(c["base"], c["dt1"], burn2), elev)
    import os
    r = evaluate(doc, c["tmp"], f"w{os.getpid()}")
    return penalty(r) + monotone_penalty(elev)


_OBJ: dict = {}


def cmd_search(a) -> int:
    import multiprocessing as mp

    import numpy as np
    import yaml
    from scipy.optimize import differential_evolution

    e508 = m._load("e508_new_vehicle", REPO / "scripts/e508_new_vehicle.py")
    base = yaml.safe_load(e508.VEHICLE.read_text())
    tmp = Path(a.out)
    tmp.mkdir(parents=True, exist_ok=True)
    _OBJ.update(base=base, dt1=a.dt1[0], tmp=tmp)
    base_el = [next(float(r["elevation_deg"]) for r in base["attitude_program"]
                    if abs(float(r["t_s"]) - t) < 1e-6) for t in PROGRAM_NODES]
    bounds = [(max(0.5, e - 9.0), min(89.0, e + 9.0)) for e in base_el] + [(370.0, 400.0)]
    x0 = np.array(base_el + [378.0 + 0.0])
    history = []

    def cb(intermediate_result):
        history.append(float(intermediate_result.fun))
        print(f"  gen {len(history):3d} best penalty {intermediate_result.fun:.4g}", flush=True)

    with mp.get_context("fork").Pool(a.workers) as pool:
        res = differential_evolution(_obj, bounds, x0=x0, seed=a.seed, popsize=a.popsize,
                                     maxiter=a.maxiter, mutation=(0.5, 1.0), recombination=0.7,
                                     init="latinhypercube", tol=0.0, polish=False,
                                     updating="deferred", workers=pool.map, callback=cb)
    x = [float(v) for v in res.x]
    doc = with_program(variant_doc(base, a.dt1[0], x[-1]), x[:-1])
    r = evaluate(doc, tmp, "best")
    (tmp / "best_vehicle.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    out = {"dt1": a.dt1[0], "penalty": float(res.fun), "x": x, "ok": r.get("ok"),
           "checks": {k: v for k, v in r.get("checks", {}).items()}}
    (tmp / "best.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("dt1", "penalty", "ok", "x")}, indent=1, default=str))
    return 0


def evaluate(doc: dict, tmp: Path, tag: str) -> dict:
    import yaml

    from orbitopt.flightlab.adapters.opentsiolkovsky import run_ascent
    from orbitopt.flightlab.contract import load_target, read_run

    e508 = m._load("e508_new_vehicle", REPO / "scripts/e508_new_vehicle.py")
    vpath = tmp / f"{tag}.yaml"
    vpath.write_text(yaml.safe_dump(doc, sort_keys=False))
    out = tmp / tag
    shutil.rmtree(out, ignore_errors=True)
    try:
        run_ascent(load_target(e508.SCENARIO), vpath, None, out, timeout_s=300.0)
        run = read_run(out)
    except Exception as exc:  # noqa: BLE001
        return {"tag": tag, "error": str(exc)[:200]}
    ck = {c.id: (bool(c.passed), c.value) for c in run.verification.checks
          if not c.id.startswith("recovery")}
    return {"tag": tag, "ok": all(v[0] for v in ck.values()), "checks": ck, "dir": str(out)}


def main(argv=None) -> int:
    import yaml

    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["scan", "search"])
    p.add_argument("--dt1", type=float, nargs="+", default=[0, 3, 6, 9])
    p.add_argument("--burn2", type=float, nargs="+", default=[378, 384, 390, 396, 402, 410, 420])
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--popsize", type=int, default=12)
    p.add_argument("--maxiter", type=int, default=40)
    p.add_argument("--seed", type=int, default=3)
    a = p.parse_args(argv)
    if a.cmd == "search":
        return cmd_search(a)
    e508 = m._load("e508_new_vehicle", REPO / "scripts/e508_new_vehicle.py")
    base = yaml.safe_load(e508.VEHICLE.read_text())
    tmp = Path(a.out)
    tmp.mkdir(parents=True, exist_ok=True)
    rows = []
    for dt1 in a.dt1:
        for b2 in a.burn2:
            r = evaluate(variant_doc(base, dt1, b2), tmp, f"dt{dt1:g}_b{b2:g}")
            ck = r.get("checks", {})
            fails = [k for k, v in ck.items() if not v[0]]
            print(f"dt1 {dt1:4g} burn2 {b2:5g}  ok={r.get('ok')}  fails={fails or '-'}  "
                  f"peri {ck.get('insertion-periapsis-above-minimum', (0, 0))[1]:.0f}  "
                  f"apo {ck.get('insertion-apoapsis-within-tolerance', (0, 0))[1]:.0f}  "
                  f"g {ck.get('max-accel-within-limit', (0, 0))[1]:.2f}  "
                  f"incl {ck.get('insertion-inclination-within-tolerance', (0, 0))[1]:.2f}"
                  if ck else f"dt1 {dt1:g} burn2 {b2:g}  {r.get('error')}", flush=True)
            rows.append(r)
    (tmp / "scan.json").write_text(json.dumps(rows, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
