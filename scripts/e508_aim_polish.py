"""Nelder-Mead polish of the corridor aim.

The recovery-script search (``e508_landing_search.py``) scores a hierarchy of gates; from a
point that already reaches the corridor but is 800 m off the pad the differential-evolution
population cannot find the narrow valley that lowers the miss. This minimises a SMOOTH
objective on the quantities the corridor report carries (miss, speed, tilt, entry-burn cutoff
height) with a local simplex, from several starting cut speeds in parallel. The result is a
starting point for the full search, not a result by itself.

    python scripts/e508_aim_polish.py --x0 '[...]' --ascent-dir D --vehicle V --out out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import get_context
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

_G: dict = {}


def _init(args_dict):
    import e508_landing_search as m
    from orbitopt.flightlab.recovery.boostback import DEFAULT_CORRIDOR

    a = argparse.Namespace(**args_dict)
    if args_dict.get("eop") == "e507":
        from orbitopt.reference_eop import freeze_reference, load_snapshot

        _G["eop"] = freeze_reference(load_snapshot(
            REPO / "tests/data/mission/reference/e5-07-physics-qualification/earth_orientation.json"))
        _G["eop"].__enter__()                    # for the life of this worker
    c = m._context(a)
    m._CTX.update(c, target_corridor=DEFAULT_CORRIDOR, sites=None, margin="base")
    _G.update(c=c, corr=DEFAULT_CORRIDOR, vt=float(args_dict.get("speed_target", 330.0)))


def _f(x):
    import numpy as np

    from orbitopt.flightlab.recovery.pipeline import fly_recovery

    c, corr = _G["c"], _G["corr"]
    try:
        out = fly_recovery(c["run"], c["target"], c["vehicle"], sep_state=c["sep"],
                           plan_options={**c["recipe"], "initial_guess": [float(v) for v in x], "search": "keep"})
    except Exception:  # noqa: BLE001
        return 1.0e7
    r = getattr(out.plan, "corridor_report", None)
    if r is None or not r.reached:
        return 1.0e6
    h0 = corr.h0_m
    cut = r.entry_cutoff_altitude_m if r.entry_cutoff_altitude_m is not None else 0.0
    return (float(r.miss_m) / 10.0 + 10.0 * max(0.0, float(r.speed_m_s) - _G["vt"])
            + 5.0 * max(0.0, h0 + 800.0 - float(cut)) + 20.0 * max(0.0, float(r.tilt_deg) - 5.0)
            + 2.0 * max(0.0, 100.0 - (float(r.propellant_available_kg) - float(r.landing_need_kg)) / 1e3))


def _run(job):
    from scipy.optimize import minimize

    x0, scale, maxfev = job
    res = minimize(_f, x0, method="Nelder-Mead",
                   options=dict(maxfev=maxfev, adaptive=True, xatol=1e-3, fatol=1e-3,
                                initial_simplex=_simplex(x0, scale)))
    return [float(v) for v in res.x], float(res.fun), int(res.nfev)


def _simplex(x0, scale):
    import numpy as np

    x0 = np.asarray(x0, dtype=float)
    pts = [x0]
    for i in range(len(x0)):
        p = x0.copy()
        p[i] += scale * (abs(x0[i]) if abs(x0[i]) > 1e-6 else 1.0)
        pts.append(p)
    return np.array(pts)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--x0", required=True)
    p.add_argument("--ascent-dir", required=True)
    p.add_argument("--vehicle", required=True)
    p.add_argument("--cuts", type=float, nargs="+", default=[330.0, 300.0, 270.0, 240.0])
    p.add_argument("--scales", type=float, nargs="+", default=[0.05, 0.15])
    p.add_argument("--maxfev", type=int, default=260)
    p.add_argument("--speed-target", type=float, default=330.0,
                   help="corridor speed [m/s] below which the objective is satisfied (a landing from a "
                        "lighter booster needs a lower arrival speed: its throttle floor is a higher TWR)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--eop", choices=("live", "e507"), default="live")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    args_dict = dict(recipe="search", entry_q_kpa=250.0, entry_g=12.0, site_km=0, cut_max=1500, landing_g=4.0,
                     sep_source="mission", x0=a.x0, ascent_dir=a.ascent_dir, vehicle=a.vehicle,
                     speed_target=a.speed_target, eop=a.eop)
    x0 = json.loads(a.x0)
    jobs = []
    for cut in a.cuts:
        for sc in a.scales:
            x = list(x0)
            x[6] = cut
            jobs.append((x, sc, a.maxfev))
    with get_context("fork").Pool(a.workers, initializer=_init, initargs=(args_dict,)) as pool:
        results = pool.map(_run, jobs)
    results.sort(key=lambda r: r[1])
    Path(a.out).write_text(json.dumps([{"x": x, "f": f, "nfev": n} for x, f, n in results], indent=1) + "\n")
    for x, f, n in results:
        print(f"f {f:9.3f} nfev {n}  x {[round(v, 2) for v in x]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
