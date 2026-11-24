"""Local polish (Nelder-Mead on the scaled decision vector) of a continuous-burn search result."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "benchmarks"))
import e11_continuous_burn_search as C  # noqa: E402
import e11_unpowered_search as U  # noqa: E402


def main() -> int:
    from scipy.optimize import minimize

    from orbitopt.flightlab.contract import load_target
    from orbitopt.reference_eop import freeze_reference, load_snapshot

    d = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2]); iters = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    lo = np.array([b[0] for b in C.BOUNDS]); hi = np.array([b[1] for b in C.BOUNDS])
    with freeze_reference(load_snapshot(U.B.REF / "earth_orientation.json")):
        U._init(d["fins"], float(d.get("vscale", 1.0)))
        U._CTX["recipe"] = {**C.RECIPE, "dt_opt": 0.5, "dt_final": 0.25}
        U._CTX["target"] = load_target(U.B.SCEN)
        C.USE_TWIN = True                       # the twin must close too
        x0 = np.clip(np.array(d["x"], dtype=float), lo, hi)
        span = hi - lo
        best = {"p": 1e9}

        def f(z):
            x = np.clip(lo + z * span, lo, hi)
            p = C.objective(x)
            if p < best["p"]:
                best.update(p=p, x=x.copy()); print(f"best {p:.4f}", flush=True)
            return p

        z0 = (x0 - lo) / span
        for rnd in range(4):
            res = minimize(f, z0 if rnd == 0 else (best["x"] - lo) / span, method="Nelder-Mead",
                           options={"maxfev": iters, "xatol": 1e-6, "fatol": 1e-9,
                                    "initial_simplex": None if rnd == 0 else
                                    (best["x"] - lo) / span + np.vstack([np.zeros(len(lo)),
                                    np.eye(len(lo)) * (0.004 / (rnd + 1))])})
            if best["p"] <= -1.0:
                break
        pl = C._plan(best["x"])
        row = {**U.B._row("polished", pl), "fins": d["fins"], "vscale": d.get("vscale", 1.0),
               "x": [float(v) for v in best["x"]], "penalty": float(best["p"]),
               "entry_cutoff_alt_m": pl.corridor_report.entry_cutoff_altitude_m,
               "gap_m": C._gap_m(pl.corridor_report), "ignitions_used": pl.ignitions_used}
        out.write_text(json.dumps(row, indent=1) + "\n")
        print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
