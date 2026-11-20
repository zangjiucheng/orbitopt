"""Benchmark: E6-05 guidance applicability extension (task #292).

Flies the E6-03 declared matrix (:mod:`...guidance_matrix`) one more time --
this time on the JSBSim channel -- and records, per cell, the native/JSBSim
difference against the committed E6-03 study: the terminal-orbit delta, the
propellant and cutoff-time parity, and the FL-403 attitude/actuator band.
Each JSBSim leg inherits the frozen separation state of the cell's own native
rigid-body stack run, so the two upper-stage legs start from the same handoff;
the stack leg itself stays E6-02's measured quantity.  See the module
docstring of :mod:`orbitopt.flightlab.sixdof.jsbsim_matrix` for what that
channel does and does not model.

Run (regenerates the committed JSON; every cell is one native stack leg plus
one JSBSim burn, so use ``--jobs`` for the whole matrix -- 17 cells take
roughly 40 min at 4 workers):

    PYTHONPATH=src python benchmarks/bench_e6_05_jsbsim_matrix.py \
        -o docs/benchmarks/e6_05_jsbsim_matrix.json --jobs 4

``--cells a220-i28.5-V1-A1,a400-i28.5-V1-A1`` flies a subset (the regression
test does exactly that, in process).  Illustrative physics, not flight
certification.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from orbitopt.flightlab.sixdof.guidance_matrix import GUIDANCE_MATRIX  # noqa: E402
from orbitopt.flightlab.sixdof.jsbsim_matrix import (  # noqa: E402
    run_jsbsim_benchmark)


def _json_safe(x):
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [_json_safe(v) for v in x]
    if hasattr(x, "item") and hasattr(x, "dtype"):  # numpy scalar
        return _json_safe(x.item())
    if isinstance(x, float):
        if not math.isfinite(x):  # a hyperbolic terminal orbit: visible, not faked
            return None if math.isnan(x) else f"{x:+.0f}"
        return round(x, 6) if abs(x) < 1e15 else f"{x:.6e}"
    return x


def main(out_path: Path | None, subset: tuple[str, ...] | None, jobs: int) -> None:
    cells = GUIDANCE_MATRIX if not subset else tuple(
        c for c in GUIDANCE_MATRIX if c.cell_id in set(subset))
    t0 = time.time()
    payload = run_jsbsim_benchmark(cells, jobs=jobs,
                                   progress=lambda r: print(
                                       f"[{r['cell_id']}] jsbsim "
                                       f"{'ins ' if r['jsbsim']['inserted'] else 'red '}"
                                       f"d_peri {r['comparison']['d_periapsis_jsbsim_minus_native_m'] / 1e3:+8.2f}km "
                                       f"d_apo {r['comparison']['d_apoapsis_jsbsim_minus_native_m'] / 1e3:+9.2f}km "
                                       f"att {r['jsbsim']['attitude_err_max_deg']:5.2f}deg "
                                       f"({'ok' if r['all_checks_passed'] else 'CHECK'}", flush=True))
    payload["command"] = " ".join(sys.argv)
    payload["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload["wall_clock_s"] = round(time.time() - t0 + payload["wall_clock_s"], 1)
    text = json.dumps(_json_safe(payload), indent=1)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n")
        print(f"wrote {out_path}")
    bad = [c["id"] for c in payload["checks"] if not c["passed"]]
    print(f"[checks] {len(payload['checks'])} checks, "
          f"{len(bad)} failed: {bad or 'none'}")
    print(f"[verdicts] insertion verdict agrees on "
          f"{payload['meta']['verdict_agreement']} of the flown cells")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--cells", type=str, default=None,
                    help="comma-separated cell ids (default: the whole matrix)")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    main(args.out, tuple(args.cells.split(",")) if args.cells else None, args.jobs)
