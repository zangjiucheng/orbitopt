"""Benchmark: E6-03 guidance generalisation matrix (task #275).

Flies the declared 17-point matrix of
:mod:`orbitopt.flightlab.sixdof.guidance_matrix` -- 5 target altitudes
x 3 inclinations x 4 upper-stage vehicle configs x 5 actuator constraint sets
(the slice that covers every axis, including the ORIGINAL F-06 stage flown
open loop) -- on both native backends through the packaged F-07 mission
runner, and records per cell:

* the insertion grading of each backend against the cell's DECLARED target
  (altitude, inclination and the matrix-wide 25 km / 1 deg tolerances);
* the cross-backend difference: the graded terminal difference plus the four
  F-05 channel metrics of the upper body;
* the burn bookkeeping that makes a miss diagnosable (did the cutoff fire, how
  much propellant is left, how far the terminal speed sits from the target
  circular speed, the terminal flight-path angle, the ideal delta-v budget);
* the verdict, and for every missed cell the attributed cause -- vehicle
  capability, guidance, or actuator constraint.

Run (this is the command that regenerates the committed JSON; the matrix is
seventeen two-backend RK4 missions, roughly a quarter of an hour of single-core
CPU):

    PYTHONPATH=src python benchmarks/bench_e6_03_guidance_matrix.py \
        -o docs/benchmarks/e6_03_guidance_matrix.json

``tests/test_flightlab_e6_03_guidance_matrix.py`` checks the committed file
against the declared expectations; ``docs/flightlab.md`` (E6-03) records the
reading. Illustrative native-plant physics, not flight certification.
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

from orbitopt.flightlab.sixdof.guidance_matrix import (  # noqa: E402
    GUIDANCE_MATRIX, run_guidance_matrix)


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


def main(out_path: Path | None) -> None:
    t0 = time.time()
    payload = run_guidance_matrix(GUIDANCE_MATRIX)
    payload["command"] = " ".join(sys.argv)
    payload["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload["wall_clock_s"] = round(time.time() - t0, 1)
    text = json.dumps(_json_safe(payload), indent=1)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n")
        print(f"wrote {out_path}")

    cov = payload["coverage"]
    print(f"[coverage] cells {cov['cells']}, altitudes "
          f"{[f'{a / 1e3:.0f}km' for a in cov['target_altitudes_m']]}, "
          f"inclinations {cov['inclinations_deg']}, vehicles "
          f"{len(cov['vehicle_ids'])}, actuator sets {len(cov['actuator_ids'])}")
    for rec in payload["cells"]:
        b = rec["backends"]
        reason = ",".join(r["code"] for r in rec["reasons"]) or "-"
        print(f"[{rec['cell_id']}] {rec['status']:4s} (declared {rec['expected']}"
              f"{', UNEXPECTED' if rec['unexpected'] else ''}) reason {reason:20s} "
              + " | ".join(
                  f"{k.replace('native-', '')}: d_peri "
                  f"{b[k]['insertion']['d_periapsis_m'] / 1e3:+8.2f}km d_apo "
                  f"{b[k]['insertion']['d_apoapsis_m'] / 1e3:+9.2f}km inc "
                  f"{b[k]['insertion']['inclination_deg']:6.2f}"
                  f"{'/ins' if b[k]['inserted'] else '/red'}"
                  for k in b))
    unexpected = [r["cell_id"] for r in payload["cells"] if r["unexpected"]]
    print(f"[verdicts] {sum(1 for r in payload['cells'] if r['status'] == 'pass')}"
          f" pass / {sum(1 for r in payload['cells'] if r['status'] == 'miss')} miss"
          f" of {len(payload['cells'])}; unexpected: {unexpected or 'none'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()
    main(args.out)
