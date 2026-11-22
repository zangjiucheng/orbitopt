"""E10-07 (task #304): full-recovery independent re-flight benchmark.

Flies the DC8 hot-staging ascent in OpenTsiolkovsky (E10-02 timeline), takes
the *actual* separation handoff, and runs
:func:`orbitopt.verify.full_recovery_reflight.independent_recovery_reflight`:
the main-model (FL-301/FL-302) recovery and an independently written
point-mass twin re-fly the same authored script from that handoff; every
acceptance quantity is compared, budget-split and band-checked. Writes the
full report to ``--out`` (``docs/benchmarks/e10_07_full_recovery_reflight.json``
when generating the committed copy). Runtime ~4 min (4 twin flights + OT).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from orbitopt.flightlab.adapters.opentsiolkovsky import run_ascent  # noqa: E402
from orbitopt.flightlab.contract import load_target  # noqa: E402
from orbitopt.verify.full_recovery_reflight import (  # noqa: E402
    independent_recovery_reflight,
)

SCENARIO = REPO / "src/orbitopt/data/flightlab/scenarios/leo_starship_class_rtls_e508_dc7.yaml"
DC8 = REPO / "src/orbitopt/data/flightlab/vehicles/starship_class_rtls_dc8.yaml"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="e1007-") as tmp:
        asc = Path(tmp) / "ascent"
        run = run_ascent(load_target(SCENARIO), DC8, None, asc, timeout_s=900)
        sep_t = next(e.t_s for e in run.events if e.kind == "separation")
        if abs(sep_t - 125.0) > 0.01:
            raise RuntimeError(f"E10-07: unexpected separation at {sep_t} s")
        rep = independent_recovery_reflight(asc, scenario_yaml=SCENARIO,
                                            vehicle_yaml=DC8)
    rep["runtime_s"] = round(time.time() - t0, 1)
    from orbitopt.verify.report import report_digest
    rep["digest"] = report_digest(rep)

    st = rep["status"]
    worst = {cid: (rep["comparison"]["deltas"][cid],
                   rep["comparison"]["bands"][cid]["unit"],
                   rep["comparison"]["bands"][cid]["limit"])
             for cid in st["checks"]}

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=1, default=float) + "\n",
                            encoding="utf-8")
        print(f"wrote {args.out}")
    print(f"status: {st['result']}   runtime: {rep['runtime_s']} s")
    for cid, (val, unit, lim) in sorted(worst.items()):
        print(f"  {'PASS' if st['checks'][cid] else 'FAIL'}  {cid:34s}"
              f" {val:10.2f} {unit:>4s}  (band {lim})")
    return 0 if st["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
