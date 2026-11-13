"""Benchmark: F-07 mission outputs -- the same two-stage mission flown on
both backends, artefacts and all (task #232).

Flies the packaged ``f07_scenario`` (the F-06 two-stage scenario) twice --
once closed-loop 6-DOF, once as the 3-DOF point-mass twin -- writes the full
output bundle (run directories, ``handoff.json``, verification reports,
scenes, comparison report) into a scratch directory, and records the study
as the JSON the acceptance asks to be able to regenerate:

* per-backend acceptance verdicts and terminal bookkeeping;
* the comparison report's status and per-body ``6dof - 3dof`` metrics on the
  four F-05 channels (altitude, speed, flight-path angle, mass);
* the wall-clock cost of the whole bundle (two RK4 missions plus scene
  conversion through the FL-002 frame bridge).

``tests/test_flightlab_sixdof_mission_output.py`` checks the committed JSON
against the regression thresholds; see ``docs/flightlab.md`` (F-07) for the
interpretation. Real RK4 physics of the F-02/F-04 reference plant -- the
pair is a few minutes of single-core CPU:

    PYTHONPATH=src python benchmarks/bench_fl607_mission_outputs.py \
        -o docs/benchmarks/fl607_mission_outputs.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from orbitopt.flightlab.sixdof.mission_output import (  # noqa: E402
    BACKENDS, BACKEND_3DOF, BACKEND_6DOF, f07_scenario,
    mission_comparison_report, run_mission, write_mission_outputs,
)
from orbitopt.verify.report import default_report_path, write_report  # noqa: E402

BODIES = ("stack", "booster", "upper")


def _backend_block(mission) -> dict:
    study = mission.study
    s = study.scenario
    return {
        "checks": {c.id: bool(c.passed) for c in study.checks},
        "separation": {k: v for k, v in study.separation.items()
                       if isinstance(v, (int, float, str))},
        "runs": {body: {"terminal": run.terminal,
                        "t_end_s": float(run.samples["t_s"][-1]),
                        "mass_end_kg": float(run.samples["mass_kg"][-1])}
                 for body, run in (("stack", study.stack),
                                   ("booster", study.booster),
                                   ("upper", study.upper))},
        "handoff": {"t_s": study.booster_handoff.t_s,
                    "frame": study.booster_handoff.frame,
                    "mass_kg": study.booster_handoff.mass_kg,
                    "dry_mass_kg": study.booster_handoff.dry_mass_kg},
        "scenario": {"name": s.name, "separation_t_s": s.separation_t_s,
                     "ignition_delay_s": s.ignition.ignition_delay_s,
                     "t_end_s": s.t_end_s, "dt_s": s.dt_s,
                     "sample_every": s.sample_every},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out", type=Path,
                    default=REPO / "docs" / "benchmarks" / "fl607_mission_outputs.json")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--sample-every", type=int, default=20)
    ap.add_argument("--keep-runs", type=Path, default=None, metavar="DIR",
                    help="write the artefact bundle here instead of a scratch dir")
    args = ap.parse_args()

    scenario = f07_scenario(dt_s=args.dt, sample_every=args.sample_every)
    t0 = time.time()
    keep = args.keep_runs is not None
    out_dir = args.keep_runs or Path(tempfile.mkdtemp(prefix="fl607-"))
    missions = {}
    for backend in BACKENDS:
        missions[backend] = run_mission(scenario, backend=backend)
    report = mission_comparison_report(missions[BACKEND_3DOF],
                                       missions[BACKEND_6DOF])
    report_path = write_report(report, default_report_path(report, out_dir))
    for backend in BACKENDS:
        write_mission_outputs(missions[backend], out_dir / backend)
    wall_s = time.time() - t0

    payload = {
        "schema": "orbitopt-fl607-1",
        "task": "#232 (F-07, epic #216)",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "wall_clock_s": round(wall_s, 1),
        "backends": {b: _backend_block(missions[b]) for b in BACKENDS},
        "comparison": {
            "status": report["status"]["result"],
            "checks": report["inputs"]["checks"],
            "bodies": {
                body: {
                    "metrics": report["inputs"]["bodies"][body]["metrics"],
                    "t_span_s": report["inputs"]["bodies"][body]["t_span_s"],
                    "run_3dof": report["inputs"]["bodies"][body]["run_3dof"],
                    "run_6dof": report["inputs"]["bodies"][body]["run_6dof"],
                } for body in BODIES},
        },
        "artefacts": {
            "kept": keep,
            "run_dirs": [str(out_dir / b) for b in BACKENDS],
            "comparison": str(report_path),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"wrote {args.out} ({payload['comparison']['status']}, "
          f"{payload['wall_clock_s']} s wall)")


if __name__ == "__main__":
    main()
