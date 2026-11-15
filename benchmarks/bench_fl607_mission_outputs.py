"""Benchmark: F-07 mission outputs -- the same two-stage mission flown on
both backends, artefacts and all (task #232), with the E1-03 shared-guidance
upper stage (task #252).

Flies the packaged ``f07_scenario`` (the F-06 two-stage scenario, guided
upper stage per task #252) twice -- once closed-loop 6-DOF, once as the
3-DOF point-mass twin -- writes the full output bundle (run directories,
``handoff.json``, verification reports, scenes, comparison report) into a
scratch directory, and records the study as the JSON the acceptance asks to
be able to regenerate:

* per-backend acceptance verdicts and terminal bookkeeping;
* the comparison report's status and per-body ``6dof - 3dof`` metrics on the
  four F-05 channels (altitude, speed, flight-path angle, mass);
* the parking-orbit insertion grading of both backends against the declared
  tolerances (``upper_insertion``, task #252);
* the residual attribution (``residual_attribution``, task #252): the total
  upper-stage terminal difference is decomposed additively into an
  *integration* rung (the 3-DOF twin reflied at ``dt_s/2``), an *actuator*
  rung (the 6-DOF upper reflied through a near-ideal servo), a *guidance*
  rung (zero by construction -- both backends fly one guidance law) and the
  dynamics/model remainder;
* the wall-clock cost of the whole bundle (two RK4 missions, two attribution
  re-flies, plus scene conversion through the FL-002 frame bridge).

``tests/test_flightlab_sixdof_mission_output.py`` checks the committed JSON
against the regression thresholds; see ``docs/flightlab.md`` (F-07) for the
interpretation. Real RK4 physics of the F-02/F-04 reference plant -- the
pair is a few minutes of single-core CPU:

    PYTHONPATH=src python benchmarks/bench_fl607_mission_outputs.py \
        -o docs/benchmarks/fl607_mission_outputs.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from orbitopt.flightlab.sixdof.gravity_turn import compare_ascent  # noqa: E402
from orbitopt.flightlab.sixdof.mission_output import (  # noqa: E402
    BACKENDS, BACKEND_3DOF, BACKEND_6DOF, F07_UPPER_GIMBAL_LIMIT_RAD,
    f07_scenario, mission_comparison_report, run_mission, write_mission_outputs,
)
from orbitopt.flightlab.sixdof.tvc import TvcActuator  # noqa: E402
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
    ap.add_argument("--no-attribution", action="store_true",
                    help="skip the two extra attribution re-flies (integration "
                         "and actuator rungs; the block records 'skipped')")
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

    # E1-03 (task #252) residual attribution -- the acceptance asks that the
    # remaining 3-DOF/6-DOF upper-stage residual be *classified*, not just
    # shrunk. Additive first-order decomposition of the upper terminal
    # altitude difference (6-DOF minus 3-DOF):
    #   integration rung: the 3-DOF twin reflied at dt_s/2 -- what the coarse
    #       step of the reference model costs (3dof@dt minus 3dof@dt/2);
    #   actuator rung: the 6-DOF upper reflied through a near-ideal servo
    #       (same travel limit, 10x slew rate, 20x faster lag) -- what the
    #       F-04 tracker lag/slew costs (6dof@standard minus 6dof@ideal);
    #   guidance rung: zero BY CONSTRUCTION -- both backends fly the one
    #       UpperStageGuidanceSpec, so no guidance delta can exist;
    #   dynamics/model remainder: total - integration - actuator (rigid-body
    #       attitude realisation: what the point-mass assumption costs).
    s = missions[BACKEND_3DOF].study.scenario
    if args.no_attribution:
        attribution = {"skipped": True}
    else:
        m3_half = run_mission(dataclasses.replace(s, dt_s=s.dt_s / 2.0),
                              backend=BACKEND_3DOF)
        ideal_actuator = TvcActuator(limit_rad=F07_UPPER_GIMBAL_LIMIT_RAD,
                                     max_slew_rate_rad_s=1.0, tau_s=0.005)
        m6_ideal = run_mission(dataclasses.replace(s, upper_actuator=ideal_actuator),
                               backend=BACKEND_6DOF)
        total = compare_ascent(missions[BACKEND_3DOF].study.upper,
                               missions[BACKEND_6DOF].study.upper)
        integration = compare_ascent(m3_half.study.upper,
                                     missions[BACKEND_3DOF].study.upper)
        actuator = compare_ascent(m6_ideal.study.upper,
                                  missions[BACKEND_6DOF].study.upper)
        total_m = float(total.summary()["d_altitude_m_terminal_m"])
        integ_m = float(integration.summary()["d_altitude_m_terminal_m"])
        act_m = float(actuator.summary()["d_altitude_m_terminal_m"])
        attribution = {
            "method": "additive first-order decomposition of the upper-stage "
                      "terminal altitude difference (6dof - 3dof)",
            "total_d_altitude_m": total_m,
            "integration_d_altitude_m": integ_m,
            "actuator_d_altitude_m": act_m,
            "guidance_d_altitude_m": 0.0,
            "dynamics_model_remainder_d_altitude_m": total_m - integ_m - act_m,
            "signs": "integration = 3dof@dt - 3dof@dt/2; actuator = 6dof@standard"
                     " - 6dof@ideal; total = 6dof - 3dof; remainder is by "
                     "subtraction",
            "integration_dt_ref_s": s.dt_s / 2.0,
            "actuator_ideal": {"limit_rad": float(ideal_actuator.limit_rad),
                               "max_slew_rate_rad_s": float(ideal_actuator.max_slew_rate_rad_s),
                               "tau_s": float(ideal_actuator.tau_s)},
        }

    report_path = write_report(report, default_report_path(report, out_dir))
    for backend in BACKENDS:
        write_mission_outputs(missions[backend], out_dir / backend)
    wall_s = time.time() - t0

    spec = missions[BACKEND_6DOF].study.scenario.upper_guidance
    payload = {
        "schema": "orbitopt-fl607-1",
        "task": "#232 (F-07, epic #216); E1-03 upper-stage guidance #252 (epic #246)",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "wall_clock_s": round(wall_s, 1),
        "upper_guidance": {
            "law": "orbitopt.flightlab.guidance.upper_stage.UpperStageGuidance"
                   " (PEG-class velocity-to-be-gained steering, shared by both"
                   " backends)",
            "target_altitude_m": float(spec.target.altitude_m),
            "periapsis_tolerance_m": float(spec.target.periapsis_tolerance_m),
            "apoapsis_tolerance_m": float(spec.target.apoapsis_tolerance_m),
            "cutoff_periapsis_margin_m": float(spec.cutoff_periapsis_margin_m),
            "command_slew_rate_deg_s": float(spec.command_slew_rate_deg_s),
            "command_alpha_limit_deg": float(spec.command_alpha_limit_deg),
            "command_heading_limit_deg": float(spec.command_heading_limit_deg),
            "gravity_term_gain": float(spec.gravity_term_gain),
        },
        "backends": {b: _backend_block(missions[b]) for b in BACKENDS},
        "comparison": {
            "status": report["status"]["result"],
            "checks": report["inputs"]["checks"],
            "upper_insertion": report["inputs"]["upper_insertion"],
            "bodies": {
                body: {
                    "metrics": report["inputs"]["bodies"][body]["metrics"],
                    "t_span_s": report["inputs"]["bodies"][body]["t_span_s"],
                    "run_3dof": report["inputs"]["bodies"][body]["run_3dof"],
                    "run_6dof": report["inputs"]["bodies"][body]["run_6dof"],
                } for body in BODIES},
        },
        "residual_attribution": attribution,
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
