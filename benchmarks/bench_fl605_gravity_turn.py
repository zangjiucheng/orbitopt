"""Benchmark: F-05 gravity-turn ascent -- 6-DOF vs. the 3-DOF point-mass
reference (task #230).

Flies the packaged ``f05_scenario`` (spike_stage booster, packaged
``spike_stage`` aero, vertical rise to t=10 s then a linear pitch-over to
18 deg by t=120 s) on all five rungs of the F-05 attribution ladder and
writes the acceptance's deliverables as JSON:

* the headline difference curves and metrics (6dof_full minus 3dof_aero) for
  the four channels -- altitude, Earth-relative speed, flight-path angle,
  mass;
* the per-rung (attribution) difference curves and metrics, isolating what
  point-mass aero, the rigid-body attitude loop, the TVC actuator lag and
  body-oriented aero each contribute;
* per-rung run records (terminal state, max-Q, apogee, burn-out, tracker
  diagnostics for the closed-loop rungs).

Real RK4 physics of the F-02/F-04 reference plant -- a study is a handful of
minutes of single-core CPU:

    PYTHONPATH=src python benchmarks/bench_fl605_gravity_turn.py \
        -o docs/benchmarks/fl605_gravity_turn_ascent.json

``tests/test_flightlab_sixdof_gravity_turn.py`` checks the committed JSON
against the regression thresholds and re-runs a short live ladder; see
``docs/flightlab.md`` (F-05) for the interpretation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from orbitopt.flightlab.sixdof.gravity_turn import (  # noqa: E402
    ATTRIBUTION_STEPS,
    DIFF_CHANNELS,
    RUNG_ORDER,
    ascent_channels,
    f05_scenario,
    run_f05_study,
)


def _run_record(run) -> dict:
    """Per-rung record, accepting a PointMassRun or a ControlledAscent."""
    ascent = run if hasattr(run, "run") else None
    r = run.run if ascent else run
    s = r.samples
    alt = s["altitude_m"]
    ch = ascent_channels(r)
    out = {
        "kind": "6dof_closed_loop" if ascent else "3dof_point_mass",
        "terminal": r.terminal,
        "t_end_s": float(s["t_s"][-1]),
        "burn_out_t_s": r.propellant_exhausted_t_s,
        "max_altitude_m": float(alt.max()),
        "max_altitude_t_s": float(s["t_s"][int(alt.argmax())]),
        "speed_end_m_s": float(ch["speed_m_s"][-1]),
        "fpa_end_deg": float(ch["fpa_deg"][-1]),
        "mass_end_kg": float(s["mass_kg"][-1]),
        "max_q": {"q_pa": r.max_q.q_pa, "t_s": r.max_q.t_s,
                  "altitude_m": r.max_q.altitude_m, "mach": r.max_q.mach},
        "alpha_max_deg": r.alpha_max_deg,
    }
    if ascent:
        out["tvc"] = {k: v for k, v in run.summary().items() if not isinstance(v, tuple)}
    return out


def _curves(diff) -> dict:
    out = {"t_s": diff.t_s.tolist()}
    for ch in DIFF_CHANNELS:
        out[f"d_{ch}"] = diff.diff[ch].tolist()
        out[f"ref_{ch}"] = diff.ref[ch].tolist()
        out[f"six_{ch}"] = diff.six[ch].tolist()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out", type=Path,
                    default=REPO / "docs" / "benchmarks" / "fl605_gravity_turn_ascent.json")
    ap.add_argument("--t-end", type=float, default=180.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--sample-every", type=int, default=20)
    args = ap.parse_args()

    t0 = time.time()
    scenario = f05_scenario(t_end_s=args.t_end, dt_s=args.dt,
                            sample_every=args.sample_every)
    study = run_f05_study(scenario)
    wall_s = time.time() - t0
    s = study.scenario

    payload = {
        "schema": "orbitopt-fl605-1",
        "task": "#230 (F-05, epic #216)",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "wall_clock_s": round(wall_s, 1),
        "scenario": {
            "name": s.name,
            "vehicle": {"name": s.vehicle.name, "dry_mass_kg": s.vehicle.dry_mass_kg,
                        "propellant_mass_kg": s.vehicle.propellant_mass_kg,
                        "inertia_kg_m2": list(s.vehicle.inertia_kg_m2),
                        "aero": "packaged spike_stage tables (orbitopt/data/flightlab/aero/)"},
            "engine": {"thrust_vacuum_n": s.vehicle.engine.thrust_vacuum_n,
                       "isp_vacuum_s": s.vehicle.engine.isp_vacuum_s,
                       "isp_sea_level_s": s.vehicle.engine.isp_sea_level_s,
                       "nozzle_exit_area_m2": s.vehicle.engine.nozzle_exit_area_m2,
                       "mount_x_cg_m": s.vehicle.engine.mount_x_cg_m,
                       "gimbal_limit_rad": s.vehicle.engine.gimbal_limit_rad},
            "launch": {"lat_geod_deg": 28.5, "lon_deg": -80.6, "heading_deg": 90.0},
            "program": {"times_s": list(s.program.times_s),
                        "pitch_deg": list(s.program.pitch_deg),
                        "heading_deg": list(s.program.heading_deg)},
            "integration": {"t_end_s": s.t_end_s, "dt_s": s.dt_s, "dt_c_s": s.dt_c_s,
                            "sample_every": s.sample_every,
                            "actuator_real": {"limit_rad": s.real_actuator().limit_rad,
                                             "max_slew_rate_rad_s": s.real_actuator().max_slew_rate_rad_s,
                                             "tau_s": s.real_actuator().tau_s},
                            "actuator_ideal": {"limit_rad": s.ideal_actuator().limit_rad,
                                               "max_slew_rate_rad_s": s.ideal_actuator().max_slew_rate_rad_s,
                                               "tau_s": s.ideal_actuator().tau_s}},
            "assumptions": list(s.assumptions),
        },
        "rungs": {name: _run_record(study.runs[name]) for name in RUNG_ORDER},
        "headline": {"label": study.headline.label,
                     "metrics": study.headline.summary(),
                     "curves": _curves(study.headline)},
        "attribution": {name: {"description": desc,
                               "metrics": study.attribution[name].summary(),
                               "curves": _curves(study.attribution[name])}
                        for name, desc in ATTRIBUTION_STEPS},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    hl = study.headline.summary()
    print(f"{args.out} written ({wall_s:.0f}s): d_alt max {hl['d_altitude_m_max_abs_m']:.1f} m,"
          f" d_speed max {hl['d_speed_m_s_max_abs_m_s']:.2f} m/s,"
          f" d_fpa max {hl['d_fpa_deg_max_abs_deg']:.3f} deg,"
          f" d_mass max {hl['d_mass_kg_max_abs_kg']:.1f} kg")


if __name__ == "__main__":
    main()
