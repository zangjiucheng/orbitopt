"""Benchmark: E6-02 guided JSBSim ascent with staging (task #274).

Flies the packaged ``e602_scenario`` (production airframe ``spike_stage_aero``
under the FL-403 PID on a 140 s gravity turn, commanded MECO and the FL-402
split at t = 95 s, delayed ignition on the upper child) on both plants -- the
JSBSim production path and the matching native rigid-body twin -- and records:

* the five per-phase metric blocks (stack powered, booster coast, upper coast
  before ignition, upper powered, upper coast after burn-out) for each plant,
  so the acceptance's powered/coast split is one file, not two greps;
* the declared phase-constraint verdicts on the JSBSim table, the staging
  bookkeeping verdicts on both paths, and the four-channel comparison;
* the scenario itself (pad, program, schedule, split fractions, the constraint
  table with its rationales) so the JSON is self-describing.

Run (this is the command that regenerates the committed JSON):

    PYTHONPATH=src python benchmarks/bench_e6_02_guided_ascent.py \
        -o docs/benchmarks/e6_02_guided_ascent.json

``tests/test_flightlab_sixdof_guided_ascent_jsbsim.py`` checks the committed
file against regression thresholds; ``docs/adr/flightlab-0010-...md`` records
the scenario decisions and the residual attribution.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from orbitopt.flightlab.sixdof.guided_ascent import (  # noqa: E402
    E602_PHASE_CONSTRAINTS, PHASES, e602_scenario, run_e602_study)


def _json_safe(x):
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [_json_safe(v) for v in x]
    if is_dataclass(x) and not isinstance(x, type):
        return _json_safe(asdict(x))
    if hasattr(x, "item") and hasattr(x, "dtype"):  # numpy scalar
        return _json_safe(x.item())
    if isinstance(x, float):
        return round(x, 6) if abs(x) < 1e15 else f"{x:.6e}"
    return x


def main(out_path: Path | None) -> None:
    t0 = time.time()
    study = run_e602_study()
    s = study.scenario
    phases = study.phase_table()
    checks = [_json_safe(c) for c in study.checks]
    failed = [c["id"] for c in checks if not c["passed"]]

    payload = {
        "meta": {
            "task": "E6-02 guided JSBSim ascent with staging (#274)",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "runtime_s": round(time.time() - t0, 1),
            "dt_s": s.dt_s, "sample_every": s.sample_every,
            "phases": list(PHASES),
            "all_checks_passed": not failed,
        },
        "scenario": {
            "name": s.name, "model": s.model, "booster_model": s.booster_model,
            "upper_model": s.upper_model,
            "pad": {"lat_deg": 28.5, "lon_deg": -80.6, "heading_deg": 90.0},
            "program_times_s": list(s.program_times_s),
            "program_elevation_deg": list(s.program_elevation_deg),
            "separation_t_s": s.separation_t_s, "ignition_delay_s": s.ignition_delay_s,
            "ignition_t_s": s.ignition_t_s, "t_end_s": s.t_end_s,
            "separation_impulse_ns": s.separation_impulse_ns,
            "split_dry_fraction": s.booster_dry_fraction,
            "split_propellant_fraction": s.booster_propellant_fraction,
            "assumptions": list(s.assumptions),
        },
        "constraints": {f"{p}|{m}": {"limit": lim, "sense": sense, "unit": unit,
                                     "rationale": why}
                        for (p, m), (lim, sense, unit, why)
                        in E602_PHASE_CONSTRAINTS.items()},
        "phases": phases,
        "bookkeeping": {"jsbsim": _json_safe(study.jsbsim.bookkeeping),
                        "native": _json_safe(study.native.bookkeeping),
                        "jsbsim_ignition_onset_err_s": study.jsbsim.ignition_onset_err_s,
                        "native_ignition_onset_err_s": study.native.ignition_onset_err_s,
                        "jsbsim_upper_burnout_t_s": study.jsbsim.upper_burnout_t_s,
                        "native_upper_burnout_t_s": study.native.upper_burnout_t_s},
        "comparison": _json_safe(study.comparison["categories"]),
        "checks": checks,
        "failed_checks": failed,
    }
    text = json.dumps(_json_safe(payload), indent=1)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n")
        print(f"wrote {out_path}")
    print(f"[phases] {len(PHASES)}, checks {len(checks)}, failed: {failed or 'none'}")
    for plant in ("jsbsim", "native"):
        print(f"[{plant}] " + " | ".join(
            f"{p}: err_max={phases[plant][p]['att_err_max_deg']:.2f}deg "
            f"alpha_max={phases[plant][p]['alpha_max_deg']:.1f}deg "
            f"maxq={phases[plant][p]['max_q_pa']:.0f}Pa"
            for p in PHASES))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()
    main(args.out)
