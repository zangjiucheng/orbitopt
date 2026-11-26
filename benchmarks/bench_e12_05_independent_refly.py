"""Benchmark: independent-model margins for LEO-RDV-001 (E12-05, task #395).

One artifact recording, for the reference rendezvous mission, every number a
reviewer needs to audit the tolerance bands declared in
:mod:`orbitopt.verify.rendezvous_refly`:

* the three error-source lines the task demands separately -- numerical
  propagation (TudatPy DOP853 tolerance axis + RK4 step axis sweeps, Kepler
  closed-form compare), approximate CW model, optimizer non-convergence --
  each with its measured value and its declared band;
* the headline acceptance quantities re-flown in TudatPy beside the main
  model's values, and the margin of each measured difference against the
  declared band (bands must carry >= 2x margin; the numbers say whether they
  do);
* the convergence tables themselves, so the observed integration orders are
  checkable rather than asserted.

Without a tudatpy backend the numerical and independent-model sections are
recorded ``null`` with the reason, and the CW/optimizer lines still measure
-- the artifact then tunes only what this environment can see (the
cross-env run on the tudat machine produces the complete document).

    PYTHONPATH=src python benchmarks/bench_e12_05_independent_refly.py \
        -o docs/benchmarks/e12_05_independent_refly.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from orbitopt.verify import rendezvous_refly as rr  # noqa: E402
from orbitopt.verify.report import provenance, to_jsonable  # noqa: E402

ARTIFACT_KIND = "orbitopt.bench.e12-05-independent-refly"


def margins(src: dict, measured_key: str, tol_key: str) -> dict:
    """measured / tolerance per declared band (inf-safe)."""
    out = {}
    for name, measured in src[measured_key].items():
        tol = src["tolerances"].get(tol_key(name))
        if tol is None or not isinstance(measured, (int, float)):
            continue
        out[name] = {"measured": measured, "tolerance": tol,
                     "margin": (float("inf") if measured == 0
                                else tol / measured)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out", default="docs/benchmarks/e12_05_independent_refly.json")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    res = rr.verify_rendezvous_refly(config_path=args.config,
                                     out_dir=Path(args.out).parent / "e12-05-refly-run")
    rep = res["report"]
    src = rep["inputs"]["error_sources"]
    plan = res["plan"]
    sealed = res["sealed"]
    mq, cw, refly = res["main_quantities"], res["cw"], res["refly"]

    doc = {
        "kind": ARTIFACT_KIND, "schema_version": "1.0",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "provenance": provenance(),
        "mission_id": rep["inputs"]["mission_id"],
        "backend": "absent (REFLY_BACKEND_MISSING)" if refly is None else "present",
        "not_checked_reasons": rep["inputs"]["not_checked_reasons"],
        "reference_plan": {
            "planner": "orbitopt.rendezvous.canonical_plan "
                       "(E12-04 plan_rendezvous_safe, constrained optimum)",
            "plan_digest": sealed.plan_digest,
            "config_sha256": sealed.config_sha256,
            "initial_states_sha256": sealed.initial_states_sha256,
            "maneuvers_sha256": sealed.maneuvers_sha256,
            "drift_end_s": plan.drift_end_s, "tof_s": plan.tof_s,
            "total_delta_v_m_s": plan.total_delta_v_m_s,
            "delta_v_budget_m_s": plan.delta_v_budget_m_s,
            "maneuvers": plan.to_dict()["maneuvers"],
        },
        "error_sources": {k: {kk: vv for kk, vv in v.items() if kk != "note"}
                          for k, v in src.items()},
        "margins": {
            "cw_vs_nonlinear": margins(src["approximate_model_cw"], "measured",
                                       lambda n: {"hold_point_position_error":
                                                  "hold_point_position_error_m",
                                                  "relative_speed": "relative_speed_m_s",
                                                  "min_approach_distance":
                                                  "min_approach_distance_m"}.get(n)),
            "refly_vs_main": (margins(src["independent_model_agreement"], "measured",
                                      lambda n: {"hold_point_position_error":
                                                 "hold_point_position_error_m",
                                                 "relative_speed": "relative_speed_m_s",
                                                 "min_approach_distance":
                                                 "min_approach_distance_m",
                                                 "total_delta_v": "total_delta_v_m_s",
                                                 "maneuver_sequence_identity":
                                                 "maneuver_sequence_m_s"}.get(n))
                              if refly is not None else None),
        },
        "acceptance_quantities": {
            "main_model": {k: mq[k] for k in
                           ("position_error_m", "relative_speed_m_s",
                            "min_approach_distance_m", "total_delta_v_m_s",
                            "min_approach_epoch_s")},
            "cw_replay": {k: cw[k] for k in
                          ("position_error_m", "relative_speed_m_s",
                           "min_approach_distance_m", "total_delta_v_m_s")},
            "refly_j2": ({"position_error_m": refly["refly"]["j2"]["position_error_m"],
                          "relative_speed_m_s": refly["refly"]["j2"]["relative_speed_m_s"],
                          "min_approach_distance_m":
                          refly["refly"]["j2"]["min_approach_distance_m"],
                          "min_approach_epoch_s": refly["refly"]["j2"]["min_approach_epoch_s"],
                          "total_delta_v_m_s": refly["plan"]["total_delta_v_m_s"]}
                         if refly is not None else None),
        },
        "convergence": refly["convergence"] if refly is not None else None,
        "declared_tolerances": rep["inputs"]["declared_tolerances"],
        "verdicts": {"headline_acceptance": rep["inputs"]["headline_acceptance"],
                     "overall": rep["status"]["result"],
                     "independent_validation": rep["status"]["independent_validation"]},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(to_jsonable(doc), indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"{out}: backend={doc['backend']} headline={doc['verdicts']['headline_acceptance']}")
    for k, v in src.items():
        print(f"  {k}: {v['result']}")
    if doc["margins"]["cw_vs_nonlinear"]:
        worst = min(m["margin"] for m in doc["margins"]["cw_vs_nonlinear"].values())
        print(f"  smallest CW margin: {worst:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
