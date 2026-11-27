"""Benchmark: do the E13-03 declared sensor errors behave as declared?

The unit tests prove each error mechanism in isolation; this benchmark checks
the *statistics a filter would see*, on the reference sensor pair and a few
deliberate variants, with enough samples that the numbers mean something:

* measurement-level NIS (``orbitopt.gnc.measurement_nis`` against the truth
  the sensor was shown) for {white noise only, noise+bias mode none,
  noise+bias mode bounded}: mean/var vs chi-square moments, z-scored, plus
  the predicted bias inflation ``sum bias^2/sigma^2``;
* covariance coverage of the polar-to-Cartesian map: empirical scatter of the
  reconstructed relative position vs the covariance the Observation claims --
  a ratio near 1.0 means the linearised covariance map (not just the channel
  sigmas) is honest at the declared geometry;
* dropout statistics: realised loss fraction vs the model prediction
  ``p + enter_p * mean_ticks`` (random + burst), on the stochastic schedules;
* reproducibility: two full streams at one seed agree bit-for-bit.

    PYTHONPATH=src python benchmarks/bench_e13_03_sensor_consistency.py \
        -o docs/benchmarks/e13_03_sensor_consistency.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from orbitopt.gnc import (
    ClockSpec,
    ControllerSpec,
    EstimatorSpec,
    measurement_nis,
    reference_task_config,
)
from orbitopt.gnc.sensors import ChannelErrorConfig, DropoutConfig, RelativeSensorModel
from orbitopt.mission.contract import Epoch
from orbitopt.mission.multispacecraft import RelativeState

SRC = "DESIGN ASSUMPTION: benchmark fixture (error parameters as marked in the config)"
EPOCH = Epoch(9000.0, "TDB")
TRUTH = RelativeState(target_id="TGT", chaser_id="CHM", epoch=EPOCH, frame="GCRS",
                     r_rel_m=(7000.0, 110.0, 25.0), v_rel_m_s=(0.0, 0.011, 0.0015))
N = 3000


def chans(*specs):
    return tuple(ChannelErrorConfig(c, s, b, SRC) for c, s, b in specs)


def relnav(sigma, bias, mode, seed=101, dropout=None):
    return RelativeSensorModel(
        "rdv", "cartesian_relnav", 1.0, 0.0, 0.25, seed,
        chans(("rel_pos_x_m", sigma, bias), ("rel_pos_y_m", sigma, bias),
              ("rel_pos_z_m", sigma, bias), ("rel_vel_x_m_s", sigma / 50, bias / 50),
              ("rel_vel_y_m_s", sigma / 50, bias / 50), ("rel_vel_z_m_s", sigma / 50,
                                                         bias / 50)),
        dropout=dropout or DropoutConfig(), bias_covariance_mode=mode, source=SRC)


def lidar(seed=42, mode="none", dropout=None):
    return RelativeSensorModel(
        "lidar", "polar_range_angle", 1.0, 0.0, 0.2, seed,
        chans(("range_m", 0.5, 0.05), ("azimuth_rad", 1e-4, 5e-5),
              ("elevation_rad", 1e-4, 5e-5), ("range_rate_m_s", 0.02, 0.005)),
        dropout=dropout or DropoutConfig(), bias_covariance_mode=mode, source=SRC)


def nis_stream(model, n):
    clock = ClockSpec(t0=EPOCH, dt_dynamics_s=0.5, sensors=(model.sensor_spec(n),),
                      estimator=EstimatorSpec(1.0, 0.0), controller=ControllerSpec(1.0, 0.0))
    fixes, kept = [], 0
    for tick in clock.sensor_schedule(model.sensor_id, (n - 1) * model.period_s):
        fix = model.measure(tick, TRUTH)
        if fix.valid:
            fixes.append(measurement_nis(fix, TRUTH))
            kept += 1
    return np.array(fixes), kept


def nis_case(name, model, dof, predicted_extra=0.0):
    a, kept = nis_stream(model, N)
    mean, var = float(a.mean()), float(a.var(ddof=1))
    z = (mean - (dof + predicted_extra)) / float(np.sqrt(2.0 * dof / len(a)))
    return {"case": name, "n_fixes": kept, "dof": dof, "nis_mean": round(mean, 4),
            "nis_var": round(var, 4), "expected_mean": dof + predicted_extra,
            "expected_var": 2.0 * dof, "z_mean": round(z, 2)}


def coverage_case():
    """Polar map: sample the reconstructed position over N frames, compare the
    empirical scatter to the covariance each Observation claims."""
    m = lidar()
    clock = ClockSpec(t0=EPOCH, dt_dynamics_s=0.5, sensors=(m.sensor_spec(N),),
                      estimator=EstimatorSpec(1.0, 0.0), controller=ControllerSpec(1.0, 0.0))
    pos, claims, truth = [], [], np.asarray(TRUTH.r_rel_m, dtype=float)
    for tick in clock.sensor_schedule("lidar", (N - 1) * m.period_s):
        obs = m.observe(tick, TRUTH, target_id="TGT", chaser_id="CHM", epoch=EPOCH)
        if obs is None:
            continue
        pos.append(obs.relative_state.r_rel_m)
        claims.append(np.array(obs.covariance_position_m2).reshape(3, 3).trace() / 3.0)
    d = np.asarray(pos) - truth
    emp = float(d.var(axis=0, ddof=1).mean())
    declared = float(np.mean(claims))
    return {"case": "polar_covariance_coverage", "n_frames": len(pos),
            "empirical_mean_var_m2": round(emp, 6), "declared_mean_var_m2": round(declared, 6),
            "coverage_ratio": round(emp / declared, 4)}


def dropout_case():
    m = lidar(dropout=DropoutConfig(random_drop_p=0.02, burst_enter_p=0.01,
                                    burst_mean_ticks=4.0))
    frac = []
    for n in (1000, 2000, 3000):
        d = m.dropout_schedule(n)
        frac.append(round(len(d) / n, 4))
    return {"case": "dropout_fraction", "predicted": 0.02 + 0.01 * 4.0,
            "measured_by_horizon": frac}


def reproducibility_case():
    a, _ = nis_stream(lidar(), 500)
    b, _ = nis_stream(lidar(), 500)
    return {"case": "seed_reproducibility", "identical": bool(np.array_equal(a, b))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()
    ref = reference_task_config()
    report = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "n_stream": N,
        "reference_config_ids": [c.sensor_id for c in ref],
        "cases": [
            nis_case("noise_only", relnav(1.0, 0.0, "none"), 6),
            nis_case("noise_bias_mode_none", relnav(1.0, 0.5, "none"), 6,
                     predicted_extra=3 * 0.5 ** 2 / 1.0 ** 2 + 3 * 0.01 ** 2 / 0.02 ** 2),
            nis_case("noise_bias_mode_bounded", relnav(1.0, 0.5, "bounded"), 6),
            nis_case("reference_lidar_bias_none", lidar(), 4,
                     predicted_extra=0.05 ** 2 / 0.5 ** 2 + 2 * (5e-5) ** 2 / (1e-4) ** 2
                                            + 0.005 ** 2 / 0.02 ** 2),
            coverage_case(), dropout_case(), reproducibility_case(),
        ],
    }
    ok = all(abs(c.get("z_mean", 0.0)) < 4.0 for c in report["cases"] if "z_mean" in c) \
        and 0.5 < coverage_case()["coverage_ratio"] < 2.0 \
        and reproducibility_case()["identical"]
    report["verdict"] = "consistent" if ok else "INCONSISTENT"
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
