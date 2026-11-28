"""Innovation-consistency tests for the E13-03 error model (task #420,
acceptance: "NEES/NIS consistent under the declared errors, or state the
failure reason").

What is *provable here* is the measurement half of the gate: for white noise
alone the innovation sum-of-squares per channel is chi-square(1), so the mean
NIS of 3000 seeded fixes sits on the number of channels and its variance on
twice that (z-scores asserted, no scipy). Bias is then shown doing exactly
what theory says: it inflates NIS by the deterministic amount
``sum_i bias_i^2 / sigma_i^2`` when the declared covariance ignores it
(``bias_covariance_mode="none"``, the honest default for a filter that must
model the bias), and consistency is restored by ``"bounded"`` when the
configuration conservatively widens the covariance instead.

The filter half -- NEES of a full EKF state -- was the documented failure
reason when this task merged: E13-02 (#419) was still in progress, so no
Estimate stream existed to form ``d' P^-1 d`` from. The measurement-level gate
below is precisely its prerequisite. #419 has since landed on this branch, so
the reason string is still asserted (the deferral stays on record) and the
check that replaces it asserts the deferral is discharged by #419's own
filter-level NEES gates. The same argument is written into
docs/gnc-contract.md.
"""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from orbitopt.gnc import ClockSpec, ControllerSpec, EstimatorSpec, measurement_nis
from orbitopt.gnc.sensors import ChannelErrorConfig, DropoutConfig, RelativeSensorModel
from orbitopt.mission.contract import Epoch
from orbitopt.mission.multispacecraft import RelativeState

SRC = "DESIGN ASSUMPTION: unit-test fixture"
EPOCH = Epoch(9000.0, "TDB")
TRUTH = RelativeState(target_id="T", chaser_id="C", epoch=EPOCH, frame="GCRS",
                      r_rel_m=(6800.0, 90.0, 15.0), v_rel_m_s=(0.0, 0.008, 0.001))

#: Why the task's filter-level acceptance clause was answered with a failure
#: reason instead of a NEES number on task #420's branch (kept on record after
#: #419 merged: see the discharge test at the end of this file).
E13_02_FAILURE_REASON = (
    "E13-02 (#419) navigation filter is not merged into this branch yet (status: In progress), "
    "so no Estimate stream exists to form filter-level NEES d'P^-1 d from. Measurement-level "
    "NIS consistency under the declared errors IS demonstrated in this file (white noise on "
    "chi-square moments; bias inflation predicted and restored by bias_covariance_mode="
    "'bounded'), which is the prerequisite the EKF consumes. On merge of #419: feed the same "
    "seeded fix stream to the filter and assert per-update NEES mean == 6 at 6 DoF.")

#: Where #419 discharges that deferral: the sealed E13-02 Monte-Carlo evidence.
E13_02_EVIDENCE = (Path(__file__).resolve().parents[1]
                   / "docs/benchmarks/e13_02_relative_nav_ekf.json")


def cfgs(sigma_p, bias_p, sigma_v, bias_v, mode, seed=21):
    def c(n, s, b):
        return ChannelErrorConfig(n, s, b, SRC)
    return RelativeSensorModel(
        "s", "cartesian_relnav", 1.0, 0.0, 0.25, seed,
        (c("rel_pos_x_m", sigma_p, bias_p), c("rel_pos_y_m", sigma_p, bias_p),
         c("rel_pos_z_m", sigma_p, bias_p), c("rel_vel_x_m_s", sigma_v, bias_v),
         c("rel_vel_y_m_s", sigma_v, bias_v), c("rel_vel_z_m_s", sigma_v, bias_v)),
        dropout=DropoutConfig(), bias_covariance_mode=mode, source=SRC)


def nis_stream(model, n=3000):
    clock = ClockSpec(t0=EPOCH, dt_dynamics_s=0.5, sensors=(model.sensor_spec(n),),
                      estimator=EstimatorSpec(1.0, 0.0), controller=ControllerSpec(1.0, 0.0))
    out = []
    for tick in clock.sensor_schedule("s", (n - 1) * model.period_s):
        fix = model.measure(tick, TRUTH)
        if fix.valid:
            out.append(measurement_nis(fix, TRUTH))
    return np.array(out)


def _z(sample_mean, expected, n, dof):
    # chi-square(k): SE of the sample mean is sqrt(2k/n)
    return (sample_mean - expected) / math.sqrt(2.0 * dof / n)


def test_white_noise_alone_sits_on_chi_square_moments():
    a = nis_stream(cfgs(1.0, 0.0, 0.02, 0.0, "none"))
    n, dof = len(a), 6
    assert n == 3000
    assert abs(a.mean() - dof) < 0.5                            # |z| < 1.4 expected
    assert abs(_z(a.mean(), dof, n, dof)) < 4.0
    assert abs(a.var(ddof=1) - 2.0 * dof) < 4.0                  # Var(chi2_6) = 12, SE ~ 0.35


def test_bias_inflates_nis_by_exactly_bias_squared_over_sigma_squared():
    bias_p, sigma_p, bias_v, sigma_v = 0.5, 1.0, 0.01, 0.02
    predicted_inflation = 3 * bias_p ** 2 / sigma_p ** 2 + 3 * bias_v ** 2 / sigma_v ** 2  # = 1.5
    none = nis_stream(cfgs(sigma_p, bias_p, sigma_v, bias_v, "none"))
    assert abs(none.mean() - (6.0 + predicted_inflation)) < 0.6
    bounded = nis_stream(cfgs(sigma_p, bias_p, sigma_v, bias_v, "bounded"))
    assert abs(bounded.mean() - 6.0) < 0.6                       # consistency restored
    assert bounded.mean() < none.mean()                          # direction of the effect
    # and the declared variance really is the widened one (sigma^2 + bias^2 per channel):
    probe = cfgs(sigma_p, bias_p, sigma_v, bias_v, "bounded")
    from orbitopt.gnc.clock import SensorTick
    fix = probe.measure(SensorTick("s", 0, 0.0, 0.25, False), TRUTH)
    assert fix.measurement_variance["rel_pos_x_m"] == pytest.approx(sigma_p ** 2 + bias_p ** 2)


def test_latency_and_record_time_are_invisible_to_measurement_nis():
    fast = cfgs(1.0, 0.0, 0.02, 0.0, "none")
    slow = RelativeSensorModel("s", "cartesian_relnav", 1.0, 0.0, 47.0, 21, fast.channels,
                               bias_covariance_mode="none", source=SRC)
    a = [measurement_nis(fast.measure(t, TRUTH), TRUTH) for t in
         ClockSpec(EPOCH, 0.5, (fast.sensor_spec(10),), EstimatorSpec(1.0, 0.0),
                   ControllerSpec(1.0, 0.0)).sensor_schedule("s", 9.0)]
    b = [measurement_nis(slow.measure(t, TRUTH), TRUTH) for t in
         ClockSpec(EPOCH, 0.5, (slow.sensor_spec(10),), EstimatorSpec(1.0, 0.0),
                   ControllerSpec(1.0, 0.0)).sensor_schedule("s", 9.0)]
    assert a == b                                                # same seed, same draws: latency
    # is bookkeeping for the filter, not a change to the measurement itself


def test_drops_do_not_disturb_the_consistency_of_what_arrives():
    model = cfgs(1.0, 0.0, 0.02, 0.0, "none")
    model = RelativeSensorModel("s", "cartesian_relnav", 1.0, 0.0, 0.25, 21, model.channels,
                                dropout=DropoutConfig(random_drop_p=0.15, burst_enter_p=0.02),
                                bias_covariance_mode="none", source=SRC)
    a = nis_stream(model, n=3000)
    assert 2200 < len(a) < 2900                                  # ~15-20% loss
    assert abs(_z(a.mean(), 6.0, len(a), 6.0)) < 4.0             # survivors stay consistent


def test_filter_level_nees_deferral_is_discharged_by_e13_02():
    """The acceptance clause '或给出失效原因' was answered with the reason above, whose
    blocker was #419. #419 has merged, so the clause is answered by its own
    filter-level NEES gate now: this test keeps the reason on record, checks the
    blocker is gone, and re-reads the sealed 6-DoF NEES evidence #419 ships."""
    assert "#419" in E13_02_FAILURE_REASON and "NEES" in E13_02_FAILURE_REASON
    assert importlib.util.find_spec("orbitopt.gnc.ekf") is not None
    assert importlib.util.find_spec("orbitopt.gnc.nav_metrics") is not None
    assert E13_02_EVIDENCE.exists(), f"run #419's benchmark first: {E13_02_EVIDENCE}"
    doc = json.loads(E13_02_EVIDENCE.read_text(encoding="utf-8"))
    gate = doc["setup"]["declared_gates"]["nees"]
    assert gate["dof"] == 6 and gate["interval"][0] > 1.0            # chi-square band at 6 DoF
    matched = [r for r in doc["cells"] if r["matched"] and r["meas"] in ("pos", "pos_vel")]
    assert matched and all(r["nees_pass_rate"] >= 0.70 for r in matched), matched
    assert all(g["pass"] for g in doc["gates"] if g["name"].startswith("consistency")), \
        [g for g in doc["gates"] if g["name"].startswith("consistency") and not g["pass"]]
