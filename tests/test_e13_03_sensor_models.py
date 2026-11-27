"""E13-03 sensor-model unit tests: every declared error behaves like its
configuration says (white-noise moments, constant bias, clock-declared
latency, random + burst dropouts), the output is a valid E13-01
Observation (timestamps, covariances, presence/absence), the task
configuration round-trips through gnc-sensor-config-1.0 with provenance
mandatory, and every construction wall has its negative test."""

import math

import numpy as np
import pytest

from orbitopt.gnc import (
    ClockSpec,
    ControllerSpec,
    EstimatorSpec,
    GNCContractError,
    GncRunLog,
    Observation,
    SensorModelError,
    measurement_nis,
    observation_from_document,
    reference_task_config,
    sensor_config_from_document,
    sensor_config_to_document,
    sensor_task_config_from_document,
    sensor_task_config_to_document,
)
from orbitopt.gnc.sensors import ChannelErrorConfig, DropoutConfig, RelativeSensorModel
from orbitopt.mission.contract import Epoch
from orbitopt.mission.multispacecraft import RelativeState

SRC = "DESIGN ASSUMPTION: unit-test fixture"
TRUTH_POS = (7100.0, 120.0, 30.0)
TRUTH_VEL = (0.0, 0.012, 0.002)


def chans(*specs):
    return tuple(ChannelErrorConfig(c, s, b, SRC) for c, s, b in specs)


def polar(sigma=0.5, bias=0.05, seed=5, mode="none", dropout=None, rr=True,
          lat=0.2, period=1.0, phase=0.0):
    spec = [("range_m", sigma, bias), ("azimuth_rad", 1e-4, 5e-5),
            ("elevation_rad", 1e-4, 5e-5)]
    if rr:
        spec.append(("range_rate_m_s", 0.02, 0.005))
    return RelativeSensorModel("s", "polar_range_angle", period, phase, lat, seed, chans(*spec),
                               dropout=dropout or DropoutConfig(), bias_covariance_mode=mode,
                               source=SRC)


def cartesian(seed=5, dropout=None, mode="none"):
    return RelativeSensorModel(
        "s", "cartesian_relnav", 2.0, 0.0, 0.5, seed,
        chans(("rel_pos_x_m", 1.0, 0.1), ("rel_pos_y_m", 1.0, 0.1),
              ("rel_pos_z_m", 1.0, 0.1), ("rel_vel_x_m_s", 0.02, 0.002),
              ("rel_vel_y_m_s", 0.02, 0.002), ("rel_vel_z_m_s", 0.02, 0.002)),
        dropout=dropout or DropoutConfig(), bias_covariance_mode=mode, source=SRC)


def truth(pos=TRUTH_POS, vel=TRUTH_VEL, t=0.0):
    return RelativeState(target_id="T", chaser_id="C", epoch=Epoch(9000.0 + t / 86400.0, "TDB"),
                         frame="GCRS", r_rel_m=pos, v_rel_m_s=vel)


def sched(model, n, drops=()):
    clock = ClockSpec(t0=Epoch(9000.0, "TDB"), dt_dynamics_s=0.1,
                      sensors=(model.sensor_spec(n),), estimator=EstimatorSpec(1.0, 0.0),
                      controller=ControllerSpec(1.0, 0.0))
    return clock.sensor_schedule(model.sensor_id, model.phase_s + (n - 1) * model.period_s)


# ---------------------------------------------------------------------------
# white noise: empirical moments match the configured sigma
# ---------------------------------------------------------------------------


def test_white_noise_moments_match_sigma():
    m = polar(sigma=1.0, bias=0.0)
    t = truth()
    values = []
    for tick in sched(m, 2000):
        if tick.dropped:
            continue
        fix = m.measure(tick, t)
        values.append(fix.measurements["range_m"] - np.linalg.norm(np.array(TRUTH_POS)))
    a = np.array(values)
    assert len(a) == 2000
    assert abs(a.mean()) < 0.10                      # 2000 samples, sigma 1 -> SE 0.022
    assert abs(a.std(ddof=1) - 1.0) < 0.06            # SE of std ~ 0.016


def test_noise_is_independent_across_ticks():
    m = polar(sigma=1.0, bias=0.0, seed=3)
    t = truth()
    ts = sched(m, 300)
    d = [m.measure(ts[i + 1], t).measurements["range_m"]
         - m.measure(ts[i], t).measurements["range_m"] for i in range(299)]
    d = np.array(d)
    assert abs(d.mean()) < 0.3
    assert abs(d.std(ddof=1) - math.sqrt(2.0)) < 0.2


def test_reproducible_same_seed_and_tick():
    m = polar()
    t = truth()
    ts = sched(m, 4)
    a = m.measure(ts[2], t)
    b = polar().measure(sched(polar(), 4)[2], t)
    assert dict(a.measurements) == dict(b.measurements)


def test_different_seed_changes_draws():
    m = polar(seed=8, bias=0.0)
    t = truth()
    ts = sched(m, 3)
    other = polar(seed=9, bias=0.0)
    assert (m.measure(ts[1], t).measurements["range_m"]
            != other.measure(sched(other, 3)[1], t).measurements["range_m"])


# ---------------------------------------------------------------------------
# bias: constant in time, exactly as declared
# ---------------------------------------------------------------------------


def test_bias_is_constant_and_declared():
    ideal = float(np.linalg.norm(np.array(TRUTH_POS)))
    m = polar(sigma=1e-9, bias=0.5, rr=False)
    t = truth()
    got = [m.measure(tk, t).measurements["range_m"] for tk in sched(m, 50)]
    assert all(abs(g - (ideal + 0.5)) < 1e-6 for g in got)


def test_bias_bounded_mode_widens_declared_variance():
    none = polar(bias=0.5, mode="none")
    bounded = polar(bias=0.5, mode="bounded")
    t = truth()
    tk = sched(none, 3)[1]
    fn, fb = none.measure(tk, t), bounded.measure(tk, t)
    assert fb.measurement_variance["range_m"] == pytest.approx(fn.measurement_variance["range_m"]
                                                               + 0.25)


# ---------------------------------------------------------------------------
# latency: exactly the clock-declared constant
# ---------------------------------------------------------------------------


def test_latency_is_the_declared_constant():
    m = polar(lat=0.37)
    for tk in sched(m, 5):
        fix = m.measure(tk, truth())
        assert fix.t_available_s == pytest.approx(fix.t_sample_s + 0.37, abs=1e-9)


def test_tick_with_wrong_arrival_time_is_refused():
    from orbitopt.gnc.clock import SensorTick
    m = polar()
    bad = SensorTick("s", 0, 0.0, 0.0 + 0.9, False)          # latency mismatch
    with pytest.raises(SensorModelError, match="latency"):
        m.measure(bad, truth())
    with pytest.raises(SensorModelError, match="not"):       # foreign sensor
        m.measure(SensorTick("other", 0, 0.0, 0.2, False), truth())


# ---------------------------------------------------------------------------
# dropouts: random, burst, explicit, and reproducible
# ---------------------------------------------------------------------------


def test_random_drop_rate_matches_config():
    m = polar(dropout=DropoutConfig(random_drop_p=0.2))
    dropped = m.dropout_schedule(5000)
    assert abs(len(dropped) / 5000 - 0.2) < 0.03              # SE ~ 0.0057 -> 5 sigma


def test_dropout_schedule_is_reproducible_and_seed_dependent():
    d = DropoutConfig(random_drop_p=0.1, burst_enter_p=0.01, burst_mean_ticks=4.0)
    a, b = polar(dropout=d).dropout_schedule(400), polar(seed=6, dropout=d).dropout_schedule(400)
    assert a == polar(dropout=d).dropout_schedule(400)        # same seed -> same mask
    assert a != b                                             # different seed -> different mask


def test_bursts_are_contiguous_runs_of_geometric_length():
    p, mean_ticks = 0.5, 3.0
    m = polar(dropout=DropoutConfig(burst_enter_p=p, burst_mean_ticks=mean_ticks))
    dropped = m.dropout_schedule(600)
    assert len(dropped) > 300                                     # enter p=0.5: most frames die
    runs = _runs(dropped)
    assert len(runs) > 40                                         # many separate loss episodes
    # configured mean is the *burst* length; consecutive-tick bursts merge into observed
    # runs, so the observable is a geometric chain of expected mean/(1-p) = 6.0
    assert abs(sum(r[1] - r[0] + 1 for r in runs) / len(runs) - mean_ticks / (1.0 - p)) < 0.9


def _runs(ticks):
    out, start, prev = [], ticks[0], ticks[0]
    for t in ticks[1:]:
        if t != prev + 1:
            out.append((start, prev))
            start = t
        prev = t
    out.append((start, prev))
    return out


def test_explicit_drops_merge_into_sensor_spec_and_blank_the_frame():
    m = polar(dropout=DropoutConfig(explicit_ticks=(1, 4)))
    spec = m.sensor_spec(6)
    assert sorted(spec.dropped_ticks) == [1, 4]
    tk = sched(m, 6)[1]
    assert tk.dropped
    fix = m.measure(tk, truth())
    assert fix.valid is False and fix.dropped_reason == "declared_drop"
    assert m.observe(tk, truth(), target_id="T", chaser_id="C", epoch=truth().epoch) is None


# ---------------------------------------------------------------------------
# E13-01 Observation contract conformance
# ---------------------------------------------------------------------------


def test_observation_round_trips_through_the_contract_schema():
    m = polar()
    tk = sched(m, 3)[2]
    obs = m.observe(tk, truth(), target_id="T", chaser_id="C", epoch=truth().epoch)
    doc = obs.to_document()
    assert doc["record_kind"] == "observation"
    back = observation_from_document(doc)
    assert back == obs
    assert back.t_available_s == pytest.approx(back.t_sample_s + m.latency_s, abs=1e-9)
    cov = np.array(back.covariance_position_m2).reshape(3, 3)
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > -1e-9 * max(cov.diagonal())


def test_polar_velocity_covariance_is_rank_one_along_los():
    m = polar()
    tk = sched(m, 2)[0]
    obs = m.observe(tk, truth(), target_id="T", chaser_id="C", epoch=truth().epoch)
    cov = np.array(obs.covariance_velocity_m2_s2).reshape(3, 3)
    assert np.linalg.matrix_rank(cov, tol=1e-12 * cov.max()) == 1
    r_hat = np.array(TRUTH_POS) / np.linalg.norm(TRUTH_POS)
    assert np.allclose(cov @ r_hat, 0.02 ** 2 * r_hat, rtol=0.02)  # cov = sigma^2 r_hat r_hat^T
    assert obs.metadata["velocity_measured"] == "true"


def test_cartesian_observation_carries_diagonal_pos_and_vel_cov():
    m = cartesian()
    tk = sched(m, 2)[0]
    obs = m.observe(tk, truth(), target_id="T", chaser_id="C", epoch=truth().epoch)
    pc = np.array(obs.covariance_position_m2).reshape(3, 3)
    vc = np.array(obs.covariance_velocity_m2_s2).reshape(3, 3)
    assert np.count_nonzero(pc - np.diag(pc.diagonal())) == 0
    assert pc.diagonal() == pytest.approx([1.0, 1.0, 1.0])
    assert vc.diagonal() == pytest.approx([4e-4, 4e-4, 4e-4])
    rs = obs.relative_state
    assert math.dist(rs.r_rel_m, TRUTH_POS) < 3.0             # 1 m noise -> sane reconstruction


def test_position_only_sensor_declares_no_velocity_covariance():
    m = RelativeSensorModel("s", "polar_range_angle", 1.0, 0.0, 0.0, 1,
                            chans(("range_m", 0.5, 0.0), ("azimuth_rad", 1e-4, 0.0),
                                  ("elevation_rad", 1e-4, 0.0)), source=SRC)
    obs = m.observe(sched(m, 2)[0], truth(), target_id="T", chaser_id="C", epoch=truth().epoch)
    assert obs.covariance_velocity_m2_s2 == ()
    assert obs.relative_state.v_rel_m_s == pytest.approx((0.0, 0.0, 0.0))
    assert obs.metadata["velocity_measured"] == "false"


def test_observations_land_in_a_sealed_run_log_and_drops_honor_the_clock():
    m = polar(dropout=DropoutConfig(explicit_ticks=(1,)), seed=17)
    n = 6
    clock = ClockSpec(t0=Epoch(9000.0, "TDB"), dt_dynamics_s=0.1,
                      sensors=(m.sensor_spec(n),), estimator=EstimatorSpec(1.0, 0.0),
                      controller=ControllerSpec(1.0, 0.0))
    log = GncRunLog.new(run_id="e13-03-log", clock=clock)
    for tk in clock.sensor_schedule("s", m.phase_s + (n - 1) * m.period_s):
        obs = m.observe(tk, truth(t=tk.t_sample_s), target_id="T", chaser_id="C",
                        epoch=clock.epoch_at(tk.t_sample_s))
        if obs is not None:
            log.append_observation(obs)
    assert len(log.records["observation"]) == 5               # six ticks, one honest absence
    assert {o.t_sample_s for o in log.records["observation"]} == \
        {0.0, 2.0, 3.0, 4.0, 5.0}
    # the sealed log refuses a record on a declared-dropped tick: the absence is enforced,
    # not just recommended. This record is built by hand -- what a broken recorder would
    # produce; the sensor model itself cannot, because its clock tick says dropped.
    ghost = Observation(record_id="obs-s-ghost", sensor_id="s", t_sample_s=1.0,
                        t_available_s=1.0 + m.latency_s, latency_s=m.latency_s,
                        relative_state=RelativeState(
                            target_id="T", chaser_id="C", epoch=clock.epoch_at(1.0),
                            frame="GCRS", r_rel_m=(TRUTH_POS[0] + 0.01, *TRUTH_POS[1:]),
                            v_rel_m_s=TRUTH_VEL),
                        covariance_position_m2=tuple(
                            [4.0 if i % 4 == 0 else 0.0 for i in range(9)]))
    with pytest.raises(GNCContractError, match="declared dropped"):
        log.append_observation(ghost)


def test_boresight_singularity_is_refused_not_guessed():
    m = polar()
    with pytest.raises(SensorModelError, match="boresight"):
        m.measure(sched(m, 2)[0], truth(pos=(1e-9, 1e-9, 9000.0)))


# ---------------------------------------------------------------------------
# task configuration: schema gate, provenance mandatory, negative walls
# ---------------------------------------------------------------------------


def test_reference_config_round_trips_and_is_schema_valid():
    cfgs = reference_task_config()
    doc = sensor_task_config_to_document(cfgs, {"task": "E13-03"})
    assert doc["schema_version"] == "1.0"
    again = sensor_task_config_from_document(doc)
    assert sensor_task_config_to_document(again) == {k: v for k, v in doc.items()
                                                     if k != "metadata"}
    assert [c.sensor_id for c in again] == ["lidar", "relnav"]


def test_every_channel_carries_its_source_or_the_document_is_rejected():
    doc = sensor_config_to_document(polar())
    doc["channels"][0].pop("source")
    with pytest.raises(SensorModelError, match="violates"):
        sensor_config_from_document(doc)


@pytest.mark.parametrize("mutate", [
    lambda d: d["channels"][0].__setitem__("sigma", 0.0),          # zero sigma refused
    lambda d: d["channels"][0].__setitem__("channel", "azimuth_deg"),  # unknown channel
    lambda d: d.__setitem__("kind", "radar"),                       # unimplemented kind
    lambda d: d.__setitem__("seed", -1),                            # seeds are non-negative
    lambda d: d.__setitem__("latency_s", -0.1),
])
def test_schema_negative_walls(mutate):
    doc = sensor_config_to_document(polar())
    mutate(doc)
    with pytest.raises(SensorModelError):
        sensor_config_from_document(doc)


def test_semantic_walls_beyond_the_schema():
    with pytest.raises(SensorModelError, match="phase"):
        polar(phase=2.0)                                          # >= period 1.0
    with pytest.raises(SensorModelError, match="velocity channels"):
        RelativeSensorModel("s", "cartesian_relnav", 1.0, 0.0, 0.0, 1,
                            chans(("rel_pos_x_m", 1.0, 0.0), ("rel_pos_y_m", 1.0, 0.0),
                                  ("rel_pos_z_m", 1.0, 0.0), ("rel_vel_x_m_s", 0.02, 0.0)),
                            source=SRC)
    with pytest.raises(SensorModelError, match="cannot measure"):
        RelativeSensorModel("s", "cartesian_relnav", 1.0, 0.0, 0.0, 1,
                            chans(("range_m", 1.0, 0.0), ("azimuth_rad", 1e-4, 0.0),
                                  ("elevation_rad", 1e-4, 0.0)), source=SRC)
    with pytest.raises(SensorModelError, match="source"):
        RelativeSensorModel("s", "polar_range_angle", 1.0, 0.0, 0.0, 1,
                            chans(("range_m", 1.0, 0.0), ("azimuth_rad", 1e-4, 0.0),
                                  ("elevation_rad", 1e-4, 0.0)), source="  ")
    with pytest.raises(SensorModelError, match="bias_covariance_mode"):
        polar(mode="generous")


def test_measurement_nis_rejects_invalid_input():
    m = polar(dropout=DropoutConfig(explicit_ticks=(1,)))
    tk = sched(m, 4)[1]
    with pytest.raises(SensorModelError, match="dropped"):
        measurement_nis(m.measure(tk, truth()), truth())
