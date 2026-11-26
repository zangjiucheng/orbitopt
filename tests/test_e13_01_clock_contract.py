"""Clock-contract tests for E13-01 (orbitopt.gnc.clock): the acceptance
criteria demand that desynchronized sampling, latency and frame drops keep
the TIMESTAMPS correct. Every timestamp here is a pure function of its tick
index, so the tests assert exact values, never approximate schedules."""
import pytest

from orbitopt.gnc import (CLOCK_KIND, ClockSpec, ControllerSpec, EstimatorSpec, GNCContractError,
                          SensorSpec, SensorTick, clock_from_document)
from orbitopt.mission.contract import Epoch, SECONDS_PER_DAY

T0 = Epoch(9300.0, "TDB")


def sensor(sid="nav", period=1.0, phase=0.25, latency=0.1, drops=()):
    return SensorSpec(sensor_id=sid, period_s=period, phase_s=phase, latency_s=latency,
                      dropped_ticks=drops)


def clock(**over):
    base = dict(t0=T0, dt_dynamics_s=0.1,
                sensors=(sensor(), sensor(sid="lidar", period=2.0, phase=1.7, latency=0.5)),
                estimator=EstimatorSpec(period_s=0.5, phase_s=0.0, latency_s=0.05),
                controller=ControllerSpec(period_s=0.2, phase_s=0.05, latency_s=0.02))
    base.update(over)
    return ClockSpec(**base)


def test_timescales_must_be_uniform():
    with pytest.raises(GNCContractError, match="linear"):
        ClockSpec(t0=Epoch(9300.0, "UTC"), dt_dynamics_s=0.1, sensors=(sensor(),),
                  estimator=EstimatorSpec(period_s=1.0, phase_s=0.0),
                  controller=ControllerSpec(period_s=1.0, phase_s=0.0))


def test_phase_is_within_one_period():
    with pytest.raises(GNCContractError, match="within one period"):
        sensor(phase=1.5)  # period 1.0: a whole-period delay is t0's job


def test_desynchronized_domains_have_the_times_they_declared():
    clk = clock()
    # two sensors, different periods AND phases; filter and controller on
    # yet another cadence -- nothing shares a period, only t0
    sched_nav = clk.sensor_schedule("nav", 3.0)
    assert [t.t_sample_s for t in sched_nav] == [0.25, 1.25, 2.25]
    sched_lidar = clk.sensor_schedule("lidar", 6.0)
    assert [t.t_sample_s for t in sched_lidar] == [1.7, 3.7, 5.7]
    assert clk.estimator_times(1.0) == (0.0, 0.5, 1.0)
    assert clk.control_ticks(0.5) == ((0.05, 0.07), (0.25, 0.27), (0.45, 0.47))


def test_latency_delivers_later_but_never_shifts_the_schedule():
    clk = clock()
    spec = clk.sensor("nav")
    before = [t.t_sample_s for t in clk.sensor_schedule("nav", 5.0)]
    slow = ClockSpec(t0=T0, dt_dynamics_s=clk.dt_dynamics_s,
                     sensors=(sensor(latency=2.0),
                              sensor(sid="lidar", period=2.0, phase=1.7, latency=0.5)),
                     estimator=clk.estimator, controller=clk.controller)
    after = [t.t_sample_s for t in slow.sensor_schedule("nav", 5.0)]
    assert before == after  # latency is delivery delay, not schedule shift
    assert all(abs(t.t_available_s - (t.t_sample_s + spec.latency_s)) < 1e-12
               for t in clk.sensor_schedule("nav", 5.0))


def test_drops_delete_records_not_timestamps():
    spec = sensor(drops=(2, 3))
    # every other tick keeps the pure phase + n*period instant, exactly
    assert spec.sample_time(0) == 0.25
    assert spec.sample_time(1) == 1.25
    assert spec.sample_time(2) == 2.25  # dropped, but the INSTANT is declared
    assert spec.sample_time(4) == 4.25  # not shifted by the two missing frames
    clk = clock(sensors=(spec,))
    sched = clk.sensor_schedule("nav", 5.0)
    assert [t.dropped for t in sched] == [False, False, True, True, False]
    assert [t.tick for t in sched] == [0, 1, 2, 3, 4]
    assert [t.t_sample_s for t in sched] == [0.25, 1.25, 2.25, 3.25, 4.25]


def test_schedule_stops_at_horizon_and_respects_phase():
    clk = clock()
    assert clk.sensor_schedule("lidar", 1.0) == ()  # phase 1.7 beyond horizon
    assert clk.sensor_schedule("nav", 0.25)[0].tick == 0


def test_epoch_at_counts_uniform_days_and_matches_delta_seconds():
    clk = clock()
    e = clk.epoch_at(86400.0)
    assert e.timescale == "TDB"
    assert e.mjd2000 == pytest.approx(T0.mjd2000 + 1.0, abs=1e-9)
    assert clk.tdb_seconds_at(3600.0) - clk.tdb_seconds_at(0.0) == pytest.approx(3600.0, abs=1e-6)


def test_dynamics_times_are_exact_multiples_not_a_drifted_accumulation():
    clk = clock(dt_dynamics_s=0.1)
    ts = clk.dynamics_times(0.5)
    assert ts == (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)  # a float accumulation would drift


def test_duplicate_sensor_ids_refused():
    with pytest.raises(GNCContractError, match="duplicate"):
        clock(sensors=(sensor(), sensor()))


def test_no_sensors_refused():
    with pytest.raises(GNCContractError, match="at least one sensor"):
        clock(sensors=())


def test_document_round_trip_and_gates():
    clk = clock()
    doc = clk.to_document()
    assert doc["kind"] == CLOCK_KIND
    back = clock_from_document(doc)
    assert back.to_document() == doc
    assert back.sensor("lidar").dropped_ticks == ()
    bad = dict(doc, schema_version="9.9")
    with pytest.raises(GNCContractError, match="schema for version"):
        clock_from_document(bad)
    bad = dict(doc, kind="orbitopt.gnc.whatever")
    with pytest.raises(GNCContractError, match="kind"):
        clock_from_document(bad)
    bad = dict(doc, t0={"mjd2000": 9300.0, "timescale": "UTC"})
    with pytest.raises(GNCContractError, match="not one of|linear"):
        clock_from_document(bad)


def test_sensor_tick_is_a_value_not_an_object_graph():
    t = SensorTick("nav", 3, 3.25, 3.35, True)
    assert (t.sensor_id, t.tick, t.dropped) == ("nav", 3, True)
