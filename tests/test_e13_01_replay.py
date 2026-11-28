"""Acceptance gate #3 of E13-01 -- the complete measurement, estimation,
control and configuration record is replayable. Runs the reference
closed-loop simulator, seals its log, replays it, and attacks the replay:
tampering must be caught twice over (canonical digest, then the causal /
schedule gates independently, so recomputing a digest does not resurrect a
doctored run)."""
import copy
import json

import pytest

import orbitopt.gnc as gnc
from orbitopt.gnc import (ClockSpec, ControllerSpec, EstimatorSpec, GNCContractError, ProportionalLaw,
                          SensorNoise, SensorSpec, simulate, write_run_log, read_run_log,
                          run_log_from_document, run_digest_of, replay_stream)
from orbitopt.mission.contract import Epoch
from orbitopt.mission.multispacecraft import MultiSpacecraftMission, SpacecraftSpec

T0 = Epoch(9300.0, "TDB")

SIM_KW = dict(
    frame="GCRS", target_id="tgt", chaser_id="chaser",
    chief_r_m=(7000000.0, 0.0, 0.0), chief_v_m_s=(0.0, 7546.1, 0.0),
    deputy_r_m=(7000040.0, 0.0, 15.0), deputy_v_m_s=(0.0, 7546.1, 0.02),
    laws={"chaser": ProportionalLaw(kp_per_s=0.01, kd=0.2)},
    noise={"navcam": SensorNoise(std_position_m=1.0, std_velocity_m_s=0.02, seed=7)},
    t_end_s=80.0, execution_scale=0.95, execution_rtol=0.05, execution_seed=3)


def make_clock():
    return ClockSpec(
        t0=T0, dt_dynamics_s=5.0,
        sensors=(SensorSpec("navcam", period_s=2.0, phase_s=0.5, latency_s=0.3,
                            dropped_ticks=(3, 9)),),
        estimator=EstimatorSpec(period_s=1.0, phase_s=0.0),
        controller=ControllerSpec(period_s=4.0, phase_s=0.0, latency_s=1.0))


def mission_doc():
    m = MultiSpacecraftMission(
        mission_id="e13-01-sim", title="closed-loop contract demo", frame="GCRS", epoch=T0,
        spacecraft=(SpacecraftSpec("tgt", "TARGET"), SpacecraftSpec("chaser", "CHASER")))
    return m.to_dict()


@pytest.fixture(scope="module")
def sealed(tmp_path_factory):
    log = simulate(run_id="e13-01-run", clock=make_clock(), mission=mission_doc(), **SIM_KW)
    path = tmp_path_factory.mktemp("gnc") / "run.json"
    write_run_log(log, path)
    return log, read_run_log(path), path


def test_run_produced_every_record_kind(sealed):
    log, replayed, _ = sealed
    counts = {k: len(v) for k, v in log.records.items()}
    assert counts["truth"] > 0 and counts["observation"] > 0 and counts["estimate"] > 0
    assert counts["command"] > 0 and counts["control_event"] > 0
    # the loop actually closed: a burn happened and was recorded as executed
    assert any(ev.executed for ev in replay_stream(replayed, "control_event"))


def test_dropped_ticks_are_absent_but_later_ticks_keep_their_time(sealed):
    log, _, _ = sealed
    spec = log.clock.sensor("navcam")
    delivered = {o.record_id.split("-")[-1] for o in log.records["observation"]}
    # tick 00003 / 00009 delivered nothing...
    assert "00003" not in delivered and "00009" not in delivered
    # ...while tick 10 kept its declared instant (drops shift nothing)
    t10 = [o for o in log.records["observation"] if o.record_id.endswith("00010")]
    assert abs(t10[0].t_sample_s - spec.sample_time(10)) < 1e-9


def test_determinism_same_seeds_same_digest(sealed):
    log, _, _ = sealed
    again = simulate(run_id="e13-01-run", clock=make_clock(), mission=mission_doc(), **SIM_KW)
    assert again.digest() == log.digest()


def test_replay_digest_is_round_trip_stable(sealed):
    log, replayed, path = sealed
    assert replayed.digest() == log.digest()
    # and the file's own claim agrees
    doc = json.loads(path.read_text())
    assert doc["run_digest"] == log.digest()


def test_mission_binding_is_the_e12_01_epoch_gate(sealed):
    doc = json.loads(sealed[2].read_text())
    doc["mission"]["epoch"]["mjd2000"] += 1.0e-3  # a day? no -- 86 s of skew
    doc["run_digest"] = run_digest_of(doc)  # attacker recomputes the seal
    with pytest.raises(GNCContractError, match="single-shared-epoch"):
        run_log_from_document(doc)


def test_tampering_is_caught_by_the_digest():
    log = simulate(run_id="e13-01-run", clock=make_clock(), **SIM_KW)
    doc = log.to_document(log.digest())
    doc["records"]["truth"][0]["position_m"][0] += 250.0  # "smooth" the plant
    with pytest.raises(GNCContractError, match="digest mismatch"):
        run_log_from_document(doc)


def test_tampering_survives_a_recomputed_digest_only_if_it_stays_honest():
    # The digest proves nobody edited the file; the GATES prove the story is
    # possible. Here we hand-edit a record AND recompute the digest: replay
    # must still refuse it on the schedule/causality gates, not wave it in.
    log = simulate(run_id="e13-01-run", clock=make_clock(), **SIM_KW)
    doc = log.to_document("")
    obs = doc["records"]["observation"][5]
    obs["t_sample_s"] += 0.35  # slide it off its declared tick
    obs["t_available_s"] += 0.35  # latency story stays intact, schedule does not
    doc["run_digest"] = run_digest_of(doc)
    with pytest.raises(GNCContractError, match="declared tick"):
        run_log_from_document(doc)


def test_command_with_an_unlogged_source_estimate_is_refused():
    log = simulate(run_id="e13-01-run", clock=make_clock(), **SIM_KW)
    doc = log.to_document("")
    cmd = doc["records"]["command"][0]
    cmd["source_estimate_id"] = "est-99999"
    doc["run_digest"] = run_digest_of(doc)
    with pytest.raises(GNCContractError, match="not in the log"):
        run_log_from_document(doc)


def test_estimate_citing_a_measurement_that_arrived_later_is_refused():
    log = simulate(run_id="e13-01-run", clock=make_clock(), **SIM_KW)
    doc = log.to_document("")
    est = doc["records"]["estimate"][2]
    last_obs = doc["records"]["observation"][-1]["record_id"]  # arrives after this update
    est["source_observation_ids"] = [last_obs]
    doc["run_digest"] = run_digest_of(doc)
    with pytest.raises(GNCContractError, match="before it arrived|not in the log"):
        run_log_from_document(doc)


def test_control_event_before_its_command_fire_time_is_refused():
    log = simulate(run_id="e13-01-run", clock=make_clock(), **SIM_KW)
    doc = log.to_document("")
    doc["records"]["control_event"][0]["t_exec_s"] -= 2.0
    doc["run_digest"] = run_digest_of(doc)
    with pytest.raises(GNCContractError, match="before the"):
        run_log_from_document(doc)


def test_duplicate_ids_are_ambiguous_provenance():
    log = simulate(run_id="e13-01-run", clock=make_clock(), **SIM_KW)
    doc = log.to_document("")
    doc["records"]["estimate"][1]["record_id"] = doc["records"]["estimate"][0]["record_id"]
    doc["run_digest"] = run_digest_of(doc)
    with pytest.raises(GNCContractError, match="used twice"):
        run_log_from_document(doc)


def test_records_on_the_log_are_the_sealed_ones_in_order(sealed):
    log, replayed, _ = sealed
    assert [o.record_id for o in replay_stream(replayed, "observation")][:3] == \
        [o.record_id for o in replay_stream(log, "observation")][:3]
    with pytest.raises(GNCContractError, match="unknown record kind"):
        replay_stream(replayed, "desire")


def test_config_records_the_loop_that_ran(sealed):
    _, replayed, _ = sealed
    cfg = dict(replayed.config)
    assert cfg["noise"]["navcam"]["std_position_m"] == 1.0
    assert cfg["execution"]["scale"] == 0.95
    assert "event_priority" in cfg


def test_plan_digest_binding_is_checked_hex64():
    with pytest.raises(GNCContractError, match="64 hex"):
        gnc.GncRunLog.new(run_id="r", clock=make_clock(), plan_digest="deadbeef")
