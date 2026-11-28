"""Contract tests for the E13-01 GNC record types (orbitopt.gnc.contracts):
document round-trips against gnc-contracts-1.0, the version gate, and the
per-record validity walls (finite numbers, symmetric covariances, declared
latency, mandatory provenance) -- every acceptance-critical invariant gets a
negative test on the wrong side of the line."""

import pytest

from orbitopt.gnc import (GNCContractError, is_valid_record_document, record_from_document,
                          truth_from_document, observation_from_document, estimate_from_document,
                          command_from_document, control_event_from_document)
from orbitopt.gnc.contracts import (CONTRACT_SCHEMA_VERSION, Command, ControlEvent, Estimate,
                                    Observation, TruthSample)
from orbitopt.mission.contract import Epoch
from orbitopt.mission.multispacecraft import is_valid_relative_state_document

MJD = 9300.0
EPOCH = Epoch(MJD, "TDB")


def truth(rec="truth-00001-chaser", sc="chaser", t=1.5):
    return TruthSample(record_id=rec, spacecraft_id=sc, t_s=t, epoch=EPOCH, frame="GCRS",
                       r_m=(7000000.0, 0.0, 0.0), v_m_s=(0.0, 7546.1, 0.0))


def observation(rec="obs-nav-0", sensor="nav", t_sample=2.0, latency=0.3):
    return Observation(record_id=rec, sensor_id=sensor, t_sample_s=t_sample,
                       t_available_s=t_sample + latency, latency_s=latency,
                       relative_state=_rs(),
                       covariance_position_m2=tuple([4.0 if i % 4 == 0 else 0.0 for i in range(9)]),
                       covariance_velocity_m2_s2=tuple([1e-4 if i % 4 == 0 else 0.0
                                                        for i in range(9)]))


def _rs():
    from orbitopt.mission.multispacecraft import RelativeState
    return RelativeState(target_id="tgt", chaser_id="chaser", epoch=EPOCH, frame="GCRS",
                         r_rel_m=(12.5, -0.4, 3.0), v_rel_m_s=(-0.05, 0.01, 0.0))


def estimate(rec="est-00001", t_update=2.5, sources=("obs-nav-0",)):
    cov = [0.0] * 36
    for i in range(6):
        cov[i * 7] = 1.0
    return Estimate(record_id=rec, estimator_id="nav", t_update_s=t_update,
                    relative_state=_rs(), covariance_rel_m2=tuple(cov),
                    source_observation_ids=tuple(sources))


def command(rec="cmd-00001", t_tick=3.0, t_execute=3.5, src="est-00001"):
    return Command(record_id=rec, actuator_id="thruster", t_tick_s=t_tick, t_execute_s=t_execute,
                   delta_v_eci_m_s=(0.1, -0.02, 0.0), source_estimate_id=src)


def control_event(rec="cev-00001", t_exec=3.5, executed=True, src="cmd-00001"):
    dv = (0.095, -0.021, 0.0) if executed else (0.0, 0.0, 0.0)
    return ControlEvent(record_id=rec, actuator_id="thruster", t_exec_s=t_exec,
                        executed=executed, commanded_delta_v_m_s=(0.1, -0.02, 0.0),
                        realized_delta_v_m_s=dv, source_command_id=src)


ALL = (truth, observation, estimate, command, control_event)


@pytest.mark.parametrize("factory", ALL)
def test_round_trip_is_stable(factory):
    rec = factory()
    doc = rec.to_document()
    assert doc["schema_version"] == CONTRACT_SCHEMA_VERSION
    again = record_from_document(doc)
    assert again.to_document() == doc
    assert again == rec


def test_observation_relative_state_is_an_e12_01_document():
    doc = observation().to_document()
    assert doc["relative_state"]["schema_version"] == "1.0"
    assert is_valid_relative_state_document(doc["relative_state"])


@pytest.mark.parametrize("factory", ALL)
def test_version_gate_refuses_unknown_versions(factory):
    doc = factory().to_document()
    doc["schema_version"] = "2.0"
    with pytest.raises(GNCContractError, match="schema for version"):
        record_from_document(doc)


@pytest.mark.parametrize("factory", ALL)
def test_unknown_extra_field_refused(factory):
    doc = factory().to_document()
    doc["surprise"] = 1
    assert not is_valid_record_document(doc)
    with pytest.raises(GNCContractError, match="gnc-contracts-1.0"):
        record_from_document(doc)


def test_record_kind_mismatch_refused():
    doc = truth().to_document()
    doc["kind"] = "observation"
    with pytest.raises(GNCContractError, match="kind"):
        record_from_document(doc)


def test_truth_rejects_non_finite_state():
    with pytest.raises(GNCContractError, match="finite"):
        TruthSample(record_id="t", spacecraft_id="c", t_s=0.0, epoch=EPOCH, frame="GCRS",
                    r_m=(float("inf"), 0.0, 0.0), v_m_s=(0.0, 0.0, 0.0))


def test_truth_refuses_non_uniform_timescale():
    with pytest.raises(GNCContractError, match="timescale"):
        TruthSample(record_id="t", spacecraft_id="c", t_s=0.0, epoch=Epoch(MJD, "UTC"),
                    frame="GCRS", r_m=(0.0, 0.0, 0.0), v_m_s=(0.0, 0.0, 0.0))


def test_observation_latency_wall():
    with pytest.raises(GNCContractError, match="latency"):
        Observation(record_id="o", sensor_id="s", t_sample_s=2.0, t_available_s=1.0,
                    latency_s=0.3, relative_state=_rs(),
                    covariance_position_m2=tuple([1.0 if i % 4 == 0 else 0.0 for i in range(9)]))


def test_observation_asymmetric_covariance_refused():
    with pytest.raises(GNCContractError, match="symmetric"):
        Observation(record_id="o", sensor_id="s", t_sample_s=2.0, t_available_s=2.3,
                    latency_s=0.3, relative_state=_rs(),
                    covariance_position_m2=(4.0, 1.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0))


def test_estimate_needs_sources():
    with pytest.raises(GNCContractError, match="source_observation_ids"):
        Estimate(record_id="e", estimator_id="nav", t_update_s=1.0, relative_state=_rs(),
                 covariance_rel_m2=tuple([1.0 if i % 7 == 0 else 0.0 for i in range(36)]),
                 source_observation_ids=())


def test_estimate_optional_velocity_covariance_round_trips_absent():
    obs = Observation(record_id="o2", sensor_id="s", t_sample_s=1.0, t_available_s=1.0,
                      latency_s=0.0, relative_state=_rs(),
                      covariance_position_m2=tuple([1.0 if i % 4 == 0 else 0.0 for i in range(9)]))
    doc = obs.to_document()
    assert "covariance_velocity_m2_s2" not in doc
    assert observation_from_document(doc) == obs


def test_command_fire_not_before_tick():
    with pytest.raises(GNCContractError, match="t_execute"):
        Command(record_id="c", actuator_id="a", t_tick_s=3.0, t_execute_s=2.0,
                delta_v_eci_m_s=(0.0, 0.0, 0.0), source_estimate_id="e")


def test_control_event_not_executed_delivers_nothing():
    with pytest.raises(GNCContractError, match="did not happen"):
        ControlEvent(record_id="x", actuator_id="a", t_exec_s=1.0, executed=False,
                     commanded_delta_v_m_s=(0.1, 0.0, 0.0),
                     realized_delta_v_m_s=(0.1, 0.0, 0.0), source_command_id="c")


def test_record_id_required_unique_key():
    with pytest.raises(GNCContractError, match="record_id"):
        truth(rec="")


def test_document_parser_set_matches_kinds():
    from orbitopt.gnc import RECORD_KINDS
    parsers = {"truth": truth_from_document, "observation": observation_from_document,
               "estimate": estimate_from_document, "command": command_from_document,
               "control_event": control_event_from_document}
    assert set(parsers) == set(RECORD_KINDS)
