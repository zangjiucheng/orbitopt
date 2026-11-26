"""Closed-loop GNC simulation contract (E13-01): five versioned data
contracts, one declared clock, one sealed replayable run log.

* :mod:`orbitopt.gnc.contracts` -- the record types: ``TruthSample`` (plant
  truth), ``Observation`` (sensor measurement + latency + covariance),
  ``Estimate`` (navigation state + covariance + cited sources), ``Command``
  (guidance decision + fire time), ``ControlEvent`` (what the actuator
  really delivered). All round-trip through
  ``orbitopt/schemas/gnc-contracts-1.0.json``, relative state included as
  E12-01's ``relative-state-1.0`` document.
* :mod:`orbitopt.gnc.clock` -- ``ClockSpec``: dynamics step, per-sensor
  period/phase/latency/drops, filter and controller cadences, counted from
  one ``Epoch`` on a uniform timescale; every timestamp a pure function of
  its tick index.
* :mod:`orbitopt.gnc.feed` -- the isolation boundary: a ``GuidanceLaw``
  sees a ``GuidanceInput`` and nothing else; pushing anything that is not an
  ``Estimate`` is a ``TypeError``, and the module cannot even name truth.
* :mod:`orbitopt.gnc.recording` -- ``GncRunLog``: append-only sealed log;
  ``read_run_log`` re-checks every reference, every declared tick and the
  canonical run digest before replay is allowed.
* :mod:`orbitopt.gnc.sim` -- ``simulate``: the reference closed loop
  (truth via E12-02 propagation, seeded noise, trivial filter, any
  GuidanceLaw, actuation error recorded) producing such a log.
"""
from .clock import (CLOCK_KIND, ClockSpec, ControllerSpec, EstimatorSpec, SensorSpec,
                    SensorTick, clock_from_document, controller_spec_from_document,
                    estimator_spec_from_document, sensor_spec_from_document)
from .contracts import (CLOCK_TIMESCALES, CONTRACT_SCHEMA_FILE, CONTRACT_SCHEMA_VERSION,
                        Command, ControlEvent, Estimate, GNCContractError, Observation,
                        RECORD_KINDS, RECORD_TYPES, TruthSample, command_from_document,
                        control_event_from_document, estimate_from_document,
                        is_valid_record_document, observation_from_document,
                        record_from_document, truth_from_document)
from .feed import GuidanceInput, GuidanceLaw, guidance_input_from, run_guidance_step
from .recording import (RUN_LOG_KIND, GncRunLog, read_run_log, replay_stream, run_digest_of,
                        run_log_from_document, write_run_log)
from .sim import ProportionalLaw, SensorNoise, simulate

__all__ = [
    "CLOCK_TIMESCALES", "CONTRACT_SCHEMA_FILE", "CONTRACT_SCHEMA_VERSION",
    "Command", "ControlEvent", "Estimate", "GNCContractError", "Observation",
    "RECORD_KINDS", "RECORD_TYPES", "TruthSample",
    "command_from_document", "control_event_from_document", "estimate_from_document",
    "is_valid_record_document", "observation_from_document", "record_from_document",
    "truth_from_document",
    "CLOCK_KIND", "ClockSpec", "ControllerSpec", "EstimatorSpec", "SensorSpec",
    "SensorTick", "clock_from_document", "controller_spec_from_document",
    "estimator_spec_from_document", "sensor_spec_from_document",
    "GuidanceInput", "GuidanceLaw", "guidance_input_from", "run_guidance_step",
    "RUN_LOG_KIND", "GncRunLog", "read_run_log", "replay_stream", "run_digest_of",
    "run_log_from_document", "write_run_log",
    "ProportionalLaw", "SensorNoise", "simulate",
]
