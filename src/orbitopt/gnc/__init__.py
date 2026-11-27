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
* :mod:`orbitopt.gnc.sensors` (E13-03) -- declared relative-measurement
  sensors: channel registry, per-channel bias + white noise, clock-declared
  latency, random/burst dropouts, seeded reproducibility, and a task
  configuration document (``gnc-sensor-config-1.0``) whose every error
  parameter carries its provenance.
"""
from .clock import (
                    CLOCK_KIND,
                    ClockSpec,
                    ControllerSpec,
                    EstimatorSpec,
                    SensorSpec,
                    SensorTick,
                    clock_from_document,
                    controller_spec_from_document,
                    estimator_spec_from_document,
                    sensor_spec_from_document,
)
from .contracts import (
                    CLOCK_TIMESCALES,
                    CONTRACT_SCHEMA_FILE,
                    CONTRACT_SCHEMA_VERSION,
                    RECORD_KINDS,
                    RECORD_TYPES,
                    Command,
                    ControlEvent,
                    Estimate,
                    GNCContractError,
                    Observation,
                    TruthSample,
                    command_from_document,
                    control_event_from_document,
                    estimate_from_document,
                    is_valid_record_document,
                    observation_from_document,
                    record_from_document,
                    truth_from_document,
)
from .feed import GuidanceInput, GuidanceLaw, guidance_input_from, run_guidance_step
from .recording import (
                    RUN_LOG_KIND,
                    GncRunLog,
                    read_run_log,
                    replay_stream,
                    run_digest_of,
                    run_log_from_document,
                    write_run_log,
)
from .sensors import (
                    CHANNELS,
                    SENSOR_CONFIG_KIND,
                    SENSOR_CONFIG_SCHEMA_FILE,
                    SENSOR_CONFIG_VERSION,
                    SENSOR_KINDS,
                    ChannelErrorConfig,
                    DropoutConfig,
                    MeasurementChannel,
                    RelativeSensorModel,
                    SensorFix,
                    SensorKind,
                    SensorModelError,
                    measurement_nis,
                    reference_task_config,
                    sensor_config_from_document,
                    sensor_config_to_document,
                    sensor_task_config_from_document,
                    sensor_task_config_to_document,
)
from .sim import ProportionalLaw, SensorNoise, simulate

__all__ = [
                    "CHANNELS",
                    "CLOCK_KIND",
                    "CLOCK_TIMESCALES",
                    "CONTRACT_SCHEMA_FILE",
                    "CONTRACT_SCHEMA_VERSION",
                    "RECORD_KINDS",
                    "RECORD_TYPES",
                    "RUN_LOG_KIND",
                    "SENSOR_CONFIG_KIND",
                    "SENSOR_CONFIG_SCHEMA_FILE",
                    "SENSOR_CONFIG_VERSION",
                    "SENSOR_KINDS",
                    "ChannelErrorConfig",
                    "ClockSpec",
                    "Command",
                    "ControlEvent",
                    "ControllerSpec",
                    "DropoutConfig",
                    "Estimate",
                    "EstimatorSpec",
                    "GNCContractError",
                    "GncRunLog",
                    "GuidanceInput",
                    "GuidanceLaw",
                    "MeasurementChannel",
                    "Observation",
                    "ProportionalLaw",
                    "RelativeSensorModel",
                    "SensorFix",
                    "SensorKind",
                    "SensorModelError",
                    "SensorNoise",
                    "SensorSpec",
                    "SensorTick",
                    "TruthSample",
                    "clock_from_document",
                    "command_from_document",
                    "control_event_from_document",
                    "controller_spec_from_document",
                    "estimate_from_document",
                    "estimator_spec_from_document",
                    "guidance_input_from",
                    "is_valid_record_document",
                    "measurement_nis",
                    "observation_from_document",
                    "read_run_log",
                    "record_from_document",
                    "reference_task_config",
                    "replay_stream",
                    "run_digest_of",
                    "run_guidance_step",
                    "run_log_from_document",
                    "sensor_config_from_document",
                    "sensor_config_to_document",
                    "sensor_spec_from_document",
                    "sensor_task_config_from_document",
                    "sensor_task_config_to_document",
                    "simulate",
                    "truth_from_document",
                    "write_run_log",
]
