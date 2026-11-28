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
* :mod:`orbitopt.gnc.ekf` -- ``RelativeNavEKF`` (E13-02): the relative-state
  EKF; CW / nonlinear (+J2) process models reusing E12-02's dynamics,
  position / velocity / range-rate / range-angle measurement models on the
  E13-01 ``Observation`` records, Joseph-form updates, ``Estimate``
  publishing with provenance.
* :mod:`orbitopt.gnc.guidance` (E13-04) -- ``CwMidCourseLaw``: the scheduled
  feedback law on the E13-01 seam (planned gate burn, CW two-point
  mid-course correction and arrival re-target from the estimate alone; the
  law cannot name the plant, ``solve_cw_intercept`` guards the near-singular
  ``Phi_12``).
* :mod:`orbitopt.gnc.closed_loop`` -- :func:`simulate_closed_loop`: the
  general Truth -> Sensor -> Navigation -> Guidance -> Command -> Execution
  error -> Truth loop over any E13-03 sensors, the E13-02 EKF and any
  GuidanceLaw, sealing a ``GncRunLog``; ``tracking_error`` /
  ``delta_v_ledger`` / ``convergence_block`` are the offline readers.
 * :mod:`orbitopt.gnc.nav_metrics` -- offline consistency and observability
   evaluation: NEES / NIS against declared chi-square intervals, sigma
   coverage, stacked observability, cluster statistics.
 * :mod:`orbitopt.gnc.constraints` / :mod:`orbitopt.gnc.lqr` /
   :mod:`orbitopt.gnc.mpc` (E13-05) -- the constrained-control research set:
   one declared constraint family (keep-out, corridor, velocity profile,
   thrust/budget), a finite-horizon CW LQR law, and a receding-horizon
   constrained MPC law that carries the constraint set inside its QP.
 * :mod:`orbitopt.gnc.supervisor` (E13-06) -- ``SafetySupervisor``: a
   covariance-inflated E12-04-envelope monitor and a nominal -> hold -> abort
   state machine wrapped around any GuidanceLaw (estimate + covariance only,
   zero thrust once supervising), with ``score_supervision`` as the offline
   truth-scoring reader (missed alarms and false alarms, counted separately).
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
                     Command,
                     ControlEvent,
                     Estimate,
                     GNCContractError,
                     Observation,
                     RECORD_KINDS,
                     RECORD_TYPES,
                     TruthSample,
                     command_from_document,
                     control_event_from_document,
                     estimate_from_document,
                     is_valid_record_document,
                     observation_from_document,
                     record_from_document,
                     truth_from_document,
)
from .ekf import (
                     CwProcess,
                     MeasurementModel,
                     NATIVE_COVARIANCE_KEY,
                     NATIVE_MEASUREMENT_KEY,
                     NonlinearRelativeProcess,
                     RangeAngle,
                     RangeRate,
                     RelativeNavEKF,
                     RelativePosition,
                     RelativeProcessModel,
                     RelativeVelocity,
                     Stacked,
                     UpdateReport,
                     frame_covariance,
                     white_noise_acceleration_q,
)
from .closed_loop import (
                     Actuator,
                     Plant,
                     convergence_block,
                     default_actuator,
                     default_plant,
                     delta_v_ledger,
                     simulate_closed_loop,
                     tracking_error,
)
from .feed import GuidanceInput, GuidanceLaw, guidance_input_from, run_guidance_step
from .constraints import ApproachConstraints, ConstraintSetError, VelocityProfile
from .lqr import FiniteHorizonCwLaw, CwLqrLaw, cw_impulse_matrices, finite_horizon_cw_lqr_gains
from .mpc import CwMpcLaw
from .guidance import (
                     GuidanceLawError,
                     MANEUVER_MODES,
                     CwMidCourseLaw,
                     ManeuverPlan,
                     ReferenceEphemeris,
                     solve_cw_intercept,
)
from .nav_metrics import (
                     ConsistencyGate,
                     chi_square_cdf,
                     chi_square_interval,
                     cluster_stats,
                     nees,
                     nis,
                     observability_rank,
                     pass_rate,
                     rmse,
                     rmse_per_axis,
                     sigma_coverage,
                     stacked_observability,
)
from .recording import (
                     GncRunLog,
                     RUN_LOG_KIND,
                     read_run_log,
                     replay_stream,
                     run_digest_of,
                     run_log_from_document,
                     write_run_log,
)
from .sensors import (
                     CHANNELS,
                     ChannelErrorConfig,
                     DropoutConfig,
                     MeasurementChannel,
                     RelativeSensorModel,
                     SENSOR_CONFIG_KIND,
                     SENSOR_CONFIG_SCHEMA_FILE,
                     SENSOR_CONFIG_VERSION,
                     SENSOR_KINDS,
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
from .supervisor import (
                     INSTANTANEOUS_MONITORS,
                     STATE_ABORT,
                     STATE_HOLD,
                     STATE_NOMINAL,
                     SUPERVISOR_STATES,
                     MonitorReading,
                     SafetyEnvelope,
                     SafetySupervisor,
                     SupervisionEvent,
                     SupervisorConfig,
                     alert_intervals,
                     score_supervision,
                     violation_intervals,
)

__all__ = [
                     "Actuator",
                     "ApproachConstraints",
                     "INSTANTANEOUS_MONITORS",
                     "STATE_ABORT",
                     "STATE_HOLD",
                     "STATE_NOMINAL",
                     "SUPERVISOR_STATES",
                     "CwMidCourseLaw",
                     "GuidanceLawError",
                     "MANEUVER_MODES",
                     "ManeuverPlan",
                     "Plant",
                     "ReferenceEphemeris",
                     "convergence_block",
                     "default_actuator",
                     "default_plant",
                     "delta_v_ledger",
                     "simulate_closed_loop",
                     "solve_cw_intercept",
                     "tracking_error",
                     "cw_impulse_matrices",
                     "finite_horizon_cw_lqr_gains",
                     "VelocityProfile",
                     "violation_intervals",
                     "CHANNELS",
                     "CLOCK_KIND",
                     "CLOCK_TIMESCALES",
                     "CONTRACT_SCHEMA_FILE",
                     "CONTRACT_SCHEMA_VERSION",
                     "ChannelErrorConfig",
                     "ClockSpec",
                     "Command",
                     "ConsistencyGate",
                     "ConstraintSetError",
                     "ControlEvent",
                     "ControllerSpec",
                     "CwLqrLaw",
                     "CwMpcLaw",
                     "CwProcess",
                     "DropoutConfig",
                     "Estimate",
                     "EstimatorSpec",
                     "FiniteHorizonCwLaw",
                     "GNCContractError",
                     "GncRunLog",
                     "GuidanceInput",
                     "GuidanceLaw",
                     "MonitorReading",
                     "SafetyEnvelope",
                     "SafetySupervisor",
                     "SupervisionEvent",
                     "SupervisorConfig",
                     "MeasurementChannel",
                     "MeasurementModel",
                     "NATIVE_COVARIANCE_KEY",
                     "NATIVE_MEASUREMENT_KEY",
                     "NonlinearRelativeProcess",
                     "Observation",
                     "ProportionalLaw",
                     "RECORD_KINDS",
                     "RECORD_TYPES",
                     "RUN_LOG_KIND",
                     "RangeAngle",
                     "RangeRate",
                     "RelativeNavEKF",
                     "RelativePosition",
                     "RelativeProcessModel",
                     "RelativeSensorModel",
                     "RelativeVelocity",
                     "SENSOR_CONFIG_KIND",
                     "SENSOR_CONFIG_SCHEMA_FILE",
                     "SENSOR_CONFIG_VERSION",
                     "SENSOR_KINDS",
                     "SensorFix",
                     "SensorKind",
                     "SensorModelError",
                     "SensorNoise",
                     "SensorSpec",
                     "SensorTick",
                     "Stacked",
                     "TruthSample",
                     "UpdateReport",
                     "chi_square_cdf",
                     "chi_square_interval",
                     "clock_from_document",
                     "cluster_stats",
                     "command_from_document",
                     "control_event_from_document",
                     "controller_spec_from_document",
                     "estimate_from_document",
                     "estimator_spec_from_document",
                     "frame_covariance",
                     "guidance_input_from",
                     "is_valid_record_document",
                     "measurement_nis",
                     "nees",
                     "nis",
                     "observability_rank",
                     "observation_from_document",
                     "pass_rate",
                     "read_run_log",
                     "record_from_document",
                     "reference_task_config",
                     "replay_stream",
                     "rmse",
                     "rmse_per_axis",
                     "run_digest_of",
                     "run_guidance_step",
                     "run_log_from_document",
                     "sensor_config_from_document",
                     "sensor_config_to_document",
                     "sensor_spec_from_document",
                     "alert_intervals",
                     "score_supervision",
                     "sensor_task_config_from_document",
                     "sensor_task_config_to_document",
                     "sigma_coverage",
                     "simulate",
                     "stacked_observability",
                     "truth_from_document",
                     "white_noise_acceleration_q",
                     "write_run_log",
]
