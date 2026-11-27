"""E13-04 (task #421): closed-loop guidance framework -- the terminal
intercept solver, stale-measurement compensation, the zero-latency actuation
path, failure attribution, and one end-to-end campaign sample flown
closed-loop.

Unit layer: the weighted-least-squares CW intercept reproduces the legacy
position-only solution at velocity_weight=0 and nulls the closing speed as
the weight grows; the EKF pulls a stale full-Cartesian reading forward to the
compensation epoch through the process model (an uncompensated stale update
injects v_rel * staleness of systematic error) and refuses compensation on
anything but the full-state model; a zero-latency controller's command
actuates at its own tick; the attributor classes failures from the run-log
evidence alone. Sample layer: one frozen campaign draw -- cluster 1, index 5
of seed base 20 000 000, an E12-10 holdout arrival_accuracy FAILURE -- flies
closed-loop to a SUCCESS on identical perturbations and execution draws. The
full paired benchmark lives in benchmarks/bench_e13_04_closed_loop_guidance.py."""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from orbitopt.gnc import (
    ChannelErrorConfig,
    ControllerSpec,
    CwMidCourseLaw,
    CwProcess,
    DropoutConfig,
    EstimatorSpec,
    GNCContractError,
    GuidanceLawError,
    ManeuverPlan,
    RangeRate,
    ReferenceEphemeris,
    RelativeNavEKF,
    RelativePosition,
    RelativeSensorModel,
    RelativeVelocity,
    Stacked,
    solve_cw_intercept,
)
from orbitopt.gnc.clock import ClockSpec
from orbitopt.gnc.closed_loop import default_plant, simulate_closed_loop
from orbitopt.gnc.recording import run_log_from_document
from orbitopt.mission.contract import Epoch
from orbitopt.mission.multispacecraft import (
    MultiSpacecraftMission,
    RelativeState,
    SpacecraftSpec,
    mission_to_dict,
)
from orbitopt.relative.cw import cw_state_transition

N = 0.001140
MU = 3.9860044150e14
T0 = Epoch(9000.0, "TDB")
SRC = "DESIGN ASSUMPTION (E13-04): unit test sensor profile"


def _sensor(period=20.0, latency=2.0, pos=1.0, vel=0.01):
    chans = tuple(ChannelErrorConfig(c, s, 0.0, SRC) for c, s in
                  (("rel_pos_x_m", pos), ("rel_pos_y_m", pos), ("rel_pos_z_m", pos),
                   ("rel_vel_x_m_s", vel), ("rel_vel_y_m_s", vel), ("rel_vel_z_m_s", vel)))
    return RelativeSensorModel("relnav", "cartesian_relnav", period, 0.0, latency, 7, chans,
                               dropout=DropoutConfig(), bias_covariance_mode="none",
                               source=SRC)


def _obs(sensor, t_sample, chief_r, rel_r, rel_v):
    truth = RelativeState(target_id="tgt", chaser_id="chaser", epoch=T0, frame="GCRS",
                          r_rel_m=tuple(float(x) for x in rel_r),
                          v_rel_m_s=tuple(float(x) for x in rel_v))
    from orbitopt.gnc.clock import SensorTick
    tick = SensorTick(sensor_id=sensor.sensor_id, tick=int(t_sample // 10),
                      t_sample_s=float(t_sample),
                      t_available_s=float(t_sample) + float(sensor.latency_s), dropped=False)
    return sensor.observe(tick, truth, target_id="tgt", chaser_id="chaser", epoch=T0)


class TestTerminalIntercept:
    def test_weight_zero_reproduces_position_only_solution(self):
        x0 = np.array([10.0, 5.0, -30.0, 0.5, 1.5, -5.2])
        dv = solve_cw_intercept(N, 60.0, x0, np.zeros(6), velocity_weight=0.0)
        phi = cw_state_transition(N, 60.0)
        rhs = np.zeros(3) - phi[:3, :3] @ x0[:3] - phi[:3, 3:] @ x0[3:]
        assert np.allclose(dv, np.linalg.solve(phi[:3, 3:], rhs), rtol=1e-6)

    def test_high_velocity_weight_nulls_the_closing_speed(self):
        x0 = np.array([10.0, 5.0, -30.0, 0.5, 1.5, -5.2])
        dv = solve_cw_intercept(N, 7.26, x0, np.zeros(6), velocity_weight=100.0)
        assert dv is not None
        phi = cw_state_transition(N, 7.26)
        x_term = phi @ (x0 + phi[:, 3:] @ dv)
        assert float(np.linalg.norm(x_term[3:])) < 0.05
        assert float(np.linalg.norm(dv)) < float(np.linalg.norm(x0[3:])) + 1.0

    def test_moderate_weight_trades_position_against_velocity(self):
        x0 = np.array([0.0, 0.0, -30.0, 0.0, 0.0, -5.0])
        phi = cw_state_transition(N, 7.26)
        outs = [solve_cw_intercept(N, 7.26, x0, np.zeros(6), velocity_weight=w)
                for w in (1.0, 200.0)]
        v = [float(np.linalg.norm((phi @ (x0 + phi[:, 3:] @ dv))[3:])) for dv in outs]
        assert v[1] < v[0]

    def test_negative_velocity_weight_raises(self):
        with pytest.raises(GuidanceLawError):
            solve_cw_intercept(N, 10.0, np.zeros(6), np.zeros(6), velocity_weight=-1.0)

    def test_ill_conditioned_returns_none_not_nan(self):
        # one orbital period: the position block of B collapses to rank 2, the
        # normal matrix is singular -- the caller must get None, never NaN
        tau_period = 2.0 * math.pi / N
        x0 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        assert solve_cw_intercept(N, tau_period, x0, np.zeros(6)) is None

    def test_maneuver_plan_validates_velocity_weight(self):
        with pytest.raises(GuidanceLawError):
            ManeuverPlan("m", 10.0, "retarget", velocity_weight=-2.0)
        assert ManeuverPlan("m", 10.0, "retarget", velocity_weight=50.0).velocity_weight == 50.0


class TestStaleMeasurementCompensation:
    """A measurement sampled at t and consumed at t + dt drags the state back
    by v_rel * dt unless it is propagated forward first; here a deputy closing
    at 6 m/s along-track 10 s before the update epoch."""

    CHIEF_R = np.array([7000000.0, 0.0, 0.0])
    CHIEF_V = np.array([0.0, math.sqrt(MU / 7000000.0), 0.0])
    RHO_T = np.array([0.0, -14940.0, 0.0])     # LVLH state at the update epoch
    RHOD_T = np.array([0.0, 6.0, 0.0])         # 6 m/s closing

    def _filters(self):
        return [RelativeNavEKF(CwProcess(N), np.concatenate((self.RHO_T, self.RHOD_T)),
                               np.diag([100.0, 100.0, 100.0, 1.0, 1.0, 1.0]),
                               q_accel_m2_s3=1e-9)
                for _ in range(2)]

    def _stale_obs(self):
        # sampled 10 s before the update epoch: the along-track reading is 60 m
        # further aft than the state at the compensation epoch
        sensor = _sensor(period=20.0, latency=2.0)
        return _obs(sensor, 100.0, self.CHIEF_R, np.array([0.0, -15000.0, 0.0]),
                    np.array([0.0, 6.0, 0.0]))

    def test_stale_update_injects_bias_compensation_removes_it(self):
        obs = self._stale_obs()
        model = Stacked((RelativePosition(), RelativeVelocity()))
        plain, comp = self._filters()
        plain.update_from_observations({"RelativePosition": obs, "RelativeVelocity": obs},
                                       model, self.CHIEF_R, self.CHIEF_V)
        comp.update_from_observations({"RelativePosition": obs, "RelativeVelocity": obs},
                                      model, self.CHIEF_R, self.CHIEF_V, compensate_to=110.0)
        # uncompensated: pulled toward the 60 m stale reading, several tens of
        # metres of along-track error against the state at the update epoch;
        # compensated: the reading lands where the truth is at t = 110
        assert plain.x[1] - self.RHO_T[1] < -25.0
        assert abs(comp.x[1] - self.RHO_T[1]) < 10.0
        assert plain.x[1] + 60.0 == pytest.approx(comp.x[1], abs=12.0)

    def test_partial_model_compensation_is_refused(self):
        filt = RelativeNavEKF(CwProcess(N), np.zeros(6), np.eye(6), q_accel_m2_s3=1e-9)
        sensor = _sensor()
        obs = _obs(sensor, 100.0, self.CHIEF_R, np.array([0.0, -15000.0, 0.0]),
                   np.array([0.0, 6.0, 0.0]))
        with pytest.raises(GNCContractError):
            filt.update_from_observation(obs, RangeRate(), self.CHIEF_R, self.CHIEF_V,
                                         compensate_to=110.0)


class TestZeroLatencyActuation:
    def test_command_actuates_at_its_own_tick(self):
        r_c = np.array([6878000.0, 0.0, 0.0])
        v_c = np.array([0.0, math.sqrt(MU / 6878000.0), 0.0])
        r_d = r_c + np.array([0.0, 1000.0, 0.0])
        v_d = v_c + np.array([0.0, 0.0, 0.0])
        tt = np.arange(0.0, 601.0, 10.0)
        from orbitopt.relative.propagation import propagate_pair
        tr = propagate_pair(r_c, v_c, r_c, v_c, list(tt), frame="GCRS", mu=MU, j2=0.0,
                            where="test e13-04")
        ref = ReferenceEphemeris(
            times_s=tuple(float(x) for x in tt),
            chief_r_m=tuple(tr.chief_r[0, i, :] for i in range(len(tt))),
            chief_v_m_s=tuple(tr.chief_v[0, i, :] for i in range(len(tt))),
            frame="GCRS", mu_m3_s2=MU)
        sensor = _sensor(period=10.0, latency=0.0)
        law = CwMidCourseLaw(
            maneuvers=(ManeuverPlan("departure", 60.0, "planned",
                                    dv_planned_eci_m_s=(0.0, -0.4, 0.0)),),
            reference=ref, t_terminal_s=600.0, terminal_rho_lvlh_m=(0.0, 0.0, 0.0),
            budget_m_s=5.0)
        clock = ClockSpec(
            t0=T0, dt_dynamics_s=5.0, sensors=(sensor.sensor_spec(70),),
            estimator=EstimatorSpec(period_s=10.0, phase_s=0.0, latency_s=0.0),
            controller=ControllerSpec(period_s=10.0, phase_s=0.0, latency_s=0.0))
        mission = mission_to_dict(MultiSpacecraftMission(
            mission_id="zero-lat", title="zero-latency unit flight", frame="GCRS", epoch=T0,
            spacecraft=(SpacecraftSpec("tgt", "TARGET"), SpacecraftSpec("chaser", "CHASER")),
            metadata={"kind": "orbitopt.gnc.campaign-sample-1.0"}))
        filt = RelativeNavEKF(CwProcess(ref.mean_motion), np.array([0.0, 1000.0, 0.0, 0.0, 0.0, 0.0]),
                              np.diag([25.0, 25.0, 25.0, 1e-4, 1e-4, 1e-4]), q_accel_m2_s3=1e-9)
        log = simulate_closed_loop(
            run_id="zero-lat", clock=clock, frame="GCRS", target_id="tgt", chaser_id="chaser",
            chief_r_m=r_c, chief_v_m_s=v_c, deputy_r_m=r_d, deputy_v_m_s=v_d,
            sensors={"relnav": sensor},
            measurement_models={"relnav": Stacked((RelativePosition(), RelativeVelocity()))},
            filter=filt, laws={"chaser": law}, t_end_s=610.0,
            plant=default_plant(frame="GCRS", mu=MU, j2=0.0), mission=mission)
        events = [e for e in log.records["control_event"] if e.executed]
        assert len(events) == 1
        assert events[0].t_exec_s == pytest.approx(60.0, abs=1.0e-9)
        assert float(np.linalg.norm(events[0].realized_delta_v_m_s)) == pytest.approx(0.4, abs=1e-9)
        log.validate()
        assert run_log_from_document(log.to_document(log.digest())).run_id == "zero-lat"


class TestFailureAttribution:
    def _stub(self, monkeypatch, cls, *, tracking_max=1.0, gap=0.0, pos_err=60.0,
              commanded=10.0, budget=25.0):
        from orbitopt.rendezvous import gnc_campaign as gc

        class Outcome:
            failure_class = cls

        class Cfg:
            delta_v_budget_m_s = budget

        outcome = Outcome()
        outcome.position_error_m = pos_err
        assessment = {"tracking": {"max_pos_m": tracking_max},
                      "ledger": {"execution_gap_by_phase_m_s": {"a": gap},
                                 "commanded_total_m_s": commanded}}
        return gc.attribute_failure(outcome, assessment, Cfg())

    def test_success_has_no_attribution(self, monkeypatch):
        assert self._stub(monkeypatch, None) is None

    def test_safety_violation_is_safety_constraint(self, monkeypatch):
        assert self._stub(monkeypatch, "safety_violation") == "safety_constraint"

    def test_navigation_when_the_estimate_was_wrong(self, monkeypatch):
        assert self._stub(monkeypatch, "arrival_accuracy", tracking_max=45.0) == "navigation"

    def test_execution_when_the_actuator_gap_explains_the_miss(self, monkeypatch):
        assert self._stub(monkeypatch, "arrival_accuracy", gap=12.0,
                          pos_err=60.0) == "execution"

    def test_guidance_when_navigation_and_execution_are_clean(self, monkeypatch):
        assert self._stub(monkeypatch, "arrival_accuracy", gap=0.1,
                          pos_err=60.0) == "guidance"

    def test_fuel_over_budget_splits_on_commanded(self, monkeypatch):
        assert self._stub(monkeypatch, "fuel_over_budget", commanded=30.0) == "guidance"
        assert self._stub(monkeypatch, "fuel_over_budget", commanded=20.0,
                          gap=1.0) == "execution"


@pytest.mark.slow
def test_campaign_failure_sample_flies_closed_loop_success():
    """Cluster 1 / index 5 of the frozen E12-10 seed base is an open-loop
    arrival_accuracy FAILURE; flown closed-loop on identical perturbations and
    execution draws it arrives inside the 50 m / 0.2 m/s gate, the realized
    delta-v stays inside the mission budget, and the schedule fires all four
    burns without refusal."""
    from orbitopt.rendezvous import load_reference_config
    from orbitopt.rendezvous import robustness as rb
    from orbitopt.rendezvous.gnc_campaign import fly_closed_loop_sample

    cfg = load_reference_config()
    campaign = dataclasses.replace(rb.load_reference_campaign(),
                                   passive_contingency_in_success=True)
    outcome, assess, log = fly_closed_loop_sample(cfg, campaign, 3416.666, 20000000, 1, 5)
    assert outcome.ok and outcome.failure_class is None, \
        f"closed-loop flight failed: {outcome.failure_class} {outcome.reason}"
    assert outcome.position_error_m < 50.0
    assert outcome.relative_speed_m_s < 0.2
    assert outcome.realized_delta_v_m_s < float(cfg.delta_v_budget_m_s)
    assert assess["attribution"] is None
    assert assess["convergence"]["converged"] is True
    assert assess["law_refusals"] == []
    assert log is not None
    log.validate()
    assert run_log_from_document(log.to_document(log.digest())).run_id == log.run_id
    led = assess["ledger"]
    assert led["commanded_by_phase_m_s"]["departure"] == pytest.approx(4.09, abs=0.05)
    assert led["commanded_by_phase_m_s"]["arrival"] > 1.0
    assert set(led["commanded_by_phase_m_s"]) == {"coast", "departure", "midcourse",
                                                  "terminal_trim", "arrival"}
    assert led["commanded_by_phase_m_s"]["terminal_trim"] > 0.0
