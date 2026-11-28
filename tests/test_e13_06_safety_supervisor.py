"""E13-06 (task #423): the safety supervisor -- covariance-inflated E12-04
monitors, the nominal -> hold -> abort state machine, and the truth-scoring
reader (missed alarms and false alarms counted separately).

Unit layer: the envelope copies an E12-04 ``SafetyCfg`` value-for-value and
its speed cap reproduces ``approach_speed_cap_m_s`` exactly; monitor margins
match the E12-04 formulas and the sigmas match the delta-method value built
from the LVLH-rotated covariance (the estimate is published inertial, so the
frame round trip has to be exact); a constructed inbound walk suppresses the
hold (its coast is the violation) and aborts directly, with the inner law's
thrust frozen from the abort tick; recovery returns to nominal; an
un-clearing hold times out; the scorer separates missed alarms from false
alarms, including against
``SafetyCheck``-shaped input from ``evaluate_trajectory_safety`` itself.
Campaign layer (slow): the E13-04 frozen failure sample flies bit-identical
with and without the supervisor (specificity -- a clean flight must not be
touched); a declared navigation fault (line-of-sight shrink after t = 60 s)
aborts the supervisor on a truth run that never violates anything -- every
alert interval a false alarm, zero missed -- and the post-abort truth coast
stays outside the keep-out sphere; a constructed collision-course insertion
drives
the truth inside the keep-out sphere and the supervisor's alert
precedes the violation (zero missed). The full campaign statistics live in
benchmarks/bench_e13_06_safety_supervisor.py."""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from orbitopt.gnc import (
    ChannelErrorConfig,
    ControllerSpec,
    DropoutConfig,
    EstimatorSpec,
    GNCContractError,
    RelativeNavEKF,
    RelativePosition,
    RelativeSensorModel,
    RelativeVelocity,
    Stacked,
)
from orbitopt.gnc.clock import ClockSpec, SensorTick
from orbitopt.gnc.closed_loop import simulate_closed_loop
from orbitopt.gnc.contracts import Estimate
from orbitopt.gnc.ekf import CwProcess, frame_covariance
from orbitopt.gnc.guidance import ReferenceEphemeris
from orbitopt.gnc.supervisor import (
    INSTANTANEOUS_MONITORS,
    STATE_ABORT,
    STATE_HOLD,
    STATE_NOMINAL,
    MonitorReading,
    SafetyEnvelope,
    SafetySupervisor,
    SupervisorConfig,
    SupervisionEvent,
    alert_intervals,
    score_supervision,
    violation_intervals,
)
from orbitopt.mission.contract import Epoch
from orbitopt.mission.multispacecraft import (
    MultiSpacecraftMission,
    RelativeState,
    SpacecraftSpec,
    mission_to_dict,
)
from orbitopt.relative.frames import eci_to_lvlh, lvlh_to_eci
from orbitopt.relative.propagation import propagate_pair

MU = 3.9860044150e14
T0 = Epoch(9000.0, "TDB")
FRAME = "GCRS"
SRC = "DESIGN ASSUMPTION (E13-06): unit test supervisor profile"

ENV_KW = dict(keep_out_radius_m=500.0, cross_track_max_m=100.0, cone_half_angle_deg=90.0,
              cone_range_m=1100.0, speed_range_knots_m=(500.0, 2500.0, 9500.0, 13000.0),
              speed_cap_m_s=(9.0, 12.0, 22.0, 26.0), passive_contingency_s=5400.0)


@pytest.fixture(scope="module")
def reference():
    """A 7000 km circular chief over two orbits -- enough table for every
    tick the unit tests use."""
    r0 = np.array([7_000_000.0, 0.0, 0.0])
    v0 = np.array([0.0, math.sqrt(MU / 7_000_000.0), 0.0])
    tt = list(np.arange(0.0, 11_000.0, 60.0))
    tr = propagate_pair(r0, v0, r0, v0, tt, frame=FRAME, mu=MU, j2=0.0, where="e13-06 test ref")
    return ReferenceEphemeris(times_s=tuple(float(x) for x in tt),
                              chief_r_m=tuple(tr.chief_r[0, i, :] for i in range(len(tt))),
                              chief_v_m_s=tuple(tr.chief_v[0, i, :] for i in range(len(tt))),
                              frame=FRAME, mu_m3_s2=MU)


class RecordingLaw:
    """Inner law stand-in: records every tick it was allowed to command."""

    def __init__(self):
        self.calls = 0

    def delta_v(self, estimate):
        self.calls += 1
        return (1.0, 0.0, 0.0)

    def phase_for(self, estimate):
        return "coast"


def _estimate(reference, t_s, rho_lvlh, rhod_lvlh, p_lvlh=None):
    """An ``Estimate`` carrying a LVLH geometry expressed the way the EKF
    publishes it: inertial relative state, inertial covariance."""
    p = (np.diag([4.0, 4.0, 4.0, 9e-6, 9e-6, 9e-6]) if p_lvlh is None else np.asarray(p_lvlh))
    cr, cv = reference.state_at(t_s)
    dr, dv = lvlh_to_eci(cr, cv, rho_lvlh, rhod_lvlh, FRAME, where="e13-06 test estimate")
    pi = frame_covariance(p, cr, cv, FRAME)
    return Estimate(record_id=f"est-{t_s:08.1f}", estimator_id="filt", t_update_s=float(t_s),
                    relative_state=RelativeState(target_id="tgt", chaser_id="chaser",
                                                 epoch=T0, frame=FRAME,
                                                 r_rel_m=tuple(float(x) for x in dr),
                                                 v_rel_m_s=tuple(float(x) for x in dv)),
                    covariance_rel_m2=tuple(float(x) for x in pi.reshape(36)),
                    source_observation_ids=("obs-1",))


class TestEnvelope:
    def test_from_safety_cfg_copies_values_and_speed_cap_matches_e12_04(self):
        from orbitopt.rendezvous import load_reference_config
        from orbitopt.rendezvous.safety import approach_speed_cap_m_s
        cfg = load_reference_config()
        env = SafetyEnvelope.from_safety_cfg(cfg.safety)
        assert env.keep_out_radius_m == cfg.safety.keep_out_radius_m
        assert env.cross_track_max_m == cfg.safety.cross_track_max_m
        assert env.cone_half_angle_deg == cfg.safety.cone_half_angle_deg
        assert env.cone_range_m == cfg.safety.cone_range_m
        assert env.passive_contingency_s == cfg.safety.passive_contingency_s
        rng = np.array([100.0, 800.0, 1700.0, 4000.0, 8200.0, 12000.0, 20000.0])
        assert np.allclose(env.approach_speed_cap_m_s(rng),
                           approach_speed_cap_m_s(cfg.safety, rng))

    def test_envelope_rejects_nonsense(self):
        with pytest.raises(GNCContractError):
            SafetyEnvelope(**{**ENV_KW, "keep_out_radius_m": 0.0})
        with pytest.raises(GNCContractError):
            SafetyEnvelope(**{**ENV_KW, "cone_half_angle_deg": 0.0})
        with pytest.raises(GNCContractError):
            SafetyEnvelope(**{**ENV_KW, "speed_range_knots_m": (100.0, 10.0),
                              "speed_cap_m_s": (1.0, 2.0)})

    def test_supervisor_config_rejects_nonsense(self):
        with pytest.raises(GNCContractError):
            SupervisorConfig(k_sigma=0.5)
        with pytest.raises(GNCContractError):
            SupervisorConfig(trip_ticks=0)
        with pytest.raises(GNCContractError):
            SupervisorConfig(hold_max_s=-1.0)


class TestMonitors:
    def test_far_and_quiet_margins_match_e12_04_formulas(self, reference):
        sup = SafetySupervisor(law=RecordingLaw(), reference=reference,
                               envelope=SafetyEnvelope(**ENV_KW))
        rho = np.array([0.0, -8000.0, 0.0])
        rhod = np.array([0.0, 0.5, 0.0])
        r = sup.monitor(_estimate(reference, 300.0, rho, rhod))
        assert r["keep_out"].active and not r["terminal_cone"].active
        assert r["keep_out"].margin == pytest.approx(7500.0, abs=0.5)
        assert r["keep_out"].lower == pytest.approx(7500.0 - 3.0 * 2.0, abs=0.5)
        assert r["cross_track"].margin == pytest.approx(100.0, abs=0.5)
        assert r["approach_speed"].margin == pytest.approx(
            float(np.interp(8000.0, ENV_KW["speed_range_knots_m"], ENV_KW["speed_cap_m_s"])) - 0.5,
            abs=0.5)
        assert all(r[name].lower > r[name].warn for name in INSTANTANEOUS_MONITORS
                   if r[name].active)
        assert not r["passive_coast"].warned

    def test_sigma_is_the_delta_method_on_the_rotated_covariance(self, reference):
        p_lvlh = np.diag([3.0 ** 2, 2.0 ** 2, 1.0 ** 2, 1e-3 ** 2, 2e-3 ** 2, 1e-3 ** 2])
        sup = SafetySupervisor(law=RecordingLaw(), reference=reference,
                               envelope=SafetyEnvelope(**ENV_KW))
        est = _estimate(reference, 600.0, np.array([300.0, -1500.0, 40.0]),
                        np.array([0.1, -2.0, 0.2]), p_lvlh=p_lvlh)
        x, p = sup.lvlh_state_cov(est)
        assert np.allclose(x[:3], [300.0, -1500.0, 40.0], atol=1e-5)
        assert np.allclose(x[3:], [0.1, -2.0, 0.2], atol=1e-8)
        assert np.allclose(p, p_lvlh, rtol=1e-8, atol=1e-8)  # frame round trip closes
        g = np.zeros(6)
        g[:3] = x[:3] / np.linalg.norm(x[:3])
        r = sup.monitor(est)
        assert r["keep_out"].sigma == pytest.approx(math.sqrt(g @ p @ g), rel=1e-6)

    def test_frame_round_trip_of_published_covariance(self, reference):
        p_lvlh = np.diag([9.0, 4.0, 1.0, 4e-6, 9e-6, 1e-6])
        est = _estimate(reference, 900.0, np.array([-200.0, -4000.0, -30.0]),
                        np.array([0.05, 1.0, -0.01]), p_lvlh=p_lvlh)
        sup = SafetySupervisor(law=RecordingLaw(), reference=reference,
                               envelope=SafetyEnvelope(**ENV_KW))
        _x, p = sup.lvlh_state_cov(est)
        assert np.allclose(p, p_lvlh, rtol=1e-8, atol=1e-8)

    def test_cone_engages_inside_the_cone_range_like_e12_04(self, reference):
        sup = SafetySupervisor(law=RecordingLaw(), reference=reference,
                               envelope=SafetyEnvelope(**ENV_KW))
        r = sup.monitor(_estimate(reference, 300.0, np.array([0.0, -900.0, 0.0]),
                                  np.array([0.0, 0.0, 0.0])))
        assert r["terminal_cone"].active
        assert r["terminal_cone"].margin == pytest.approx(90.0, abs=0.1)  # dead on -V-bar
        r2 = sup.monitor(_estimate(reference, 300.0, np.array([0.0, 900.0, 0.0]),
                                   np.array([0.0, 0.0, 0.0])))
        assert r2["terminal_cone"].margin == pytest.approx(-90.0, abs=0.1)  # dead ahead

    def test_passive_coast_predicts_the_keepout_hit(self, reference):
        sup = SafetySupervisor(law=RecordingLaw(), reference=reference,
                               envelope=SafetyEnvelope(**ENV_KW))
        r = sup.monitor(_estimate(reference, 100.0, np.array([0.0, -900.0, 0.0]),
                                  np.array([0.0, 3.0, 0.0])))
        assert r["passive_coast"].lower < 0.0 and r["passive_coast"].warned
        # static at 3 km: the CW coast holds range; the only risk is the
        # covariance growth over the horizon, and 3 km of margin swallows it
        r2 = sup.monitor(_estimate(reference, 100.0, np.array([0.0, -3000.0, 0.0]),
                                   np.array([0.0, 0.0, 0.0])))
        assert r2["passive_coast"].lower > 200.0 and not r2["passive_coast"].warned

    def test_estimate_identity_is_enforced(self, reference):
        sup = SafetySupervisor(law=RecordingLaw(), reference=reference,
                               envelope=SafetyEnvelope(**ENV_KW))
        with pytest.raises(GNCContractError):
            sup.monitor("not an estimate")


class TestStateMachine:
    def test_inbound_walk_suppresses_hold_then_aborts_and_freezes_thrust(self, reference):
        # Closing straight at the chief from 900 m: the coast from every tick
        # is predicted inside the keep-out sphere (passive lower < 0), so a
        # hold would commit TO the violation -- the supervisor suppresses the
        # hold, keeps the law in command, and goes straight to abort when an
        # instantaneous lower bound crosses zero.
        inner = RecordingLaw()
        sup = SafetySupervisor(law=inner, reference=reference,
                               envelope=SafetyEnvelope(**ENV_KW))
        seen = []
        for k in range(30):
            t = 100.0 + 10.0 * k
            rho = np.array([0.0, -900.0 + 3.0 * (t - 100.0), 0.0])
            dv = sup.delta_v(_estimate(reference, t, rho, np.array([0.0, 3.0, 0.0])))
            seen.append((t, sup.state, dv))
        assert STATE_HOLD not in {s for _t, s, _d in seen}
        order = [e.to_dict() for e in sup.events]
        assert order[0]["state_to"] == STATE_NOMINAL
        assert order[0]["trigger"] == "hold_suppressed_passive_unsafe"
        # first real alert: the direct abort the acceptance criterion allows
        # as hold/abort once an instantaneous lower bound crosses zero
        assert (order[1]["state_from"], order[1]["state_to"]) == (STATE_NOMINAL, STATE_ABORT)
        assert order[1]["trigger"] == "keep_out"
        t_abort = order[1]["t_s"]
        assert all(d == (0.0, 0.0, 0.0) for t, _s, d in seen if t >= t_abort - 1e-9)
        assert inner.calls >= 2  # law in command during the suppressed warnings
        n_after = sum(1 for t, _s, _d in seen if t >= t_abort - 1e-9)
        assert inner.calls == len(seen) - n_after  # ... and frozen after the abort

    def test_recovery_returns_to_nominal_and_releases_the_law(self, reference):
        # hold on a cross-track warning -- containment, not a doomed coast,
        # so the state can legally come back once the geometry clears (the
        # hold entry itself requires a passive-green coast). Tiny q and a
        # tight covariance keep the passive-coast lower above zero (a real
        # 3e-8 coast uncertainty at 1 km would NOT be recoverable: see
        # test_passive_coast_predicts_the_keepout_hit). The 30 m band is
        # pinned here, not by the default (the mission's own cross-track
        # design margin is ~30 m, so the default band is smaller).
        tight = np.diag([0.2 ** 2] * 3 + [1e-8] * 3)
        cfg = SupervisorConfig(trip_ticks=1, q_accel_m2_s3=1e-13, warn_cross_track_m=30.0)
        inner = RecordingLaw()
        sup = SafetySupervisor(law=inner, reference=reference,
                               envelope=SafetyEnvelope(**ENV_KW), config=cfg)
        states, calls_at = [], []
        for k in range(24):
            t = 100.0 + 10.0 * k
            if k < 8:  # static at z = +75 m: margin 25 < warn 30 -> held
                rho, rhod = np.array([0.0, -1500.0, 75.0]), np.zeros(3)
            else:      # correction burns done: quiet station-keeping point
                rho, rhod = np.array([0.0, -8000.0, 0.0]), np.zeros(3)
            dv = sup.delta_v(_estimate(reference, t, rho, rhod, p_lvlh=tight))
            states.append(sup.state)
            calls_at.append((t, inner.calls))
            if sup.state == STATE_HOLD:
                assert dv == (0.0, 0.0, 0.0)  # held means held
        assert states[0] == STATE_HOLD
        assert sup.state == STATE_NOMINAL and states.count(STATE_HOLD) < 12
        assert sup.events[-1].trigger == "recovery"
        assert calls_at[-1][1] > calls_at[1][1]  # the inner law commands again after recovery

    def test_unclearing_hold_times_out_to_abort(self, reference):
        # Same cross-track warning, but the geometry never clears: only the
        # hold timeout can end it (passive coast stays clean, lower stays
        # positive, so no trip fires).
        tight = np.diag([0.2 ** 2] * 3 + [1e-8] * 3)
        sup = SafetySupervisor(law=RecordingLaw(), reference=reference,
                               envelope=SafetyEnvelope(**ENV_KW),
                               config=SupervisorConfig(trip_ticks=1, q_accel_m2_s3=1e-13,
                                                       warn_cross_track_m=30.0))
        est = lambda t: _estimate(reference, t, np.array([0.0, -1500.0, 75.0]),  # noqa: E731
                                  np.zeros(3), p_lvlh=tight)
        sup.step(est(100.0))
        assert sup.state == STATE_HOLD
        sup.step(est(200.0))
        assert sup.state == STATE_HOLD  # 100 s held: inside the 180 s budget
        sup.step(est(300.0))
        assert sup.state == STATE_ABORT and sup.events[-1].trigger == "hold_timeout"
        assert sup.delta_v(est(400.0)) == (0.0, 0.0, 0.0)


class TestScoring:
    @staticmethod
    def _series(violated_from, t1, dt=10.0):
        t = np.arange(0.0, 1200.0, dt)
        m = np.where((t >= violated_from) & (t <= t1), -50.0, 20.0)
        return {"times_s": t, "margins": m}

    def test_clean_scoring_counts_nothing(self):
        score = score_supervision([], {"keep_out": self._series(9e9, 0.0)}, t_end_s=1200.0)
        assert score["counts"] == {"alerts": 0, "violation_intervals": 0, "missed": 0,
                                   "false_alarms": 0}

    def test_violation_with_early_alert_is_detected_not_false(self):
        ev = [SupervisionEvent(300.0, STATE_NOMINAL, STATE_HOLD, "keep_out", 150.0, "warn"),
              SupervisionEvent(600.0, STATE_HOLD, STATE_ABORT, "passive_coast", 20.0, "doomed")]
        score = score_supervision(ev, {"keep_out": self._series(350.0, 600.0)}, t_end_s=1200.0)
        assert score["counts"]["missed"] == 0 and score["counts"]["false_alarms"] == 0
        assert score["alert_intervals"][0]["start_s"] == 300.0
        assert score["alert_intervals"][0]["end_s"] == 1200.0  # still aborted at run end

    def test_late_alert_is_a_miss_and_early_is_not(self):
        late = [SupervisionEvent(900.0, STATE_NOMINAL, STATE_HOLD, "keep_out", -20.0, "late")]
        score = score_supervision(late, {"keep_out": self._series(350.0, 600.0)}, t_end_s=1200.0)
        assert score["counts"]["missed"] == 1
        assert score["checks"]["keep_out"]["detected"] == [False]

    def test_alert_with_no_violation_anywhere_is_a_false_alarm(self):
        ev = [SupervisionEvent(300.0, STATE_NOMINAL, STATE_HOLD, "cross_track", 5.0, "warn")]
        score = score_supervision(ev, {"keep_out": self._series(0, 0, 9e9)}, t_end_s=1200.0)
        assert score["counts"]["false_alarms"] == 1
        assert score["false_alarms"][0]["trigger"] == "cross_track"

    def test_millions_of_flickering_samples_merge_into_one_window(self):
        t = np.arange(0.0, 500.0, 10.0)
        m = np.where((np.abs(t - 100.0) < 15) | (np.abs(t - 300.0) < 15), -1.0, 1.0)
        assert violation_intervals(t, m, merge_gap_s=0.0) == [[90.0, 110.0], [290.0, 310.0]]
        assert violation_intervals(t, m, merge_gap_s=250.0) == [[90.0, 310.0]]

    def test_scorer_consumes_e12_04_safety_checks_directly(self):
        from orbitopt.rendezvous import load_reference_config
        from orbitopt.rendezvous.safety import evaluate_trajectory_safety
        cfg = load_reference_config()
        t = np.arange(0.0, 600.0, 10.0)
        rho = np.zeros((t.size, 3))
        rho[:, 1] = 2000.0 - 4.0 * t  # straight-line V-bar closure 2 km -> -400 m
        rhod = np.tile(np.array([0.0, -4.0, 0.0]), (t.size, 1))
        checks = evaluate_trajectory_safety(cfg, t, rho, rhod)
        ev = [SupervisionEvent(100.0, STATE_NOMINAL, STATE_HOLD, "passive_coast", 150.0, "w"),
              SupervisionEvent(250.0, STATE_HOLD, STATE_ABORT, "keep_out", 20.0, "d")]
        score = score_supervision(ev, checks, t_end_s=float(t[-1]))
        assert checks["keep_out"].violation_count > 0
        assert score["counts"]["missed"] == 0 and score["counts"]["false_alarms"] == 0

    def test_alert_intervals_merge_over_hold_to_abort(self):
        ev = [SupervisionEvent(100.0, STATE_NOMINAL, STATE_HOLD, "a", 0.0, ""),
              SupervisionEvent(200.0, STATE_HOLD, STATE_ABORT, "b", 0.0, ""),
              SupervisionEvent(400.0, STATE_ABORT, STATE_NOMINAL, "c", 0.0, "")]
        assert alert_intervals(ev, 500.0) == [{"start_s": 100.0, "end_s": 400.0, "trigger": "b"}]
        # the escalation trigger (hold -> abort) names the merged window


# ---------------------------------------------------------------------------
# campaign layer: the three acceptance quantities on a real campaign sample
# ---------------------------------------------------------------------------

def _campaign_sample_args():
    from orbitopt.rendezvous import load_reference_config
    from orbitopt.rendezvous import robustness as rb
    cfg = load_reference_config()
    campaign = dataclasses.replace(rb.load_reference_campaign(),
                                   passive_contingency_in_success=True)
    return cfg, campaign, 3416.666, 20000000, 1, 5


def _los_shrink_hook(d_m: float, t_onset_s: float):
    """Navigation fault: every measurement after ``t_onset_s`` reports the
    target ``d_m`` closer along the line of sight (an inertial LOS shrink, so
    it needs no frame knowledge). The plant and the log keep the truth."""
    def hook(obs):
        if obs.t_sample_s < t_onset_s:
            return obs
        r = np.asarray(obs.relative_state.r_rel_m, dtype=float)
        rng = float(np.linalg.norm(r))
        shrunk = r * max(rng - d_m, 10.0) / rng
        rs = dataclasses.replace(obs.relative_state, r_rel_m=tuple(float(x) for x in shrunk))
        return dataclasses.replace(obs, relative_state=rs)
    return hook


@pytest.mark.slow
def test_clean_sample_is_untouched_by_the_supervisor():
    """Specificity: the E13-04 frozen failure sample, closed-loop clean, must
    fly bit-identical with the supervisor attached -- nominal throughout,
    zero events, zero alerts, and the same outcome the unsupervised run
    produced."""
    from orbitopt.rendezvous.gnc_campaign import fly_closed_loop_sample
    cfg, campaign, tof, sb, c, i = _campaign_sample_args()
    base, _a0, _l0 = fly_closed_loop_sample(cfg, campaign, tof, sb, c, i)
    outcome, assess, log = fly_closed_loop_sample(cfg, campaign, tof, sb, c, i,
                                                  supervisor=SupervisorConfig())
    sup_rec = assess["supervision"]
    assert sup_rec["final_state"] == STATE_NOMINAL
    assert sup_rec["events"] == []
    assert sup_rec["score"]["counts"] == {"alerts": 0, "violation_intervals": 0, "missed": 0,
                                          "false_alarms": 0}
    assert sup_rec["post_abort"]["aborted"] is False
    assert outcome.failure_class == base.failure_class
    assert outcome.position_error_m == pytest.approx(base.position_error_m, abs=0.01)
    assert outcome.realized_delta_v_m_s == pytest.approx(base.realized_delta_v_m_s, abs=1e-9)
    assert log is not None


@pytest.mark.slow
def test_nav_fault_false_alarm_and_passive_safe_post_abort():
    """A declared navigation fault (measurements shrink 7.5 km along the LOS
    from t = 60 s, during the drift leg) drives the supervisor to abort
    although the truth never violates anything: the scoring must count every
    alert interval as a false alarm, zero missed, zero truth violations --
    and the post-abort trajectory is the burn-free drift coast, which the
    independent truth re-check must certify outside the keep-out sphere."""
    from orbitopt.rendezvous.gnc_campaign import fly_closed_loop_sample
    cfg, campaign, tof, sb, c, i = _campaign_sample_args()
    outcome, assess, log = fly_closed_loop_sample(
        cfg, campaign, tof, sb, c, i, supervisor=SupervisorConfig(),
        observation_hook=_los_shrink_hook(7500.0, 60.0))
    rec = assess["supervision"]
    assert rec["final_state"] == STATE_ABORT
    counts = rec["score"]["counts"]
    assert counts["missed"] == 0
    # every alert is a false alarm on a truth run that never violated; the
    # fault's filter transient can cycle a hold (passive-unsafe resume), so
    # the count is structural, not a hard-coded one
    assert counts["alerts"] >= 1
    assert counts["false_alarms"] == counts["alerts"]
    assert not any(v["detected"] for v in rec["score"]["violations"])
    assert all(not chk["violated"] for chk in rec["score"]["checks"].values())
    post = rec["post_abort"]
    assert post["aborted"] and post["keep_out_passed"]
    assert post["window_s"] > 1800.0
    assert post["keep_out_min_margin_m"] > 0.0
    assert log is not None


# ---------------------------------------------------------------------------
# negative test: a constructed perturbation that enters the keep-out sphere
# must trigger hold/abort, and the alert must precede the violation
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_collision_course_insertion_is_alerted_before_crossing():
    """Constructed negative: a chaser inserted 1.2 km radially BELOW the
    target and rising at 6 m/s -- an insertion error on a collision course
    the guidance never models (the law here commands nothing at all) and the
    filter's process noise never expects. CW drift carries the closure
    through the keep-out sphere at t ~ 190 s; the supervisor's alert must
    precede the violation (zero missed)."""
    from orbitopt.rendezvous import load_reference_config
    from orbitopt.rendezvous.safety import evaluate_trajectory_safety
    from orbitopt.rendezvous.gnc_campaign import _sensor, ClosedLoopSampleConfig
    from orbitopt.gnc.guidance import ReferenceEphemeris

    cfg = load_reference_config()
    clc = ClosedLoopSampleConfig()
    frame = cfg.dynamics.frame
    t_end = 600.0
    om = math.sqrt(MU / 7_000_000.0 ** 3)
    r0c = np.array([7_000_000.0, 0.0, 0.0])
    v0c = np.array([0.0, om * 7_000_000.0, 0.0])
    r0d = np.array([7_000_000.0 - 1200.0, 0.0, 0.0])   # 1.2 km radially below
    v0d = np.array([6.0, om * 7_000_000.0, 0.0])      # rising at 6 m/s
    sensor, sensor_spec = _sensor(clc, 11, math.ceil(t_end / clc.sensor_period_s) + 2)
    clock = ClockSpec(
        t0=T0, dt_dynamics_s=10.0, sensors=(sensor_spec,),
        estimator=EstimatorSpec(period_s=10.0, phase_s=0.0, latency_s=0.0),
        controller=ControllerSpec(period_s=10.0, phase_s=0.0, latency_s=0.0))
    tt = list(np.arange(0.0, t_end + 60.0, 60.0))
    tr = propagate_pair(r0c, v0c, r0c, v0c, tt, frame=frame, mu=MU, j2=1.08262668e-3,
                        where="e13-06 negative ref")
    ref = ReferenceEphemeris(times_s=tuple(float(x) for x in tt),
                             chief_r_m=tuple(tr.chief_r[0, i, :] for i in range(len(tt))),
                             chief_v_m_s=tuple(tr.chief_v[0, i, :] for i in range(len(tt))),
                             frame=frame, mu_m3_s2=MU)
    filt = RelativeNavEKF(CwProcess(ref.mean_motion),
                          np.array([-1200.0, 0.0, 0.0, 6.0, 0.0, 0.0]),
                          np.diag([10.0 ** 2] * 3 + [0.01 ** 2] * 3),
                          q_accel_m2_s3=clc.q_accel_m2_s3)

    class NoThrust:
        def delta_v(self, estimate):
            return (0.0, 0.0, 0.0)

        def phase_for(self, estimate):
            return "coast"

    sup = SafetySupervisor(law=NoThrust(), reference=ref,
                           envelope=SafetyEnvelope.from_safety_cfg(cfg.safety),
                           config=SupervisorConfig(), target_id="tgt", chaser_id="chaser")
    mission = mission_to_dict(MultiSpacecraftMission(
        mission_id="e13-06-negative", title="E13-06 negative test", frame=frame, epoch=T0,
        spacecraft=(SpacecraftSpec("tgt", "TARGET"), SpacecraftSpec("chaser", "CHASER")),
        metadata={"kind": "orbitopt.gnc.campaign-sample-1.0"}))
    log = simulate_closed_loop(
        run_id="e13-06-kick", clock=clock, frame=frame, target_id="tgt", chaser_id="chaser",
        chief_r_m=r0c, chief_v_m_s=v0c, deputy_r_m=r0d, deputy_v_m_s=v0d,
        sensors={"relnav": sensor},
        measurement_models={"relnav": Stacked((RelativePosition(), RelativeVelocity()))},
        filter=filt, laws={"chaser": sup}, t_end_s=t_end,
        mission=mission, config_extra={})

    # truth samples for the independent E12-04 re-check (from the log only)
    chaser = [r for r in log.records["truth"] if r.spacecraft_id == "chaser"]
    tgt = {r.t_s: r for r in log.records["truth"] if r.spacecraft_id == "tgt"}
    rows = []
    for r in chaser:
        if r.t_s not in tgt:
            continue
        rho, rhod = eci_to_lvlh(np.asarray(tgt[r.t_s].r_m), np.asarray(tgt[r.t_s].v_m_s),
                                np.asarray(r.r_m) - np.asarray(tgt[r.t_s].r_m),
                                np.asarray(r.v_m_s) - np.asarray(tgt[r.t_s].v_m_s),
                                frame, where="e13-06 test kick")
        rows.append((r.t_s, rho, rhod))
    t = np.array([x[0] for x in rows])
    rel = np.hstack((np.vstack([x[1] for x in rows]), np.vstack([x[2] for x in rows])))
    checks = evaluate_trajectory_safety(cfg, t, rel[:, :3], rel[:, 3:])
    ko = checks["keep_out"]
    assert ko.violation_count > 0, "the constructed perturbation must cross the keep-out sphere"
    assert sup.state == STATE_ABORT
    score = score_supervision(sup.events, checks, t_end_s=float(t[-1]))
    assert score["counts"]["missed"] == 0, (
        f"supervisor missed the crossing: {score['violations']} vs alerts "
        f"{score['alert_intervals']}")
    trig = [e for e in sup.events if e.state_to != STATE_NOMINAL]
    # With a collision-course coast the hold is passive-unsafe and therefore
    # suppressed -- the direct hold/abort the acceptance criterion allows is
    # the hard trip here.
    assert trig[0].state_to in (STATE_HOLD, STATE_ABORT)
    assert trig[0].t_s < float(ko.worst_epoch_s)  # the alert precedes the violation
