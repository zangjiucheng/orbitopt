"""E13-05 (task #422): constrained control research -- the shared law base
bookkeeping (scheduled terminal pair, arc reference, budget reserve) and the
two arms' decision behaviour, at unit speed.

The velocity profile validates its own geometry and grades violations; the
Riccati sequence is shaped by its arguments; the base law rejects
out-of-order / mis-weighted / mis-phased scheduled burns and an arc
reference that does not end on the hold point; a due burn fires exactly
once, at the first tick at or after its epoch, regardless of the feedback
window; the reserve the feedback sees shrinks as burns fire; and a tick
inside the window on an arc-tracking estimate with closing speed commands a
corrective, saturation-respecting pulse. The benchmark
(benchmarks/bench_e13_05_constrained_control.py) flies the paired campaigns;
nothing here needs a full flight.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from orbitopt.gnc import (
    ApproachConstraints,
    ConstraintSetError,
    CwLqrLaw,
    CwMpcLaw,
    Estimate,
    FiniteHorizonCwLaw,
    GuidanceLawError,
    ManeuverPlan,
    ReferenceEphemeris,
    VelocityProfile,
    cw_impulse_matrices,
    finite_horizon_cw_lqr_gains,
)
from orbitopt.mission.contract import Epoch
from orbitopt.mission.multispacecraft import RelativeState
from orbitopt.relative.frames import eci_to_lvlh, lvlh_to_eci

N = 0.001140
MU = 3.9860044150e14
T0 = Epoch(9000.0, "TDB")
HOLD = np.array([0.0, -1000.0, 0.0])


def _profile():
    return VelocityProfile(range_knots_m=(500.0, 1000.0, 2500.0), cap_m_s=(9.0, 12.0, 26.0))


def _constraints(budget=25.0):
    return ApproachConstraints(keep_out_radius_m=500.0, cross_track_max_m=100.0,
                               cone_cos_half=0.0, cone_range_m=1100.0, velocity=_profile(),
                               u_max_m_s=1.0, budget_m_s=budget,
                               hold_point_rho_norm_m=1000.0)


def _reference():
    r_c = np.array([7000e3, 0.0, 0.0])
    v_c = np.array([0.0, math.sqrt(MU / 7000e3), 0.0])
    tt = np.arange(0.0, 1201.0, 10.0)
    from orbitopt.relative.propagation import propagate_pair
    tr = propagate_pair(r_c, v_c, r_c, v_c, list(tt), frame="GCRS", mu=MU, j2=0.0,
                        where="test e13-05")
    return ReferenceEphemeris(times_s=tuple(float(x) for x in tt),
                              chief_r_m=tuple(tr.chief_r[0, i, :] for i in range(len(tt))),
                              chief_v_m_s=tuple(tr.chief_v[0, i, :] for i in range(len(tt))),
                              frame="GCRS", mu_m3_s2=MU)


def _arc(t_end=1200.0):
    """Straight-line (t, rho, rho_dot) rows from (1000, -1000, 0) to the hold
    point at rest: structure enough to exercise the reference plumbing."""
    rows = []
    for t in np.arange(0.0, t_end + 1.0, 100.0):
        f = t / t_end
        rate = np.array([-1000.0 / t_end, 0.0, 0.0]) if f < 1.0 else np.zeros(3)
        rows.append([t, 1000.0 * (1.0 - f), -1000.0, 0.0, *rate])
    return np.asarray(rows)


def _estimate(now, ref, rho_lvlh, v_lvlh):
    cr, cv = ref.state_at(now)
    r_eci, v_eci = lvlh_to_eci(cr, cv, np.asarray(rho_lvlh, float),
                               np.asarray(v_lvlh, float), "GCRS", where="test e13-05")
    st = RelativeState(target_id="tgt", chaser_id="chaser", epoch=T0, frame="GCRS",
                       r_rel_m=tuple(float(x) for x in r_eci),
                       v_rel_m_s=tuple(float(x) for x in v_eci))
    return Estimate(record_id=f"est-{now:.0f}", estimator_id="ekf", t_update_s=float(now),
                    relative_state=st, covariance_rel_m2=tuple(np.eye(6).ravel()),
                    source_observation_ids=("obs-1",))


def _law(law_cls=FiniteHorizonCwLaw, *, ref=None, burns=(), arc=None, **kw):
    ref = ref if ref is not None else _reference()
    common = dict(reference=ref,
                  planned_burn=ManeuverPlan("departure", 0.0, "planned",
                             dv_planned_eci_m_s=(0.0, 0.0, 1.0e-9)),
                  t_feedback_start_s=10.0, t_terminal_s=1200.0, cadence_s=10.0,
                  terminal_lead_s=100.0, terminal_rho_lvlh_m=HOLD,
                  constraints=kw.pop("constraints", _constraints()),
                  q_diag=[1 / 50.0 ** 2] * 3 + [1 / 0.2 ** 2] * 3, r_diag=[1.0] * 3,
                  q_terminal_diag=[25 / 50.0 ** 2] * 3 + [25 / 0.2 ** 2] * 3,
                  scheduled_burns=burns, trajectory_ref=arc)
    common.update(kw)
    return law_cls(**common)


def _burn(epoch, label="terminal_trim", cap=3.0, vw=0.25):
    return ManeuverPlan(label, epoch, "retarget", dv_cap_m_s=cap, velocity_weight=vw)


def _warm(law, ref):
    """First delta_v call always spends the departure burn (planned zero here);
    do it once so later calls exercise the phase under test."""
    law.delta_v(_estimate(100.0, ref, HOLD, np.zeros(3)))


class TestVelocityProfile:
    def test_rejects_unsorted_knots(self):
        with pytest.raises(ConstraintSetError):
            VelocityProfile(range_knots_m=(1000.0, 500.0), cap_m_s=(9.0, 12.0))

    def test_rejects_misaligned_caps(self):
        with pytest.raises(ConstraintSetError):
            VelocityProfile(range_knots_m=(500.0, 1000.0), cap_m_s=(9.0,))

    def test_rejects_loosening_caps(self):
        with pytest.raises(ConstraintSetError):
            VelocityProfile(range_knots_m=(500.0, 1000.0), cap_m_s=(12.0, 9.0))

    def test_cap_interpolates(self):
        assert _profile().cap(600.0) == pytest.approx(9.6)

    def test_violated_counts(self):
        c = _constraints()
        assert sum(c.violated([0.0, -600.0, 0.0], [0.0, 8.0, 0.0]).values()) == 0
        assert sum(c.violated([0.0, -600.0, 0.0], [0.0, -13.0, 0.0]).values()) == 1


class TestGains:
    def test_shapes(self):
        k, p = finite_horizon_cw_lqr_gains(N, 10.0, [1.0] * 6, [1.0] * 3, [10.0] * 6,
                                           max_steps=8)
        assert k.shape == (9, 3, 6) and p.shape == (9, 6, 6)
        a, b = cw_impulse_matrices(N, 10.0)
        assert a.shape == (6, 6) and b.shape == (6, 3)
        assert np.linalg.norm(k[8]) > 0.0


class TestValidation:
    def test_burns_must_be_sorted(self):
        with pytest.raises(GuidanceLawError, match="firing order"):
            _law(burns=(_burn(600.0, "arrival"), _burn(300.0)))

    def test_burns_must_retarget(self):
        with pytest.raises(GuidanceLawError, match="retarget"):
            _law(burns=(ManeuverPlan("x", 600.0, "planned", dv_planned_eci_m_s=(0.0, 0.0, 1.0)),))

    def test_burns_must_be_after_first_tick(self):
        with pytest.raises(GuidanceLawError, match="after the first feedback tick"):
            _law(burns=(_burn(15.0),))

    def test_burns_need_cap_and_weight(self):
        with pytest.raises(GuidanceLawError, match="cap"):
            _law(burns=(_burn(600.0, cap=0.0),))

    def test_arc_must_end_on_hold(self):
        bad = _arc()
        bad[-1, 1] = -800.0
        with pytest.raises(GuidanceLawError, match="hold"):
            _law(arc=bad)

    def test_arc_needs_three_rows(self):
        with pytest.raises(GuidanceLawError):
            _law(arc=np.array([[0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [100.0, 0.0, -1000.0, 0.0, 0.0, 0.0, 0.0]]))


class TestReserveAndFiring:
    def test_reserve_shrinks_when_burn_fires(self):
        ref = _reference()
        law = _law(ref=ref, burns=(_burn(1100.0, cap=3.0),
                                   _burn(1180.0, "arrival", 8.0, 100.0)))
        assert law.budget_available_m_s == pytest.approx(25.0 - 11.0)
        _warm(law, ref)
        est = _estimate(1110.0, ref, HOLD + np.array([0.0, -60.0, 0.0]), np.zeros(3))
        dv = law.delta_v(est)
        spent = math.hypot(*dv)
        assert spent > 0.05 and law.phase_for(est) == "terminal_trim"
        assert law.budget_available_m_s == pytest.approx(25.0 - spent - 8.0)

    def test_due_burn_fires_outside_window(self):
        ref = _reference()
        law = _law(ref=ref, burns=(_burn(1100.0, cap=3.0),))
        _warm(law, ref)
        est = _estimate(1195.0, ref, HOLD + np.array([0.0, -50.0, 0.0]), np.zeros(3))
        law.delta_v(est)
        assert law.phase_for(est) == "terminal_trim"  # window closed at 1100 already

    def test_coast_after_all_burns_fired(self):
        ref = _reference()
        law = _law(ref=ref, burns=(_burn(1100.0, cap=3.0),))
        _warm(law, ref)
        law.delta_v(_estimate(1110.0, ref, HOLD + np.array([2.0, 0.0, 0.0]), np.zeros(3)))
        est = _estimate(1150.0, ref, HOLD, np.zeros(3))
        assert law.delta_v(est) == (0.0, 0.0, 0.0)
        assert law.phase_for(est) == "coast"

    def test_feedback_pulse_on_arc_brakes_closing_speed(self):
        ref = _reference()
        arc = _arc()
        law = _law(CwLqrLaw, ref=ref, arc=arc)
        _warm(law, ref)
        est = _estimate(600.0, ref, arc[6, 1:4] + np.array([10.0, 0.0, 0.0]),
                        np.array([0.0, -6.0, 0.0]))
        dv = law.delta_v(est)
        mag = math.hypot(*dv)
        assert law.phase_for(est) == "feedback"
        assert 0.002 < mag <= 1.0 + 1e-9
        cr, cv = ref.state_at(600.0)
        _rho, u_lvlh = eci_to_lvlh(cr, cv, np.zeros(3), np.asarray(dv), "GCRS",
                                   where="test e13-05")
        assert u_lvlh[1] > 0.0  # brakes the -6 m/s closing error (terminal weight >> position)


class TestBudgetLedger:
    def test_reserve_is_not_spent_by_feedback(self):
        ref = _reference()
        law = _law(CwLqrLaw, ref=ref, constraints=_constraints(budget=6.0),
                   burns=(_burn(1100.0, cap=3.0),))
        for k in range(30):
            law.delta_v(_estimate(20.0 + 10.0 * k, ref,
                                  np.array([900.0 - 10.0 * k, -1000.0, 0.0]), np.zeros(3)))
        assert law.commanded_total_m_s <= 6.0 - 3.0 + 1e-9


class TestMpcLaw:
    def test_u_max_and_solve_time_record(self):
        ref = _reference()
        law = _law(CwMpcLaw, ref=ref, arc=_arc(), horizon_steps=6)
        _warm(law, ref)
        est = _estimate(600.0, ref, _arc()[6, 1:4] + np.array([20.0, 0.0, -30.0]),
                        np.array([0.0, -8.0, 0.0]))
        dv = law.delta_v(est)
        assert law.phase_for(est) == "feedback"
        assert math.hypot(*dv) <= 1.0 + 1e-9
        assert law.solve_time_stats["n_decisions"] == 1

    def test_horizon_validation(self):
        with pytest.raises(GuidanceLawError):
            _law(CwMpcLaw, horizon_steps=0)
