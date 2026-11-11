"""FL-403 (#208): JSBSim 6-DOF PID gimbal tracking of the 3-DOF ascent program.

Acceptance thresholds are declared up front (module constants below). The
closed-loop tests need the recorded 3-DOF reference fixture and ``jsbsim``
(opt-in pip dep: ``pip install -r envs/sixdof-requirements.txt``); both
skip cleanly when missing.

Declared GO criteria (ADR docs/adr/flightlab-0001-jsbsim-6dof.md):
* settled attitude tracking (t >= 60 s, after the program-shaping window):
  max <= 2 deg, RMS <= 0.6 deg -- the early-window error is the documented
  program-infeasibility transient near vertical, reported but not gating;
* gimbal saturation 0 % of the burn, command never beyond travel limits;
* stable integration: dt 0.01 vs 0.005 agree, no non-finite states;
* MECO state deviation vs the 3-DOF reference within the (generous,
  transient-inclusive) bounds below.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

pytest.importorskip("jsbsim")

FX = os.path.join(os.path.dirname(__file__), "fixtures", "flightlab", "opentsiolkovsky",
                  "fl403_spike")
pytestmark = pytest.mark.skipif(
    not os.path.isfile(os.path.join(FX, "ot_out", "trajectory.csv")),
    reason="FL-403 3-DOF reference fixture not recorded "
           "(scripts/fl403_record_reference.py)")

from orbitopt.flightlab.adapters.opentsiolkovsky import AttitudePoint, Vehicle  # noqa: E402
from orbitopt.flightlab.sixdof.tracking import (  # noqa: E402
    AttitudeProgram,
    ReferenceTrajectory,
    chart_attitude,
    chart_error_deg,
    attitude_error_body,
    run_tracking_spike,
    so3_log,
)

SETTLE_T_S = 60.0
SETTLED_MAX_DEG = 2.0
SETTLED_RMS_DEG = 0.6
FULL_MAX_DEG = 30.0            # transient-inclusive, program-shaping window
D_ALT_TOL_M = 6000.0           # transient-inclusive MECO deviation bounds
D_SPEED_TOL_M_S = 200.0
D_FPA_TOL_DEG = 1.0
DT_CONVERGE_ALT_M = 300.0      # dt 0.01 vs 0.005 altitude agreement


def _fixture():
    veh = Vehicle.load(os.path.join(FX, "ascent_vehicle.yaml"))
    return AttitudeProgram.from_vehicle(veh), ReferenceTrajectory.from_csv(
        os.path.join(FX, "ot_out", "trajectory.csv")), veh.stages[0]["engine"]["burn_end_s"]


def _settled(res):
    errs = [chart_error_deg(s, res.program) for s in res.samples if s["t_s"] >= SETTLE_T_S]
    return max(errs), math.sqrt(sum(e * e for e in errs) / len(errs))


def test_so3_log_matches_axis_angle():
    # matrix log verified against a Rodrigues-built rotation matrix (no
    # quaternion convention involved)
    for axis in (np.array([1.0, 0.0, 0.0]), np.array([0.3, -0.8, 0.5]) / math.sqrt(1.58)):
        axis = axis / np.linalg.norm(axis)
        for ang in (1e-4, 0.3, 1.7, 3.0):
            kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                           [-axis[1], axis[0], 0]])
            r = np.eye(3) + math.sin(ang) * kx + (1 - math.cos(ang)) * (kx @ kx)
            w = so3_log(r)
            assert np.linalg.norm(w - ang * axis) < 1e-6


def test_so3_log_identity():
    assert np.linalg.norm(so3_log(np.eye(3))) < 1e-12


def test_attitude_error_on_real_sample():
    prog, _, _ = _fixture()
    # short no-control run: use the real sampled state (all conventions as
    # the driver reports them) to test the error machinery, not its math
    from orbitopt.flightlab.sixdof.jsbsim_driver import InitialConditions, SixDofSim
    sim = SixDofSim(0.01)
    sim.initialize(InitialConditions(lat_geod_deg=0.0, lon_deg=0.0, alt_m=0.0,
                                     pitch_deg=90.0, heading_deg=90.0, roll_deg=90.0))
    sim.ignite(1.0)
    sim.run_for(14.0, lambda t, s: None)  # pitch-over under way: chart not at the pole
    s = sim.samples[-1]
    az_a, el_a = chart_attitude(s)

    def prog_at(el_off):
        # chart_attitude reports radians; AttitudePoint takes degrees
        az_d, el_d = math.degrees(az_a), math.degrees(el_a)
        return AttitudeProgram.from_points([
            AttitudePoint(0.0, az_d, el_d + el_off),
            AttitudePoint(1.0e6, az_d, el_d + el_off)])

    assert np.linalg.norm(attitude_error_body(s, prog_at(0.0))) < math.radians(0.05)
    # commanded ABOVE the actual attitude must want a nose-up correction
    assert attitude_error_body(s, prog_at(+2.0))[1] > math.radians(1.5)
    assert attitude_error_body(s, prog_at(-2.0))[1] < math.radians(-1.5)


def test_closed_loop_full_profile_tracking():
    prog, ref, meco = _fixture()
    res = run_tracking_spike(ref, prog, meco, dt_s=0.01)
    m = res.metrics
    assert m.finite
    assert m.att_err_max_deg <= FULL_MAX_DEG
    assert m.gimbal_sat_fraction_of_burn == 0.0
    assert m.gimbal_max_deg < 5.0
    s_max, s_rms = _settled(res)
    assert s_max <= SETTLED_MAX_DEG
    assert s_rms <= SETTLED_RMS_DEG
    assert abs(m.d_alt_m) <= D_ALT_TOL_M
    assert abs(m.d_speed_m_s) <= D_SPEED_TOL_M_S
    assert abs(m.d_fpa_deg) <= D_FPA_TOL_DEG


def test_dt_halving_convergence():
    prog, ref, meco = _fixture()
    coarse = run_tracking_spike(ref, prog, meco, dt_s=0.01).metrics
    fine = run_tracking_spike(ref, prog, meco, dt_s=0.005).metrics
    assert fine.finite
    assert abs(coarse.alt_climb_6dof_m - fine.alt_climb_6dof_m) < DT_CONVERGE_ALT_M
    assert abs(coarse.speed_6dof_m_s - fine.speed_6dof_m_s) < 20.0
    assert abs(coarse.fpa_6dof_deg - fine.fpa_6dof_deg) < 0.5
