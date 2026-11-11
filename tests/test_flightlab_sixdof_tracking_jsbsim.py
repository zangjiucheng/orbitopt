"""FL-403 (#208): JSBSim 6-DOF PID gimbal tracking of the 3-DOF ascent program.

Acceptance thresholds are declared up front (module constants below). The
closed-loop tests need the recorded 3-DOF reference fixture and ``jsbsim``
(opt-in pip dep: ``pip install -r envs/sixdof-requirements.txt``); both
skip cleanly when missing.

Declared GO criteria (ADR docs/adr/flightlab-0001-jsbsim-6dof.md):
* settled attitude tracking (t >= 60 s): max <= 2 deg, RMS <= 0.6 deg;
* gimbal saturation 0 % of the burn, command never beyond travel limits;
* stable integration: dt 0.01 vs 0.005 agree, no non-finite states;
* MECO state deviation vs the 3-DOF reference within the bounds below.

Regression guards added with the quaternion->DCM fix: the whole-burn error
bound is the same 2 deg (it was 30 deg while a transposed DCM plus a launch
IC tuned against it produced a ~24 deg early "transient"), and the MECO
bounds sit ~10x above the measured deviations (51 m / 2 m/s / 0.02 deg).
"""
from __future__ import annotations

import math
import os
import warnings

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
from orbitopt.flightlab.sixdof.jsbsim_driver import (  # noqa: E402
    InitialConditions,
    SixDofSim,
    quat_from_dcm,
)
from orbitopt.flightlab.sixdof.tracking import (  # noqa: E402
    AttitudeProgram,
    ReferenceTrajectory,
    chart_attitude,
    chart_error_deg,
    attitude_error_body,
    dcm_from_quat,
    run_tracking_spike,
    so3_log,
)

LAUNCH_IC = InitialConditions(lat_geod_deg=0.0, lon_deg=0.0, alt_m=0.0,
                              pitch_deg=90.0, heading_deg=90.0, roll_deg=-90.0)

SETTLE_T_S = 60.0
SETTLED_MAX_DEG = 2.0
SETTLED_RMS_DEG = 0.6
FULL_MAX_DEG = 2.0             # whole burn, pitch-over kick included
D_ALT_TOL_M = 500.0            # MECO deviation bounds vs the 3-DOF reference
D_SPEED_TOL_M_S = 20.0
D_FPA_TOL_DEG = 0.2
GIMBAL_TRAVEL_DEG = 5.0
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


def _tec2b(sim):
    with warnings.catch_warnings():  # get_Tec2b returns a deprecated numpy.matrix
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        return np.asarray(sim.fdm.get_propagate().get_Tec2b())


def test_dcm_from_quat_inverts_quat_from_dcm():
    # dcm_from_quat must rebuild exactly the matrix the driver encoded
    # (Tec2b), not its transpose -- the transpose mis-steered FL-403.
    rng = np.random.default_rng(7)
    for _ in range(20):
        q_mat, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        q_mat *= np.sign(np.linalg.det(q_mat))
        assert np.allclose(dcm_from_quat(quat_from_dcm(q_mat)), q_mat, atol=1e-12)


def test_launch_ic_matches_program_frame():
    """Ground truth from JSBSim's own Tec2b (not through dcm_from_quat): at
    (lat, lon) = (0, 0) ECEF x/y/z = up/east/north, and the program frame at
    (el=90, az=90) is body x = up, y = south, z = east. A nose-down pitch
    gimbal then tips the nose east, the program's pitch-over direction."""
    sim = SixDofSim(0.01)
    sim.initialize(LAUNCH_IC)
    tec2b = _tec2b(sim)
    assert np.allclose(tec2b, [[1, 0, 0], [0, 0, -1], [0, 1, 0]], atol=1e-6)
    prog = AttitudeProgram.from_points([AttitudePoint(0.0, 90.0, 90.0),
                                        AttitudePoint(1.0e6, 90.0, 90.0)])
    assert np.linalg.norm(attitude_error_body(sim.samples[-1], prog)) < 1e-6
    sim.ignite(1.0)
    sim.run_for(3.0, lambda t, s: s.set_gimbal(-math.radians(0.2) if t >= 2.0 else 0.0, 0.0))
    nose = _tec2b(sim)[0]
    assert nose[1] > 1e-3 and abs(nose[2]) < 1e-6  # east, no north component


def test_attitude_error_on_real_sample():
    prog, _, _ = _fixture()
    # short open-loop run: use the real sampled state (all conventions as
    # the driver reports them) to test the error machinery, not its math.
    # An uncontrolled vertical stage stays vertical, so a nose-down gimbal
    # pulse tips it off the chart pole first.
    sim = SixDofSim(0.01)
    sim.initialize(LAUNCH_IC)
    sim.ignite(1.0)
    sim.run_for(14.0, lambda t, s: s.set_gimbal(-math.radians(0.5) if 2.0 <= t < 4.0 else 0.0,
                                                0.0))
    s = sim.samples[-1]
    az_a, el_a = chart_attitude(s)
    assert math.degrees(el_a) < 85.0  # off the pole

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
    assert m.gimbal_max_deg < GIMBAL_TRAVEL_DEG
    burn = [s for s in res.samples if s["t_s"] <= meco]
    assert max(max(abs(s["gimbal_pitch_cmd_rad"]), abs(s["gimbal_yaw_cmd_rad"]))
               for s in burn) < math.radians(GIMBAL_TRAVEL_DEG)
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
