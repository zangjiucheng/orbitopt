"""Checks for orbitopt.flightlab.recovery.landing (FL-303 G-FOLD landing
burn with independent re-integration).

Tolerances -- declared up front, before any result is looked at:

    AGREE_FUEL_TOL      gfold vs the cvxpy reference: planned fuel must
                        agree to 0.5 % of the gfold figure (same SOCP,
                        different discretization plumbing and solver).
    AGREE_THRUST_TOL    Max node-wise thrust-vector difference over the
                        max plan thrust: 10 %. The min-fuel optimum is
                        not node-wise unique when the arc has slack, so
                        this is a wide band; the cases chosen are
                        lateral-offset, near-tight arcs where the profile
                        is nearly determined.
    AGREE_POS_TOL_M     Node-wise position agreement to 25 m, velocity to
                        3 m/s -- the arc, not the micrometre.
    LAND_*              The nominal RTLS case lands inside the default
                        LandingLimits purely from the drag-laden
                        re-integration verdict.

The three agreement cases (B/C/R3) are lateral-offset insertions solved
at a fixed TOF derived from gfold's own minimum-feasible line search
plus a 5 % stretch -- the minimum-TOF arc is a dive with an under-
determined thrust profile, and a stretched one is the honest comparison
ground. Case A (near-vertical, slack) is deliberately excluded from
node-wise agreement: both solvers reach the same 6784 kg answer to
0.002 % while routing the thrust differently mid-arc.

Why the nominal mass is ~54 t and the plan ends *above* the pad: this
vehicle's 40 % throttle floor is ~517 kN ambient, so the engine cannot
hover below ~52.7 t and can never zero its descent rate under thrust.
The plan therefore terminates at the engine-cutoff state
(``cutoff_height_m`` / ``cutoff_descent_m_s``, default 0.25 m / 1.2 m/s)
and the vehicle coasts the last step ballistically; touchdown speed
comes from the propagator's ground-event bisection either way.
"""
from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from orbitopt.flightlab.contract import read_run, write_run
from orbitopt.flightlab.frames import geodetic_to_ecef
from orbitopt.flightlab.recovery import (
    LANDING_INFEASIBLE,
    LANDING_TILT_EXCEEDED,
    LANDING_VERTICAL_SPEED_EXCEEDED,
    BoosterModel,
    BoosterState,
    LandingBurnEnu,
    LandingInfeasibleError,
    LandingSolverUnavailableError,
    RecoveryTarget,
    fly_landing_burn,
    landing_result_to_run,
    solve_landing_burn,
)
from orbitopt.flightlab.recovery.boostback import _enu_basis

pytest.importorskip("gfold")
pytest.importorskip("cvxpy")

REPO_ROOT = Path(__file__).resolve().parents[1]
VEHICLE_YAML = REPO_ROOT / "src/orbitopt/data/flightlab/vehicles/synthetic_two_stage.yaml"

EPOCH = datetime(2026, 11, 10, 12, 0, 0)
PAD_LAT, PAD_LON, PAD_ALT = 28.608, -80.604, 0.0

AGREE_FUEL_TOL = 0.005
AGREE_THRUST_TOL = 0.10
AGREE_POS_TOL_M = 25.0
AGREE_VEL_TOL_M_S = 3.0

PAD_ECEF = np.asarray(geodetic_to_ecef(PAD_LAT, PAD_LON, PAD_ALT), dtype=np.float64)
_BAS = np.vstack([np.asarray(b, dtype=np.float64) for b in _enu_basis(PAD_LAT, PAD_LON)])

VEHICLE = BoosterModel.load(VEHICLE_YAML)
TARGET = RecoveryTarget(kind="rtls", name="LZ-33", latitude_deg=PAD_LAT,
                        longitude_deg=PAD_LON, altitude_m=PAD_ALT)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# (position_enu, velocity_enu, mass_kg)
CASE_B = ((2500.0, -800.0, 3000.0), (-30.0, 5.0, -250.0), 47000.0)
CASE_C = ((-1500.0, 400.0, 2800.0), (5.0, -10.0, -320.0), 43000.0)
CASE_R3 = ((-900.0, -1600.0, 2900.0), (8.0, -18.0, -270.0), 45000.0)
CASE_A = ((40.0, 10.0, 2500.0), (2.0, -3.0, -180.0), 46000.0)

# Nominal RTLS insertion: heavy enough that the 40 % throttle floor sits at
# (or below) weight, low and slow enough that burn-averaged drag is a few
# hundredths of a m/s^2, so the ballistic cutoff coast stays inside the
# vertical-speed limit.
CASE_NOMINAL = ((80.0, 0.0, 200.0), (-10.0, 0.0, -10.0), 54000.0)
NOMINAL_TOF_S = 25.0

# Flat-feasible, drag-failing (the G3 exhibit): fast insertion at ~46 t,
# below the hover mass, where G-FOLD's drag-free arc claims a landing the
# re-integration refuses.
CASE_G3 = ((60.0, 0.0, 400.0), (-8.0, 0.0, -55.0), 46000.0)
G3_TOF_S = 12.0

# Nothing gets you from -450 m/s at 150 m in under 10 s: infeasible.
CASE_INFEASIBLE = ((0.0, 0.0, 150.0), (0.0, 0.0, -450.0), 45000.0)


def _enu_state(pos, vel, m0, t0=200.0):
    return LandingBurnEnu(t_s=t0, position_enu_m=pos, velocity_enu_m=vel,
                          mass_kg=m0)


def _ecef_state(pos, vel, m0, t0=200.0):
    r = PAD_ECEF + _BAS.T @ np.asarray(pos, dtype=np.float64)
    v = _BAS.T @ np.asarray(vel, dtype=np.float64)
    return BoosterState(t_s=t0, epoch_utc=EPOCH, frame="ECEF",
                        position_m=tuple(float(c) for c in r),
                        velocity_m_s=tuple(float(c) for c in v),
                        mass_kg=m0)


def _solve(case, solver, tof_s, tilt_limit_deg=90.0):
    pos, vel, m0 = case
    return solve_landing_burn(_enu_state(pos, vel, m0), VEHICLE,
                              solver=solver, tof_s=tof_s, n=16,
                              glide_slope_deg=20.0,
                              tilt_limit_deg=tilt_limit_deg)


def _tight_tof(case):
    """gfold's minimum-feasible TOF + 5 % -- the fixed arc both solvers
    are compared on (deterministic; the solver line search is exact)."""
    pos, vel, m0 = case
    plan = solve_landing_burn(_enu_state(pos, vel, m0), VEHICLE,
                              solver="gfold", n=16, tof_min_s=8.0,
                              tof_max_s=120.0, glide_slope_deg=20.0,
                              tilt_limit_deg=90.0)
    return 1.05 * plan.tof_commanded_s


def _pair_metrics(case, tof_s):
    pg = _solve(case, "gfold", tof_s)
    pc = _solve(case, "cvxpy", tof_s)
    fuel_g, fuel_c = pg.fuel_planned_kg, pc.fuel_planned_kg
    tg = np.asarray(pg.thrust_enu_n)
    tc = np.asarray(pc.thrust_enu_n)
    dth = float(np.max(np.linalg.norm(tg - tc, axis=1))
                / np.max(np.linalg.norm(tg, axis=1)))
    dpos = float(np.max(np.linalg.norm(
        np.asarray(pg.position_enu_m) - np.asarray(pc.position_enu_m), axis=1)))
    dvel = float(np.max(np.linalg.norm(
        np.asarray(pg.velocity_enu_m) - np.asarray(pc.velocity_enu_m), axis=1)))
    return {"relfuel": abs(fuel_g - fuel_c) / fuel_g, "dth": dth,
            "dpos": dpos, "dvel": dvel, "fuel_g": fuel_g, "fuel_c": fuel_c,
            "gap": pc.lossless_gap}


# ---------------------------------------------------------------------------
# acceptance 1: gfold vs cvxpy agreement on the reference cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", [CASE_B, CASE_C, CASE_R3],
                         ids=["B", "C", "R3"])
def test_gfold_cvxpy_agreement(case):
    m = _pair_metrics(case, _tight_tof(case))
    assert m["relfuel"] <= AGREE_FUEL_TOL
    assert m["dth"] <= AGREE_THRUST_TOL
    assert m["dpos"] <= AGREE_POS_TOL_M
    assert m["dvel"] <= AGREE_VEL_TOL_M_S
    assert m["gap"] <= 1e-6  # cvxpy reference: lossless, duality gap gone


# ---------------------------------------------------------------------------
# acceptance 2: nominal RTLS lands under drag-laden re-integration
# ---------------------------------------------------------------------------

def test_nominal_rtls_lands_after_reintegration():
    pos, vel, m0 = CASE_NOMINAL
    res = fly_landing_burn(_enu_state(pos, vel, m0), _ecef_state(pos, vel, m0),
                           TARGET, VEHICLE, solver="gfold",
                           tof_s=NOMINAL_TOF_S, n=16, glide_slope_deg=20.0)
    assert res.feasible
    assert res.landed, res.reason_codes
    assert res.reason_codes == ()
    assert res.track is not None and res.track.impact is not None
    # the verdict is the propagator's: cutoff happens *above* the pad and
    # the touchdown is the ballistic coast's, graded independently
    assert res.plan.position_enu_m[-1][2] == pytest.approx(0.25, abs=1e-6)
    imp = res.track.impact
    assert imp.vertical_speed_m_s > -3.0
    # propellant stayed above the reserve: check id present and green
    ids = {c.id: c for c in res.checks}
    assert ids["landing-propellant-reserve"].passed
    assert len(res.checks) == 6


# ---------------------------------------------------------------------------
# G3 exhibit: flat-model-feasible, drag-failing must report failed
# ---------------------------------------------------------------------------

def test_drag_breaks_the_flat_solution_and_it_is_reported():
    pos, vel, m0 = CASE_G3
    res = fly_landing_burn(_enu_state(pos, vel, m0), _ecef_state(pos, vel, m0),
                           TARGET, VEHICLE, solver="gfold",
                           tof_s=G3_TOF_S, n=16, glide_slope_deg=20.0)
    assert res.feasible, "G-FOLD itself must find this arc"
    assert not res.landed
    assert LANDING_VERTICAL_SPEED_EXCEEDED in res.reason_codes


# ---------------------------------------------------------------------------
# acceptance 3: infeasible SOCP never green
# ---------------------------------------------------------------------------

def test_infeasible_is_reported_not_green():
    pos, vel, m0 = CASE_INFEASIBLE
    with pytest.raises(LandingInfeasibleError):
        solve_landing_burn(_enu_state(pos, vel, m0), VEHICLE, solver="gfold",
                           tof_s=6.0, n=16, glide_slope_deg=20.0)
    res = fly_landing_burn(_enu_state(pos, vel, m0), _ecef_state(pos, vel, m0),
                           TARGET, VEHICLE, solver="gfold",
                           tof_s=6.0, n=16, glide_slope_deg=20.0)
    assert not res.feasible and not res.landed
    assert res.reason_codes == (LANDING_INFEASIBLE,)
    assert res.plan is None


def test_missing_solver_is_not_reported_infeasible(monkeypatch):
    """No gfold in the environment is an environment gap, not a physics
    verdict: it must never surface as LandingInfeasibleError (which the
    flow reports as LANDING_INFEASIBLE)."""
    monkeypatch.setitem(sys.modules, "gfold", None)
    pos, vel, m0 = CASE_A
    with pytest.raises(LandingSolverUnavailableError, match="not installed") as exc:
        solve_landing_burn(_enu_state(pos, vel, m0), VEHICLE, solver="gfold",
                           tof_s=20.0, n=16)
    assert not isinstance(exc.value, LandingInfeasibleError)
    with pytest.raises(LandingSolverUnavailableError):
        fly_landing_burn(_enu_state(pos, vel, m0), _ecef_state(pos, vel, m0),
                         TARGET, VEHICLE, solver="gfold", tof_s=20.0, n=16)


def test_off_pin_solver_is_not_reported_infeasible(monkeypatch):
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.0.1")
    pos, vel, m0 = CASE_A
    with pytest.raises(LandingSolverUnavailableError, match="pinned"):
        solve_landing_burn(_enu_state(pos, vel, m0), VEHICLE, solver="gfold",
                           tof_s=20.0, n=16)


# ---------------------------------------------------------------------------
# the tilt cone is real in the reference implementation (gfold's own
# max_angle is inert at these geometries, so the tilt limit is enforced
# by the cvxpy model; a slack arc must become infeasible when it binds)
# ---------------------------------------------------------------------------

def test_tilt_cone_binds_the_cvxpy_reference():
    # the cone only binds near the minimum-feasible arc (~19.9 s line
    # search here); a slack arc admits the same near-vertical solution
    tof = 19.0
    loose = _solve(CASE_A, "cvxpy", tof, tilt_limit_deg=90.0)
    assert loose.fuel_planned_kg > 0.0
    with pytest.raises(LandingInfeasibleError):
        _solve(CASE_A, "cvxpy", tof, tilt_limit_deg=40.0)


# ---------------------------------------------------------------------------
# plan mirrors the cutoff design + guidance stops at the last node
# ---------------------------------------------------------------------------

def test_plan_terminates_at_cutoff_state():
    pos, vel, m0 = CASE_B
    plan = _solve(CASE_B, "gfold", _tight_tof(CASE_B))
    end_pos = np.asarray(plan.position_enu_m[-1])
    end_vel = np.asarray(plan.velocity_enu_m[-1])
    assert end_pos[2] == pytest.approx(0.25, abs=1e-6)
    assert end_vel[2] == pytest.approx(-1.2, abs=1e-6)
    assert plan.tof_commanded_s == pytest.approx(_tight_tof(CASE_B), rel=1e-9)


# ---------------------------------------------------------------------------
# FlightRun plumbing: body track + landing_start / touchdown events,
# green only when re-integration agrees
# ---------------------------------------------------------------------------

def test_flight_run_roundtrip(tmp_path):
    pos, vel, m0 = CASE_NOMINAL
    res = fly_landing_burn(_enu_state(pos, vel, m0), _ecef_state(pos, vel, m0),
                           TARGET, VEHICLE, solver="gfold",
                           tof_s=NOMINAL_TOF_S, n=16, glide_slope_deg=20.0)
    run = landing_result_to_run(res, run_id="fl303-landing-nominal",
                                target_sha256="0" * 64,
                                vehicle_sha256="1" * 64,
                                git_sha="a" * 40,
                                created_utc="2026-11-10T12:00:00Z")
    kinds = [e.kind for e in run.events]
    assert kinds == ["landing_start", "touchdown"]
    assert run.verification.optimizer_converged
    assert run.verification.physics_verified
    assert len(run.tracks) == 1
    write_run(tmp_path, run)
    back = read_run(tmp_path)
    assert [e.kind for e in back.events] == kinds
    assert back.verification == run.verification


def test_flight_run_green_requires_landing(tmp_path):
    """The G3 attempt writes a run too -- converged solver, red physics."""
    pos, vel, m0 = CASE_G3
    res = fly_landing_burn(_enu_state(pos, vel, m0), _ecef_state(pos, vel, m0),
                           TARGET, VEHICLE, solver="gfold",
                           tof_s=G3_TOF_S, n=16, glide_slope_deg=20.0)
    run = landing_result_to_run(res, run_id="fl303-landing-g3",
                                target_sha256="0" * 64,
                                vehicle_sha256="1" * 64,
                                git_sha="a" * 40,
                                created_utc="2026-11-10T12:00:00Z")
    assert [e.kind for e in run.events] == ["landing_start", "impact"]
    assert run.verification.optimizer_converged
    assert not run.verification.physics_verified
    assert not run.verification.mission_constraints_passed
