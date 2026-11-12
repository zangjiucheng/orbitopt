"""Checks for orbitopt.flightlab.recovery.propagator (FL-301 booster 3-DOF).

Tolerances -- declared up front, before any result is looked at (FL-301
acceptance criteria):

    FREEFALL_SPEED_REL   = 5e-3   Vacuum radial fall from the pole vs the
    FREEFALL_TIME_REL    = 5e-3     energy integral / Kepler fall-time
        closed form (impact at the WGS84 polar radius). RK4 at dt = 0.05 s;
        the ground crossing is bisected to ~mm, so the slack is the step error
        accumulated over a ~200 s ballistic fall.
    TSIOK_REL            = 1e-5   Constant thrust / constant mdot in the
        gravity-off vacuum limit vs Tsiolkovsky dV = Isp_vac * g0 * ln(m0/m1).
        The burn is split exactly at the dry mass (mdot constant while on),
        so this tests the integrator and the mass bookkeeping, not the
        atmosphere model; 1e-5 rel covers the RK4 error on a ~51 s burn.
    CORIOLIS_REL         = 0.05   East deflection of a vacuum drop from the
        equator vs the first-order estimate (1/3) * omega * g * t_fall^3: the
        first-order formula ignores the (small) g variation over 20 km and the
        centrifugal contribution to t_fall; 5 % is its own error, not the
        propagator's.
    CONVERGENCE_DT_RATIO = 4      Halving dt must shrink the impact-time
    CONVERGENCE_MAX_D_T_S = 2e-4    change by >= 4x, and the finer pair must
    CONVERGENCE_MAX_D_V_M_S = 0.02  agree to < 0.2 ms / 2 cm/s on a 40 km
        aero-braked drop (dt = 0.5 vs 0.25). The impact-*speed* difference is
        only required to decrease: the entry density table is piecewise
        (kinks at the layer boundaries), which caps the observed global order
        near 2 there -- the absolute bound above is what "converged" means.
    OT_POS_TOL_M         = 5e3    Ballistic impact vs the frozen OpenTsiolkovsky
    OT_TIME_TOL_S        = 2.0    stage-1 debris fixture, US76-direct density
    OT_SPEED_TOL_M_S     = 30.0   model: position/time/speed. The residual is
    OT_ENTRY_*           (below)  the frame bridge / ground-model difference
                                  (spherical-mean ground vs WGS84 ellipsoid),
                                  not the dynamics: the whole trajectory
                                  agrees to ~10 m/s through entry.
    OT_ENTRY_SPEED_TOL_M_S = 250.0  Same fixture with the *default* entry
        density model (Vallado's piecewise-exponential US76 fit, ~15 % light
        near sea level): position/time stay inside the tolerances above, but
        the shallower near-surface air costs impact speed -- booked, because
        the FL-301 scope pins the entry reuse as the default.

Reference sources: the analytic cases are textbook closed forms (radial free
fall, Tsiolkovsky, Coriolis first-order); the cross-check oracle is the
frozen OpenTsiolkovsky fixture recorded by FL-004 under the
``tests/fixtures/flightlab/opentsiolkovsky`` license header, run with
OT's own inferred aero table (Cd from the stage ca table, A = pi/4 * 3.7^2).
"""
from __future__ import annotations

import math
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from orbitopt.entry import G0_M_S2, MU_EARTH_M3_S2 as MU
from orbitopt.entry import height_m as entry_height_m
from orbitopt.flightlab import frames
from orbitopt.flightlab.contract import EVENT_KINDS, FLIGHT_PROVENANCE_TAGS
from orbitopt.flightlab.recovery.propagator import (
    AltitudeThreshold,
    BoosterEngine,
    BoosterModel,
    BoosterPhysics,
    BoosterState,
    RecoveryError,
    ThrustCmd,
    ballistic_return,
    density_us76_kg_m3,
    propagate,
    separation_state_from_run,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VEHICLE_YAML = REPO_ROOT / "src/orbitopt/data/flightlab/vehicles/synthetic_two_stage.yaml"
OT_FIXTURE = REPO_ROOT / "tests/fixtures/flightlab/opentsiolkovsky/synthetic_out"

FREEFALL_SPEED_REL = 5e-3
FREEFALL_TIME_REL = 5e-3
TSIOK_REL = 1e-5
CORIOLIS_REL = 0.05
CONVERGENCE_DT_RATIO = 4.0
CONVERGENCE_MAX_D_T_S = 2e-4
CONVERGENCE_MAX_D_V_M_S = 0.02
OT_POS_TOL_M = 5e3
OT_TIME_TOL_S = 2.0
OT_SPEED_TOL_M_S = 30.0
OT_ENTRY_SPEED_TOL_M_S = 250.0

EPOCH = datetime(2026, 1, 1)
B_POLAR = frames.WGS84_A * math.sqrt(1.0 - frames.WGS84_F * (2.0 - frames.WGS84_F))


def _vehicle(provenance="illustrative", dry=30000.0, reserve=0.0, prop=0.0):
    engine = BoosterEngine(
        count=1, usable_for=("boostback", "landing"), thrust_vacuum_n=1_800_000.0,
        isp_sea_level_s=282.0, isp_vacuum_s=311.0, nozzle_exit_area_m2=5.0,
        throttle_min=0.4, throttle_max=1.0, restarts_max=3)
    return BoosterModel(
        name="test-booster", provenance=provenance, dry_mass_kg=dry,
        landing_propellant_reserve_kg=reserve, reference_area_m2=10.7516,
        cd_mach=((0.0, 0.35), (0.8, 0.38), (1.1, 0.55), (1.5, 0.5),
                 (2.5, 0.4), (4.0, 0.32), (8.0, 0.26)),
        engine=engine)


# ---------------------------------------------------------------------------
# analytic case 1: vacuum free fall vs the closed form
# ---------------------------------------------------------------------------


def test_vacuum_free_fall_matches_closed_form():
    """Drop from rest over the pole, point-mass gravity, vacuum, no rotation:
    impact speed is the energy integral, impact time the Kepler fall time."""
    h0 = 200_000.0
    r0 = B_POLAR + h0
    st = BoosterState(0.0, EPOCH, "ECEF", (0.0, 0.0, r0), (0.0, 0.0, 0.0), 30_000.0)
    ph = BoosterPhysics(gravity="point_mass", atmosphere=False, rotation=False)
    tr = propagate(st, _vehicle(), None, 400.0, events=("ground",),
                   dt_s=0.05, sample_s=10.0, physics=ph)
    assert tr.termination == "ground_impact"
    v_expect = math.sqrt(2.0 * MU * (1.0 / B_POLAR - 1.0 / r0))
    x = B_POLAR / r0
    t_expect = math.sqrt(r0 ** 3 / (2.0 * MU)) * (
        math.acos(math.sqrt(x)) + math.sqrt(x * (1.0 - x)))
    assert tr.impact.speed_m_s == pytest.approx(v_expect, rel=FREEFALL_SPEED_REL)
    assert tr.impact.t_s == pytest.approx(t_expect, rel=FREEFALL_TIME_REL)
    assert tr.impact.latitude_deg == pytest.approx(90.0, abs=1e-9)
    assert tr.impact.vertical_speed_m_s < 0.0
    assert [e.kind for e in tr.events] == ["ground"]
    assert tr.propellant_remaining_kg == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# analytic case 2: constant thrust / constant mdot vs Tsiolkovsky
# ---------------------------------------------------------------------------


def test_constant_thrust_matches_tsiolkovsky():
    """Gravity-off vacuum burn to depletion: dV = Isp_vac * g0 * ln(m0/m1),
    the mass lands exactly on dry, one exhaustion event, one ignition."""
    dry, prop = 20_000.0, 10_000.0
    veh = BoosterModel(
        name="tsiok", provenance="illustrative", dry_mass_kg=dry,
        landing_propellant_reserve_kg=0.0, reference_area_m2=1.0,
        cd_mach=((0.0, 0.35), (8.0, 0.26)),
        engine=BoosterEngine(count=1, usable_for=("landing",),
                             thrust_vacuum_n=600_000.0, isp_sea_level_s=282.0,
                             isp_vacuum_s=311.0, nozzle_exit_area_m2=0.0,
                             throttle_min=0.4))
    st = BoosterState(0.0, EPOCH, "ECEF", (0.0, 0.0, 6_778_137.0),
                      (0.0, 0.0, 0.0), dry + prop)
    ph = BoosterPhysics(gravity="off", atmosphere=False, rotation=False)
    tr = propagate(st, veh, lambda t, s: ThrustCmd(1.0, (0.0, 0.0, 1.0)), 200.0,
                   events=("burn",), dt_s=0.1, sample_s=5.0, physics=ph)
    dv = float(np.linalg.norm(tr.velocity_ecef_m_s[-1] - tr.velocity_ecef_m_s[0]))
    dv_expect = 311.0 * G0_M_S2 * math.log((dry + prop) / dry)
    assert dv == pytest.approx(dv_expect, rel=TSIOK_REL)
    assert tr.termination == "time_limit"
    assert tr.mass_kg.min() >= dry - 1e-9
    assert tr.mass_kg[-1] == pytest.approx(dry, abs=1e-6)
    assert tr.ignitions_used == 1
    starts = [e for e in tr.events if e.kind == "burn_start"]
    ends = [e for e in tr.events if e.kind == "burn_end"]
    assert len(starts) == 1 and len(ends) == 1
    assert ends[0].data["reason"] == "PROPELLANT_EXHAUSTED"
    # constant mdot: the bisected burn duration is m_prop / mdot
    burn_s = float(ends[0].t_s)
    mdot_expect = 600_000.0 / (311.0 * G0_M_S2)
    assert burn_s == pytest.approx(prop / mdot_expect, rel=1e-3)


# ---------------------------------------------------------------------------
# analytic case 3: Coriolis deflection sanity
# ---------------------------------------------------------------------------


def test_coriolis_deflection_sanity():
    """Vacuum drop from rest over the equator (lon 0) in the rotating frame:
    the impact is deflected EAST, within 5 % of the first-order estimate
    (1/3) omega g t_f^3; the non-rotating twin misses by far more."""
    h0 = 20_000.0
    r0 = frames.WGS84_A + h0
    st = BoosterState(0.0, EPOCH, "ECEF", (r0, 0.0, 0.0), (0.0, 0.0, 0.0), 30_000.0)
    rot = propagate(st, _vehicle(), None, 200.0, events=("ground",), dt_s=0.02,
                    physics=BoosterPhysics(gravity="point_mass", atmosphere=False,
                                           rotation=True))
    still = propagate(st, _vehicle(), None, 200.0, events=("ground",), dt_s=0.02,
                      physics=BoosterPhysics(gravity="point_mass", atmosphere=False,
                                             rotation=False))
    g_eff = 9.78
    t_f = math.sqrt(2.0 * h0 / g_eff)
    defl_expect = (1.0 / 3.0) * float(frames.EARTH_ROTATION_RATE) * g_eff * t_f ** 3
    y_rot = rot.impact.position_ecef_m[1]
    y_still = still.impact.position_ecef_m[1]
    assert y_rot > 0.0                      # east
    assert abs(y_still) < 1e-6              # no rotation -> no deflection
    assert y_rot == pytest.approx(defl_expect, rel=CORIOLIS_REL)
    # sanity: sane order of magnitude for a 64 s fall (~60 m, not km-scale)
    assert 10.0 < y_rot < 200.0


# ---------------------------------------------------------------------------
# step-size convergence on an aero-braked drop (stiff near-surface region)
# ---------------------------------------------------------------------------


def _aero_drop(dt_s: float):
    st = BoosterState(0.0, EPOCH, "ECEF", (0.0, 0.0, B_POLAR + 40_000.0),
                      (0.0, 0.0, 0.0), 30_000.0)
    return propagate(st, _vehicle(), None, 900.0, events=(), dt_s=dt_s,
                     sample_s=10.0, physics=BoosterPhysics(
                         gravity="point_mass", rotation=False))


def test_step_size_convergence():
    tr1, tr05, tr025 = (_aero_drop(dt) for dt in (1.0, 0.5, 0.25))
    for tr in (tr1, tr05, tr025):
        assert tr.termination == "ground_impact"
    d1 = abs(tr1.impact.t_s - tr05.impact.t_s)
    d2 = abs(tr05.impact.t_s - tr025.impact.t_s)
    dpos1 = abs(tr1.impact.speed_m_s - tr05.impact.speed_m_s)
    dpos2 = abs(tr05.impact.speed_m_s - tr025.impact.speed_m_s)
    assert d2 < d1 / CONVERGENCE_DT_RATIO
    assert dpos2 < dpos1  # speed converges monotonically (see docstring: the
    # entry density table's layer kinks cap the observed order below 4 here)
    assert d2 < CONVERGENCE_MAX_D_T_S
    assert dpos2 < CONVERGENCE_MAX_D_V_M_S
    assert d2 > 0.0 and dpos2 > 0.0  # the runs are distinct, not cached twins


# ---------------------------------------------------------------------------
# propellant and restart bookkeeping
# ---------------------------------------------------------------------------


def test_propellant_never_negative_and_dry_engine_stays_dark():
    """Full-throttle retro burn from a 60 km coast until the tanks run dry:
    the mass never dips below dry, the engine auto-shuts once with a single
    PROPELLANT_EXHAUSTED note, and the coast continues afterwards."""
    dry, prop = 20_000.0, 4_000.0
    veh = _vehicle(dry=dry)
    st = BoosterState(0.0, EPOCH, "ECEF", (0.0, 0.0, B_POLAR + 60_000.0),
                      (0.0, 0.0, 0.0), dry + prop)
    tr = propagate(st, veh, lambda t, s: ThrustCmd(1.0, (0.0, 0.0, 1.0)), 900.0,
                   events=("burn",), dt_s=0.1, sample_s=2.0)
    assert tr.mass_kg.min() >= dry - 1e-9
    assert (tr.mass_kg >= dry - 1e-9).all()
    assert tr.propellant_remaining_kg == pytest.approx(0.0, abs=1e-6)
    ends = [e for e in tr.events if e.kind == "burn_end"]
    exhausted = [e for e in ends if e.data.get("reason") == "PROPELLANT_EXHAUSTED"]
    assert len(exhausted) == 1  # the dry engine notes once, then goes quiet
    assert len(ends) == 1
    assert tr.termination == "ground_impact"  # the dry engine never reignites
    assert tr.ignitions_used == 1


def test_burn_beyond_restart_count_is_rejected():
    """Three commanded ignitions against restarts_max=3 pass with exactly 3
    counted; a fourth raises RecoveryError naming the exhausted count."""
    st = BoosterState(0.0, EPOCH, "ECEF", (0.0, 0.0, B_POLAR + 60_000.0),
                      (0.0, 0.0, 0.0), 30_000.0)
    veh = BoosterModel(
        name="restart", provenance="illustrative", dry_mass_kg=20_000.0,
        landing_propellant_reserve_kg=0.0, reference_area_m2=10.7516,
        cd_mach=_vehicle().cd_mach,
        engine=BoosterEngine(count=1, usable_for=("boostback", "landing"),
                             thrust_vacuum_n=1_800_000.0, isp_sea_level_s=282.0,
                             isp_vacuum_s=311.0, nozzle_exit_area_m2=5.0,
                             throttle_min=0.4, throttle_max=1.0, restarts_max=3))

    def pulses(t, s, windows=((10.0, 12.0), (30.0, 32.0), (50.0, 52.0))):
        for a, b in windows:
            if a <= t < b:
                return ThrustCmd(1.0, (0.0, 0.0, 1.0))
        return None

    tr = propagate(st, veh, pulses, 100.0, events=("burn",), dt_s=0.2)
    assert tr.ignitions_used == 3
    assert len([e for e in tr.events if e.kind == "burn_start"]) == 3
    assert tr.mass_kg.min() >= veh.dry_mass_kg - 1e-9

    with pytest.raises(RecoveryError, match="restart count exhausted"):
        propagate(st, veh,
                  lambda t, s: pulses(t, s, ((10.0, 12.0), (30.0, 32.0),
                                             (50.0, 52.0), (70.0, 72.0))),
                  100.0, events=("burn",), dt_s=0.2)


def test_throttle_below_min_flies_at_min_zero_shuts_down():
    st = BoosterState(0.0, EPOCH, "ECEF", (0.0, 0.0, B_POLAR + 60_000.0),
                      (0.0, 0.0, 0.0), 30_000.0)
    tr = propagate(st, _vehicle(dry=20_000.0),
                   lambda t, s: ThrustCmd(0.1, (0.0, 0.0, 1.0)), 60.0,
                   events=("burn",), dt_s=0.5, sample_s=1.0)
    assert tr.throttle.max() == pytest.approx(0.4)  # engine minimum, not 0.1
    tr_off = propagate(st, _vehicle(dry=20_000.0),
                       lambda t, s: ThrustCmd(0.0, (0.0, 0.0, 1.0)), 60.0,
                       events=("burn",), dt_s=0.5)
    assert tr_off.ignitions_used == 0
    assert tr_off.throttle.max() == 0.0


# ---------------------------------------------------------------------------
# events, track shape, contract bridge
# ---------------------------------------------------------------------------


OT_SEPARATION = dict(  # frozen FL-004 fixture: stage-1 debris at separation
    t_s=207.0, lat=29.783806060197332, lon=-79.16172810358658,
    alt=189100.17303361185, mass=30_000.0,
    vel_enu=(1964.68061098682, 2020.6358064197857, -1639.565739067858),
)
OT_IMPACT = dict(t_s=739.9388172614653, lat=37.97350280095533,
                 lon=-67.4461869373751, speed=1726.1388877350541)


def _ot_separation_state():
    d = OT_SEPARATION
    r0 = frames.geodetic_to_ecef(d["lat"], d["lon"], d["alt"])
    p, l = math.radians(d["lat"]), math.radians(d["lon"])
    east = np.array([-math.sin(l), math.cos(l), 0.0])
    north = np.array([-math.sin(p) * math.cos(l), -math.sin(p) * math.sin(l), math.cos(p)])
    up = np.array([math.cos(p) * math.cos(l), math.cos(p) * math.sin(l), math.sin(p)])
    n_, e_, dn_ = d["vel_enu"]
    v0 = e_ * east + n_ * north - dn_ * up
    return BoosterState(d["t_s"], EPOCH, "ECEF", tuple(float(c) for c in r0),
                        tuple(float(c) for c in v0), d["mass"])


def test_ot_separation_state_reproduces_fixture_velocity_enu():
    """The ECEF state built from the fixture's geodetic + ENU separation row
    round-trips through frames.ecef_to_geodetic (fixture sanity)."""
    st = _ot_separation_state()
    lat, lon, h = frames.ecef_to_geodetic(np.asarray(st.position_m))
    d = OT_SEPARATION
    assert float(lat) == pytest.approx(d["lat"], abs=1e-9)
    assert float(lon) == pytest.approx(d["lon"], abs=1e-9)
    assert float(h) == pytest.approx(d["alt"], abs=1e-2)
    assert st.mass_kg == d["mass"] and st.t_s == d["t_s"]


@pytest.mark.skipif(not OT_FIXTURE.exists(), reason="FL-004 OT fixture not recorded")
def test_ballistic_impact_cross_check_against_opentsiolkovsky():
    """FL-301's headline cross-check: the unpowered return of the jettisoned
    stage from the fixture's separation state vs the fixture's recorded
    impact (declared tolerances in the module docstring)."""
    st = _ot_separation_state()
    tr = ballistic_return(st, _vehicle(), 950.0, dt_s=1.0, sample_s=5.0,
                          events=("apogee", AltitudeThreshold(80e3, name="entry_interface")),
                          physics=BoosterPhysics(gravity="point_mass",
                                                 density_model="ussa1976"))
    assert tr.termination == "ground_impact"
    imp = tr.impact
    d_ns = (imp.latitude_deg - OT_IMPACT["lat"]) * 111_320.0
    d_ew = (imp.longitude_deg - OT_IMPACT["lon"]) * 111_320.0 * math.cos(
        math.radians(imp.latitude_deg))
    assert math.hypot(d_ns, d_ew) < OT_POS_TOL_M
    assert abs(imp.t_s - OT_IMPACT["t_s"]) < OT_TIME_TOL_S
    assert abs(imp.speed_m_s - OT_IMPACT["speed"]) < OT_SPEED_TOL_M_S
    kinds = [e.kind for e in tr.events]
    assert "apogee" in kinds and "entry_interface" in kinds
    # the recorded trajectory's own events bracket ours
    assert min(e.t_s for e in tr.events) > OT_SEPARATION["t_s"]


@pytest.mark.skipif(not OT_FIXTURE.exists(), reason="FL-004 OT fixture not recorded")
def test_ballistic_cross_check_default_entry_density_booked_gap():
    """Same fixture, default (entry-reuse) density: position/time stay inside
    the tolerances, impact speed is booked separately (exponential-fit
    density runs light near sea level -> less deceleration in the last km)."""
    st = _ot_separation_state()
    tr = ballistic_return(st, _vehicle(), 950.0, dt_s=1.0, sample_s=10.0,
                          events=(), physics=BoosterPhysics(gravity="point_mass"))
    imp = tr.impact
    d_ns = (imp.latitude_deg - OT_IMPACT["lat"]) * 111_320.0
    d_ew = (imp.longitude_deg - OT_IMPACT["lon"]) * 111_320.0 * math.cos(
        math.radians(imp.latitude_deg))
    assert math.hypot(d_ns, d_ew) < OT_POS_TOL_M
    assert abs(imp.t_s - OT_IMPACT["t_s"]) < OT_TIME_TOL_S
    assert abs(imp.speed_m_s - OT_IMPACT["speed"]) < OT_ENTRY_SPEED_TOL_M_S
    assert imp.speed_m_s > OT_IMPACT["speed"]  # lighter air near the ground


def test_density_models_agree_up_high_differ_low():
    """entry's fit and the direct US76 table are within a few percent above
    30 km and differ by ~15 % at 3 km (the booked cross-check gap)."""
    from orbitopt.entry import density_kg_m3
    assert density_us76_kg_m3(40_000.0) == pytest.approx(density_kg_m3(40_000.0), rel=0.05)
    assert density_us76_kg_m3(3_000.0) > density_kg_m3(3_000.0) * 1.10
    assert density_us76_kg_m3(0.0) == pytest.approx(1.225, rel=1e-3)


def test_ground_is_terminal_even_without_ground_event_spec():
    st = BoosterState(0.0, EPOCH, "ECEF", (0.0, 0.0, B_POLAR + 10_000.0),
                      (0.0, 0.0, 0.0), 30_000.0)
    tr = propagate(st, _vehicle(), None, 400.0, events=(), dt_s=0.1,
                   physics=BoosterPhysics(atmosphere=False, rotation=False))
    assert tr.termination == "ground_impact"
    assert tr.impact is not None
    assert tr.events == ()  # "ground" was not requested


def test_track_arrays_are_aligned_and_monotonic():
    tr = ballistic_return(_ot_separation_state(), _vehicle(), 950.0,
                          dt_s=1.0, sample_s=5.0, events=("apogee",),
                          physics=BoosterPhysics(gravity="point_mass",
                                                 density_model="ussa1976"))
    n = len(tr.times_s)
    for arr in (tr.mass_kg, tr.position_ecef_m, tr.velocity_ecef_m_s, tr.height_m_arr,
                tr.throttle, tr.mach, tr.dynamic_pressure_pa):
        assert len(arr) == n
    assert np.all(np.diff(tr.times_s) > 0)
    # the propagator's ruler is entry.height_m (first order in f); the
    # geodetic-exact fixture altitude reads ~20 m higher at 189 km
    assert tr.height_m_arr[0] == pytest.approx(
        entry_height_m(np.asarray(_ot_separation_state().position_m)), abs=1.0)
    assert tr.max_dynamic_pressure_pa > 1e4
    assert tr.max_proper_g > 1.0


def test_body_track_bridge_and_event_mapping():
    tr = ballistic_return(_ot_separation_state(), _vehicle(), 950.0,
                          dt_s=1.0, sample_s=5.0, events=("apogee", "ground"),
                          physics=BoosterPhysics(gravity="point_mass",
                                                 density_model="ussa1976"))
    bt = tr.to_body_track("GCRS", body_id="stage1_debris")
    assert bt.frame == "GCRS" and bt.body_id == "stage1_debris"
    assert bt.epoch_utc.endswith("Z")
    assert len(bt.t_s) == len(bt.position_m) == len(tr.times_s)
    assert set(bt.extra_columns) == {"throttle", "mach", "dynamic_pressure_pa",
                                     "altitude_m", "latitude_deg", "longitude_deg"}
    assert bt.extra_columns["altitude_m"] == pytest.approx(tuple(tr.height_m_arr))
    evs = tr.to_contract_events(body_id="stage1_debris")
    assert set(e.kind for e in evs) <= set(EVENT_KINDS)
    kinds = [e.kind for e in evs]
    assert "impact" in kinds  # ground -> impact
    with pytest.raises(frames.UnknownFrameError):
        tr.to_body_track("MARS")
    # an ECEF BodyTrack is a pass-through copy of the flown state
    bt_e = tr.to_body_track("ECEF")
    assert np.allclose(np.asarray(bt_e.position_m), tr.position_ecef_m)


def test_gcrs_input_state_matches_ecef_input():
    """The same physical separation state expressed in GCRS must reproduce
    the ECEF-defined run (the frame bridge is the only difference)."""
    st_e = _ot_separation_state()
    r = np.asarray(st_e.position_m).reshape(1, 3)
    v = np.asarray(st_e.velocity_m_s).reshape(1, 3)
    r_g, v_g = frames.to_gcrs(r, v, "ECEF", EPOCH, np.array([st_e.t_s]))
    st_g = BoosterState(st_e.t_s, EPOCH, "GCRS", tuple(float(c) for c in r_g[0]),
                        tuple(float(c) for c in v_g[0]), st_e.mass_kg)
    ph = BoosterPhysics(gravity="point_mass", density_model="ussa1976")
    kw = dict(events=(), dt_s=1.0, sample_s=5.0, physics=ph)
    a = ballistic_return(st_e, _vehicle(), 950.0, **kw)
    b = ballistic_return(st_g, _vehicle(), 950.0, **kw)
    assert a.impact.t_s == pytest.approx(b.impact.t_s, abs=0.01)
    assert a.impact.latitude_deg == pytest.approx(b.impact.latitude_deg, abs=1e-5)
    assert a.impact.longitude_deg == pytest.approx(b.impact.longitude_deg, abs=1e-5)


def test_separation_state_from_run_roundtrip():
    """separation_state_from_run reads the first sample of the named track
    back out of a BodyTrack -- and says so cleanly when the body is absent."""
    tr = ballistic_return(_ot_separation_state(), _vehicle(), 300.0,
                          dt_s=1.0, sample_s=5.0,
                          physics=BoosterPhysics(gravity="point_mass",
                                                 density_model="ussa1976"))
    bt = tr.to_body_track("GCRS", body_id="stage1_debris")
    run = types.SimpleNamespace(run_id="roundtrip", tracks=(bt,))
    st = separation_state_from_run(run)
    assert st.frame == "GCRS" and st.mass_kg == OT_SEPARATION["mass"]
    assert st.t_s == pytest.approx(OT_SEPARATION["t_s"])
    with pytest.raises(RecoveryError, match="no track"):
        separation_state_from_run(run, body_id="booster")


# ---------------------------------------------------------------------------
# vehicle YAML booster section
# ---------------------------------------------------------------------------


def test_vehicle_yaml_booster_section_loads_with_provenance():
    m = BoosterModel.load(VEHICLE_YAML)
    assert m.provenance in FLIGHT_PROVENANCE_TAGS
    assert m.dry_mass_kg > 0.0
    assert m.landing_propellant_reserve_kg > 0.0
    assert m.reference_area_m2 == pytest.approx(math.pi / 4.0 * 3.7 ** 2, rel=1e-3)
    assert {"boostback", "landing"} <= set(m.engine.usable_for)
    assert 0.0 < m.engine.throttle_min < 1.0
    assert m.engine.restarts_max >= 2
    # hover authority at dry mass, sea level, full throttle
    sea_level_thrust = (m.engine.thrust_vacuum_n
                        - m.engine.nozzle_exit_area_m2 * 101_325.0)
    assert sea_level_thrust > 2.0 * m.dry_mass_kg * G0_M_S2
    for key, tag in m.provenance_by_field.items():
        assert tag in FLIGHT_PROVENANCE_TAGS, key
    # the OT adapter's keys are untouched: stages still present
    import yaml
    doc = yaml.safe_load(VEHICLE_YAML.read_text(encoding="utf-8"))
    assert len(doc["stages"]) == 2 and "booster" in doc
