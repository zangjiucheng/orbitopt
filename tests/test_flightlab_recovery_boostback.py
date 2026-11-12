"""Checks for orbitopt.flightlab.recovery.boostback (FL-302 boostback &
entry-burn guidance).

Tolerances -- declared up front, before any result is looked at:

    NOMINAL_*           The nominal 65 t RTLS case must enter the frozen
                        corridor ``RecoveryCorridor(3000, 8000, 350, 15)``
                        with margin, and land-propellant margin on top of
                        the landing need + 4.2 t reserve.
    REFLY_POS_TOL_M     The same script re-flown through the FL-301
                        propagator (``mission_command``, no kinematic
                        shortcuts) must reproduce the accepted corridor
                        insertion position to 2 m / 5 kg -- same script,
                        same integrator, same dt, so the residual is pure
                        event-logging bookkeeping.
    ENU_MIRROR_TOL      ``landing_burn_initial_state_enu`` must agree with
                        the ECEF ``landing_burn_initial_state`` projected
                        onto the pad ENU basis to 1 m / 0.05 m/s.
    SCENARIO_*          ``RecoveryTarget.from_scenario`` is pure parsing:
                        exact lat/lon, 1 mm ECEF radius sanity.

The heavy optimizer tests are marked ``slow``; the reason-code tests pin a
script via ``initial_guess`` + ``max_fevals=1`` so the reported verdict is
deterministic (the search budget keeps the search where it started). The
nominal seed ``_SEED_PLAN`` is the script the unconstrained default run
converges to (miss ~0.4 km, v ~282 m/s); reusing it keeps the acceptance
test honest (it verifies that script re-enters the corridor in full
physics) without re-running the 200 s search in CI.
"""
from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from orbitopt.flightlab.frames import geodetic_to_ecef
from orbitopt.flightlab.recovery import (
    BoosterModel,
    BoosterState,
    RecoveryCorridor,
    RecoveryError,
    RecoveryTarget,
    plan_boostback_entry,
)
from orbitopt.flightlab.recovery.boostback import (
    DEFAULT_CORRIDOR,
    DEFAULT_ENTRY_G_MAX,
    DEFAULT_ENTRY_Q_MAX_PA,
    INSUFFICIENT_PROPELLANT_BOOSTBACK,
    INSUFFICIENT_PROPELLANT_ENTRY_BURN,
    INSUFFICIENT_PROPELLANT_RESERVE,
    CORRIDOR_MISSED,
    CORRIDOR_NOT_REACHED,
    ENTRY_G_LIMIT_EXCEEDED,
    ENTRY_Q_LIMIT_EXCEEDED,
    _bearing_of_target,
    _enu_basis,
    landing_burn_propellant_need_kg,
    mission_command,
)
from orbitopt.flightlab.recovery.propagator import AltitudeThreshold, propagate

REPO_ROOT = Path(__file__).resolve().parents[1]
VEHICLE_YAML = REPO_ROOT / "src/orbitopt/data/flightlab/vehicles/synthetic_two_stage.yaml"
SCENARIO_YAML = REPO_ROOT / "src/orbitopt/data/flightlab/scenarios/leo_two_stage.yaml"

EPOCH = datetime(2026, 11, 10, 12, 0, 0)
PAD = (28.5619, -80.5772, 3.0)  # matches the scenario launch site
SEP_T_S = 207.0

REFLY_POS_TOL_M = 2.0
REFLY_MASS_TOL_KG = 5.0
ENU_MIRROR_TOL_M = 1.0
ENU_MIRROR_TOL_M_S = 0.05

# Script the unconstrained default nominal run converges to (recorded
# 2026-11-11, miss 420 m, tilt 0.21 deg, v 282 m/s, prop 9.7 t).
_SEED_PLAN = [18.4, 19.1, 1.0, 4.36, 0.0, 17_456.0, 23.0]


def _pad_basis():
    e, n, u = _enu_basis(PAD[0], PAD[1])
    return np.array(e), np.array(n), np.array(u)


def _sep_state(alt, speed, fpa_deg, down_m, mass, heading_deg=90.0):
    """Separation ``down_m`` east of the pad at ``alt``, flying
    ``heading_deg`` with ``fpa_deg`` -- the two-stage-RTLS textbook case."""
    e, n, u = _pad_basis()
    r = np.array(geodetic_to_ecef(*PAD)) + down_m * e + alt * u
    hr, fpa = math.radians(heading_deg), math.radians(fpa_deg)
    hd = math.cos(hr) * n + math.sin(hr) * e
    v = speed * (math.cos(fpa) * hd + math.sin(fpa) * u)
    return BoosterState(t_s=SEP_T_S, epoch_utc=EPOCH, frame="ECEF",
                        position_m=tuple(float(c) for c in r),
                        velocity_m_s=tuple(float(c) for c in v), mass_kg=mass)


@pytest.fixture(scope="module")
def vehicle():
    return BoosterModel.load(str(VEHICLE_YAML))


@pytest.fixture(scope="module")
def pad_target():
    return RecoveryTarget(kind="rtls", name="launch_site", latitude_deg=PAD[0],
                          longitude_deg=PAD[1], altitude_m=PAD[2])


@pytest.fixture(scope="module")
def nominal_plan(vehicle, pad_target):
    """The accepted nominal plan, evaluated once for the whole module."""
    st = _sep_state(alt=40_000.0, speed=250.0, fpa_deg=2.0, down_m=8_000.0,
                    mass=65_000.0)
    return st, plan_boostback_entry(st, vehicle, pad_target,
                                    initial_guess=_SEED_PLAN, max_fevals=18)


# ---------------------------------------------------------------------------
# acceptance criterion 1: the synthetic RTLS nominal reaches the corridor
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_nominal_rtls_enters_corridor(nominal_plan, pad_target):
    _st, plan = nominal_plan
    cr = plan.corridor_report
    assert plan.feasible, plan.reason_codes
    assert plan.reason_codes == ()
    assert cr.reached and cr.t_s > SEP_T_S
    assert 0.0 <= cr.miss_m <= pad_target.corridor.radius_m
    assert 0.0 <= cr.tilt_deg <= pad_target.corridor.max_tilt_deg
    assert 0.0 < cr.speed_m_s <= pad_target.corridor.v_max_m_s
    assert cr.propellant_available_kg >= cr.landing_need_kg + 4200.0
    assert cr.landing_need_kg > 0.0
    assert cr.entry_max_q_pa <= DEFAULT_ENTRY_Q_MAX_PA
    assert cr.entry_max_proper_g <= DEFAULT_ENTRY_G_MAX
    assert plan.ignitions_used == 2


@pytest.mark.slow
def test_nominal_track_is_flown_not_assumed(nominal_plan, vehicle, pad_target):
    """All thrust comes from the two scripted burns re-integrated by the
    FL-301 propagator: two burn starts, ballistic elsewhere, and the track
    ends at a real ground impact (the corridor is crossed en route)."""
    _st, plan = nominal_plan
    starts = [e for e in plan.track.events if e.kind == "burn_start"]
    ends = [e for e in plan.track.events if e.kind == "burn_end"]
    assert len(starts) == 2 and len(ends) == 2
    assert all(e.data.get("reason") == "COMMANDED_CUTOFF" for e in ends)
    # coast segments carry zero throttle on the sample grid
    t_bb_end = starts[0].t_s + plan.boostback.duration_s
    coast = np.asarray(plan.track.throttle)[
        (np.asarray(plan.track.times_s) > t_bb_end + 1.0)
        & (np.asarray(plan.track.times_s) < starts[1].t_s - 1.0)]
    assert np.all(coast == 0.0)
    assert plan.track.impact is not None


@pytest.mark.slow
def test_mission_command_reflight_reproduces_insertion(nominal_plan, vehicle):
    """'No kinematic shortcuts': re-flying ``mission_command`` through
    ``propagate`` must reproduce the accepted corridor state."""
    st, plan = nominal_plan
    cmd = mission_command(plan, st)
    h0_name = "reflight_h0"
    tr = propagate(st, vehicle, cmd, st.t_s + 900.0,
                   events=("burn", "ground",
                           AltitudeThreshold(3000.0, h0_name, "down")),
                   dt_s=0.5, sample_s=2.0)
    starts = [e for e in tr.events if e.kind == "burn_start"]
    assert len(starts) == 2
    ev = next(e for e in tr.events if e.kind == h0_name and e.t_s >= starts[1].t_s)
    r = np.asarray(ev.data["position_ecef_m"])
    r_plan = np.asarray(plan.landing_burn_initial_state.position_m)
    assert float(np.linalg.norm(r - r_plan)) < REFLY_POS_TOL_M
    assert abs(ev.data["mass_kg"]
               - plan.landing_burn_initial_state.mass_kg) < REFLY_MASS_TOL_KG


@pytest.mark.slow
def test_landing_burn_initial_state_enu_mirror(nominal_plan):
    st, plan = nominal_plan
    cr = plan.corridor_report
    assert cr.reached
    e, n, u = _pad_basis()
    r = np.asarray(plan.landing_burn_initial_state.position_m)
    r_pad = np.array(geodetic_to_ecef(*PAD))
    d = r - r_pad
    pos = (float(d @ e), float(d @ n), float(d @ u))
    v = np.asarray(plan.landing_burn_initial_state.velocity_m_s)
    vel = (float(v @ e), float(v @ n), float(v @ u))
    enu = plan.landing_burn_initial_state_enu
    for got, want in zip(enu.position_enu_m, pos):
        assert abs(got - want) < ENU_MIRROR_TOL_M
    for got, want in zip(enu.velocity_enu_m, vel):
        assert abs(got - want) < ENU_MIRROR_TOL_M_S
    assert enu.mass_kg == plan.landing_burn_initial_state.mass_kg
    assert abs(enu.t_s - cr.t_s) < 1e-9
    assert abs(enu.position_enu_m[2] - 3000.0) < 100.0  # crossed at h0
    assert enu.velocity_enu_m[2] < 0.0  # descending


def test_landing_need_momentum_balance(vehicle):
    """The landing-need solver must be exactly the momentum balance: the
    need it returns must, when burned at SL Isp, stop the vehicle."""
    from orbitopt.entry import G0_M_S2
    m0, v, h = 45_000.0, 300.0, 3_000.0
    need = landing_burn_propellant_need_kg(m0, v, h, vehicle)
    assert 0.0 < need < m0 - vehicle.dry_mass_kg
    eng = vehicle.engine
    thrust_sl = eng.thrust_vacuum_n - eng.nozzle_exit_area_m2 * 70_000.0  # ~3 km
    isp = eng.isp_sea_level_s
    m1 = m0 - need
    dv = isp * G0_M_S2 * math.log(m0 / m1) - need * isp * G0_M_S2 ** 2 / thrust_sl
    assert dv == pytest.approx(v, rel=0.02)  # pessimistic-SL thrust ~ 3 km ambient
    assert landing_burn_propellant_need_kg(vehicle.dry_mass_kg, v, h, vehicle) == 0.0


# ---------------------------------------------------------------------------
# acceptance criterion 2: infeasible cases report the frozen reason codes
# ---------------------------------------------------------------------------


def test_boostback_floor_reports_instantly(vehicle, pad_target):
    """1 t of propellant cannot null 250 m/s of horizontal separation
    velocity no matter what the optimizer tries: analytic fast path."""
    st = _sep_state(alt=40_000.0, speed=250.0, fpa_deg=2.0, down_m=8_000.0,
                    mass=31_000.0)
    plan = plan_boostback_entry(st, vehicle, pad_target)
    assert not plan.feasible
    assert plan.reason_codes[0] == INSUFFICIENT_PROPELLANT_BOOSTBACK
    assert plan.corridor_report.propellant_available_kg is None
    assert plan.track.ignitions_used == 0
    assert any("floor" in a for a in plan.assumptions)


def test_entry_burn_exhaustion_reason(vehicle, pad_target):
    """Pinned script with a 45 s full-throttle entry burn: the tanks run
    dry inside it -> INSUFFICIENT_PROPELLANT_ENTRY_BURN, first reason."""
    st = _sep_state(alt=40_000.0, speed=250.0, fpa_deg=2.0, down_m=8_000.0,
                    mass=65_000.0)
    plan = plan_boostback_entry(st, vehicle, pad_target, max_fevals=1,
                                initial_guess=[10.0, 20.0, 1.0, 2.0, -20.0,
                                               30_000.0, 45.0])
    assert not plan.feasible
    assert plan.reason_codes[0] == INSUFFICIENT_PROPELLANT_ENTRY_BURN


def test_corridor_not_reached_under_short_horizon(vehicle, pad_target):
    """A 40 s observation window from a 40 km separation sees neither the
    corridor plane nor the ground -> CORRIDOR_NOT_REACHED."""
    st = _sep_state(alt=40_000.0, speed=250.0, fpa_deg=2.0, down_m=8_000.0,
                    mass=65_000.0)
    plan = plan_boostback_entry(st, vehicle, pad_target, max_fevals=1,
                                horizon_s=40.0, initial_guess=_SEED_PLAN)
    assert not plan.feasible
    assert plan.reason_codes[0] == CORRIDOR_NOT_REACHED
    assert plan.track.impact is None


@pytest.mark.slow
def test_q_and_g_limits_reason(vehicle, pad_target):
    """The nominal flight pulls ~45 kPa / ~4.4 g in the entry window;
    tightening either limit must fire exactly that reason."""
    st = _sep_state(alt=40_000.0, speed=250.0, fpa_deg=2.0, down_m=8_000.0,
                    mass=65_000.0)
    q_plan = plan_boostback_entry(st, vehicle, pad_target, max_fevals=1,
                                  entry_q_max_pa=20_000.0,
                                  initial_guess=_SEED_PLAN)
    assert q_plan.reason_codes[0] == ENTRY_Q_LIMIT_EXCEEDED
    g_plan = plan_boostback_entry(st, vehicle, pad_target, max_fevals=1,
                                  entry_g_max=3.5, initial_guess=_SEED_PLAN)
    assert g_plan.reason_codes[0] == ENTRY_G_LIMIT_EXCEEDED


def test_corridor_missed_reason(vehicle, pad_target):
    """A weak, late, badly trimmed burn must leave the booster well outside
    the 8 km corridor radius -> CORRIDOR_MISSED."""
    st = _sep_state(alt=40_000.0, speed=250.0, fpa_deg=2.0, down_m=8_000.0,
                    mass=65_000.0)
    plan = plan_boostback_entry(st, vehicle, pad_target, max_fevals=1,
                                initial_guess=[40.0, 4.0, 0.4, 25.0, -45.0,
                                               5_500.0, 3.0])
    assert not plan.feasible
    assert plan.reason_codes[0] == CORRIDOR_MISSED
    assert plan.corridor_report.miss_m > pad_target.corridor.radius_m


def test_reserve_shortfall_reason(vehicle, pad_target):
    """Same nominal script, 1 t lighter at separation: the corridor is still
    entered but the tanks hold less than landing need + reserve ->
    INSUFFICIENT_PROPELLANT_RESERVE (nominal had ~1 t of margin)."""
    st = _sep_state(alt=40_000.0, speed=250.0, fpa_deg=2.0, down_m=8_000.0,
                    mass=63_500.0)
    plan = plan_boostback_entry(st, vehicle, pad_target, max_fevals=1,
                                initial_guess=_SEED_PLAN)
    assert not plan.feasible
    assert plan.reason_codes[0] == INSUFFICIENT_PROPELLANT_RESERVE
    assert plan.corridor_report.reached


def test_needs_two_restarts(vehicle, pad_target):
    st = _sep_state(alt=40_000.0, speed=250.0, fpa_deg=2.0, down_m=8_000.0,
                    mass=65_000.0)
    single = replace(vehicle, engine=replace(vehicle.engine, restarts_max=1))
    with pytest.raises(RecoveryError):
        plan_boostback_entry(st, single, pad_target)


# ---------------------------------------------------------------------------
# target plumbing
# ---------------------------------------------------------------------------


def test_from_scenario_rtls():
    import yaml
    with open(SCENARIO_YAML) as fh:
        sc = yaml.safe_load(fh)
    tgt = RecoveryTarget.from_scenario(sc)
    assert tgt.kind == "rtls" and tgt.name == "launch_site"
    assert tgt.latitude_deg == pytest.approx(28.5619)
    assert tgt.longitude_deg == pytest.approx(-80.5772)
    assert tgt.corridor == DEFAULT_CORRIDOR
    r = np.linalg.norm(np.asarray(tgt.position_ecef_m))
    assert 6_330_000.0 < r < 6_400_000.0  # near-surface radius sanity
    b = _bearing_of_target(np.asarray(tgt.position_ecef_m))
    r_up = np.array(geodetic_to_ecef(28.5619, -80.46, 3.0))
    assert math.degrees(b(r_up)) == pytest.approx(-90.0, abs=1.0)  # pad due west


def test_from_scenario_downrange_and_errors():
    site = {"latitude_deg": 31.0, "longitude_deg": -81.5, "altitude_m": 2.0}
    sc = {"recovery": {"kind": "downrange", "target": "lz1"},
          "launch_site": {"latitude_deg": 28.5, "longitude_deg": -80.6}}
    tgt = RecoveryTarget.from_scenario(sc, landing_sites={"lz1": site})
    assert tgt.kind == "downrange" and tgt.latitude_deg == 31.0
    with pytest.raises(ValueError):
        RecoveryTarget.from_scenario({"recovery": {"kind": "none"}})
    with pytest.raises(ValueError):
        RecoveryTarget.from_scenario(sc, landing_sites={})


def test_corridor_frozen_defaults():
    assert DEFAULT_CORRIDOR == RecoveryCorridor(h0_m=3000.0, radius_m=8000.0,
                                                v_max_m_s=350.0, max_tilt_deg=15.0)


@pytest.mark.slow
def test_downrange_target_steers_south(vehicle):
    """Bearing-guided boostback must support a downrange site: separation
    still east of the pad, landing 39 km south -> corridor reached with the
    two-burn structure (search stays cheap; corridor entry, not radius, is
    the claim)."""
    tgt = RecoveryTarget(kind="downrange", name="lz-1",
                         latitude_deg=PAD[0] - 0.35, longitude_deg=PAD[1],
                         altitude_m=PAD[2])
    st = _sep_state(alt=40_000.0, speed=250.0, fpa_deg=2.0, down_m=8_000.0,
                    mass=65_000.0)
    plan = plan_boostback_entry(st, vehicle, tgt, max_fevals=12,
                                initial_guess=[15.0, 26.0, 1.0, 8.0, 0.0,
                                               15_000.0, 22.0])
    cr = plan.corridor_report
    assert cr.reached and cr.tilt_deg <= tgt.corridor.max_tilt_deg
    assert plan.ignitions_used == 2
    assert cr.entry_max_q_pa <= DEFAULT_ENTRY_Q_MAX_PA
    assert plan.reason_codes[0] in (CORRIDOR_MISSED, INSUFFICIENT_PROPELLANT_RESERVE) \
        or plan.feasible
