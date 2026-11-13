"""Checks for FL-305 closed-loop terminal landing guidance
(orbitopt.flightlab.recovery.landing: TerminalGuidance,
terminal_guidance_command, fly_landing_burn_closed_loop) and its pipeline
switch (``plan_options["landing"]``).

The pure-law tests pin the guidance's own contract -- coast / ignite /
burn / cutoff decisions, the tilt limit enforced in the loop, the thrust
cap -- on hand-computed states, no solver needed. The flight tests fly the
law through the FL-301 propagator and are graded by FL-303's unchanged
touchdown checks (the guidance never grades itself):

* the textbook-RTLS corridor handoff FL-302 produces with the re-pinned
  warm start (3.0 km, 306 m/s down, 0.35 km off, 41.7 t) lands inside
  every limit, reserve included -- the same state FL-303's single open-loop
  plan hits at ~-85 m/s;
* FL-303's own G3 exhibit (flat-feasible, drag-failing) lands closed-loop;
* an infeasible corridor state stays LANDING_INFEASIBLE, never green.

R-05 (epic #217) adds the actuator-aware overlay: the pure-law TVC slew
limit (clipped vs unconstrained commands, rate-bounded step tracking) and,
flown, the :class:`ActuatorMargin` report on the closed-loop result --
constraints held as commanded, metrics in the assumption line and the
touchdown event data, FL-305 baseline measured alongside.
"""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from orbitopt.entry import G0_M_S2
from orbitopt.flightlab.frames import geodetic_to_ecef
from orbitopt.flightlab.recovery import (
    LANDING_INFEASIBLE,
    TERMINAL_GUIDANCE_DEFAULT,
    TERMINAL_GUIDANCE_R05,
    TOUCHDOWN_CHECK_IDS,
    BoosterModel,
    BoosterState,
    LandingBurnEnu,
    RecoveryTarget,
    TerminalGuidance,
    fly_landing_burn,
    fly_landing_burn_closed_loop,
    landing_events,
    terminal_guidance_command,
)
from orbitopt.flightlab.recovery.boostback import _enu_basis
from orbitopt.flightlab.recovery.landing import DEFAULT_LANDING_LIMITS
from orbitopt.flightlab.recovery.pipeline import LANDING_MODES, _landing_flight

REPO_ROOT = Path(__file__).resolve().parents[1]
VEHICLE_YAML = REPO_ROOT / "src/orbitopt/data/flightlab/vehicles/synthetic_two_stage.yaml"
VEHICLE = BoosterModel.load(VEHICLE_YAML)

EPOCH = datetime(2026, 11, 10, 12, 0, 0)
PAD = (28.5619, -80.5772, 3.0)  # the E2E scenario's launch site
PAD_ECEF = np.asarray(geodetic_to_ecef(*PAD), dtype=np.float64)
_BAS = np.vstack([np.asarray(b, dtype=np.float64) for b in _enu_basis(PAD[0], PAD[1])])
SITE = RecoveryTarget(kind="rtls", name="launch_site", latitude_deg=PAD[0],
                      longitude_deg=PAD[1], altitude_m=PAD[2])

G = 9.79
T_MAX = 1.3e6   # ~ sea-level thrust of the synthetic landing engine
T_MIN = 0.4 * T_MAX

# FL-302's corridor handoff for the textbook RTLS separation with the
# FL-305 warm start (pipeline.NOMINAL_PLAN_GUESS), recorded from the chain.
CASE_TEXTBOOK_HANDOFF = ((-351.15, 21.11, 3015.87), (-1.16, 0.15, -306.01), 41660.9)
# FL-303's G3 exhibit, re-sited on this pad (flat-feasible, drag-failing).
CASE_G3 = ((60.0, 0.0, 400.0), (-8.0, 0.0, -55.0), 46000.0)
CASE_INFEASIBLE = ((0.0, 0.0, 150.0), (0.0, 0.0, -450.0), 45000.0)


def _states(case, t0=300.0):
    pos, vel, m0 = case
    enu = LandingBurnEnu(t_s=t0, position_enu_m=pos, velocity_enu_m=vel, mass_kg=m0)
    r = PAD_ECEF + _BAS.T @ np.asarray(pos, dtype=np.float64)
    v = _BAS.T @ np.asarray(vel, dtype=np.float64)
    ecef = BoosterState(t_s=t0, epoch_utc=EPOCH, frame="ECEF",
                        position_m=tuple(float(c) for c in r),
                        velocity_m_s=tuple(float(c) for c in v), mass_kg=m0)
    return enu, ecef


def _cmd(pos, vel, mass=40_000.0, *, ignited=True, guidance=TERMINAL_GUIDANCE_DEFAULT):
    return terminal_guidance_command(pos, vel, mass, gravity_m_s2=G,
                                     thrust_max_n=T_MAX, thrust_min_n=T_MIN,
                                     ignited=ignited, guidance=guidance)


def _tilt_deg(cmd) -> float:
    f = np.asarray(cmd.thrust_enu_n)
    return math.degrees(math.acos(f[2] / np.linalg.norm(f)))


# ---------------------------------------------------------------------------
# the law, as a pure function
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"design_throttle": 0.0}, {"design_throttle": 1.2},
    {"touchdown_speed_m_s": 0.0}, {"tilt_limit_deg": 90.0},
    {"max_thrust_accel_g": 1.0}, {"settle_s": -1.0}, {"cutoff_speed_m_s": 0.0},
    {"tvc_rate_limit_deg_s": -1.0},
])
def test_guidance_policy_rejects_nonsense(kwargs):
    with pytest.raises(ValueError):
        TerminalGuidance(**kwargs)


def test_coasts_until_the_profile_needs_the_design_deceleration():
    # 3 km up at 200 m/s: a_req = (200^2 - 1.5^2) / 6000 ~ 6.7 m/s^2, far
    # below 0.9 * 1.3 MN / 40 t - g ~ 19.5 m/s^2 -> engine stays off
    high = _cmd((0.0, 0.0, 3000.0), (0.0, 0.0, -200.0), ignited=False)
    assert high.phase == "coast" and high.thrust_n == 0.0
    assert high.required_decel_m_s2 == pytest.approx((200.0 ** 2 - 1.5 ** 2) / 6000.0)
    # 1 km up at 200 m/s: a_req ~ 20 m/s^2 > design -> lights
    low = _cmd((0.0, 0.0, 1000.0), (0.0, 0.0, -200.0), ignited=False)
    assert low.phase == "burn" and low.thrust_n > 0.0


def test_vertical_burn_flies_the_constant_deceleration_profile():
    h, vz, m = 800.0, -120.0, 38_000.0
    cmd = _cmd((0.0, 0.0, h), (0.0, 0.0, vz), m)
    a_req = (vz * vz - 1.5 ** 2) / (2.0 * h)
    assert cmd.phase == "burn"
    np.testing.assert_allclose(cmd.thrust_enu_n, (0.0, 0.0, m * (a_req + G)), rtol=1e-12)


def test_tilt_limit_is_enforced_in_the_loop():
    # 1.5 km off the pad, drifting away: the divert wants far more lateral
    # acceleration than the tilt cone allows -- the command must ride the cone
    cmd = _cmd((1500.0, -400.0, 900.0), (40.0, -10.0, -150.0))
    assert cmd.phase == "burn"
    assert _tilt_deg(cmd) == pytest.approx(TERMINAL_GUIDANCE_DEFAULT.tilt_limit_deg, abs=1e-9)
    # and it pushes back toward the pad
    f = np.asarray(cmd.thrust_enu_n)
    assert f[0] < 0.0 and f[1] > 0.0


def test_thrust_is_capped_at_the_acceleration_limit():
    m = 35_000.0
    cmd = _cmd((0.0, 0.0, 50.0), (0.0, 0.0, -150.0), m)  # a_req ~ 225 m/s^2
    assert cmd.thrust_n == pytest.approx(m * 4.5 * G0_M_S2)


def test_no_hover_cutoff_only_when_the_ballistic_arrival_is_soft():
    m = 35_000.0  # floor 520 kN > weight 343 kN: no hover
    # 1 m up at -1.2 m/s: arrival sqrt(1.44 + 2 g) ~ 4.6 m/s -> keep burning
    assert _cmd((0.0, 0.0, 1.0), (0.0, 0.0, -1.2), m).phase == "burn"
    # 0.15 m up at -1.2 m/s: arrival ~ 2.0 m/s <= 2.2 -> final cutoff
    cut = _cmd((0.0, 0.0, 0.15), (0.0, 0.0, -1.2), m)
    assert cut.phase == "cutoff" and cut.thrust_n == 0.0
    # same state on a vehicle that can hover (floor below weight): the
    # profile is still followable, so no cutoff
    hover = terminal_guidance_command((0.0, 0.0, 0.15), (0.0, 0.0, -1.2), m,
                                      gravity_m_s2=G, thrust_max_n=T_MAX,
                                      thrust_min_n=0.1 * T_MAX, ignited=True)
    assert hover.phase == "burn"


def test_divert_aware_ignition_lights_early_for_a_large_offset():
    state = ((0.0, 0.0, 3000.0), (0.0, 0.0, -300.0))
    off = ((2500.0, 0.0, 3000.0), (0.0, 0.0, -300.0))
    assert _cmd(*state, ignited=False).phase == "coast"
    assert _cmd(*off, ignited=False).phase == "burn"
    lazy = TerminalGuidance(divert_ignition_fraction=0.0)
    assert _cmd(*off, ignited=False, guidance=lazy).phase == "coast"


def test_lateral_drift_is_damped_in_the_settle_window():
    # t_go = 2 * 10 / (5 + 1.5) ~ 3.1 s < settle + min t_go: PD regime
    cmd = _cmd((0.0, 0.0, 10.0), (2.0, 0.0, -5.0))
    assert cmd.phase == "burn" and cmd.thrust_enu_n[0] < 0.0


# ---------------------------------------------------------------------------
# R-05 (epic #217): TVC slew limit as a pure-law actuator constraint
# ---------------------------------------------------------------------------

OFF_PAD = ((1500.0, -400.0, 900.0), (40.0, -10.0, -150.0))  # divert rides the cone


def _cmd_rate(pos, vel, prev_dir, dt, guidance):
    return terminal_guidance_command(pos, vel, 40_000.0, gravity_m_s2=G,
                                     thrust_max_n=T_MAX, thrust_min_n=T_MIN,
                                     ignited=True, guidance=guidance,
                                     prev_dir_enu=prev_dir, dt_s=dt)


def test_tvc_rate_limit_clips_the_direction_slew():
    r = TerminalGuidance(tvc_rate_limit_deg_s=20.0)
    cmd = _cmd_rate(*OFF_PAD, (0.0, 0.0, 1.0), 0.1, r)
    assert cmd.phase == "burn" and cmd.rate_clipped
    assert _tilt_deg(cmd) == pytest.approx(2.0, abs=1e-9)  # 20 deg/s x 0.1 s
    # the unconstrained law rides the whole 30 deg cone on this state ...
    plain = _cmd(*OFF_PAD)
    assert not plain.rate_clipped
    assert _tilt_deg(plain) == pytest.approx(30.0, abs=1e-9)
    # ... and with the rate limit at 0 (the FL-305 default) the law returns
    # the baseline command to the bit
    off = _cmd_rate(*OFF_PAD, (0.0, 0.0, 1.0), 0.1, TERMINAL_GUIDANCE_DEFAULT)
    assert off.thrust_enu_n == plain.thrust_enu_n
    # a long dt gives the slew more room than the cone: the cone binds, no clip
    slow = _cmd_rate(*OFF_PAD, (0.0, 0.0, 1.0), 5.0, r)
    assert _tilt_deg(slow) == pytest.approx(30.0, abs=1e-9)
    assert not slow.rate_clipped


def test_rate_limited_direction_tracks_the_demand_within_the_actuator_rate():
    # step the law from gimbal neutral toward the cone-bound divert demand:
    # every step turns by at most rate * dt, and the tilt converges on the
    # cone -- the constraint is tracked, not evaded
    r = TerminalGuidance(tvc_rate_limit_deg_s=20.0)
    prev = np.array([0.0, 0.0, 1.0])
    tilts = []
    for _ in range(20):
        cmd = _cmd_rate(*OFF_PAD, tuple(prev), 0.1, r)
        d = np.asarray(cmd.thrust_enu_n)
        d = d / np.linalg.norm(d)
        assert math.degrees(math.acos(float(np.dot(prev, d)))) / 0.1 <= 20.0 + 1e-6
        tilts.append(_tilt_deg(cmd))
        prev = d
    assert tilts[0] == pytest.approx(2.0, abs=1e-9)
    assert tilts[-1] == pytest.approx(30.0, abs=1e-6)


# ---------------------------------------------------------------------------
# pipeline switch
# ---------------------------------------------------------------------------

def test_landing_options_select_and_validate_the_flow():
    assert LANDING_MODES == ("closed-loop", "single")
    assert _landing_flight({"mode": "single"}) is fly_landing_burn
    assert callable(_landing_flight(None))
    assert callable(_landing_flight({"design_throttle": 0.85}))
    with pytest.raises(ValueError, match="landing mode"):
        _landing_flight({"mode": "hover"})
    with pytest.raises(ValueError, match="single"):
        _landing_flight({"mode": "single", "design_throttle": 0.8})
    with pytest.raises(TypeError):
        _landing_flight({"desing_throttle": 0.8})
    with pytest.raises(ValueError):
        _landing_flight({"design_throttle": 2.0})


# ---------------------------------------------------------------------------
# flown through the propagator, graded by FL-303's checks
# ---------------------------------------------------------------------------

def _fly(case, **kwargs):
    pytest.importorskip("gfold")
    enu, ecef = _states(case)
    return fly_landing_burn_closed_loop(enu, ecef, SITE, VEHICLE, **kwargs)


def test_textbook_corridor_handoff_lands_closed_loop():
    res = _fly(CASE_TEXTBOOK_HANDOFF)
    ids = tuple(c.id for c in res.checks)
    assert ids == TOUCHDOWN_CHECK_IDS  # the FL-303 referee, unchanged
    assert res.landed, res.reason_codes
    assert res.reason_codes == ()
    lim = DEFAULT_LANDING_LIMITS
    assert abs(res.vertical_speed_m_s) <= lim.vertical_speed_max_m_s
    assert res.lateral_speed_m_s <= lim.lateral_speed_max_m_s
    assert res.position_error_m <= lim.position_tol_m
    assert res.max_tilt_deg <= TERMINAL_GUIDANCE_DEFAULT.tilt_limit_deg + 1e-9
    margin = {c.id: c for c in res.checks}["landing-propellant-reserve"].value
    assert margin > 0.0
    assert res.ignitions_used == 1  # one light, one final cutoff
    assert any("FL-305" in a for a in res.assumptions)


def test_g3_exhibit_fails_open_loop_but_lands_closed_loop():
    pytest.importorskip("gfold")
    enu, ecef = _states(CASE_G3)
    open_loop = fly_landing_burn(enu, ecef, SITE, VEHICLE, solver="gfold",
                                 tof_s=12.0, n=16, glide_slope_deg=20.0)
    assert not open_loop.landed
    closed = _fly(CASE_G3)
    assert closed.landed, closed.reason_codes


def test_infeasible_corridor_state_is_never_green():
    res = _fly(CASE_INFEASIBLE)
    assert not res.feasible and not res.landed
    assert res.reason_codes == (LANDING_INFEASIBLE,)
    assert res.track is None


# ---------------------------------------------------------------------------
# R-05: flown actuator constraints and control-margin reporting
# ---------------------------------------------------------------------------

def test_r05_preset_lands_with_the_actuator_constraints_held():
    res = _fly(CASE_TEXTBOOK_HANDOFF, guidance=TERMINAL_GUIDANCE_R05)
    assert res.landed, res.reason_codes
    assert res.ignitions_used == 1
    m = res.actuator_margin
    assert m is not None
    # every constraint the guidance promised, held as *commanded*:
    assert m.tvc_rate_limit_deg_s == 20.0
    assert m.slew_deg_s_max <= m.tvc_rate_limit_deg_s + 1e-6  # TVC rate
    assert m.tilt_margin_deg_min >= -1e-9                     # TVC angle
    assert m.throttle_floor_room_min >= -1e-12                # floor
    assert m.throttle_headroom_min >= -1e-12                  # ceiling
    assert m.burn_time_s > 0.0
    # the metrics travel in the run's report: an assumption line and the
    # touchdown event data (the referee checks stay FL-303's, unchanged)
    assert any("R-05 actuator margins" in a for a in res.assumptions)
    touch = [e for e in landing_events(res) if e.kind in ("touchdown", "impact")]
    assert touch and "control_margin" in touch[0].data
    assert touch[0].data["control_margin"]["tvc_rate_limit_deg_s"] == 20.0


def test_closed_loop_reports_the_baseline_control_margin_too():
    res = _fly(CASE_TEXTBOOK_HANDOFF)  # FL-305 default guidance, unchanged
    m = res.actuator_margin
    assert res.landed and m is not None and m.tvc_rate_limit_deg_s == 0.0
    # ADR flightlab-0002's reserved ignition authority (1 - design_throttle
    # = 10 %), *measured* as flown, not asserted by fiat
    assert 0.05 <= m.throttle_headroom_at_ignition <= 0.2
    # and the honest indictment of the unlimited baseline: a TVC slew no real
    # actuator delivers, which the R-05 preset above keeps at its 20 deg/s
    assert m.slew_deg_s_max > TERMINAL_GUIDANCE_R05.tvc_rate_limit_deg_s
