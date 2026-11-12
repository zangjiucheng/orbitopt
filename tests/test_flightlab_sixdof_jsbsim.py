"""FL-401 (#206): JSBSim 6-DOF spike -- analytic checks on the minimal
liquid stage (orbitopt/data/flightlab/jsbsim/aircraft/spike_stage).

Acceptance thresholds are declared up front (module constants below) and the
tests only compare against them. Skipped without ``jsbsim`` (opt-in pip dep:
``pip install -r envs/sixdof-requirements.txt``).
"""
from __future__ import annotations

import math

import pytest

pytest.importorskip("jsbsim")

from orbitopt.flightlab.contract import BodyTrack, REQUIRED_CSV_COLUMNS  # noqa: E402
from orbitopt.flightlab.sixdof.jsbsim_driver import (  # noqa: E402
    EXTRA_COLUMNS,
    G0_M_S2,
    SPIKE_STAGE,
    InitialConditions,
    SixDofSim,
)

# --------------------------------------------------------------------------
# Declared thresholds
# --------------------------------------------------------------------------
DT_S = 0.01
#: Vertical hop: relative error of the Earth-relative vertical velocity gain
#: vs. Tsiolkovsky minus integrated (gravity - centrifugal) loss.
HOP_DV_REL_TOL = 1e-3
#: Vacuum thrust vs. Isp * g0 * mdot (relative).
THRUST_REL_TOL = 1e-4
#: Mass vs. m0 - mdot*t: JSBSim's mass balance reports the mass one frame
#: stale (it runs before propulsion drains the tanks), so allow one frame of
#: flow plus 1 kg of XML lbs<->kg rounding.
MASS_FRAMES_LAG = 1
MASS_ABS_TOL_KG = 1.0
#: TVC moment vs. T * sin(delta) * (x_thruster - x_cg) (relative).
MOMENT_REL_TOL = 1e-2
#: Body pitch rate vs. integral of M / Iyy (relative).
RATE_REL_TOL = 2e-2
#: Actuator: first-order lag (tau, with Tustin's half-step delay) error as a
#: fraction of the command; rate/travel limits honoured to 1e-6 rad(/s).
LAG_ABS_TOL_FRAC = 1e-2
LIMIT_TOL = 1e-6
#: Timestep halving (0.01 s -> 0.005 s, 20 s gimballed burn), final state.
HALVING_ATTITUDE_TOL_DEG = 0.05
HALVING_POS_REL_TOL = 2e-3  # of the distance flown
HALVING_VEL_REL_TOL = 2e-3  # of the speed
HALVING_RATE_REL_TOL = 1e-4  # body rates
HALVING_MASS_TOL_KG = 1.01 * SPIKE_STAGE["mdot_max_kg_s"] * DT_S / 2  # one (fine) frame of flow

OMEGA_EARTH = 7.2921150e-5
THRUSTER_X_M = 42.0  # structural x (aft of the nose) in the XML
HOP_IC = InitialConditions(lat_geod_deg=0.0, lon_deg=0.0, alt_m=300e3, pitch_deg=90.0)


def _hop(dt=DT_S, duration=30.0, throttle=1.0):
    sim = SixDofSim(dt)
    sim.initialize(HOP_IC)
    sim.ignite(throttle)
    sim.run_for(duration)
    return sim


def _radius_m(s):
    return math.sqrt(s["x_m"] ** 2 + s["y_m"] ** 2 + s["z_m"] ** 2)


# --------------------------------------------------------------------------
# Vertical vacuum-ish hop
# --------------------------------------------------------------------------


@pytest.mark.parametrize("throttle", [1.0, 0.5])
def test_vertical_hop_matches_tsiolkovsky_with_constant_mdot(throttle):
    sim = _hop(throttle=throttle)
    s = sim.samples
    mdot = throttle * SPIKE_STAGE["mdot_max_kg_s"]
    m0 = SPIKE_STAGE["dry_mass_kg"] + SPIKE_STAGE["propellant_mass_kg"]
    assert s[0]["mass_kg"] == pytest.approx(m0, abs=MASS_ABS_TOL_KG)

    # Constant mdot: m(t) = m0 - mdot*t (reported one frame stale).
    for row in s[1::100]:
        expected = m0 - mdot * row["t_s"]
        assert abs(row["mass_kg"] - expected) <= MASS_FRAMES_LAG * mdot * DT_S + MASS_ABS_TOL_KG

    # Thrust = Isp * g0 * mdot (vacuum: token nozzle exit area). The engine
    # lights one frame after the ignition command (frame 1 still reads 0).
    assert s[1]["thrust_n"] == 0.0
    for row in s[2::300]:
        assert row["thrust_n"] == pytest.approx(SPIKE_STAGE["isp_s"] * G0_M_S2 * mdot, rel=THRUST_REL_TOL)
        assert row["throttle"] == pytest.approx(throttle)

    # Tsiolkovsky along the local vertical, minus (gravity - centrifugal)
    # loss integrated over JSBSim's own gravity samples (equatorial launch).
    loss = 0.0
    for a, b in zip(s, s[1:]):
        ga = a["gravity_m_s2"] - OMEGA_EARTH**2 * _radius_m(a)
        gb = b["gravity_m_s2"] - OMEGA_EARTH**2 * _radius_m(b)
        loss += 0.5 * (ga + gb) * (b["t_s"] - a["t_s"])
    dv_analytic = SPIKE_STAGE["isp_s"] * G0_M_S2 * math.log(s[0]["mass_kg"] / s[-1]["mass_kg"]) - loss
    dv_sim = -(s[-1]["v_down_m_s"] - s[0]["v_down_m_s"])
    assert dv_sim == pytest.approx(dv_analytic, rel=HOP_DV_REL_TOL)
    # It really flew straight up.
    assert s[-1]["theta_deg"] == pytest.approx(90.0, abs=1e-6)
    assert s[-1]["altitude_m"] > HOP_IC.alt_m


def test_mass_cg_and_inertia_follow_the_propellant():
    s = _hop(duration=20.0).samples
    rows = s[::100]
    for key in ("mass_kg", "ixx_kg_m2", "iyy_kg_m2", "izz_kg_m2"):
        values = [r[key] for r in rows]
        assert all(b < a for a, b in zip(values, values[1:])), key
    # Forward LOX drains 2.5x faster than aft fuel -> CG moves aft.
    assert rows[-1]["cg_x_m"] > rows[0]["cg_x_m"]


def test_engine_starves_at_propellant_depletion():
    sim = _hop(duration=101.0)
    last = sim.samples[-1]
    assert last["thrust_n"] == 0.0
    assert last["mass_kg"] == pytest.approx(SPIKE_STAGE["dry_mass_kg"], abs=MASS_ABS_TOL_KG)


# --------------------------------------------------------------------------
# Gimbal / TVC
# --------------------------------------------------------------------------


def _gimbal_step(pitch_cmd, yaw_cmd, duration=1.0, dt=DT_S):
    sim = SixDofSim(dt)
    sim.initialize(HOP_IC)
    sim.ignite(1.0)
    sim.set_gimbal(pitch_cmd, yaw_cmd)
    sim.run_for(duration)
    return sim


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_pitch_gimbal_step_gives_right_signed_moment_and_rate(sign):
    s = _gimbal_step(sign * 0.01, 0.0).samples
    later = s[2:]  # frame 1: engine not yet lit (one-frame ignition latency)
    assert all(r["pitch_moment_n_m"] * sign > 0.0 for r in later)
    assert all(r["gimbal_pitch_actual_rad"] * sign > 0.0 for r in later)
    assert s[-1]["q_rad_s"] * sign > 0.0
    # No cross-axis moment; the only yaw rate is Earth-rotation coupling
    # (JSBSim body rates are relative to the rotating ECEF frame).
    assert abs(s[-1]["yaw_moment_n_m"]) < 1e-6
    assert abs(s[-1]["r_rad_s"]) < 1e-4 * abs(s[-1]["q_rad_s"])

    # Magnitude: M = T sin(delta) * lever arm (thruster aft of the live CG).
    for r in later[::10]:
        arm = THRUSTER_X_M - r["cg_x_m"]
        expected = r["thrust_n"] * math.sin(r["gimbal_pitch_actual_rad"]) * arm
        assert r["pitch_moment_n_m"] == pytest.approx(expected, rel=MOMENT_REL_TOL)

    # Rate: q(t) = integral of M / Iyy.
    q = 0.0
    for a, b in zip(s, s[1:]):
        q += 0.5 * (a["pitch_moment_n_m"] / a["iyy_kg_m2"] + b["pitch_moment_n_m"] / b["iyy_kg_m2"]) * (
            b["t_s"] - a["t_s"])
    assert s[-1]["q_rad_s"] == pytest.approx(q, rel=RATE_REL_TOL)


def test_yaw_gimbal_step_gives_right_signed_moment_and_rate():
    s = _gimbal_step(0.0, 0.01).samples
    assert all(r["yaw_moment_n_m"] > 0.0 for r in s[2:])
    assert s[-1]["r_rad_s"] > 0.0
    assert abs(s[-1]["pitch_moment_n_m"]) < 1e-6
    assert abs(s[-1]["q_rad_s"]) < 1e-4 * abs(s[-1]["r_rad_s"])


def test_gimbal_actual_follows_command_with_configured_lag():
    cmd = 0.01  # small: initial slew cmd/tau = 0.05 rad/s, below the rate limit
    tau = SPIKE_STAGE["gimbal_lag_tau_s"]
    s = _gimbal_step(cmd, -cmd, duration=1.0).samples
    for r in s[1:]:
        expected = cmd * (1.0 - math.exp(-(r["t_s"] - DT_S / 2) / tau))
        assert r["gimbal_pitch_cmd_rad"] == cmd
        assert abs(r["gimbal_pitch_actual_rad"] - expected) <= LAG_ABS_TOL_FRAC * cmd
        assert abs(r["gimbal_yaw_actual_rad"] + expected) <= LAG_ABS_TOL_FRAC * cmd


def test_gimbal_actual_honours_rate_and_travel_limits():
    s = _gimbal_step(0.5, 0.0, duration=1.0).samples
    act = [r["gimbal_pitch_actual_rad"] for r in s]
    slews = [(b - a) / DT_S for a, b in zip(act, act[1:])]
    limit = SPIKE_STAGE["gimbal_rate_limit_rad_s"]
    assert max(slews) <= limit + LIMIT_TOL
    assert max(slews) >= limit - LIMIT_TOL  # the limiter is actually active
    assert max(act) <= SPIKE_STAGE["gimbal_limit_rad"] + LIMIT_TOL
    assert act[-1] == pytest.approx(SPIKE_STAGE["gimbal_limit_rad"], abs=LIMIT_TOL)


# --------------------------------------------------------------------------
# Timestep halving
# --------------------------------------------------------------------------


def _gimballed_burn(dt, duration=20.0, cmd=0.005):
    sim = SixDofSim(dt)
    sim.initialize(InitialConditions(alt_m=300e3, pitch_deg=60.0, heading_deg=90.0))
    sim.ignite(1.0)

    def control(t, s):
        on = 1.0 - 1e-9 <= t < 2.0 - 1e-9
        s.set_gimbal(cmd if on else 0.0, -cmd / 2 if on else 0.0)

    sim.run_for(duration, control)
    return sim.samples


def test_timestep_halving_changes_results_below_tolerance():
    coarse, fine = _gimballed_burn(DT_S), _gimballed_burn(DT_S / 2)
    a, b, a0 = coarse[-1], fine[-1], coarse[0]
    assert a["t_s"] == pytest.approx(b["t_s"])

    pos = lambda r: (r["x_m"], r["y_m"], r["z_m"])  # noqa: E731
    vel = lambda r: (r["vx_m_s"], r["vy_m_s"], r["vz_m_s"])  # noqa: E731
    flown = math.dist(pos(a), pos(a0))
    speed = math.hypot(*vel(a))
    assert math.dist(pos(a), pos(b)) <= HALVING_POS_REL_TOL * flown
    assert math.dist(vel(a), vel(b)) <= HALVING_VEL_REL_TOL * speed

    dot = abs(sum(a[k] * b[k] for k in ("q_w", "q_x", "q_y", "q_z")))
    assert math.degrees(2.0 * math.acos(min(1.0, dot))) <= HALVING_ATTITUDE_TOL_DEG
    for k in ("q_rad_s", "r_rad_s"):
        assert a[k] == pytest.approx(b[k], rel=HALVING_RATE_REL_TOL)
    assert abs(a["mass_kg"] - b["mass_kg"]) <= HALVING_MASS_TOL_KG


# --------------------------------------------------------------------------
# BodyTrack export
# --------------------------------------------------------------------------


def test_body_track_carries_contract_and_extra_columns(tmp_path):
    from orbitopt.flightlab.contract import FlightRun  # noqa: F401  (contract import sanity)

    sim = _gimbal_step(0.01, 0.0, duration=0.5)
    track = sim.to_body_track(every=10)
    assert isinstance(track, BodyTrack)
    assert track.frame == "ECEF"
    assert list(track.extra_columns) == list(EXTRA_COLUMNS)
    n = len(track.t_s)
    assert n == 6 and track.t_s[0] == 0.0 and track.t_end_s == pytest.approx(0.5)
    assert all(len(v) == n for v in track.extra_columns.values())
    assert not set(EXTRA_COLUMNS) & set(REQUIRED_CSV_COLUMNS)
    for i in range(n):
        qn = sum(track.extra_columns[k][i] ** 2 for k in ("q_w", "q_x", "q_y", "q_z"))
        assert qn == pytest.approx(1.0, abs=1e-12)
    assert track.extra_columns["gimbal_pitch_cmd_rad"][-1] == 0.01
    assert 0.0 < track.extra_columns["gimbal_pitch_actual_rad"][-1] < 0.01
    assert track.extra_columns["thrust_n"][-1] == pytest.approx(SPIKE_STAGE["thrust_vac_n"], rel=THRUST_REL_TOL)
    # ECEF radius ~ Earth radius + 300 km.
    r0 = math.hypot(*track.position_m[0])
    assert r0 == pytest.approx(6378137.0 + 300e3, rel=1e-6)


def test_quaternion_points_the_nose_along_the_local_vertical():
    # Equator / prime meridian, nose straight up: the body x axis expressed
    # in ECEF (first row of R(q), R = ECEF->body) is +X_ecef.
    sim = SixDofSim(DT_S)
    sim.initialize(HOP_IC)
    w, x, y, z = (sim.samples[0][k] for k in ("q_w", "q_x", "q_y", "q_z"))
    nose_ecef = (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))
    assert nose_ecef == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)
