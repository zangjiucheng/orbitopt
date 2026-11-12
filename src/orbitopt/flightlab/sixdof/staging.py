"""Two-instance staging handoff spike (FL-402, task #207).

Proves, with the ``jsbsim`` Python bindings (1.3.3), that two *independent
top-level* ``FGFDMExec`` instances can run side by side in one process and
exchange a staged vehicle's state without leakage.  The recipe:

1. Fly ``spike_stage`` (#206) for ``burn_s`` and freeze the state
   (:func:`capture_frozen_state`): ECEF position/velocity, attitude as a
   scalar-first quaternion, body rates, total mass, tank contents.
2. Allocate dry mass and remaining propellant between the two child bodies
   (:class:`StageSplit`) and spawn each as a *fresh* top-level instance of
   its own aircraft model (``spike_booster`` / ``spike_upper``, the same
   airframe segments of the stack), driven only by ``ic/...`` properties
   derived from the frozen state (:func:`spawn_stage`).
3. Optionally apply a separation impulse (default 0 N.s) along the shared
   body +x, momentum-conservingly: +J x_hat to the upper stage, -J x_hat to
   the booster; the momentum residual is bookkept and logged.
4. Step both instances interleaved in one process for ``coast_s`` >= 60 s.

Why separate top-level instances: JSBSim's multi-FDM support is a parent
FGFDMExec feeding *child* FDMs every step (tied flight), which is not a
free separation after stage door; independent instances are the only way to
fly receding bodies.  That support is by design but was unproven for
side-by-side free flyers -- this module is the proof, and the checks in
``tests/test_flightlab_sixdof_staging_jsbsim.py`` assert: mass closure,
momentum residual under a declared tolerance, position/attitude continuity
across the handoff, bit-identical reruns, step-order independence
(A,B vs B,A), and a valid three-body ``FlightRun``.

Injection notes (jsbsim 1.3.3, all verified by experiment):
``ic/h-sl-ft`` is *geoid*-referenced while the frozen state is ECEF, so the
geodetic latitude / h-sl pair is solved by fixed-point iteration against the
achieved ECEF position (converges to ~1e-9 m in <= 3 iterations); ECEF
velocity is injected through ``ic/vn|ve|vd-fps`` (rotating-frame match is
exact); Euler angles come from ``Tbn = Tec2b @ R_n2e`` at the frozen point.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Mapping

from orbitopt.flightlab.contract import (
    BodyTrack, Check, Event, FlightRun, Verification,
)
from orbitopt.flightlab.sixdof.jsbsim_driver import (
    FT_M, LB_KG, InitialConditions, SixDofSim,
)

#: Tolerance the continuity assertions use [m] for the position handoff.
HANDOFF_POSITION_TOL_M = 1e-3
#: Tolerance for the attitude handoff [rad] and the ECEF velocity handoff [m/s].
HANDOFF_ATTITUDE_TOL_RAD = 1e-6
HANDOFF_VELOCITY_TOL_M_S = 1e-6
#: Declared momentum bookkeeping tolerance, relative to the stack |p| at
#: separation (absolute floor below: JSBSim round-trips state through
#: ft/lb/deg properties, so residuals land at the ~1e-8 relative level).
MOMENTUM_RESIDUAL_TOL_REL = 1e-6
MOMENTUM_RESIDUAL_FLOOR_N_S = 1.0

BOOSTER_MODEL = "spike_booster"
UPPER_MODEL = "spike_upper"

WGS84_A_M = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


# --------------------------------------------------------------------------
# WGS84 helpers (the handoff's only outside-JSBSim math)
# --------------------------------------------------------------------------

def _meridional_radius(lat_rad: float) -> float:
    s = math.sin(lat_rad)
    w = 1.0 - WGS84_E2 * s * s
    return WGS84_A_M * (1.0 - WGS84_E2) / w ** 1.5


def _ecef_to_geodetic(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Bowring's closed form: (lat_rad, lon_rad, h_m) from ECEF [m]."""
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    b = WGS84_A_M * math.sqrt(1.0 - WGS84_E2)
    th = math.atan2(z, p * (1.0 - WGS84_E2))
    lat = math.atan2(z + b * WGS84_E2 / (1 - WGS84_E2) ** 0.5 * math.sin(th) ** 3,
                     p - WGS84_A_M * WGS84_E2 * math.cos(th) ** 3)
    w = 1.0 - WGS84_E2 * math.sin(lat) ** 2
    h = p / math.sqrt(w) - WGS84_A_M
    return lat, lon, h


def _ned_to_ecef(lat_rad: float, lon_rad: float) -> tuple[tuple, tuple, tuple]:
    """Rows of the NED->ECEF rotation (numpy-free; returns 3 tuples)."""
    sl, cl = math.sin(lat_rad), math.cos(lat_rad)
    sn, cs = math.sin(lon_rad), math.cos(lon_rad)
    return ((-sl * cs, -sl * sn, cl), (-sn, cs, 0.0), (-cl * cs, -cl * sn, -sl))


def _mat3_row(r: tuple[tuple, tuple, tuple], i: int) -> tuple:
    return r[i]


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _quat_to_dcm(q) -> tuple[tuple, tuple, tuple]:
    """Body-from-ECEF DCM (``Tec2b``: v_body = R v_ecef) from scalar-first q."""
    w, x, y, z = q
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )


def euler_from_quaternion(q, lat_rad: float, lon_rad: float) -> tuple[float, float, float]:
    """JSBSim IC Euler angles (phi, theta, psi) [rad] for scalar-first q.

    The quaternion is the ECEF->body attitude, so first rotate it into the
    local NED frame (``Tbn = Tec2b @ R_n2e``) and read the 3-2-1 Euler
    angles off that; matches JSBSim's ``attitude/*-deg`` round-trip to ~1e-10.
    """
    tec2b = _quat_to_dcm(q)
    rn = _ned_to_ecef(lat_rad, lon_rad)  # rows = N, E, D basis vectors in ECEF
    # Tbn[i][j] = row i of Tec2b dotted with column j of R_n2e (= row j).
    tbn = tuple(tuple(_dot(row, rn[j]) for j in range(3)) for row in tec2b)
    phi = math.atan2(tbn[1][2], tbn[2][2])
    theta = math.asin(max(-1.0, min(1.0, -tbn[0][2])))
    psi = math.atan2(tbn[0][1], tbn[0][0])
    return phi, theta, psi


def _angle_between_quats(q1, q2) -> float:
    d = abs(q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3])
    return 2.0 * math.acos(min(1.0, d))


# --------------------------------------------------------------------------
# Frozen state and mass split
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FrozenState:
    """The stack state at separation, everything in SI, ECEF frame."""

    t_s: float
    mass_kg: float
    position_m: tuple[float, float, float]
    velocity_m_s: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]  # scalar-first, Tec2b
    rates_rad_s: tuple[float, float, float]
    tank_propellant_kg: tuple[float, float]  # (oxidizer, fuel)
    sample: Mapping[str, float]

    @property
    def dry_mass_kg(self) -> float:
        return self.mass_kg - self.tank_propellant_kg[0] - self.tank_propellant_kg[1]


def capture_frozen_state(sim: SixDofSim) -> FrozenState:
    """Freeze the current state of a running stack instance.

    Total mass is taken as empty weight + tank contents, *not*
    ``inertia/mass-slugs``: JSBSim's rocket model samples the mass before
    the current step's propellant drain, so ``mass-slugs`` runs one step
    (mdot*dt) high while the engine burns (13.23 lb = 6 kg here). The
    empty+tank figure is what the handoff must reproduce on the children.
    """
    s = sim.sample()
    ox, fuel = sim.tank_contents_kg()
    mass_kg = sim["inertia/empty-weight-lbs"] * LB_KG + ox + fuel
    return FrozenState(
        t_s=s["t_s"], mass_kg=mass_kg,
        position_m=(s["x_m"], s["y_m"], s["z_m"]),
        velocity_m_s=(s["vx_m_s"], s["vy_m_s"], s["vz_m_s"]),
        quaternion=(s["q_w"], s["q_x"], s["q_y"], s["q_z"]),
        rates_rad_s=(s["p_rad_s"], s["q_rad_s"], s["r_rad_s"]),
        tank_propellant_kg=(ox, fuel), sample=s,
    )


@dataclass(frozen=True)
class StageSplit:
    """Declared staging bookkeeping: what fraction of the stack's dry mass /
    remaining propellant goes to the booster (the rest to the upper stage)."""

    booster_dry_fraction: float = 0.4
    booster_propellant_fraction: float = 0.4

    def allocate(self, frozen: FrozenState) -> dict:
        db = self.booster_dry_fraction * frozen.dry_mass_kg
        dp = self.booster_propellant_fraction * sum(frozen.tank_propellant_kg)
        # Propellant split keeps the mixture ratio: scale each tank the same.
        ox, fu = frozen.tank_propellant_kg
        f = dp / (ox + fu) if (ox + fu) > 0 else 0.0
        return {
            "booster": {"dry_kg": db, "ox_kg": ox * f, "fuel_kg": fu * f},
            "upper": {"dry_kg": frozen.dry_mass_kg - db,
                      "ox_kg": ox * (1.0 - f), "fuel_kg": fu * (1.0 - f)},
        }


def spawn_stage(model: str, frozen: FrozenState, alloc: Mapping[str, float], *,
                dt_s: float, impulse_dv_ecef_m_s: tuple = (0.0, 0.0, 0.0),
                root=None) -> tuple[SixDofSim, dict]:
    """Fresh top-level instance of ``model`` initialized from ``frozen``.

    ``alloc`` carries this body's share (``dry_kg`` is in the aircraft XML's
    ``emptywt``; ``ox_kg``/``fuel_kg`` go into the tanks); the optional
    separation ``impulse_dv_ecef_m_s`` is added to the inherited velocity.
    Returns the sim plus handoff residuals (position [m], velocity [m/s],
    attitude [rad], mass [kg]) measured *after* ``run_ic()``.
    """
    sim = SixDofSim(dt_s, model=model, root=root)
    tx, ty, tz = frozen.position_m
    lat, lon, h = _ecef_to_geodetic(tx, ty, tz)
    phi, theta, psi = euler_from_quaternion(frozen.quaternion, lat, lon)
    vx, vy, vz = (frozen.velocity_m_s[i] + impulse_dv_ecef_m_s[i] for i in range(3))
    hsl_m = h
    lat_rad = lat
    rows = _ned_to_ecef(lat_rad, lon)
    vn, ve, vd = (_dot(rows[i], (vx, vy, vz)) for i in range(3))
    err = float("inf")
    for _ in range(8):
        sim.initialize_ic({
            "ic/lat-geod-deg": math.degrees(lat_rad),
            "ic/long-gc-deg": math.degrees(lon),
            "ic/h-sl-ft": hsl_m / FT_M,
            "ic/phi-deg": math.degrees(phi),
            "ic/theta-deg": math.degrees(theta),
            "ic/psi-true-deg": math.degrees(psi),
            "ic/vn-fps": vn / FT_M,
            "ic/ve-fps": ve / FT_M,
            "ic/vd-fps": vd / FT_M,
            "ic/p-rad_sec": frozen.rates_rad_s[0],
            "ic/q-rad_sec": frozen.rates_rad_s[1],
            "ic/r-rad_sec": frozen.rates_rad_s[2],
            "propulsion/tank[0]/contents-lbs": alloc["ox_kg"] / LB_KG,
            "propulsion/tank[1]/contents-lbs": alloc["fuel_kg"] / LB_KG,
        })
        s0 = sim.samples[0]
        e = (tx - s0["x_m"], ty - s0["y_m"], tz - s0["z_m"])
        err = math.sqrt(_dot(e, e))
        if err < 1e-6:
            break
        rows = _ned_to_ecef(lat_rad, lon)
        dn, _, dd = (_dot(rows[i], e) for i in range(3))
        lat_rad += dn / _meridional_radius(lat_rad)
        hsl_m += -dd
    s0 = sim.samples[0]
    vel_err = math.sqrt(sum((s0[k] - v) ** 2 for k, v in
                            zip(("vx_m_s", "vy_m_s", "vz_m_s"), (vx, vy, vz))))
    att_err = _angle_between_quats(
        (s0["q_w"], s0["q_x"], s0["q_y"], s0["q_z"]), frozen.quaternion)
    mass_err = s0["mass_kg"] - (alloc["dry_kg"] + alloc["ox_kg"] + alloc["fuel_kg"])
    return sim, {"position_err_m": err, "velocity_err_m_s": vel_err,
                 "attitude_err_rad": att_err, "mass_err_kg": mass_err}


# --------------------------------------------------------------------------
# Scenario
# --------------------------------------------------------------------------

@dataclass
class StagingReplay:
    """Everything a test needs to re-derive the spike's claims."""

    frozen: FrozenState
    allocation: Mapping
    stack: SixDofSim
    booster: SixDofSim
    upper: SixDofSim
    handoff: Mapping  # per-body spawn residuals from spawn_stage()
    bookkeeping: Mapping  # mass/momentum numbers at the freeze instant
    separation_impulse_ns: float
    step_order: str
    dt_s: float
    coast_s: float


def run_staging_scenario(*, dt_s: float = 0.01, burn_s: float = 20.0,
                         coast_s: float = 65.0, separation_impulse_ns: float = 0.0,
                         step_order: str = "ab", throttle: float = 1.0,
                         split: StageSplit = StageSplit(), root=None) -> StagingReplay:
    """Fly the stack, freeze, hand off, coast both children.

    ``coast_s`` must be >= 60 s for the acceptance claim; both instances are
    stepped from the same loop, in the order given (``"ab"`` booster-then-
    upper, ``"ba"`` the reverse) to prove the instances do not share state.
    """
    if coast_s < 60.0:
        raise ValueError("FL-402 requires coasting both instances for >= 60 s")
    if step_order not in ("ab", "ba"):
        raise ValueError("step_order must be 'ab' or 'ba'")
    stack = SixDofSim(dt_s, root=root)
    stack.initialize(InitialConditions(alt_m=0.0, pitch_deg=90.0))
    stack.ignite(throttle)
    stack.run_for(burn_s)
    frozen = capture_frozen_state(stack)
    alloc = split.allocate(frozen)

    # Impulse along shared body +x (row 0 of the ECEF->body DCM = +x in ECEF).
    xb = _mat3_row(_quat_to_dcm(frozen.quaternion), 0)
    j = separation_impulse_ns
    m_b = alloc["booster"]["dry_kg"] + alloc["booster"]["ox_kg"] + alloc["booster"]["fuel_kg"]
    m_u = alloc["upper"]["dry_kg"] + alloc["upper"]["ox_kg"] + alloc["upper"]["fuel_kg"]
    booster, hb = spawn_stage(BOOSTER_MODEL, frozen, alloc["booster"], dt_s=dt_s,
                              impulse_dv_ecef_m_s=tuple(-j * xb[i] / m_b for i in range(3)),
                              root=root)
    upper, hu = spawn_stage(UPPER_MODEL, frozen, alloc["upper"], dt_s=dt_s,
                            impulse_dv_ecef_m_s=tuple(j * xb[i] / m_u for i in range(3)),
                            root=root)

    b0, u0 = booster.samples[0], upper.samples[0]
    p0 = tuple(frozen.mass_kg * v for v in frozen.velocity_m_s)
    p_after = tuple(b0["mass_kg"] * b0[k] + u0["mass_kg"] * u0[k]
                    for k in ("vx_m_s", "vy_m_s", "vz_m_s"))
    bookkeeping = {
        "mass_before_kg": frozen.mass_kg,
        "dry_split_kg": (alloc["booster"]["dry_kg"], alloc["upper"]["dry_kg"]),
        "propellant_split_kg": (alloc["booster"]["ox_kg"] + alloc["booster"]["fuel_kg"],
                                alloc["upper"]["ox_kg"] + alloc["upper"]["fuel_kg"]),
        "mass_after_kg": b0["mass_kg"] + u0["mass_kg"],
        "mass_residual_kg": b0["mass_kg"] + u0["mass_kg"] - frozen.mass_kg,
        "momentum_before_kg_m_s": p0,
        "momentum_after_kg_m_s": p_after,
        "momentum_residual_kg_m_s": math.sqrt(sum((p_after[i] - p0[i]) ** 2 for i in range(3))),
        "dv_booster_m_s": j / m_b,
        "dv_upper_m_s": j / m_u,
        "position_continuity_err_m": max(hb["position_err_m"], hu["position_err_m"]),
        "attitude_continuity_err_rad": max(hb["attitude_err_rad"], hu["attitude_err_rad"]),
    }

    n = round(coast_s / dt_s)
    for _ in range(n):
        if step_order == "ab":
            booster.step(); upper.step()
        else:
            upper.step(); booster.step()
    return StagingReplay(frozen, alloc, stack, booster, upper,
                         {"booster": hb, "upper": hu}, bookkeeping,
                         j, step_order, dt_s, coast_s)


def to_flight_run(replay: StagingReplay, *, run_id: str = "fl-402-staging-spike",
                  epoch_utc: str = "2026-01-01T00:00:00Z", every: int = 50) -> FlightRun:
    """The replay as a three-body FlightResult v1 run (bodies ``stack``,
    ``booster``, ``upper``; the children carry ``parent_body_id=stack`` and
    are re-based onto the stack clock at the separation instant)."""
    bk, frozen = replay.bookkeeping, replay.frozen
    mk = lambda tag: hashlib.sha256(f"fl-402:{tag}".encode()).hexdigest()  # noqa: E731
    p_norm = math.sqrt(sum(v * v for v in bk["momentum_before_kg_m_s"]))
    mom_tol = max(MOMENTUM_RESIDUAL_FLOOR_N_S, MOMENTUM_RESIDUAL_TOL_REL * p_norm)
    ok_mass = abs(bk["mass_residual_kg"]) < 1e-6 * max(1.0, bk["mass_before_kg"])
    ok_mom = bk["momentum_residual_kg_m_s"] < mom_tol
    ok_pos = bk["position_continuity_err_m"] < HANDOFF_POSITION_TOL_M
    ok_att = bk["attitude_continuity_err_rad"] < HANDOFF_ATTITUDE_TOL_RAD
    ok_vel = (max(replay.handoff["booster"]["velocity_err_m_s"],
                  replay.handoff["upper"]["velocity_err_m_s"]) < HANDOFF_VELOCITY_TOL_M_S)
    checks = (
        Check("mass-closure", ok_mass, bk["mass_residual_kg"], 0.0, "kg",
              "" if ok_mass else "children masses != frozen stack mass"),
        Check("momentum-residual", ok_mom, bk["momentum_residual_kg_m_s"], mom_tol, "N.s", ""),
        Check("position-continuity", ok_pos, bk["position_continuity_err_m"],
              HANDOFF_POSITION_TOL_M, "m", ""),
        Check("attitude-continuity", ok_att, bk["attitude_continuity_err_rad"],
              HANDOFF_ATTITUDE_TOL_RAD, "rad", ""),
        Check("velocity-continuity", ok_vel,
              max(replay.handoff["booster"]["velocity_err_m_s"],
                  replay.handoff["upper"]["velocity_err_m_s"]),
              HANDOFF_VELOCITY_TOL_M_S, "m/s", ""),
    )
    return FlightRun(
        run_id=run_id,
        target_sha256=mk("no-flight-target-impulse-scenario"),
        vehicle_sha256=mk("spike-stage-split-into-booster-and-upper"),
        git_sha="0" * 40,
        created_utc=epoch_utc,
        engines={"spike_engine": "vacuum Isp 320 s, 600 kg/s (spike_engine.xml)"},
        assumptions=(
            "E4 feasibility spike: JSBSim 6-DOF, zero aerodynamics, spherical+J2 gravity;"
            " numbers are illustrative, not a flight-certified run.",
            "Staging handoff injects state through JSBSim ic/ properties only"
            " (no internal state access); position converges by h-sl/lat iteration.",
            f"Separation impulse {replay.separation_impulse_ns} N.s along body +x,"
            " applied momentum-conservatively (+J upper / -J booster).",
            f"Children stepped interleaved '{replay.step_order}' for "
            f"{replay.coast_s} s in one process; separate top-level FGFDMExec instances.",
        ),
        tracks=(
            replay.stack.to_body_track(body_id="stack", epoch_utc=epoch_utc, every=every),
            replay.booster.to_body_track(body_id="booster", epoch_utc=epoch_utc, every=every,
                                         parent_body_id="stack", t_offset_s=frozen.t_s),
            replay.upper.to_body_track(body_id="upper", epoch_utc=epoch_utc, every=every,
                                       parent_body_id="stack", t_offset_s=frozen.t_s),
        ),
        events=(
            Event(0.0, "ignition", "stack", {"throttle": 1.0}),
            Event(frozen.t_s, "separation", "stack",
                  {"impulse_ns": replay.separation_impulse_ns,
                   "mass_kg": frozen.mass_kg}),
            Event(frozen.t_s, "separation", "booster", {"dv_m_s": -bk["dv_booster_m_s"]}),
            Event(frozen.t_s, "separation", "upper", {"dv_m_s": bk["dv_upper_m_s"]}),
        ),
        verification=Verification(
            optimizer_converged=None,
            physics_verified=all(c.passed for c in checks),
            mission_constraints_passed=None,
            checks=checks,
        ),
    )
