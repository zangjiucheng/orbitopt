"""E8-05 independent dynamics validation of the E5-07 Starship-class reference.

E8-01 (``orbitopt.verify.full_chain``) proved the E5-07 chain extracts the
recorded OpenTsiolkovsky run without drift, but -- as that module's honesty
notes already said -- comparing the chain against OpenTsiolkovsky's *own*
raw output cannot count as independent-model evidence for the ascent: the
ascent was computed *by* that program. This module closes the evidence gap
with two genuinely separately formulated recomputations of the E5-07
reference mission, both graded against the sealed recording
(``PROVENANCE.json`` digests re-checked via
:func:`orbitopt.verify.full_chain.verify_reference_digests`) and both
labelled ``independent_model`` in the E4-03 sense:

* **Ascent (and stage-1 debris re-entry).** A native point-mass twin built
  only from the vehicle YAML and the recorded scenario: the repo's 3-DOF
  plant (``propagate_point_mass``, its own force laws, its own RK4, US76 via
  :mod:`orbitopt.flightlab.atmosphere`, J2 + rotating Earth from
  :mod:`orbitopt.flightlab.sixdof.rigidbody`) driven *open-loop* by the
  authored attitude program and throttle schedules -- no event-driven
  feedback -- plus a separately written variable-mass-free ballistic
  re-entry integrator for the stage-1 debris (power-off CA polar, RK4 at
  0.05 s, geodetic event interpolation). OpenTsiolkovsky is a separately
  developed external program (ADR-0007); the twin is repo code. The shared
  inputs -- the attitude program, throttle schedule and vehicle constants
  read from the same authored YAML -- are design inputs, recorded in
  ``inputs.assumption_differences``.
  The two models differ where the recorded program's internals are opaque:
  OpenTsiolkovsky's own engine-Isp law (pad Isp 329.26 s rising with
  altitude, read back from the raw state history) differs from the authored
  YAML law (327 s sea-level Raptor, linear in pressure ratio), and its
  atmosphere model is its own. Those differences are not hand-waved: every
  event discrepancy is decomposed into an E8-02-style error budget --
  a *measured* engine line (re-run with the recorded pad Isp), a *measured*
  aero/atmosphere line (re-run with density x1.03), a *measured* numerical
  line (re-run at dt/2), a declared-zero frame/time line (same epoch, same
  Earth rate, same launch-aligned ECI convention -- the pad state matches to
  < 1 m at t = 0) and a declared-zero controller line (both fly the same
  authored program; schedule-tracking implementation differences are
  charged to the *residual*, which is reported, never zeroed).
* **Powered landing.** No landing is flown in E5-07 (the gap is documented
  in ``RECOVERY_GAP.md``), so the landing burn is validated on a *declared
  design corridor* at the pad (a corridor-insertion state, a design input
  like E5-05's): a gfold SOCP arc is solved for the Starship-class booster
  in its three-engine landing configuration (the vehicle file itself
  records that the thirteen-engine single set cannot throttle below weight
  and that the real vehicle shuts down to three engines for the landing
  burn), re-flown twice through two independently written models of the
  same commanded profile: the FL-301 propagator (fixed-step RK4 0.1 s, J2
  gravity, "entry" density table, thrust/acceleration command replay) and a
  newly written independent re-integrator (own RK4 at 0.05 s, own WGS-84
  geodetic conversion, IGF-80 normal gravity, own US76 ISA tables,
  physical-thrust FOH replay). Touchdown position, vertical/lateral speed,
  end-of-burn state and remaining propellant are compared between the two
  re-flights.

Neither report contains ``historical_telemetry`` evidence anywhere: the
reference vehicle is a public estimate, not flown hardware.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REFERENCE_ROOT = (Path(__file__).resolve().parents[3] / "tests" / "data" /
                  "mission" / "reference" / "e5-07-physics-qualification")
_VEHICLE_YAML = (Path(__file__).resolve().parents[1] / "data" / "flightlab" /
                 "vehicles" / "starship_class_public_estimate.yaml")
_EPOCH_ISO = "2026-11-10T00:00:00+00:00"
OMEGA_EARTH_RAD_S = 7.2921159e-5
_G0 = 9.80665
_MU_EARTH = 398600441800000.0
_J2 = 1.08262668e-3
_R_EARTH = 6378137.0
_A_WGS84 = 6378137.0
_F_WGS84 = 1.0 / 298.257223563
_E2_WGS84 = _F_WGS84 * (2.0 - _F_WGS84)

#: Launch site of the E5-07 scenario (Kourou LC, geodetic deg / alt m) and the
#: authored pad azimuth -- read from the scenario, mirrored here as constants
#: so the twin's initial state is independent of the sealed event records.
LAUNCH_LAT_DEG = 5.2392
LAUNCH_LON_DEG = -52.7685
LAUNCH_ALT_M = 15.0
LAUNCH_HEADING_DEG = 36.0

#: Twin integration steps (probed; the numerical budget line halves them).
TWIN_DT_S = 0.02
DEBRIS_DT_S = 0.05

#: Declared powered-landing corridor insertion (pad-local ENU metres and m/s)
#: and the insertion mass. Design inputs, declared here and embedded in the
#: report: the insertion state is 800 m of ENU up (geodetic + the pad's 15 m)
#: and 150 m downrange west, descending at 55 m/s with 15 m/s of lateral
#: speed -- low and slow enough that a three-engine (shut-down) landing
#: configuration, whose full-throttle thrust-to-weight is 1.7 and throttle-
#: floor thrust-to-weight 0.65 at this mass, can null both velocities and
#: arrive at the FL-303 cutoff criterion. The mass is dry (275 t) + the
#: 150 t landing reserve + a 31 t landing-burn margin carried on the
#: vehicle. ``design_input`` is also why the FL-301 grading of the same
#: re-flight is labelled ``design_input``, not validation.
LANDING_CORRIDOR_ENU_M = (150.0, 0.0, 800.0)
LANDING_CORRIDOR_V_ENU_M_S = (-15.0, 0.0, -55.0)
LANDING_INSERTION_MASS_KG = 456000.0
LANDING_ENGINES = 3

#: Recorded engine Isp at the launch pad, measured from the raw OpenTsiolkovsky
#: state history (column ``isp_s`` at t = 0); drives the engine budget line.
OT_PAD_ISP_S = 329.26

_TOL: dict[str, dict[str, Any]] = {}


def _band(check_id: str, unit: str, limit: float, basis: str) -> None:
    _TOL[check_id] = {"unit": unit, "limit": limit, "basis": basis}


# --- ascent twin bands ------------------------------------------------------
_band("ascent.meco.state_position", "m", 400.0,
      "point-mass twin vs the recorded OT state history at MECO; measured "
      "~2.2e2 m nominal, almost all of it the engine-Isp line; band = "
      "measured + ~80% margin for probe-interaction terms charged to it")
_band("ascent.meco.state_velocity", "m/s", 25.0,
      "measured ~15 m/s nominal, engine-Isp line; margin on MECO mass-deficit "
      "line (Isp law mismatch ~1.2% of stage-1 burn)")
_band("ascent.meco.mass", "kg", 20000.0,
      "stage-1 burnout mass deficit is the engine-Isp line: pad 329.26 s vs "
      "authored 327 s (linear laws), ~1.2% of the 1.2 Mt stage-1 propellant")
_band("ascent.separation.state_position", "m", 600.0,
      "coasted 3 s from MECO; measured ~2.4e2 m; band covers the carried MECO "
      "offset plus coast propagation of the same discrepancy")
_band("ascent.separation.state_velocity", "m/s", 15.0,
      "the MECO velocity offset, coasted 3 s nearly unchanged; measured ~5.4 "
      "m/s with the same engine-line origin")
_band("ascent.separation.debris_mass", "kg", 12000.0,
      "debris mass = twin separation mass - 1.3 Mt ship vs recorded body mass; "
      "engine line again; measured ~6.4e3 kg")
_band("ascent.seco.state_position", "m", 5000.0,
      "378 s of stage-2 burn accumulate the (smaller, +0.3 s equivalent) Isp-"
      "law offset plus gravity-loss-path drift; measured ~2.7e3 m")
_band("ascent.seco.state_velocity", "m/s", 20.0,
      "measured ~7 m/s; margin for the stage-2 mass-flow-line offset")
_band("ascent.seco.mass", "kg", 2000.0,
      "measured ~5e1 kg: the stage-2 Isp line is declared from the recorded "
      "vacuum Isp directly, so only integration-path terms remain")
for _cid, _unit, _lim, _bas in (
        ("ascent.meco.residual_position", "m", 250.0,
         "nominal MECO position discrepancy minus the measured engine/aero/"
         "numerical budget lines; what remains is interaction terms, engine-"
         "model shape differences and throttle-schedule tracking"),
        ("ascent.meco.residual_velocity", "m/s", 18.0,
         "residual MECO velocity discrepancy after the three measured lines: "
         "the part of the engine-model difference invisible from the raw "
         "state history (ambient thrust correction and mass-flow shape) "
         "coupled through the gravity-turn path; measured 13.2 m/s"),
        ("ascent.separation.residual_position", "m", 350.0,
         "residual at separation: carries the MECO residual forward through "
         "3 s of coast, plus coast-integration cross terms"),
        ("ascent.seco.residual_position", "m", 2500.0,
         "residual at SECO after 378 s of stage-2 burn: accumulated path "
         "difference not explained by the three measured probe lines"),
):
    _band(_cid, _unit, _lim, _bas)

_band("ascent.maxq.value", "Pa", 1200.0,
      "max-q agreement of two point-mass models sharing the CA polar: "
      "measured ~3e1 Pa (-0.08%); band ~3%")
_band("ascent.maxq.time", "s", 2.0,
      "max-q is a broad maximum; phase offset follows the thrust-rise line; "
      "measured -0.9 s")
_band("ascent.seco.orbit_semi_major_axis", "m", 15000.0,
      "osculating elements from both SECO states with the same closed form; "
      "the semi-major axis follows the velocity discrepancy, da ~ 2*a*dv/v "
      "(about 1.75 km per m/s here), so the band is the SECO velocity band "
      "seen through vis-viva; measured 8.5 km")
_band("ascent.seco.orbit_eccentricity", "-", 3.0e-3,
      "tracks the SECO state band; measured ~1e-4")
_band("ascent.seco.orbit_inclination", "deg", 0.05,
      "the attitude program fixes the azimuth profile; inclination "
      "discrepancy is second-order in the state offset")
_band("ascent.debris_impact.time", "s", 8.0,
      "emergent check: debris ballistic range is sensitive to the (engine-"
      "line-shifted) separation state and to CA power-off +/-30% (vehicle "
      "file); measured +2 s")
_band("ascent.debris_impact.position", "km", 12.0,
      "great-circle impact-point distance; measured ~3.5 km")
_band("ascent.debris_impact.speed", "%", 12.0,
      "impact speed vs recorded (both Earth-relative); measured ~8.5%; the "
      "power-off CA table is an illustrative blunt-body polar")
# --- landing bands ----------------------------------------------------------
_band("landing.cutoff_position", "m", 15.0,
      "re-flight disagreement at the commanded cutoff node: the two "
      "re-integrators fly the same profile through different gravity/atmosphere/"
      "replay implementations; measured ~6 m vertical")
_band("landing.cutoff_velocity", "m/s", 2.5,
      "same comparison at the cutoff node; measured ~0.9 m/s")
_band("landing.touchdown_position", "m", 10.0,
      "touchdown position disagreement between the two re-flights; measured "
      "~0.3 m (both converge toward the pad target)")
_band("landing.touchdown_vertical_speed", "m/s", 15.0,
      "touchdown vertical-speed disagreement: amplifies the cutoff-state "
      "offset ballistically (the cutoff is a fixed-clock engine stop, so a "
      "cutoff-height error dh enters as sqrt(v^2+2g dh)); the *design limit* "
      "3 m/s is graded on the FL-301 re-flight alone (design_input)")
_band("landing.touchdown_lateral_speed", "m/s", 3.0,
      "measured ~0.25 m/s")
_band("landing.propellant", "kg", 750.0,
      "propellant-burned disagreement at touchdown between the re-flights "
      "(same Isp law, different replay/integration); measured ~1.6e2 kg")

_LANDING_CHECK_BANDS = ("landing.cutoff_position", "landing.cutoff_velocity",
                        "landing.touchdown_position",
                        "landing.touchdown_vertical_speed",
                        "landing.touchdown_lateral_speed", "landing.propellant")
_ASCENT_CHECK_IDS = tuple(k for k in _TOL if k.startswith("ascent."))


def declared_tolerances() -> dict[str, dict[str, Any]]:
    """Declared bands with their basis, per E8-02 audit discipline: every
    band is measured-then-margined, none is a bare round number."""
    return {k: dict(v) for k, v in _TOL.items()}


# ---------------------------------------------------------------------------
# shared geometry helpers (own implementations -- that is the point)
# ---------------------------------------------------------------------------

def _enu_basis(lat_deg: float, lon_deg: float) -> np.ndarray:
    la, lo = math.radians(lat_deg), math.radians(lon_deg)
    sl, cl, sn = math.sin(la), math.cos(la), math.sin(lo)
    return np.vstack([[-sn, math.cos(lo), 0.0],
                      [-sl * math.cos(lo), -sl * math.sin(lo), cl],
                      [cl * math.cos(lo), cl * math.sin(lo), sl]])


def _rot_z(v: np.ndarray, th: float) -> np.ndarray:
    c, s = math.cos(th), math.sin(th)
    return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]])


def _ecef_to_eci(t_s: float, r_ecef: np.ndarray, v_ecef_rel: np.ndarray,
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Launch-aligned ECEF-at-t=0 ECI rotation (the convention verified against
    the recorded raw columns: the pad state matches to < 1 m at t = 0)."""
    w = OMEGA_EARTH_RAD_S * t_s
    return _rot_z(r_ecef, -w), _rot_z(v_ecef_rel + np.cross(
        np.array([0.0, 0.0, OMEGA_EARTH_RAD_S]), r_ecef), -w)


def _geodetic(r: np.ndarray) -> tuple[float, float, float]:
    """Own WGS-84 inverse (rad, rad, m) -- independent of frames.py by intent."""
    x, y, z = (float(c) for c in r)
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    phi = math.atan2(z, p * (1.0 - _E2_WGS84)) if p > 0.0 else math.copysign(
        math.pi / 2, z)
    for _ in range(8):
        s = math.sin(phi)
        n_ = _A_WGS84 / math.sqrt(1.0 - _E2_WGS84 * s * s)
        phi = math.atan2(z + _E2_WGS84 * n_ * s, p)
    s = math.sin(phi)
    n_ = _A_WGS84 / math.sqrt(1.0 - _E2_WGS84 * s * s)
    h = p / math.cos(phi) - n_ if abs(phi) < 1.2 else z / s - n_ * (1.0 - _E2_WGS84)
    return phi, lon, h


def _isa_us76(h_m: float) -> tuple[float, float, float]:
    """Own US76 pressure (Pa), density (kg/m3) and speed of sound (m/s), to
    32 km -- independent of flightlab.atmosphere by intent (landing twin)."""
    h = max(float(h_m), 0.0)
    if h <= 11000.0:
        base = 1.0 - 6.5e-3 * h / 288.15
        t_k = 288.15 * base
        p = 101325.0 * base ** 5.25588
        rho = 1.225 * base ** 4.25588
    elif h <= 20000.0:
        t_k = 216.65
        p = 22632.0 * math.exp(-1.21632e-4 * (h - 11000.0))
        rho = p / (287.05287 * t_k)
    else:
        base = 1.0 + 1.0e-3 * (h - 20000.0) / 288.15
        t_k = 216.65 + 1.0e-3 * (h - 20000.0) * 1000.0 / 1000.0
        t_k = 216.65 + (h - 20000.0) * 1.0e-3
        p = 5474.87 * base ** -12.25588
        rho = p / (287.05287 * t_k)
    return p, rho, math.sqrt(287.05287 * 1.4 * t_k)


# ---------------------------------------------------------------------------
# ascent: the native point-mass twin
# ---------------------------------------------------------------------------

def _sched_level(table: Sequence[Sequence[float]]) -> tuple[Any, float]:
    """Throttle-level schedule from a (t_s, thrust_n) table with the recorded
    OpenTsiolkovsky semantics: the table is evaluated on consecutive original-
    order pairs (never sorted); a duplicated time instant is an instantaneous
    step whose *later* value holds from that instant."""
    pts = [(float(a), float(b)) for a, b in table]
    peak = max(v for _, v in pts)

    def level(t: float) -> float:
        if t <= pts[0][0]:
            return pts[0][1] / peak
        for (ta, va), (tb, vb) in zip(pts, pts[1:]):
            if ta <= t <= tb:
                if tb == ta:
                    return vb / peak
                f = (t - ta) / (tb - ta)
                return (va + f * (vb - va)) / peak
        return pts[-1][1] / peak
    return level, peak


class _SchedGuidance:
    """Open-loop attitude + throttle: the authored waypoint program and the
    authored throttle schedule only -- no feedback of the flown state, so the
    twin cannot chase the recorded trajectory."""

    def __init__(self, program: Any, level_fn: Any, t_offset_s: float = 0.0) -> None:
        self._program, self._level, self._t0 = program, level_fn, t_offset_s

    def command_from_state(self, t_s: float, r: Any, v: Any, m: Any):
        from orbitopt.flightlab import frames
        lat, lon, _ = frames.ecef_to_geodetic(np.asarray(r, dtype=float))
        return (self._program.dcm_at(t_s, float(lat), float(lon)),
                self._level(t_s - self._t0))


def _twin_ascent(dt_s: float, isp_sl_stage1: float | None = None,
                 density_scale: float = 1.0, ) -> dict[str, Any]:
    """Fly pad -> MECO -> coast -> separation -> SECO with the point-mass twin
    and integrate the stage-1 debris to impact; return the event states."""
    import yaml
    from orbitopt.flightlab import frames
    from orbitopt.flightlab.sixdof.aero import AeroModel
    from orbitopt.flightlab.sixdof.gravity_turn import WaypointAttitudeProgram, propagate_point_mass
    from orbitopt.flightlab.sixdof.rigidbody import (RigidBodyEngine, RigidBodyPhysics,
                                                     RigidBodyState, RigidBodyVehicle)
    doc = yaml.safe_load(_VEHICLE_YAML.read_text(encoding="utf-8"))
    prog = doc["attitude_program"]
    program = WaypointAttitudeProgram(tuple(p["t_s"] for p in prog),
                                      tuple(p["azimuth_deg"] for p in prog),
                                      tuple(p["elevation_deg"] for p in prog),
                                      (0.0,) * len(prog))

    def stage(spec: Mapping[str, Any], isp_sl: float):
        lev, peak = _sched_level(spec["engine"]["thrust_vac_n"])
        ca = spec["aero"]["ca"]
        eng = RigidBodyEngine(thrust_vacuum_n=peak,
                              isp_vacuum_s=float(spec["engine"]["isp_vac_s"]),
                              isp_sea_level_s=isp_sl,
                              nozzle_exit_area_m2=float(spec["engine"]["nozzle_exit_area_m2"]))
        aero = AeroModel.axisymmetric(name="twin", ca_vs_mach=(
            tuple(p[0] for p in ca), tuple(p[1] for p in ca)),
            cn_alpha_per_rad=float(spec["aero"]["cn_alpha_per_rad"]),
            diameter_m=float(spec["aero"]["diameter_m"]))
        dry = float(spec["mass_dry_kg"])
        veh = RigidBodyVehicle(name="twin", dry_mass_kg=dry,
                               propellant_mass_kg=float(spec["mass_initial_kg"]) - dry,
                               inertia_kg_m2=(1e7, 1e9, 1e9), engine=eng, aero=aero)
        return veh, lev

    s1, s2 = doc["stages"][0], doc["stages"][1]
    # the stage blocks record vacuum Isp only; the sea-level values come from
    # the vehicle's Raptor sea-level figure (stage 1) and the authored sea-
    # level/vacuum ratio applied to the vacuum-optimised engine (stage 2) --
    # declared design assumptions, shared by every probe.
    eng = doc["booster"]["engines"]
    sl_ratio = float(eng["isp_sea_level_s"]) / float(eng["isp_vacuum_s"])
    v1, lev1 = stage(s1, isp_sl_stage1 if isp_sl_stage1 is not None else
                     float(eng["isp_sea_level_s"]))
    v2, lev2 = stage(s2, float(s2["engine"]["isp_vac_s"]) * sl_ratio)
    physics = RigidBodyPhysics(density_scale=density_scale)
    state0 = RigidBodyState.from_launch(lat_geod_deg=LAUNCH_LAT_DEG,
                                        lon_deg=LAUNCH_LON_DEG,
                                        alt_m=LAUNCH_ALT_M,
                                        heading_deg=LAUNCH_HEADING_DEG,
                                        pitch_deg=90.0,
                                        mass_kg=float(s1["mass_initial_kg"])
                                        + float(s2["mass_initial_kg"]))

    def run1(t_end: float, dt: float):
        return propagate_point_mass(state0, v1, program, t_end, dt_s=dt,
                                    sample_every=max(1, int(round(dt / dt_s * 100))),
                                    physics=physics,
                                    guidance=_SchedGuidance(program, lev1))

    burn1 = run1(135.0, dt_s)
    mq = burn1.max_q

    def at(run: Any, t: float) -> dict:
        i = int(np.searchsorted(run.samples["t_s"], t))
        return {k: run.samples[k][i] for k in ("x_m", "y_m", "z_m", "vx_m_s",
                                              "vy_m_s", "vz_m_s", "mass_kg")}

    m_s = at(burn1, 135.0)
    r_meco = np.array([m_s["x_m"], m_s["y_m"], m_s["z_m"]])
    v_meco = np.array([m_s["vx_m_s"], m_s["vy_m_s"], m_s["vz_m_s"]])
    st_meco = RigidBodyState(t_s=135.0, position_m=r_meco, velocity_m_s=v_meco,
                             quaternion=(1.0, 0.0, 0.0, 0.0), omega_rad_s=(0.0, 0.0, 0.0),
                             mass_kg=float(m_s["mass_kg"]))
    coast = propagate_point_mass(st_meco, v1, program, 138.0, dt_s=dt_s,
                                 sample_every=1, physics=physics, throttle=0.0)
    s_s = at(coast, 138.0)
    r_sep = np.array([s_s["x_m"], s_s["y_m"], s_s["z_m"]])
    v_sep = np.array([s_s["vx_m_s"], s_s["vy_m_s"], s_s["vz_m_s"]])
    m_sep = float(s_s["mass_kg"])
    ship_kg = float(s2["mass_initial_kg"])
    st2 = RigidBodyState(t_s=138.0, position_m=r_sep, velocity_m_s=v_sep,
                         quaternion=(1.0, 0.0, 0.0, 0.0), omega_rad_s=(0.0, 0.0, 0.0),
                         mass_kg=ship_kg)
    burn2 = propagate_point_mass(st2, v2, program, 516.0, dt_s=dt_s,
                                 sample_every=max(1, int(round(dt_s / dt_s * 100))),
                                 physics=physics,
                                 guidance=_SchedGuidance(program, lev2, 138.0))
    e_s = at(burn2, 516.0)
    r_seco = np.array([e_s["x_m"], e_s["y_m"], e_s["z_m"]])
    v_seco = np.array([e_s["vx_m_s"], e_s["vy_m_s"], e_s["vz_m_s"]])
    debris = _debris_to_impact(r_sep, v_sep, m_sep - ship_kg,
                               s1["aero"]["ca_power_off"], DEBRIS_DT_S)
    return {"meco": (r_meco, v_meco, float(m_s["mass_kg"])),
            "separation": (r_sep, v_sep, m_sep),
            "seco": (r_seco, v_seco, float(e_s["mass_kg"])),
            "maxq": (float(mq.q_pa), float(mq.t_s)),
            "debris_mass_kg": m_sep - ship_kg, "impact": debris}


def _debris_to_impact(r0: np.ndarray, v0: np.ndarray, mass_kg: float,
                      ca_off: Sequence[Sequence[float]], dt: float) -> dict:
    """Ballistic re-entry of the spent stage-1: power-off CA polar, velocity-
    aligned, point mass, J2 + Coriolis + centrifugal, US76 (altitude only);
    own RK4, linear interpolation of the geodetic-altitude ground crossing."""
    from orbitopt.flightlab import atmosphere
    s_ref = math.pi * float(_twin_diameter_m()) ** 2 / 4.0
    ca = np.asarray(ca_off, dtype=float)
    om = np.array([0.0, 0.0, OMEGA_EARTH_RAD_S])

    def deriv(y: np.ndarray) -> np.ndarray:
        r, v = y[:3], y[3:6]
        rn = float(np.linalg.norm(r))
        a = -(_MU_EARTH / rn ** 3) * r
        z2 = (r[2] / rn) ** 2
        k = 1.5 * _J2 * _MU_EARTH * _R_EARTH ** 2 / rn ** 5
        a = a + k * np.array([r[0] * (1.0 - 5.0 * z2), r[1] * (1.0 - 5.0 * z2),
                              r[2] * (3.0 - 5.0 * z2)])
        a = a - 2.0 * np.cross(om, v) - np.cross(om, np.cross(om, r))
        alt = _geodetic(r)[2]
        rho = atmosphere.density_us76_kg_m3(alt)
        if rho > 0.0:
            v_rel = v + np.cross(om, r)
            speed = float(np.linalg.norm(v_rel))
            if speed > 0.0:
                mach = speed / atmosphere.speed_of_sound_m_s(alt)
                cd = float(np.interp(mach, ca[:, 0], ca[:, 1]))
                a = a - (0.5 * rho * speed * speed * cd * s_ref / mass_kg) * (v_rel / speed)
        return np.concatenate([v, a])

    t, y = 138.0, np.concatenate([r0, v0])
    h_prev = _geodetic(y[:3])[2]
    while t < 600.0:
        k1 = deriv(y)
        k2 = deriv(y + dt / 2 * k1)
        k3 = deriv(y + dt / 2 * k2)
        k4 = deriv(y + dt * k3)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        h = _geodetic(y[:3])[2]
        if h <= 0.0 < h_prev:
            lat, lon, _ = _geodetic(y[:3])
            tc = t - dt * h / (h - h_prev)
            vr = y[3:6] + np.cross(om, y[:3])
            return {"t_s": float(tc), "lat_deg": math.degrees(float(lat)),
                    "lon_deg": math.degrees(float(lon)),
                    "speed_earth_m_s": float(np.linalg.norm(vr))}
        h_prev = h
    return {"t_s": None, "lat_deg": None, "lon_deg": None,
            "speed_earth_m_s": None}


def _twin_diameter_m() -> float:
    import yaml
    doc = yaml.safe_load(_VEHICLE_YAML.read_text(encoding="utf-8"))
    return float(doc["stages"][0]["aero"]["diameter_m"])


# ---------------------------------------------------------------------------
# the recorded reference (sealed inputs)
# ---------------------------------------------------------------------------

def _load_recorded(reference_root: Path) -> dict[str, Any]:
    ascent = Path(reference_root) / "mission-optimized" / "launch" / "ascent"
    raw = list(csv.DictReader((ascent / "opentsiolkovsky" / "trajectory.csv").open(encoding="utf-8")))
    events = json.loads((ascent / "events.json").read_text(encoding="utf-8"))
    summary = json.loads((ascent / "opentsiolkovsky" / "summary.json").read_text(encoding="utf-8"))

    def event(kind: str, body: str | None = None) -> dict:
        for e in events:
            if e["kind"] == kind and (body is None or e.get("body_id") == body):
                return e
        raise KeyError(kind)

    return {"raw": raw, "events": events, "summary": summary,
            "event_times": {k: float(event(k, b)["t_s"]) for k, b in
                            (("meco", None), ("separation", "stage2"),
                             ("seco", None), ("max_q", None),
                             ("impact", "stage1_debris"))},
            "impact": next(b["impact"] for b in summary["bodies"]
                           if b.get("impact")),
            "debris_mass_kg": next(b["mass_kg"] for b in summary["bodies"]
                                   if b.get("impact"))}


_RAW_COLS = ("pos_eci_x_m", "pos_eci_y_m", "pos_eci_z_m", "vel_eci_x_mps",
             "vel_eci_y_mps", "vel_eci_z_mps", "mass_kg")


def _raw_interp(rec: Mapping[str, Any], t: float) -> dict[str, float]:
    """Linear interpolation of the raw 2-s state history at a 2-s grid time.
    The contract event records carry no state; the raw history is the only
    state evidence, and the declared grid times land on samples exactly."""
    raw = rec["raw"]
    for row in raw:
        if abs(float(row["time_s"]) - t) < 1e-6:
            return {c: float(row[c]) for c in _RAW_COLS}
    lo = raw[int(t // 2)]
    hi = raw[min(int(t // 2) + 1, len(raw) - 1)]
    f = (t - float(lo["time_s"])) / (float(hi["time_s"]) - float(lo["time_s"]))
    return {c: float(lo[c]) + f * (float(hi[c]) - float(lo[c])) for c in _RAW_COLS}


def _rec_state(rec: Mapping[str, Any], t: float) -> tuple[np.ndarray, np.ndarray, float]:
    s = _raw_interp(rec, t)
    r = np.array([s["pos_eci_x_m"], s["pos_eci_y_m"], s["pos_eci_z_m"]])
    v = np.array([s["vel_eci_x_mps"], s["vel_eci_y_mps"], s["vel_eci_z_mps"]])
    return r, v, s["mass_kg"]


# ---------------------------------------------------------------------------
# landing: gfold plan, FL-301 re-flight, independent re-integration
# ---------------------------------------------------------------------------

def _pad_state_and_basis() -> tuple[np.ndarray, np.ndarray]:
    from orbitopt.flightlab import frames
    pad = frames.geodetic_to_ecef(LAUNCH_LAT_DEG, LAUNCH_LON_DEG,
                                  LAUNCH_ALT_M).reshape(3)
    return pad, _enu_basis(LAUNCH_LAT_DEG, LAUNCH_LON_DEG)


def independent_landing_refly(plan_times: np.ndarray, plan_thrust_enu: np.ndarray,
                              insertion_state: Mapping[str, Any],
                              vehicle: Any, dt_s: float = 0.05) -> dict[str, Any]:
    """The independent re-flight of a solved landing plan: own RK4 at ``dt_s``,
    own WGS-84 geodesy, IGF-80 normal gravity (no spherical harmonics), own
    US76 tables, physical-thrust FOH command replay (FL-301 replays the node
    *accelerations* through the current mass; here the bounded physical
    thrust vectors are replayed directly -- a deliberately different mapping),
    zero wind. Returns the touchdown state and the cutoff-node state."""
    th = np.asarray(plan_thrust_enu, dtype=float)
    ts = np.asarray(plan_times, dtype=float)
    dry = float(vehicle.dry_mass_kg)
    eng = vehicle.engine
    cd_tab = vehicle.cd_mach
    s_ref = float(vehicle.reference_area_m2)
    n_eng = float(eng.count)
    pad, bas = _pad_state_and_basis()
    h_pad = _geodetic(pad)[2]
    om = np.array([0.0, 0.0, OMEGA_EARTH_RAD_S])

    def cmd(t: float) -> np.ndarray:
        if t >= ts[-1]:
            return np.zeros(3)
        if t <= ts[0]:
            return th[0].copy()
        k = int(np.searchsorted(ts, t)) - 1
        f = (t - ts[k]) / (ts[k + 1] - ts[k])
        return (1.0 - f) * th[k] + f * th[k + 1]

    def g_igf80(r: np.ndarray) -> np.ndarray:
        phi, lon, h = _geodetic(r)
        s2 = math.sin(phi) ** 2
        g = 9.780327 * (1.0 + 0.0053024 * s2 - 0.0000058 * math.sin(2.0 * phi) ** 2)
        g *= 1.0 - 2.0 * h / _A_WGS84
        up = np.array([math.cos(phi) * math.cos(lon), math.cos(phi) * math.sin(lon),
                       math.sin(phi)])
        return -g * up

    def deriv(t: float, y: np.ndarray) -> np.ndarray:
        r, v, m = y[:3], y[3:6], float(y[6])
        a = g_igf80(r)
        h = _geodetic(r)[2]
        p, rho, a_sound = _isa_us76(h)
        if rho > 1e-12:
            spd = float(np.linalg.norm(v))  # air at rest in the ECEF frame
            if spd > 1e-3:
                mach = spd / a_sound
                ms = [m_ for m_, _ in cd_tab]
                cs = [c for _, c in cd_tab]
                cd = float(np.interp(mach, ms, cs))
                a = a - 0.5 * rho * cd * s_ref * spd * v / m
        if t < ts[-1] and m > dry:
            t_ecef = bas.T @ cmd(t)
            fmax = eng.thrust_vacuum_n - eng.nozzle_exit_area_m2 * p
            t_mag = float(np.linalg.norm(t_ecef))
            lvl = min(max(t_mag / (n_eng * fmax), eng.throttle_min), eng.throttle_max)
            f = n_eng * lvl * max(fmax, 0.0)
            isp = eng.isp_sea_level_s + (eng.isp_vacuum_s - eng.isp_sea_level_s) * (
                1.0 - min(p, 101325.0) / 101325.0)
            a = a + f * t_ecef / (t_mag * m)
            return np.concatenate([v, a, [-(f / (isp * _G0))]])
        return np.concatenate([v, a, [0.0]])

    r0 = pad + bas.T @ np.asarray(insertion_state["position_enu_m"], float)
    v0 = bas.T @ np.asarray(insertion_state["velocity_enu_m"], float)
    y = np.concatenate([r0, v0, [float(insertion_state["mass_kg"])]])
    t = float(insertion_state["t_s"])
    t_cap = float(ts[-1]) + 60.0
    cutoff: dict[str, Any] | None = None
    while t < t_cap:
        k1 = deriv(t, y)
        k2 = deriv(t + dt_s / 2, y + dt_s / 2 * k1)
        k3 = deriv(t + dt_s / 2, y + dt_s / 2 * k2)
        k4 = deriv(t + dt_s, y + dt_s * k3)
        y2 = y + dt_s / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if cutoff is None and t < ts[-1] <= t + dt_s:
            cutoff = {"t_s": float(ts[-1]), "pos_enu_m": tuple(bas @ (y2[:3] - pad)),
                      "vel_enu_m": tuple(bas @ y2[3:6]), "mass_kg": float(y2[6])}
        h0, h1 = _geodetic(y[:3])[2], _geodetic(y2[:3])[2]
        if h1 < h_pad <= h0:
            lo_f, hi_f = 0.0, 1.0
            for _ in range(40):
                mid = 0.5 * (lo_f + hi_f)
                ym = y + (dt_s * mid) / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                if _geodetic(ym[:3])[2] < h_pad:
                    hi_f = mid
                else:
                    lo_f = mid
            y = y + (dt_s * hi_f) / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt_s * hi_f
            break
        y, t = y2, t + dt_s
    e = bas @ (y[:3] - pad)
    vv = bas @ y[3:6]
    return {"touchdown": {"t_s": float(t), "pos_enu_m": tuple(float(c) for c in e),
                          "vel_enu_m": tuple(float(c) for c in vv),
                          "mass_kg": float(y[6])},
            "cutoff": cutoff}


def _landing_pair(vehicle_yaml: Path, engines: int) -> dict[str, Any]:
    """Solve the gfold arc, re-fly it with FL-301 and with the independent
    integrator; return everything the checks need."""
    import dataclasses
    from datetime import datetime, timezone
    from orbitopt.flightlab.recovery.landing import LandingBurnEnu, fly_landing_burn
    from orbitopt.flightlab.recovery.boostback import RecoveryTarget
    from orbitopt.flightlab.recovery.propagator import BoosterModel, BoosterState
    v0 = BoosterModel.load(vehicle_yaml)
    vehicle = dataclasses.replace(v0, engine=dataclasses.replace(v0.engine,
                                                                 count=engines))
    pad, bas = _pad_state_and_basis()
    pos = np.array(LANDING_CORRIDOR_ENU_M)
    vel = np.array(LANDING_CORRIDOR_V_ENU_M_S)
    enu = LandingBurnEnu(t_s=200.0, position_enu_m=tuple(float(c) for c in pos),
                        velocity_enu_m=tuple(float(c) for c in vel),
                        mass_kg=LANDING_INSERTION_MASS_KG)
    st = BoosterState(t_s=200.0, epoch_utc=datetime.fromisoformat(_EPOCH_ISO),
                      frame="ECEF",
                      position_m=tuple(float(c) for c in pad + bas.T @ pos),
                      velocity_m_s=tuple(float(c) for c in bas.T @ vel),
                      mass_kg=LANDING_INSERTION_MASS_KG)
    target = RecoveryTarget(kind="rtls", name="kourou",
                            latitude_deg=LAUNCH_LAT_DEG, longitude_deg=LAUNCH_LON_DEG,
                            altitude_m=LAUNCH_ALT_M)
    result = fly_landing_burn(enu, st, target, vehicle)
    if not result.feasible or result.plan is None:
        raise RuntimeError(f"landing plan infeasible: {result.reason_codes}")
    plan = result.plan
    state = {"t_s": 200.0, "position_enu_m": tuple(float(c) for c in pos),
             "velocity_enu_m": tuple(float(c) for c in vel),
             "mass_kg": LANDING_INSERTION_MASS_KG}
    indep = independent_landing_refly(np.asarray(plan.times_s),
                                      np.asarray(plan.thrust_enu_n), state, vehicle)
    tr = result.track
    t_arr = np.asarray(tr.times_s)
    idx = int(np.argmin(np.abs(t_arr - plan.times_s[-1])))
    e_cut = bas @ (np.asarray(tr.position_ecef_m[idx]) - pad)
    v_cut = bas @ np.asarray(tr.velocity_ecef_m_s[idx])
    fl301_cut = {"t_s": float(plan.times_s[-1]),
                 "pos_enu_m": tuple(float(c) for c in e_cut),
                 "vel_enu_m": tuple(float(c) for c in v_cut),
                 "mass_kg": float(tr.mass_kg[idx])}
    return {"result": result, "plan": plan, "vehicle": vehicle,
            "indep": indep, "fl301_cutoff": fl301_cut}


# ---------------------------------------------------------------------------
# grading
# ---------------------------------------------------------------------------

class ReferenceNotFound(FileNotFoundError):
    """The sealed E5-07 reference tree is absent."""


def _orbit_elements(r_eci: np.ndarray, v_eci: np.ndarray) -> dict:
    from orbitopt.flightlab.checks import orbital_elements
    return orbital_elements(np.asarray(r_eci, float), np.asarray(v_eci, float))


def verify_independent_dynamics(reference_root: Path | None = None, *,
                                check_digests: bool = True, workers: int = 4,
                                ) -> dict:
    """Recompute the E5-07 ascent/debris and the declared landing corridor with
    independent dynamics models and grade them inside the declared bands.

    Returns a ``verification-report-1.0`` document of kind
    ``independent-dynamics`` whose ascent checks carry
    ``independent_model`` evidence for the native point-mass twin (and nothing
    else -- OpenTsiolkovsky-vs-OpenTsiolkovsky extraction checks are not
    counted here; see ``full_chain``), with the per-event discrepancies split
    into the measured engine / aero-atmosphere / numerical lines, the declared
    frame-time and controller lines, and a reported residual; the landing
    checks carry ``independent_model`` evidence for the two-model re-flight
    comparison and ``design_input`` evidence for the corridor and the FL-301
    gates.
    """
    from concurrent.futures import ProcessPoolExecutor
    from orbitopt.verify.full_chain import verify_reference_digests
    from orbitopt.verify.report import (
        CHECK_MODEL, CHECK_NUMERICAL, EVIDENCE_ANALYTICAL, EVIDENCE_DESIGN_INPUT,
        EVIDENCE_INDEPENDENT_MODEL, provenance, report_digest, status_from_checks,
        to_jsonable)

    root = Path(reference_root) if reference_root is not None else REFERENCE_ROOT
    if not (root / "PROVENANCE.json").is_file():
        raise ReferenceNotFound(f"{root}: sealed E5-07 reference tree not found")
    prov = json.loads((root / "PROVENANCE.json").read_text(encoding="utf-8"))
    digests = verify_reference_digests(root) if check_digests else None
    if digests is not None and not digests["ok"]:
        raise RuntimeError(f"E5-07 reference tamper check failed: {digests['failures']}")
    rec = _load_recorded(root)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            "nominal": pool.submit(_twin_ascent, TWIN_DT_S, None, 1.0),
            "engine": pool.submit(_twin_ascent, TWIN_DT_S, OT_PAD_ISP_S, 1.0),
            "aero": pool.submit(_twin_ascent, TWIN_DT_S, None, 1.03),
            "numerical": pool.submit(_twin_ascent, TWIN_DT_S / 2.0, None, 1.0),
        }
        twin = {k: f.result() for k, f in futures.items()}
    lp = _landing_pair(_VEHICLE_YAML, LANDING_ENGINES)

    details: list[dict] = []
    checks: dict[str, bool] = {}
    categories: dict[str, str] = {}
    evidence_map: dict[str, str] = {}

    def add(check_id: str, category: str, evidence: str, value: float,
            reference: str) -> None:
        band = _TOL[check_id]
        limit = band["limit"]
        passed = abs(float(value)) <= float(limit)
        details.append({"id": check_id, "category": category, "evidence": evidence,
                        "unit": band["unit"], "limit": limit, "value": float(value),
                        "passed": passed, "reference": reference,
                        "basis": band["basis"]})
        checks[check_id] = passed
        categories[check_id] = category
        evidence_map[check_id] = evidence

    # --- ascent twin vs the recorded raw state history -----------------------
    deltas: dict[str, dict[str, float]] = {}
    for variant, res in twin.items():
        d: dict[str, float] = {}
        for ev_key, t in (("meco", rec["event_times"]["meco"]),
                          ("separation", rec["event_times"]["separation"]),
                          ("seco", rec["event_times"]["seco"])):
            r, v, m = res[ev_key]
            rr, rv, rm = _rec_state(rec, t)
            r_t, v_t = _ecef_to_eci(t, r, v)
            d[f"{ev_key}_pos"] = float(np.linalg.norm(r_t - rr))
            d[f"{ev_key}_vel"] = float(np.linalg.norm(v_t - rv))
            d[f"{ev_key}_mass"] = float(m - rm)
        d["separation_debris_mass"] = float(res["debris_mass_kg"]
                                            - rec["debris_mass_kg"])
        d["maxq_value"] = float(res["maxq"][0] - rec["summary"]["maxima"]
                                ["dynamic_pressure_pa"]["value"])
        d["maxq_time"] = float(res["maxq"][1] - rec["event_times"]["max_q"])
        deltas[variant] = d

    def impact_delta(variant: str) -> tuple[float, float, float]:
        imp = twin[variant].get("impact") or {}
        ref = rec["impact"]
        if imp.get("t_s") is None:
            return (math.inf, math.inf, math.inf)
        dt_imp = imp["t_s"] - ref["time_s"]
        a1 = math.radians(imp["lat_deg"])
        o1 = math.radians(imp["lon_deg"])
        a2 = math.radians(ref["latitude_deg"])
        o2 = math.radians(ref["longitude_deg"])
        hav = (math.sin((a2 - a1) / 2) ** 2
               + math.cos(a1) * math.cos(a2) * math.sin((o2 - o1) / 2) ** 2)
        dist_km = 2 * 6371.0 * math.asin(min(1.0, math.sqrt(hav)))
        dspd = 100.0 * (imp["speed_earth_m_s"] - ref["speed_mps"]) / ref["speed_mps"]
        return (dt_imp, dist_km, dspd)

    budget: dict[str, dict[str, Any]] = {}
    for q, ev in (("meco_pos", "meco"), ("meco_vel", "meco"), ("meco_mass", "meco"),
                  ("separation_pos", "separation"), ("separation_vel", "separation"),
                  ("separation_debris_mass", "separation"),
                  ("seco_pos", "seco"), ("seco_vel", "seco"), ("seco_mass", "seco")):
        def mag(variant: str) -> float:
            v = deltas[variant][q]
            return abs(v) if q.endswith("_mass") else v
        nominal = mag("nominal")
        lines = {}
        for line in ("engine", "aero", "numerical"):
            lines[line] = nominal - mag(line)
        lines["frame_time"] = 0.0   # same epoch/rate/ECI convention; pad < 1 m at t=0
        lines["controller"] = 0.0   # same authored program; tracking implementation -> residual
        lines["residual"] = nominal - sum(lines[k] for k in
                                          ("engine", "aero", "numerical",
                                           "frame_time", "controller"))
        budget[q] = {"measured_delta": nominal, "lines": lines}

    for _q, _cid in (("meco_pos", "ascent.meco.residual_position"),
                     ("meco_vel", "ascent.meco.residual_velocity"),
                     ("separation_pos", "ascent.separation.residual_position"),
                     ("seco_pos", "ascent.seco.residual_position")):
        add(_cid, CHECK_MODEL, EVIDENCE_INDEPENDENT_MODEL,
            budget[_q]["lines"]["residual"],
            "nominal discrepancy minus the measured error-budget lines "
            "(reported, never zeroed; E8-02 accounting discipline)")

    tn = twin["nominal"]
    for ev_key, t in (("meco", rec["event_times"]["meco"]),
                      ("separation", rec["event_times"]["separation"]),
                      ("seco", rec["event_times"]["seco"])):
        add(f"ascent.{ev_key}.state_position", CHECK_MODEL,
            EVIDENCE_INDEPENDENT_MODEL, deltas["nominal"][f"{ev_key}_pos"],
            "point-mass twin (repo 3-DOF plant, open-loop authored program) "
            "vs the raw OpenTsiolkovsky state history interpolated at the "
            "declared event epoch")
        add(f"ascent.{ev_key}.state_velocity", CHECK_MODEL,
            EVIDENCE_INDEPENDENT_MODEL, deltas["nominal"][f"{ev_key}_vel"],
            "same comparison, velocity")
    add("ascent.meco.mass", CHECK_MODEL, EVIDENCE_INDEPENDENT_MODEL,
        deltas["nominal"]["meco_mass"], "twin stage-1 burnout mass vs raw")
    add("ascent.separation.debris_mass", CHECK_MODEL, EVIDENCE_INDEPENDENT_MODEL,
        deltas["nominal"]["separation_debris_mass"],
        "twin separation mass minus the 1.2 Mt ship vs the recorded stage-1 body mass")
    add("ascent.seco.mass", CHECK_MODEL, EVIDENCE_INDEPENDENT_MODEL,
        deltas["nominal"]["seco_mass"], "twin stage-2 burnout mass vs raw")
    add("ascent.maxq.value", CHECK_MODEL, EVIDENCE_INDEPENDENT_MODEL,
        deltas["nominal"]["maxq_value"],
        "twin max-q vs the recorded maxima (dynamic_pressure_pa)")
    add("ascent.maxq.time", CHECK_MODEL, EVIDENCE_INDEPENDENT_MODEL,
        deltas["nominal"]["maxq_time"], "same, epoch of the max")

    # SECO orbit: same closed form on both states; a/e/i are invariant under
    # the (common, epoch-fixed) launch-aligned rotation, so the comparison is
    # frame-convention-free.
    r_t, v_t, _ = tn["seco"]
    r_eci, v_eci = _ecef_to_eci(rec["event_times"]["seco"], r_t, v_t)
    el_twin = _orbit_elements(r_eci, v_eci)
    rr, rv, _ = _rec_state(rec, rec["event_times"]["seco"])
    el_rec = _orbit_elements(rr, rv)
    add("ascent.seco.orbit_semi_major_axis", CHECK_MODEL,
        EVIDENCE_INDEPENDENT_MODEL,
        el_twin["semi_major_axis_m"] - el_rec["semi_major_axis_m"],
        "closed-form elements of the twin SECO state vs the recorded raw "
        "state at the same epoch (same formula both sides)")
    add("ascent.seco.orbit_eccentricity", CHECK_MODEL, EVIDENCE_INDEPENDENT_MODEL,
        el_twin["eccentricity"] - el_rec["eccentricity"], "same comparison")
    add("ascent.seco.orbit_inclination", CHECK_MODEL, EVIDENCE_INDEPENDENT_MODEL,
        el_twin["inclination_deg"] - el_rec["inclination_deg"], "same comparison")

    dt_i, dist_i, dsp_i = impact_delta("nominal")
    add("ascent.debris_impact.time", CHECK_MODEL, EVIDENCE_INDEPENDENT_MODEL,
        dt_i, "twin power-off debris re-entry (own RK4, CA polar, US76) "
              "impact epoch vs the recorded impact event")
    add("ascent.debris_impact.position", CHECK_MODEL, EVIDENCE_INDEPENDENT_MODEL,
        dist_i, "great-circle distance between impact sites")
    add("ascent.debris_impact.speed", CHECK_MODEL, EVIDENCE_INDEPENDENT_MODEL,
        dsp_i, "Earth-relative impact speed difference, percent of recorded")

    # --- landing: two independent re-flights of one solved plan --------------
    res = lp["result"]
    fl_c, in_c = lp["fl301_cutoff"], lp["indep"]["cutoff"]
    in_t = lp["indep"]["touchdown"]
    tr = res.track
    fl_touch = res.touchdown_enu_m
    d_cut_p = float(np.linalg.norm(np.subtract(fl_c["pos_enu_m"], in_c["pos_enu_m"])))
    d_cut_v = float(np.linalg.norm(np.subtract(fl_c["vel_enu_m"], in_c["vel_enu_m"])))
    d_td_p = float(np.linalg.norm(np.subtract(fl_touch, in_t["pos_enu_m"])))
    d_td_v = float(abs(res.vertical_speed_m_s - in_t["vel_enu_m"][2]))
    d_td_lat = float(math.hypot(res.lateral_speed_m_s
                                - math.hypot(in_t["vel_enu_m"][0],
                                             in_t["vel_enu_m"][1]), 0.0))
    d_prop = float(abs(tr.mass_kg[-1] - in_t["mass_kg"]))
    add("landing.cutoff_position", CHECK_MODEL, EVIDENCE_INDEPENDENT_MODEL,
        d_cut_p, "FL-301 re-flight vs independent re-integrator at the "
                 "commanded cutoff node (pad-local ENU)")
    add("landing.cutoff_velocity", CHECK_MODEL, EVIDENCE_INDEPENDENT_MODEL,
        d_cut_v, "same, velocity")
    add("landing.touchdown_position", CHECK_MODEL, EVIDENCE_INDEPENDENT_MODEL,
        d_td_p, "touchdown position disagreement between the two re-flights")
    add("landing.touchdown_vertical_speed", CHECK_MODEL,
        EVIDENCE_INDEPENDENT_MODEL, d_td_v,
        "vertical-speed disagreement at touchdown (cutoff-state offset "
        "amplified ballistically; the design limit itself is graded on the "
        "FL-301 re-flight, design_input)")
    add("landing.touchdown_lateral_speed", CHECK_MODEL,
        EVIDENCE_INDEPENDENT_MODEL, d_td_lat, "lateral-speed disagreement")
    add("landing.propellant", CHECK_MODEL, EVIDENCE_INDEPENDENT_MODEL, d_prop,
        "remaining-propellant disagreement at the end of the burn")

    gate_vals = {c.id: c for c in res.checks}
    util = max(
        abs(gate_vals["landing-position"].value) / gate_vals["landing-position"].limit,
        abs(gate_vals["landing-vertical-speed"].value)
        / gate_vals["landing-vertical-speed"].limit,
        abs(gate_vals["landing-lateral-speed"].value)
        / gate_vals["landing-lateral-speed"].limit,
        0.0 if gate_vals["landing-propellant-reserve"].passed else math.inf,
        0.0 if res.landed else math.inf)
    band = {"unit": "-", "limit": 1.0,
            "basis": "FL-301 design gates on the shared re-flight (gfold arc, "
                     "FL-303 re-integration): worst gate utilisation; the "
                     "corridor and the 3-engine landing configuration are "
                     "design inputs, so this is the design grading, not "
                     "independent validation"}
    details.append({"id": "landing.fl301_gates", "category": CHECK_MODEL,
                    "evidence": EVIDENCE_DESIGN_INPUT, "unit": "-", "limit": 1.0,
                    "value": util, "passed": util <= 1.0,
                    "reference": "FL-301 landing checks of the re-flown plan",
                    "basis": band["basis"]})
    checks["landing.fl301_gates"] = util <= 1.0
    categories["landing.fl301_gates"] = CHECK_MODEL
    evidence_map["landing.fl301_gates"] = EVIDENCE_DESIGN_INPUT

    status = status_from_checks(checks, categories, evidence_map)
    report = {
        "schemaVersion": "1.0",
        "kind": "independent-dynamics",
        "label": ("E5-07 Starship-class reference: ascent and stage-1 debris "
                  "recomputed with a native point-mass twin (E8-05), and the "
                  "declared landing corridor re-flown through two independent "
                  "dynamics models"),
        "status": status,
        "inputs": {
            "reference": {
                "root": str(root), "description": prov.get("reference"),
                "provenance_files_checked": (digests or {}).get("files_checked"),
                "provenance_ok": None if digests is None else digests["ok"],
                "toolchain": prov.get("toolchain"),
                "git_sha_recording": prov.get("git_sha_recording"),
            },
            "tolerances": declared_tolerances(),
            "check_details": details,
            "evidence_map": dict(sorted(evidence_map.items())),
            "error_budget": budget,
            "probes": {
                "engine": f"twin re-run with the recorded pad Isp {OT_PAD_ISP_S} s "
                          "(the difference between the authored YAML Isp law and "
                          "the recorded OpenTsiolkovsky Isp law)",
                "aero": "twin re-run with atmosphere density x1.03 (US76 vs the "
                        "recorded program's own atmosphere model, ~3% band near "
                        "sea level)",
                "numerical": f"twin re-run at dt/2 ({TWIN_DT_S / 2} s)",
            },
            "landing_corridor": {
                "position_enu_m": list(LANDING_CORRIDOR_ENU_M),
                "velocity_enu_m": list(LANDING_CORRIDOR_V_ENU_M_S),
                "mass_kg": LANDING_INSERTION_MASS_KG, "t_s": 200.0,
                "engines": LANDING_ENGINES,
                "plan_tof_s": float(lp["plan"].times_s[-1] - lp["plan"].times_s[0]),
                "fl301_position_error_m": res.position_error_m,
                "fl301_vertical_speed_m_s": res.vertical_speed_m_s,
                "fl301_lateral_speed_m_s": res.lateral_speed_m_s,
                "fl301_final_mass_kg": float(tr.mass_kg[-1]),
                "independent_touchdown": in_t, "independent_cutoff": lp["indep"]["cutoff"],
            },
            "assumption_differences": [
                "ascent: the two models share the authored attitude program, "
                "throttle schedules and vehicle constants (design inputs); they "
                "differ in formulation (external OpenTsiolkovsky vs the repo "
                "3-DOF plant), in the engine Isp law (YAML linear-in-pressure "
                "law, pad 327 s, vs the recorded pad 329.26 s law -- the "
                "measured engine line), the atmosphere (US76 x1.03 band -- the "
                "measured aero line) and the integrator step (RK4 0.02 s vs the "
                "recorded program's own scheme -- the measured numerical line).",
                "ascent frame: both sides expressed in the launch-aligned "
                "ECI-at-t=0 convention (raw columns rotated by R3(-omega*t)); "
                "the t=0 pad state agrees to < 1 m, which is why the frame/time "
                "line is declared zero rather than fitted.",
                "debris: the twin's re-entry is a separately written variable-"
                "mass-free ballistic integrator with the power-off CA polar "
                "from the same YAML; OpenTsiolkovsky integrates its own debris "
                "body; the impact comparison is therefore an emergent, "
                "widest-band check.",
                "landing: no landing is flown in E5-07 (RECOVERY_GAP.md); the "
                "corridor insertion state is a declared design input. Both "
                "re-flights share the gfold plan nodes and the 3-engine "
                "landing configuration declared from the vehicle file's own "
                "note; they differ in gravity (J2 point-mass + centrifugal vs "
                "IGF-80 normal gravity with the centrifugal absorbed, Coriolis "
                "omitted -- below 0.15 m on this arc), atmosphere (FL-301's "
                "entry density table vs independent US76 tables), command "
                "replay (node-acceleration FOH x current mass vs physical-"
                "thrust FOH with the engine's own ambient correction and "
                "throttle limits) and integrator (RK4 0.1 s vs RK4 0.05 s).",
                "no check in this report uses historical_telemetry evidence: "
                "the reference vehicle is a public estimate, not flown "
                "hardware, and there is no telemetry to compare against.",
            ],
            "honesty_notes": [
                "E8-01's comparisons of the chain against OpenTsiolkovsky's "
                "own raw output are extraction/serialisation checks and are "
                "relabelled numerical_consistency/analytical in full_chain; "
                "the independent ascent-model evidence for the E5-07 reference "
                "is this module's twin, and only this module's.",
                "Residual lines are reported, never zeroed; where a probe "
                "over-explains (negative line), the surplus is carried in the "
                "residual.",
            ],
        },
        "propagations": [],
        "residuals": [],
        "targets": {"converged": True},
        "provenance": provenance(),
    }
    report = to_jsonable(report)
    report["digest"] = report_digest(report)
    return report


def main(argv=None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        prog="python -m orbitopt.verify.independent_dynamics",
        description="Recompute the E5-07 reference ascent and landing corridor "
                    "with independent dynamics models (E8-05).")
    ap.add_argument("--reference", type=Path, default=REFERENCE_ROOT)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    report = verify_independent_dynamics(args.reference, workers=args.workers)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n",
                            encoding="utf-8")
        print(f"wrote {args.out}")
    status = report["status"]
    for d in report["inputs"]["check_details"]:
        mark = "ok  " if d["passed"] else "FAIL"
        print(f"{mark} {d['id']:<44} {d['value']:+.6g} [{d['evidence']}]")
    print(f"independent dynamics: {status['result']}"
          f" ({sum(status['checks'].values())}/{len(status['checks'])} checks;"
          f" independent validation: {status['independent_validation']['result']})")
    return 0 if status["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
