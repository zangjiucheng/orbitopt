"""V-02 (task #241): cross-model comparison harness -- 3-DOF / 6-DOF / TudatPy.

One scenario, several physics backends, automatic difference metrics. The
same :class:`CrossModelScenario` (a spike-stage ascent to MECO plus a
gravity coast -- the FL-403 spike lineage) is flown by every backend that is
available in this environment, the resulting trajectories are resampled onto
a common time grid, and the pairwise discrepancy is quantified per channel:

* **position** [m] and **velocity** [m/s] -- max / RMS along the overlap and
  the difference at the end of the shared phase window;
* **mass** [kg] -- whenever both backends track propellant (3-DOF vs 6-DOF
  during the burn, handoff continuity across the coast);
* **classical orbital elements** [a m, e -, i/RAAN/argP deg] -- at the end
  of each shared phase window, from each backend's own GCRS state
  (``orbitopt.flightlab.checks.orbital_elements``, the Gate-B convention).

Every number is written into one verification-report-1.0 document (kind
``cross-model``) as one residual per backend pair and phase, each carrying
its tolerance block and ``within_tolerance``, so the report's ``status`` --
and therefore the pipeline gate, which grades candidates through
``VerificationGate.evaluate`` on exactly these reports -- is the threshold
verdict. Thresholds are set through :class:`CrossModelThresholds` (defaults
plus dotted-key overrides) and on the CLI via ``--threshold``, *not* by
editing constants: a metric without a resolved threshold is *recorded* but
never *checked* (task #243: absent evidence is never a pass).

Backends
--------
``three_dof``
    Point-mass ascent integrated here (scipy DOP853, ``MU_EARTH``/``J2``
    from :mod:`orbitopt.entry`) in the Earth-fixed frame with Coriolis and
    centrifugal terms and the FL-403 azimuth/elevation program evaluated in
    the local NED frame at the *current* vehicle position -- the same
    convention the JSBSim tracking loop uses. Always available; it is also
    the reference state producer (the MECO handoff) for the coast backends.
``six_dof``
    The JSBSim ``spike_stage`` (FL-401 model, same vehicle numbers as the
    scenario defaults) flown closed-loop on the scenario program with the
    FL-403 PID gimbal controller
    (``orbitopt.flightlab.sixdof.tracking.run_tracking_spike``), mass read
    from the tanks. Requires ``jsbsim`` (opt-in: ``envs/sixdof-requirements.txt``).
``tudatpy``
    The 3-DOF MECO state propagated as a coast arc with tudatpy under the
    scenario's coast force model (``orbitopt.mission.propagation.propagate_injected_orbit``,
    the M-05 machinery: same GM/radius/J2 triple, GCRS in / GCRS out).
    Requires ``tudatpy``.
``two_body_analytic``
    The 3-DOF MECO state propagated analytically (closed-form two-body,
    :func:`orbitopt.verify.benchmarks.propagate_kepler_analytic`). The same
    *model* as a point-mass coast re-run by independent numerics, so this
    pair is categorised ``numerical_consistency``; every other pair compares
    different models and is categorised ``model_agreement``. Always
    available, so at least one real cross-check runs on a base install.

Frames: every track is bridged to GCRS (``orbitopt.flightlab.frames.to_gcrs``
per sample for Earth-fixed tracks) at the scenario epoch, so all differences
are computed in one inertial frame; element angles RAAN/argP are only
*gated* when the inclination is well defined (> 1 deg on both sides) --
they are still recorded (an equatorial FL-403 MECO has an ill-conditioned
node).

A backend that is not installed is recorded (``status: unavailable``) and
never faked; a backend that fails at runtime is recorded with its error and
the harness continues with the rest. With zero comparable pairs the report
reads ``not_checked``.

    from orbitopt.verify.cross_model import run_cross_model
    res = run_cross_model(thresholds={"coast.position_m": "100"})
    print(res.report["status"]["result"])     # gate-gradeable
    # in a pipeline gate:
    gate.register(0, scenario="cross-model"); gate.evaluate(0, [res.report])

Run from the command line (exit code 1 if a gated check fails):

    python -m orbitopt.verify.cross_model [--backends ...] [--threshold K=V]
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Callable, Mapping, Sequence

import numpy as np

from orbitopt.entry import J2_EARTH, MU_EARTH_M3_S2, R_EARTH_M
from orbitopt.flightlab.adapters.opentsiolkovsky import AttitudePoint
from orbitopt.flightlab.frames import EARTH_ROTATION_RATE, WGS84_A, WGS84_F
from orbitopt.verify.report import (
    CHECK_MODEL,
    CHECK_NUMERICAL,
    REPORT_DIR_ENV,
    build_report,
    default_report_path,
    finalize_report,
    format_status,
    status_result,
)

__all__ = [
    "BACKENDS",
    "CrossModelScenario",
    "CrossModelThresholds",
    "CrossModelVerification",
    "DEFAULT_BACKENDS",
    "DEFAULT_PROGRAM",
    "DEFAULT_THRESHOLDS",
    "ModelTrack",
    "STATUS_FAILED",
    "STATUS_RAN",
    "STATUS_UNAVAILABLE",
    "THRESHOLD_METRICS",
    "THRESHOLD_PHASES",
    "compare_tracks",
    "register_backend",
    "run_cross_model",
]

G0_M_S2 = 9.80665
OMEGA_EARTH = np.array([0.0, 0.0, EARTH_ROTATION_RATE])
MU_EARTH = MU_EARTH_M3_S2
_WGS84_A, _WGS84_F = WGS84_A, WGS84_F

STATUS_RAN = "ran"
STATUS_UNAVAILABLE = "unavailable"
STATUS_FAILED = "failed"

#: Elements whose *difference* is always gated. RAAN/argP depend on a
#: well-defined node; they are gated only when both sides have i > 1 deg
#: (recorded regardless).
GATED_ELEMENTS_ALWAYS = ("semi_major_axis_m", "eccentricity", "inclination_deg")
GATED_ELEMENTS_CONDITIONAL = ("raan_deg", "argument_of_perigee_deg")
MIN_INCLINATION_FOR_ANGLES_DEG = 1.0
MIN_ECCENTRICITY_FOR_ANGLES = 0.01

#: Every threshold metric name, per phase (the full threshold key space).
THRESHOLD_PHASES = ("ascent", "coast")
THRESHOLD_METRICS = ("position_m", "velocity_m_s", "mass_kg",
                     "semi_major_axis_m", "eccentricity", "inclination_deg",
                     "raan_deg", "argument_of_perigee_deg")

#: Defaults, deliberately loose compared to what the models actually agree
#: to (FL-403 measured 51 m / 2 m/s MECO deviation between JSBSim and the
#: OT reference; point-mass coast pairs agree to centimetres): the gate
#: catches a *broken* cross-model stack, not the last centimetre of physics.
#: ``None`` = record the metric, do not gate it.
DEFAULT_THRESHOLDS: dict = {
    "ascent": {"position_m": 500.0, "velocity_m_s": 20.0, "mass_kg": 500.0,
               "semi_major_axis_m": 20_000.0, "eccentricity": 0.05,
               "inclination_deg": 0.5, "raan_deg": 1.0,
               "argument_of_perigee_deg": 1.0},
    "coast": {"position_m": 200.0, "velocity_m_s": 0.05, "mass_kg": 1.0,
              "semi_major_axis_m": 500.0, "eccentricity": 0.005,
              "inclination_deg": 0.05, "raan_deg": 0.1,
              "argument_of_perigee_deg": 0.1},
}

#: Same waypoints as the FL-403 reference vehicle's attitude program
#  (tests/fixtures/.../fl403_spike/ascent_vehicle.yaml): vertical hold to
#  T+8 s, pitch over to 18 deg elevation by T+50 s, hold to MECO at T+105 s.
DEFAULT_PROGRAM: tuple[AttitudePoint, ...] = (
    AttitudePoint(t_s=0.0, azimuth_deg=90.0, elevation_deg=90.0),
    AttitudePoint(t_s=8.0, azimuth_deg=90.0, elevation_deg=90.0),
    AttitudePoint(t_s=50.0, azimuth_deg=90.0, elevation_deg=18.0),
    AttitudePoint(t_s=105.0, azimuth_deg=90.0, elevation_deg=18.0),
)


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CrossModelScenario:
    """One spike-stage ascent + coast scenario flown by every backend.
    Defaults mirror the FL-403 3-DOF/6-DOF comparison fixture: 10 t dry +
    60 t propellant, Isp 320 s, 600 kg/s, 100 s burn, MECO event at T+105 s,
    equator launch due east, zero aerodynamics on every side (so the
    comparison is thrust + gravity only)."""

    name: str = "spike_stage_meco_coast"
    epoch_mjd2000: float = 9497.0  # 2026-01-01T00:00:00Z (the FL-403 target epoch)
    latitude_deg: float = 0.0
    longitude_deg: float = 0.0
    altitude_m: float = 0.0
    dry_mass_kg: float = 10_000.0
    propellant_mass_kg: float = 60_000.0
    isp_s: float = 320.0
    mdot_kg_s: float = 600.0
    burn_start_s: float = 0.0
    burn_end_s: float = 100.0
    meco_t_s: float = 105.0  # MECO *event* (5 s of coast after burnout, FL-403)
    coast_duration_s: float = 300.0
    program: tuple[AttitudePoint, ...] = DEFAULT_PROGRAM
    ascent_j2: bool = False
    coast_j2: bool = False
    ascent_output_step_s: float = 0.5
    coast_output_samples: int = 121

    def __post_init__(self):
        if self.dry_mass_kg <= 0.0 or self.propellant_mass_kg < 0.0:
            raise ValueError("dry_mass_kg must be > 0 and propellant_mass_kg >= 0")
        if not (0.0 <= self.burn_start_s < self.burn_end_s <= self.meco_t_s):
            raise ValueError(f"need burn_start < burn_end <= meco_t, got "
                             f"{self.burn_start_s}/{self.burn_end_s}/{self.meco_t_s}")
        if self.coast_duration_s <= 0.0:
            raise ValueError("coast_duration_s must be > 0 (the coast phase feeds the coast pairs)")
        if self.mdot_kg_s <= 0.0 or self.isp_s <= 0.0:
            raise ValueError("mdot_kg_s and isp_s must be > 0")
        pts = tuple(self.program)
        if len(pts) < 2 or any(not (p.t_s >= 0.0) for p in pts) \
                or any(pts[i + 1].t_s <= pts[i].t_s for i in range(len(pts) - 1)):
            raise ValueError("program must be >= 2 AttitudePoints with strictly increasing t_s")
        object.__setattr__(self, "program", pts)
        burn_s = self.burn_end_s - self.burn_start_s
        if self.mdot_kg_s * burn_s > self.propellant_mass_kg + 1e-9:
            raise ValueError(f"mdot {self.mdot_kg_s} kg/s over {burn_s} s exceeds the "
                             f"propellant load {self.propellant_mass_kg} kg")

    @property
    def thrust_n(self) -> float:
        """Vacuum thrust, same definition as the spike stage: mdot * Isp * g0."""
        return self.mdot_kg_s * self.isp_s * G0_M_S2

    def mass_at(self, t: float) -> float:
        """Wet mass at mission time ``t`` (constant mdot during the burn, then dry)."""
        burned = self.mdot_kg_s * (min(max(float(t), self.burn_start_s), self.burn_end_s)
                                   - self.burn_start_s)
        return self.dry_mass_kg + self.propellant_mass_kg - burned

    def attitude_at(self, t: float) -> tuple[float, float]:
        """(elevation, azimuth) [rad] by piecewise-linear interpolation of
        the program (held outside its ends)."""
        ts = np.array([p.t_s for p in self.program], dtype=float)
        el = np.array([p.elevation_deg for p in self.program], dtype=float)
        az = np.array([p.azimuth_deg for p in self.program], dtype=float)
        return (math.radians(float(np.interp(t, ts, el))),
                math.radians(float(np.interp(t, ts, az))))

    def epoch_utc(self) -> datetime:
        from orbitopt.timescales import mjd2000_to_utc_datetime

        return mjd2000_to_utc_datetime(self.epoch_mjd2000).astimezone(timezone.utc)

    def to_dict(self) -> dict:
        return {"name": self.name, "epoch_mjd2000": self.epoch_mjd2000,
                "latitude_deg": self.latitude_deg, "longitude_deg": self.longitude_deg,
                "altitude_m": self.altitude_m, "dry_mass_kg": self.dry_mass_kg,
                "propellant_mass_kg": self.propellant_mass_kg, "isp_s": self.isp_s,
                "mdot_kg_s": self.mdot_kg_s, "burn_start_s": self.burn_start_s,
                "burn_end_s": self.burn_end_s, "meco_t_s": self.meco_t_s,
                "coast_duration_s": self.coast_duration_s, "thrust_n": self.thrust_n,
                "ascent_j2": bool(self.ascent_j2), "coast_j2": bool(self.coast_j2),
                "program": [{"t_s": p.t_s, "elevation_deg": p.elevation_deg,
                             "azimuth_deg": p.azimuth_deg} for p in self.program]}


# ---------------------------------------------------------------------------
# Tracks and backend outcomes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelTrack:
    """One backend's trajectory: mission-relative time [s], GCRS states
    [n,6] (m, m/s) and optionally mass [n] kg."""

    label: str
    t_s: np.ndarray
    states_gcrs: np.ndarray
    mass_kg: np.ndarray | None = None

    def __post_init__(self):
        t = np.asarray(self.t_s, dtype=float)
        s = np.asarray(self.states_gcrs, dtype=float)
        if t.ndim != 1 or t.size < 2 or s.shape != (t.size, 6):
            raise ValueError(f"{self.label}: track needs t (n,) and states (n,6), n>=2")
        if not np.all(np.diff(t) > 0.0):
            raise ValueError(f"{self.label}: track times must strictly increase")
        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(t))):
            raise ValueError(f"{self.label}: track contains non-finite samples")
        if self.mass_kg is not None:
            m = np.asarray(self.mass_kg, dtype=float)
            if m.shape != (t.size,) or not np.all(np.isfinite(m)):
                raise ValueError(f"{self.label}: mass must be finite (n,)")
            object.__setattr__(self, "mass_kg", m)
        object.__setattr__(self, "t_s", t)
        object.__setattr__(self, "states_gcrs", s)

    @property
    def t_start(self) -> float:
        return float(self.t_s[0])

    @property
    def t_end(self) -> float:
        return float(self.t_s[-1])


@dataclass(frozen=True)
class BackendOutcome:
    """What one backend produced for one scenario: never a fake track --
    an unavailable backend says so, a failed one carries its error."""

    label: str
    status: str
    detail: str | None = None
    ascent: ModelTrack | None = None
    coast: ModelTrack | None = None

    def to_dict(self) -> dict:
        return {"backend": self.label, "status": self.status, "detail": self.detail,
                "ascent": None if self.ascent is None else _track_summary(self.ascent),
                "coast": None if self.coast is None else _track_summary(self.coast)}


def _track_summary(track: ModelTrack) -> dict:
    return {"t_start_s": track.t_start, "t_end_s": track.t_end,
            "samples": int(track.t_s.size),
            "mass_kg": None if track.mass_kg is None else float(track.mass_kg[-1])}


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CrossModelThresholds:
    """Gate thresholds per phase and metric, with dotted-key overrides:
    ``"<metric>"`` / ``"<phase>.<metric>"`` / ``"<pair>.<phase>.<metric>"``
    (most specific wins). A resolved value of ``None`` records the metric
    without gating it; the string ``"none"`` in an override means the same.
    This is the knob the pipeline gate reads -- the report's checks are
    exactly the gated metrics."""

    overrides: Mapping[str, float | None] = field(default_factory=dict)

    def resolve(self, pair: str, phase: str, metric: str) -> float | None:
        for key in (f"{pair}.{phase}.{metric}", f"{phase}.{metric}", metric):
            if key in self.overrides:
                value = self.overrides[key]
                return None if value is None else float(value)
        return DEFAULT_THRESHOLDS[phase][metric]

    def to_dict(self) -> dict:
        return {"defaults": {phase: dict(DEFAULT_THRESHOLDS[phase]) for phase in THRESHOLD_PHASES},
                "overrides": {k: (None if v is None else float(v))
                              for k, v in sorted(self.overrides.items())}}

    @staticmethod
    def coerce(arg: "CrossModelThresholds | Mapping | None") -> "CrossModelThresholds":
        if arg is None:
            return CrossModelThresholds()
        if isinstance(arg, CrossModelThresholds):
            return arg
        return CrossModelThresholds(overrides=parse_thresholds(arg))


def parse_thresholds(mapping: Mapping) -> dict:
    """Validate a ``{dotted key: value}`` threshold mapping (values may be
    numbers or ``None``/``"none"`` to un-gate a metric)."""
    out: dict[str, float | None] = {}
    for key, value in dict(mapping).items():
        _validate_threshold_key(str(key))
        if value is None or (isinstance(value, str) and value.strip().lower() == "none"):
            out[str(key)] = None
        else:
            f = float(value)
            if not math.isfinite(f) or f <= 0.0:
                raise ValueError(f"threshold {key}={value!r}: must be a finite number > 0 (or None)")
            out[str(key)] = f
    return out


def _validate_threshold_key(key: str) -> None:
    parts = key.split(".")
    metric = parts[-1]
    if metric not in THRESHOLD_METRICS:
        raise ValueError(f"threshold key {key!r}: unknown metric {metric!r} "
                         f"(have {THRESHOLD_METRICS})")
    if len(parts) == 2 and parts[0] not in THRESHOLD_PHASES:
        raise ValueError(f"threshold key {key!r}: unknown phase {parts[0]!r} "
                         f"(have {THRESHOLD_PHASES})")
    if len(parts) > 3:
        raise ValueError(f"threshold key {key!r}: at most <pair>.<phase>.<metric>")


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

# label -> (producer(scenario, context) -> (ascent|None, coast|None), availability)
BACKENDS: dict[str, tuple[Callable, Callable[[], bool]]] = {}

#: Backends run by :func:`run_cross_model` when ``backends=None``.
DEFAULT_BACKENDS: tuple[str, ...] = ("three_dof", "six_dof", "tudatpy", "two_body_analytic")


def register_backend(key: str, producer: Callable,
                     available: Callable[[], bool] = lambda: True) -> None:
    """Add (or replace) a backend -- the extension point for tests (a fake
    backend flying a known orbit) and for future propagators.
    ``producer(scenario, context)`` returns ``(ascent_track|None,
    coast_track|None)``; ``context`` is a dict the harness fills as backends
    run (it carries ``"three_dof": BackendOutcome`` once that backend has
    run, so coast backends can start from the same handoff state)."""
    BACKENDS[key] = (producer, available)


def _module_available(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


register_backend("three_dof", lambda s, c: _produce_three_dof(s))
register_backend("six_dof", lambda s, c: _produce_six_dof(s, c), lambda: _module_available("jsbsim"))
register_backend("tudatpy", lambda s, c: _produce_tudatpy(s, c), lambda: _module_available("tudatpy"))
register_backend("two_body_analytic", lambda s, c: _produce_analytic(s, c))


# ---------------------------------------------------------------------------
# 3-DOF point-mass backend (scipy; ascent in ECEF, coast in GCRS)
# ---------------------------------------------------------------------------

def _ned_basis(lat_rad: float, lon_rad: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit east/north/up vectors in ECEF at a geodetic position."""
    sl, cl = math.sin(lat_rad), math.cos(lat_rad)
    so, co = math.sin(lon_rad), math.cos(lon_rad)
    e = np.array([-so, co, 0.0])
    n = np.array([-sl * co, -sl * so, cl])
    u = np.array([cl * co, cl * so, sl])
    return e, n, u


def _geodetic_latlon(ecef) -> tuple[float, float]:
    """Scalar WGS84 geodetic (lat, lon) [rad] of one ECEF position -- the
    cheap inverse of :func:`orbitopt.flightlab.frames.geodetic_to_ecef`
    (Bowring seed + fixed point, agrees with the batched
    ``ecef_to_geodetic`` to ~1e-13 rad); hot loop, so pure stdlib math."""
    x, y, z = (float(c) for c in np.asarray(ecef, dtype=float).reshape(3))
    e2 = _WGS84_F * (2.0 - _WGS84_F)
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    if p == 0.0:
        return (math.copysign(math.pi / 2.0, z), lon)
    theta = math.atan2(p, z * (1.0 - _WGS84_F))
    sin_t, cos_t = math.sin(theta), math.cos(theta)
    ep2 = e2 / (1.0 - e2)
    lat = math.atan2(z + ep2 * (_WGS84_A / math.sqrt(1.0 + ep2 ** 2 * sin_t ** 2)) * sin_t ** 3,
                     p - e2 * (_WGS84_A / math.sqrt(1.0 + ep2 ** 2 * sin_t ** 2)) * cos_t ** 3)
    for _ in range(3):
        n_rm = _WGS84_A / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
        lat = math.atan2(z + e2 * n_rm * math.sin(lat), p)
    return lat, lon


def _thrust_direction_ecef(scenario: CrossModelScenario, r_ecef, t: float) -> np.ndarray:
    """Program thrust unit vector in ECEF: elevation/azimuth in the NED
    frame at the vehicle's *current* geodetic position (the FL-403 tracking
    convention), rotated to ECEF."""
    el, az = scenario.attitude_at(t)
    lat, lon = _geodetic_latlon(r_ecef)
    e, n, u = _ned_basis(lat, lon)
    return (u * math.sin(el) + n * math.cos(el) * math.cos(az)
            + e * math.cos(el) * math.sin(az))


def _gravity_point_mass(r: np.ndarray, j2: bool) -> np.ndarray:
    """Point mass (+ optional J2 zonal) on a position in the frame the
    J2 figure is defined in (body-fixed: the caller rotates when needed)."""
    rn = float(np.linalg.norm(r))
    a = -MU_EARTH * r / rn ** 3
    if j2:
        z2 = r[2] * r[2] / (rn * rn)
        k = -1.5 * J2_EARTH * MU_EARTH * R_EARTH_M ** 2 / rn ** 5
        a = a + k * np.array([r[0] * (1.0 - 5.0 * z2), r[1] * (1.0 - 5.0 * z2),
                              r[2] * (3.0 - 5.0 * z2)])
    return a


def _to_gcrs_pair(states_ecef: np.ndarray, t: np.ndarray, epoch: datetime):
    from orbitopt.flightlab.frames import to_gcrs

    return to_gcrs(states_ecef[:, :3], states_ecef[:, 3:], "ECEF", epoch, t)


def _produce_three_dof(scenario: CrossModelScenario):
    from scipy.integrate import solve_ivp

    from orbitopt.flightlab.frames import geodetic_to_ecef

    r0 = np.asarray(geodetic_to_ecef(scenario.latitude_deg, scenario.longitude_deg,
                                     scenario.altitude_m), dtype=float).reshape(3)
    v0 = np.zeros(3)  # at rest on the rotating Earth: zero *ECEF* velocity
    # (the to_gcrs bridge below supplies the +465 m/s of ω×r the inertial
    # frame sees; the ECEF equations of motion carry it via Coriolis +
    # centrifugal terms).
    omega = OMEGA_EARTH
    meco, b0, b1 = scenario.meco_t_s, scenario.burn_start_s, scenario.burn_end_s

    def rhs_ascent(t, s):
        r, v = s[:3], s[3:]
        a = _gravity_point_mass(r, scenario.ascent_j2)
        a = a - 2.0 * np.cross(omega, v) - np.cross(omega, np.cross(omega, r))
        if b0 <= t < b1:
            a = a + (scenario.thrust_n / scenario.mass_at(t)) * \
                _thrust_direction_ecef(scenario, r, t)
        return np.concatenate([v, a])

    t_ascent = np.arange(0.0, meco + 1e-9, scenario.ascent_output_step_s)
    if t_ascent[-1] < meco - 1e-6:
        t_ascent = np.append(t_ascent, meco)
    sol = solve_ivp(rhs_ascent, (0.0, meco), np.concatenate([r0, v0]), t_eval=t_ascent,
                    method="DOP853", rtol=1e-10, atol=1e-5)
    if not sol.success:
        raise RuntimeError(f"3-DOF ascent integration failed: {sol.message}")
    t = np.asarray(sol.t, dtype=float)
    ascent = ModelTrack("three_dof", t,
                        np.column_stack(_to_gcrs_pair(sol.y[:6].T, t, scenario.epoch_utc())),
                        np.array([scenario.mass_at(x) for x in t]))

    # Coast from the MECO state, integrated in GCRS under the scenario's
    # coast force model (point mass by default -> the coast pairs are gateable
    # physics agreement, not integration noise). J2 needs the body-fixed
    # figure, so the position is rotated ECEF-ward by the sidereal angle that
    # carries the MECO sample's own GCRS<->ECEF bridge forward in time.
    state0 = ascent.states_gcrs[-1].copy()
    lon_ecef = _geodetic_latlon(_from_gcrs_pair(state0[np.newaxis, :3],
                                                state0[np.newaxis, 3:], t[-1:],
                                                scenario.epoch_utc())[0][0])[1]
    theta0 = math.atan2(float(state0[1]), float(state0[0])) - lon_ecef

    def rhs_coast(dt, s):
        r = s[:3]
        if scenario.coast_j2:
            th = theta0 + float(OMEGA_EARTH[2]) * float(dt)
            c, sn = math.cos(th), math.sin(th)
            a = _gravity_point_mass(np.array([c * r[0] + sn * r[1],
                                              -sn * r[0] + c * r[1], r[2]]), True)
            a = np.array([c * a[0] - sn * a[1], sn * a[0] + c * a[1], a[2]])
        else:
            a = _gravity_point_mass(r, False)
        return np.concatenate([s[3:], a])

    t_coast = np.linspace(meco, meco + scenario.coast_duration_s,
                          int(scenario.coast_output_samples))
    sol = solve_ivp(rhs_coast, (meco, meco + scenario.coast_duration_s), state0,
                    t_eval=t_coast, method="DOP853", rtol=1e-11, atol=1e-6)
    if not sol.success:
        raise RuntimeError(f"3-DOF coast integration failed: {sol.message}")
    coast = ModelTrack("three_dof", np.asarray(sol.t, dtype=float), sol.y[:6].T,
                       np.full(sol.t.size, float(ascent.mass_kg[-1])))
    return ascent, coast


def _from_gcrs_pair(r_gcrs, v_gcrs, t: np.ndarray, epoch: datetime):
    from orbitopt.flightlab.frames import from_gcrs

    return from_gcrs(r_gcrs, v_gcrs, "ECEF", epoch, t)


# ---------------------------------------------------------------------------
# 6-DOF backend (JSBSim spike stage, FL-403 PID tracking of the 3-DOF program)
# ---------------------------------------------------------------------------

def _produce_six_dof(scenario: CrossModelScenario, context: dict):
    if (abs(scenario.latitude_deg) > 1e-9 or abs(scenario.longitude_deg) > 1e-9
            or abs(scenario.altitude_m) > 1e-6):
        raise NotImplementedError(
            "the six_dof adapter flies the FL-403 spike model (equatorial-range "
            f"launch IC at (0, 0, 0)); scenario {scenario.name!r} launches at "
            f"({scenario.latitude_deg}, {scenario.longitude_deg}, {scenario.altitude_m} m)")
    three = context.get("three_dof")
    if three is None or three.ascent is None:
        raise RuntimeError("the six_dof backend flies closed-loop on the 3-DOF program; "
                           "include 'three_dof' in the backend list")
    reference = _tracking_reference(scenario, three.ascent)

    from orbitopt.flightlab.sixdof.jsbsim_driver import InitialConditions
    from orbitopt.flightlab.sixdof.tracking import AttitudeProgram, run_tracking_spike

    result = run_tracking_spike(
        reference, AttitudeProgram.from_points(list(scenario.program)),
        scenario.meco_t_s, dt_s=0.01,
        ic=InitialConditions(lat_geod_deg=scenario.latitude_deg,
                             lon_deg=scenario.longitude_deg,
                             alt_m=scenario.altitude_m,
                             pitch_deg=90.0, heading_deg=90.0, roll_deg=-90.0))
    up_to_meco = [s for s in result.sim.samples if s["t_s"] <= scenario.meco_t_s + 1e-9]
    if len(up_to_meco) < 2:
        raise RuntimeError("JSBSim produced fewer than two samples up to MECO")
    stride = max(1, int(round(0.1 / result.sim.dt_s)))  # ~10 Hz: report stays small
    samples = up_to_meco[::stride]
    last = up_to_meco[-1]
    if samples[-1] is not last:
        samples.append(last)
    t = np.array([s["t_s"] for s in samples], dtype=float)
    keep = np.concatenate(([True], np.diff(t) > 0.0))
    rows = [s for s, k in zip(samples, keep) if k]
    t = t[keep]
    ecef = np.array([[s["x_m"], s["y_m"], s["z_m"], s["vx_m_s"], s["vy_m_s"], s["vz_m_s"]]
                     for s in rows], dtype=float)
    mass = np.array([s["mass_kg"] for s in rows], dtype=float)
    r_g, v_g = _to_gcrs_pair(ecef, t, scenario.epoch_utc())
    return ModelTrack("six_dof", t, np.column_stack([r_g, v_g]), mass), None


def _tracking_reference(scenario: CrossModelScenario, ascent: ModelTrack):
    """Rebuild the FL-403 tracking loop's ``ReferenceTrajectory`` from the
    3-DOF GCRS ascent: geodetic altitude, ECEF-relative speed and flight-path
    angle come from the same states bridged back to ECEF (the OT columns the
    JSBSim comparison reads are all Earth-relative)."""
    from orbitopt.flightlab.frames import ecef_to_geodetic
    from orbitopt.flightlab.sixdof.tracking import ReferenceTrajectory

    r_e, v_e = _from_gcrs_pair(ascent.states_gcrs[:, :3], ascent.states_gcrs[:, 3:],
                               ascent.t_s, scenario.epoch_utc())
    _lat, _lon, alt = ecef_to_geodetic(r_e)
    speed = np.linalg.norm(v_e, axis=1)
    up = r_e / np.linalg.norm(r_e, axis=1)[:, None]
    fpa = np.degrees(np.arcsin(np.clip(-np.einsum("ij,ij->i", v_e, up)
                                       / np.maximum(speed, 1e-9), -1.0, 1.0)))
    return ReferenceTrajectory(
        t_s=ascent.t_s, altitude_m=np.asarray(alt, dtype=float),
        speed_earth_m_s=speed, fpa_deg=fpa,
        elevation_deg=np.array([math.degrees(scenario.attitude_at(t)[0]) for t in ascent.t_s]),
        mach=speed / 340.29, mass_kg=ascent.mass_kg.copy())


# ---------------------------------------------------------------------------
# TudatPy backend (M-05 coast machinery from the 3-DOF MECO handoff)
# ---------------------------------------------------------------------------

def _produce_tudatpy(scenario: CrossModelScenario, context: dict):
    from orbitopt.flightlab.checks import orbital_elements
    from orbitopt.mission.contract import Epoch, HandoffState
    from orbitopt.mission.propagation import (
        MU_EARTH as MU_M05, PropagationForceModel, propagate_injected_orbit)

    three = context.get("three_dof")
    if three is None or three.ascent is None:
        raise RuntimeError("the tudatpy backend coasts the 3-DOF MECO state; "
                           "include 'three_dof' in the backend list")
    final = three.ascent.states_gcrs[-1]
    el0 = orbital_elements(final[:3], final[3:], MU_M05)
    if not math.isfinite(el0["period_s"]) or el0["period_s"] <= 0.0:
        raise RuntimeError("3-DOF MECO state is not on a closed orbit; the tudatpy coast "
                           "backend (M-05 machinery) propagates whole-period arcs")
    handoff = HandoffState(
        epoch=Epoch(mjd2000=scenario.epoch_mjd2000 + scenario.meco_t_s / 86400.0,
                    timescale="UTC"),
        frame="GCRS",
        r_m=tuple(float(x) for x in final[:3]),
        v_m_s=tuple(float(x) for x in final[3:]),
        mass_kg=float(three.ascent.mass_kg[-1]),
        metadata={"producer": "orbitopt.verify.cross_model"})
    prop = propagate_injected_orbit(
        handoff, PropagationForceModel(j2=bool(scenario.coast_j2)),
        revolutions=scenario.coast_duration_s / el0["period_s"])
    t = scenario.meco_t_s + np.asarray(prop.epochs_s, dtype=float)
    states = np.asarray(prop.states_gcrs, dtype=float)
    return None, ModelTrack("tudatpy", t, states,
                            np.full(t.size, float(handoff.mass_kg)))


# ---------------------------------------------------------------------------
# Analytic two-body coast backend (numerical-consistency pair, always present)
# ---------------------------------------------------------------------------

def _produce_analytic(scenario: CrossModelScenario, context: dict):
    from orbitopt.verify.benchmarks import propagate_kepler_analytic

    three = context.get("three_dof")
    if three is None or three.ascent is None:
        raise RuntimeError("the two_body_analytic backend coasts the 3-DOF MECO state; "
                           "include 'three_dof' in the backend list")
    state0 = three.ascent.states_gcrs[-1]
    t = np.linspace(scenario.meco_t_s, scenario.meco_t_s + scenario.coast_duration_s,
                    int(scenario.coast_output_samples))
    states = np.array([propagate_kepler_analytic(state0, float(dt), MU_EARTH)
                       for dt in t - scenario.meco_t_s])
    return None, ModelTrack("two_body_analytic", t, states,
                            np.full(t.size, float(three.ascent.mass_kg[-1])))


# ---------------------------------------------------------------------------
# Pairwise difference metrics
# ---------------------------------------------------------------------------

def _wrap_deg(x: float) -> float:
    return (x + 180.0) % 360.0 - 180.0


def compare_tracks(a: ModelTrack, b: ModelTrack, samples: int = 64) -> dict:
    """Difference metrics between two tracks over their shared time window:
    cubic-Hermite-resampled position/velocity (and linear-interpolated mass)
    differences -- max / RMS along the arc and at the shared end epoch --
    plus the classical-element difference at the shared end epoch. Raw
    numbers only; thresholds and gating are applied by the report builder."""
    from orbitopt.flightlab.checks import orbital_elements
    from orbitopt.verify.events import HermiteDenseOutput

    lo, hi = max(a.t_start, b.t_start), min(a.t_end, b.t_end)
    if hi <= lo:
        raise ValueError(f"tracks {a.label}/{b.label} share no time window "
                         f"([{a.t_start}, {a.t_end}] vs [{b.t_start}, {b.t_end}])")
    t = np.linspace(lo, hi, int(samples))
    da, db = HermiteDenseOutput(a.t_s, a.states_gcrs), HermiteDenseOutput(b.t_s, b.states_gcrs)
    sa = np.array([da(x) for x in t])
    sb = np.array([db(x) for x in t])
    dr = np.linalg.norm(sa[:, :3] - sb[:, :3], axis=1)
    dv = np.linalg.norm(sa[:, 3:] - sb[:, 3:], axis=1)
    out = {
        "window_s": [float(lo), float(hi)],
        "max_position_m": float(dr.max()), "rms_position_m": float(np.sqrt((dr ** 2).mean())),
        "final_position_m": float(dr[-1]),
        "max_velocity_m_s": float(dv.max()), "rms_velocity_m_s": float(np.sqrt((dv ** 2).mean())),
        "final_velocity_m_s": float(dv[-1]),
    }
    if a.mass_kg is not None and b.mass_kg is not None:
        dm = np.abs(np.interp(t, a.t_s, a.mass_kg) - np.interp(t, b.t_s, b.mass_kg))
        out["max_mass_kg"] = float(dm.max())
        out["final_mass_kg"] = float(dm[-1])
    el_a = orbital_elements(sa[-1, :3], sa[-1, 3:], MU_EARTH)
    el_b = orbital_elements(sb[-1, :3], sb[-1, 3:], MU_EARTH)
    out["elements"] = {
        "semi_major_axis_m": el_a["semi_major_axis_m"] - el_b["semi_major_axis_m"],
        "eccentricity": el_a["eccentricity"] - el_b["eccentricity"],
        "inclination_deg": el_a["inclination_deg"] - el_b["inclination_deg"],
        "raan_deg": _wrap_deg(el_a["raan_deg"] - el_b["raan_deg"]),
        "argument_of_perigee_deg": _wrap_deg(el_a["argument_of_perigee_deg"]
                                             - el_b["argument_of_perigee_deg"]),
    }
    out["final_inclination_deg"] = [float(el_a["inclination_deg"]), float(el_b["inclination_deg"])]
    out["final_eccentricity"] = [float(el_a["eccentricity"]), float(el_b["eccentricity"])]
    return out


# ---------------------------------------------------------------------------
# Report / gate wiring
# ---------------------------------------------------------------------------

def _pair_summary(pair: str, phase: str, metrics: Mapping,
                  thresholds: CrossModelThresholds) -> dict:
    """Split raw difference metrics into gated checks against thresholds."""
    gated = {
        "position_m": metrics["final_position_m"],
        "velocity_m_s": metrics["final_velocity_m_s"],
        "mass_kg": metrics.get("final_mass_kg"),
        "semi_major_axis_m": metrics["elements"]["semi_major_axis_m"],
        "eccentricity": metrics["elements"]["eccentricity"],
        "inclination_deg": metrics["elements"]["inclination_deg"],
        "raan_deg": metrics["elements"]["raan_deg"],
        "argument_of_perigee_deg": metrics["elements"]["argument_of_perigee_deg"],
    }
    incl_a, incl_b = metrics["final_inclination_deg"]
    ecc_a, ecc_b = metrics.get("final_eccentricity", [0.0, 0.0])
    planar = min(incl_a, incl_b) > MIN_INCLINATION_FOR_ANGLES_DEG
    # node line needs inclination; the perigee direction also needs a real
    # ellipse -- near-degenerate angles are recorded, never gated (task #243:
    # no pass without meaningful evidence, and no fail on numerical noise).
    angles_ok = {"raan_deg": planar,
                 "argument_of_perigee_deg": planar
                 and min(ecc_a, ecc_b) > MIN_ECCENTRICITY_FOR_ANGLES}
    limits, checks, recorded = {}, {}, {}
    for metric, value in gated.items():
        limit = thresholds.resolve(pair, phase, metric)
        if value is None or limit is None \
                or (metric in GATED_ELEMENTS_CONDITIONAL and not angles_ok[metric]):
            recorded[metric] = None if value is None else float(abs(value))
            continue
        limits[metric] = float(limit)
        checks[metric] = bool(abs(float(value)) <= float(limit))
    return {"limits": limits, "checks": checks, "recorded_only": recorded}


def _residual_from_metrics(pair: str, phase: str, metrics: Mapping, summary: Mapping,
                           category: str) -> dict:
    residual = {
        "name": f"{pair}.{phase}",
        "reference": f"cross-model ({pair.replace('_vs_', ' vs ')}, {phase} phase)",
        "category": category,
        "position_residual_m": abs(metrics["final_position_m"]),
        "velocity_residual_m_s": abs(metrics["final_velocity_m_s"]),
        "max_position_m": metrics["max_position_m"],
        "rms_position_m": metrics["rms_position_m"],
        "max_velocity_m_s": metrics["max_velocity_m_s"],
        "rms_velocity_m_s": metrics["rms_velocity_m_s"],
        "elements_residual": metrics["elements"],
        "window_s": metrics["window_s"],
        "tolerance": dict(summary["limits"]),
        "within_tolerance": bool(summary["checks"]) and all(summary["checks"].values()),
    }
    if "max_mass_kg" in metrics:
        residual["mass_residual_kg"] = abs(metrics["final_mass_kg"])
        residual["max_mass_kg"] = metrics["max_mass_kg"]
    residual["elements_within_tolerance"] = {
        m: bool(c) for m, c in summary["checks"].items()
        if m in GATED_ELEMENTS_ALWAYS + GATED_ELEMENTS_CONDITIONAL}
    return residual


@dataclass
class CrossModelVerification:
    """One harness run: the outcomes, the pairwise metrics, the thresholds
    that graded them and the digested verification report."""

    scenario: CrossModelScenario
    thresholds: CrossModelThresholds
    backends: tuple[BackendOutcome, ...]
    pair_metrics: tuple[Mapping, ...]
    report: dict
    report_path: object = None

    @property
    def result(self) -> str:
        return status_result(self.report["status"])

    def to_dict(self) -> dict:
        return {"scenario": self.scenario.to_dict(),
                "thresholds": self.thresholds.to_dict(),
                "backends": [b.to_dict() for b in self.backends],
                "pairs": [dict(p) for p in self.pair_metrics],
                "status": self.report["status"],
                "report_file": str(self.report_path) if self.report_path else None}


def run_cross_model(scenario: CrossModelScenario | None = None,
                    backends: Sequence[str] | None = None,
                    thresholds: "CrossModelThresholds | Mapping | None" = None,
                    report_path=None) -> CrossModelVerification:
    """Fly ``scenario`` on every requested backend (default: all built-ins),
    compute the pairwise difference metrics and build the ``cross-model``
    verification report. A backend that is unavailable or raises never
    aborts the run -- it is recorded in ``report["inputs"]["backends"]``.
    The report honours ``$ORBITOPT_VERIFICATION_REPORT_DIR`` like every
    other verify entry point (and ``report_path`` overrides it)."""
    import os

    scenario = scenario or CrossModelScenario()
    thresholds = CrossModelThresholds.coerce(thresholds)
    keys = list(DEFAULT_BACKENDS) if backends is None else list(dict.fromkeys(backends))
    unknown = [k for k in keys if k not in BACKENDS]
    if unknown:
        raise ValueError(f"unknown backends {unknown}; have {sorted(BACKENDS)}")

    # three_dof first: its MECO handoff feeds every coast backend.
    order = (["three_dof"] if "three_dof" in keys else []) + [k for k in keys if k != "three_dof"]
    context: dict[str, BackendOutcome] = {}
    outcomes: list[BackendOutcome] = []
    for key in order:
        producer, available = BACKENDS[key]
        if not available():
            outcomes.append(BackendOutcome(key, STATUS_UNAVAILABLE,
                                          detail=f"{key} is not installed in this environment"))
            continue
        try:
            ascent, coast = producer(scenario, context)
        except Exception as exc:  # noqa: BLE001 -- a backend failure is data, not a crash
            outcomes.append(BackendOutcome(key, STATUS_FAILED,
                                          detail=f"{type(exc).__name__}: {exc}"))
            continue
        outcome = BackendOutcome(key, STATUS_RAN, ascent=ascent, coast=coast)
        context[key] = outcome
        outcomes.append(outcome)
    outcomes.sort(key=lambda o: keys.index(o.label))

    ran = {o.label: o for o in outcomes if o.status == STATUS_RAN}
    pairs: list[dict] = []
    residuals: list[dict] = []
    for phase, attr in (("ascent", "ascent"), ("coast", "coast")):
        with_track = [lbl for lbl in sorted(ran) if getattr(ran[lbl], attr) is not None]
        for i, la in enumerate(with_track):
            for lb in with_track[i + 1:]:
                pair = f"{la}_vs_{lb}"
                category = (CHECK_NUMERICAL if {la, lb} == {"three_dof", "two_body_analytic"}
                            else CHECK_MODEL)
                metrics = compare_tracks(getattr(ran[la], attr), getattr(ran[lb], attr))
                summary = _pair_summary(pair, phase, metrics, thresholds)
                pairs.append({"pair": pair, "phase": phase, "category": category,
                              "metrics": dict(metrics), "tolerance": summary["limits"],
                              "checks": summary["checks"],
                              "recorded_only": summary["recorded_only"]})
                residuals.append(_residual_from_metrics(pair, phase, metrics, summary, category))

    report = build_report(
        "cross-model", scenario.name, propagations=[], residuals=residuals,
        inputs={"scenario": scenario.to_dict(), "thresholds": thresholds.to_dict(),
                "backends": [o.to_dict() for o in outcomes], "pairs": pairs})
    report = finalize_report(report, report_path)
    path = report_path
    if path is None and os.environ.get(REPORT_DIR_ENV):
        path = default_report_path(report, os.environ[REPORT_DIR_ENV])
    return CrossModelVerification(scenario=scenario, thresholds=thresholds,
                                  backends=tuple(outcomes), pair_metrics=tuple(pairs),
                                  report=report, report_path=path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    """Run the cross-model harness; print a per-backend and per-pair summary;
    exit 1 on a failed gate check (``not_checked`` exits 0 with a loud line:
    absence of evidence is not a failure *or* a pass)."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m orbitopt.verify.cross_model",
        description="V-02: fly one scenario on every available backend (3-DOF point-mass, "
                    "6-DOF JSBSim, TudatPy, analytic two-body) and grade the pairwise "
                    "position/velocity/mass/orbital-element differences into a "
                    "verification report.")
    parser.add_argument("--backends", default=",".join(DEFAULT_BACKENDS),
                        help=f"comma list from {sorted(BACKENDS)} (default: all)")
    parser.add_argument("--threshold", action="append", default=[], metavar="KEY=VALUE",
                        help="override one gate threshold: KEY is <metric>, <phase>.<metric> "
                             "or <pair>.<phase>.<metric>; VALUE 'none' records without gating")
    parser.add_argument("--coast-duration", type=float, default=None, metavar="S",
                        help="override the scenario coast duration")
    parser.add_argument("-o", "--output", default=None, metavar="PATH",
                        help="report path (default: $ORBITOPT_VERIFICATION_REPORT_DIR)")
    parser.add_argument("--json", action="store_true",
                        help="print the verification report document (schema- and "
                             "digest-validated) instead of the text summary")
    args = parser.parse_args(argv)

    overrides = {}
    for item in args.threshold:
        key, _, value = item.partition("=")
        overrides[key] = value
    scenario = CrossModelScenario()
    if args.coast_duration is not None:
        scenario = replace(scenario, coast_duration_s=float(args.coast_duration))
    res = run_cross_model(scenario, backends=args.backends.split(","),
                          thresholds=overrides, report_path=args.output)
    if args.json:
        print(json.dumps(res.report, indent=2, sort_keys=True))
        return 0 if res.result != "failed" else 1
    for o in res.backends:
        line = f"[{o.status:>11}] {o.label}"
        if o.detail:
            line += f" -- {o.detail}"
        print(line)
    for p in res.pair_metrics:
        n_ok = sum(p["checks"].values())
        verdict = "PASS" if n_ok == len(p["checks"]) else ("RECORD" if not p["checks"] else "FAIL")
        print(f"[{verdict:6}] {p['pair']} ({p['phase']}): "
              f"dpos {p['metrics']['final_position_m']:.3g} m (max {p['metrics']['max_position_m']:.3g}), "
              f"dvel {p['metrics']['final_velocity_m_s']:.3g} m/s, "
              f"da {p['metrics']['elements']['semi_major_axis_m']:.3g} m"
              + (f" [{n_ok}/{len(p['checks'])} checks]" if p["checks"] else " [recorded, not gated]"))
    print(f"verification {format_status(res.report['status'])}"
          + (f" -> {res.report_path}" if res.report_path else ""))
    return 1 if res.result == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
