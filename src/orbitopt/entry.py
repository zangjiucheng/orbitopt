"""Atmospheric entry of a lifting vehicle: a 3-DOF point-mass model.

Integrated in the Earth-centred inertial frame (J2000 equator) with

* gravity: point mass + J2,
* an atmosphere that co-rotates with the Earth: Vallado's piecewise
  exponential density model (*Fundamentals of Astrodynamics and
  Applications*, 4th ed., Table 8-4: US Standard Atmosphere 1976 below
  ~1000 km, CIRA-72 above), height measured above the WGS-84 ellipsoid,
* drag along the air-relative velocity and lift perpendicular to it, rolled
  about it by a bank angle (0 = lift straight up, + = toward the right of
  the direction of flight).

Point mass means no attitude dynamics and no heating model: constant
aerodynamic coefficients stand for a trimmed attitude. Good for "where does
a vehicle with this ballistic coefficient and L/D come down, and how hard
does it decelerate" -- not for loads or thermal design.

numpy + scipy only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

MU_EARTH_M3_S2 = 3.986004418e14
R_EARTH_M = 6378137.0
J2_EARTH = 1.08262668e-3
OMEGA_EARTH_RAD_S = 7.2921150e-5
G0_M_S2 = 9.80665
_WGS84_F = 1.0 / 298.257223563
ENTRY_INTERFACE_M = 120_000.0

# Vallado Table 8-4: base height [km], base density [kg/m^3], scale height [km].
_ATMOSPHERE = np.array([
    (0.0, 1.225, 7.249), (25.0, 3.899e-2, 6.349), (30.0, 1.774e-2, 6.682), (40.0, 3.972e-3, 7.554),
    (50.0, 1.057e-3, 8.382), (60.0, 3.206e-4, 7.714), (70.0, 8.770e-5, 6.549), (80.0, 1.905e-5, 5.799),
    (90.0, 3.396e-6, 5.382), (100.0, 5.297e-7, 5.877), (110.0, 9.661e-8, 7.263), (120.0, 2.438e-8, 9.473),
    (130.0, 8.484e-9, 12.636), (140.0, 3.845e-9, 16.149), (150.0, 2.070e-9, 22.523), (180.0, 5.464e-10, 29.740),
    (200.0, 2.789e-10, 37.105), (250.0, 7.248e-11, 45.546), (300.0, 2.418e-11, 53.628), (350.0, 9.518e-12, 53.298),
    (400.0, 3.725e-12, 58.515), (450.0, 1.585e-12, 60.828), (500.0, 6.967e-13, 63.822), (600.0, 1.454e-13, 71.835),
    (700.0, 3.614e-14, 88.667), (800.0, 1.170e-14, 124.64), (900.0, 5.245e-15, 181.05), (1000.0, 3.019e-15, 268.00),
])


def density_kg_m3(h_m: float) -> float:
    """Atmospheric density at height ``h_m`` above the ellipsoid (0 below
    ground is clamped to sea level)."""
    h_km = max(h_m, 0.0) / 1000.0
    i = int(np.searchsorted(_ATMOSPHERE[:, 0], h_km, side="right")) - 1
    h0, rho0, scale = _ATMOSPHERE[max(i, 0)]
    return float(rho0 * math.exp(-(h_km - h0) / scale))


def height_m(r: np.ndarray) -> float:
    """Height above the WGS-84 ellipsoid, to first order in the flattening
    (metres-level near the surface -- ample for an atmosphere lookup)."""
    rn = float(np.linalg.norm(r))
    sin_lat = r[2] / rn
    return rn - R_EARTH_M * (1.0 - _WGS84_F * sin_lat * sin_lat)


@dataclass(frozen=True)
class EntryVehicle:
    mass_kg: float
    area_m2: float
    cd: float
    cl: float

    @property
    def ballistic_coefficient(self) -> float:
        return self.mass_kg / (self.cd * self.area_m2)

    @property
    def lift_to_drag(self) -> float:
        return self.cl / self.cd


def flat_plate_coefficients(alpha_deg: float) -> tuple[float, float]:
    """(CD, CL) of a flat plate at angle of attack ``alpha`` in hypersonic
    Newtonian flow: CN = 2 sin^2(alpha), CD = CN sin(alpha), CL = CN cos(alpha)
    (L/D = cot(alpha)). A first estimate for a body flying belly-first."""
    a = math.radians(alpha_deg)
    cn = 2.0 * math.sin(a) ** 2
    return cn * math.sin(a), cn * math.cos(a)


def _gravity(r: np.ndarray) -> np.ndarray:
    x, y, z = r
    rn2 = x * x + y * y + z * z
    rn = math.sqrt(rn2)
    k = 1.5 * J2_EARTH * (R_EARTH_M * R_EARTH_M) / rn2
    zz = 5.0 * z * z / rn2
    f = -MU_EARTH_M3_S2 / (rn2 * rn)
    return np.array([f * x * (1.0 + k * (1.0 - zz)), f * y * (1.0 + k * (1.0 - zz)), f * z * (1.0 + k * (3.0 - zz))])


def _accelerations(r: np.ndarray, v: np.ndarray, vehicle: EntryVehicle, bank_rad: float):
    """(total acceleration, aerodynamic acceleration) at state (r, v)."""
    g = _gravity(r)
    omega = np.array([0.0, 0.0, OMEGA_EARTH_RAD_S])
    v_air = v - np.cross(omega, r)
    speed = float(np.linalg.norm(v_air))
    rho = density_kg_m3(height_m(r))
    if rho < 1e-14 or speed < 1e-6:
        return g, np.zeros(3)
    q = 0.5 * rho * speed * speed
    v_hat = v_air / speed
    up = r / np.linalg.norm(r)
    n1 = up - np.dot(up, v_hat) * v_hat           # "lift up": in the vertical plane, normal to v
    n1_norm = np.linalg.norm(n1)
    n1 = n1 / n1_norm if n1_norm > 1e-12 else np.cross(v_hat, [0.0, 0.0, 1.0])
    right = np.cross(v_hat, n1)                    # horizontal, to the right of the direction of flight
    lift_hat = math.cos(bank_rad) * n1 + math.sin(bank_rad) * right
    aero = (q * vehicle.area_m2 / vehicle.mass_kg) * (-vehicle.cd * v_hat + vehicle.cl * lift_hat)
    return g + aero, aero


@dataclass
class EntryTrajectory:
    times_s: np.ndarray        # from the start of the integration
    states: np.ndarray         # (N, 6) m, m/s, inertial
    heights_m: np.ndarray
    aero_g: np.ndarray         # aerodynamic deceleration, in g
    dynamic_pressure_pa: np.ndarray
    ended_at_height: bool      # reached the stop height (vs. ran out of time)

    def first_crossing_s(self, height: float) -> float | None:
        """First time the trajectory descends through ``height`` [m]
        (linearly interpolated between samples)."""
        h = self.heights_m
        idx = np.flatnonzero((h[:-1] >= height) & (h[1:] < height))
        if idx.size == 0:
            return None
        i = int(idx[0])
        f = (h[i] - height) / (h[i] - h[i + 1])
        return float(self.times_s[i] + f * (self.times_s[i + 1] - self.times_s[i]))


def fly_entry(state0, vehicle: EntryVehicle, bank_deg: float, stop_height_m: float = 1000.0,
              max_time_s: float = 7200.0, sample_s: float = 2.0, rtol: float = 1e-9, atol: float = 1e-6) -> EntryTrajectory:
    """Integrate from ``state0`` (inertial, m and m/s) with a constant bank
    angle until the vehicle descends to ``stop_height_m``."""
    bank = math.radians(bank_deg)

    def rhs(_t, y):
        a, _aero = _accelerations(y[:3], y[3:], vehicle, bank)
        return np.concatenate([y[3:], a])

    def reached(_t, y):
        return height_m(y[:3]) - stop_height_m
    reached.terminal = True
    reached.direction = -1

    sol = solve_ivp(rhs, (0.0, max_time_s), np.asarray(state0, dtype=float), method="DOP853",
                    rtol=rtol, atol=atol, events=reached, dense_output=True, max_step=60.0)
    t_end = float(sol.t[-1])
    times = np.unique(np.concatenate([np.arange(0.0, t_end, sample_s), [t_end]]))
    states = sol.sol(times).T
    heights = np.array([height_m(s[:3]) for s in states])
    aero_g, qbar = [], []
    for s in states:
        _a, aero = _accelerations(s[:3], s[3:], vehicle, bank)
        aero_g.append(np.linalg.norm(aero) / G0_M_S2)
        v_air = s[3:] - np.cross([0.0, 0.0, OMEGA_EARTH_RAD_S], s[:3])
        qbar.append(0.5 * density_kg_m3(height_m(s[:3])) * float(v_air @ v_air))
    return EntryTrajectory(times_s=times, states=states, heights_m=heights, aero_g=np.array(aero_g),
                           dynamic_pressure_pa=np.array(qbar), ended_at_height=bool(sol.t_events[0].size))
