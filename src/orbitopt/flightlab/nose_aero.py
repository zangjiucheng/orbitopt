"""Nose-shape aerodynamics: axial-force tables from a declared nose and body.

A vehicle file's ``geometry.nose`` declares the nose (shape, length) on the
stack's top stage; this module turns it, with the body it sits on, into the
three numbers OpenTsiolkovsky's ``aero`` block takes: ``ca`` (axial force
coefficient while the engines burn), ``ca_power_off`` (the same with the
base drag of a coasting stage) and ``cn_alpha_per_rad``. The tables in a
vehicle YAML are *generated* from this (``python -m orbitopt.flightlab.nose_aero
VEHICLE``) and a test holds the committed numbers equal to the generated
ones, so the aero cannot drift from the geometry it is declared with.

The model is the textbook preliminary-design one, **an estimate good to
roughly +-30 %**, not CFD:

* nose wave drag -- the *tangent-cone* method: the pressure at each point of
  the nose is that of a cone with the local surface inclination, from the
  exact Taylor-Maccoll solution; where the cone's shock would detach (a blunt
  tip, low supersonic Mach) the modified-Newtonian pressure takes over. No
  wave drag below Mach 0.9, a smooth rise to Mach 1.3 (the transonic range is
  an interpolation, not a prediction);
* skin friction -- compressible turbulent flat plate (Prandtl-Schlichting) on
  the wetted nose and body at a fixed Reynolds number;
* base drag -- Missile DATCOM's power-off correlation (0.12 + 0.13 M^2 below
  Mach 1, 0.25 / M above); with the engines burning the jet fills the base and
  none is charged.

Not modelled: protuberances (grid fins, flaps, legs), angle-of-attack drag,
real-gas and heating effects. Forces are referred to the body's cross
section, the reference area of the OT ``aero`` block.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
from scipy.integrate import solve_ivp

__all__ = [
    "DEFAULT_MACH",
    "GAMMA",
    "NOSE_SHAPES",
    "NoseSpec",
    "base_drag_ca",
    "ca_table",
    "cone_pressure_coefficient",
    "normal_force_slope_per_rad",
    "skin_friction_ca",
    "wave_drag_ca",
    "wetted_area_m2",
]

GAMMA = 1.4
NOSE_SHAPES = ("cone", "tangent_ogive", "haack")
#: Mach grid of the generated tables
DEFAULT_MACH = (0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
#: Reynolds number (body length) for the skin-friction estimate; a launcher's
#: ascent spends most of its drag-relevant time between 1e7 and 1e9
DEFAULT_REYNOLDS = 3.0e8
_TRANSONIC = (0.9, 1.3)


@dataclass(frozen=True)
class NoseSpec:
    """A nose of revolution: ``shape`` (``cone``, ``tangent_ogive`` or the
    minimum-wave-drag ``haack`` Von Karman body), its ``length_m`` and the
    ``base_diameter_m`` it meets the body at."""

    shape: str
    length_m: float
    base_diameter_m: float

    def __post_init__(self) -> None:
        if self.shape not in NOSE_SHAPES:
            raise ValueError(f"nose shape must be one of {NOSE_SHAPES}, got {self.shape!r}")
        if self.length_m <= 0.0 or self.base_diameter_m <= 0.0:
            raise ValueError("nose length and base diameter must be positive")

    @property
    def fineness(self) -> float:
        """Length over base diameter."""
        return self.length_m / self.base_diameter_m

    def profile(self, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
        """``(x, r)`` from the tip (x = 0) to the base (x = length), metres."""
        length, radius = self.length_m, 0.5 * self.base_diameter_m
        x = np.linspace(0.0, length, n + 1)
        if self.shape == "cone":
            r = radius * x / length
        elif self.shape == "tangent_ogive":
            rho = (radius**2 + length**2) / (2.0 * radius)
            r = np.sqrt(np.maximum(rho**2 - (length - x) ** 2, 0.0)) + radius - rho
        else:  # haack (Von Karman)
            theta = np.arccos(1.0 - 2.0 * x / length)
            r = radius / math.sqrt(math.pi) * np.sqrt(theta - 0.5 * np.sin(2.0 * theta))
        return x, np.maximum(r, 0.0)


# --------------------------------------------------------------------------
# pressure on a cone: Taylor-Maccoll
# --------------------------------------------------------------------------

def _pitot_cp(mach: float) -> float:
    """Stagnation-point pressure coefficient behind a normal shock (Rayleigh)."""
    g = GAMMA
    m2 = mach * mach
    ratio = (((g + 1) ** 2 * m2 / (4 * g * m2 - 2 * (g - 1))) ** (g / (g - 1))
             * (1 - g + 2 * g * m2) / (g + 1))
    return 2.0 / (g * m2) * (ratio - 1.0)


def _tm_rhs(theta: float, y: np.ndarray) -> list[float]:
    g = GAMMA
    vr, vt = y
    a2 = 0.5 * (g - 1.0) * (1.0 - vr * vr - vt * vt)
    return [vt, (vt * vt * vr - a2 * (2.0 * vr + vt / math.tan(theta))) / (a2 - vt * vt)]


def _cone_surface(mach: float, beta: float) -> tuple[float, float] | None:
    """For a shock of angle ``beta``: the cone half-angle it carries and the
    surface pressure coefficient, by integrating Taylor-Maccoll from the shock
    to the surface (the ray where the cross-flow velocity vanishes)."""
    g = GAMMA
    m1n2 = (mach * math.sin(beta)) ** 2
    if m1n2 <= 1.0:
        return None
    tan_delta = 2.0 / math.tan(beta) * (m1n2 - 1.0) / (mach * mach * (g + math.cos(2.0 * beta)) + 2.0)
    delta = math.atan(tan_delta)                              # flow turning across the shock
    m2n2 = (1.0 + 0.5 * (g - 1.0) * m1n2) / (g * m1n2 - 0.5 * (g - 1.0))
    m2 = math.sqrt(m2n2) / math.sin(beta - delta)
    v = math.sqrt(0.5 * (g - 1.0) * m2 * m2 / (1.0 + 0.5 * (g - 1.0) * m2 * m2))   # V / Vmax
    y0 = [v * math.cos(beta - delta), -v * math.sin(beta - delta)]

    def surface(_theta, y):  # noqa: ARG001
        return y[1]

    surface.terminal = True  # type: ignore[attr-defined]
    surface.direction = 1.0  # type: ignore[attr-defined]
    sol = solve_ivp(_tm_rhs, (beta, 1e-4), y0, events=surface, rtol=1e-9, atol=1e-11,
                    max_step=0.01)
    if not len(sol.t_events[0]):
        return None
    theta_c = float(sol.t_events[0][0])
    vr, vt = (float(q) for q in sol.y_events[0][0])
    vc2 = vr * vr + vt * vt
    mc2 = 2.0 / (g - 1.0) * vc2 / (1.0 - vc2)
    p2_p1 = 1.0 + 2.0 * g / (g + 1.0) * (m1n2 - 1.0)
    pc_p2 = ((1.0 + 0.5 * (g - 1.0) * m2 * m2) / (1.0 + 0.5 * (g - 1.0) * mc2)) ** (g / (g - 1.0))
    return theta_c, 2.0 * (p2_p1 * pc_p2 - 1.0) / (g * mach * mach)


@lru_cache(maxsize=256)
def _cone_curve(mach: float) -> tuple[np.ndarray, np.ndarray]:
    """``(half-angle [rad], Cp)`` samples of the attached-shock cone family at
    one Mach number, from a vanishing cone up to the largest attached one."""
    beta_lo = math.asin(1.0 / mach) + 1e-6
    betas = np.linspace(beta_lo, 0.5 * math.pi - 1e-3, 90)
    pts = [(0.0, 0.0)]
    for beta in betas:
        got = _cone_surface(mach, float(beta))
        if got is None:
            continue
        theta, cp = got
        if theta <= pts[-1][0] or not math.isfinite(cp):
            break
        pts.append((theta, cp))
    arr = np.asarray(pts)
    return arr[:, 0], arr[:, 1]


def cone_pressure_coefficient(mach: float, half_angle_rad: float) -> float | None:
    """Surface pressure coefficient of a sharp cone of this half-angle at
    zero incidence (Taylor-Maccoll); ``None`` where the shock detaches."""
    if mach <= 1.0:
        return None
    theta, cp = _cone_curve(round(float(mach), 6))
    if half_angle_rad > theta[-1]:
        return None
    return float(np.interp(half_angle_rad, theta, cp))


# --------------------------------------------------------------------------
# the three drag components
# --------------------------------------------------------------------------

def wave_drag_ca(nose: NoseSpec, mach: float) -> float:
    """Nose pressure drag over the body cross section (see the module docstring
    for the method and its transonic interpolation)."""
    lo, hi = _TRANSONIC
    if mach <= lo:
        return 0.0
    m_eval = max(mach, hi)
    x, r = nose.profile()
    dr = np.diff(r)
    dx = np.diff(x)
    delta = np.arctan2(dr, dx)
    r_mid = 0.5 * (r[1:] + r[:-1])
    cp0 = _pitot_cp(m_eval)
    cp = np.empty_like(delta)
    for i, d in enumerate(delta):
        attached = cone_pressure_coefficient(m_eval, float(d))
        cp[i] = attached if attached is not None else cp0 * math.sin(d) ** 2
    radius = 0.5 * nose.base_diameter_m
    ca = float(2.0 / radius**2 * np.sum(cp * r_mid * dr))
    if mach < hi:                                   # smooth rise across the transonic range
        s = (mach - lo) / (hi - lo)
        ca *= s * s * (3.0 - 2.0 * s)
    return ca


def wetted_area_m2(nose: NoseSpec, body_length_m: float) -> float:
    """Lateral area of the nose plus the cylinder behind it."""
    x, r = nose.profile()
    dr, dx = np.diff(r), np.diff(x)
    nose_area = float(np.sum(2.0 * math.pi * 0.5 * (r[1:] + r[:-1]) * np.hypot(dr, dx)))
    return nose_area + math.pi * nose.base_diameter_m * max(body_length_m, 0.0)


def skin_friction_ca(mach: float, wetted_m2: float, ref_area_m2: float,
                     reynolds: float = DEFAULT_REYNOLDS) -> float:
    """Turbulent flat-plate skin friction, compressibility-corrected, over
    the reference area."""
    cf = 0.455 / math.log10(reynolds) ** 2.58 / (1.0 + 0.144 * mach * mach) ** 0.65
    return cf * wetted_m2 / ref_area_m2


def base_drag_ca(mach: float) -> float:
    """Power-off base drag over the base (= reference) area, Missile DATCOM."""
    return 0.12 + 0.13 * mach * mach if mach < 1.0 else 0.25 / mach


def normal_force_slope_per_rad() -> float:
    """CN-alpha of a slender nose-and-cylinder body, per radian, over the base
    area: slender-body theory gives 2 for the nose and nothing for the
    cylinder behind it."""
    return 2.0


def ca_table(nose: NoseSpec, body_length_m: float, *, power_on: bool,
             mach: Sequence[float] = DEFAULT_MACH, reynolds: float = DEFAULT_REYNOLDS,
             decimals: int = 4) -> list[list[float]]:
    """``[[mach, ca], ...]`` for a stage whose nose is ``nose`` and whose body
    behind it is ``body_length_m`` long -- the ``ca`` (``power_on``) or
    ``ca_power_off`` curve of the vehicle's OT ``aero`` block."""
    ref = 0.25 * math.pi * nose.base_diameter_m**2
    wetted = wetted_area_m2(nose, body_length_m)
    rows = []
    for m in mach:
        ca = wave_drag_ca(nose, m) + skin_friction_ca(m, wetted, ref, reynolds)
        if not power_on:
            ca += base_drag_ca(m)
        rows.append([float(m), round(ca, decimals)])
    return rows


def main(argv: list[str] | None = None) -> int:
    """Print the generated ``aero`` curves of a vehicle file with a
    ``geometry.nose`` block (for pasting into the YAML)."""
    import argparse

    from orbitopt.flightlab import vehicle_catalog as vc

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("vehicle", help="catalog name or path of a vehicle YAML")
    args = parser.parse_args(argv)
    doc = vc.load_vehicle_file(args.vehicle)
    for k, stage in enumerate(stage_nose_aero(doc), start=1):
        print(f"stage {k}: body {stage['body_length_m']:.1f} m behind a "
              f"{stage['nose'].shape} nose, L/D {stage['nose'].fineness:.2f}")
        print("  ca:", stage["ca"])
        print("  ca_power_off:", stage["ca_power_off"])
        print("  cn_alpha_per_rad:", stage["cn_alpha_per_rad"])
    return 0


def stage_nose_aero(doc) -> list[dict]:
    """The generated aero curves of each stage of a vehicle document that
    declares ``geometry.nose``: every stage carries the nose of the stack's top
    stage, with the body from its own bottom up to that nose base. ``[]`` when
    the document declares no nose."""
    geo = doc.get("geometry") or {}
    nose_doc = geo.get("nose")
    stages = geo.get("stages") or []
    if not nose_doc or not stages:
        return []
    diameter = float((doc["stages"][-1].get("aero") or {}).get("diameter_m", 0.0))
    nose = NoseSpec(str(nose_doc["shape"]), float(nose_doc["length_m"]),
                    float(nose_doc.get("base_diameter_m") or diameter))
    out = []
    for k in range(len(stages)):
        above = sum(float(s["length_m"]) for s in stages[k:])
        body = above - nose.length_m
        out.append({"nose": nose, "body_length_m": body,
                    "ca": ca_table(nose, body, power_on=True),
                    "ca_power_off": ca_table(nose, body, power_on=False),
                    "cn_alpha_per_rad": normal_force_slope_per_rad()})
    return out


if __name__ == "__main__":
    raise SystemExit(main())
