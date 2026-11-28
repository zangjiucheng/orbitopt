"""Constraint model for constrained feedback control (E13-05).

One frozen record, :class:`ApproachConstraints`, carries the constraint
families the E13-05 research compares LQR and MPC under, stated in the
relative frame the laws work in (LVLH, metres and m/s):

* **keep-out** -- a sphere centred on the target (the LVLH origin):
  ``|rho| >= keep_out_radius_m``;
* **approach corridor** -- cross-track containment ``|z| <=
  cross_track_max_m`` plus the terminal half-cone about ``-V-bar``: inside
  ``cone_range_m`` the line of sight must stay within ``cone_half_angle_deg``
  of straight aft, the same geometry ADR rendezvous-0002 fixes for the
  mission config (90 deg reads "never in front of the target");
* **velocity profile** -- a range-dependent relative-speed ceiling,
  piecewise-linear through knots and constant outside them (the mission
  config's approach-speed family, restated for the prediction model);
* **thrust / delta-v limits** -- a per-pulse magnitude cap (the actuator
  envelope at the control cadence) and the law-side cumulative delta-v
  budget (the law's own ledger, exactly as :class:`CwMidCourseLaw` keeps
  one: commanded, never realized).

The values are the same families :mod:`orbitopt.rendezvous.safety` grades,
declared independently here so the guidance layer stays free of campaign
imports; :meth:`ApproachConstraints.from_safety_values` builds them from a
mission config's safety block, and :meth:`margins` reports signed margins
(negative = violated) on arbitrary ``(rho, rho_dot)`` samples -- the same
margin convention the safety module uses, so predicted and realized counts
are directly comparable.

Like every guidance-side module this one names the planned chief motion
"reference" and carries no plant handle: constraints are DECLARATIONS the
optimizer consumes, not observations of the vehicle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .contracts import GNCContractError


class ConstraintSetError(GNCContractError):
    """A constraint declaration is malformed (inverted bounds, unsorted
    knots, swallowed hold point)."""


def _finite(name: str, value, *, positive: bool = True) -> float:
    v = float(value)
    if not math.isfinite(v) or (v <= 0.0 if positive else v < 0.0):
        raise ConstraintSetError(f"{name} must be finite and {'> 0' if positive else '>= 0'}, "
                                 f"got {value!r}")
    return v


@dataclass(frozen=True)
class VelocityProfile:
    """Range-dependent relative-speed ceiling: piecewise-linear through
    ``(range_knots_m, cap_m_s)``, constant below the first and above the last
    knot. Caps must not decrease with range (the ceiling tightens as the
    chaser closes -- the mission's approach-speed family read the same way)."""

    range_knots_m: tuple
    cap_m_s: tuple

    def __post_init__(self):
        knots = tuple(float(x) for x in self.range_knots_m)
        caps = tuple(float(x) for x in self.cap_m_s)
        if len(knots) < 2 or len(knots) != len(caps):
            raise ConstraintSetError("VelocityProfile needs >= 2 aligned knots/caps, got "
                                     f"{self.range_knots_m!r} / {self.cap_m_s!r}")
        if any(b <= a for a, b in zip(knots, knots[1:])):
            raise ConstraintSetError("VelocityProfile range_knots_m must strictly increase")
        if any(b < a for a, b in zip(caps, caps[1:])):
            raise ConstraintSetError("VelocityProfile cap_m_s must be non-decreasing with range "
                                     "(the speed ceiling tightens as the chaser closes)")
        if not all(math.isfinite(x) for x in knots + caps):
            raise ConstraintSetError("VelocityProfile knots/caps must be finite")
        object.__setattr__(self, "range_knots_m", knots)
        object.__setattr__(self, "cap_m_s", caps)

    def cap(self, ranges_m):
        """The speed ceiling at each range (np.interp: constant beyond the
        end knots, same read the safety module gives)."""
        return np.interp(np.asarray(ranges_m, dtype=np.float64),
                         np.asarray(self.range_knots_m, dtype=np.float64),
                         np.asarray(self.cap_m_s, dtype=np.float64))


@dataclass(frozen=True)
class ApproachConstraints:
    """Everything the constrained laws must respect, in LVLH. ``cone_cos_half``
    is ``cos(half-angle)`` of the terminal cone about -V-bar (0.0 = 90 deg:
    never forward of the target); the cone only binds within ``cone_range_m``.
    ``u_max_m_s`` caps each single commanded pulse; ``budget_m_s`` caps the
    law's cumulative COMMANDED delta-v including the planned departure burn."""

    keep_out_radius_m: float
    cross_track_max_m: float
    cone_cos_half: float
    cone_range_m: float
    velocity: VelocityProfile
    u_max_m_s: float
    budget_m_s: float
    hold_point_rho_norm_m: float

    def __post_init__(self):
        ko = _finite("keep_out_radius_m", self.keep_out_radius_m)
        xt = _finite("cross_track_max_m", self.cross_track_max_m)
        ch = float(self.cone_cos_half)
        if not -1.0 <= ch <= 1.0:
            raise ConstraintSetError(f"cone_cos_half must lie in [-1, 1], got {self.cone_cos_half!r}")
        cr = _finite("cone_range_m", self.cone_range_m)
        um = _finite("u_max_m_s", self.u_max_m_s)
        bu = _finite("budget_m_s", self.budget_m_s)
        hr = _finite("hold_point_rho_norm_m", self.hold_point_rho_norm_m)
        if ko >= hr:
            raise ConstraintSetError(f"keep_out_radius_m {ko:.0f} m swallows the hold point at "
                                     f"{hr:.0f} m -- no feasible approach exists")
        if not isinstance(self.velocity, VelocityProfile):
            raise ConstraintSetError(f"velocity must be a VelocityProfile, got {self.velocity!r}")
        if float(self.velocity.cap_m_s[-1]) < 0.0:
            raise ConstraintSetError("velocity caps must be >= 0")
        object.__setattr__(self, "keep_out_radius_m", ko)
        object.__setattr__(self, "cross_track_max_m", xt)
        object.__setattr__(self, "cone_cos_half", ch)
        object.__setattr__(self, "cone_range_m", cr)
        object.__setattr__(self, "u_max_m_s", um)
        object.__setattr__(self, "budget_m_s", bu)
        object.__setattr__(self, "hold_point_rho_norm_m", hr)

    @classmethod
    def from_safety_values(cls, *, keep_out_radius_m: float, cross_track_max_m: float,
                           cone_half_angle_deg: float, cone_range_m: float,
                           speed_range_knots_m, speed_cap_m_s,
                           u_max_m_s: float, budget_m_s: float,
                           hold_point_rho_norm_m: float) -> "ApproachConstraints":
        """Build the law-side declaration from a mission config's safety block
        (values, not the config object -- the gnc layer imports no campaign
        types)."""
        return cls(keep_out_radius_m=float(keep_out_radius_m),
                   cross_track_max_m=float(cross_track_max_m),
                   cone_cos_half=math.cos(math.radians(float(cone_half_angle_deg))),
                   cone_range_m=float(cone_range_m),
                   velocity=VelocityProfile(tuple(float(x) for x in speed_range_knots_m),
                                            tuple(float(x) for x in speed_cap_m_s)),
                   u_max_m_s=float(u_max_m_s), budget_m_s=float(budget_m_s),
                   hold_point_rho_norm_m=float(hold_point_rho_norm_m))

    def margins(self, rho, rho_dot) -> dict:
        """Signed margins (negative = violated) for the four trajectory
        families on aligned ``(n, 3)`` samples, in the convention of
        :func:`orbitopt.rendezvous.safety.evaluate_trajectory_safety` so the
        realized-violation counts the benchmark reports and the margins the
        optimizer predicts are the same numbers. The cone margin is ``+inf``
        outside ``cone_range_m`` (the constraint is simply not applicable)."""
        rho = np.asarray(rho, dtype=np.float64)
        rhod = np.asarray(rho_dot, dtype=np.float64)
        if rho.ndim == 1:
            rho = rho[None, :]
            rhod = rhod[None, :]
        if rho.shape != rhod.shape or rho.shape[1] != 3:
            raise ConstraintSetError(f"margins needs aligned (n, 3) samples, got "
                                     f"{rho.shape} / {rhod.shape}")
        rng = np.linalg.norm(rho, axis=1)
        out = {
            "keep_out": rng - self.keep_out_radius_m,
            "cross_track": self.cross_track_max_m - np.abs(rho[:, 2]),
            "approach_speed": self.velocity.cap(rng) - np.linalg.norm(rhod, axis=1),
        }
        cone = -rho[:, 1] - self.cone_cos_half * rng
        cone = np.where(rng <= self.cone_range_m, cone, np.inf)
        out["terminal_cone"] = cone
        return out

    def violated(self, rho, rho_dot) -> dict:
        """Violation sample COUNT per family (margin < 0), aligned with the
        safety module's violation_count."""
        return {name: int(np.count_nonzero(m < 0.0))
                for name, m in self.margins(rho, rho_dot).items()}
