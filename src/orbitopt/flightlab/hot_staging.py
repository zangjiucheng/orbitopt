"""E10-02 (task #299): the HOT-STAGING / THROTTLED-MECO timing contract.

One document, three models
--------------------------
A vehicle file with a ``staging:`` block declares a hot-staging timeline:
the stage-1 thrust ramps *continuously* down to the throttle floor before
shutdown, the ship lights while the stages are still connected, and the
booster's MECO lands at or after separation -- the cold-separation coast
between MECO and separation is gone. This module is the single, stdlib-only
authority for that timeline: it parses and validates the ``staging:`` block,
orders the events, measures the stage-1 thrust curve against the declared
slew / step limits, and builds the composite (stack) thrust curve that any
single-body ascent model flies while the stages are still mated.

Consumers: the OpenTsiolkovsky adapter (folds the timeline into the OT stage
config), the native point-mass twin (E8-05), the native rigid-body two-stage
plant (F-06), the 6-DOF staging handoff plumbing, and the E10-02 comparison
record. Each backend flies what it can represent of the timeline; the
declared simplifications live with the model adapters and are re-stated here:

* **OT / any single-body stack model**: the stack flies the composite curve
  (booster + plume-corrected ship thrust) up to separation, at which point
  the composite thrust ends -- a stage that has separated can no longer
  thrust in these models. A booster MECO *after* separation (the DC8
  timeline) is therefore represented on the stack side by "burn continues
  through separation"; the booster's post-separation burn (0.5 s at the
  floor, 2.6 t of propellant, ~1 m/s of booster-only delta-v) is charged to
  the backend difference, never silently absorbed.
* **JSBSim staging (FL-402 path)**: the stack flies its own thrust, and the
  upper child spawns *engine-live* at separation -- the ship is already at
  commanded throttle with no ignition transient and no unpowered coast,
  which is the handoff-side representation of ignition-before-separation.

Timing conventions (all absolute seconds after liftoff, like the attitude
program): ``throttle_down_start_s < ship_ignition_s < separation_s <=
booster_meco_s <= stage-1 burn_end_s``. The ship's overlap burn
(``[ship_ignition_s, separation_s]``) happens on the ship's own engine clock,
so a config keyed to the ship's ignition shifts the OT stage-2 curve by
``separation_s - ship_ignition_s`` and deducts the overlap propellant from
the stage-2 initial mass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

__all__ = ["EVENT_ORDER", "HotStaging", "HotStagingError", "ThrustCurveReport",
           "g0", "interp_thrust", "load_hot_staging"]

G0 = 9.80665

#: The four declared events, in the order the acceptance requires them.
EVENT_ORDER = ("throttle_down", "ship_ignition", "separation", "booster_meco")


class HotStagingError(ValueError):
    """The staging block (or the curve it must match) violates the contract."""


def g0() -> float:
    return G0


def interp_thrust(curve: Sequence[Sequence[float]], t: float) -> float:
    """Piecewise-linear thrust curve lookup, clamped at the ends; the first
    of two samples sharing a time wins (a commanded step is its post-step
    value from that instant on, matching OT's and the twin's convention)."""
    pts = [[float(a), float(b)] for a, b in curve]
    if not pts:
        raise HotStagingError("thrust curve is empty")
    if t <= pts[0][0]:
        return pts[0][1]
    for (t0, f0), (t1, f1) in zip(pts, pts[1:]):
        if t0 <= t < t1:
            span = (t1 - t0) or 1.0
            return f0 + (f1 - f0) * (t - t0) / span
        if t0 == t1 == t:
            return f1
    return pts[-1][1]


def _integrated_flow_kg(curve: Sequence[Sequence[float]], isp_s: float,
                        t0: float, t1: float) -> float:
    """Integral of thrust/isp/g0 over [t0, t1] for a piecewise-linear curve
    (trapezoid over the curve's own vertices inside the window, clamped)."""
    pts = sorted([(max(t0, float(a)), max(t0, min(t1, float(b)))) for a, b in curve])
    keys = sorted({t0, t1, *{float(a) for a, _ in curve if t0 <= float(a) <= t1}})
    total = 0.0
    for ta, tb in zip(keys, keys[1:]):
        total += 0.5 * (interp_thrust(curve, ta) + interp_thrust(curve, tb)) * (tb - ta)
    return total / (float(isp_s) * G0)


@dataclass(frozen=True)
class ThrustCurveReport:
    """Measurement of a stage-1 curve against the staging block's declared
    limits. ``max_step_fraction`` excludes the terminal MECO shutoff (the
    shutdown is a commanded engine stop, not a throttle transient; the
    acceptance's "no step" clause is about throttle behaviour *before*
    MECO, and the shutoff floor is separately declared)."""

    max_step_fraction: float
    max_slew_fraction_per_s: float
    throttle_down_slew_fraction_per_s: float
    floor_fraction: float
    cut_level_n: float

    def violations(self, declared_step_max: float, declared_slew_max: float,
                   declared_floor: float | None = None) -> tuple[str, ...]:
        out = []
        if self.max_step_fraction > declared_step_max + 1e-12:
            out.append(f"thrust step {self.max_step_fraction:.4f} exceeds declared "
                       f"{declared_step_max}")
        if self.max_slew_fraction_per_s > declared_slew_max + 1e-12:
            out.append(f"thrust slew {self.max_slew_fraction_per_s:.4f} /s exceeds "
                       f"declared {declared_slew_max} /s")
        if declared_floor is not None and abs(self.floor_fraction - declared_floor) > 1e-6:
            out.append(f"MECO cut level is {self.floor_fraction:.4f} of full thrust, "
                       f"declared floor {declared_floor}")
        return tuple(out)


@dataclass(frozen=True)
class HotStaging:
    """The vehicle file's ``staging:`` block, validated. All times are
    absolute seconds after liftoff."""

    throttle_down_start_s: float
    ship_ignition_s: float
    separation_s: float
    booster_meco_s: float
    #: Axial load fraction the ship plume takes off the booster while the
    #  stages are mated (simplified interference model; DESIGN ASSUMPTION --
    #  no public plume-interference figure for a mated Starship pair).
    plume_axial_loss_fraction: float = 0.04
    #: Declared throttle-slew limit over the throttle-down phase (fraction of
    #  full thrust per second). Public Raptor sources give the 40-100 % range
    #  but no slew rate; this limit is a DESIGN ASSUMPTION (see the vehicle
    #  file's staging provenance entry).
    throttle_slew_max_fraction_per_s: float = 0.05
    #: Declared maximum commanded thrust step before MECO (fraction of full
    #  thrust). A continuous ramp makes this vacuous; the number is what the
    #  acceptance's step gate measures against.
    thrust_step_max_fraction: float = 0.175
    #: The level the stage-1 curve must reach at cutoff (fraction of full):
    #  the Raptor throttle floor, so the terminal shutoff is the smallest
    #  possible one for the engine set.
    throttle_floor_fraction: float = 0.40

    # -- construction ------------------------------------------------------

    @staticmethod
    def load(doc: Mapping[str, Any]) -> "HotStaging":
        """Parse and validate a ``staging:`` mapping from a vehicle document."""
        if not isinstance(doc, Mapping):
            raise HotStagingError("staging block must be a mapping")
        for k in ("ship_ignition_s", "separation_s", "booster_meco_s"):
            if k not in doc:
                raise HotStagingError(f"staging block is missing {k!r}")
        kw = dict(
            throttle_down_start_s=float(doc.get("throttle_down_start_s",
                                                doc["ship_ignition_s"])),
            ship_ignition_s=float(doc["ship_ignition_s"]),
            separation_s=float(doc["separation_s"]),
            booster_meco_s=float(doc["booster_meco_s"]),
        )
        for opt, cast in (("plume_axial_loss_fraction", float),
                          ("throttle_slew_max_fraction_per_s", float),
                          ("thrust_step_max_fraction", float),
                          ("throttle_floor_fraction", float)):
            if opt in doc:
                kw[opt] = cast(doc[opt])
        hs = HotStaging(**kw)
        hs._validate_order()
        return hs

    def _validate_order(self) -> None:
        if not self.throttle_down_start_s < self.ship_ignition_s:
            raise HotStagingError(
                f"throttle-down must start before ship ignition "
                f"({self.throttle_down_start_s} >= {self.ship_ignition_s})")
        if not self.ship_ignition_s < self.separation_s:
            raise HotStagingError(
                "hot staging requires ship ignition BEFORE separation "
                f"({self.ship_ignition_s} >= {self.separation_s})")
        if not self.separation_s <= self.booster_meco_s:
            raise HotStagingError(
                f"booster MECO ({self.booster_meco_s}) must not precede separation "
                f"({self.separation_s}) -- the mated ship cannot push a stack the "
                "booster has already left")
        if not 0.0 <= self.plume_axial_loss_fraction < 1.0:
            raise HotStagingError("plume_axial_loss_fraction must be in [0, 1)")
        if self.throttle_slew_max_fraction_per_s <= 0.0:
            raise HotStagingError("throttle_slew_max_fraction_per_s must be positive")
        if not 0.0 < self.throttle_floor_fraction <= 1.0:
            raise HotStagingError("throttle_floor_fraction must be in (0, 1]")

    # -- event timeline ------------------------------------------------------

    def events(self) -> tuple[tuple[str, float], ...]:
        """The four declared events, ordered; the acceptance's timeline check
        reads exactly this."""
        seq = (("throttle_down", self.throttle_down_start_s),
               ("ship_ignition", self.ship_ignition_s),
               ("separation", self.separation_s),
               ("booster_meco", self.booster_meco_s))
        return tuple(sorted(seq, key=lambda e: e[1]))

    @property
    def overlap_s(self) -> float:
        """Time the ship thrusts while the stages are still connected."""
        return self.separation_s - self.ship_ignition_s

    @property
    def meco_after_separation_s(self) -> float:
        return self.booster_meco_s - self.separation_s

    def stack_powered(self, stage1_curve: Sequence[Sequence[float]],
                      ship_curve: Sequence[Sequence[float]],
                      t: float) -> bool:
        """True if the mated stack has thrust at ``t``: the ship igniting
        before separation must leave no unpowered window between MECO-side
        events and separation."""
        if t < self.ship_ignition_s:
            return interp_thrust(stage1_curve, t) > 0.0
        if t <= self.separation_s:
            return interp_thrust(ship_curve, t - self.ship_ignition_s) > 0.0
        return False

    def no_coast_window(self, stage1_curve: Sequence[Sequence[float]],
                        ship_curve: Sequence[Sequence[float]],
                        span_s: float = 0.25) -> bool:
        """Sample every ``span_s`` from the throttle-down start to
        separation: the stack must be powered at every sample (the DC7
        3-s unpowered coast would fail this)."""
        t = self.throttle_down_start_s
        while t < self.separation_s:
            if not self.stack_powered(stage1_curve, ship_curve, t):
                return False
            t += span_s
        return True

    # -- curve checks ----------------------------------------------------------

    def check_stage1_curve(self, curve: Sequence[Sequence[float]],
                           burn_end_s: float) -> ThrustCurveReport:
        """Measure the stage-1 thrust curve against the declared limits:
        steps (same-time vertex pairs) and slew (vertex-to-vertex slope) in
        fractions of full thrust; the level the curve is cut at."""
        pts = [[float(a), float(b)] for a, b in curve]
        if pts[-1][0] > burn_end_s + 1e-9:
            raise HotStagingError("stage-1 curve extends past burn_end_s")
        full = max(p[1] for p in pts)
        if full <= 0.0:
            raise HotStagingError("stage-1 thrust curve is all zero")
        max_step = 0.0
        max_slew = 0.0
        down_slew = 0.0
        for (t0, f0), (t1, f1) in zip(pts, pts[1:]):
            step = abs(f1 - f0) / full
            if t0 == t1:
                max_step = max(max_step, step)
            else:
                slew = abs(f1 - f0) / full / (t1 - t0)
                max_slew = max(max_slew, slew)
                if t0 >= self.throttle_down_start_s - 1e-9 and f1 < f0:
                    down_slew = max(down_slew, slew)
        cut = interp_thrust(pts, burn_end_s)
        return ThrustCurveReport(max_step_fraction=max_step,
                                 max_slew_fraction_per_s=max_slew,
                                 throttle_down_slew_fraction_per_s=down_slew,
                                 floor_fraction=cut / full,
                                 cut_level_n=cut)

    # -- the composite (stack) representation --------------------------------

    def composite_stack_curve(self, stage1_curve: Sequence[Sequence[float]],
                              ship_curve: Sequence[Sequence[float]],
                              ship_isp_s: float) -> tuple[list[list[float]], float]:
        """Fold the connected-phase ship thrust into the stage-1 curve: the
        curve a single-body stack model flies -- booster thrust plus the
        plume-corrected ship thrust over the overlap window, ending at
        separation. Returns ``(curve, overlap_ship_burn_kg)``; the burn is
        the ship's own-clock propellant spent while mated (physical flow at
        the ship's Isp), which the config must deduct from the stage-2
        initial mass."""
        t_ig = self.ship_ignition_s
        t_sep = self.separation_s
        head = [[float(a), float(b)] for a, b in stage1_curve if float(a) < t_ig]
        f_b = interp_thrust(stage1_curve, t_ig)
        eta = 1.0 - self.plume_axial_loss_fraction

        def ship_n(t_abs: float) -> float:
            return interp_thrust(ship_curve, t_abs - t_ig) * eta

        def total_n(t_abs: float) -> float:
            return interp_thrust(stage1_curve, t_abs) + ship_n(t_abs)

        pts = head + [[t_ig, f_b + ship_n(t_ig)]]
        for a, _ in ship_curve:
            tt = t_ig + float(a)
            if t_ig < tt < t_sep:
                pts.append([tt, total_n(tt)])
        pts.append([t_sep, total_n(t_sep)])
        pts.append([t_sep, 0.0])
        burn_kg = _integrated_flow_kg(ship_curve, ship_isp_s, 0.0, self.overlap_s)
        return pts, burn_kg

    def ship_burn_kg(self, ship_curve: Sequence[Sequence[float]],
                     ship_isp_s: float) -> float:
        """Ship propellant spent over the overlap window on the ship's own
        clock (the amount a config deducts from the stage-2 initial mass)."""
        return _integrated_flow_kg(ship_curve, ship_isp_s, 0.0, self.overlap_s)


def load_hot_staging(vehicle_doc: Mapping[str, Any]) -> HotStaging | None:
    """The validated staging contract of a parsed vehicle document, or
    ``None`` when the document declares no hot staging (a cold-separation
    vehicle like DC7)."""
    block = vehicle_doc.get("staging")
    if block is None:
        return None
    return HotStaging.load(block)
