"""E3-02 (epic #248): the joint recovery objective as chance constraints.

ADR flightlab-0003 (R-01) put the corridor divert and the corridor fuel
margin into the FL-302 objective as two *additive soft terms*, each with its
own hand-picked weight. That fixed R-01's dominant failure class but left
two things the epic observed (benchmarks/bench_r07_nominal_dispersed.py):
the stacked design gains nothing over R-01 alone, and the touchdown tail
grows (max landing radius 338 m -> 833 m), because a weighted sum lets one
soft term buy itself off against another: a 1.7 km divert costs ~3 penalty
here and ~0.05 against the corridor radius, and the 30-evaluation search
spends it wherever the *sum* is cheapest.

This module states the same coupling the other way round: fuel margin,
divert feasibility and landing probability enter as **constraints**, and
the probability one is a genuine **chance constraint** with a deterministic
equivalent, so there is no weight to tune -- the constraints carry their
own scale (kg, m, dimensionless probability), and violations aggregate as a
worst-normalized-violation exact penalty (:data:`JOINT_PENALTY_SCALE`),
which is exact in the usual sense (once the scale exceeds the constraint
multipliers the constrained minimizer and the penalized minimizer coincide)
and cannot trade one constraint against another.

Naming (E7-01, task #277, explicit by decision): this is **penalized
optimization**, not a constrained optimizer -- the search is the same
Nelder-Mead simplex FL-302 always used, steering the exact penalty. What
the constraint statement buys over R-01's additive weights is that
violations cannot trade against each other and the penalty is exact; the
feasibility *verdict* is not the penalty value but the post-hoc check
``worst_violation <= violation_tolerance`` (:meth:`JointChanceSpec
.satisfies`). FL-302 enforces it: an accepted handoff whose worst violation
exceeds the tolerance is never reported as satisfying the chance
constraints -- it comes back ``feasible=False`` with reason code
``JOINT_CHANCE_VIOLATED``.

The three constraints, evaluated on the corridor-handoff numbers FL-302
already measures (miss, propellant, landing need):

1. **Fuel margin (chance constraint).** Under the declared dispersion bands
   the actual propellant loss about the momentum-balance landing need is a
   zero-mean random shortfall ``L`` with scale
   :func:`propellant_loss_sigma_kg`; requiring ``P(margin >= L) >=
   p_fuel_min`` is equivalent (normal model) to the deterministic bar
   ``margin >= phi_inv(p_fuel_min) * sigma_loss`` -- a *quantile* margin,
   not the fixed 1000 kg R-01 asked for. At the default bands the scale is
   a few hundred kg per ~10 t of burn exposure, which lands the bar near
   R-01's design point (that agreement is the calibration, not a given).
2. **Divert feasibility (chance constraint on the scatter).** Closed-loop
   touchdown scatter (ADR flightlab-0002: the law homes on the pad; the
   residual grows with the divert the corridor hands the landing burn) is
   modelled Rayleigh with ``sigma = sigma_floor_m + sigma_divert_gain *
   miss``. Requiring the ``scatter_quantile`` of the touchdown radius to
   stay inside ``scatter_budget_m`` has the closed-form equivalent
   ``miss <= effective_divert_budget_m`` (a Rayleigh quantile is linear in
   sigma) -- the divert budget is *derived* from a scatter requirement
   instead of being asserted, and ``P(divert closes)`` is read off the same
   model as the Rayleigh CDF at the budget.
3. **Landing probability.** The joint event: ``p_land = p_fuel * p_divert``
   must meet ``min_landing_probability`` -- the planner is told the success
   *probability* it must deliver, the quantity the Monte Carlo grades.

All calibration numbers are DESIGN ASSUMPTIONs anchored on the recorded
Monte Carlo evidence (see the defaults' docstrings and ADR flightlab-0005);
every one of them is a field of :class:`JointChanceSpec`, and the bands
mirror the FL-304 default dispersions the paired benchmark disperses with.
The module is pure math on the corridor handoff numbers -- it never flies a
trajectory (FL-302's rule: every trajectory is one ``propagate()`` call).
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DEFAULT_FUEL_QUANTILE",
    "DEFAULT_JOINT_VIOLATION_TOLERANCE",
    "DEFAULT_MIN_LANDING_PROBABILITY",
    "DEFAULT_SCATTER_BUDGET_M",
    "DEFAULT_SCATTER_QUANTILE",
    "JOINT_PENALTY_SCALE",
    "JointChanceSpec",
    "JointVerdict",
    "evaluate_joint_chance",
    "joint_penalty",
    "phi",
    "phi_inv",
    "propellant_loss_sigma_kg",
    "rayleigh_quantile_factor",
]

#: Chance level of the fuel-margin constraint: the corridor margin must
#: cover the ``p_fuel_min`` quantile of the dispersion loss. DESIGN
#: ASSUMPTION: 0.9 -- one-in-ten samples may exceed their margin; the
#: FL-305-baseline population failed reserve 13 times in 32 (0.59 covered),
#: R-01's fixed 1000 kg bar covered ~0.9 at these bands.
DEFAULT_FUEL_QUANTILE = 0.9

#: Lower bound on the joint landing probability ``p_fuel * p_divert``.
#: DESIGN ASSUMPTION: 0.8 -- comfortably above the recorded success rates
#: (FL-305 0.438, R-01 0.531 at n=32, seed 1), so meeting it moves the
#: whole population, not the accepted sample.
DEFAULT_MIN_LANDING_PROBABILITY = 0.8

#: Touchdown-radius scatter budget [m] the ``scatter_quantile`` is taken
#: against. DESIGN ASSUMPTION: 1000 m -- the same landing-zone order R-01's
#: divert budget asserted; here it is the *scatter* requirement the divert
#: budget is derived from, and it sits between the FL-305-baseline observed
#: max touchdown radius (338 m) and the R-01/R-07 tail (833 m) that this
#: task is meant to pull back.
DEFAULT_SCATTER_BUDGET_M = 1000.0

#: Quantile level of the touchdown-radius chance constraint. DESIGN
#: ASSUMPTION: 0.9 -- with the Rayleigh model the 90 % radius is
#: ``sqrt(2 ln 10) sigma ~ 2.15 sigma``.
DEFAULT_SCATTER_QUANTILE = 0.9

#: Exact-penalty scale on the worst normalized joint violation. Chosen at
#: the floor of the impact branch of the FL-302 penalty (25.0): violating a
#: joint constraint is graded at least as hard as crashing away from the
#: pad, so no corridor gain buys the violation off; unlike R-01's per-term
#: weights this scale is not a trade-off knob -- exactness says any larger
#: value leads to the same constrained solution.
JOINT_PENALTY_SCALE = 25.0

#: Tolerance on the worst normalized joint violation (:attr:
#: `JointVerdict.worst_violation`) within which the accepted handoff counts
#: as *satisfying* the joint chance constraints (E7-01 feasibility
#: semantics: a violation above it is never reported as satisfaction --
#: reason code ``JOINT_CHANCE_VIOLATED``). DESIGN ASSUMPTION 1e-6: a
#: numerical floor, not a trade-off knob -- the constraints are stated
#: exactly and the penalty search may leave a residual down to
#: double-precision noise; anything above this floor is a real violation.
DEFAULT_JOINT_VIOLATION_TOLERANCE = 1e-6

#: Uniform half-width -> 1-sigma conversion (rectangular distribution).
_UNIFORM_SIGMA = 1.0 / math.sqrt(3.0)

#: First-order gain translating an aero (density / Cd) fractional 1-sigma
#: into a propellant-use fractional 1-sigma. DESIGN ASSUMPTION calibrated
#: on the recorded corridor fuel-margin spread (FL-305 baseline min -1461 kg
#: over ~20 t of burn exposure -> ~0.04 total fractional sigma); the two
#: aero bands would contribute ~2x that bare, and only the drag-decelerated
#: fraction of the profile couples through.
_AERO_COUPLING = 0.3


def phi(z: float) -> float:
    """Standard normal CDF (math.erf; no scipy import in this module)."""
    return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))


def phi_inv(p: float) -> float:
    """Inverse standard normal CDF on (0, 1): bisection on the monotone
    :func:`phi` (60 iterations -> full double precision at negligible
    cost; called once per constraint evaluation, not per simplex step)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"phi_inv needs p in (0, 1), got {p}")
    lo, hi = -40.0, 40.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def rayleigh_quantile_factor(q: float) -> float:
    """``sqrt(-2 ln(1-q))``: the multiplier from Rayleigh sigma to its
    ``q``-quantile of radius."""
    if not 0.0 < q < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {q}")
    return math.sqrt(-2.0 * math.log(1.0 - q))


@dataclass(frozen=True)
class JointChanceSpec:
    """One joint chance-constraint specification for FL-302 (module
    docstring). ``p_fuel_min`` is the fuel quantile level,
    ``min_landing_probability`` the joint landing-probability bound,
    ``scatter_budget_m``/``scatter_quantile`` the touchdown-radius chance
    requirement the divert feasibility follows from, ``sigma_floor_m`` /
    ``sigma_divert_gain`` the touchdown-scatter model and ``*_pct`` the
    dispersion bands the loss scale is computed from (defaults mirror the
    FL-304 :class:`~orbitopt.flightlab.recovery.montecarlo.Dispersions`)."""

    p_fuel_min: float = DEFAULT_FUEL_QUANTILE
    min_landing_probability: float = DEFAULT_MIN_LANDING_PROBABILITY
    scatter_budget_m: float = DEFAULT_SCATTER_BUDGET_M
    scatter_quantile: float = DEFAULT_SCATTER_QUANTILE
    #: Touchdown-scatter floor [m] 1-sigma with zero corridor miss (the
    #: dispersion-driven closed-loop residual; DESIGN ASSUMPTION, the
    #: nominal landed subset scatters at the few-metre level and the
    #: dispersion-only band needs an order 10^1 m floor).
    sigma_floor_m: float = 40.0
    #: Proportionality of the touchdown 1-sigma on the corridor divert
    #: (closed-loop guidance closes most of the divert; the *unclosable
    #: fraction* carries into touchdown). DESIGN ASSUMPTION 0.5 anchored on
    #: the recorded pairs: handoff p50 ~0.9 km with landed p95 at metres,
    #: and the dispersion tail (handoff max ~2-3 km -> ~0.8 km touchdown).
    sigma_divert_gain: float = 0.5
    thrust_pct: float = 5.0
    isp_pct: float = 3.0
    density_pct: float = 10.0
    cd_pct: float = 8.0
    #: E7-01 feasibility semantics: the accepted handoff satisfies the joint
    #: chance constraints iff its worst normalized violation is within this
    #: tolerance (:meth:`satisfies`; 0.0 demands exact satisfaction).
    violation_tolerance: float = DEFAULT_JOINT_VIOLATION_TOLERANCE

    def __post_init__(self) -> None:
        if not 0.0 < self.p_fuel_min < 1.0:
            raise ValueError(f"p_fuel_min must be in (0, 1), got {self.p_fuel_min}")
        if not 0.0 < self.min_landing_probability < 1.0:
            raise ValueError("min_landing_probability must be in (0, 1), got "
                             f"{self.min_landing_probability}")
        if not 0.0 < self.scatter_quantile < 1.0:
            raise ValueError(f"scatter_quantile must be in (0, 1), got {self.scatter_quantile}")
        if self.scatter_budget_m <= 0.0:
            raise ValueError("scatter_budget_m must be positive")
        if self.sigma_floor_m < 0.0 or self.sigma_divert_gain < 0.0:
            raise ValueError("scatter-model floor and gain cannot be negative")
        for name in ("thrust_pct", "isp_pct", "density_pct", "cd_pct"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if self.violation_tolerance < 0.0:
            raise ValueError(f"violation_tolerance must be >= 0, got "
                             f"{self.violation_tolerance}")

    @classmethod
    def from_option(cls, option: "JointChanceSpec | Mapping[str, Any] | None"
                    ) -> "JointChanceSpec | None":
        """Planner-entry coercion: ``None`` passes through (R-01 mode), a
        spec passes through, a mapping of field names builds one (unknown
        keys raise, so a typo fails fast like the landing options do)."""
        if option is None or isinstance(option, JointChanceSpec):
            return option
        if not isinstance(option, Mapping):
            raise TypeError(f"joint_chance must be a JointChanceSpec or a mapping, "
                            f"got {type(option).__name__}")
        return cls(**dict(option))

    @classmethod
    def from_dispersion_bands(cls, bands: Any, **overrides: Any) -> "JointChanceSpec":
        """Build from anything carrying ``thrust_pct``/``isp_pct``/
        ``density_pct``/``cd_pct`` attributes (e.g. a FL-304
        ``Dispersions``), so the chance model sees exactly the bands the
        Monte Carlo flies; remaining fields default or take ``overrides``."""
        return cls(thrust_pct=float(bands.thrust_pct), isp_pct=float(bands.isp_pct),
                   density_pct=float(bands.density_pct), cd_pct=float(bands.cd_pct),
                   **overrides)

    # -- deterministic equivalents ------------------------------------------

    def propellant_loss_sigma_kg(self, exposure_kg: float) -> float:
        """1-sigma of the propellant shortfall about the momentum-balance
        landing need for ``exposure_kg`` of propellant burned-under-uncertainty
        (boostback + entry burn actually spent plus the landing need).
        First-order multiplicative model (module docstring): the uniform
        bands convert to sigmas, the aero bands couple through
        :data:`_AERO_COUPLING`."""
        if exposure_kg < 0.0:
            raise ValueError(f"exposure_kg cannot be negative, got {exposure_kg}")
        f = _UNIFORM_SIGMA / 100.0
        frac = math.sqrt((self.thrust_pct * f) ** 2
                         + (self.isp_pct * f) ** 2
                         + (_AERO_COUPLING * self.density_pct * f) ** 2
                         + (_AERO_COUPLING * self.cd_pct * f) ** 2)
        return float(exposure_kg) * frac

    def required_margin_kg(self, exposure_kg: float) -> float:
        """Deterministic equivalent of ``P(margin >= loss) >= p_fuel_min``:
        the quantile bar ``phi_inv(p_fuel_min) * sigma`` (may be negative
        for sub-median levels -- then no margin is required)."""
        return phi_inv(self.p_fuel_min) * self.propellant_loss_sigma_kg(exposure_kg)

    def touchdown_sigma_m(self, miss_m: float) -> float:
        """Predicted touchdown 1-sigma [m] for a corridor handoff ``miss_m``
        from the pad (linear Rayleigh model, module docstring)."""
        return self.sigma_floor_m + self.sigma_divert_gain * abs(float(miss_m))

    def divert_budget_m(self) -> float:
        """Deterministic equivalent of the scatter chance constraint: the
        corridor miss that puts the ``scatter_quantile`` touchdown radius
        exactly at ``scatter_budget_m`` (Rayleigh quantiles are linear in
        sigma). 0 when the scatter floor alone busts the budget."""
        sigma_max = self.scatter_budget_m / rayleigh_quantile_factor(self.scatter_quantile)
        return max(0.0, (sigma_max - self.sigma_floor_m) / self.sigma_divert_gain)

    def p_divert(self, miss_m: float) -> float:
        """Rayleigh CDF of the touchdown radius at the scatter budget: the
        probability the scatter chance constraint holds for a handoff
        ``miss_m``."""
        s = self.touchdown_sigma_m(miss_m)
        if s <= 0.0:
            return 1.0
        r = self.scatter_budget_m / s
        return 1.0 - math.exp(-0.5 * r * r)

    def satisfies(self, worst_violation: float) -> bool:
        """E7-01 feasibility semantics (module docstring): does a joint
        verdict with this worst normalized violation satisfy the
        constraints? Satisfaction is ``worst_violation <=
        violation_tolerance`` -- a violation *above* the tolerance is never
        satisfaction (FL-302 reports ``JOINT_CHANCE_VIOLATED``)."""
        return float(worst_violation) <= self.violation_tolerance


@dataclass(frozen=True)
class JointVerdict:
    """The joint constraint read-out at one corridor handoff: the component
    probabilities, the deterministic-equivalent bars, and the normalized
    violations (each >= 0, active constraint > 0). ``violations`` is ordered
    (fuel, divert, landing probability)."""

    p_fuel: float
    p_divert: float
    p_land: float
    loss_sigma_kg: float
    required_margin_kg: float
    divert_budget_m: float
    violations: tuple[float, float, float]

    @property
    def worst_violation(self) -> float:
        return max(0.0, *self.violations)

    def to_dict(self) -> dict[str, Any]:
        return {"p_fuel": self.p_fuel, "p_divert": self.p_divert,
                "p_land": self.p_land, "loss_sigma_kg": self.loss_sigma_kg,
                "required_margin_kg": self.required_margin_kg,
                "divert_budget_m": self.divert_budget_m,
                "violations": [float(g) for g in self.violations],
                "worst_violation": self.worst_violation}


def evaluate_joint_chance(spec: JointChanceSpec, *, miss_m: float,
                          margin_kg: float, exposure_kg: float) -> JointVerdict:
    """Grade one corridor handoff against the joint chance constraints.
    ``margin_kg`` is propellant at the corridor minus landing need minus
    vehicle reserve; ``exposure_kg`` the propellant actually spent to get
    there plus the landing need (the mass whose usage the dispersions
    perturb)."""
    sigma = spec.propellant_loss_sigma_kg(exposure_kg)
    p_fuel = phi(margin_kg / sigma) if sigma > 0.0 else (
        1.0 if margin_kg > 0.0 else (0.5 if margin_kg == 0.0 else 0.0))
    p_div = spec.p_divert(miss_m)
    p_land = p_fuel * p_div
    z = phi_inv(spec.p_fuel_min)
    req = z * sigma
    budget = spec.divert_budget_m()
    # normalized violations (module docstring): each constraint has its own
    # natural scale -- sigma of the loss, the scatter budget, p_min itself
    g_fuel = max(0.0, (req - margin_kg) / sigma) if sigma > 0.0 else 0.0
    g_div = max(0.0, (miss_m - budget) / spec.scatter_budget_m)
    g_prob = max(0.0, (spec.min_landing_probability - p_land)
                 / spec.min_landing_probability)
    return JointVerdict(p_fuel=p_fuel, p_divert=p_div, p_land=p_land,
                        loss_sigma_kg=sigma, required_margin_kg=req,
                        divert_budget_m=budget,
                        violations=(g_fuel, g_div, g_prob))


def joint_penalty(verdict: JointVerdict) -> float:
    """Exact-penalty contribution of the joint constraints: the *worst*
    normalized violation, squared, at :data:`JOINT_PENALTY_SCALE` (module
    docstring: a worst-violation exact penalty, not an additive weight --
    no constraint can buy itself off against another)."""
    w = verdict.worst_violation
    return JOINT_PENALTY_SCALE * w * w
