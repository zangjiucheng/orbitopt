"""F-03 (task #228): mass, centre of gravity and inertia tensor as functions of
the propellant consumed.

JSBSim recomputes the flying vehicle's mass properties from the tank contents
every frame (``FGMassBalance``); what this module adds is the *declared* model
-- the numbers a scenario config says the vehicle must have at the two ends of
its propellant load, interpolated in between. That makes the 6-DOF backend's
mass bookkeeping checkable (``orbitopt.flightlab.adapters.sixdof`` turns it
into per-run checks: flow integral vs. mass change, CG/inertia at full load
and at burnout) and gives every other consumer one answer to "what does this
vehicle weigh, where is it balanced and how does it tumble at 37 % propellant?".

Model
-----
Two declared endpoints, ``full`` (tanks at capacity) and ``empty`` (burnout /
dry), each a :class:`MassState`: total mass, CG and the full symmetric
inertia tensor *about that CG*, in the vehicle's structural frame (JSBSim's:
``x`` positive aft of the nose, ``y`` right, ``z`` up; the driver reports
``inertia/cg-*-in`` and ``inertia/i*-slugs_ft2`` in that same frame).

The interpolation is linear in the **remaining propellant mass** -- not in
time, not in ``delta-v`` -- and it is built so that both endpoints are exact
by construction. The two endpoints are decomposed into a dry structure plus
one lumped propellant load::

    m_p     = m_full - m_empty                        (propellant capacity)
    r_p     = (m_full c_full - m_empty c_empty) / m_p (propellant centroid)
    I_own   = I_full - shift(I_empty, m_empty, c_full - c_empty)
              - m_p * parallel_axis(r_p - c_full)     (propellant's own inertia)

and the state at burn fraction ``alpha`` (1 = full, 0 = empty) is the exact
superposition of the two parts, both moved to the CG this state declares::

    m(c)   = m_empty + alpha * m_p
    c(c)   = (m_empty c_empty + alpha m_p r_p) / m(c)
    I(c)   = shift(I_empty, m_empty, c_empty - c) + alpha * I_own
             + alpha * m_p * parallel_axis(r_p - c)

with ``shift`` the parallel-axis transport. ``alpha = 1`` and ``alpha = 0``
reproduce the declared endpoints to rounding, by construction.

Everything else follows from the parallel-axis theorem, so the CG follows the
correct hyperbola ``1/m(c)`` (linear interpolation of the CG itself would put
it in the wrong place whenever the propellant does not sit at the dry CG) and
the tensor stays symmetric and positive semi-definite along the whole burn.

The model is exact for a vehicle whose propellant centroid does not move while
it drains (one tank, or several tanks draining proportionally, which is what
the JSBSim models here do). For a vehicle whose tanks drain *sequentially* the
centroid moves, and the endpoints stay exact while mid-burn values carry the
lumping error -- which is why the backend checks the *endpoints* strictly and
reports mid-burn agreement as its own, looser, claim.

Usage::

    from orbitopt.flightlab.sixdof.mass_properties import load_mass_model
    model = load_mass_model({"full": {...}, "empty": {...}})
    at_half = model.properties.at_fraction(0.5)

Pure Python: importing this module needs neither ``jsbsim`` nor ``numpy``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from functools import cached_property
from typing import Any, Mapping

#: Component order of a symmetric inertia tensor: the diagonal, then the three
#: independent off-diagonals (tensor entries, not JSBSim's negated products).
TENSOR_COMPONENTS = ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")

#: Relative slack with which a declared tensor is accepted as positive
#: semi-definite (config round-tripping, kg*m^2-scaled).
_PSD_REL_TOL = 1e-9


class MassPropertiesError(ValueError):
    """A declared mass model is inconsistent -- the offending key is named, so
    a scenario typo fails here, not in the middle of a burn."""


# --------------------------------------------------------------------------
# the tensor helpers (3x3 symmetric, plain tuples: no numpy dependency)
# --------------------------------------------------------------------------

def _num(value: Any, where: str, key: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise MassPropertiesError(f"{where}: {key}={value!r} is not a number") from exc
    if not math.isfinite(v):
        raise MassPropertiesError(f"{where}: {key} must be finite, got {v}")
    return v


def _as_tensor(values: Any, where: str) -> tuple[float, float, float, float, float, float]:
    """Validate the six tensor components in :data:`TENSOR_COMPONENTS` order,
    from a 6-sequence or a mapping keyed by the component names (the three
    products may be omitted from a mapping and default to 0.0)."""
    if isinstance(values, Mapping):
        unknown = sorted(set(values) - set(TENSOR_COMPONENTS))
        if unknown:
            raise MassPropertiesError(f"{where}: unknown inertia component(s): "
                                      f"{', '.join(unknown)}")
        missing = [k for k in TENSOR_COMPONENTS[:3] if k not in values]
        if missing:
            raise MassPropertiesError(f"{where}: inertia is missing component(s): "
                                      f"{', '.join(missing)} (the products may be "
                                      "omitted, the diagonal may not)")
        seq = [_num(values.get(k, 0.0), where, str(k)) for k in TENSOR_COMPONENTS]
    else:
        seq = list(values or ())
        if len(seq) != 6:
            raise MassPropertiesError(f"{where}: inertia needs the 6 components "
                                      f"{TENSOR_COMPONENTS}, got {len(seq)}")
        seq = [_num(v, where, TENSOR_COMPONENTS[i]) for i, v in enumerate(seq)]
    return tuple(seq)  # type: ignore[return-value]


def _vec(values: Any, where: str) -> tuple[float, float, float]:
    seq = list(values or ())
    if len(seq) != 3:
        raise MassPropertiesError(f"{where}: expected 3 numbers, got {len(seq)}")
    seq = [_num(v, where, f"component {i}") for i, v in enumerate(seq)]
    return (seq[0], seq[1], seq[2])


def _tensor_matrix(i: tuple[float, float, float, float, float, float]
                   ) -> tuple[tuple[float, float, float], ...]:
    ixx, iyy, izz, ixy, ixz, iyz = i
    return ((ixx, ixy, ixz), (ixy, iyy, iyz), (ixz, iyz, izz))


def parallel_axis(i: tuple[float, float, float, float, float, float], mass_kg: float,
                  d: tuple[float, float, float]
                  ) -> tuple[float, float, float, float, float, float]:
    """The tensor ``i``, given about a centre of gravity at offset ``d`` from
    the new reference point, moved to that reference point:
    ``I + m (|d|^2 E - d d^T)``. ``d = 0`` is the identity."""
    dx, dy, dz = d
    dd = dx * dx + dy * dy + dz * dz
    ixx, iyy, izz, ixy, ixz, iyz = i
    return (ixx + mass_kg * (dy * dy + dz * dz),
            iyy + mass_kg * (dx * dx + dz * dz),
            izz + mass_kg * (dx * dx + dy * dy),
            ixy - mass_kg * dx * dy,
            ixz - mass_kg * dx * dz,
            iyz - mass_kg * dy * dz)


def _tensor_add(a, b, scale_b: float = 1.0):
    return tuple(x + scale_b * y for x, y in zip(a, b))  # type: ignore[return-value]


def _leading_minors(i):
    m = _tensor_matrix(i)
    d1 = m[0][0]
    d2 = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    d3 = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
          - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
          + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    return d1, d2, d3


def _require_psd(i, where: str) -> None:
    """Sylvester's criterion (leading principal minors >= 0) -- an inertia
    tensor with a negative principal moment is not a physical body."""
    scale = max(1.0, max(abs(v) for v in i))
    d1, d2, d3 = _leading_minors(i)
    if d1 < -_PSD_REL_TOL * scale or d2 < -_PSD_REL_TOL * scale ** 2 \
            or d3 < -_PSD_REL_TOL * scale ** 3:
        raise MassPropertiesError(f"{where}: inertia tensor is not positive "
                                  f"semi-definite (minors {d1:.6g}, {d2:.6g}, {d3:.6g})")


# --------------------------------------------------------------------------
# states
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MassState:
    """Mass, CG and inertia tensor of one propellant load, in the vehicle's
    structural frame (JSBSim's ``<location>`` frame: ``x`` aft of the nose) --
    the CG and the tensor are about the CG this state declares."""

    mass_kg: float
    cg_m: tuple[float, float, float]
    inertia_kg_m2: tuple[float, float, float, float, float, float]

    def __post_init__(self) -> None:
        if not math.isfinite(self.mass_kg) or self.mass_kg <= 0.0:
            raise MassPropertiesError(f"mass_kg must be > 0 and finite, got {self.mass_kg}")
        object.__setattr__(self, "cg_m", _vec(self.cg_m, "cg_m"))
        object.__setattr__(self, "inertia_kg_m2", _as_tensor(self.inertia_kg_m2, "inertia"))
        _require_psd(self.inertia_kg_m2, "inertia")

    @property
    def inertia_matrix(self) -> tuple[tuple[float, float, float], ...]:
        """The tensor as a symmetric 3x3 row-major matrix."""
        return _tensor_matrix(self.inertia_kg_m2)

    @classmethod
    def from_mapping(cls, doc: Any, where: str) -> "MassState":
        if not isinstance(doc, Mapping):
            raise MassPropertiesError(f"{where}: expected a mapping, got "
                                      f"{type(doc).__name__}")
        unknown = sorted(set(doc) - {"mass_kg", "cg_m", "inertia_kg_m2"})
        if unknown:
            raise MassPropertiesError(f"{where}: unknown key(s): {', '.join(unknown)}")
        for key in ("mass_kg", "cg_m", "inertia_kg_m2"):
            if key not in doc:
                raise MassPropertiesError(f"{where}: required key {key!r} is missing")
        try:
            mass = float(doc["mass_kg"])
        except (TypeError, ValueError) as exc:
            raise MassPropertiesError(f"{where}: mass_kg={doc['mass_kg']!r} is not a "
                                      "number") from exc
        return cls(mass_kg=mass,
                   cg_m=_vec(doc["cg_m"], f"{where}: cg_m"),
                   inertia_kg_m2=_as_tensor(doc["inertia_kg_m2"], f"{where}: inertia_kg_m2"))

    def as_mapping(self) -> dict[str, Any]:
        return {"mass_kg": self.mass_kg, "cg_m": list(self.cg_m),
                "inertia_kg_m2": dict(zip(TENSOR_COMPONENTS, self.inertia_kg_m2))}


@dataclass(frozen=True)
class MassModelTolerances:
    """Declared slack of the F-03 checks (the FL-403 bounds pattern: the limits
    the per-run checks carry come from the config, not from code).

    * ``flow_lag_frames`` / ``mass_abs_kg`` -- the mass-conservation check's
      allowance for JSBSim's one-frame-late mass balance (``FGMassBalance``
      integrates before the tanks are drained) plus XML unit rounding;
    * ``cg_abs_m`` / ``inertia_rel`` -- endpoint agreement on CG (per axis, m)
      and on the tensor (worst component, relative);
    * ``interpolation_cg_abs_m`` / ``interpolation_inertia_rel`` -- the same
      claims along the burn, with slack for the lumped-propellant idealisation
      and for the integrator's one-frame-late mass balance;
    * ``burnout_fraction`` -- how empty the last sample must be (of the
      declared propellant capacity) for the burnout endpoint to count.
    """

    flow_lag_frames: float = 2.0
    mass_abs_kg: float = 2.0
    cg_abs_m: float = 0.02
    inertia_rel: float = 2e-3
    interpolation_cg_abs_m: float = 0.05
    interpolation_inertia_rel: float = 2e-2
    burnout_fraction: float = 1e-3

    def as_mapping(self) -> dict[str, float]:
        return {"flow_lag_frames": self.flow_lag_frames, "mass_abs_kg": self.mass_abs_kg,
                "cg_abs_m": self.cg_abs_m, "inertia_rel": self.inertia_rel,
                "interpolation_cg_abs_m": self.interpolation_cg_abs_m,
                "interpolation_inertia_rel": self.interpolation_inertia_rel,
                "burnout_fraction": self.burnout_fraction}


@dataclass(frozen=True)
class MassModelConfig:
    """A scenario's ``mass_model`` section: the declared endpoints plus the
    tolerances the F-03 checks are judged against."""

    properties: MassProperties
    tolerances: MassModelTolerances = field(default_factory=MassModelTolerances)

    def as_mapping(self) -> dict[str, Any]:
        return {"full": self.properties.full.as_mapping(),
                "empty": self.properties.empty.as_mapping(),
                "tolerances": self.tolerances.as_mapping()}


@dataclass(frozen=True)
class MassProperties:
    """The declared two-endpoint model: interpolate mass / CG / inertia in the
    remaining propellant mass (module docstring for the derivation)."""

    full: MassState
    empty: MassState

    def __post_init__(self) -> None:
        if self.full.mass_kg <= self.empty.mass_kg:
            raise MassPropertiesError(
                f"full-load mass ({self.full.mass_kg} kg) must exceed the burnout/dry "
                f"mass ({self.empty.mass_kg} kg): the model's propellant mass would be "
                "non-positive")

    @cached_property
    def propellant_mass_kg(self) -> float:
        """Declared propellant capacity (``m_full - m_empty``)."""
        return self.full.mass_kg - self.empty.mass_kg

    @cached_property
    def propellant_cg_m(self) -> tuple[float, float, float]:
        """CG of the whole declared propellant load, structural frame."""
        mp = self.propellant_mass_kg
        f, e = self.full, self.empty
        return tuple((f.mass_kg * c - e.mass_kg * d) / mp
                     for c, d in zip(f.cg_m, e.cg_m))  # type: ignore[return-value]

    @cached_property
    def propellant_inertia_kg_m2(self):
        """The propellant load's own inertia tensor (about the propellant
        centroid), back out of the full-load endpoint. Must be positive
        semi-definite -- if it is not, the declared endpoints cannot be the
        same vehicle."""
        f, e = self.full, self.empty
        zero = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        dry_at_full_cg = parallel_axis(e.inertia_kg_m2, e.mass_kg,
                                      tuple(a - b for a, b in zip(e.cg_m, f.cg_m)))
        prop_at_full_cg = parallel_axis(zero, self.propellant_mass_kg,
                                        tuple(a - b for a, b in
                                              zip(self.propellant_cg_m, f.cg_m)))
        own = _tensor_add(_tensor_add(f.inertia_kg_m2, dry_at_full_cg, scale_b=-1.0),
                          prop_at_full_cg, scale_b=-1.0)
        _require_psd(own, "propellant own inertia (the full/empty endpoints "
                          "are inconsistent)")
        return own

    def at_fraction(self, burn_fraction: float) -> MassState:
        """The mass properties at ``burn_fraction`` of the declared propellant
        load remaining (``1.0`` == :attr:`full`, ``0.0`` == :attr:`empty`)."""
        if not math.isfinite(burn_fraction) or not 0.0 <= burn_fraction <= 1.0:
            raise ValueError(f"burn_fraction must lie in [0, 1], got {burn_fraction}")
        e = self.empty
        mp = self.propellant_mass_kg
        zero = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        mass = e.mass_kg + burn_fraction * mp
        cg = tuple((e.mass_kg * c + burn_fraction * mp * p) / mass
                   for c, p in zip(e.cg_m, self.propellant_cg_m))
        dry = parallel_axis(e.inertia_kg_m2, e.mass_kg,
                            tuple(c - g for c, g in zip(e.cg_m, cg)))
        inertia = _tensor_add(dry, self.propellant_inertia_kg_m2, scale_b=burn_fraction)
        inertia = _tensor_add(inertia,
                              parallel_axis(zero, burn_fraction * mp,
                                            tuple(p - c for p, c in
                                                  zip(self.propellant_cg_m, cg))))
        return MassState(mass_kg=mass, cg_m=cg,  # type: ignore[arg-type]
                         inertia_kg_m2=inertia)

    def at_propellant_mass_kg(self, propellant_mass_kg: float) -> MassState:
        """Same as :meth:`at_fraction`, with the remaining propellant in kg
        (what a simulator's tank contents say)."""
        frac = _num(propellant_mass_kg, "at_propellant_mass_kg", "propellant_mass_kg")
        if not 0.0 <= frac <= self.propellant_mass_kg:
            raise MassPropertiesError(
                f"propellant mass {frac} kg is outside [0, "
                f"{self.propellant_mass_kg}] kg")
        return self.at_fraction(frac / self.propellant_mass_kg)

    def fraction_at_mass_kg(self, mass_kg: float) -> float:
        """The burn fraction that corresponds to a total mass (the quantity a
        simulator reports), i.e. ``(m - m_empty) / m_propellant``."""
        frac = (mass_kg - self.empty.mass_kg) / self.propellant_mass_kg
        if not 0.0 <= frac <= 1.0:
            raise MassPropertiesError(f"mass {mass_kg} kg is outside the declared "
                                      f"load [{self.empty.mass_kg}, "
                                      f"{self.full.mass_kg}] kg")
        return frac

    def as_mapping(self) -> dict[str, Any]:
        return {"full": self.full.as_mapping(), "empty": self.empty.as_mapping()}


def load_mass_model(doc: Any, where: str = "mass_model") -> MassModelConfig:
    """Parse + validate a scenario's ``mass_model`` section::

        mass_model:
          full:  {mass_kg: 70000.0, cg_m: [18.2857143, 0.0, 0.0],
                  inertia_kg_m2: {ixx: 111760.0, iyy: 4212000.0, izz: 4212000.0}}
          empty: {mass_kg: 10000.0, cg_m: [20.0, 0.0, 0.0],
                  inertia_kg_m2: {ixx: 3.4e4, iyy: 1.7e6, izz: 1.7e6}}
          tolerances: {flow_lag_frames: 2.0, mass_abs_kg: 2.0, cg_abs_m: 0.02,
                       inertia_rel: 0.002, interpolation_cg_abs_m: 0.05,
                       interpolation_inertia_rel: 0.02, burnout_fraction: 0.001}  # optional

    Off-diagonal inertia components may be omitted (they default to 0.0).
    Raises :class:`MassPropertiesError` (a ``ValueError``) naming the key."""
    if not isinstance(doc, Mapping):
        raise MassPropertiesError(f"{where}: expected a mapping, got {type(doc).__name__}")
    unknown = sorted(set(doc) - {"full", "empty", "tolerances"})
    if unknown:
        raise MassPropertiesError(f"{where}: unknown key(s): {', '.join(unknown)}")
    for key in ("full", "empty"):
        if key not in doc:
            raise MassPropertiesError(f"{where}: required key {key!r} is missing")
    props = MassProperties(full=MassState.from_mapping(doc["full"], f"{where}: full"),
                           empty=MassState.from_mapping(doc["empty"], f"{where}: empty"))
    tol_doc = doc.get("tolerances") or {}
    if not isinstance(tol_doc, Mapping):
        raise MassPropertiesError(f"{where}: tolerances: expected a mapping, got "
                                  f"{type(tol_doc).__name__}")
    unknown = sorted(set(tol_doc) - {f.name for f in fields(MassModelTolerances)})
    if unknown:
        raise MassPropertiesError(f"{where}: tolerances has unknown key(s): "
                                  f"{', '.join(unknown)}")
    defaults = MassModelTolerances()
    values: dict[str, float] = {}
    for tol_field in fields(MassModelTolerances):
        key, default = tol_field.name, getattr(defaults, tol_field.name)
        if key not in tol_doc or tol_doc[key] is None:
            values[key] = default
            continue
        try:
            v = float(tol_doc[key])
        except (TypeError, ValueError) as exc:
            raise MassPropertiesError(f"{where}: tolerances.{key}={tol_doc[key]!r} "
                                      "is not a number") from exc
        if not 0.0 <= v:
            raise MassPropertiesError(f"{where}: tolerances.{key} must be >= 0, got {v}")
        values[key] = v
    if values["burnout_fraction"] > 1.0:
        raise MassPropertiesError(f"{where}: tolerances.burnout_fraction must be <= 1, "
                                  f"got {values['burnout_fraction']}")
    return MassModelConfig(properties=props, tolerances=MassModelTolerances(**values))
