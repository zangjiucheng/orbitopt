"""Aerodynamic coefficient model for the 6-DOF Flight Lab plant (F-02, task
#227): force/moment coefficients as tables in Mach, angle of attack and
sideslip, interpolated, turned into body-axis force and moment by the dynamic
pressure, plus the max-Q diagnostic the same run has to report.

The model
---------
Six coefficients, each its own :class:`CoefficientTable`, in the body axes of
:class:`orbitopt.flightlab.sixdof.rigidbody` (x forward, y right, z down):

======================  ===============================  ==========================
coefficient             force / moment                   sign convention
======================  ===============================  ==========================
``ca``                  axial, ``Fx = -q S ca``          drag is positive ``ca``
``cn``                  normal, ``Fz = -q S cn``         ``cn > 0`` lifts toward -z
``cy``                  side, ``Fy = +q S cy``           ``cy < 0`` restores ``beta > 0``
``cl``, ``cm``, ``cn_yaw``  ``M = q S (b cl, c cm, c cn_yaw)``   roll / pitch / yaw
======================  ===============================  ==========================

Static stability therefore reads ``cn_alpha > 0``, ``cm_alpha < 0``,
``cy_beta < 0``, ``cn_yaw_beta < 0`` -- the same weathervane sign the 3-DOF
recovery propagator assumes when it points the booster into the airflow
(:mod:`orbitopt.flightlab.recovery.propagator` has no attitude, so it gets
that stability for free; here it has to come from the tables).

A table may depend on any subset of the three axes (``ca`` from Mach alone is
a perfectly good model -- that is exactly what the flight-vehicle schema's
``aero: {ca: [[M, ca], ...]}`` carries); the interpolation is N-linear over
whatever axes a table declares, so a 1-D table reduces to
``numpy.interp`` semantics and a 3-D one to trilinear.

Out of scope here, deliberately: rate derivatives (``cm_q``, ``cl_p``, ...)
as *table axes*. :attr:`AeroModel.pitch_damping_cm_q` and friends are single
linear constants standing in for them, because the 6-DOF plant needs some
pitch damping to fly, and a proper fourth table axis is F-05's business once
there are measured curves to put in it. Grid fins, landing legs, fairings and
flow reversal are likewise not modelled -- see docs/flightlab.md.

Interpolation and extrapolation
-------------------------------
Each table clamps, extrapolates linearly along its end segment, or raises, at
or beyond an axis end (:attr:`CoefficientTable.extrapolation`,
:data:`EXTRAPOLATION_POLICIES`). ``clamp`` is the default because it matches
both ``numpy.interp`` (what the 3-DOF propagator uses for ``Cd(M)``) and
OpenTsiolkovsky's Cd table, so a shared vehicle means shared numbers across
the toolchain; ``error`` is what a formal campaign wants when a run leaves the
validated envelope, and the aero check that reports it is F-07's.

Data
----
:class:`AeroModel` loads from a plain mapping (:meth:`AeroModel.from_document`)
so a coefficient dump from CFD, a wind-tunnel reduction or a hand-written
vehicle file all reach the same object; the packaged
``data/flightlab/aero/*.yaml`` are ``axisymmetric`` source documents (a Mach
breakout of ``ca``, a normal-curve slope, a static margin) that
:meth:`AeroModel.axisymmetric` turns into the three-dimensional tables -- the
same promotion the ``cn_alpha_per_rad`` field of the flight-vehicle schema
implies, made explicit and provenance-tagged.
"""
from __future__ import annotations

import bisect
import itertools
import json
import math
from dataclasses import dataclass, field
from functools import cached_property
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import jsonschema
import numpy as np
import yaml

__all__ = [
    "AERO_AXES",
    "AeroError",
    "AeroModel",
    "CoefficientTable",
    "EXTRAPOLATION_POLICIES",
    "MaxQ",
    "REQUIRED_COEFFICIENTS",
    "aero_schema",
    "angle_of_attack_deg",
    "dynamic_pressure_pa",
    "load_aero_model",
    "load_packaged_aero",
    "max_q_of",
    "sideslip_deg",
    "validate_aero_document",
]

#: The canonical aero axes, in table-axis order. Every table axis must be one
#: of these, in this relative order.
AERO_AXES: tuple[str, ...] = ("mach", "alpha_deg", "beta_deg")

#: How a table behaves at or beyond the end of one of its axes: ``clamp`` holds
#: the end value (``numpy.interp`` semantics), ``linear`` continues the slope of
#: the end segment (``numpy.interp`` is ``clamp``, so this one is *not* the
#: cross-toolchain default -- it exists because extrapolating a wave-drag rise
#: off the table end is sometimes what you want to see), ``error`` refuses.
EXTRAPOLATION_POLICIES: tuple[str, ...] = ("clamp", "linear", "error")

#: The six coefficients a 6-DOF aero model must supply.
REQUIRED_COEFFICIENTS: tuple[str, ...] = ("ca", "cn", "cy", "cl", "cm", "cn_yaw")


class AeroError(ValueError):
    """A malformed aero document, an out-of-envelope read under ``error``
    extrapolation, or an unusable coefficient evaluation."""


#: The JSON Schema every aero document is validated against on load
#: (``data/flightlab/schemas/flight-aero-1.0.json``), the same pattern as the
#: flight-vehicle catalog.
_SCHEMA_FILE = "flight-aero-1.0.json"


def aero_schema() -> dict:
    """The packaged ``flight-aero-1.0`` JSON Schema."""
    schema = resources.files("orbitopt").joinpath(
        "data", "flightlab", "schemas", _SCHEMA_FILE)
    return json.loads(schema.read_text(encoding="utf-8"))


def validate_aero_document(doc: Mapping[str, Any]) -> None:
    """Schema-validate an aero document; raise :class:`AeroError` naming every
    violation. :meth:`AeroModel.from_document` calls this, so nothing that
    flies has to remember to."""
    validator = jsonschema.Draft202012Validator(aero_schema())
    lines = []
    for error in sorted(validator.iter_errors(doc), key=lambda e: str(list(e.absolute_path))):
        path = json.dumps(list(error.absolute_path)) or "(root)"
        if error.context:  # oneOf/anyOf: report why each alternative was rejected
            lines.extend(f"  {path}: {sub.message}" for sub in error.context[:4])
        else:
            lines.append(f"  {path}: {error.message}")
    if lines:
        raise AeroError(f"aero document fails {_SCHEMA_FILE}: {len(lines)} violation(s):\n"
                        + "\n".join(lines))


# --------------------------------------------------------------------------
# coefficient tables
# --------------------------------------------------------------------------

def _freeze(values: Any, where: str) -> tuple:
    """Nested lists -> nested tuples (a frozen dataclass holds immutable
    premises), validating that every level is a non-rectangular-free list of
    numbers."""
    if isinstance(values, (list, tuple)):
        return tuple(_freeze(v, where) for v in values)
    if isinstance(values, np.ndarray):
        return _freeze(values.tolist(), where)
    if isinstance(values, bool) or not isinstance(values, (int, float)):
        raise AeroError(f"{where}: expected a number or a (nested) list of numbers, got {values!r}")
    return float(values)


def _shape_of(values: Any) -> tuple[int, ...]:
    shape = []
    node = values
    while isinstance(node, (tuple, list)):
        shape.append(len(node))
        if not node:
            raise AeroError("empty coefficient table")
        node = node[0]
    return tuple(shape)


def _check_rectangular(values: Any, shape: tuple[int, ...], where: str) -> None:
    if not shape:
        if isinstance(values, (tuple, list)):
            raise AeroError(f"{where}: expected a scalar, got a list")
        return
    if not isinstance(values, (tuple, list)) or len(values) != shape[0]:
        got = len(values) if isinstance(values, (tuple, list)) else "scalar"
        raise AeroError(f"{where}: expected {shape[0]} entries, got {got}")
    for entry in values:
        _check_rectangular(entry, shape[1:], where)


@dataclass(frozen=True)
class CoefficientTable:
    """One aerodynamic coefficient as a table over any ordered subset of
    :data:`AERO_AXES`, interpolated N-linearly between nodes.

    ``values`` is nested, outermost axis first, matching ``axes``, with one
    entry per node: a ``(mach, alpha_deg)`` table on 3 Mach nodes and 5 AoA
    nodes holds a ``[3][5]`` nest, a Mach-only table a flat list of 3. (The
    flight-vehicle schema's ``ca: [[M, ca], ...]`` pair-list form is the input
    of :meth:`AeroModel.axisymmetric`, not of a table: the pairs are split
    into axis nodes and values there.)
    """

    #: Key in an :class:`AeroModel`'s table mapping (``ca``, ``cn``, ...).
    name: str
    #: The axes this table is tabulated on, in :data:`AERO_AXES` order.
    axes: tuple[str, ...]
    #: Node coordinates of each axis, strictly increasing, >= 2 nodes.
    axis_values: Mapping[str, tuple[float, ...]]
    #: Tabulated coefficient values, nested outer-axis-first (see the class docstring).
    values: tuple
    #: One of :data:`EXTRAPOLATION_POLICIES`.
    extrapolation: str = "clamp"
    #: Free-text description shown in reports.
    label: str = ""
    #: Where the numbers came from (a ``data_provenance``-style mapping or None).
    provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise AeroError("a coefficient table needs a name")
        axes = tuple(self.axes)
        if not axes:
            raise AeroError(f"{self.name}: a table needs at least one axis")
        unknown = [axis for axis in axes if axis not in AERO_AXES]
        if unknown:
            raise AeroError(f"{self.name}: unknown axis/axes {unknown}; use {AERO_AXES}")
        if len(set(axes)) != len(axes):
            raise AeroError(f"{self.name}: duplicate axis in {axes}")
        if any(AERO_AXES.index(axes[i]) > AERO_AXES.index(axes[i + 1])
               for i in range(len(axes) - 1)):
            raise AeroError(f"{self.name}: axes {axes} are not in {AERO_AXES} order")
        object.__setattr__(self, "axes", axes)
        if self.extrapolation not in EXTRAPOLATION_POLICIES:
            raise AeroError(f"{self.name}: extrapolation {self.extrapolation!r} is not one of "
                            f"{EXTRAPOLATION_POLICIES}")
        frozen = _freeze(self.values, f"{self.name}.values")
        shape = _shape_of(frozen)
        _check_rectangular(frozen, shape, f"{self.name}.values")
        nodes: dict[str, tuple[float, ...]] = {}
        for axis in axes:
            if axis not in self.axis_values:
                raise AeroError(f"{self.name}: no node values for axis {axis}")
            values = tuple(float(v) for v in self.axis_values[axis])
            if len(values) < 2:
                raise AeroError(f"{self.name}: axis {axis} needs at least 2 nodes")
            if any(b <= a for a, b in zip(values, values[1:])):
                raise AeroError(f"{self.name}: axis {axis} nodes must increase strictly: {values}")
            nodes[axis] = values
        want = tuple(len(nodes[axis]) for axis in axes)
        if shape != want:
            raise AeroError(f"{self.name}: values are shaped {shape}, the {len(axes)} axis/axes "
                            f"{axes} want {want}")
        object.__setattr__(self, "values", frozen)
        object.__setattr__(self, "axis_values", nodes)

    @property
    def shape(self) -> tuple[int, ...]:
        """Node counts, outer axis first."""
        return _shape_of(self.values)

    @cached_property
    def grid(self) -> np.ndarray:
        """The tabulated values as an array (same nesting as :attr:`values`)."""
        return np.asarray(self.values, dtype=float)

    @cached_property
    def _node_lists(self) -> list[list[float]]:
        return [[float(v) for v in self.axis_values[axis]] for axis in self.axes]

    @cached_property
    def _step_lists(self) -> list[list[float]]:
        return [[b - a for a, b in zip(nodes, nodes[1:])] for nodes in self._node_lists]

    @cached_property
    def _grid_lists(self) -> list:
        """Nested Python lists of the grid: the scalar fast path indexes these
        instead of the ndarray, because this environment spends ~30 us on a
        numpy scalar ``grid[(i, j, k)]`` read and 30 ns on a list one."""
        return self.grid.tolist()

    def evaluate(self, **coords: Any) -> Any:
        """The coefficient at ``mach`` / ``alpha_deg`` / ``beta_deg``.

        Axes the table does not tabulate are ignored, so a Mach-only table
        reads fine with all three coordinates given. Coordinates may be
        scalars or arrays of a common broadcast shape; a scalar query returns a
        ``float``, anything else an :class:`numpy.ndarray`.
        """
        query = []
        for axis in self.axes:
            if axis not in coords:
                raise AeroError(f"{self.name}: evaluation needs a value for axis {axis}")
            query.append(coords[axis])
        extra = sorted(set(coords) - set(AERO_AXES))
        if extra:
            raise AeroError(f"{self.name}: unknown coordinate(s) {extra}; axes are {AERO_AXES}")
        if all(getattr(q, "ndim", 0) == 0 for q in query):  # np.ndim costs 7 us here
            return _interp_scalar(self.axes, self._node_lists, self._step_lists,
                                  self._grid_lists, [float(q) for q in query],
                                  self.extrapolation, self.name)
        return _interpolate(self.axes, self.axis_values, self.grid, query, self.extrapolation,
                            self.name)

    @classmethod
    def from_document(cls, name: str, doc: Mapping[str, Any],
                      axes: Mapping[str, Sequence[float]] | None = None) -> CoefficientTable:
        """Build one table from an aero document's ``coefficients:<name>``
        block: ``{axes: [...], values: ..., label: ..., extrapolation: ...,
        provenance: ...}``. Axis nodes come from the block or from the
        document-level ``axes:`` the loader passes in ``axes``."""
        if not isinstance(doc, Mapping):
            raise AeroError(f"coefficient {name}: expected a mapping, got {type(doc).__name__}")
        table_axes = tuple(doc.get("axes") or ())
        if not table_axes:
            raise AeroError(f"coefficient {name}: no 'axes' given")
        nodes: dict[str, tuple[float, ...]] = {}
        for axis in table_axes:
            source = doc.get(axis) or (axes or {}).get(axis)
            if source is None:
                raise AeroError(f"coefficient {name}: no nodes for axis {axis}")
            nodes[axis] = tuple(float(v) for v in source)
        if "values" not in doc:
            raise AeroError(f"coefficient {name}: no 'values'")
        return cls(name=name, axes=table_axes, axis_values=nodes, values=doc["values"],
                   extrapolation=str(doc.get("extrapolation", "clamp")),
                   label=str(doc.get("label", "")),
                   provenance=doc.get("provenance"))

    def to_document(self, inline_axes: bool = True) -> dict[str, Any]:
        """The table as a plain mapping, ready for ``yaml.safe_dump`` -- and
        readable by :meth:`from_document` again."""
        doc: dict[str, Any] = {"axes": list(self.axes),
                               "values": np.asarray(self.values, dtype=float).tolist()}
        if inline_axes:
            for axis in self.axes:
                doc[axis] = list(self.axis_values[axis])
        if self.extrapolation != "clamp":
            doc["extrapolation"] = self.extrapolation
        if self.label:
            doc["label"] = self.label
        if self.provenance:
            doc["provenance"] = dict(self.provenance)
        return doc


def _bracket(nodes: np.ndarray, query: np.ndarray, policy: str, where: str,
             axis: str) -> tuple[np.ndarray, np.ndarray]:
    """Index of the segment left end and the segment fraction for one axis.

    ``clamp`` clips the query into the range first, so the fraction lands in
    ``[0, 1]``; ``linear`` leaves the query alone and keeps the end segment, so
    the fraction runs past the ends; ``error`` refuses a query outside.
    """
    lo, hi = float(nodes[0]), float(nodes[-1])
    q = np.asarray(query, dtype=float)
    if policy == "error" and bool(np.any((q < lo) | (q > hi))):
        raise AeroError(f"{where}: {axis} is outside the table range [{lo}, {hi}] "
                        f"(read asked for {float(np.min(q)):.6g} to {float(np.max(q)):.6g}) "
                        f"and extrapolation='error'")
    q = np.clip(q, lo, hi) if policy == "clamp" else q
    step = np.diff(nodes)
    idx = np.clip(np.searchsorted(nodes, q, side="right") - 1, 0, len(nodes) - 2)
    return idx, (q - nodes[idx]) / step[idx]


def _interp_scalar(axes: tuple[str, ...], nodes: list[list[float]], steps: list[list[float]],
                   grid: list, qvals: Sequence[float], policy: str, where: str) -> float:
    """Scalar fast path of :func:`_interpolate` -- bisect and float arithmetic
    only. The 6-DOF plant hits this thousands of times per run and this
    environment's numpy spends microseconds per tiny array op, so the vector
    path below stays for array queries but scalars never build one."""
    picks: list[int] = []
    fracs: list[float] = []
    for axis, nodes_a, step_a, q in zip(axes, nodes, steps, qvals):
        lo, hi = nodes_a[0], nodes_a[-1]
        if policy == "error" and not lo <= q <= hi:
            raise AeroError(f"{where}: {axis} is outside the table range [{lo}, {hi}] "
                            f"(read asked for {q:.6g}) and extrapolation='error'")
        q = min(max(q, lo), hi) if policy == "clamp" else q
        i = min(max(bisect.bisect_right(nodes_a, q) - 1, 0), len(nodes_a) - 2)
        picks.append(i)
        fracs.append((q - nodes_a[i]) / step_a[i])
    out = 0.0
    for corner in itertools.product((0, 1), repeat=len(picks)):
        weight = 1.0
        node = grid
        for d, bit in enumerate(corner):
            node = node[picks[d] + bit]
            weight *= fracs[d] if bit else 1.0 - fracs[d]
        out += weight * node
    return float(out)


def _interpolate(axes: tuple[str, ...], axis_values: Mapping[str, tuple[float, ...]],
                 grid: np.ndarray, query: Sequence[Any], policy: str,
                 where: str) -> np.ndarray:
    """N-linear interpolation of ``grid`` at ``query`` (one entry per axis),
    broadcasting the queries against each other; always at least 1-D."""
    coords = [np.atleast_1d(np.asarray(q, dtype=float)) for q in query]
    shape = np.broadcast_shapes(*[c.shape for c in coords])
    coords = [np.broadcast_to(c, shape) for c in coords]
    nodes = [np.asarray(tuple(float(v) for v in axis_values[axis]), dtype=float)
             for axis in axes]
    idx: list[np.ndarray] = []
    frac: list[np.ndarray] = []
    for axis, q, n in zip(axes, coords, nodes):
        i, f = _bracket(n, q, policy, where, axis)
        idx.append(i)
        frac.append(f)
    out = np.zeros(shape, dtype=float)
    for corner in itertools.product((0, 1), repeat=len(axes)):
        weight = np.ones(shape, dtype=float)
        picks: list[np.ndarray] = []
        for d, bit in enumerate(corner):
            picks.append(idx[d] + bit)
            weight = weight * (frac[d] if bit else 1.0 - frac[d])
        out += weight * grid[tuple(picks)]
    return out


# --------------------------------------------------------------------------
# angles
# --------------------------------------------------------------------------

def angle_of_attack_deg(v_body: Sequence[float]) -> float:
    """Angle of attack [deg] of a body-axis air-relative velocity:
    ``atan2(v_z, v_x)``, so a body flying nose-up to its flight path reads
    positive (z is down)."""
    v = np.asarray(v_body, dtype=float)
    if v.shape != (3,):
        raise AeroError(f"v_body must have 3 components, got shape {v.shape}")
    u, w = float(v[0]), float(v[2])
    if math.hypot(u, w) < 1e-9:
        return 0.0
    return math.degrees(math.atan2(w, u))


def sideslip_deg(v_body: Sequence[float]) -> float:
    """Sideslip angle [deg] of a body-axis air-relative velocity:
    ``asin(v_y / |v|)``, positive when the vehicle is sliding to its right, so
    a weathervane-stable airframe sees a restoring (negative-``y``) side force
    -- ``cy_beta < 0`` in the sign convention of this module's table."""
    v = np.asarray(v_body, dtype=float)
    if v.shape != (3,):
        raise AeroError(f"v_body must have 3 components, got shape {v.shape}")
    speed = float(np.linalg.norm(v))
    if speed < 1e-9:
        return 0.0
    return math.degrees(math.asin(max(-1.0, min(1.0, float(v[1]) / speed))))


def dynamic_pressure_pa(density_kg_m3: float, air_speed_m_s: float) -> float:
    """``q = 1/2 rho V^2`` -- re-exported from
    :mod:`orbitopt.flightlab.atmosphere` so aero code can read both halves of
    the model from one place."""
    from orbitopt.flightlab.atmosphere import dynamic_pressure_pa as _q

    return _q(density_kg_m3, air_speed_m_s)


# --------------------------------------------------------------------------
# max-Q
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MaxQ:
    """Peak dynamic pressure of a run, with when and where it happened.

    The time is the sample time of the largest recorded ``q``, refined by
    parabolic interpolation through the three samples around it when the peak
    is interior (a fixed-step run sees max-Q to within a step otherwise, and
    the F-02 acceptance wants the *time*, not just the value). ``q_pa`` is the
    recorded sample value, never the parabola's vertex: the reported number is
    always one the integrator actually evaluated."""

    q_pa: float
    t_s: float
    altitude_m: float | None = None
    mach: float | None = None
    interpolated_t_s: bool = False

    @property
    def q_kpa(self) -> float:
        """The same peak in kPa, the unit mission control prints."""
        return self.q_pa / 1000.0


def _peak_time(t: np.ndarray, q: np.ndarray) -> tuple[float, int, bool]:
    i = int(np.argmax(q))
    refined = False
    ti = float(t[i])
    if 0 < i < len(q) - 1:
        y0, y1, y2 = float(q[i - 1]), float(q[i]), float(q[i + 1])
        dt = float(t[i + 1] - t[i]) if float(t[i + 1]) != float(t[i]) else float(t[i] - t[i - 1])
        denom = y0 - 2.0 * y1 + y2
        if dt and abs(denom) > 1e-12 * max(1.0, abs(y1)):
            offset = 0.5 * (y0 - y2) / denom
            if abs(offset) <= 1.0:
                ti = float(t[i]) + offset * dt
                refined = True
    return ti, i, refined


def max_q_of(times_s: Sequence[float], q_bar_pa: Sequence[float], *,
             altitude_m: Sequence[float] | None = None,
             mach: Sequence[float] | None = None) -> MaxQ:
    """Max-Q of a sampled trajectory: the largest recorded dynamic pressure and
    its (parabola-refined) time. Raises on an empty or all-NaN sample set."""
    t = np.asarray(times_s, dtype=float).ravel()
    q = np.asarray(q_bar_pa, dtype=float).ravel()
    if t.shape != q.shape:
        raise AeroError(f"times ({t.shape}) and dynamic pressure ({q.shape}) must match")
    if q.size == 0:
        raise AeroError("no samples to take a max-Q from")
    if not np.any(np.isfinite(q)):
        raise AeroError("no finite dynamic pressure samples")
    q = np.where(np.isfinite(q), q, -np.inf)
    ti, i, refined = _peak_time(t, np.where(np.isfinite(q), q, -np.inf))
    kwargs: dict[str, Any] = {}
    if altitude_m is not None:
        alt = np.asarray(altitude_m, dtype=float).ravel()
        kwargs["altitude_m"] = float(alt[i]) if alt.shape == q.shape and np.isfinite(alt[i]) else None
    if mach is not None:
        m = np.asarray(mach, dtype=float).ravel()
        kwargs["mach"] = float(m[i]) if m.shape == q.shape and np.isfinite(m[i]) else None
    return MaxQ(q_pa=float(np.max(q)), t_s=float(ti), interpolated_t_s=refined, **kwargs)


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AeroModel:
    """The six body-axis coefficient tables of one vehicle, plus the reference
    geometry that turns coefficients into force and moment.

    ``reference_area_m2`` is the usual frontal area; ``reference_span_m`` sizes
    the roll moment and ``reference_length_m`` the pitch and yaw moments (one
    length for both, as booster models do).
    """

    name: str
    reference_area_m2: float
    reference_span_m: float
    reference_length_m: float
    #: The six coefficient tables, keyed by :data:`REQUIRED_COEFFICIENTS`.
    tables: Mapping[str, CoefficientTable]
    #: Default extrapolation for tables that do not state one.
    extrapolation: str = "clamp"
    #: Roll-rate derivative ``cl_p`` [per rad of p b/(2V)]; 0 disables. See the
    #: module docstring -- rate derivatives are not tables (yet).
    roll_damping_cl_p: float = 0.0
    #: Pitch-rate derivative ``cm_q`` [per rad of q c/(2V)].
    pitch_damping_cm_q: float = 0.0
    #: Yaw-rate derivative ``cn_yaw_r`` [per rad of r c/(2V)].
    yaw_damping_cn_yaw_r: float = 0.0
    #: Free-text description.
    label: str = ""
    #: Where the numbers came from; ``provenance["tag"]`` is one of
    #: ``official`` / ``public_estimate`` / ``illustrative``.
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for what, value in (("reference_area_m2", self.reference_area_m2),
                            ("reference_span_m", self.reference_span_m),
                            ("reference_length_m", self.reference_length_m)):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise AeroError(f"{self.name or 'aero model'}: {what} must be positive")
        missing = [key for key in REQUIRED_COEFFICIENTS if key not in self.tables]
        if missing:
            raise AeroError(f"{self.name or 'aero model'}: no table for coefficient(s) {missing}")
        extra = sorted(set(self.tables) - set(REQUIRED_COEFFICIENTS))
        if extra:
            raise AeroError(f"{self.name or 'aero model'}: unknown coefficient(s) {extra}")
        tables = dict(self.tables)
        for key, table in tables.items():
            if table.name != key:
                raise AeroError(f"{key}: table is named {table.name!r}; the model key wins")
            if table.extrapolation == "clamp" and self.extrapolation != "clamp":
                tables[key] = CoefficientTable(
                    name=table.name, axes=table.axes, axis_values=table.axis_values,
                    values=table.values, extrapolation=self.extrapolation,
                    label=table.label, provenance=table.provenance)
        object.__setattr__(self, "tables", tables)
        if self.extrapolation not in EXTRAPOLATION_POLICIES:
            raise AeroError(f"extrapolation {self.extrapolation!r} is not one of "
                            f"{EXTRAPOLATION_POLICIES}")

    # -- reading ----------------------------------------------------------

    @property
    def provenance_tag(self) -> str:
        """``official`` / ``public_estimate`` / ``illustrative``, defaulting to
        ``illustrative``: an untagged model is never read as a claim."""
        return str(self.provenance.get("tag", "illustrative"))

    def coefficients(self, mach: Any, alpha_deg: Any, beta_deg: Any) -> dict[str, Any]:
        """All six coefficients at one (or a broadcast set of) flight state.

        ``alpha_deg`` / ``beta_deg`` are the air-relative angles (see
        :func:`angle_of_attack_deg` / :func:`sideslip_deg`); a zero-speed state
        is a valid query and simply reads its angles as zero.
        """
        out: dict[str, Any] = {}
        for key in REQUIRED_COEFFICIENTS:
            out[key] = self.tables[key].evaluate(mach=mach, alpha_deg=alpha_deg,
                                                 beta_deg=beta_deg)
        return out

    def force_and_moment(self, q_bar_pa: float, mach: float, alpha_deg: float, beta_deg: float,
                         omega_rad_s: Sequence[float] | None = None,
                         speed_of_sound_m_s: float | None = None
                         ) -> tuple[np.ndarray, np.ndarray]:
        """Body-axis aerodynamic force [N] and moment [N m] at a flight state.

        ``q_bar_pa`` is the dynamic pressure the caller computed from the
        air-relative speed (the integrator does that from its atmosphere
        model), ``mach`` the air-relative Mach, and the angles in degrees in
        the conventions of this module. ``omega_rad_s`` (body rates) feeds the
        linear rate damping, which needs the air-relative speed to build its
        dimensionless rates -- that is ``mach * speed_of_sound_m_s``, so
        ``speed_of_sound_m_s`` comes along whenever the rates do. Without
        ``omega_rad_s`` the read is table-only.
        """
        if q_bar_pa < 0.0 or not math.isfinite(q_bar_pa):
            raise AeroError(f"dynamic pressure must be finite and >= 0, got {q_bar_pa}")
        coeff = self.coefficients(float(mach), float(alpha_deg), float(beta_deg))
        force = q_bar_pa * self.reference_area_m2 * np.array([
            -float(coeff["ca"]), float(coeff["cy"]), -float(coeff["cn"])])
        moment = q_bar_pa * self.reference_area_m2 * np.array([
            self.reference_span_m * float(coeff["cl"]),
            self.reference_length_m * float(coeff["cm"]),
            self.reference_length_m * float(coeff["cn_yaw"])])
        if omega_rad_s is None:
            return force, moment
        w = np.asarray(omega_rad_s, dtype=float)
        if w.shape != (3,):
            raise AeroError(f"omega_rad_s must have 3 components, got shape {w.shape}")
        if speed_of_sound_m_s is None or speed_of_sound_m_s <= 0.0:
            raise AeroError("rate damping needs speed_of_sound_m_s to build p/(2V), q/(2V), r/(2V)")
        speed = abs(float(mach)) * float(speed_of_sound_m_s)
        if speed <= 0.0 or q_bar_pa <= 0.0:
            return force, moment
        two_v = 2.0 * speed
        d_roll = self.roll_damping_cl_p * float(w[0]) * self.reference_span_m / two_v
        d_pitch = self.pitch_damping_cm_q * float(w[1]) * self.reference_length_m / two_v
        d_yaw = self.yaw_damping_cn_yaw_r * float(w[2]) * self.reference_length_m / two_v
        added = (q_bar_pa * self.reference_area_m2) * np.array([
            self.reference_span_m * d_roll, self.reference_length_m * d_pitch,
            self.reference_length_m * d_yaw])
        return force, moment + added

    # -- writing / loading -------------------------------------------------

    def to_document(self) -> dict[str, Any]:
        """A plain mapping that :meth:`from_document` reads back exactly."""
        doc: dict[str, Any] = {
            "schema_version": "flight-aero-1.0",
            "kind": "coefficient-tables",
            "name": self.name,
            "reference": {"area_m2": self.reference_area_m2,
                          "span_m": self.reference_span_m,
                          "length_m": self.reference_length_m},
            "coefficients": {key: self.tables[key].to_document(inline_axes=True)
                             for key in REQUIRED_COEFFICIENTS},
        }
        if self.label:
            doc["label"] = self.label
        if self.extrapolation != "clamp":
            doc["extrapolation"] = self.extrapolation
        if any((self.roll_damping_cl_p, self.pitch_damping_cm_q, self.yaw_damping_cn_yaw_r)):
            doc["damping"] = {"cl_p": self.roll_damping_cl_p,
                              "cm_q": self.pitch_damping_cm_q,
                              "cn_yaw_r": self.yaw_damping_cn_yaw_r}
        if self.provenance:
            doc["provenance"] = dict(self.provenance)
        return doc

    @classmethod
    def from_document(cls, doc: Mapping[str, Any]) -> AeroModel:
        """Build a model from a mapping (``yaml.safe_load`` of a
        ``flight-aero-1.0`` document, or built in memory).

        ``kind: axisymmetric`` documents -- a Mach breakout of ``ca``, a
        normal-curve slope, a static margin -- are promoted to full tables on
        the three axes first; ``kind: coefficient-tables`` (the default, and
        what :meth:`to_document` writes) is read table by table."""
        if not isinstance(doc, Mapping):
            raise AeroError(f"aero document must be a mapping, got {type(doc).__name__}")
        validate_aero_document(doc)
        kind = str(doc.get("kind", "coefficient-tables"))
        if kind == "axisymmetric":
            return cls.axisymmetric(**cls._axisymmetric_kwargs(doc))
        if kind != "coefficient-tables":
            raise AeroError(f"unknown aero document kind {kind!r}; expected 'axisymmetric' "
                            f"or 'coefficient-tables'")
        ref = doc.get("reference")
        if not isinstance(ref, Mapping):
            raise AeroError("aero document needs a 'reference' mapping (area_m2, span_m, length_m)")
        shared_axes = {axis: doc[axis] for axis in AERO_AXES if axis in doc}
        tables_doc = doc.get("coefficients")
        if not isinstance(tables_doc, Mapping):
            raise AeroError("aero document needs a 'coefficients' mapping")
        default_policy = str(doc.get("extrapolation", "clamp"))
        tables: dict[str, CoefficientTable] = {}
        for key in REQUIRED_COEFFICIENTS:
            if key not in tables_doc:
                raise AeroError(f"aero document {doc.get('name', '?')}: no table for {key}")
            tables[key] = CoefficientTable.from_document(key, tables_doc[key], shared_axes)
        damping = doc.get("damping") or {}
        name = str(doc.get("name", "aero"))
        try:
            area = float(ref["area_m2"])
            length = float(ref["length_m"])
        except KeyError as exc:
            raise AeroError(f"aero document {doc.get('name', '?')}: 'reference' is missing "
                            f"{exc}") from exc
        return cls(name=name,
                   reference_area_m2=area,
                   reference_span_m=float(ref.get("span_m", length)),
                   reference_length_m=length,
                   tables=tables, extrapolation=default_policy,
                   roll_damping_cl_p=float(damping.get("cl_p", 0.0)),
                   pitch_damping_cm_q=float(damping.get("cm_q", 0.0)),
                   yaw_damping_cn_yaw_r=float(damping.get("cn_yaw_r", 0.0)),
                   label=str(doc.get("label", "")),
                   provenance=dict(doc.get("provenance") or {}))

    @staticmethod
    def _axisymmetric_kwargs(doc: Mapping[str, Any]) -> dict[str, Any]:
        ref = doc.get("reference")
        if not isinstance(ref, Mapping):
            raise AeroError("axisymmetric aero document needs a 'reference' mapping")
        ca = doc.get("ca")
        if not isinstance(ca, (list, tuple)) or not ca:
            raise AeroError("axisymmetric aero document needs a 'ca' Mach breakout")
        try:
            mach_nodes = [float(pair[0]) for pair in ca]
            ca_nodes = [float(pair[1]) for pair in ca]
        except (TypeError, ValueError, IndexError) as exc:
            raise AeroError(f"axisymmetric 'ca' must be [[mach, ca], ...] pairs: {exc}") from exc
        if any(b <= a for a, b in zip(mach_nodes, mach_nodes[1:])):
            raise AeroError(f"axisymmetric 'ca' Mach nodes must increase strictly: {mach_nodes}")
        return dict(name=str(doc.get("name", "axisymmetric-aero")),
                    diameter_m=ref.get("diameter_m"),
                    reference_area_m2=ref.get("area_m2"),
                    reference_length_m=ref.get("length_m", 1.0),
                    reference_span_m=ref.get("span_m"),
                    ca_vs_mach=(mach_nodes, ca_nodes),
                    cn_alpha_per_rad=float(doc["cn_alpha_per_rad"]),
                    static_margin=float(doc.get("static_margin", 0.0)),
                    mach_nodes=doc.get("mach_nodes"),
                    alpha_nodes=doc.get("alpha_deg_nodes"),
                    beta_nodes=doc.get("beta_deg_nodes"),
                    extrapolation=str(doc.get("extrapolation", "clamp")),
                    roll_damping_cl_p=float((doc.get("damping") or {}).get("cl_p", 0.0)),
                    pitch_damping_cm_q=float((doc.get("damping") or {}).get("cm_q", 0.0)),
                    yaw_damping_cn_yaw_r=float((doc.get("damping") or {}).get("cn_yaw_r", 0.0)),
                    label=str(doc.get("label", "")),
                    provenance=dict(doc.get("provenance") or {}))

    @classmethod
    def axisymmetric(cls, *, name: str, ca_vs_mach: tuple[Sequence[float], Sequence[float]],
                     cn_alpha_per_rad: float, static_margin: float = 0.0,
                     reference_area_m2: float | None = None, diameter_m: float | None = None,
                     reference_length_m: float = 1.0, reference_span_m: float | None = None,
                     mach_nodes: Sequence[float] | None = None,
                     alpha_nodes: Sequence[float] | None = None,
                     beta_nodes: Sequence[float] | None = None,
                     extrapolation: str = "clamp",
                     roll_damping_cl_p: float = 0.0, pitch_damping_cm_q: float = 0.0,
                     yaw_damping_cn_yaw_r: float = 0.0, label: str = "",
                     provenance: Mapping[str, Any] | None = None) -> AeroModel:
        """Promote the axisymmetric description the flight-vehicle schema
        carries (a ``ca`` breakout in Mach, a normal-curve slope per radian, a
        static margin) into full tables on Mach, AoA and sideslip.

        The generated tables are, with ``a`` the AoA and ``b`` the sideslip in
        **radians**, ``S`` the reference area, ``c`` the reference length:

        * ``cn = cn_alpha * a`` -- linear normal curve, slope constant in Mach;
        * ``cy = -cn_alpha * b`` -- the same slope restoring a sideslip;
        * ``ca = ca_mach(M) + cn_alpha * (a^2 + b^2)`` -- tabulated wave/pressure
          drag plus the induced drag ``L . a`` of that normal force;
        * ``cm = -static_margin * cn_alpha * a`` -- a positive ``static_margin``
          is weathervane stable (``cm_alpha < 0``);
        * ``cl = 0``, ``cn_yaw = -static_margin * cn_alpha * b`` -- the sideslip
          restoring yaw of an axisymmetric body at the same margin; roll is
          left to :attr:`roll_damping_cl_p`.

        This is an *illustrative* model shape, and the packaged instances say
        so: it is the linear theory a hand-built booster gets from a diameter
        and a curve slope, not a CFD reduction. Transonic normal-force growth,
        fin stall and grid fins are all outside it (F-05 will put real curves
        in real tables).
        """
        if reference_area_m2 is None:
            if diameter_m is None:
                raise AeroError("axisymmetric aero needs reference_area_m2 or diameter_m")
            reference_area_m2 = math.pi * float(diameter_m) ** 2 / 4.0
        if diameter_m is not None and reference_span_m is None:
            reference_span_m = float(diameter_m)
        if reference_span_m is None:
            reference_span_m = reference_length_m
        mach = np.asarray(mach_nodes if mach_nodes is not None else ca_vs_mach[0], dtype=float)
        ca_m = np.interp(mach, np.asarray(ca_vs_mach[0], dtype=float),
                         np.asarray(ca_vs_mach[1], dtype=float))
        alpha = np.asarray(alpha_nodes if alpha_nodes is not None
                           else (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0), dtype=float)
        beta = np.asarray(beta_nodes if beta_nodes is not None
                          else (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0), dtype=float)
        a = np.radians(alpha)[None, :, None]
        b = np.radians(beta)[None, None, :]
        slope = float(cn_alpha_per_rad)
        grid_ca = ca_m[:, None, None] + slope * (a ** 2 + b ** 2)
        grid_cn = slope * np.broadcast_to(a, (len(mach), len(alpha), len(beta)))
        grid_cy = -slope * np.broadcast_to(b, (len(mach), len(alpha), len(beta)))
        grid_cm = -float(static_margin) * slope * np.broadcast_to(a, grid_ca.shape)
        grid_yaw = -float(static_margin) * slope * np.broadcast_to(b, grid_ca.shape)
        grid_zero = np.zeros(grid_ca.shape)
        nodes = {"mach": mach, "alpha_deg": alpha, "beta_deg": beta}
        made = {"ca": ("Axial force coefficient (drag positive)", grid_ca),
                "cn": ("Normal force coefficient", grid_cn),
                "cy": ("Side force coefficient", grid_cy),
                "cl": ("Roll moment coefficient", grid_zero),
                "cm": ("Pitch moment coefficient", grid_cm),
                "cn_yaw": ("Yaw moment coefficient", grid_yaw)}
        tables = {key: CoefficientTable(
            name=key, axes=("mach", "alpha_deg", "beta_deg"), axis_values=nodes,
            values=entry[1].tolist(), extrapolation=extrapolation, label=entry[0],
            provenance=dict(provenance or {}))
            for key, entry in made.items()}
        return cls(name=name, reference_area_m2=float(reference_area_m2),
                   reference_span_m=float(reference_span_m),
                   reference_length_m=float(reference_length_m), tables=tables,
                   extrapolation=extrapolation, roll_damping_cl_p=roll_damping_cl_p,
                   pitch_damping_cm_q=pitch_damping_cm_q,
                   yaw_damping_cn_yaw_r=yaw_damping_cn_yaw_r, label=label,
                   provenance=dict(provenance or {}))

    def write(self, path: str | Path) -> Path:
        """Write the model as a ``flight-aero-1.0`` YAML document."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(self.to_document(), sort_keys=False, width=79,
                                        default_flow_style=None), encoding="utf-8")
        return target


def load_aero_model(path: str | Path) -> AeroModel:
    """Read a ``flight-aero-1.0`` aero document (YAML or JSON) from disk."""
    text = Path(path).read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    if not isinstance(doc, Mapping):
        raise AeroError(f"{path}: expected a mapping, got {type(doc).__name__}")
    return AeroModel.from_document(doc)


def load_packaged_aero(name: str) -> AeroModel:
    """Load one of the packaged aero models by bare name (``spike-booster`` ->
    ``data/flightlab/aero/spike-booster.yaml``), the same resource lookup as
    the vehicle catalog."""
    return load_aero_model(resources.files("orbitopt").joinpath("data", "flightlab", "aero",
                                                                f"{name}.yaml"))
