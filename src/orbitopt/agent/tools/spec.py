"""What a scientific tool declares: its schemas, units, assumptions, validity domain, cost and tests.

A ``ToolSpec`` is pure data. ``spec_errors`` lists every way a spec breaks
the registry contract; ``ToolRegistry.register`` refuses a spec with any.
Every failure the registry reports is a ``ToolError`` whose ``code`` is a
``ToolErrorCode``; callers branch on the code, never on the message.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal, Mapping

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from orbitopt.optimization import Expression, ExpressionError, UnitError, parse_unit

TOOL_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*\.v[1-9][0-9]*$")
"""Dotted lower-case names ending in a version segment, such as ``math.erf.v1``.

Every id that matches is also a valid IR ``ToolCall.tool_id``. A new version
of a tool is a new id; a registered id never changes meaning.
"""

COST_CLASSES = ("trivial", "cheap", "moderate", "expensive")
"""Relative cost per call: under 1 ms, under 10 ms, under 1 s, more than that."""

NUMERIC_TYPES = ("number", "integer")


class ToolErrorCode(str, Enum):
    """Why the registry refused a spec or a call."""

    INVALID_SPEC = "invalid_spec"
    DUPLICATE_TOOL = "duplicate_tool"
    SIGNATURE_MISMATCH = "signature_mismatch"
    UNKNOWN_TOOL = "unknown_tool"
    NOT_ALLOWED = "not_allowed"
    INVALID_ARGS = "invalid_args"
    OUT_OF_DOMAIN = "out_of_domain"
    INVALID_OUTPUT = "invalid_output"
    TOOL_FAILED = "tool_failed"
    BUDGET_EXCEEDED = "budget_exceeded"
    TIMEOUT = "timeout"
    UNIT_MISMATCH = "unit_mismatch"


REFUSALS = frozenset({ToolErrorCode.UNKNOWN_TOOL, ToolErrorCode.NOT_ALLOWED,
                      ToolErrorCode.BUDGET_EXCEEDED, ToolErrorCode.TIMEOUT})
"""Call outcomes decided by policy (allowlist, budget), not by the tool and its arguments.

A refused call is logged but never replayed: replaying it would test the
policy of the session that made it, not the tool.
"""


class ToolError(Exception):
    """A refused spec or call; ``code`` says why and ``tool_id`` names the tool involved."""

    def __init__(self, code: ToolErrorCode, tool_id: str, message: str):
        self.code = ToolErrorCode(code)
        self.tool_id = tool_id
        super().__init__(f"{tool_id}: [{self.code.value}] {message}")


class DomainError(ValueError):
    """Raised by a tool for valid-looking input it refuses to compute, such as ``a >= b``.

    Use it for preconditions that ``ToolSpec.domain`` intervals cannot
    express. The registry reports it as ``ToolErrorCode.OUT_OF_DOMAIN``.
    """


@dataclass(frozen=True)
class Interval:
    """A closed range of an input; ``None`` leaves that side open. Arrays are checked element by element."""

    lower: float | None = None
    upper: float | None = None

    def contains(self, value: float) -> bool:
        return (self.lower is None or value >= self.lower) and (self.upper is None or value <= self.upper)


@dataclass(frozen=True)
class CostModel:
    """Declared cost of one call.

    ``evaluations`` is an IR expression over the numeric inputs giving the
    expected number of evaluations of the underlying model (integrand
    evaluations, time steps), such as ``"3*n"``. It is what a caller's
    ``CallBudget.max_evaluations`` is charged.
    """

    cost_class: Literal["trivial", "cheap", "moderate", "expensive"]
    evaluations: str = "1"


@dataclass(frozen=True)
class ToolSpec:
    """The contract of one scientific tool.

    ``input_schema`` and ``output_schema`` are JSON Schema objects. Every
    numeric field (``number`` or ``integer``, or an array of them) has an
    SI-coherent unit in ``input_units`` or ``output_units``, ``""`` for
    dimensionless; an array's unit applies to each element. An IR
    ``ToolCall`` reads one scalar numeric output, which it names in
    ``output``. ``domain`` declares the input ranges the tool was verified on;
    a call outside them is refused, never extrapolated. A stochastic tool
    (``deterministic=False``) must take a required integer ``seed``.
    ``replay_tolerance`` is the relative tolerance replay allows (``0``
    means the canonical results must be identical). ``tested_by`` names the
    tests that verify the tool against an independent reference.
    """

    tool_id: str
    description: str
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    input_units: Mapping[str, str]
    output_units: Mapping[str, str]
    assumptions: tuple[str, ...]
    domain: Mapping[str, Interval]
    cost: CostModel
    tested_by: tuple[str, ...]
    deterministic: bool = True
    replay_tolerance: float = 0.0

    def to_dict(self) -> dict:
        """JSON-ready form, for an agent prompt or a provider tool list."""
        return {
            "tool_id": self.tool_id, "description": self.description,
            "input_schema": dict(self.input_schema), "output_schema": dict(self.output_schema),
            "input_units": dict(self.input_units), "output_units": dict(self.output_units),
            "assumptions": list(self.assumptions),
            "domain": {k: {"lower": v.lower, "upper": v.upper} for k, v in self.domain.items()},
            "cost": {"cost_class": self.cost.cost_class, "evaluations": self.cost.evaluations},
            "deterministic": self.deterministic,
            "replay_tolerance": self.replay_tolerance, "tested_by": list(self.tested_by),
        }


@dataclass(frozen=True)
class Tool:
    """A spec and the pure Python callable that implements it.

    ``fn`` takes the inputs as keyword arguments and returns a dict that
    satisfies ``output_schema``.
    """

    spec: ToolSpec
    fn: Callable[..., Mapping[str, Any]] = field(repr=False)


def numeric_fields(schema: Mapping[str, Any]) -> dict[str, bool]:
    """Numeric properties of an object schema, mapped to whether each is an array of numbers."""
    found = {}
    for name, prop in schema.get("properties", {}).items():
        if prop.get("type") in NUMERIC_TYPES:
            found[name] = False
        elif prop.get("type") == "array" and prop.get("items", {}).get("type") in NUMERIC_TYPES:
            found[name] = True
    return found


def _schema_errors(schema: Any, what: str, closed: bool) -> list[str]:
    if not isinstance(schema, Mapping) or schema.get("type") != "object":
        return [f"{what} must be a JSON Schema with type 'object'"]
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        return [f"{what} is not a valid JSON Schema: {exc.message}"]
    errors = []
    if closed and schema.get("additionalProperties") is not False:
        errors.append(f"{what} must set additionalProperties to false")
    unknown = set(schema.get("required", ())) - set(schema.get("properties", {}))
    if unknown:
        errors.append(f"{what} requires undeclared properties {sorted(unknown)}")
    return errors


def _unit_errors(units: Mapping[str, str], schema: Mapping[str, Any], what: str) -> list[str]:
    errors = []
    numeric = numeric_fields(schema)
    for name in sorted(numeric.keys() - units.keys()):
        errors.append(f"numeric {what} field {name!r} has no unit")
    for name, text in sorted(units.items()):
        if name not in numeric:
            errors.append(f"{what} unit given for {name!r}, which is not a numeric {what} field")
            continue
        try:
            unit = parse_unit(text)
        except UnitError as exc:
            errors.append(f"{what} field {name!r}: {exc}")
            continue
        if not math.isclose(unit.scale, 1.0, rel_tol=1e-12):
            errors.append(f"{what} field {name!r} unit {text!r} is not SI-coherent (scale {unit.scale:g})")
    return errors


def spec_errors(spec: ToolSpec) -> list[str]:
    """Every way ``spec`` breaks the registry contract; empty when it is sound."""
    if not isinstance(spec.tool_id, str) or not TOOL_ID_PATTERN.match(spec.tool_id):
        return [f"tool id {spec.tool_id!r} must be dotted lower-case names ending in a version, like 'math.erf.v1'"]
    errors = _schema_errors(spec.input_schema, "input_schema", closed=True)
    errors += _schema_errors(spec.output_schema, "output_schema", closed=False)
    if errors:
        return errors
    if not (isinstance(spec.description, str) and spec.description.strip()):
        errors.append("a tool needs a description")
    errors += _unit_errors(spec.input_units, spec.input_schema, "input")
    errors += _unit_errors(spec.output_units, spec.output_schema, "output")

    inputs = numeric_fields(spec.input_schema)
    for name, interval in spec.domain.items():
        if name not in inputs:
            errors.append(f"domain given for {name!r}, which is not a numeric input")
        elif interval.lower is not None and interval.upper is not None and interval.lower > interval.upper:
            errors.append(f"domain of {name!r} has lower {interval.lower} above upper {interval.upper}")

    if spec.cost.cost_class not in COST_CLASSES:
        errors.append(f"cost class {spec.cost.cost_class!r} is not one of {list(COST_CLASSES)}")
    try:
        names = Expression(spec.cost.evaluations).names
        scalars = {name for name, is_array in inputs.items()
                   if not is_array and name in spec.input_schema.get("required", ())}
        if not names <= scalars:
            errors.append(f"cost evaluations {spec.cost.evaluations!r} reads {sorted(names - scalars)}, "
                          "which are not required scalar inputs")
    except ExpressionError as exc:
        errors.append(f"cost evaluations {spec.cost.evaluations!r}: {exc}")

    if not spec.deterministic:
        seed = spec.input_schema.get("properties", {}).get("seed", {})
        if seed.get("type") != "integer" or "seed" not in spec.input_schema.get("required", ()):
            errors.append("a stochastic tool must take a required integer 'seed'")
    if not (isinstance(spec.replay_tolerance, (int, float)) and math.isfinite(spec.replay_tolerance)
            and spec.replay_tolerance >= 0):
        errors.append("replay_tolerance must be finite and non-negative")
    if not spec.assumptions or not all(isinstance(a, str) and a.strip() for a in spec.assumptions):
        errors.append("a tool must state its model assumptions")
    if not spec.tested_by:
        errors.append("a tool must name the tests that verify it (tested_by)")
    return errors
