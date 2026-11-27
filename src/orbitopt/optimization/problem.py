"""The optimization problem IR: variables, parameters, objectives, hard constraints.

The canonical mathematics is ``min f(x; theta)`` subject to ``g_i(x; theta) <= 0``
and ``h_j(x; theta) = 0``, but the IR stores what the user said: an objective
keeps its ``sense`` and a constraint keeps ``lhs <sense> rhs``. Compilers
normalise downstream.

Construction never validates, so a malformed problem (for example one an LLM
proposed) can still be built and asked for all its errors at once:
``validate_problem`` returns them as typed ``ProblemError`` values, and
``parse_problem`` is the boundary for untrusted dict input that raises one
``InvalidProblemError`` carrying all of them. The JSON Schema
``schemas/optimization-problem-1.1.json`` checks structure only (types,
required fields, enumerations); every value rule lives in ``validate_problem``
so it reports one specific ``ErrorCode`` whether the problem came from Python
or JSON. A 1.0 document still parses: it is checked against the 1.0 schema
and its missing 1.1 fields take their defaults (see ``docs/optimization/ir.md``).
"""

from __future__ import annotations

import hashlib
import json
import keyword
import math
import numbers
from dataclasses import dataclass, field, fields
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from jsonschema import Draft202012Validator

from .errors import ErrorCode, InvalidProblemError, ProblemError
from .expressions import (RESERVED_NAMES, TOOL_ID_PATTERN, EvaluationError, Expression, ExpressionError, ToolCall,
                          ToolCallError)
from .units import Unit, UnitError, format_dims, parse_unit

SCHEMA_VERSION = "1.1"
PROBLEM_SCHEMA = "optimization-problem-1.1.json"
PROBLEM_SCHEMAS = {"1.0": "optimization-problem-1.0.json", SCHEMA_VERSION: PROBLEM_SCHEMA}
"""The problem schema for each ``schemaVersion`` that still parses; ``to_dict`` writes the newest."""
VARIABLE_KINDS = ("continuous", "integer", "binary", "discrete")
OBJECTIVE_SENSES = ("minimize", "maximize")
CONSTRAINT_SENSES = ("<=", ">=", "==")
ORIGINS = ("user", "compiler", "agent")
STATUSES = ("solver_converged", "constraints_satisfied", "independently_verified", "optimality_certified")
"""The four result statuses; ``OptimizationResult`` has one boolean field per name."""
INDEPENDENT_METHODS = ("analytical", "reference_solution", "alternate_solver")
DIGEST_SIGNIFICANT_DIGITS = 12

VariableKind = Literal["continuous", "integer", "binary", "discrete"]
Origin = Literal["user", "compiler", "agent"]
DomainViolation = Literal["below_lower", "above_upper", "not_integer", "not_a_choice"]
DOMAIN_VIOLATIONS = ("below_lower", "above_upper", "not_integer", "not_a_choice")
"""Every reason ``Variable.domain_violations`` can give."""


def tolerance_scale(reference: float) -> float:
    """What ``constraint_tol`` is relative to for a right-hand side or bound: ``|reference|``, or 1 when it is 0.

    A constraint ``g <sense> rhs`` holds when its violation is at most
    ``constraint_tol * tolerance_scale(rhs)``, and a bound or discrete choice
    the same way. IR 1.0 used ``max(1, |rhs|)``, which let a tiny right-hand
    side drown in the tolerance: ``error <= 1e-10`` accepted errors up to
    ``1e-6``. Under this rule it accepts up to ``1e-10 * (1 + 1e-6)``. A zero
    right-hand side carries no magnitude, so its scale stays 1.
    """
    return abs(reference) or 1.0


def _set(obj: Any, **values: Any) -> None:
    for name, value in values.items():
        object.__setattr__(obj, name, value)


def _expression(value: Any) -> Any:
    return Expression(value) if isinstance(value, str) else value


@dataclass(frozen=True)
class Variable:
    """A decision variable and its domain.

    ``continuous`` and ``integer`` use ``lower``/``upper`` (``None`` is an open
    side); ``binary`` is {0, 1} and sets neither; ``discrete`` takes one of
    ``choices``. ``unit`` applies to bounds, choices and ``initial``.
    """

    name: str
    kind: VariableKind = "continuous"
    lower: float | None = None
    upper: float | None = None
    choices: tuple[float, ...] = ()
    unit: str = ""
    initial: float | None = None
    description: str = ""

    def __post_init__(self):
        if isinstance(self.choices, list):
            _set(self, choices=tuple(self.choices))

    def domain_violations(self, value: float, tol: float) -> tuple[DomainViolation, ...]:
        """Why a finite ``value`` lies outside this variable's domain; empty when it lies inside.

        A bound or choice ``r`` is met within ``tol * tolerance_scale(r)``,
        integrality within ``tol``; a binary variable is an integer in
        [0, 1]. This is the per-variable half of ``constraints_satisfied``.
        """
        if self.kind == "discrete":
            met = any(abs(value - c) <= tol * tolerance_scale(c) for c in self.choices)
            return () if met else ("not_a_choice",)
        lower, upper = (0.0, 1.0) if self.kind == "binary" else (self.lower, self.upper)
        found: list[DomainViolation] = []
        if lower is not None and value < lower - tol * tolerance_scale(lower):
            found.append("below_lower")
        if upper is not None and value > upper + tol * tolerance_scale(upper):
            found.append("above_upper")
        if self.kind in ("integer", "binary") and abs(value - round(value)) > tol:
            found.append("not_integer")
        return tuple(found)


@dataclass(frozen=True)
class Parameter:
    """A fixed named constant (theta) with a unit."""

    name: str
    value: float
    unit: str = ""
    description: str = ""


@dataclass(frozen=True)
class Objective:
    """One objective; several objectives make the problem multi-objective.

    ``unit`` is the unit the value is reported in and must match the
    expression's dimension. A plain string expression is wrapped in
    ``Expression``.
    """

    name: str
    expression: Expression | ToolCall
    sense: Literal["minimize", "maximize"] = "minimize"
    unit: str = ""
    origin: Origin = "user"
    description: str = ""

    def __post_init__(self):
        _set(self, expression=_expression(self.expression))


@dataclass(frozen=True)
class Constraint:
    """A hard constraint ``expression <sense> rhs``, both sides in ``unit``.

    ``locked`` defaults to ``origin == "user"``; only a user constraint may be
    locked, and locked constraints are what ``hard_constraint_fingerprint``
    protects. A plain string expression is wrapped in ``Expression``.
    """

    name: str
    expression: Expression | ToolCall
    sense: Literal["<=", ">=", "=="]
    rhs: float
    unit: str = ""
    origin: Origin = "user"
    locked: bool | None = None
    description: str = ""

    def __post_init__(self):
        _set(self, expression=_expression(self.expression))
        if self.locked is None:
            _set(self, locked=self.origin == "user")

    def residual(self, lhs: Any) -> Any:
        """Canonical residual: ``g <= 0`` for an inequality, ``h = 0`` for an equality.

        ``lhs - rhs`` for ``<=`` and ``==``, ``rhs - lhs`` for ``>=``; a
        positive inequality residual is a violation.
        """
        return self.rhs - lhs if self.sense == ">=" else lhs - self.rhs

    def violation(self, residual: float) -> float:
        """How far a canonical residual breaks the constraint: ``|h|`` for an equality, ``g`` otherwise."""
        return abs(residual) if self.sense == "==" else residual

    def allowance(self, tol: float) -> float:
        """The largest violation that still satisfies the constraint: ``tol * tolerance_scale(rhs)``."""
        return tol * tolerance_scale(self.rhs)


@dataclass(frozen=True)
class Budget:
    """Resource limits; each is optional, and positive when present.

    ``max_evaluations`` counts evaluations of the problem and ``max_wall_s``
    is wall-clock seconds; every solver adapter enforces both.
    ``max_iterations`` is the agent runtime's loop cap (planner proposals,
    accepted or refused); no solver reads it, and a backend's own iteration
    limit is ``Solver.solve(..., max_backend_iterations=...)``.
    ``max_tokens`` counts LLM tokens in the agent runtime.
    """

    max_evaluations: int | None = None
    max_iterations: int | None = None
    max_wall_s: float | None = None
    max_tokens: int | None = None


@dataclass(frozen=True)
class VerificationPolicy:
    """Which result statuses must be true for acceptance, and the thresholds behind them.

    ``constraint_tol``: a constraint or bound holds when its violation is at
    most ``constraint_tol * tolerance_scale(rhs or bound)``, that is relative
    to ``|rhs|``, or absolute when ``rhs`` is 0. ``objective_rtol``: an
    independent check agrees when ``|f - f_ref| <= objective_rtol * max(1, |f_ref|)``.
    ``optimality_gap``: the relative gap an optimality certificate must close.
    ``independent_methods`` are the methods that may set ``independently_verified``.
    """

    required: tuple[str, ...] = ("solver_converged", "constraints_satisfied", "independently_verified")
    independent_methods: tuple[str, ...] = INDEPENDENT_METHODS
    constraint_tol: float = 1e-6
    objective_rtol: float = 1e-6
    optimality_gap: float = 1e-4

    def __post_init__(self):
        _set(self, required=tuple(self.required), independent_methods=tuple(self.independent_methods))


@dataclass(frozen=True)
class Evaluation:
    """Objective values in their declared units and canonical constraint residuals, by name."""

    objectives: dict[str, Any]
    residuals: dict[str, Any]


@dataclass(frozen=True, kw_only=True)
class OptimizationProblem:
    """A complete optimization problem: the contract between NL, solvers, tools, agent and guard."""

    id: str
    title: str = ""
    description: str = ""
    variables: tuple[Variable, ...]
    parameters: tuple[Parameter, ...] = ()
    objectives: tuple[Objective, ...]
    constraints: tuple[Constraint, ...] = ()
    budget: Budget = Budget()
    verification: VerificationPolicy = VerificationPolicy()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        _set(self, **{name: tuple(getattr(self, name))
                      for name in ("variables", "parameters", "objectives", "constraints")})

    def to_dict(self) -> dict:
        """The JSON-ready dict form; every field is written, defaults included."""
        return {"schemaVersion": SCHEMA_VERSION, **to_plain(self)}

    @classmethod
    def from_dict(cls, data: Mapping) -> "OptimizationProblem":
        """Inverse of ``to_dict`` for schema-valid input; use ``parse_problem`` for untrusted input."""

        def items(key, build):
            return tuple(build(**item) for item in data.get(key, ()))

        def with_expression(build):
            return lambda **item: build(**{**item, "expression": _expression_from(item["expression"])})

        return cls(
            id=data["id"],
            title=data.get("title", ""),
            description=data.get("description", ""),
            variables=items("variables", Variable),
            parameters=items("parameters", Parameter),
            objectives=items("objectives", with_expression(Objective)),
            constraints=items("constraints", with_expression(Constraint)),
            budget=Budget(**data.get("budget", {})),
            verification=VerificationPolicy(**data.get("verification", {})),
            metadata=dict(data.get("metadata", {})),
        )

    def digest(self) -> str:
        """SHA-256 of every semantic field; ``title``, ``description``s and ``metadata`` are excluded.

        Order of variables, objectives and constraints is part of the problem
        (it fixes solver vector layouts) and therefore of the digest. See
        ``canonical_digest`` for the float quantisation.
        """
        data = self.to_dict()
        for key in ("title", "description", "metadata"):
            data.pop(key)
        for key in ("variables", "parameters", "objectives", "constraints"):
            data[key] = [_without(item, "description") for item in data[key]]
        return canonical_digest(data)

    def hard_constraint_fingerprint(self) -> str:
        """Digest of what an agent must not weaken: user objectives, locked constraints, domains, parameters.

        Covers user-origin objectives, locked constraints (always user-origin),
        every variable's domain (kind, bounds, choices, unit) and every
        parameter's value and unit: widening a bound or raising a parameter a
        locked constraint reads weakens it as surely as editing its text.
        Excludes initial guesses, budget, verification, descriptions, metadata
        and compiler/agent items, so adding a tightening agent constraint keeps
        the fingerprint. Order-insensitive (sorted by name).
        """

        def entries(items, keep, drop=("description",)):
            return sorted((_without(to_plain(item), *drop) for item in items if keep(item)), key=lambda d: d["name"])

        return canonical_digest({
            "objectives": entries(self.objectives, lambda o: o.origin == "user"),
            "constraints": entries(self.constraints, lambda c: c.locked),
            "variables": entries(self.variables, lambda v: True, ("description", "initial")),
            "parameters": entries(self.parameters, lambda p: True),
        })

    def evaluate(self, x: Mapping[str, Any], tools: Mapping[str, Callable[..., Any]] | None = None) -> Evaluation:
        """Evaluate every objective and constraint at ``x`` (values in each variable's unit).

        Names are converted to SI-coherent values before evaluation, so
        ``x [km] + y [m]`` is correct; each result is converted back to its
        objective or constraint unit. ``tools`` maps a ``ToolCall.tool_id`` to
        a callable taking the evaluated arguments and the constants as
        keywords (SI in, SI out) and returning a number, or an object from
        which ``ToolCall.output`` is read. Values may be floats or numpy
        arrays. Raises ``EvaluationError``, and ``ToolCallError`` (its
        subclass) for anything that goes wrong in or around a tool. Assumes a
        valid problem.
        """
        missing = {v.name for v in self.variables} - x.keys()
        if missing:
            raise EvaluationError(f"no value for variables {sorted(missing)}")
        si = {v.name: x[v.name] * parse_unit(v.unit).scale for v in self.variables}
        si.update({p.name: p.value * parse_unit(p.unit).scale for p in self.parameters})

        def value(item):
            expr, scale = item.expression, parse_unit(item.unit).scale
            if not isinstance(expr, ToolCall):
                return expr.evaluate(si) / scale
            raw = _call(expr, si, tools)
            try:
                return raw / scale
            except TypeError:
                raise ToolCallError(expr.tool_id, f"returned {type(raw).__name__}, not a number") from None

        return Evaluation(objectives={o.name: value(o) for o in self.objectives},
                          residuals={c.name: c.residual(value(c)) for c in self.constraints})

    def constraints_satisfied(self, x: Mapping[str, float], residuals: Mapping[str, float]) -> bool:
        """Whether scalar ``x`` lies in every variable domain and every residual is within tolerance.

        This is the one definition of the ``constraints_satisfied`` status,
        using ``verification.constraint_tol`` (see ``VerificationPolicy``).
        """
        tol = self.verification.constraint_tol
        for v in self.variables:
            value = x.get(v.name)
            if not (_number(value) and math.isfinite(value)) or v.domain_violations(value, tol):
                return False
        for c in self.constraints:
            r = residuals.get(c.name)
            if r is None or not math.isfinite(r) or c.violation(r) > c.allowance(tol):
                return False
        return True


def _call(call: ToolCall, si: Mapping[str, Any], tools: Mapping[str, Callable[..., Any]] | None) -> Any:
    """The value of one tool reference: the tool's number, or its ``output`` field."""
    tool = (tools or {}).get(call.tool_id)
    if tool is None:
        raise ToolCallError(call.tool_id, "no tool is registered for this id")
    kwargs = {arg: Expression(src).evaluate(si) for arg, src in call.args.items()}
    try:
        raw = tool(**kwargs, **call.constants)
    except Exception as exc:  # a tool is third-party code: whatever it raises, the point cannot be evaluated
        raise ToolCallError(call.tool_id, f"raised {type(exc).__name__}: {exc}") from exc
    if call.output is None:
        if isinstance(raw, Mapping):
            raise ToolCallError(call.tool_id, f"returned the outputs {sorted(raw)}; the call names none of them")
        return raw
    if not isinstance(raw, Mapping) or call.output not in raw:
        raise ToolCallError(call.tool_id, f"returned no output {call.output!r}")
    return raw[call.output]


def _expression_from(data: Any) -> Expression | ToolCall:
    if isinstance(data, Mapping):
        return ToolCall(data["tool_id"], dict(data.get("args", {})), dict(data.get("constants", {})),
                        data.get("output"))
    return Expression(data)


def to_plain(obj: Any) -> Any:
    """IR objects as JSON-ready values: dataclasses as dicts, expressions as a string or a tool object.

    This is the form ``to_dict``, the digest and the hard-constraint
    fingerprint are built from.
    """
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Expression):
        return obj.source
    if isinstance(obj, ToolCall):
        return {"tool_id": obj.tool_id, "args": dict(obj.args), "constants": to_plain(dict(obj.constants)),
                "output": obj.output}
    if hasattr(obj, "__dataclass_fields__"):
        return {f.name: to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_plain(item) for item in obj]
    if isinstance(obj, Mapping):
        return {key: to_plain(value) for key, value in obj.items()}
    return obj


def _without(data: dict, *keys: str) -> dict:
    return {k: v for k, v in data.items() if k not in keys}


def _quantise(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        value = float(value)
        if not math.isfinite(value):
            return repr(value)
        return float(f"{value:.{DIGEST_SIGNIFICANT_DIGITS}g}") + 0.0
    if isinstance(value, Mapping):
        return {str(k): _quantise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_quantise(v) for v in value]
    raise TypeError(f"cannot digest {type(value).__name__}")


def canonical_digest(data: Any) -> str:
    """SHA-256 hex digest of canonical JSON: sorted keys, compact, floats at 12 significant digits.

    Every number (ints included) is rounded to 12 significant digits and
    ``-0.0`` becomes ``0.0`` before serialisation, so last-bit noise from a
    different machine, BLAS or compiler does not change the digest (the E12
    rendezvous plan digests hashed exact float reprs and did). Lists and
    tuples serialise alike; mapping order is irrelevant; non-finite floats
    become the strings ``"nan"``, ``"inf"``, ``"-inf"``.

    Caveat: two values that differ only in noise but straddle a rounding
    boundary of the 12th digit (``...4999999`` against ``...5000001``) still
    digest differently. Quantisation makes such a split rare, not impossible;
    compare the numbers with a tolerance before calling two digests a real change.
    """
    text = json.dumps(_quantise(data), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@lru_cache(maxsize=None)
def load_schema(name: str) -> dict:
    """Load a packaged JSON Schema from ``orbitopt/schemas`` by file name."""
    path = Path(__file__).resolve().parent.parent / "schemas" / name
    return json.loads(path.read_text(encoding="utf-8"))


def schema_errors(document: Any, schema_name: str) -> list[ProblemError]:
    """Validate a dict against a packaged schema; one ``ErrorCode.SCHEMA`` error per violation."""
    validator = Draft202012Validator(load_schema(schema_name))
    found = sorted(validator.iter_errors(document), key=lambda e: [str(p) for p in e.absolute_path])
    return [ProblemError(ErrorCode.SCHEMA, "/" + "/".join(str(p) for p in e.absolute_path), e.message)
            for e in found]


def schema_for(document: Any, schemas: Mapping[str, str]) -> str:
    """The schema a document's ``schemaVersion`` names, or the newest when it names no known version."""
    version = document.get("schemaVersion") if isinstance(document, Mapping) else None
    return schemas.get(version, schemas[SCHEMA_VERSION])


def parse_problem(data: Mapping) -> OptimizationProblem:
    """Parse untrusted dict input: schema, then ``from_dict``, then ``validate_problem``.

    The schema is the one ``data["schemaVersion"]`` names (``PROBLEM_SCHEMAS``),
    so a 1.0 document parses with every 1.1 field at its default. Raises one
    ``InvalidProblemError`` carrying every error found at the first failing
    stage.
    """
    errors = schema_errors(data, schema_for(data, PROBLEM_SCHEMAS))
    if errors:
        raise InvalidProblemError(errors)
    problem = OptimizationProblem.from_dict(data)
    errors = validate_problem(problem)
    if errors:
        raise InvalidProblemError(errors)
    return problem


def _number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


class _Collector:
    def __init__(self):
        self.errors: list[ProblemError] = []

    def __call__(self, code: ErrorCode, path: str, message: str) -> None:
        self.errors.append(ProblemError(code, path, message))

    def unit(self, text: Any, path: str) -> Unit | None:
        try:
            return parse_unit(text)
        except (UnitError, TypeError) as exc:
            self(ErrorCode.UNKNOWN_UNIT, path, str(exc))
            return None

    def enum(self, value: Any, allowed: tuple[str, ...], path: str) -> bool:
        if value not in allowed:
            self(ErrorCode.INVALID_VALUE, path, f"{value!r} is not one of {list(allowed)}")
            return False
        return True

    def finite(self, value: Any, path: str) -> bool:
        if not _number(value):
            self(ErrorCode.INVALID_VALUE, path, f"{value!r} is not a number")
            return False
        if not math.isfinite(value):
            self(ErrorCode.NON_FINITE, path, f"{value!r} is not finite")
            return False
        return True

    def unique(self, items, path: str, identifier: bool, seen: set[str] | None = None) -> None:
        seen = set() if seen is None else seen
        for i, item in enumerate(items):
            name = item.name
            duplicate = isinstance(name, str) and name in seen
            if isinstance(name, str):
                seen.add(name)  # declared even when invalid, so expressions do not cascade into unknown names
            ok = isinstance(name, str) and (
                (name.isidentifier() and not keyword.iskeyword(name) and not name.startswith("_")
                 and name not in RESERVED_NAMES) if identifier else bool(name.strip()))
            if not ok:
                self(ErrorCode.INVALID_NAME, f"{path}/{i}/name",
                     f"{name!r} is not a valid name" + (" (an identifier, not a keyword, function, sum or range, "
                                                        "not starting with '_')" if identifier else ""))
            elif duplicate:
                self(ErrorCode.DUPLICATE_NAME, f"{path}/{i}/name", f"name {name!r} is declared twice")


def _check_variable(err: _Collector, v: Variable, path: str) -> None:
    if not err.enum(v.kind, VARIABLE_KINDS, f"{path}/kind"):
        return
    bounds_ok = all(err.finite(getattr(v, side), f"{path}/{side}")
                    for side in ("lower", "upper") if getattr(v, side) is not None)
    if v.kind in ("continuous", "integer"):
        if v.choices:
            err(ErrorCode.DOMAIN_FIELDS, f"{path}/choices", f"a {v.kind} variable takes bounds, not choices")
        if bounds_ok and v.lower is not None and v.upper is not None and v.lower > v.upper:
            err(ErrorCode.BOUNDS_ORDER, f"{path}/lower", f"lower {v.lower} exceeds upper {v.upper}")
        if bounds_ok and v.kind == "integer":
            for side in ("lower", "upper"):
                bound = getattr(v, side)
                if bound is not None and bound != int(bound):
                    err(ErrorCode.INTEGER_BOUNDS, f"{path}/{side}", f"integer variable bound {bound} is not an integer")
    else:
        for side in ("lower", "upper"):
            if getattr(v, side) is not None:
                err(ErrorCode.DOMAIN_FIELDS, f"{path}/{side}", f"a {v.kind} variable takes no bounds")
        if v.kind == "binary" and v.choices:
            err(ErrorCode.DOMAIN_FIELDS, f"{path}/choices", "a binary variable takes no choices")
        if v.kind == "discrete":
            if not v.choices:
                err(ErrorCode.EMPTY_CHOICES, f"{path}/choices", "a discrete variable needs at least one choice")
            elif all(err.finite(c, f"{path}/choices/{i}") for i, c in enumerate(v.choices)) \
                    and len(set(v.choices)) != len(v.choices):
                err(ErrorCode.INVALID_VALUE, f"{path}/choices", "choices must be distinct")
    if v.initial is not None and err.finite(v.initial, f"{path}/initial") and v.domain_violations(v.initial, 0.0):
        err(ErrorCode.INITIAL_OUT_OF_DOMAIN, f"{path}/initial", f"initial guess {v.initial} is outside the domain")


def _check_source(err: _Collector, source: Any, path: str, declared: set[str]) -> Expression | None:
    if not isinstance(source, str):
        err(ErrorCode.INVALID_VALUE, path, "an expression source must be a string")
        return None
    expr = Expression(source)
    try:
        names = expr.names
    except ExpressionError as exc:
        err(exc.code, path, str(exc))
        return None
    for name in sorted(names - declared):
        err(ErrorCode.UNKNOWN_NAME, path, f"name {name!r} is not a declared variable or parameter")
    return expr if names <= declared else None


def _check_constant(err: _Collector, value: Any, path: str) -> None:
    """A JSON string, boolean, finite number, or an array of these (nested arrays allowed)."""
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _check_constant(err, item, f"{path}/{i}")
    elif not isinstance(value, (str, bool)):
        err.finite(value, path)


def _check_expression(err: _Collector, item: Objective | Constraint, path: str,
                      declared: set[str], dims: dict) -> None:
    target = err.unit(item.unit, f"{path}/unit")
    expr = item.expression
    if isinstance(expr, ToolCall):
        if not (isinstance(expr.tool_id, str) and TOOL_ID_PATTERN.match(expr.tool_id)):
            err(ErrorCode.INVALID_NAME, f"{path}/expression/tool_id",
                f"tool id {expr.tool_id!r} must be dotted lower-case identifiers")
        for arg, source in expr.args.items():
            if not (isinstance(arg, str) and arg.isidentifier()):
                err(ErrorCode.INVALID_NAME, f"{path}/expression/args", f"tool argument {arg!r} is not an identifier")
            _check_source(err, source, f"{path}/expression/args/{arg}", declared)
        for arg, constant in expr.constants.items():
            where = f"{path}/expression/constants/{arg}"
            if not (isinstance(arg, str) and arg.isidentifier()):
                err(ErrorCode.INVALID_NAME, f"{path}/expression/constants", f"tool argument {arg!r} is not an identifier")
            elif arg in expr.args:
                err(ErrorCode.DUPLICATE_NAME, where, f"tool argument {arg!r} is both an expression and a constant")
            _check_constant(err, constant, where)
        if expr.output is not None and not (isinstance(expr.output, str) and expr.output.isidentifier()):
            err(ErrorCode.INVALID_NAME, f"{path}/expression/output", f"tool output {expr.output!r} is not an identifier")
        return
    if not isinstance(expr, Expression):
        err(ErrorCode.INVALID_VALUE, f"{path}/expression", "expected an Expression or a ToolCall")
        return
    checked = _check_source(err, expr.source, f"{path}/expression", declared)
    if checked is None or target is None or not checked.names <= dims.keys():
        return
    try:
        found = checked.dimension(dims)
    except ExpressionError as exc:
        err(exc.code, f"{path}/expression", str(exc))
        return
    if found != target.dims:
        err(ErrorCode.UNIT_MISMATCH, f"{path}/unit",
            f"expression has dimension {format_dims(found)} but unit {item.unit!r} is {format_dims(target.dims)}")


def validate_problem(problem: OptimizationProblem) -> list[ProblemError]:
    """Every semantic error in ``problem``, each with a specific ``ErrorCode``; empty when valid."""
    err = _Collector()
    if not (isinstance(problem.id, str) and problem.id.strip()):
        err(ErrorCode.INVALID_VALUE, "/id", "a problem needs a non-empty id")
    if not problem.variables:
        err(ErrorCode.NO_VARIABLES, "/variables", "a problem needs at least one variable")
    if not problem.objectives:
        err(ErrorCode.NO_OBJECTIVES, "/objectives", "a problem needs at least one objective")

    names: set[str] = set()
    err.unique(problem.variables, "/variables", identifier=True, seen=names)
    err.unique(problem.parameters, "/parameters", identifier=True, seen=names)
    dims = {}
    for group, items in (("variables", problem.variables), ("parameters", problem.parameters)):
        for i, item in enumerate(items):
            unit = err.unit(item.unit, f"/{group}/{i}/unit")
            if unit is not None and isinstance(item.name, str):
                dims[item.name] = unit.dims
    for i, v in enumerate(problem.variables):
        _check_variable(err, v, f"/variables/{i}")
    for i, p in enumerate(problem.parameters):
        err.finite(p.value, f"/parameters/{i}/value")

    err.unique(problem.objectives, "/objectives", identifier=False)
    for i, o in enumerate(problem.objectives):
        path = f"/objectives/{i}"
        err.enum(o.sense, OBJECTIVE_SENSES, f"{path}/sense")
        err.enum(o.origin, ORIGINS, f"{path}/origin")
        _check_expression(err, o, path, names, dims)

    err.unique(problem.constraints, "/constraints", identifier=False)
    for i, c in enumerate(problem.constraints):
        path = f"/constraints/{i}"
        err.enum(c.sense, CONSTRAINT_SENSES, f"{path}/sense")
        err.enum(c.origin, ORIGINS, f"{path}/origin")
        if c.rhs is None:
            err(ErrorCode.MISSING_RHS, f"{path}/rhs", "a constraint needs a right-hand side")
        else:
            err.finite(c.rhs, f"{path}/rhs")
        if c.locked and c.origin != "user":
            err(ErrorCode.LOCKED_NON_USER, f"{path}/locked", f"only user constraints may be locked, not {c.origin!r}")
        _check_expression(err, c, path, names, dims)

    for f in fields(Budget):
        value = getattr(problem.budget, f.name)
        path = f"/budget/{f.name}"
        if value is None:
            continue
        if f.name != "max_wall_s" and not (isinstance(value, int) and not isinstance(value, bool)):
            err(ErrorCode.INVALID_VALUE, path, f"{value!r} is not an integer")
        elif err.finite(value, path) and value <= 0:
            err(ErrorCode.NON_POSITIVE_BUDGET, path, f"budget limits must be positive, got {value}")

    policy = problem.verification
    for i, status in enumerate(policy.required):
        err.enum(status, STATUSES, f"/verification/required/{i}")
    for i, method in enumerate(policy.independent_methods):
        err.enum(method, INDEPENDENT_METHODS, f"/verification/independent_methods/{i}")
    if "independently_verified" in policy.required and not policy.independent_methods:
        err(ErrorCode.INVALID_VALUE, "/verification/independent_methods",
            "independently_verified is required but no independent method is allowed")
    for name in ("constraint_tol", "objective_rtol", "optimality_gap"):
        value = getattr(policy, name)
        if not (_number(value) and math.isfinite(value) and value > 0):
            err(ErrorCode.INVALID_TOLERANCE, f"/verification/{name}", f"{name} must be positive and finite")
    return err.errors
