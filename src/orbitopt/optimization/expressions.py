"""Safe arithmetic expressions and tool references for objectives and constraints.

An ``Expression`` is a source string such as ``"x**2 + y**2"`` that passes a
whitelist walk over its Python syntax tree: numeric literals, names,
``+ - * / **``, unary ``-``/``+`` and calls to the functions in ``FUNCTIONS``.
Everything else (attribute access, subscripts, strings, comparisons, lambdas,
comprehensions, keyword arguments, names starting with ``_``) is rejected
before anything runs. Integer literals become floats before compilation, so
no expression can trigger unbounded integer arithmetic such as ``9**9**9``.
Evaluation compiles only the validated tree, never the raw text, and runs it
with no builtins.

A ``ToolCall`` names a scientific tool by registered id. The IR only types the
reference; the registry is E14-04.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache, reduce
from typing import Any, Mapping

from .errors import ErrorCode
from .units import DIMENSIONLESS, Dims, format_dims

FUNCTIONS: dict[str, int | None] = {
    "abs": 1, "sqrt": 1, "exp": 1, "log": 1, "log10": 1,
    "sin": 1, "cos": 1, "tan": 1, "asin": 1, "acos": 1, "atan": 1, "atan2": 2,
    "sinh": 1, "cosh": 1, "tanh": 1, "min": None, "max": None,
}
"""Callable functions and their arity; ``None`` means two or more arguments."""

MAX_SOURCE_LENGTH = 2000
MAX_NODES = 500
MAX_DEPTH = 200
MAX_EXPONENT = 100.0
"""A literal exponent larger than this in magnitude is rejected."""

TOOL_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")

_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load, ast.Constant, ast.Call,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)
_MATH = {"abs": abs, "min": min, "max": max,
         **{name: getattr(math, name) for name in FUNCTIONS if hasattr(math, name)}}


class ExpressionError(ValueError):
    """An expression that is not in the safe subset, or whose units disagree."""

    def __init__(self, code: ErrorCode, message: str):
        self.code = code
        super().__init__(message)


class EvaluationError(ArithmeticError):
    """Evaluating a valid expression failed: a domain error, overflow, missing value or tool."""


@dataclass(frozen=True)
class _Parsed:
    tree: ast.Expression
    code: Any
    names: frozenset[str]


def _literal(node: ast.AST) -> float | None:
    """The value of a numeric literal, optionally negated; ``None`` otherwise."""
    sign = 1.0
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        sign = -1.0 if isinstance(node.op, ast.USub) else 1.0
        node = node.operand
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return sign * float(node.value)
    return None


def _check_node(node: ast.AST, callees: set[int]) -> None:
    if isinstance(node, ast.Constant):
        if type(node.value) not in (int, float):
            raise ExpressionError(ErrorCode.BAD_LITERAL, f"literal {node.value!r} is not a number")
        try:
            finite = math.isfinite(float(node.value))
        except OverflowError:
            finite = False
        if not finite:
            raise ExpressionError(ErrorCode.BAD_LITERAL, "numeric literals must be finite")
    elif isinstance(node, ast.BitXor):
        raise ExpressionError(ErrorCode.UNSAFE_EXPRESSION, "'^' is not allowed; write powers as '**'")
    elif not isinstance(node, _ALLOWED):
        raise ExpressionError(ErrorCode.UNSAFE_EXPRESSION, f"{type(node).__name__} is not allowed in an expression")
    elif isinstance(node, ast.Name) and node.id.startswith("_"):
        raise ExpressionError(ErrorCode.UNSAFE_EXPRESSION, f"name {node.id!r} may not start with '_'")
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        exponent = _literal(node.right)
        if exponent is not None and abs(exponent) > MAX_EXPONENT:
            raise ExpressionError(ErrorCode.EXPONENT_TOO_LARGE,
                                  f"literal exponent {exponent:g} exceeds {MAX_EXPONENT:g} in magnitude")
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExpressionError(ErrorCode.UNSAFE_EXPRESSION, "only named functions may be called")
        name = node.func.id
        if name not in FUNCTIONS:
            raise ExpressionError(ErrorCode.UNKNOWN_FUNCTION, f"function {name!r} is not one of {sorted(FUNCTIONS)}")
        if node.keywords:
            raise ExpressionError(ErrorCode.UNSAFE_EXPRESSION, f"{name}() takes no keyword arguments")
        arity = FUNCTIONS[name]
        if (len(node.args) != arity) if arity is not None else (len(node.args) < 2):
            expected = arity if arity is not None else "two or more"
            raise ExpressionError(ErrorCode.WRONG_ARITY, f"{name}() takes {expected} arguments, got {len(node.args)}")
        callees.add(id(node.func))


class _FloatLiterals(ast.NodeTransformer):
    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        return ast.copy_location(ast.Constant(float(node.value)), node)


@lru_cache(maxsize=4096)
def _parse(source: str) -> _Parsed:
    if not isinstance(source, str):
        raise ExpressionError(ErrorCode.INVALID_VALUE, "an expression source must be a string")
    if len(source) > MAX_SOURCE_LENGTH:
        raise ExpressionError(ErrorCode.EXPRESSION_TOO_COMPLEX, f"expression exceeds {MAX_SOURCE_LENGTH} characters")
    try:
        tree = ast.parse(source.strip(), mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
        raise ExpressionError(ErrorCode.SYNTAX_ERROR, f"expression does not parse: {exc}") from None
    callees: set[int] = set()
    names: set[str] = set()
    stack, count = [(tree, 0)], 0
    while stack:
        node, depth = stack.pop()
        count += 1
        if count > MAX_NODES or depth > MAX_DEPTH:
            raise ExpressionError(ErrorCode.EXPRESSION_TOO_COMPLEX,
                                  f"expression exceeds {MAX_NODES} nodes or nesting depth {MAX_DEPTH}")
        _check_node(node, callees)
        if isinstance(node, ast.Name) and id(node) not in callees:
            names.add(node.id)
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    tree = ast.fix_missing_locations(_FloatLiterals().visit(tree))
    return _Parsed(tree, compile(tree, "<expression>", "eval"), frozenset(names))


@lru_cache(maxsize=1)
def _numpy_functions() -> dict[str, Any]:
    import numpy as np

    table = {name: getattr(np, name) for name in ("abs", "sqrt", "exp", "log", "log10", "sin", "cos", "tan",
                                                   "sinh", "cosh", "tanh")}
    table.update(asin=np.arcsin, acos=np.arccos, atan=np.arctan, atan2=np.arctan2,
                 min=lambda *a: reduce(np.minimum, a), max=lambda *a: reduce(np.maximum, a))
    return table


def _dimension(node: ast.AST, units: Mapping[str, Dims]) -> Dims:
    if isinstance(node, ast.Constant):
        return DIMENSIONLESS
    if isinstance(node, ast.Name):
        if node.id not in units:
            raise ExpressionError(ErrorCode.UNKNOWN_NAME, f"name {node.id!r} is not a declared variable or parameter")
        return units[node.id]
    if isinstance(node, ast.UnaryOp):
        return _dimension(node.operand, units)
    if isinstance(node, ast.BinOp):
        left, right = _dimension(node.left, units), _dimension(node.right, units)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            _same(left, right, "terms of a sum")
            return left
        if isinstance(node.op, ast.Mult):
            return tuple(a + b for a, b in zip(left, right))
        if isinstance(node.op, ast.Div):
            return tuple(a - b for a, b in zip(left, right))
        _same(right, DIMENSIONLESS, "an exponent")
        if left == DIMENSIONLESS:
            return DIMENSIONLESS
        power = _literal(node.right)
        if power is None:
            raise ExpressionError(ErrorCode.UNIT_MISMATCH,
                                  "a quantity with units can only be raised to a literal power")
        return tuple(d * Fraction(power).limit_denominator(1000) for d in left)
    args = [_dimension(arg, units) for arg in node.args]
    name = node.func.id
    if name in ("abs", "min", "max"):
        for other in args[1:]:
            _same(args[0], other, f"arguments of {name}()")
        return args[0]
    if name == "atan2":
        _same(args[0], args[1], "arguments of atan2()")
        return DIMENSIONLESS
    if name == "sqrt":
        return tuple(d / 2 for d in args[0])
    _same(args[0], DIMENSIONLESS, f"the argument of {name}()")
    return DIMENSIONLESS


def _same(a: Dims, b: Dims, what: str) -> None:
    if a != b:
        raise ExpressionError(ErrorCode.UNIT_MISMATCH,
                              f"{what} must share a dimension, got {format_dims(a)} and {format_dims(b)}")


@dataclass(frozen=True)
class Expression:
    """A safe arithmetic expression over declared variable and parameter names."""

    source: str

    @property
    def names(self) -> frozenset[str]:
        """Variable and parameter names the expression reads; raises ``ExpressionError`` if unsafe."""
        return _parse(self.source).names

    def dimension(self, units: Mapping[str, Dims]) -> Dims:
        """Infer the result dimension from the dimension of each name; raises ``ExpressionError``.

        Sums, ``min``/``max``/``atan2`` arguments and exponents must agree;
        a dimensioned base takes only a literal power; transcendental
        functions take dimensionless arguments.
        """
        return _dimension(_parse(self.source).tree.body, units)

    def evaluate(self, values: Mapping[str, Any]) -> Any:
        """Evaluate over plain numbers (returns ``float``) or numpy arrays (vectorised).

        Values are used as given, with no unit conversion. Raises
        ``ExpressionError`` if the source is unsafe and ``EvaluationError`` on
        a domain error, overflow, complex result or missing value.
        """
        parsed = _parse(self.source)
        missing = parsed.names - values.keys()
        if missing:
            raise EvaluationError(f"no value for {sorted(missing)}")
        namespace = {name: values[name] for name in parsed.names}
        vectorised = any(not isinstance(v, (int, float)) for v in namespace.values())
        namespace.update(_numpy_functions() if vectorised else _MATH)
        try:
            result = eval(parsed.code, {"__builtins__": {}}, namespace)  # noqa: S307 - validated tree only
        except (ArithmeticError, ValueError, TypeError) as exc:
            raise EvaluationError(f"{self.source!r}: {exc}") from exc
        if isinstance(result, complex):
            raise EvaluationError(f"{self.source!r} has a complex value")
        return result if vectorised else float(result)


@dataclass(frozen=True)
class ToolCall:
    """A reference to a registered scientific tool, called with expression-valued arguments.

    ``args`` maps each tool argument to an expression source over declared
    names. The tool receives SI-coherent argument values and must return an
    SI-coherent value; the objective or constraint ``unit`` declares its
    dimension, which the E14-04 registry is responsible for checking.
    """

    tool_id: str
    args: Mapping[str, str] = field(default_factory=dict)

    @property
    def names(self) -> frozenset[str]:
        """Names read by all arguments; raises ``ExpressionError`` if any argument is unsafe."""
        return frozenset().union(*(Expression(src).names for src in self.args.values()))
