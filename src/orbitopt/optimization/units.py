"""Unit strings of the optimization IR: parse, scale to SI, check dimensions.

A unit string is a product of symbols from ``SYMBOLS`` with integer or
fractional powers, written the way people write units: ``"km/s"``,
``"kg*m/s^2"``, ``"m**3/(kg*s**2)"``. The empty string and ``"1"`` mean
dimensionless. Angles (``rad``, ``deg``) are dimensionless, as in SI, so an
angle may be compared with a plain number; ``deg`` still scales by pi/180.

The table is deliberately small and implemented here rather than borrowed from
astropy: astropy is only present in the conda compute environment, and the IR
must validate and evaluate identically in a base install (jsonschema + pyyaml).
``tests/test_e14_01_problem_ir.py`` cross-checks every symbol against
astropy where it is installed.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

BASE_DIMENSIONS = ("m", "kg", "s", "A", "K", "mol")
"""Order of the exponents in ``Unit.dims``."""

Dims = tuple[Fraction, ...]


def _dims(m=0, kg=0, s=0, A=0, K=0, mol=0) -> Dims:
    return tuple(Fraction(p) for p in (m, kg, s, A, K, mol))


DIMENSIONLESS = _dims()

SYMBOLS: dict[str, tuple[float, Dims]] = {
    "rad": (1.0, DIMENSIONLESS),
    "deg": (math.pi / 180.0, DIMENSIONLESS),
    "m": (1.0, _dims(m=1)),
    "mm": (1e-3, _dims(m=1)),
    "cm": (1e-2, _dims(m=1)),
    "km": (1e3, _dims(m=1)),
    "AU": (1.495978707e11, _dims(m=1)),
    "kg": (1.0, _dims(kg=1)),
    "g": (1e-3, _dims(kg=1)),
    "t": (1e3, _dims(kg=1)),
    "s": (1.0, _dims(s=1)),
    "min": (60.0, _dims(s=1)),
    "h": (3600.0, _dims(s=1)),
    "day": (86400.0, _dims(s=1)),
    "Hz": (1.0, _dims(s=-1)),
    "N": (1.0, _dims(m=1, kg=1, s=-2)),
    "kN": (1e3, _dims(m=1, kg=1, s=-2)),
    "J": (1.0, _dims(m=2, kg=1, s=-2)),
    "kJ": (1e3, _dims(m=2, kg=1, s=-2)),
    "MJ": (1e6, _dims(m=2, kg=1, s=-2)),
    "W": (1.0, _dims(m=2, kg=1, s=-3)),
    "kW": (1e3, _dims(m=2, kg=1, s=-3)),
    "Pa": (1.0, _dims(m=-1, kg=1, s=-2)),
    "kPa": (1e3, _dims(m=-1, kg=1, s=-2)),
    "A": (1.0, _dims(A=1)),
    "K": (1.0, _dims(K=1)),
    "mol": (1.0, _dims(mol=1)),
}
"""Every accepted unit symbol: its SI scale factor and dimension exponents."""


class UnitError(ValueError):
    """A unit string that does not parse, or names a symbol outside ``SYMBOLS``."""


@dataclass(frozen=True)
class Unit:
    """A parsed unit: multiply a value in this unit by ``scale`` to get SI."""

    scale: float
    dims: Dims

    @property
    def dimensionless(self) -> bool:
        return self.dims == DIMENSIONLESS


def _times(a: Unit, b: Unit, sign: int) -> Unit:
    return Unit(a.scale * b.scale ** sign, tuple(x + sign * y for x, y in zip(a.dims, b.dims)))


def _exponent(node: ast.AST) -> Fraction:
    negative = isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
    if negative:
        node = node.operand
    if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
            and all(isinstance(n, ast.Constant) and type(n.value) is int for n in (node.left, node.right))
            and node.right.value != 0):
        value = Fraction(node.left.value, node.right.value)
    elif isinstance(node, ast.Constant) and type(node.value) is int:
        value = Fraction(node.value)
    else:
        raise UnitError("a unit exponent must be an integer or a ratio of integers")
    if abs(value) > 12:
        raise UnitError("a unit exponent must be at most 12 in magnitude")
    return -value if negative else value


def _unit_of(node: ast.AST) -> Unit:
    if isinstance(node, ast.Name):
        if node.id not in SYMBOLS:
            raise UnitError(f"unknown unit symbol {node.id!r}")
        return Unit(*SYMBOLS[node.id])
    if isinstance(node, ast.Constant) and node.value == 1 and type(node.value) is int:
        return Unit(1.0, DIMENSIONLESS)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div)):
        return _times(_unit_of(node.left), _unit_of(node.right), 1 if isinstance(node.op, ast.Mult) else -1)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        base, power = _unit_of(node.left), _exponent(node.right)
        return Unit(base.scale ** float(power), tuple(d * power for d in base.dims))
    raise UnitError("a unit is a product of symbols with powers, such as 'kg*m/s^2'")


@lru_cache(maxsize=1024)
def parse_unit(text: str) -> Unit:
    """Parse a unit string into its SI scale and dimension; raise ``UnitError``.

    ``^`` and ``**`` both mean a power. Parentheses group. Only symbols in
    ``SYMBOLS`` and the literal ``1`` are accepted. The result is cached.
    """
    if not isinstance(text, str):
        raise UnitError("a unit must be a string")
    if text.strip() in ("", "1"):
        return Unit(1.0, DIMENSIONLESS)
    if len(text) > 64:
        raise UnitError("a unit string must be at most 64 characters")
    try:
        tree = ast.parse(text.replace("^", "**"), mode="eval")
    except SyntaxError as exc:
        raise UnitError(f"unit {text!r} does not parse") from exc
    return _unit_of(tree.body)


def format_dims(dims: Dims) -> str:
    """Render dimension exponents for an error message, such as ``m*s^-1``."""
    parts = [sym if p == 1 else f"{sym}^{p}" for sym, p in zip(BASE_DIMENSIONS, dims) if p]
    return "*".join(parts) or "dimensionless"
