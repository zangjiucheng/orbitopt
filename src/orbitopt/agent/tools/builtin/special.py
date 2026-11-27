"""Standard special functions of one real argument: erf, erfc, gamma, lgamma, log1p, expm1, J0, J1.

Each tool is one row of ``_FUNCTIONS``. The domain is the range the tests
verify against 20-digit mpmath values; outside it the tool refuses.
"""

from __future__ import annotations

import math
from typing import Callable

from ..spec import CostModel, Interval, Tool, ToolSpec
from ._common import inputs, number, outputs, tests


def _bessel(order: int) -> Callable[[float], float]:
    def evaluate(x: float) -> float:
        from scipy import special

        return float(special.j0(x) if order == 0 else special.j1(x))

    return evaluate


_LIBM = "Evaluated in IEEE-754 double precision by the C library through Python's math module."
_CEPHES = "Evaluated in IEEE-754 double precision by scipy.special (Cephes)."

_FUNCTIONS = (
    ("math.erf.v1", "Error function erf(x) = (2/sqrt(pi)) * integral_0^x exp(-t^2) dt.", math.erf, -10.0, 10.0, _LIBM),
    ("math.erfc.v1", "Complementary error function erfc(x) = 1 - erf(x), without cancellation for large x.",
     math.erfc, -10.0, 25.0, _LIBM),
    ("math.gamma.v1", "Gamma function Gamma(x) for positive x.", math.gamma, 1e-3, 170.0, _LIBM),
    ("math.lgamma.v1", "Natural log of the Gamma function, ln Gamma(x), for positive x.", math.lgamma, 1e-3, 1e6, _LIBM),
    ("math.log1p.v1", "ln(1 + x), accurate for small x.", math.log1p, -0.999999, 1e6, _LIBM),
    ("math.expm1.v1", "exp(x) - 1, accurate for small x.", math.expm1, -50.0, 50.0, _LIBM),
    ("math.bessel_j0.v1", "Bessel function of the first kind of order 0, J0(x).", _bessel(0), -100.0, 100.0, _CEPHES),
    ("math.bessel_j1.v1", "Bessel function of the first kind of order 1, J1(x).", _bessel(1), -100.0, 100.0, _CEPHES),
)


def _tool(tool_id: str, description: str, fn: Callable[[float], float], lower: float, upper: float,
          how: str) -> Tool:
    def call(x: float) -> dict:
        return {"value": fn(x)}

    spec = ToolSpec(
        tool_id=tool_id, description=description,
        input_schema=inputs({"x": number("Real argument.")}),
        output_schema=outputs({"value": number("Function value.")}),
        input_units={"x": ""}, output_units={"value": ""},
        assumptions=(how, "Real, dimensionless argument and value."),
        domain={"x": Interval(lower, upper)},
        cost=CostModel("trivial"),
        tested_by=tests("test_special_functions_match_mpmath_references"),
    )
    return Tool(spec, call)


TOOLS = tuple(_tool(*row) for row in _FUNCTIONS)
