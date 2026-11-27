"""Quadrature with a-posteriori error estimates, and the accuracy-against-cost tool behind the E14 integration family.

The integrand is an IR expression over the single name ``t``, such as
``"exp(-t**2)"``; it is parsed by the IR's safe expression evaluator, never
by Python's ``eval`` on raw text.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable

from orbitopt.optimization import EvaluationError, Expression, ExpressionError

from ..spec import CostModel, DomainError, Interval, Tool, ToolSpec
from ._common import inputs, integer, number, outputs, tests, text

RULES = ("gauss_legendre", "trapezoid", "simpson", "gauss_legendre_t2")
"""Rules of ``numerics.quadrature_cost.v1``, in the order of the E14 benchmark's ``QUADRATURE_RULES``."""

INTEGRAND = text("Integrand as an IR expression over t, such as 'exp(-t**2)'.", maxLength=2000)
LIMITS = {"a": number("Lower limit."), "b": number("Upper limit.")}
LIMIT_DOMAIN = {"a": Interval(-1e6, 1e6), "b": Interval(-1e6, 1e6)}
INTEGRAND_ASSUMPTION = "The integrand is finite on [a, b]; a non-finite value at any node is an error, not a result."


def compile_integrand(source: str) -> Callable:
    """A vectorised ``f(t)`` for an IR expression over ``t``; ``DomainError`` if the expression is unusable."""
    expr = Expression(source)
    try:
        names = expr.names
    except ExpressionError as exc:
        raise DomainError(f"integrand {source!r}: {exc}") from None
    if not names <= {"t"}:
        raise DomainError(f"integrand {source!r} may only read t, not {sorted(names - {'t'})}")

    def f(t):
        import numpy as np

        t = np.asarray(t, dtype=float)
        with np.errstate(all="ignore"):
            try:
                y = expr.evaluate({"t": t})
            except EvaluationError as exc:
                raise DomainError(f"integrand {source!r}: {exc}") from None
        return np.broadcast_to(np.asarray(y, dtype=float), t.shape)

    return f


@lru_cache(maxsize=None)
def _legendre(n: int):
    import numpy as np

    return np.polynomial.legendre.leggauss(n)


def gauss(f: Callable, a: float, b: float, n: int) -> float:
    """``n``-point Gauss-Legendre on [a, b]."""
    import numpy as np

    t, w = _legendre(n)
    return float(0.5 * (b - a) * np.dot(w, f(0.5 * (b - a) * t + 0.5 * (a + b))))


def gauss_legendre(integrand: str, a: float, b: float, n: int) -> dict:
    f = compile_integrand(integrand)
    low, high = gauss(f, a, b, n), gauss(f, a, b, 2 * n)
    return {"value": low, "error_estimate": abs(high - low), "evaluations": 3 * n}


def adaptive_simpson(integrand: str, a: float, b: float, tol: float, max_evaluations: int) -> dict:
    """Adaptive Simpson (Lyness): split until |S_left + S_right - S| <= 15 * tol_i, halving tol_i at each split."""
    f = compile_integrand(integrand)
    m = 0.5 * (a + b)
    fa, fm, fb = (float(v) for v in f([a, m, b]))
    stack = [(a, b, fa, fm, fb, (b - a) / 6.0 * (fa + 4.0 * fm + fb), tol)]
    total, error, evaluations, converged = 0.0, 0.0, 3, True
    while stack:
        lo, hi, flo, fmid, fhi, whole, eps = stack.pop()
        mid = 0.5 * (lo + hi)
        if evaluations + 2 > max_evaluations or mid in (lo, hi):
            total += whole
            converged = False
            continue
        fl, fr = (float(v) for v in f([0.5 * (lo + mid), 0.5 * (mid + hi)]))
        evaluations += 2
        left = (mid - lo) / 6.0 * (flo + 4.0 * fl + fmid)
        right = (hi - mid) / 6.0 * (fmid + 4.0 * fr + fhi)
        delta = left + right - whole
        if abs(delta) <= 15.0 * eps:
            total += left + right + delta / 15.0
            error += abs(delta) / 15.0
        else:
            stack.append((mid, hi, fmid, fr, fhi, right, 0.5 * eps))
            stack.append((lo, mid, flo, fl, fmid, left, 0.5 * eps))
    return {"value": total, "error_estimate": error, "evaluations": evaluations, "converged": converged}


def rule_value(f: Callable, a: float, b: float, rule: str, n: int) -> float:
    """Rule ``rule`` with exactly ``n`` integrand evaluations; ``DomainError`` where the rule is undefined."""
    import numpy as np

    if rule == "gauss_legendre":
        return gauss(f, a, b, n)
    if rule == "gauss_legendre_t2":  # x = a + (b - a) s^2, dx = 2 (b - a) s ds, s in [0, 1]
        t, w = _legendre(n)
        s = 0.5 * (t + 1.0)
        return float(0.5 * np.dot(w, f(a + (b - a) * s ** 2) * 2.0 * (b - a) * s))
    if rule == "trapezoid" and n >= 2:
        y = f(np.linspace(a, b, n))
        return float((b - a) / (n - 1) * (np.sum(y) - 0.5 * (y[0] + y[-1])))
    if rule == "simpson" and n >= 3 and n % 2 == 1:
        y = f(np.linspace(a, b, n))
        return float((b - a) / (n - 1) / 3.0 * (y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2])))
    raise DomainError(f"rule {rule!r} is not defined for n={n} (trapezoid needs n >= 2, simpson an odd n >= 3)")


REFERENCE_RTOL = 1e-12
"""The reference integral is refused when QUADPACK's error estimate exceeds this times max(1, |value|)."""


def reference_integral(f: Callable, a: float, b: float) -> tuple[float, float]:
    """QUADPACK's adaptive Gauss-Kronrod asked for 1e-14, with its error estimate.

    QUADPACK often stops at round-off just short of 1e-14; that result is
    kept. ``DomainError`` when the estimate is worse than ``REFERENCE_RTOL``
    (a singular or divergent integrand).
    """
    from scipy.integrate import quad

    value, error, *_ = quad(lambda t: float(f(t)), a, b, epsabs=1e-14, epsrel=1e-14, limit=500, full_output=1)
    if not error <= REFERENCE_RTOL * max(1.0, abs(value)):
        raise DomainError(f"the reference integral did not converge (estimated error {error:.3g})")
    return value, error


def quadrature_cost(integrand: str, a: float, b: float, rule: str, n: int) -> dict:
    f = compile_integrand(integrand)
    value = rule_value(f, a, b, rule, n)
    reference, reference_error = reference_integral(f, a, b)
    return {"value": value, "abs_error": abs(value - reference), "reference": reference,
            "reference_error": reference_error, "evaluations": n}


_ESTIMATE = {"value": number("Integral estimate."), "error_estimate": number("Estimated absolute error of value."),
             "evaluations": integer("Integrand evaluations used.")}

TOOLS = (
    Tool(ToolSpec(
        tool_id="numerics.gauss_legendre.v1",
        description="n-point Gauss-Legendre quadrature of an integrand on [a, b], with the error estimate "
                    "|Q_n - Q_2n| from the 2n-point rule.",
        input_schema=inputs({"integrand": INTEGRAND, **LIMITS, "n": integer("Number of Gauss points.")}),
        output_schema=outputs(_ESTIMATE),
        input_units={"a": "", "b": "", "n": ""}, output_units={"value": "", "error_estimate": "", "evaluations": ""},
        assumptions=(INTEGRAND_ASSUMPTION,
                     "The estimate |Q_n - Q_2n| is the error of Q_n when Q_2n has converged much further, as it "
                     "does for an integrand analytic near [a, b]; it is not a bound in general."),
        domain={**LIMIT_DOMAIN, "n": Interval(1, 256)},
        cost=CostModel("cheap", "3*n"),
        tested_by=tests("test_gauss_legendre_matches_closed_forms_and_its_estimate_tracks_the_error"),
    ), gauss_legendre),
    Tool(ToolSpec(
        tool_id="numerics.adaptive_simpson.v1",
        description="Adaptive Simpson quadrature of an integrand on [a, b] to an absolute tolerance, within a cap "
                    "on integrand evaluations.",
        input_schema=inputs({"integrand": INTEGRAND, **LIMITS, "tol": number("Absolute error tolerance."),
                             "max_evaluations": integer("Cap on integrand evaluations.")}),
        output_schema=outputs({**_ESTIMATE, "converged": {"type": "boolean",
                                                          "description": "Every interval met its tolerance."}}),
        input_units={"a": "", "b": "", "tol": "", "max_evaluations": ""},
        output_units={"value": "", "error_estimate": "", "evaluations": ""},
        assumptions=(INTEGRAND_ASSUMPTION,
                     "The error estimate sums |S2 - S|/15 over accepted intervals (Richardson); it is reliable for a "
                     "smooth integrand. When converged is false, intervals left unrefined are not in the estimate."),
        domain={**LIMIT_DOMAIN, "tol": Interval(1e-14, 1.0), "max_evaluations": Interval(5, 200_000)},
        cost=CostModel("moderate", "max_evaluations"),
        tested_by=tests("test_adaptive_simpson_meets_its_tolerance_on_closed_forms"),
    ), adaptive_simpson),
    Tool(ToolSpec(
        tool_id="numerics.quadrature_cost.v1",
        description="Accuracy against cost of one quadrature setting: run rule with exactly n integrand "
                    "evaluations and report its absolute error against a 1e-14 adaptive reference integral.",
        input_schema=inputs({"integrand": INTEGRAND, **LIMITS, "rule": text("Quadrature rule.", enum=list(RULES)),
                             "n": integer("Integrand evaluations the rule may use.")}),
        output_schema=outputs({"value": number("Rule estimate."),
                               "abs_error": number("|value - reference|."),
                               "reference": number("Reference integral (QUADPACK, 1e-14)."),
                               "reference_error": number("QUADPACK's error estimate of the reference."),
                               "evaluations": integer("Integrand evaluations the rule used.")}),
        input_units={"a": "", "b": "", "n": ""},
        output_units={"value": "", "abs_error": "", "reference": "", "reference_error": "", "evaluations": ""},
        assumptions=(INTEGRAND_ASSUMPTION,
                     "abs_error is meaningful down to about reference_error (near 1e-14), not below it.",
                     "Trapezoid and Simpson use n equally spaced points; Simpson needs an odd n >= 3. "
                     "gauss_legendre_t2 applies Gauss-Legendre after x = a + (b - a) s^2, which suits an "
                     "integrand that is steep near a.",
                     "The cost charged is the rule's n evaluations; the reference integral is overhead."),
        domain={**LIMIT_DOMAIN, "n": Interval(1, 512)},
        cost=CostModel("cheap", "n"),
        tested_by=tests("test_quadrature_cost_reproduces_benchmark_minimum_evaluations",
                        "test_quadrature_cost_rules_match_the_benchmark_definitions"),
    ), quadrature_cost),
)
