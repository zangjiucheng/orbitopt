"""E14-04 scientific tool registry: allowlisted, validated, logged and replayable tool calls.

Every built-in is checked against a reference that does not share its code:
20-digit mpmath values (recorded below, re-checked when mpmath is installed),
closed-form integrals and series solutions, and the E14 benchmark's own
quadrature definitions and verified minimum costs.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import pytest

from orbitopt.agent.tools import (
    BUILTIN_TOOLS,
    CallBudget,
    Caller,
    CallLog,
    CostModel,
    Interval,
    Tool,
    ToolError,
    ToolErrorCode,
    ToolRegistry,
    ToolSpec,
    bind_problem,
    builtin_registry,
    replay,
)
from orbitopt.agent.tools.calllog import record_digest
from orbitopt.agent.tools.spec import spec_errors
from orbitopt.optimization import (
    Constraint,
    Objective,
    OptimizationProblem,
    Parameter,
    ToolCall,
    Variable,
    validate_problem,
)

REPO = Path(__file__).resolve().parents[1]
Code = ToolErrorCode
AGENT = Caller("agent-under-test")


def _benchmark_problems():
    if "e14_spec" not in sys.modules:
        pkg = REPO / "benchmarks/e14_spec"
        spec = importlib.util.spec_from_file_location("e14_spec", pkg / "__init__.py",
                                                      submodule_search_locations=[str(pkg)])
        module = importlib.util.module_from_spec(spec)
        sys.modules["e14_spec"] = module
        spec.loader.exec_module(module)
    return sys.modules["e14_spec"].problems


def full_view(log=None):
    registry = builtin_registry()
    return registry.view([s.tool_id for s in registry.list()], log)


def call_code(view, tool_id, args, caller=AGENT):
    with pytest.raises(ToolError) as info:
        view.call(tool_id, args, caller=caller)
    return info.value.code


def scalar_tool(tool_id="test.echo.v1", fn=None, **spec_changes):
    spec = ToolSpec(
        tool_id=tool_id, description="Test tool.",
        input_schema={"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"],
                      "additionalProperties": False},
        output_schema={"type": "object", "properties": {"value": {"type": "number"}}, "required": ["value"],
                       "additionalProperties": False},
        input_units={"x": ""}, output_units={"value": ""}, assumptions=("A test double.",), domain={},
        cost=CostModel("trivial"), tested_by=("tests/test_e14_04_tool_registry.py::scalar_tool",))
    return Tool(dataclasses.replace(spec, **spec_changes), fn or (lambda x: {"value": x}))


# ------------------------------------------------------------------------------------------------ access control
def test_only_registered_and_allowed_tools_can_be_called():
    registry = builtin_registry()
    view = registry.view(["math.erf.v1"])
    assert [s.tool_id for s in view.list()] == ["math.erf.v1"]
    assert view.call("math.erf.v1", {"x": 0.5}, caller=AGENT) == {"value": math.erf(0.5)}
    assert call_code(view, "math.gamma.v1", {"x": 2.0}) == Code.NOT_ALLOWED
    assert call_code(view, "math.nonexistent.v1", {"x": 2.0}) == Code.UNKNOWN_TOOL
    with pytest.raises(ToolError) as info:
        view.get("math.gamma.v1")
    assert info.value.code == Code.NOT_ALLOWED
    with pytest.raises(ToolError) as info:
        registry.view(["math.erf.v1", "math.typo.v1"])
    assert info.value.code == Code.UNKNOWN_TOOL

    outcomes = [(r.tool_id, r.outcome, r.code) for r in view.log.records]
    assert outcomes == [("math.erf.v1", "ok", None),
                        ("math.gamma.v1", "refused", Code.NOT_ALLOWED),
                        ("math.nonexistent.v1", "refused", Code.UNKNOWN_TOOL)]
    assert [e.status for e in replay(view.log, builtin_registry()).entries] == ["exact", "not_executed", "not_executed"]


# ------------------------------------------------------------------------------------ built-ins against references
# mpmath 1.4.1 at mp.dps = 30, printed to 20 digits: mp.erf(x), mp.erfc(x), mp.gamma(x), mp.loggamma(x),
# mp.log1p(x), mp.expm1(x), mp.besselj(0, x), mp.besselj(1, x) on the exact binary value of each float x.
MPMATH_REFERENCES = {
    "math.erf.v1": [(0.5, "0.52049987781304653768"), (-2.0, "-0.99532226501895273416"),
                    (3.0, "0.99997790950300141456"), (1e-08, "1.1283791670955125599e-8")],
    "math.erfc.v1": [(0.5, "0.47950012218695346232"), (5.0, "1.5374597944280348502e-12"),
                     (-1.0, "1.8427007929497148693")],
    "math.gamma.v1": [(0.5, "1.7724538509055160273"), (2.5, "1.3293403881791370205"), (10.0, "362880.0"),
                      (0.01, "99.432585119150601632")],
    "math.lgamma.v1": [(0.5, "0.57236494292470008707"), (100.0, "359.13420536957539878"),
                       (12345.678, "103959.91990554605982")],
    "math.log1p.v1": [(1e-10, "9.9999999995000003644e-11"), (3.0, "1.3862943611198906188"),
                      (-0.5, "-0.69314718055994530942")],
    "math.expm1.v1": [(1e-10, "1.0000000000500000364e-10"), (-1.0, "-0.6321205588285576784"),
                      (20.0, "485165194.40979027797")],
    "math.bessel_j0.v1": [(1.0, "0.76519768655796655145"), (2.5, "-0.048383776468197996327"),
                          (50.0, "0.055812327669251815005"), (-7.0, "0.30007927051955559665")],
    "math.bessel_j1.v1": [(1.0, "0.44005058574493351596"), (-2.0, "-0.5767248077568733872"),
                          (30.0, "-0.11875106261662293652")],
}


def test_special_functions_match_mpmath_references():
    view = full_view()
    worst = 0.0
    for tool_id, cases in MPMATH_REFERENCES.items():
        for x, reference in cases:
            got = view.call(tool_id, {"x": x}, caller=AGENT)["value"]
            rel = abs(got - float(reference)) / abs(float(reference))
            worst = max(worst, rel)
            assert rel <= 1e-13, (tool_id, x, got, reference)
    assert worst <= 1e-13
    mpmath = pytest.importorskip("mpmath")
    mpmath.mp.dps = 30
    functions = {"math.erf.v1": mpmath.erf, "math.erfc.v1": mpmath.erfc, "math.gamma.v1": mpmath.gamma,
                 "math.lgamma.v1": mpmath.loggamma, "math.log1p.v1": mpmath.log1p, "math.expm1.v1": mpmath.expm1,
                 "math.bessel_j0.v1": lambda x: mpmath.besselj(0, x),
                 "math.bessel_j1.v1": lambda x: mpmath.besselj(1, x)}
    for tool_id, cases in MPMATH_REFERENCES.items():
        for x, reference in cases:
            assert mpmath.almosteq(functions[tool_id](mpmath.mpf(x)), mpmath.mpf(reference), rel_eps=1e-19)


def test_gauss_legendre_matches_closed_forms_and_its_estimate_tracks_the_error():
    view = full_view()
    # n = 3 points integrate a degree-5 polynomial exactly: int_-1^2 (t^5 - 3 t^2 + 1) dt = 21/2 - 9 + 3 = 4.5
    exact = view.call("numerics.gauss_legendre.v1", {"integrand": "t**5 - 3*t**2 + 1", "a": -1, "b": 2, "n": 3},
                      caller=AGENT)
    assert abs(exact["value"] - 4.5) <= 1e-14 and exact["error_estimate"] <= 1e-14 and exact["evaluations"] == 9
    for n in (2, 3, 4, 5):
        out = view.call("numerics.gauss_legendre.v1", {"integrand": "exp(t)", "a": 0, "b": 1, "n": n}, caller=AGENT)
        actual = abs(out["value"] - (math.e - 1.0))
        assert abs(out["error_estimate"] - actual) <= 1e-3 * actual + 1e-15, (n, out, actual)
    assert call_code(view, "numerics.gauss_legendre.v1",
                     {"integrand": "exp(s)", "a": 0, "b": 1, "n": 4}) == Code.OUT_OF_DOMAIN


def test_adaptive_simpson_meets_its_tolerance_on_closed_forms():
    view = full_view()
    cases = [("exp(-t**2)", 0.0, 1.0, 0.5 * math.sqrt(math.pi) * math.erf(1.0)),
             ("1/sqrt(t+0.001)", 0.0, 1.0, 2.0 * (math.sqrt(1.001) - math.sqrt(0.001))),
             ("cos(20*t)", 0.0, 1.0, math.sin(20.0) / 20.0)]
    for integrand, a, b, exact in cases:
        for tol in (1e-6, 1e-10):
            out = view.call("numerics.adaptive_simpson.v1", {"integrand": integrand, "a": a, "b": b, "tol": tol,
                                                             "max_evaluations": 200_000}, caller=AGENT)
            assert out["converged"] and abs(out["value"] - exact) <= tol, (integrand, tol, out)
            assert out["error_estimate"] <= tol and out["evaluations"] % 2 == 1
    capped = view.call("numerics.adaptive_simpson.v1", {"integrand": "1/sqrt(t+0.001)", "a": 0, "b": 1,
                                                        "tol": 1e-12, "max_evaluations": 9}, caller=AGENT)
    assert not capped["converged"] and capped["evaluations"] <= 9


INTEGRATION_EXACT = {
    "integ-exp-1e-10": math.e - 1.0,
    "integ-runge-1e-8": 0.4 * math.atan(5.0),
    "integ-nearsing-1e-6": 2.0 * (math.sqrt(1.001) - math.sqrt(0.001)),
    "integ-nearsing-1e-10": 2.0 * (math.sqrt(1.001) - math.sqrt(0.001)),
    "integ-peak-1e-8": 10.0 * (math.atan(7.0) + math.atan(3.0)),
    "integ-oscillatory-1e-10": math.sin(20.0) / 20.0,
    "integ-gaussian-1e-12": 0.5 * math.sqrt(math.pi) * math.erf(1.0),
}


def test_quadrature_cost_reproduces_benchmark_minimum_evaluations():
    P = _benchmark_problems()
    view = full_view()
    instances = [i for i in P.INSTANCES if i.family == "integration"]
    assert {i.id for i in instances} == INTEGRATION_EXACT.keys()
    for inst in instances:
        a, b = inst.data["interval"]
        tau = inst.data["tau"]
        found = None
        for n in range(1, P.N_MAX + 1):
            for rule in P.QUADRATURE_RULES:
                args = {"integrand": inst.data["integrand"], "a": a, "b": b, "rule": rule, "n": n}
                try:
                    out = view.call("numerics.quadrature_cost.v1", args, caller=AGENT)
                except ToolError as exc:
                    assert exc.code == Code.OUT_OF_DOMAIN  # trapezoid n=1, simpson with even n
                    continue
                assert abs(out["reference"] - INTEGRATION_EXACT[inst.id]) <= 1e-13 * max(1.0, abs(out["reference"]))
                if out["abs_error"] <= tau:
                    found = (float(P.QUADRATURE_RULES.index(rule)), float(n))
                    break
            if found:
                break
        assert found == inst.reference.x and found[1] == inst.reference.value, (inst.id, found, inst.reference)
    report = replay(view.log, builtin_registry())
    assert report.ok and {e.status for e in report.entries} == {"exact"}


def test_quadrature_cost_rules_match_the_benchmark_definitions():
    import numpy as np

    P = _benchmark_problems()
    view = full_view()
    f = lambda t: 1.0 / np.sqrt(t + 0.001)  # noqa: E731
    for index, rule in enumerate(P.QUADRATURE_RULES):
        for n in (1, 2, 3, 4, 9, 64):
            expected = P.quadrature(f, 0.0, 1.0, index, n)
            args = {"integrand": "1/sqrt(t+0.001)", "a": 0.0, "b": 1.0, "rule": rule, "n": n}
            if math.isnan(expected):
                assert call_code(view, "numerics.quadrature_cost.v1", args) == Code.OUT_OF_DOMAIN
            else:
                got = view.call("numerics.quadrature_cost.v1", args, caller=AGENT)
                assert abs(got["value"] - expected) <= 1e-14 * abs(expected) and got["evaluations"] == n


def test_layered_wall_matches_hand_calculation_and_conserves_flux():
    view = full_view()
    # one layer: R = 1/10 + 0.1/1 + 1/10 = 0.3 m^2 K/W, q = 30 K / 0.3 = 100 W/m^2
    one = view.call("heat.layered_wall.v1", {"thicknesses": [0.1], "conductivities": [1.0], "t_hot": 330.0,
                                             "t_cold": 300.0, "h_hot": 10.0, "h_cold": 10.0}, caller=AGENT)
    assert abs(one["heat_flux"] - 100.0) <= 1e-12 and abs(one["resistance"] - 0.3) <= 1e-15
    assert all(abs(t - e) <= 1e-12 for t, e in zip(one["surface_temperatures"], (320.0, 310.0)))
    # brick, insulation, plaster: the same flux crosses every film and layer
    L, k, h_hot, h_cold, t_hot, t_cold = [0.1, 0.05, 0.015], [0.72, 0.04, 0.22], 25.0, 8.0, 295.0, 263.0
    out = view.call("heat.layered_wall.v1", {"thicknesses": L, "conductivities": k, "t_hot": t_hot,
                                             "t_cold": t_cold, "h_hot": h_hot, "h_cold": h_cold}, caller=AGENT)
    q, s = out["heat_flux"], out["surface_temperatures"]
    assert len(s) == 4
    fluxes = [h_hot * (t_hot - s[0]), *(k[i] * (s[i] - s[i + 1]) / L[i] for i in range(3)), h_cold * (s[3] - t_cold)]
    assert all(abs(x - q) <= 1e-10 * q for x in fluxes)
    assert abs(q - 32.0 / (1 / 25 + 0.1 / 0.72 + 0.05 / 0.04 + 0.015 / 0.22 + 1 / 8)) <= 1e-12
    assert call_code(view, "heat.layered_wall.v1", {"thicknesses": L, "conductivities": k[:2], "t_hot": t_hot,
                                                    "t_cold": t_cold, "h_hot": 1.0, "h_cold": 1.0}) == Code.OUT_OF_DOMAIN


def slab_series(length, diffusivity, time_s, position, t0, tb, terms=4000):
    """Analytic solution: T = Tb + (T0 - Tb) sum_{n odd} 4/(n pi) sin(n pi x / L) exp(-alpha (n pi / L)^2 t)."""
    total = sum(4.0 / (n * math.pi) * math.sin(n * math.pi * position / length)
                * math.exp(-diffusivity * (n * math.pi / length) ** 2 * time_s) for n in range(1, 2 * terms, 2))
    return tb + (t0 - tb) * total


def test_transient_slab_matches_series_solution_with_measured_orders():
    view = full_view()
    base = {"length": 0.1, "diffusivity": 1e-5, "time": 100.0, "position": 0.05, "initial_temperature": 400.0,
            "boundary_temperature": 300.0}
    exact = slab_series(0.1, 1e-5, 100.0, 0.05, 400.0, 300.0)

    def error(nx, nt, theta):
        out = view.call("heat.transient_slab.v1", {**base, "nx": nx, "nt": nt, "theta": theta}, caller=AGENT)
        return abs(out["temperature"] - exact)

    def orders(errors):
        return [math.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]

    space = orders([error(nx, 4000, 0.5) for nx in (10, 20, 40, 80)])
    implicit_time = orders([error(400, nt, 1.0) for nt in (10, 20, 40, 80)])
    cn_time = orders([error(400, nt, 0.5) for nt in (10, 20, 40, 80)])
    explicit = orders([error(nx, math.ceil(1e-5 * 100.0 / (0.4 * (0.1 / nx) ** 2)), 0.0) for nx in (10, 20, 40, 80)])
    assert all(1.9 <= p <= 2.5 for p in space), space
    assert all(0.9 <= p <= 1.1 for p in implicit_time), implicit_time
    assert all(1.9 <= p <= 2.3 for p in cn_time), cn_time
    assert all(1.9 <= p <= 2.1 for p in explicit), explicit
    assert error(80, 4000, 0.5) <= 2e-4
    # an explicit step past r = 1/2 is refused, not run into an oscillating answer
    assert call_code(view, "heat.transient_slab.v1", {**base, "nx": 80, "nt": 100, "theta": 0.0}) == Code.OUT_OF_DOMAIN


# ---------------------------------------------------------------------------------------- validation of every call
def test_domain_violations_are_refused_not_extrapolated():
    ran = []
    registry = ToolRegistry([scalar_tool(fn=lambda x: ran.append(x) or {"value": x},
                                         domain={"x": Interval(0.0, 1.0)})])
    view = registry.view(["test.echo.v1"])
    assert call_code(view, "test.echo.v1", {"x": 1.5}) == Code.OUT_OF_DOMAIN
    assert view.call("test.echo.v1", {"x": 1.0}, caller=AGENT) == {"value": 1.0}
    assert ran == [1.0]
    builtin = full_view()
    for tool_id, x in (("math.gamma.v1", -0.5), ("math.erf.v1", 11.0), ("math.bessel_j0.v1", 1e4)):
        assert call_code(builtin, tool_id, {"x": x}) == Code.OUT_OF_DOMAIN
    slab = {"length": 0.1, "diffusivity": 1e-5, "time": 10.0, "position": 0.2, "initial_temperature": 400.0,
            "boundary_temperature": 300.0, "nx": 10, "nt": 10, "theta": 1.0}
    assert call_code(builtin, "heat.transient_slab.v1", slab) == Code.OUT_OF_DOMAIN
    assert [r.outcome for r in builtin.log.records] == ["error"] * 4


def test_invalid_arguments_are_rejected_before_the_tool_runs():
    view = full_view()
    for args in ({"x": float("nan")}, {"x": "0.5"}, {"x": True}, {}, {"x": 0.5, "y": 1.0}, [0.5]):
        assert call_code(view, "math.erf.v1", args) == Code.INVALID_ARGS, args
    assert call_code(view, "numerics.gauss_legendre.v1",
                     {"integrand": "exp(t)", "a": 0, "b": 1, "n": 2.5}) == Code.INVALID_ARGS
    assert replay(view.log, builtin_registry()).ok


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "1.0", None, [1.0]])
def test_nan_inf_and_ill_typed_outputs_are_rejected(bad):
    registry = ToolRegistry([scalar_tool(fn=lambda x: {"value": bad})])
    view = registry.view(["test.echo.v1"])
    assert call_code(view, "test.echo.v1", {"x": 1.0}) == Code.INVALID_OUTPUT
    record = view.log.records[-1]
    assert record.result is None and record.outcome == "error"


def test_malformed_outputs_and_failing_tools_are_errors_not_results():
    import numpy as np

    cases = {"test.missing.v1": lambda x: {},
             "test.extra.v1": lambda x: {"value": x, "debug": 1},
             "test.notdict.v1": lambda x: x,
             "test.array.v1": lambda x: {"value": np.array([x])},
             "test.raises.v1": lambda x: 1 / 0}
    registry = ToolRegistry([scalar_tool(tool_id, fn) for tool_id, fn in cases.items()])
    view = registry.view(cases)
    codes = {tool_id: call_code(view, tool_id, {"x": 1.0}) for tool_id in cases}
    assert codes == {**{k: Code.INVALID_OUTPUT for k in cases}, "test.raises.v1": Code.TOOL_FAILED}
    assert replay(view.log, registry).ok


# ------------------------------------------------------------------------------------------- registration contract
def test_registration_rejects_duplicates_and_signature_schema_mismatches():
    registry = ToolRegistry([scalar_tool()])

    def code_of(tool):
        with pytest.raises(ToolError) as info:
            registry.register(tool)
        return info.value.code

    assert code_of(scalar_tool()) == Code.DUPLICATE_TOOL
    assert code_of(scalar_tool("test.wrong_name.v1", lambda y: {"value": y})) == Code.SIGNATURE_MISMATCH
    assert code_of(scalar_tool("test.extra_param.v1", lambda x, y: {"value": x})) == Code.SIGNATURE_MISMATCH
    assert code_of(scalar_tool("test.defaulted.v1", lambda x=1.0: {"value": x})) == Code.SIGNATURE_MISMATCH
    assert code_of(scalar_tool("test.kwargs.v1", lambda **kw: {"value": 0.0})) == Code.SIGNATURE_MISMATCH
    for bad_id in ("test.unversioned", "math.erf@1", "Test.caps.v1"):
        assert code_of(scalar_tool(bad_id)) == Code.INVALID_SPEC, bad_id
    unsound = {
        "non-SI unit": {"input_units": {"x": "km"}},
        "missing unit": {"output_units": {}},
        "open input schema": {"input_schema": {"type": "object", "properties": {"x": {"type": "number"}},
                                               "required": ["x"]}},
        "value field not numeric": {"value_field": "label"},
        "bad cost class": {"cost": CostModel("free")},
        "cost reads a non-input": {"cost": CostModel("cheap", "3*n")},
        "domain on a non-input": {"domain": {"y": Interval(0.0, 1.0)}},
        "reversed domain": {"domain": {"x": Interval(1.0, 0.0)}},
        "no assumptions": {"assumptions": ()},
        "no tests": {"tested_by": ()},
    }
    for name, change in unsound.items():
        assert code_of(scalar_tool("test.unsound.v1", **change)) == Code.INVALID_SPEC, name
    assert [s.tool_id for s in registry.list()] == ["test.echo.v1"]


def test_stochastic_tool_requires_a_seed():
    import random

    unseeded = scalar_tool("test.noise.v1", lambda x: {"value": x + random.random()}, deterministic=False)
    with pytest.raises(ToolError) as info:
        ToolRegistry([unseeded])
    assert info.value.code == Code.INVALID_SPEC

    def noisy(x, seed):
        return {"value": x + random.Random(seed).gauss(0.0, 1.0)}

    seeded = scalar_tool("test.noise.v1", noisy, deterministic=False, input_units={"x": "", "seed": ""}, input_schema={
        "type": "object", "properties": {"x": {"type": "number"}, "seed": {"type": "integer"}},
        "required": ["x", "seed"], "additionalProperties": False})
    registry = ToolRegistry([seeded])
    view = registry.view(["test.noise.v1"])
    assert call_code(view, "test.noise.v1", {"x": 1.0}) == Code.INVALID_ARGS
    first = view.call("test.noise.v1", {"x": 1.0, "seed": 7}, caller=AGENT)
    assert view.call("test.noise.v1", {"x": 1.0, "seed": 7}, caller=AGENT) == first
    assert [e.status for e in replay(view.log, ToolRegistry([seeded])).entries] == ["exact"] * 3


def test_every_builtin_is_sound_and_names_tests_that_exist():
    defined = {node.name for node in ast.parse(Path(__file__).read_text(encoding="utf-8")).body
               if isinstance(node, ast.FunctionDef)}
    assert len({t.spec.tool_id for t in BUILTIN_TOOLS}) == len(BUILTIN_TOOLS) == 13
    for tool in BUILTIN_TOOLS:
        assert spec_errors(tool.spec) == []
        json.dumps(tool.spec.to_dict(), allow_nan=False)
        for ref in tool.spec.tested_by:
            path, name = ref.split("::")
            assert path == "tests/test_e14_04_tool_registry.py" and name in defined, (tool.spec.tool_id, ref)


# ------------------------------------------------------------------------------------------------ log and replay
def mixed_session():
    view = full_view()
    view.call("math.gamma.v1", {"x": 4.5}, caller=AGENT)
    view.call("numerics.gauss_legendre.v1", {"integrand": "cos(t)", "a": 0, "b": 2, "n": 6}, caller=AGENT)
    view.call("heat.layered_wall.v1", {"thicknesses": [0.2, 0.05], "conductivities": [1.4, 0.04], "t_hot": 293.0,
                                       "t_cold": 268.0, "h_hot": 8.0, "h_cold": 25.0}, caller=Caller("other"))
    view.call("heat.transient_slab.v1", {"length": 0.02, "diffusivity": 1e-6, "time": 60.0, "position": 0.005,
                                         "initial_temperature": 350.0, "boundary_temperature": 290.0, "nx": 40,
                                         "nt": 200, "theta": 0.5}, caller=AGENT)
    for tool_id, args in (("math.gamma.v1", {"x": -3.0}), ("math.unknown.v1", {"x": 1.0})):
        with pytest.raises(ToolError):
            view.call(tool_id, args, caller=AGENT)
    return view.log


def test_call_log_round_trips_and_replays_exactly_on_a_fresh_registry():
    log = mixed_session()
    assert log.verify() == []
    text = log.to_json()
    loaded = CallLog.from_json(text)
    assert loaded.records == log.records and loaded.to_json() == text
    report = replay(loaded, builtin_registry())
    assert report.ok and report.integrity == ()
    assert [e.status for e in report.entries] == ["exact"] * 5 + ["not_executed"]
    assert [r.caller for r in loaded.records] == ["agent-under-test", "agent-under-test", "other",
                                                  "agent-under-test", "agent-under-test", "agent-under-test"]
    assert loaded.usage("agent-under-test") == log.usage("agent-under-test") == (4, 1 + 18 + 200 + 0)
    data = json.loads(text)
    for broken in ({**data, "format": "other"},
                   {**data, "records": [{**data["records"][0], "extra": 1}]},
                   {**data, "records": [{**data["records"][4], "error": {"code": "bogus", "message": ""}}]}):
        with pytest.raises(ValueError):
            CallLog.from_json(json.dumps(broken))


def test_record_digests_ignore_last_bit_noise():
    log = mixed_session()
    body = log.records[0].body()
    noisy = dict(body, result={"value": math.nextafter(body["result"]["value"], math.inf)})
    assert noisy != body and record_digest(noisy) == record_digest(body)


def _tampered(log, edit):
    data = json.loads(log.to_json())
    edit(data["records"])
    return CallLog.from_json(json.dumps(data))


def _reseal(log):
    data = json.loads(log.to_json())
    prev = ""
    for i, record in enumerate(data["records"]):
        record.update(seq=i, prev_digest=prev)
        record.pop("digest")
        record["digest"] = prev = record_digest(record)
    return CallLog.from_json(json.dumps(data))


@pytest.mark.parametrize("edit", [
    lambda rs: rs[0]["args"].update(x=4.6),
    lambda rs: rs[1]["result"].update(value=rs[1]["result"]["value"] + 1e-6),
    lambda rs: rs.insert(0, rs.pop(2)),
    lambda rs: rs.pop(1),
], ids=["args", "result", "order", "dropped"])
def test_tampered_log_fails_its_digest_and_replay(edit):
    tampered = _tampered(mixed_session(), edit)
    assert tampered.verify() != []
    assert not replay(tampered, builtin_registry()).ok


@pytest.mark.parametrize("edit", [
    lambda rs: rs[0]["args"].update(x=4.6),
    lambda rs: rs[1]["result"].update(value=rs[1]["result"]["value"] + 1e-6),
    lambda rs: rs[4].update(error={"code": "invalid_args", "message": "forged"}),
], ids=["args", "result", "error"])
def test_resealed_tampering_is_caught_by_re_execution(edit):
    forged = _reseal(_tampered(mixed_session(), edit))
    assert forged.verify() == []
    report = replay(forged, builtin_registry())
    assert not report.ok and len(report.mismatches) == 1


def test_replay_tolerance_is_declared_per_tool():
    jitter = iter([0.0, 1e-12])

    def jittery(x):
        return {"value": x + next(jitter)}

    tolerant = scalar_tool("test.jitter.v1", jittery, replay_tolerance=1e-9)
    view = ToolRegistry([tolerant]).view(["test.jitter.v1"])
    view.call("test.jitter.v1", {"x": 0.3}, caller=AGENT)
    assert [e.status for e in replay(view.log, ToolRegistry([tolerant])).entries] == ["within_tolerance"]
    jitter = iter([0.0, 1e-12])
    strict = scalar_tool("test.jitter.v1", jittery)
    view = ToolRegistry([strict]).view(["test.jitter.v1"])
    view.call("test.jitter.v1", {"x": 0.3}, caller=AGENT)
    assert [e.status for e in replay(view.log, ToolRegistry([strict])).entries] == ["mismatch"]


# ------------------------------------------------------------------------------------------------------- budgets
def test_budget_caps_calls_evaluations_and_wall_time():
    view = full_view()
    two_calls = Caller("two", CallBudget(max_calls=2))
    view.call("math.erf.v1", {"x": 0.1}, caller=two_calls)
    view.call("math.erf.v1", {"x": 0.2}, caller=two_calls)
    assert call_code(view, "math.erf.v1", {"x": 0.3}, two_calls) == Code.BUDGET_EXCEEDED
    assert call_code(view, "math.erf.v1", {"x": "bad"}, two_calls) == Code.BUDGET_EXCEEDED  # counted before args
    view.call("math.erf.v1", {"x": 0.3}, caller=AGENT)  # budgets are per caller

    twenty = Caller("twenty", CallBudget(max_evaluations=20))
    gl = {"integrand": "exp(t)", "a": 0, "b": 1}
    view.call("numerics.gauss_legendre.v1", {**gl, "n": 5}, caller=twenty)  # charged 3n = 15
    assert call_code(view, "numerics.gauss_legendre.v1", {**gl, "n": 2}, twenty) == Code.BUDGET_EXCEEDED
    view.call("numerics.gauss_legendre.v1", {**gl, "n": 1}, caller=twenty)  # 15 + 3 = 18 still fits
    assert view.log.usage("twenty") == (2, 18.0)

    slow = ToolRegistry([scalar_tool("test.slow.v1", lambda x: time.sleep(0.05) or {"value": x})])
    slow_view = slow.view(["test.slow.v1"])
    assert call_code(slow_view, "test.slow.v1", {"x": 1.0}, Caller("hurried", CallBudget(max_wall_s_per_call=0.01))) \
        == Code.TIMEOUT
    assert slow_view.log.records[-1].result is None
    assert [r.outcome for r in view.log.records].count("refused") == 3


# ----------------------------------------------------------------------------------------------- IR integration
def slab_problem(unit="K", time_arg="t"):
    return OptimizationProblem(
        id="slab-cooling-time",
        variables=(Variable("t", lower=0.5, upper=30.0, unit="min"),),
        parameters=(Parameter("L", 2.0, "cm"), Parameter("alpha", 1e-6, "m^2/s"), Parameter("t0", 350.0, "K"),
                    Parameter("tb", 290.0, "K")),
        objectives=(Objective("cooling_time", "t", unit="min"),),
        constraints=(Constraint("centre_cool", ToolCall("heat.transient_slab.v1", {
            "length": "L", "diffusivity": "alpha", "time": time_arg, "position": "L/2", "initial_temperature": "t0",
            "boundary_temperature": "tb", "nx": "40", "nt": "400", "theta": "0.5"}), "<=", 300.0, unit=unit),),
    )


def test_ir_tool_output_unit_mismatch_is_rejected_at_binding():
    view = full_view()
    for problem in (slab_problem(unit="W"), slab_problem(time_arg="L")):
        assert validate_problem(problem) == []  # the IR alone cannot see either mistake
        with pytest.raises(ToolError) as info:
            bind_problem(problem, view, AGENT)
        assert info.value.code == Code.UNIT_MISMATCH
    assert len(view.log) == 0
    with pytest.raises(ToolError) as info:
        bind_problem(slab_problem(), builtin_registry().view(["math.erf.v1"]), AGENT)
    assert info.value.code == Code.NOT_ALLOWED


def test_ir_problem_with_tool_constraint_converts_units_and_solves():
    from scipy.optimize import brentq

    problem = slab_problem()
    assert validate_problem(problem) == []
    view = full_view()
    tools = bind_problem(problem, view, AGENT)
    residual = lambda t: problem.evaluate({"t": t}, tools=tools).residuals["centre_cool"]  # noqa: E731
    t_min = brentq(residual, 0.5, 30.0, xtol=1e-10)  # minutes; the temperature falls monotonically
    expected = brentq(lambda s: slab_series(0.02, 1e-6, s, 0.01, 350.0, 290.0) - 300.0, 30.0, 1800.0, xtol=1e-9) / 60
    assert abs(t_min - expected) <= 1e-3 * expected, (t_min, expected)
    call = view.log.records[0]
    assert call.args["length"] == 0.02 and call.args["time"] == pytest.approx(60 * 0.5) and call.args["nx"] == 40
    assert replay(view.log, builtin_registry()).ok


def test_ir_problem_with_tool_objective_solves_end_to_end():
    from scipy.optimize import minimize_scalar

    problem = OptimizationProblem(
        id="bessel-j0-minimum", variables=(Variable("x", lower=2.0, upper=5.0, initial=3.0),),
        objectives=(Objective("j0", ToolCall("math.bessel_j0.v1", {"x": "x"})),))
    assert validate_problem(problem) == []
    view = full_view()
    tools = bind_problem(problem, view, AGENT)
    found = minimize_scalar(lambda x: problem.evaluate({"x": x}, tools=tools).objectives["j0"],
                            bounds=(2.0, 5.0), method="bounded", options={"xatol": 1e-10})
    # J0' = -J1, so the minimum is at the first zero of J1 (mpmath besseljzero(1, 1), 30 digits)
    assert found.success and abs(found.x - 3.83170597020751231561) <= 1e-6
    assert abs(found.fun - -0.402759395702552972096) <= 1e-12
    assert len(view.log) == found.nfev
    assert replay(view.log, builtin_registry()).ok


def test_import_stays_light():
    source = ("import sys, orbitopt.agent.tools as t; t.builtin_registry();"
              "heavy = [m for m in sys.modules if m.split('.')[0] in ('numpy', 'scipy')"
              " or m.startswith(('orbitopt.mission', 'orbitopt.flightlab', 'orbitopt.rendezvous'))];"
              "print(heavy)")
    out = subprocess.run([sys.executable, "-c", source], cwd=REPO, capture_output=True, text=True,
                         env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"}, check=True)
    assert out.stdout.strip() == "[]"
