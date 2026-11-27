"""E14-01: the optimization IR contract (ADR optimization-0001).

Schema validity, exact round trip, machine-independent digest, unit checking,
specific rejection of invalid problems, safe expression evaluation, the
hard-constraint fingerprint, the result contract and the documented worked
example.
"""

import ast
import dataclasses
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

from orbitopt.optimization import (
    PROBLEM_SCHEMA,
    PROBLEM_SCHEMAS,
    RESULT_SCHEMA,
    RESULT_SCHEMAS,
    STATUSES,
    SYMBOLS,
    Budget,
    Constraint,
    ErrorCode,
    EvaluationError,
    Evidence,
    MAX_EXPANDED_NODES,
    MAX_NODES,
    Expression,
    ExpressionError,
    InvalidProblemError,
    Objective,
    OptimizationProblem,
    OptimizationResult,
    Parameter,
    ParetoPoint,
    Provenance,
    Termination,
    ToolCall,
    ToolCallError,
    UnitError,
    Variable,
    VerificationPolicy,
    canonical_digest,
    tolerance_scale,
    parse_problem,
    parse_unit,
    schema_errors,
    schema_for,
    validate_problem,
)

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "optimization" / "ir.md"
replace = dataclasses.replace


def quadratic(**overrides) -> OptimizationProblem:
    """min x^2 + y^2 s.t. x + y >= 1, x, y >= 0 (the documented worked example)."""
    fields = dict(
        id="quadratic",
        variables=(Variable("x", lower=0.0), Variable("y", lower=0.0)),
        objectives=(Objective("f", "x**2 + y**2"),),
        constraints=(Constraint("halfplane", "x + y", ">=", 1.0),),
    )
    return OptimizationProblem(**{**fields, **overrides})


def rich() -> OptimizationProblem:
    """Every feature of the IR in one problem."""
    return OptimizationProblem(
        id="transfer-demo",
        title="Rich IR example",
        description="Exercises every field.",
        variables=(
            Variable("dv1", lower=0.0, upper=3.5, unit="km/s", initial=1.0, description="first burn"),
            Variable("tof", lower=600.0, upper=None, unit="s"),
            Variable("n_rev", kind="integer", lower=0, upper=3),
            Variable("use_drag", kind="binary", initial=1),
            Variable("thrust", kind="discrete", choices=(200.0, 400.0, 800.0), unit="N"),
            Variable("theta", lower=-90.0, upper=90.0, unit="deg"),
        ),
        parameters=(Parameter("dv_max", 4000.0, unit="m/s"), Parameter("mass", 1200.0, unit="kg")),
        objectives=(
            Objective("total_dv", "dv1", unit="km/s"),
            Objective("payload", ToolCall("mission.payload_margin", {"dv": "dv1", "t": "tof"},
                                          {"model": "two_body", "stages": [1, 2.5], "drag": True}, "margin"),
                      sense="maximize", unit="kg", origin="compiler"),
        ),
        constraints=(
            Constraint("dv_cap", "dv1", "<=", 4.0, unit="km/s"),
            Constraint("accel", "thrust / mass", "<=", 1.0, unit="m/s^2"),
            Constraint("aim", "sin(theta)", "==", 0.0, origin="agent"),
            Constraint("tof_floor", "tof", ">=", 0.25, unit="h", origin="user", locked=False),
        ),
        budget=Budget(max_evaluations=5000, max_iterations=200, max_wall_s=60.0, max_tokens=20000),
        verification=VerificationPolicy(required=("solver_converged", "constraints_satisfied"),
                                        independent_methods=("alternate_solver",), constraint_tol=1e-8),
        metadata={"source": "test", "nl_prompt": "minimise dv"},
    )


def codes(problem):
    return [(e.code, e.path) for e in validate_problem(problem)]


def test_well_formed_problems_are_valid_and_schema_valid():
    for problem in (quadratic(), rich()):
        assert validate_problem(problem) == []
        assert schema_errors(problem.to_dict(), PROBLEM_SCHEMA) == []


def test_round_trip_through_json_is_exact_and_keeps_the_digest():
    for problem in (quadratic(), rich()):
        text = json.dumps(problem.to_dict())
        back = parse_problem(json.loads(text))
        assert back == problem
        assert back.to_dict() == problem.to_dict()
        assert back.digest() == problem.digest()


def test_minimal_dict_takes_defaults_and_locks_user_constraints():
    problem = parse_problem({
        "schemaVersion": "1.1", "id": "p",
        "variables": [{"name": "x", "lower": -1, "upper": 1}],
        "objectives": [{"name": "f", "expression": "x**2"}],
        "constraints": [{"name": "c", "expression": "x", "sense": "<=", "rhs": 0.5},
                        {"name": "a", "expression": "x", "sense": ">=", "rhs": -0.5, "origin": "agent"}],
    })
    assert [c.locked for c in problem.constraints] == [True, False]
    assert problem.budget == Budget() and problem.verification == VerificationPolicy()


@pytest.mark.parametrize("mutate, path", [
    (lambda d: d["constraints"][0].pop("rhs"), "/constraints/0"),
    (lambda d: d.pop("objectives"), "/"),
    (lambda d: d["variables"][0].update(kind="real"), "/variables/0/kind"),
    (lambda d: d["variables"][0].update(lower="0"), "/variables/0/lower"),
    (lambda d: d.update(surprise=1), "/"),
    (lambda d: d["budget"].update(max_evaluations=1.5), "/budget/max_evaluations"),
])
def test_schema_rejects_malformed_dicts_with_a_path(mutate, path):
    data = quadratic().to_dict()
    mutate(data)
    with pytest.raises(InvalidProblemError) as info:
        parse_problem(data)
    assert {(e.code, e.path) for e in info.value.errors} == {(ErrorCode.SCHEMA, path)}


def test_invalid_problem_error_carries_every_error_at_once():
    bad = quadratic(variables=(Variable("x", lower=2.0, upper=1.0), Variable("y"), Variable("x")),
                    budget=Budget(max_iterations=0))
    data = json.loads(json.dumps(bad.to_dict()))
    with pytest.raises(InvalidProblemError) as info:
        parse_problem(data)
    assert {e.code for e in info.value.errors} == {
        ErrorCode.BOUNDS_ORDER, ErrorCode.DUPLICATE_NAME, ErrorCode.NON_POSITIVE_BUDGET}
    assert set(info.value.errors) == set(validate_problem(bad))


# Pinned: the digest is a cross-machine contract. If the canonical form must
# change, that is a new IR version, not an edit to this number. IR 1.0 pinned
# b246af29...; the 1.1 digest hashes the 1.1 form (schemaVersion "1.1").
QUADRATIC_DIGEST = "3c09d1026773084a2e7a003cd3dde04439b5359db9f4447211d35e125510016c"


def test_digest_is_pinned_across_machines():
    assert quadratic().digest() == QUADRATIC_DIGEST


QUADRATIC_1_0 = {
    "schemaVersion": "1.0", "id": "quadratic",
    "variables": [{"name": "x", "lower": 0.0}, {"name": "y", "lower": 0.0}],
    "objectives": [{"name": "f", "expression": "x**2 + y**2"}],
    "constraints": [{"name": "halfplane", "expression": "x + y", "sense": ">=", "rhs": 1.0}],
}


def test_a_1_0_document_parses_into_the_1_1_problem_and_writes_back_as_1_1():
    assert schema_for(QUADRATIC_1_0, PROBLEM_SCHEMAS) == "optimization-problem-1.0.json"
    problem = parse_problem(QUADRATIC_1_0)
    assert problem == quadratic()
    assert problem.to_dict()["schemaVersion"] == "1.1"
    assert problem.digest() == QUADRATIC_DIGEST
    assert schema_errors(problem.to_dict(), PROBLEM_SCHEMA) == []


def test_an_unknown_schema_version_is_a_schema_error():
    with pytest.raises(InvalidProblemError) as info:
        parse_problem({**QUADRATIC_1_0, "schemaVersion": "2.0"})
    assert {(e.code, e.path) for e in info.value.errors} == {(ErrorCode.SCHEMA, "/schemaVersion")}


def test_a_1_0_result_document_parses_with_the_1_1_defaults():
    data = {"schemaVersion": "1.0", "problem_digest": "0" * 64, "solver_id": "s", "termination": "converged",
            "evidence": [{"kind": "solver", "summary": "ran"}]}
    assert schema_errors(data, schema_for(data, RESULT_SCHEMAS)) == []
    result = OptimizationResult.from_dict(data)
    assert result == OptimizationResult(problem_digest="0" * 64, solver_id="s", termination=Termination.CONVERGED,
                                        evidence=(Evidence("solver", "ran"),))
    assert schema_errors(result.to_dict(), RESULT_SCHEMA) == []


def test_digest_ignores_key_order_containers_and_last_bit_noise():
    problem = rich()
    data = problem.to_dict()
    reordered = json.loads(json.dumps(dict(reversed(list(data.items())))))
    for item in reordered["variables"]:
        item.update(dict(reversed(list(item.items()))))
    assert OptimizationProblem.from_dict(reordered).digest() == problem.digest()
    assert canonical_digest({"a": [1, 2.0], "b": {"y": 1, "x": -0.0}}) == \
        canonical_digest({"b": {"x": 0.0, "y": 1.0}, "a": (1.0, 2)})
    noisy = quadratic(constraints=(Constraint("halfplane", "x + y", ">=", 0.1 + 0.2),))
    clean = quadratic(constraints=(Constraint("halfplane", "x + y", ">=", 0.3),))
    assert 0.1 + 0.2 != 0.3 and noisy.digest() == clean.digest()


def test_digest_ignores_titles_descriptions_and_metadata():
    problem = quadratic()
    cosmetic = replace(problem, title="t", description="d", metadata={"k": 1},
                       variables=tuple(replace(v, description="note") for v in problem.variables))
    assert cosmetic.digest() == problem.digest()


@pytest.mark.parametrize("change", [
    lambda p: replace(p, constraints=(replace(p.constraints[0], rhs=1.1),)),
    lambda p: replace(p, constraints=(replace(p.constraints[0], sense="<="),)),
    lambda p: replace(p, constraints=()),
    lambda p: replace(p, variables=(replace(p.variables[0], lower=-1.0), p.variables[1])),
    lambda p: replace(p, variables=(replace(p.variables[0], kind="integer"), p.variables[1])),
    lambda p: replace(p, variables=(replace(p.variables[0], unit="m"), p.variables[1])),
    lambda p: replace(p, variables=(replace(p.variables[0], initial=0.2), p.variables[1])),
    lambda p: replace(p, objectives=(replace(p.objectives[0], expression=Expression("x**2 + 2*y**2")),)),
    lambda p: replace(p, objectives=(replace(p.objectives[0], sense="maximize"),)),
    lambda p: replace(p, parameters=(Parameter("k", 1.0),)),
    lambda p: replace(p, budget=Budget(max_evaluations=10)),
    lambda p: replace(p, verification=VerificationPolicy(constraint_tol=1e-3)),
    lambda p: replace(p, id="other"),
])
def test_digest_changes_with_every_semantic_field(change):
    assert change(quadratic()).digest() != quadratic().digest()


def test_fingerprint_survives_changes_that_do_not_weaken_the_user_problem():
    problem = rich()
    fingerprint = problem.hard_constraint_fingerprint()
    harmless = [
        replace(problem, constraints=problem.constraints + (Constraint("extra", "dv1", ">=", 0.1, unit="km/s",
                                                                       origin="agent"),)),
        replace(problem, constraints=tuple(reversed(problem.constraints))),
        replace(problem, variables=(replace(problem.variables[0], initial=2.0),) + problem.variables[1:]),
        replace(problem, budget=Budget(max_evaluations=10), title="renamed", metadata={}),
        replace(problem, objectives=problem.objectives[:1]),  # the dropped objective is compiler-origin
        replace(problem, constraints=tuple(replace(c, rhs=0.1) if c.name == "tof_floor" else c
                                           for c in problem.constraints)),  # user unlocked it
    ]
    for variant in harmless:
        assert validate_problem(variant) == []
        assert variant.hard_constraint_fingerprint() == fingerprint


def test_fingerprint_catches_every_weakening():
    problem = rich()
    dv_cap = problem.constraints[0]
    weakened = [
        replace(problem, constraints=(replace(dv_cap, rhs=5.0),) + problem.constraints[1:]),
        replace(problem, constraints=problem.constraints[1:]),
        replace(problem, constraints=(replace(dv_cap, origin="agent", locked=False),) + problem.constraints[1:]),
        replace(problem, variables=(replace(problem.variables[0], upper=10.0),) + problem.variables[1:]),
        replace(problem, variables=problem.variables[:2] + (replace(problem.variables[2], kind="continuous"),)
                + problem.variables[3:]),
        replace(problem, parameters=(Parameter("dv_max", 9000.0, unit="m/s"), problem.parameters[1])),
        replace(problem, objectives=(replace(problem.objectives[0], expression=Expression("0.0 * dv1")),)
                + problem.objectives[1:]),
    ]
    for variant in weakened:
        assert variant.hard_constraint_fingerprint() != problem.hard_constraint_fingerprint()


@pytest.mark.parametrize("text, scale", [
    ("", 1.0), ("1", 1.0), ("km/s", 1e3), ("kg*m/s^2", 1.0), ("m**3/(kg*s**2)", 1.0),
    ("1/s", 1.0), ("deg", math.pi / 180), ("km^(1/2)", 1e3 ** 0.5), ("kN*km", 1e6),
])
def test_unit_strings_parse_to_their_si_scale(text, scale):
    assert parse_unit(text).scale == pytest.approx(scale, rel=1e-15)


def test_equivalent_units_share_a_dimension_and_angles_are_dimensionless():
    assert parse_unit("N").dims == parse_unit("kg*m/s^2").dims
    assert parse_unit("J/s").dims == parse_unit("W").dims
    assert parse_unit("deg").dimensionless and parse_unit("rad").dimensionless
    assert parse_unit("m").dims != parse_unit("s").dims


@pytest.mark.parametrize("text", ["furlong", "m/", "m^x", "m + s", "2*m", "m^1.5", "m^99", "m.s", "kg(m)"])
def test_malformed_or_unknown_units_are_rejected(text):
    with pytest.raises(UnitError):
        parse_unit(text)


def test_unit_table_agrees_with_astropy():
    u = pytest.importorskip("astropy.units")
    bases = [u.m, u.kg, u.s, u.A, u.K, u.mol]
    for symbol, (scale, dims) in SYMBOLS.items():
        target = u.dimensionless_unscaled
        for base, power in zip(bases, dims):
            target = target * base ** power
        converted = (1 * u.Unit(symbol)).to(target, equivalencies=u.dimensionless_angles()).value
        assert converted == pytest.approx(scale, rel=1e-12), symbol


def test_mixed_scale_units_evaluate_in_si():
    problem = OptimizationProblem(
        id="units",
        variables=(Variable("a", unit="km"), Variable("b", unit="m"), Variable("theta", unit="deg")),
        objectives=(Objective("length", "a + b", unit="m"), Objective("s", "sin(theta)")),
        constraints=(Constraint("cap", "a + b", "<=", 2.0, unit="km"),),
    )
    assert validate_problem(problem) == []
    ev = problem.evaluate({"a": 1.0, "b": 500.0, "theta": 30.0})
    assert ev.objectives["length"] == pytest.approx(1500.0)
    assert ev.objectives["s"] == pytest.approx(0.5)
    assert ev.residuals["cap"] == pytest.approx(-0.5)


def test_compatible_units_and_dimensionless_terms_are_accepted():
    problem = OptimizationProblem(
        id="ok-units",
        variables=(Variable("r", unit="km"), Variable("v", unit="m/s"), Variable("k")),
        parameters=(Parameter("mu", 3.986004418e14, unit="m^3/s^2"),),
        objectives=(Objective("energy", "v**2/2 - mu/r", unit="J/kg"),),
        constraints=(Constraint("ratio", "k + 2*k**2 + exp(k)", "<=", 3.0),
                     Constraint("circ", "sqrt(mu/r)", ">=", 7.0, unit="km/s")),
    )
    assert validate_problem(problem) == []


X, Y = Variable("x", lower=0.0), Variable("y", lower=0.0)
L, T = Variable("x", unit="m"), Variable("y", unit="s")


def objective(source, **kw):
    return {"objectives": (Objective("f", source, **kw),)}


def constraint(source, sense=">=", rhs=1.0, **kw):
    return {"constraints": (Constraint("c", source, sense, rhs, **kw),)}


INVALID = [
    ("duplicate variable", {"variables": (X, Y, replace(Y, name="x"))}, ErrorCode.DUPLICATE_NAME,
     "/variables/2/name"),
    ("variable/parameter clash", {"parameters": (Parameter("y", 1.0),)}, ErrorCode.DUPLICATE_NAME,
     "/parameters/0/name"),
    ("keyword name", {"variables": (X, Y, Variable("lambda"))}, ErrorCode.INVALID_NAME, "/variables/2/name"),
    ("function name", {"variables": (X, Y, Variable("sin"))}, ErrorCode.INVALID_NAME, "/variables/2/name"),
    ("sum as a name", {"variables": (X, Y, Variable("sum"))}, ErrorCode.INVALID_NAME, "/variables/2/name"),
    ("duplicate constraint", {"constraints": (Constraint("c", "x + y", ">=", 1.0),) * 2},
     ErrorCode.DUPLICATE_NAME, "/constraints/1/name"),
    ("unknown name", objective("x**2 + z"), ErrorCode.UNKNOWN_NAME, "/objectives/0/expression"),
    ("no variables", {"variables": (), "objectives": (Objective("f", "1.0"),), "constraints": ()},
     ErrorCode.NO_VARIABLES, "/variables"),
    ("no objectives", {"objectives": ()}, ErrorCode.NO_OBJECTIVES, "/objectives"),
    ("lower above upper", {"variables": (replace(X, lower=2.0, upper=1.0), Y)}, ErrorCode.BOUNDS_ORDER,
     "/variables/0/lower"),
    ("nan bound", {"variables": (replace(X, lower=math.nan), Y)}, ErrorCode.NON_FINITE, "/variables/0/lower"),
    ("inf bound", {"variables": (replace(X, upper=math.inf), Y)}, ErrorCode.NON_FINITE, "/variables/0/upper"),
    ("integer with fractional bound", {"variables": (replace(X, kind="integer", upper=2.5), Y)},
     ErrorCode.INTEGER_BOUNDS, "/variables/0/upper"),
    ("discrete with no choices", {"variables": (Variable("x", kind="discrete"), Y)}, ErrorCode.EMPTY_CHOICES,
     "/variables/0/choices"),
    ("binary with bounds", {"variables": (Variable("x", kind="binary", upper=1.0), Y)}, ErrorCode.DOMAIN_FIELDS,
     "/variables/0/upper"),
    ("unknown kind", {"variables": (replace(X, kind="real"), Y)}, ErrorCode.INVALID_VALUE, "/variables/0/kind"),
    ("initial outside bounds", {"variables": (replace(X, initial=-1.0), Y)}, ErrorCode.INITIAL_OUT_OF_DOMAIN,
     "/variables/0/initial"),
    ("attribute access", objective("x.real"), ErrorCode.UNSAFE_EXPRESSION, "/objectives/0/expression"),
    ("__import__", objective("__import__('os')"), ErrorCode.UNKNOWN_FUNCTION,
     "/objectives/0/expression"),
    ("dunder name", objective("x + __builtins__"), ErrorCode.UNSAFE_EXPRESSION, "/objectives/0/expression"),
    ("subscript of an expression", objective("(x + y)[0]"), ErrorCode.UNSAFE_EXPRESSION, "/objectives/0/expression"),
    ("subscript of an undeclared name", objective("x[0] + y"), ErrorCode.UNKNOWN_NAME, "/objectives/0/expression"),
    ("slice", objective("x[0:2]"), ErrorCode.UNSAFE_EXPRESSION, "/objectives/0/expression"),
    ("non-whitelisted call", objective("open(x)"), ErrorCode.UNKNOWN_FUNCTION, "/objectives/0/expression"),
    ("keyword argument", objective("max(x, y, key=x)"), ErrorCode.UNSAFE_EXPRESSION, "/objectives/0/expression"),
    ("wrong arity", objective("atan2(x)"), ErrorCode.WRONG_ARITY, "/objectives/0/expression"),
    ("string literal", objective("x + 'y'"), ErrorCode.BAD_LITERAL, "/objectives/0/expression"),
    ("boolean literal", objective("x + True"), ErrorCode.BAD_LITERAL, "/objectives/0/expression"),
    ("lambda", objective("(lambda: x)()"), ErrorCode.UNSAFE_EXPRESSION, "/objectives/0/expression"),
    ("list comprehension", objective("sum([x for x in y])"), ErrorCode.UNSAFE_EXPRESSION, "/objectives/0/expression"),
    ("sum over a name", objective("sum(x for i in y)"), ErrorCode.UNSAFE_EXPRESSION, "/objectives/0/expression"),
    ("filtered sum", objective("sum(x for i in range(3) if i)"), ErrorCode.UNSAFE_EXPRESSION,
     "/objectives/0/expression"),
    ("generator outside sum", objective("max((x for i in range(3)), y)"), ErrorCode.UNSAFE_EXPRESSION,
     "/objectives/0/expression"),
    ("range outside sum", objective("x + range(3)"), ErrorCode.UNKNOWN_FUNCTION, "/objectives/0/expression"),
    ("fractional index", objective("sum(x for i in range(2.5))"), ErrorCode.UNSAFE_EXPRESSION,
     "/objectives/0/expression"),
    ("huge sum", objective("sum(x for i in range(1000000000))"), ErrorCode.EXPRESSION_TOO_COMPLEX,
     "/objectives/0/expression"),
    ("exponent through an index", objective("sum(x**i for i in range(200))"), ErrorCode.EXPONENT_TOO_LARGE,
     "/objectives/0/expression"),
    ("comparison", objective("x < y"), ErrorCode.UNSAFE_EXPRESSION, "/objectives/0/expression"),
    ("caret power", objective("x ^ 2"), ErrorCode.UNSAFE_EXPRESSION, "/objectives/0/expression"),
    ("huge exponent", objective("x ** 99999999999"), ErrorCode.EXPONENT_TOO_LARGE, "/objectives/0/expression"),
    ("overflowing literal", objective("x * 1e999"), ErrorCode.BAD_LITERAL, "/objectives/0/expression"),
    ("too long", objective("+".join(["x"] * 1200)), ErrorCode.EXPRESSION_TOO_COMPLEX, "/objectives/0/expression"),
    ("syntax error", objective("x +"), ErrorCode.SYNTAX_ERROR, "/objectives/0/expression"),
    ("sum of length and time", {"variables": (L, T), **objective("x + y", unit="m"), "constraints": ()},
     ErrorCode.UNIT_MISMATCH, "/objectives/0/expression"),
    ("rhs of another dimension", {"variables": (L, T), **objective("x", unit="m"), **constraint("y", unit="m")},
     ErrorCode.UNIT_MISMATCH, "/constraints/0/unit"),
    ("dimensioned trig argument", {"variables": (L, T), **objective("sin(x)"), "constraints": ()},
     ErrorCode.UNIT_MISMATCH, "/objectives/0/expression"),
    ("unknown unit", {"variables": (replace(X, unit="furlong"), Y)}, ErrorCode.UNKNOWN_UNIT, "/variables/0/unit"),
    ("constraint without rhs", constraint("x + y", rhs=None), ErrorCode.MISSING_RHS, "/constraints/0/rhs"),
    ("nan rhs", constraint("x + y", rhs=math.nan), ErrorCode.NON_FINITE, "/constraints/0/rhs"),
    ("bad sense", constraint("x + y", sense="<"), ErrorCode.INVALID_VALUE, "/constraints/0/sense"),
    ("locked agent constraint", constraint("x + y", origin="agent", locked=True), ErrorCode.LOCKED_NON_USER,
     "/constraints/0/locked"),
    ("zero evaluation budget", {"budget": Budget(max_evaluations=0)}, ErrorCode.NON_POSITIVE_BUDGET,
     "/budget/max_evaluations"),
    ("negative wall budget", {"budget": Budget(max_wall_s=-1.0)}, ErrorCode.NON_POSITIVE_BUDGET,
     "/budget/max_wall_s"),
    ("zero tolerance", {"verification": VerificationPolicy(constraint_tol=0.0)}, ErrorCode.INVALID_TOLERANCE,
     "/verification/constraint_tol"),
    ("unknown status", {"verification": VerificationPolicy(required=("happy",))}, ErrorCode.INVALID_VALUE,
     "/verification/required/0"),
    ("bad tool id", objective(ToolCall("Not A Tool", {"a": "x"})), ErrorCode.INVALID_NAME,
     "/objectives/0/expression/tool_id"),
    ("unsafe tool argument", objective(ToolCall("orbit.dv", {"a": "x.__class__"})), ErrorCode.UNSAFE_EXPRESSION,
     "/objectives/0/expression/args/a"),
    ("argument both expression and constant", objective(ToolCall("orbit.dv", {"a": "x"}, {"a": 1.0})),
     ErrorCode.DUPLICATE_NAME, "/objectives/0/expression/constants/a"),
    ("non-finite constant", objective(ToolCall("orbit.dv", {"a": "x"}, {"b": [1.0, math.inf]})), ErrorCode.NON_FINITE,
     "/objectives/0/expression/constants/b/1"),
    ("object constant", objective(ToolCall("orbit.dv", {"a": "x"}, {"b": {"k": 1}})), ErrorCode.INVALID_VALUE,
     "/objectives/0/expression/constants/b"),
    ("bad output name", objective(ToolCall("orbit.dv", {"a": "x"}, output="dv total")), ErrorCode.INVALID_NAME,
     "/objectives/0/expression/output"),
]


@pytest.mark.parametrize("label, overrides, code, path", INVALID, ids=[case[0] for case in INVALID])
def test_invalid_problem_is_rejected_with_one_specific_error(label, overrides, code, path):
    assert codes(quadratic(**overrides)) == [(code, path)]


def test_enough_distinct_rejections():
    assert len({case[2] for case in INVALID}) >= 15


NUMPY_CASES = [
    ("x**2 + y**2", lambda np, x, y: x ** 2 + y ** 2),
    ("sqrt(x**2 + y**2) - atan2(y, x)", lambda np, x, y: np.sqrt(x ** 2 + y ** 2) - np.arctan2(y, x)),
    ("exp(-x) * sin(y) + log(1 + x**2)", lambda np, x, y: np.exp(-x) * np.sin(y) + np.log(1 + x ** 2)),
    ("max(x, y, 0.5) - min(x, -y)", lambda np, x, y: np.maximum(np.maximum(x, y), 0.5) - np.minimum(x, -y)),
    ("-x / (1 + abs(y)) + cosh(x / 3) - tanh(y)", lambda np, x, y: -x / (1 + np.abs(y)) + np.cosh(x / 3) - np.tanh(y)),
    ("log10(1 + x**2) + asin(y / 10) + acos(y / 10) + atan(x) + tan(x / 4) + cos(y) + sinh(y / 2)",
     lambda np, x, y: (np.log10(1 + x ** 2) + np.arcsin(y / 10) + np.arccos(y / 10) + np.arctan(x)
                       + np.tan(x / 4) + np.cos(y) + np.sinh(y / 2))),
    ("x**-2 + +y", lambda np, x, y: x ** -2.0 + y),
]


@pytest.mark.parametrize("source, reference", NUMPY_CASES, ids=[c[0] for c in NUMPY_CASES])
def test_evaluator_matches_numpy_vectorised_and_scalar(source, reference):
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(14)
    x, y = rng.uniform(0.2, 3.0, 64), rng.uniform(-3.0, 3.0, 64)
    expected = reference(np, x, y)
    expr = Expression(source)
    np.testing.assert_allclose(expr.evaluate({"x": x, "y": y}), expected, rtol=1e-13)
    scalar = [expr.evaluate({"x": float(a), "y": float(b)}) for a, b in zip(x, y)]
    assert all(isinstance(v, float) for v in scalar)
    np.testing.assert_allclose(scalar, expected, rtol=1e-13)


@pytest.mark.parametrize("source, values", [
    ("sqrt(x)", {"x": -1.0}),
    ("1 / x", {"x": 0.0}),
    ("x ** 0.5", {"x": -4.0}),
    ("9 ** 9 ** 9", {}),
    ("exp(x)", {"x": 1e6}),
    ("x + y", {"x": 1.0}),
])
def test_evaluation_failures_raise_one_error_type(source, values):
    with pytest.raises(EvaluationError):
        Expression(source).evaluate(values)


def _rosenbrock(x):
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))


@pytest.mark.parametrize("n", [2, 20, 200])
def test_an_indexed_sum_reads_scalars_x0_to_xn_and_matches_its_unrolled_value(n):
    expr = Expression(f"sum(100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range({n} - 1))")
    assert expr.names == {f"x{i}" for i in range(n)}
    point = [math.cos(3.0 * i) * 2.0 for i in range(n)]
    assert expr.evaluate({f"x{i}": v for i, v in enumerate(point)}) == pytest.approx(_rosenbrock(point), rel=1e-13)
    assert Expression(f"sum(x[i] for i in range({n}))").evaluate({f"x{i}": 1.0 for i in range(n)}) == float(n)


@pytest.mark.parametrize("source, expected", [
    ("sum(i * x[i] for i in range(4))", 0 * 1 + 1 * 2 + 2 * 3 + 3 * 4),
    ("sum(x[i] for i in range(2, 4))", 3 + 4),
    ("sum(x[i] for i in range(3, 1))", 0.0),
    ("sum(sum(x[i] * x[j] for j in range(i)) for i in range(4))", 1 * 2 + 1 * 3 + 2 * 3 + 1 * 4 + 2 * 4 + 3 * 4),
    ("x[3] - x[2 * 1]", 4 - 3),
    ("sum(x[3 - i] ** 2 for i in range(-1 + 1, 4))", 1 + 4 + 9 + 16),
])
def test_indices_are_integer_arithmetic_over_literals_and_enclosing_indices(source, expected):
    assert Expression(source).evaluate({f"x{i}": float(i + 1) for i in range(4)}) == float(expected)


def test_the_expanded_tree_is_plain_public_syntax_and_counts_its_own_node_limit():
    tree = Expression("sum(sqrt(x[i]) * k for i in range(1000))").tree
    allowed = (ast.Constant, ast.Name, ast.UnaryOp, ast.BinOp, ast.Call, ast.Load, ast.Add, ast.Sub, ast.Mult,
               ast.Div, ast.Pow, ast.USub, ast.UAdd)
    nodes = list(ast.walk(tree))
    assert all(isinstance(n, allowed) for n in nodes) and len(nodes) > MAX_NODES
    assert all(isinstance(n.value, float) for n in nodes if isinstance(n, ast.Constant))
    depth = lambda n: 1 + max((depth(c) for c in ast.iter_child_nodes(n)), default=0)  # noqa: E731
    assert depth(tree) < 30  # a long sum expands into a balanced tree, not a chain
    # Both limits count every node ast.walk yields: a term x_i * 2 is 5 (BinOp, Mult, Name, Load, Constant), each
    # joining addition 2 (BinOp, Add), the root 1. So 2857 terms are 19998 nodes and 2858 terms 20005.
    terms = (MAX_EXPANDED_NODES + 1) // 7
    assert Expression(f"sum(x[i] * 2 for i in range({terms}))").names
    with pytest.raises(ExpressionError) as info:
        Expression(f"sum(x[i] * 2 for i in range({terms + 1}))").names
    assert info.value.code is ErrorCode.EXPRESSION_TOO_COMPLEX


def test_units_hold_through_a_sum():
    problem = OptimizationProblem(
        id="sum-units", variables=tuple(Variable(f"d{i}", lower=0.0, unit="km") for i in range(3)),
        objectives=(Objective("total", "sum(d[i] for i in range(3))", unit="m"),),
        constraints=(Constraint("mixed", "sum(d[i] for i in range(3)) + 1", "<=", 5.0, unit="km"),))
    assert codes(problem) == [(ErrorCode.UNIT_MISMATCH, "/constraints/0/expression")]
    good = replace(problem, constraints=())
    assert validate_problem(good) == []
    assert good.evaluate({"d0": 1.0, "d1": 2.0, "d2": 3.0}).objectives["total"] == pytest.approx(6000.0)


def test_unsafe_source_is_rejected_before_anything_runs(tmp_path):
    marker = tmp_path / "ran"
    source = f"__import__('pathlib').Path({str(marker)!r}).touch()"
    with pytest.raises(ExpressionError) as info:
        Expression(source).evaluate({})
    assert info.value.code is ErrorCode.UNSAFE_EXPRESSION
    assert not marker.exists()


def test_tool_calls_evaluate_through_a_registry_in_si():
    problem = OptimizationProblem(
        id="tool",
        variables=(Variable("d", lower=0.0, unit="km"),),
        objectives=(Objective("cost", ToolCall("orbit.cost", {"dist": "2 * d"}), unit="km"),),
    )
    assert validate_problem(problem) == []
    seen = {}

    def cost(dist):
        seen["dist"] = dist
        return dist + 1000.0

    assert problem.evaluate({"d": 3.0}, tools={"orbit.cost": cost}).objectives["cost"] == pytest.approx(7.0)
    assert seen == {"dist": 6000.0}
    with pytest.raises(ToolCallError):
        problem.evaluate({"d": 3.0})


def _tool_problem(call: ToolCall) -> OptimizationProblem:
    return OptimizationProblem(id="tool", variables=(Variable("d", lower=0.0, unit="km"),),
                               objectives=(Objective("cost", call, unit="km"),))


def test_a_tool_call_passes_constants_unchanged_and_reads_the_output_it_names():
    seen = {}

    def survey(dist, model, layers):
        seen.update(dist=dist, model=model, layers=layers)
        return {"cost": dist + 1000.0, "count": 3.0}

    call = ToolCall("orbit.survey", {"dist": "2 * d"}, {"model": "j2", "layers": [0.5, 1.5]}, output="cost")
    problem = _tool_problem(call)
    assert validate_problem(problem) == []
    assert problem.evaluate({"d": 3.0}, tools={"orbit.survey": survey}).objectives["cost"] == pytest.approx(7.0)
    assert seen == {"dist": 6000.0, "model": "j2", "layers": [0.5, 1.5]}
    count = _tool_problem(replace(call, output="count"))
    assert count.evaluate({"d": 3.0}, tools={"orbit.survey": survey}).objectives["cost"] == pytest.approx(3e-3)
    assert count.digest() != problem.digest()
    assert _tool_problem(replace(call, constants={"model": "j4", "layers": [0.5, 1.5]})).digest() != problem.digest()
    back = parse_problem(json.loads(json.dumps(problem.to_dict())))
    assert back == problem and back.objectives[0].expression.constants == {"model": "j2", "layers": [0.5, 1.5]}


@pytest.mark.parametrize("call, tool", [
    (ToolCall("orbit.survey", {"dist": "d"}), lambda dist: {"cost": dist}),           # several outputs, none named
    (ToolCall("orbit.survey", {"dist": "d"}, output="cost"), lambda dist: dist),      # a number, an output named
    (ToolCall("orbit.survey", {"dist": "d"}, output="total"), lambda dist: {"cost": dist}),
    (ToolCall("orbit.survey", {"dist": "d"}), lambda dist: 1.0 / 0.0),                # the tool raises
    (ToolCall("orbit.survey", {"dist": "d"}), lambda dist: "fast"),                   # not a number
])
def test_every_tool_failure_is_a_tool_call_error_that_is_an_evaluation_error(call, tool):
    with pytest.raises(ToolCallError) as info:
        _tool_problem(call).evaluate({"d": 1.0}, tools={"orbit.survey": tool})
    assert isinstance(info.value, EvaluationError) and info.value.tool_id == "orbit.survey"


def test_a_tool_exception_keeps_its_cause():
    class Refused(Exception):
        pass

    def refuse(dist):
        raise Refused("outside the validity domain")

    with pytest.raises(ToolCallError) as info:
        _tool_problem(ToolCall("orbit.survey", {"dist": "d"})).evaluate({"d": 1.0}, tools={"orbit.survey": refuse})
    assert isinstance(info.value.__cause__, Refused) and "outside the validity domain" in str(info.value)


def test_a_1_0_tool_call_parses_with_no_constants_and_no_output():
    doc = {**QUADRATIC_1_0, "objectives": [{"name": "f", "expression": {"tool_id": "orbit.cost", "args": {"a": "x"}}}]}
    assert schema_errors(doc, PROBLEM_SCHEMAS["1.0"]) == []
    call = parse_problem(doc).objectives[0].expression
    assert call == ToolCall("orbit.cost", {"a": "x"}) and call.constants == {} and call.output is None
    with_constants = {**doc, "objectives": [{"name": "f", "expression": {
        "tool_id": "orbit.cost", "args": {"a": "x"}, "constants": {"k": 1}, "output": "v"}}]}
    assert {e.path for e in schema_errors(with_constants, PROBLEM_SCHEMAS["1.0"])} == {"/objectives/0/expression"}
    assert parse_problem({**with_constants, "schemaVersion": "1.1"}).objectives[0].expression.output == "v"


def test_importing_the_ir_pulls_in_no_other_orbitopt_module_or_numpy():
    code = ("import sys, orbitopt.optimization as o; o.Expression('sqrt(x)').evaluate({'x': 4.0}); "
            "print(sorted(m for m in sys.modules if m.startswith('orbitopt.') "
            "and not m.startswith('orbitopt.optimization')), 'numpy' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={"PYTHONPATH": str(ROOT / "src")}).stdout
    assert out.strip() == "[] False"


def test_result_statuses_are_independent_booleans_that_default_false():
    assert {f.name for f in dataclasses.fields(OptimizationResult)} >= set(STATUSES)
    base = OptimizationResult(problem_digest="0" * 64, solver_id="slsqp", termination=Termination.CONVERGED)
    assert [getattr(base, s) for s in STATUSES] == [False] * 4
    for status in STATUSES:
        one = replace(base, **{status: True})
        assert [getattr(one, s) for s in STATUSES] == [s == status for s in STATUSES]


def test_result_round_trips_and_validates_against_its_schema():
    result = OptimizationResult(
        problem_digest=quadratic().digest(), solver_id="scipy.slsqp", termination="max_iterations",
        solver_converged=True, constraints_satisfied=True, x={"x": 0.5, "y": 0.5},
        objective_values={"f": 0.5}, constraint_residuals={"halfplane": 0.0}, evaluations=42, iterations=7,
        wall_time_s=0.01, message="ok",
        evidence=(Evidence("analytical", "KKT point of a convex QP", {"lambda": 1.0}, producer="kkt",
                           problem_digest=quadratic().digest()),),
        front=(ParetoPoint({"x": 0.5, "y": 0.5}, {"f": 0.5}), ParetoPoint({"x": 1.0, "y": 0.0}, {"f": 1.0})),
        provenance=Provenance(cache_key=quadratic().digest()),
    )
    data = json.loads(json.dumps(result.to_dict()))
    assert schema_errors(data, RESULT_SCHEMA) == []
    assert OptimizationResult.from_dict(data) == result
    assert data["termination"] == "max_iterations"
    data["problem_digest"] = "not-a-digest"
    data["termination"] = "gave_up"
    assert {e.path for e in schema_errors(data, RESULT_SCHEMA)} == {"/problem_digest", "/termination"}


def _error_problem(rhs: float, sense: str = "<=") -> OptimizationProblem:
    return OptimizationProblem(id="tiny-rhs", variables=(Variable("err"),), objectives=(Objective("f", "err"),),
                               constraints=(Constraint("accuracy", "err", sense, rhs),))


@pytest.mark.parametrize("rhs, sense, value, holds", [
    (1e-10, "<=", 1e-10 * (1 + 0.5e-6), True),   # within constraint_tol of the bound, relative to it
    (1e-10, "<=", 1e-10 * (1 + 2e-6), False),
    (1e-10, "<=", 1e-9, False),                  # IR 1.0 accepted this: 1e-9 - 1e-10 <= 1e-6 * max(1, 1e-10)
    (-1e-10, ">=", -1e-10 * (1 + 2e-6), False),
    (0.0, "<=", 0.5e-6, True),                   # a zero rhs has no magnitude: the tolerance is absolute
    (0.0, "<=", 2e-6, False),
    (1e3, "<=", 1e3 + 0.5e-3, True),             # large rhs: relative, as in IR 1.0
    (1e3, "==", 1e3 - 2e-3, False),
])
def test_constraint_tolerance_is_relative_to_the_rhs_or_absolute_at_zero(rhs, sense, value, holds):
    problem = _error_problem(rhs, sense)
    assert problem.constraints_satisfied({"err": value}, problem.evaluate({"err": value}).residuals) is holds
    allowance = problem.constraints[0].allowance(problem.verification.constraint_tol)
    assert allowance == 1e-6 * (abs(rhs) or 1.0) == 1e-6 * tolerance_scale(rhs)


@pytest.mark.parametrize("variable, value, reasons", [
    (Variable("x", lower=1e-9, upper=2.0), 0.9e-9, ("below_lower",)),
    (Variable("x", lower=1e-9, upper=2.0), 1e-9 * (1 - 0.5e-6), ()),
    (Variable("x", lower=0.0, upper=2.0), -0.5e-6, ()),
    (Variable("x", lower=0.0, upper=2.0), 2.1, ("above_upper",)),
    (Variable("n", kind="integer", lower=0, upper=3), 1.5, ("not_integer",)),
    (Variable("n", kind="integer", lower=0, upper=3), 4.5, ("above_upper", "not_integer")),
    (Variable("b", kind="binary"), 1.0, ()),
    (Variable("b", kind="binary"), -1.0, ("below_lower",)),
    (Variable("b", kind="binary"), 0.5, ("not_integer",)),
    (Variable("c", kind="discrete", choices=(1e-6, 2.0)), 1e-6 * (1 + 0.5e-6), ()),
    (Variable("c", kind="discrete", choices=(1e-6, 2.0)), 2e-6, ("not_a_choice",)),
])
def test_domain_violations_name_every_part_of_the_domain_a_value_breaks(variable, value, reasons):
    assert variable.domain_violations(value, 1e-6) == reasons
    problem = OptimizationProblem(id="d", variables=(variable,), objectives=(Objective("f", variable.name),))
    assert problem.constraints_satisfied({variable.name: value}, {}) is (reasons == ())


@pytest.mark.parametrize("termination", ["unbounded", "stalled"])
def test_unbounded_and_stalled_are_1_1_terminations(termination):
    data = OptimizationResult(problem_digest="0" * 64, solver_id="s", termination=termination).to_dict()
    assert schema_errors(data, RESULT_SCHEMA) == []
    assert OptimizationResult.from_dict(data).termination is Termination(termination)
    assert {e.path for e in schema_errors({**data, "schemaVersion": "1.0"}, RESULT_SCHEMAS["1.0"])} >= {"/termination"}


def _doc_problem_json():
    block = re.search(r"```json\n(.*?)```", DOC.read_text(encoding="utf-8"), re.S)
    return json.loads(block.group(1))


def test_documented_worked_example_builds_validates_and_evaluates():
    problem = parse_problem(_doc_problem_json())
    optimum = {"x": 0.5, "y": 0.5}
    ev = problem.evaluate(optimum)
    assert ev.objectives == {"distance_sq": 0.5}
    assert ev.residuals == {"halfplane": 0.0}
    assert problem.constraints_satisfied(optimum, ev.residuals)
    off = {"x": 0.4, "y": 0.4}
    assert not problem.constraints_satisfied(off, problem.evaluate(off).residuals)
    assert not problem.constraints_satisfied({"x": -0.1, "y": 1.2}, problem.evaluate({"x": -0.1, "y": 1.2}).residuals)


def test_documented_worked_example_drives_a_solver_to_the_reference_optimum():
    optimize = pytest.importorskip("scipy.optimize")
    problem = parse_problem(_doc_problem_json())
    names = [v.name for v in problem.variables]
    point = lambda z: dict(zip(names, map(float, z)))  # noqa: E731
    sol = optimize.minimize(
        lambda z: problem.evaluate(point(z)).objectives["distance_sq"], x0=[1.0, 0.0], method="SLSQP",
        bounds=[(v.lower, v.upper) for v in problem.variables],
        constraints=[{"type": "ineq", "fun": lambda z, c=c: -problem.evaluate(point(z)).residuals[c.name]}
                     for c in problem.constraints])
    assert sol.fun == pytest.approx(0.5, abs=1e-8)
    assert sol.x == pytest.approx([0.5, 0.5], abs=1e-6)
    assert problem.constraints_satisfied(point(sol.x), problem.evaluate(point(sol.x)).residuals)
