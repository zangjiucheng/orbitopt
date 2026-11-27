"""E14-03: the solver registry solves the same IR problem with every compatible solver, honestly.

The acceptance corpus is the E14-10 benchmark catalogue (``benchmarks/e14_spec``), translated here into the IR. The
neutral expression grammar is IR 1.1 syntax once ``n`` and scalar data are inlined and each data vector ``s`` is
declared as parameters ``s0, s1, ...``. Every compatible available solver runs every instance once (the
``matrix`` fixture); the tests then check the results against the catalogue's verified references and tolerances,
the budget, the seeds, and the honesty rules of ``orbitopt.optimization.solvers.base``.
"""
from __future__ import annotations

import ast
import dataclasses
import importlib.util
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from orbitopt.optimization import (
    RESULT_SCHEMA,
    Budget,
    Constraint,
    ErrorCode,
    InvalidProblemError,
    Objective,
    OptimizationProblem,
    Parameter,
    Termination,
    ToolCall,
    Variable,
    VerificationPolicy,
    schema_errors,
    validate_problem,
)
from orbitopt.optimization.classify import Form, ProblemClass, classify, linear_coefficients
from orbitopt.optimization.solvers import SolverRegistry, default_registry

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "benchmarks/e14_spec"


def _load():
    if "e14_spec" not in sys.modules:
        spec = importlib.util.spec_from_file_location("e14_spec", PKG / "__init__.py",
                                                      submodule_search_locations=[str(PKG)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules["e14_spec"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["e14_spec"]


spec = _load()
P, A, M = spec.problems, spec.arms, spec.metrics
by_id = P.by_id
REGISTRY = default_registry()
TOOL_ID = "bench.quadrature_error"


# ------------------------------------------------------------------------------------- benchmark catalogue -> IR
class _Inline(ast.NodeTransformer):
    """Replace ``n`` and each scalar datum by its value; ``x[i]``, ``s[i]`` and sums are IR 1.1 syntax already."""

    def __init__(self, inst):
        self.values = {"n": inst.dimension, **{k: v for k, v in inst.data.items()
                                               if isinstance(v, (int, float)) and not isinstance(v, bool)}}

    def visit_Name(self, node):
        return ast.Constant(self.values[node.id]) if node.id in self.values else node


def ir_source(inst, expr: str) -> str:
    return ast.unparse(_Inline(inst).visit(ast.parse(expr, mode="eval")).body)


def data_parameters(inst) -> tuple[Parameter, ...]:
    """Each numeric data vector ``s`` as the parameters ``s0, s1, ...`` that ``s[i]`` reads."""
    return tuple(Parameter(f"{name}{i}", float(v)) for name, values in sorted(inst.data.items())
                 if isinstance(values, tuple) and all(isinstance(v, (int, float)) for v in values)
                 for i, v in enumerate(values))


def to_ir(inst) -> OptimizationProblem:
    def variable(i, lo, hi):
        if inst.variable_type == "binary":
            return Variable(f"x{i}", kind="binary")
        return Variable(f"x{i}", kind=inst.variable_type, lower=lo, upper=hi)

    constraints = []
    for c in inst.constraints:
        if "quadrature_error" in c.expr:
            constraints.append(Constraint(c.name, ToolCall(TOOL_ID, {"rule": "x0", "n": "x1"}), "<=",
                                          float(inst.data["tau"])))
        else:
            constraints.append(Constraint(c.name, ir_source(inst, c.expr), "==" if c.kind == "eq" else "<=", 0.0))
    return OptimizationProblem(
        id=inst.id, variables=tuple(variable(i, lo, hi) for i, (lo, hi) in enumerate(inst.bounds)),
        parameters=(Parameter("pi", math.pi), Parameter("e", math.e), *data_parameters(inst)),
        objectives=tuple(Objective(f"f{k}", ir_source(inst, e), sense=inst.sense)
                         for k, e in enumerate(inst.objective_exprs)),
        constraints=tuple(constraints),
        # the catalogue's hard-constraint tolerance; the IR needs a positive one (integration instances use 0)
        verification=VerificationPolicy(constraint_tol=max(inst.tolerance.constraint_abs, 1e-15)))


def tools_for(inst):
    if "quadrature_error" not in inst.blackboxes:
        return None
    return {TOOL_ID: lambda rule, n: inst.blackboxes["quadrature_error"](rule, n)}


def hypervolume_2d(points, ref) -> float:
    """Area dominated by ``points`` and bounded by ``ref``: a staircase over the non-dominated points by f1."""
    front, best = [], math.inf
    for f1, f2 in sorted(p for p in points if p[0] < ref[0] and p[1] < ref[1]):
        if f2 < best:
            front.append((f1, f2))
            best = f2
    return sum(((front[i + 1][0] if i + 1 < len(front) else ref[0]) - f1) * (ref[1] - f2)
               for i, (f1, f2) in enumerate(front))


def front_of(result) -> list[tuple[float, ...]]:
    return [tuple(p.objective_values.values()) for p in result.front]


def correct(inst, result) -> bool:
    """The catalogue's success rule, scored with its own callables: feasible, and the signed gap within tolerance."""
    ref, tol = inst.reference, inst.tolerance
    allowed = tol.objective_abs + tol.objective_rel * abs(ref.value)
    if ref.indicator == "hypervolume":
        achieved = hypervolume_2d(front_of(result), ref.reference_point)
        return M.optimality_gap(ref.value, achieved, inst.sense, "hypervolume") <= allowed / max(1.0, abs(ref.value))
    if len(result.x) != inst.dimension:
        return False
    x = [result.x[f"x{i}"] for i in range(inst.dimension)]
    achieved = inst.objectives[0](x)
    return (spec.faults.max_violation(inst, x) <= tol.constraint_abs
            and M.optimality_gap(ref.value, achieved, inst.sense) <= allowed / max(1.0, abs(ref.value)))


def budget_for(inst) -> Budget:
    return Budget(max_evaluations=A.BUDGET.evaluations(inst.dimension))


CORPUS = list(P.INSTANCES)


def test_the_whole_corpus_translates_to_valid_ir():
    assert {inst.id: validate_problem(to_ir(inst)) for inst in CORPUS} == {inst.id: [] for inst in CORPUS}


def test_rosenbrock_d20_is_one_indexed_sum_that_matches_the_catalogue():
    """IR 1.0 could not state it: unrolled, its objective has more than 500 nodes. As an indexed sum it has 40."""
    inst = by_id("rosenbrock-d20")
    problem = to_ir(inst)
    source = problem.objectives[0].expression
    assert source.source.startswith("sum(") and len(source.names) == 20
    assert sum(1 for _ in ast.walk(ast.parse(source.source, mode="eval"))) < 100
    unrolled = " + ".join(f"100*(x{i + 1} - x{i}**2)**2 + (1 - x{i})**2" for i in range(19))
    assert validate_problem(dataclasses.replace(problem, objectives=(Objective("f0", unrolled),)))[0].code \
        == ErrorCode.EXPRESSION_TOO_COMPLEX
    rng = np.random.default_rng(20)
    for point in [list(inst.reference.x), *(list(rng.uniform(-5.0, 10.0, 20)) for _ in range(5))]:
        value = problem.evaluate({f"x{i}": float(v) for i, v in enumerate(point)}).objectives["f0"]
        assert value == pytest.approx(inst.objectives[0](point), rel=1e-12, abs=1e-12)
    assert problem.evaluate({f"x{i}": 1.0 for i in range(20)}).objectives["f0"] == inst.reference.value == 0.0
    assert classify(problem).problem_class == ProblemClass.BOX_SMOOTH


@pytest.mark.parametrize("inst", CORPUS, ids=lambda i: i.id)
def test_translated_ir_computes_the_catalogue_callables(inst):
    problem, rng = to_ir(inst), np.random.default_rng(7)
    for _ in range(5):
        point = [float(round(v)) if inst.variable_type != "continuous" else float(v)
                 for v in (lo + (hi - lo) * rng.random() for lo, hi in inst.bounds)]
        ev = problem.evaluate({f"x{i}": v for i, v in enumerate(point)}, tools_for(inst))
        for k, fn in enumerate(inst.objectives):
            assert ev.objectives[f"f{k}"] == pytest.approx(fn(point), rel=1e-12, abs=1e-12)
        for c in inst.constraints:
            assert ev.residuals[c.name] == pytest.approx(c.fn(point), rel=1e-12, abs=1e-12)


@pytest.fixture(scope="session")
def matrix():
    """(instance id, solver id) -> (instance, result, seconds) for every compatible available solver."""
    out = {}
    for inst in CORPUS:
        problem = to_ir(inst)
        for solver in REGISTRY.compatible(problem):
            started = time.perf_counter()
            result = solver.solve(problem, budget=budget_for(inst), seed=1, tools=tools_for(inst))
            out[inst.id, solver.capability.solver_id] = (inst, result, time.perf_counter() - started)
    return out


def outcome(inst, result) -> str:
    ok = correct(inst, result)
    if result.solver_converged:
        return "ok" if ok else "converged-wrong"
    return f"ok({result.termination.value})" if ok else result.termination.value


def matrix_table(matrix) -> str:
    solvers = sorted({s for _, s in matrix})
    lines = ["| instance | class | " + " | ".join(solvers) + " |", "|---" * (len(solvers) + 2) + "|"]
    for inst in CORPUS:
        cells = [outcome(*matrix[inst.id, s][:2]) if (inst.id, s) in matrix else "-" for s in solvers]
        lines.append(f"| {inst.id} | {classify(to_ir(inst)).problem_class.value} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def problem(*variables, objective="x", constraints=(), sense="minimize", **kwargs):
    objectives = objective if isinstance(objective, tuple) else (Objective("f", objective, sense=sense),)
    return OptimizationProblem(id="t", variables=variables, objectives=objectives,
                               constraints=tuple(constraints), **kwargs)


BOX = (Variable("x", lower=-2.0, upper=3.0), Variable("y", lower=-2.0, upper=3.0))


# --------------------------------------------------------------------------------------------------- classification
@pytest.mark.parametrize("source, form", [
    ("x + 2*y - 3", Form.LINEAR), ("x/2", Form.LINEAR), ("x/(1+1)", Form.LINEAR), ("x**1.0", Form.LINEAR),
    ("sqrt(2)*x", Form.LINEAR), ("k*x", Form.LINEAR), ("x**0", Form.LINEAR), ("0*x**3", Form.LINEAR),
    ("x*y - y*x + x", Form.LINEAR), ("(x+1)**2 - x**2", Form.LINEAR), ("(x+1)**3 - x**3 - 3*x**2", Form.LINEAR),
    ("sin(x) - sin(x) + y", Form.LINEAR), ("abs(x) - abs(x)", Form.LINEAR), ("exp(x)*0", Form.LINEAR),
    ("x*y", Form.QUADRATIC), ("x*x", Form.QUADRATIC), ("x**2", Form.QUADRATIC), ("x**k", Form.QUADRATIC),
    ("x*(y + 1)", Form.QUADRATIC), ("(x*y)/2", Form.QUADRATIC), ("-(x - y)**2", Form.QUADRATIC),
    ("x**3", Form.SMOOTH), ("x*y*x", Form.SMOOTH), ("1/x", Form.SMOOTH), ("x/y", Form.SMOOTH), ("x**2/x", Form.SMOOTH),
    ("x**0.5", Form.SMOOTH), ("x**-1", Form.SMOOTH), ("2**x", Form.SMOOTH), ("x**y", Form.SMOOTH),
    ("sin(x)", Form.SMOOTH), ("exp(x) + y", Form.SMOOTH), ("sin(x)**2 + cos(x)**2", Form.SMOOTH),
    ("sqrt(x**2)", Form.SMOOTH), ("(x + y)**100", Form.SMOOTH),
    ("abs(x)", Form.NONSMOOTH), ("min(x, y)", Form.NONSMOOTH), ("max(x, 1) + y", Form.NONSMOOTH),
    ("sin(abs(x))", Form.NONSMOOTH), ("x*abs(2)", Form.LINEAR),
])
def test_classify_decides_the_form_exactly_from_the_syntax_tree(source, form):
    p = problem(*BOX, objective=source, parameters=(Parameter("k", 2.0),))
    assert classify(p).objective_forms == (form,)


@pytest.mark.parametrize("objective, constraints, sense, kinds, expected", [
    ("2*x + 3*y", [("x + y", ">=", 1.0)], "minimize", "cc", ProblemClass.LP),
    ("x", [], "minimize", "cc", ProblemClass.LP),
    ("x**2 + x*y + y**2", [("x + y", ">=", 1.0)], "minimize", "cc", ProblemClass.CONVEX_QP),
    ("x**2 + 3*x*y + y**2", [("x + y", ">=", 1.0)], "minimize", "cc", ProblemClass.NLP),  # indefinite Hessian
    ("-(x**2) - y**2", [("x + y", "<=", 1.0)], "maximize", "cc", ProblemClass.CONVEX_QP),
    ("x**2", [("x + y", "<=", 1.0)], "maximize", "cc", ProblemClass.NLP),
    ("x**2 + y**2", [("x**2 + y", "<=", 1.0)], "minimize", "cc", ProblemClass.NLP),  # quadratic constraint
    ("x**2 - y**2", [], "minimize", "cc", ProblemClass.BOX_SMOOTH),
    ("sin(x) + y**4", [], "minimize", "cc", ProblemClass.BOX_SMOOTH),
    ("abs(x) + y", [], "minimize", "cc", ProblemClass.NONSMOOTH),
    ("x + y", [("x - y", "==", 0.0)], "maximize", "ci", ProblemClass.MILP),
    ("x*y", [], "minimize", "ci", ProblemClass.MINLP),
    ("x", [("abs(y)", "<=", 1.0)], "minimize", "cb", ProblemClass.MINLP),
])
def test_classify_reports_the_headline_class(objective, constraints, sense, kinds, expected):
    y = {"c": Variable("y", lower=-2.0, upper=3.0), "i": Variable("y", kind="integer", lower=-2, upper=3),
         "b": Variable("y", kind="binary")}[kinds[1]]
    variables = [Variable("x", lower=-2.0, upper=3.0), y]
    p = problem(*variables, objective=objective, sense=sense,
                constraints=[Constraint(f"c{i}", e, s, r) for i, (e, s, r) in enumerate(constraints)])
    assert classify(p).problem_class == expected


def test_classify_black_box_and_multi_objective_and_bounds():
    tool = problem(*BOX, objective="x", constraints=[Constraint("t", ToolCall("bench.tool", {"a": "x"}), "<=", 1.0)])
    assert classify(tool).problem_class == ProblemClass.BLACK_BOX and classify(tool).form == Form.BLACK_BOX
    mo = problem(*BOX, objective=(Objective("a", "x"), Objective("b", "x + y")))
    assert classify(mo).problem_class == ProblemClass.MULTI_OBJECTIVE and classify(mo).multiobjective
    assert classify(problem(*BOX)).bounded
    assert not classify(problem(Variable("x", lower=0.0), objective="x")).bounded
    assert classify(problem(Variable("x", kind="discrete", choices=(1.0, 2.0)))).bounded


def test_linear_coefficients_are_exact_in_the_variables_own_units():
    p = problem(Variable("a", lower=0.0, unit="km"), Variable("b", lower=0.0, unit="m"),
                objective=(Objective("f", "a + 2*b + 3*a - a + 5", unit="m"),))
    assert linear_coefficients(p, p.objectives[0]) == ({"a": 3000.0, "b": 2.0}, 5.0)
    with pytest.raises(ValueError):
        linear_coefficients(p, Objective("g", "a*b", unit="m**2"))


CATALOGUE_CLASS = {"lp": {ProblemClass.LP}, "milp": {ProblemClass.MILP}, "convex_qp": {ProblemClass.CONVEX_QP},
                   "black_box": {ProblemClass.BLACK_BOX}, "multi_objective": {ProblemClass.MULTI_OBJECTIVE},
                   "nlp": {ProblemClass.BOX_SMOOTH, ProblemClass.NLP},
                   "global": {ProblemClass.BOX_SMOOTH}}  # multimodality is not decidable: global reads as smooth


@pytest.mark.parametrize("inst", CORPUS, ids=lambda i: i.id)
def test_classify_agrees_with_the_catalogue_where_the_class_is_decidable(inst):
    assert classify(to_ir(inst)).problem_class in CATALOGUE_CLASS[inst.problem_class]


# ------------------------------------------------------------------------------------------- the benchmark matrix
def test_every_result_is_schema_valid_and_reports_the_irs_own_evaluation(matrix):
    for (iid, sid), (inst, r, _) in matrix.items():
        p = to_ir(inst)
        assert schema_errors(r.to_dict(), RESULT_SCHEMA) == [], (iid, sid)
        assert r.problem_digest == p.digest() and r.solver_id == sid
        assert not r.independently_verified and not r.optimality_certified
        assert 0 < r.evaluations <= budget_for(inst).max_evaluations
        assert r.solver_converged == (r.termination == Termination.CONVERGED)
        if not r.x:
            continue
        ev = p.evaluate(r.x, tools_for(inst))
        assert r.objective_values == pytest.approx(ev.objectives)
        assert r.constraint_residuals == pytest.approx(ev.residuals)
        assert r.constraints_satisfied == p.constraints_satisfied(r.x, ev.residuals)
        assert r.constraints_satisfied or not r.solver_converged


def promises_optimum(capability, problem_class) -> bool:
    """Whether the solver's convergence flag means the global optimum on this class of problem."""
    return capability.convergence == "certificate" or (
        capability.convergence == "first_order" and problem_class in (ProblemClass.LP, ProblemClass.CONVEX_QP))


def test_no_solver_claims_convergence_with_a_wrong_answer_where_its_convergence_means_optimal(matrix):
    promised = 0
    for (iid, sid), (inst, r, _) in matrix.items():
        if r.solver_converged and promises_optimum(REGISTRY.get(sid).capability, classify(to_ir(inst)).problem_class):
            promised += 1
            assert correct(inst, r), (iid, sid, r.objective_values, r.message)
    assert promised >= 15  # LP, MILP and convex QP cells, so the check is not vacuous


@pytest.mark.parametrize("inst", [i for i in CORPUS if i.problem_class in ("lp", "milp", "convex_qp")],
                         ids=lambda i: i.id)
def test_every_linear_or_convex_instance_is_solved_by_a_solver_whose_convergence_means_optimal(matrix, inst):
    cls = classify(to_ir(inst)).problem_class
    assert any(r.solver_converged and correct(inst, r) and promises_optimum(REGISTRY.get(sid).capability, cls)
               for (iid, sid), (_, r, _) in matrix.items() if iid == inst.id), matrix_table(matrix)


def test_a_backend_that_claims_success_at_an_infeasible_point_is_reported_infeasible():
    from orbitopt.optimization.solvers import AdapterSolver, Outcome

    class Liar(AdapterSolver):
        CAPABILITY = dataclasses.replace(REGISTRY.get("scipy.slsqp").CAPABILITY, solver_id="test.liar")

        def _run(self, run):
            run.evaluator([0.0, 0.0])
            return Outcome(converged=True, message="trust me")

    p = problem(*BOX, objective="x + y", constraints=[Constraint("c", "x + y", ">=", 1.0)])
    r = Liar().solve(p)
    assert r.termination == Termination.INFEASIBLE and not r.solver_converged and not r.constraints_satisfied


def test_one_problem_solved_by_several_solvers_through_the_identical_call():
    for iid in ("lp-wyndor-d2", "cqp-worked-d2", "rosenbrock-d2"):
        inst = by_id(iid)
        p = to_ir(inst)
        ids = [s.capability.solver_id for s in REGISTRY.compatible(p)]
        solved = [sid for sid in ids if correct(inst, REGISTRY.solve_with(p, sid, budget=budget_for(inst), seed=1))]
        assert len(solved) >= 2, (iid, ids, solved)


def test_maximisation_is_negated_by_the_adapter_and_reported_in_its_declared_sense():
    p = to_ir(by_id("lp-wyndor-d2"))
    for sid in ("scipy.highs", "scipy.slsqp"):
        r = REGISTRY.solve_with(p, sid)
        assert r.objective_values["f0"] == pytest.approx(36.0, abs=1e-7) and r.x == pytest.approx({"x0": 2, "x1": 6})


# ----------------------------------------------------------------------------------------------------- budgets
BUDGET_CASES = [("scipy.lbfgsb", "rosenbrock-d5"), ("scipy.slsqp", "rosenbrock-disk-d2"),
                ("scipy.differential_evolution", "rastrigin-d5"), ("scipy.highs", "knapsack-p07-n15"),
                ("pygmo.nsga2", "mo-zdt1-d5"), ("cma.cma_es", "rosenbrock-d5")]


@dataclasses.dataclass
class Calls:
    count: int = 0
    delay_s: float = 0.0


@pytest.fixture
def counted(monkeypatch):
    """Counts every real call of OptimizationProblem.evaluate, independently of the adapter's counter."""
    calls, original = Calls(), OptimizationProblem.evaluate

    def evaluate(self, x, tools=None):
        calls.count += 1
        time.sleep(calls.delay_s)
        return original(self, x, tools)

    monkeypatch.setattr(OptimizationProblem, "evaluate", evaluate)
    return calls


def _solver_or_skip(sid):
    solver = REGISTRY.get(sid)
    if not solver.capability.available:
        pytest.skip(solver.capability.unavailable_reason)
    return solver


@pytest.mark.parametrize("sid, iid", BUDGET_CASES)
@pytest.mark.parametrize("limit", [1, 3, 9, 60])
def test_a_tiny_evaluation_budget_is_never_exceeded_by_one_evaluation(counted, sid, iid, limit):
    solver, inst = _solver_or_skip(sid), by_id(iid)
    r = solver.solve(to_ir(inst), budget=Budget(max_evaluations=limit), seed=1)
    assert counted.count == r.evaluations <= limit
    assert schema_errors(r.to_dict(), RESULT_SCHEMA) == []
    if sid == "scipy.highs":  # HiGHS reads coefficients and spends one evaluation, on its answer
        assert r.evaluations == 1 and r.termination == Termination.CONVERGED
    elif limit < 10:
        assert r.termination == Termination.MAX_EVALUATIONS and not r.solver_converged
    else:
        assert r.termination == Termination.MAX_EVALUATIONS or r.evaluations < limit


@pytest.mark.parametrize("sid, iid", BUDGET_CASES)
def test_the_wall_clock_budget_stops_every_solver(counted, sid, iid):
    solver, inst = _solver_or_skip(sid), by_id(iid)
    counted.delay_s = 0.002
    r = solver.solve(to_ir(inst), budget=Budget(max_wall_s=0.05), seed=1)
    assert r.wall_time_s < 0.05 + 0.5
    if sid != "scipy.highs":
        assert r.termination == Termination.TIMEOUT and not r.solver_converged


# --------------------------------------------------------------------------------------------- seeds, availability
@pytest.mark.parametrize("sid, iid", BUDGET_CASES)
def test_the_same_seed_and_budget_give_the_identical_result(sid, iid):
    solver, inst = _solver_or_skip(sid), by_id(iid)
    budget = Budget(max_evaluations=1500)

    def run(seed):
        d = solver.solve(to_ir(inst), budget=budget, seed=seed).to_dict()
        d.pop("wall_time_s")
        return d

    assert run(5) == run(5)
    if solver.capability.solver_id in ("scipy.differential_evolution", "pygmo.nsga2", "cma.cma_es"):
        assert run(5)["x"] != run(6)["x"]


def test_an_unavailable_backend_is_reported_not_raised(monkeypatch):
    monkeypatch.setitem(sys.modules, "pygmo", None)
    nsga2 = REGISTRY.get("pygmo.nsga2")
    assert not nsga2.capability.available and "pygmo" in nsga2.capability.unavailable_reason
    p = to_ir(by_id("mo-schaffer-d1"))
    assert REGISTRY.compatible(p) == ()
    r = REGISTRY.solve_with(p, "pygmo.nsga2")
    assert r.termination == Termination.ERROR and r.evaluations == 0 and "pygmo" in r.message
    assert schema_errors(r.to_dict(), RESULT_SCHEMA) == []


def test_cma_es_is_registered_with_a_reason_when_cma_is_missing():
    cap = REGISTRY.get("cma.cma_es").capability
    installed = importlib.util.find_spec("cma") is not None
    assert cap.available == installed
    assert installed or "'cma'" in cap.unavailable_reason


# ---------------------------------------------------------------------------------------------- exactness, failure
@pytest.mark.parametrize("inst", [i for i in CORPUS if i.problem_class == "milp"], ids=lambda i: i.id)
def test_highs_solves_every_knapsack_exactly(inst):
    r = REGISTRY.solve_with(to_ir(inst), "scipy.highs")
    assert r.termination == Termination.CONVERGED and r.constraints_satisfied and r.evaluations == 1
    assert r.objective_values["f0"] == inst.reference.value
    if inst.reference.x is not None:
        assert [r.x[f"x{i}"] for i in range(inst.dimension)] == list(inst.reference.x)


def test_discrete_variables_are_searched_over_their_choices():
    p = problem(Variable("x", kind="discrete", choices=(1.0, 2.5, 4.0)),
                Variable("y", kind="integer", lower=0, upper=3), objective="x + y", sense="maximize",
                constraints=[Constraint("cap", "x + y", "<=", 5.5)])
    for solver in REGISTRY.compatible(p):
        r = solver.solve(p, seed=2)
        assert r.x == {"x": 2.5, "y": 3.0} and r.constraints_satisfied, solver.capability.solver_id


INFEASIBLE = {
    "lp": problem(*BOX, objective="x + y", constraints=[Constraint("far", "x + y", ">=", 7.0)]),
    "nlp": problem(*BOX, objective="x**2 + y", constraints=[Constraint("ring", "x**2 + y**2", "<=", -1.0)]),
    "milp": problem(Variable("x", kind="binary"), Variable("y", kind="binary"), objective="x - y",
                    constraints=[Constraint("three", "x + y", ">=", 3.0)]),
}


@pytest.mark.parametrize("kind", sorted(INFEASIBLE))
def test_an_infeasible_problem_is_reported_infeasible_by_every_solver_that_can_take_it(kind):
    solvers = REGISTRY.compatible(INFEASIBLE[kind])
    assert len(solvers) >= 2
    for solver in solvers:
        r = solver.solve(INFEASIBLE[kind], seed=1)
        assert r.termination == Termination.INFEASIBLE, (solver.capability.solver_id, r.message)
        assert not r.solver_converged and not r.constraints_satisfied


def test_highs_reports_an_unbounded_lp_as_unbounded():
    p = problem(Variable("x", lower=0.0), Variable("y", lower=0.0), objective="-x",
                constraints=[Constraint("c", "x - y", "<=", 1.0)])
    r = REGISTRY.solve_with(p, "scipy.highs", seed=1)
    assert r.termination == Termination.UNBOUNDED and not r.solver_converged and not r.constraints_satisfied
    assert schema_errors(r.to_dict(), RESULT_SCHEMA) == []


def test_solver_evidence_names_its_producer_and_problem_and_a_single_objective_result_has_no_front():
    p = to_ir(by_id("lp-wyndor-d2"))
    r = REGISTRY.solve_with(p, "scipy.highs", seed=1)
    assert r.evidence[0].producer == "scipy.highs" and r.evidence[0].problem_digest == p.digest() == r.problem_digest
    assert r.front == () and "front" not in r.evidence[0].data


def test_a_backend_exception_becomes_an_error_result():
    def broken(a):
        raise RuntimeError("tool fell over")

    p = problem(*BOX, objective="x", constraints=[Constraint("t", ToolCall("bench.broken", {"a": "x"}), "<=", 1.0)])
    r = REGISTRY.solve_with(p, "scipy.differential_evolution", tools={"bench.broken": broken})
    assert r.termination == Termination.ERROR and "tool fell over" in r.message and not r.solver_converged


@pytest.mark.parametrize("sid", ["scipy.slsqp", "scipy.differential_evolution"])
def test_no_evaluated_point_means_no_feasibility_claim_even_without_constraints(sid):
    p = problem(*BOX, objective="log(x - 10)")  # undefined on the whole box, and nothing else to satisfy
    r = REGISTRY.solve_with(p, sid, budget=Budget(max_evaluations=50))
    assert r.evaluations > 0 and r.termination == Termination.ERROR
    assert not r.constraints_satisfied and r.x == {} and r.objective_values == {}


def test_a_multi_objective_result_carries_a_feasible_non_dominated_front():
    inst = by_id("mo-two-centres-d2")
    r = REGISTRY.solve_with(to_ir(inst), "pygmo.nsga2", budget=budget_for(inst), seed=1)
    front = front_of(r)
    assert len(front) >= 10 and tuple(r.objective_values.values()) == min(front)
    assert all(set(p.x) == {"x0", "x1"} for p in r.front) and r.x in [p.x for p in r.front]
    assert not any(all(a <= b for a, b in zip(q, p)) and q != p for p in front for q in front)


# --------------------------------------------------------------------------------------------- registry contract
def test_compatible_is_ranked_most_specialised_first():
    assert [s.capability.solver_id for s in REGISTRY.compatible(to_ir(by_id("lp-wyndor-d2")))] == [
        "scipy.highs", "scipy.slsqp", "scipy.differential_evolution"]
    box = [s.capability.solver_id for s in REGISTRY.compatible(to_ir(by_id("rosenbrock-d5")))]
    assert box == ["scipy.lbfgsb", "scipy.slsqp"] + (["cma.cma_es"] if REGISTRY.get("cma.cma_es").capability.available
                                                     else []) + ["scipy.differential_evolution"]
    assert SolverRegistry(reversed([REGISTRY.get(c.solver_id) for c in REGISTRY.capabilities()])).compatible(
        to_ir(by_id("rosenbrock-d5"))) == REGISTRY.compatible(to_ir(by_id("rosenbrock-d5")))


def test_malformed_requests_raise_and_impossible_ones_return_an_error_result():
    with pytest.raises(InvalidProblemError):
        REGISTRY.compatible(problem(Variable("x"), objective="x + nope"))
    with pytest.raises(KeyError):
        REGISTRY.get("scipy.nope")
    with pytest.raises(ValueError):
        REGISTRY.register(REGISTRY.get("scipy.slsqp"))
    p = to_ir(by_id("knapsack-p02-n5"))
    with pytest.raises(ValueError):
        REGISTRY.solve_with(p, "scipy.highs", options={"popsize": 3})
    r = REGISTRY.solve_with(p, "scipy.lbfgsb")
    assert r.termination == Termination.ERROR and r.evaluations == 0 and "integer" in r.message


def test_the_registry_and_classifier_import_no_numpy_until_a_solver_runs():
    code = ("import sys; from orbitopt.optimization import Objective, OptimizationProblem, Variable; "
            "from orbitopt.optimization.solvers import default_registry; "
            "p = OptimizationProblem(id='t', variables=(Variable('x', lower=0, upper=1),), "
            "objectives=(Objective('f', 'x**2'),)); "
            "print([s.capability.solver_id for s in default_registry().compatible(p)], 'numpy' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONPATH": str(REPO / "src")}).stdout
    assert out.strip().endswith("False")
