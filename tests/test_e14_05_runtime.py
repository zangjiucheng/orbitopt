"""E14-05 agent iteration runtime: budgets, policy, failure handling, verification, persistence and replay.

Every test drives ``orbitopt.agent.runtime`` through its public API. No test needs an API key or a network:
the LLM planner runs on ``MockProvider`` scripts and ``ReplayProvider`` cassettes. Expected optima come from
the E14-10 catalogue (``benchmarks/e14_spec``), scored with the catalogue's own callables.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from orbitopt.agent.providers import MockProvider, RecordingProvider, ReplayProvider
from orbitopt.agent.runtime import (
    FailureKind,
    LLMPlanner,
    Policy,
    Proposal,
    RecordedPlanner,
    RecordKind,
    Remedy,
    RuleBasedPlanner,
    RunStatus,
    StopReason,
    UnsafePathError,
    ViolationCode,
    CancelPoint,
    CancelReason,
    known_tool_ids,
    load_trace,
    orbitopt_version,
    loosened,
    check_run,
    replay,
    save_run,
    solve,
    trace_digest,
    upgrade_trace,
)
from orbitopt.agent.tools import builtin_registry
from orbitopt.optimization import (
    Budget,
    Constraint,
    Evidence,
    Objective,
    OptimizationProblem,
    OptimizationResult,
    Parameter,
    Provenance,
    Termination,
    ToolCall,
    Variable,
    VerificationPolicy,
)
from orbitopt.optimization.classify import Form
from orbitopt.optimization.solvers import (
    DifferentialEvolutionSolver,
    HighsSolver,
    LbfgsbSolver,
    OptionSpec,
    SlsqpSolver,
    SolverRegistry,
    default_registry,
)
from orbitopt.optimization.verification import (FindingCode, Outcome, ReferenceSolution, canonical_json, file_digest,
                                                result_digest, verify_result, verify_run)

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
P, A, M, F = spec.problems, spec.arms, spec.metrics, spec.faults
TOOL_ID = "bench.quadrature_error"


# ------------------------------------------------------------------------------------- catalogue -> IR (as E14-03)
class _Inline(ast.NodeTransformer):
    def __init__(self, inst):
        self.values = {"n": inst.dimension, **{k: v for k, v in inst.data.items()
                                               if isinstance(v, (int, float)) and not isinstance(v, bool)}}

    def visit_Name(self, node):
        return ast.Constant(self.values[node.id]) if node.id in self.values else node


def to_ir(inst) -> OptimizationProblem:
    def source(expr: str) -> str:
        return ast.unparse(_Inline(inst).visit(ast.parse(expr, mode="eval")).body)

    def variable(i, lo, hi):
        if inst.variable_type == "binary":
            return Variable(f"x{i}", kind="binary")
        return Variable(f"x{i}", kind=inst.variable_type, lower=lo, upper=hi)

    vectors = tuple(Parameter(f"{name}{i}", float(v)) for name, values in sorted(inst.data.items())
                    if isinstance(values, tuple) and all(isinstance(v, (int, float)) for v in values)
                    for i, v in enumerate(values))
    constraints = tuple(
        Constraint(c.name, ToolCall(TOOL_ID, {"rule": "x0", "n": "x1"}), "<=", float(inst.data["tau"]))
        if "quadrature_error" in c.expr else
        Constraint(c.name, source(c.expr), "==" if c.kind == "eq" else "<=", 0.0) for c in inst.constraints)
    return OptimizationProblem(
        id=inst.id, variables=tuple(variable(i, lo, hi) for i, (lo, hi) in enumerate(inst.bounds)),
        parameters=(Parameter("pi", math.pi), Parameter("e", math.e), *vectors),
        objectives=tuple(Objective(f"f{k}", source(e), sense=inst.sense) for k, e in enumerate(inst.objective_exprs)),
        constraints=constraints,
        verification=VerificationPolicy(constraint_tol=max(inst.tolerance.constraint_abs, 1e-15)))


def tools_for(inst):
    if "quadrature_error" not in inst.blackboxes:
        return None
    return {TOOL_ID: lambda rule, n: inst.blackboxes["quadrature_error"](rule, n)}


def correct(inst, result) -> bool:
    """The catalogue's success rule, scored with its own callables."""
    ref, tol = inst.reference, inst.tolerance
    x = [result.x[f"x{i}"] for i in range(inst.dimension)]
    allowed = tol.objective_abs + tol.objective_rel * abs(ref.value)
    return (F.max_violation(inst, x) <= tol.constraint_abs
            and M.optimality_gap(ref.value, inst.objectives[0](x), inst.sense) <= allowed / max(1.0, abs(ref.value)))


def independently_correct(inst, problem, result) -> bool:
    """A fresh guard grade against the catalogue's stored optimum, outside the runtime."""
    evidence = ReferenceSolution(id="catalogue", value=inst.reference.value, source=inst.reference.derivation,
                                 tolerance=max(1e-9, inst.tolerance.objective_abs), problem_digest=problem.digest(),
                                 result_digest=result_digest(result))
    return verify_result(problem, result, evidence=[evidence]).verdict is Outcome.PASSED and correct(inst, result)


# ------------------------------------------------------------------------------------- helpers
class Counter:
    """An independent evaluation counter: a tool every evaluation of the problem has to call once."""

    def __init__(self, delay: float = 0.0):
        self.calls = 0
        self.delay = delay

    def __call__(self, x, y):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return (x - 1.0) ** 2 + (y + 0.5) ** 2


def counted_problem(**budget) -> OptimizationProblem:
    return OptimizationProblem(
        id="counted", variables=(Variable("x", lower=-2.0, upper=2.0), Variable("y", lower=-2.0, upper=2.0)),
        objectives=(Objective("f", ToolCall("test.counted.v1", {"x": "x", "y": "y"})),), budget=Budget(**budget))


class AnyFormLocal(LbfgsbSolver):
    """L-BFGS-B declared able to take a black box, so a tool-evaluated problem has two solvers and an alternate."""

    CAPABILITY = replace(LbfgsbSolver.CAPABILITY, solver_id="test.lbfgsb_any", max_form=Form.BLACK_BOX)


def counted_registry() -> SolverRegistry:
    return SolverRegistry([DifferentialEvolutionSolver(), AnyFormLocal()])


WORKED = OptimizationProblem(
    id="worked", variables=(Variable("x", lower=0.0, upper=10.0), Variable("y", lower=0.0, upper=10.0)),
    objectives=(Objective("distance_sq", "x**2 + y**2"),),
    constraints=(Constraint("halfplane", "x + y", ">=", 1.0),))


def solves(outcome):
    return [r for r in outcome.trace.iterations if r.kind is RecordKind.SOLVE]


def every_result_answers(outcome, problem) -> bool:
    return all(r.problem_digest == problem.digest() for label, r in outcome.results.items())


def scripted(replies):
    return LLMPlanner(MockProvider([json.dumps(r) if isinstance(r, dict) else r for r in replies]))


# ------------------------------------------------------------------------------------- budgets
@pytest.mark.parametrize("limit", [20, 100, 600, 3000])
def test_evaluation_budget_holds_on_an_independent_counter(limit):
    counter = Counter()
    outcome = solve(counted_problem(max_evaluations=limit), planner=RuleBasedPlanner(), registry=counted_registry(),
                    tools={"test.counted.v1": counter})
    assert counter.calls <= limit
    assert outcome.trace.usage.evaluations == counter.calls
    assert sum(r.evaluations for r in outcome.trace.iterations) + \
        sum(v.evaluations for v in outcome.trace.verifications) == counter.calls


def test_a_solver_that_ignores_its_budget_is_stopped_by_the_runtime():
    class Rogue(DifferentialEvolutionSolver):
        CAPABILITY = replace(DifferentialEvolutionSolver.CAPABILITY, solver_id="test.rogue")

        def solve(self, problem, **kwargs):
            for _ in range(10**6):
                problem.evaluate({"x": 0.0, "y": 0.0}, kwargs.get("tools"))
            raise AssertionError("unreachable")

    counter = Counter()
    outcome = solve(counted_problem(max_evaluations=200), planner=RuleBasedPlanner(),
                    registry=SolverRegistry([Rogue()]), tools={"test.counted.v1": counter})
    first = solves(outcome)[0]
    assert first.diagnosis.kind is FailureKind.SOLVER_ERROR and first.result is None
    assert counter.calls == first.evaluations <= first.decision.evaluations < 200
    assert outcome.status is RunStatus.NO_FEASIBLE


@pytest.mark.parametrize("limit", [1, 2, 4])
def test_iteration_limit_counts_every_proposal(limit):
    always = {"action": "solve", "solver_id": "scipy.differential_evolution", "evaluations": 30}
    outcome = solve(counted_problem(max_evaluations=5000), planner=scripted([always] * 10),
                    budget=Budget(max_iterations=limit), registry=counted_registry(),
                    tools={"test.counted.v1": Counter()})
    assert len(outcome.trace.iterations) == limit
    assert outcome.stop_reason is StopReason.MAX_ITERATIONS


@pytest.mark.parametrize("limit", [500, 1200])
def test_token_budget_stops_the_llm_planner(limit):
    always = {"action": "solve", "solver_id": "scipy.differential_evolution", "evaluations": 30}
    outcome = solve(counted_problem(max_evaluations=10_000), planner=scripted([always] * 50),
                    budget=Budget(max_tokens=limit, max_iterations=50), registry=counted_registry(),
                    tools={"test.counted.v1": Counter()})
    tokens = [r.tokens for r in outcome.trace.iterations]
    assert outcome.stop_reason is StopReason.MAX_TOKENS
    assert sum(tokens[:-1]) < limit <= sum(tokens) == outcome.trace.usage.tokens
    assert all(t > 0 for t in tokens)


def test_wall_clock_budget_stops_a_slow_run():
    counter = Counter(delay=0.005)
    started = time.perf_counter()
    outcome = solve(counted_problem(max_evaluations=100_000, max_wall_s=0.4), planner=RuleBasedPlanner(),
                    registry=counted_registry(), tools={"test.counted.v1": counter})
    elapsed = time.perf_counter() - started
    assert (outcome.stop_reason, outcome.status) == (StopReason.MAX_WALL_S, RunStatus.BUDGET_EXHAUSTED)
    assert outcome.trace.verifications == ()  # no time is left to verify in
    assert elapsed < 0.4 + 0.5  # the overrun is the evaluation in flight and the guard's screen
    assert solves(outcome)[0].result.termination is Termination.TIMEOUT


def test_limits_come_from_the_problem_budget_then_the_defaults():
    outcome = solve(counted_problem(max_evaluations=123, max_tokens=77), planner=RuleBasedPlanner(),
                    registry=counted_registry(), tools={"test.counted.v1": Counter()})
    limits = outcome.trace.limits
    assert (limits.max_evaluations, limits.max_tokens, limits.max_iterations, limits.max_wall_s) == (123, 77, 20, 600.0)
    overridden = solve(counted_problem(max_evaluations=123), planner=RuleBasedPlanner(),
                       budget=Budget(max_evaluations=50), registry=counted_registry(),
                       tools={"test.counted.v1": Counter()})
    assert overridden.trace.limits.max_evaluations == 50
    default = solve(WORKED, planner=RuleBasedPlanner())
    assert default.trace.limits.max_evaluations == A.BUDGET.evaluations(2)


# ------------------------------------------------------------------------------------- the policy
MALICIOUS = [
    ({"action": "solve", "solver_id": "scipy.slsqp", "constraints": [{"name": "halfplane", "rhs": 0.0}]},
     ViolationCode.PROBLEM_EDIT),
    ({"action": "solve", "solver_id": "scipy.slsqp", "objective": "x"}, ViolationCode.PROBLEM_EDIT),
    ({"action": "solve", "solver_id": "scipy.slsqp", "bounds": {"x": [-10, 10]}}, ViolationCode.PROBLEM_EDIT),
    ({"action": "solve", "solver_id": "scipy.slsqp", "verification": {"required": []}}, ViolationCode.PROBLEM_EDIT),
    ({"action": "solve", "solver_id": "scipy.slsqp", "options": {"constraint_tol": 1.0}}, ViolationCode.PROBLEM_EDIT),
    ({"action": "solve", "solver_id": "scipy.slsqp", "max_evaluations": 10**9}, ViolationCode.PROBLEM_EDIT),
    ({"action": "solve", "solver_id": "evil.solver"}, ViolationCode.UNKNOWN_SOLVER),
    ({"action": "solve", "solver_id": "scipy.lbfgsb"}, ViolationCode.SOLVER_NOT_ALLOWED),
    ({"action": "solve", "solver_id": "scipy.slsqp", "evaluations": 10**12}, ViolationCode.BUDGET_OVERREACH),
    ({"action": "solve", "solver_id": "scipy.slsqp", "max_iterations": 10**9}, ViolationCode.BUDGET_OVERREACH),
    ({"action": "solve", "solver_id": "scipy.slsqp", "x0": {"x": -5.0, "y": 0.0}}, ViolationCode.X0_OUT_OF_DOMAIN),
    ({"action": "solve", "solver_id": "scipy.slsqp", "x0": {"z": 1.0}}, ViolationCode.X0_UNKNOWN_VARIABLE),
    ({"action": "solve", "solver_id": "scipy.slsqp", "options": {"ftol": 0.5}}, ViolationCode.INVALID_OPTION),
    ({"action": "solve", "solver_id": "scipy.slsqp", "options": {"popsize": 5}}, ViolationCode.INVALID_OPTION),
    ("this is not JSON", ViolationCode.MALFORMED),
]


def test_a_malicious_planner_never_changes_the_problem():
    good = {"action": "solve", "solver_id": "scipy.slsqp", "rationale": "honest"}
    replies = [reply for reply, _ in MALICIOUS] + [good, {"action": "stop"}]
    # the malformed reply is retried once by the provider: script the retry too
    replies.insert(len(MALICIOUS), "still not JSON")
    outcome = solve(WORKED, planner=scripted(replies), policy=Policy(max_violations=len(MALICIOUS) + 1))
    rejected = [r for r in outcome.trace.iterations if r.kind is RecordKind.REJECTED]
    assert [r.violations[0].code for r in rejected] == [code for _, code in MALICIOUS]
    assert all(r.decision is None and r.result is None for r in rejected)
    assert len(solves(outcome)) == 1 and solves(outcome)[0].decision.solver_id == "scipy.slsqp"
    assert every_result_answers(outcome, WORKED)
    assert outcome.trace.fingerprint == WORKED.hard_constraint_fingerprint()
    assert outcome.status is RunStatus.VERIFIED
    assert outcome.result.x["x"] + outcome.result.x["y"] >= 1.0 - 1e-6
    assert outcome.result.objective_values["distance_sq"] == pytest.approx(0.5, abs=1e-8)


def test_policy_can_stop_at_the_first_violation_or_after_a_few():
    relax = {"action": "solve", "solver_id": "scipy.slsqp", "constraints": []}
    stopped = solve(WORKED, planner=scripted([relax] * 5), policy=Policy(on_violation="stop"))
    assert (stopped.status, stopped.stop_reason) == (RunStatus.POLICY_VIOLATION, StopReason.POLICY_VIOLATION)
    assert len(stopped.trace.iterations) == 1 and stopped.trace.usage.evaluations == 0 and stopped.result is None
    capped = solve(WORKED, planner=scripted([relax] * 5), policy=Policy(max_violations=3))
    assert len(capped.trace.iterations) == 3 and capped.status is RunStatus.POLICY_VIOLATION


def test_policy_limits_solvers_options_and_start_points():
    proposals = [{"action": "solve", "solver_id": "scipy.differential_evolution"},
                 {"action": "solve", "solver_id": "scipy.slsqp", "options": {"ftol": 1e-9}},
                 {"action": "solve", "solver_id": "scipy.slsqp", "x0": {"x": 1.0, "y": 1.0}},
                 {"action": "solve", "solver_id": "scipy.slsqp"}, {"action": "stop"}]
    policy = Policy(allowed_solvers=("scipy.slsqp",), allow_options=False, allow_x0=False, max_violations=5)
    outcome = solve(WORKED, planner=scripted(proposals), policy=policy)
    codes = [r.violations[0].code for r in outcome.trace.iterations if r.kind is RecordKind.REJECTED]
    assert codes == [ViolationCode.SOLVER_NOT_ALLOWED, ViolationCode.INVALID_OPTION, ViolationCode.X0_OUT_OF_DOMAIN]
    assert [r.decision.solver_id for r in solves(outcome)] == ["scipy.slsqp"]


# ------------------------------------------------------------------------------------- failure handling
class Crashing(SlsqpSolver):
    CAPABILITY = replace(SlsqpSolver.CAPABILITY, solver_id="backend.crashing")

    def solve(self, problem, **kwargs):
        raise RuntimeError("backend segfault")


def test_a_crashing_solver_is_excluded_and_the_fallback_verifies():
    registry = SolverRegistry([Crashing(), SlsqpSolver(), DifferentialEvolutionSolver()])
    outcome = solve(WORKED, planner=RuleBasedPlanner(), registry=registry)
    first, second = solves(outcome)[:2]
    assert first.decision.solver_id == "backend.crashing" and first.diagnosis.kind is FailureKind.SOLVER_ERROR
    assert second.decision.solver_id == "scipy.slsqp"
    assert "backend.crashing" not in {r.decision.solver_id for r in solves(outcome)[1:]}
    assert outcome.status is RunStatus.VERIFIED and outcome.result.solver_id == "scipy.slsqp"


def test_an_infeasible_problem_stays_infeasible():
    problem = OptimizationProblem(
        id="impossible", variables=(Variable("x", lower=0.0, upper=1.0), Variable("y", lower=0.0, upper=1.0)),
        objectives=(Objective("f", "x + y"),), constraints=(Constraint("too_far", "x + y", ">=", 3.0),))
    outcome = solve(problem, planner=RuleBasedPlanner())
    assert outcome.status is RunStatus.NO_FEASIBLE
    kinds = [r.diagnosis.kind for r in solves(outcome)]
    assert kinds[0] is FailureKind.INFEASIBLE and set(kinds) <= {FailureKind.INFEASIBLE, FailureKind.BUDGET}
    assert [r.decision.solver_id for r in solves(outcome)][:3] == ["scipy.highs", "scipy.slsqp",
                                                                    "scipy.differential_evolution"]
    assert every_result_answers(outcome, problem) and outcome.trace.fingerprint == problem.hard_constraint_fingerprint()
    assert outcome.result is not None and outcome.report.statuses["constraints_satisfied"].outcome is Outcome.FAILED


def test_an_unbounded_problem_stops_without_adding_bounds():
    problem = OptimizationProblem(
        id="unbounded", variables=(Variable("x"), Variable("y", lower=0.0, upper=1.0)),
        objectives=(Objective("f", "x"),), constraints=(Constraint("cap", "x + y", "<=", 5.0),))
    outcome = solve(problem, planner=RuleBasedPlanner())
    first = solves(outcome)[0]
    assert first.result.termination is Termination.UNBOUNDED and first.diagnosis.kind is FailureKind.UNBOUNDED
    assert first.diagnosis.remedies == (Remedy.STOP,)
    assert len(solves(outcome)) == 1 and outcome.status is RunStatus.NO_FEASIBLE


def test_evaluation_errors_lead_to_a_new_start_then_a_new_solver():
    def broken(x, y):
        raise ValueError("model diverged")

    outcome = solve(counted_problem(max_evaluations=2000), planner=RuleBasedPlanner(), registry=counted_registry(),
                    tools={"test.counted.v1": broken})
    first = solves(outcome)[0]
    assert first.result.termination is Termination.ERROR and first.diagnosis.kind is FailureKind.EVALUATION_ERROR
    assert first.diagnosis.remedies[0] is Remedy.CHANGE_INITIAL_GUESS
    assert solves(outcome)[1].decision.x0 is not None
    assert outcome.status is RunStatus.NO_FEASIBLE
    assert outcome.result is None and outcome.report is None and outcome.statuses == {}  # no evaluated point, no claim


class Stalling(SlsqpSolver):
    """Honest about stalling: it reports ``stalled`` until its tolerance is loosened."""

    CAPABILITY = replace(SlsqpSolver.CAPABILITY, solver_id="test.stalling")

    def solve(self, problem, *, options=None, **kwargs):
        result = super().solve(problem, options=options, **kwargs)
        if (options or {}).get("ftol", 1e-10) < 1e-9:
            return replace(result, termination=Termination.STALLED, solver_converged=False, message="line search")
        return result


def test_a_stalled_solver_gets_a_looser_solver_tolerance_never_a_looser_problem():
    registry = SolverRegistry([Stalling(), DifferentialEvolutionSolver()])
    outcome = solve(WORKED, planner=RuleBasedPlanner(), registry=registry)
    first, second = solves(outcome)[:2]
    assert first.diagnosis.kind is FailureKind.STALLED and Remedy.LOOSEN_SOLVER_TOLERANCE in first.diagnosis.remedies
    assert second.decision.solver_id == "test.stalling" and second.decision.options == {"ftol": 1e-8}
    assert second.diagnosis.kind is FailureKind.VERIFIED_LOCAL
    assert outcome.status is RunStatus.VERIFIED and every_result_answers(outcome, WORKED)


class Tunable(SlsqpSolver):
    """An SLSQP whose declared options are its own: a tolerance the runtime has never heard of."""

    CAPABILITY = replace(SlsqpSolver.CAPABILITY, solver_id="test.tunable",
                         options={"ftol": OptionSpec(float, 1e-16, 1e-2, tolerance=True, default=1e-10),
                                  "precision": OptionSpec(float, 1e-12, 1e-3, tolerance=True, default=1e-9)})

    def solve(self, problem, *, options=None, **kwargs):
        return super().solve(problem, options={k: v for k, v in (options or {}).items() if k == "ftol"} or None,
                             **kwargs)


def test_the_runtime_takes_option_schemas_from_the_solver_declaration():
    registry = SolverRegistry([Tunable(), DifferentialEvolutionSolver()])
    capability = registry.get("test.tunable").capability
    proposal = {"action": "solve", "solver_id": "test.tunable", "options": {"precision": 1e-6}}
    outcome = solve(WORKED, planner=scripted([proposal, {**proposal, "options": {"precision": 0.5}},
                                              {"action": "stop"}]), registry=registry)
    accepted, refused = outcome.trace.iterations[:2]
    assert accepted.kind is RecordKind.SOLVE and accepted.decision.options == {"precision": 1e-6}
    assert refused.kind is RecordKind.REJECTED and refused.violations[0].code is ViolationCode.INVALID_OPTION
    assert "must lie in" in refused.violations[0].message
    assert loosened({}, capability) == pytest.approx({"ftol": 1e-8, "precision": 1e-7})
    assert prompt_state_options(registry, "test.tunable")["precision"] == {"type": "float", "min": 1e-12, "max": 1e-3}


def test_budget_max_iterations_caps_the_loop_and_a_decision_caps_the_backend():
    capped = {"action": "solve", "solver_id": "scipy.slsqp", "max_iterations": 1, "x0": {"x": 2.0, "y": 2.0}}
    outcome = solve(WORKED, planner=scripted([capped, capped, capped]), budget=Budget(max_iterations=2))
    assert outcome.stop_reason is StopReason.MAX_ITERATIONS and len(outcome.trace.iterations) == 2
    first = solves(outcome)[0]
    assert first.result.termination is Termination.MAX_ITERATIONS and first.result.iterations <= 1
    assert first.result.evidence[0].data["budget"]["max_backend_iterations"] == 1
    no_cap = solve(to_ir(P.by_id("lp-wyndor-d2")), planner=scripted([{"action": "solve", "solver_id": "scipy.highs", "max_iterations": 3},
                                              {"action": "stop"}]))
    assert no_cap.trace.iterations[0].violations[0].code is ViolationCode.INVALID_OPTION


def prompt_state_options(registry, solver_id):
    seen = []

    def spy(request):
        seen.append(json.loads(request.messages[-1].content))
        return json.dumps({"action": "stop"})

    solve(WORKED, planner=LLMPlanner(MockProvider(spy)), registry=registry)
    return next(s["options"] for s in seen[0]["solvers"] if s["solver_id"] == solver_id)


def test_a_tiny_budget_ends_as_budget_exhausted():
    inst = P.by_id("rosenbrock-d5")
    outcome = solve(to_ir(inst), planner=RuleBasedPlanner(), budget=Budget(max_evaluations=60))
    assert outcome.status is RunStatus.BUDGET_EXHAUSTED and outcome.stop_reason is StopReason.MAX_EVALUATIONS
    assert outcome.trace.usage.evaluations <= 60


def test_heuristic_convergence_is_not_taken_for_an_optimum():
    """With differential evolution alone, nothing independent can confirm its point, however good."""
    inst = P.by_id("knapsack-p07-n15")
    problem = to_ir(inst)
    registry = SolverRegistry([DifferentialEvolutionSolver()])
    outcome = solve(problem, planner=RuleBasedPlanner(), registry=registry, seed=1)
    first = solves(outcome)[0]
    assert first.diagnosis.kind in (FailureKind.HEURISTIC_STOP, FailureKind.BUDGET)
    assert outcome.status in (RunStatus.UNVERIFIED_BEST, RunStatus.BUDGET_EXHAUSTED)
    assert outcome.report.statuses["independently_verified"].outcome is Outcome.NOT_ASSESSED


# ------------------------------------------------------------------------------------- cancellation
def test_a_set_event_cancels_before_anything_runs():
    event = threading.Event()
    event.set()
    outcome = solve(WORKED, planner=RuleBasedPlanner(), cancel=event)
    assert (outcome.status, outcome.stop_reason) == (RunStatus.CANCELLED, StopReason.CANCELLED)
    assert outcome.trace.iterations == () and outcome.trace.usage.evaluations == 0


def test_cancellation_between_iterations_keeps_the_best_so_far_unverified():
    checks = []
    outcome = solve(to_ir(P.by_id("rastrigin-d5")), planner=RuleBasedPlanner(),
                    cancel=lambda: checks.append(1) or len(checks) > 2)
    assert outcome.status is RunStatus.CANCELLED and len(outcome.trace.iterations) == 2
    assert outcome.trace.verifications == () and outcome.result is not None
    assert outcome.report.verdict is not Outcome.PASSED
    assert outcome.trace.cancel == CancelPoint(2, CancelReason.BEFORE_ITERATION)
    assert outcome.trace.to_dict()["outcome"]["cancel"] == {"iteration": 2, "reason": "before_iteration"}


@pytest.mark.parametrize("when, reason", [("first", CancelReason.BEFORE_ITERATION), ("third", CancelReason.BEFORE_ITERATION),
                                          ("verification", CancelReason.BEFORE_VERIFICATION)])
def test_a_cancelled_run_replays_equal(tmp_path, when, reason):
    """The trace records where the cancel fired; replay cancels there, so it reproduces the whole trace."""
    problem = to_ir(P.by_id("rastrigin-d5"))
    records = len(solve(problem, planner=RuleBasedPlanner()).trace.iterations)  # one check before each proposal
    fire = {"first": 1, "third": 3, "verification": records + 1}[when]
    checks = []
    cancel = lambda: checks.append(1) or len(checks) >= fire  # noqa: E731
    outcome = solve(problem, planner=RuleBasedPlanner(), cancel=cancel)
    assert outcome.status is RunStatus.CANCELLED and outcome.trace.cancel.reason is reason
    digest = save_run(outcome, tmp_path)
    report = replay(tmp_path, expected_manifest_digest=digest)
    assert report.equal, (report.integrity, report.differences)
    assert report.replayed_digest == report.recorded_digest == outcome.trace.digest()


def test_a_cancel_point_cancels_a_run_exactly_there():
    problem = to_ir(P.by_id("rastrigin-d5"))
    outcome = solve(problem, planner=RuleBasedPlanner(), cancel=CancelPoint(1, CancelReason.BEFORE_ITERATION))
    assert outcome.status is RunStatus.CANCELLED and len(outcome.trace.iterations) == 1
    assert outcome.trace.cancel == CancelPoint(1, CancelReason.BEFORE_ITERATION)
    never = solve(problem, planner=RuleBasedPlanner(), cancel=CancelPoint(50, CancelReason.BEFORE_ITERATION))
    assert never.status is RunStatus.VERIFIED and never.trace.cancel is None


def _as_trace_1_0(run: Path) -> dict:
    """Rewrite a saved run's trace as agent-trace-1.0 (no cancel point) and reseal the manifest."""
    trace = json.loads((run / "trace.json").read_text())
    trace["schemaVersion"] = "1.0"
    del trace["outcome"]["cancel"], trace["provider"], trace["model"], trace["orbitopt_version"]
    trace["digest"] = trace_digest(trace)
    (run / "trace.json").write_bytes(canonical_json(trace))
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["files"]["trace.json"] = file_digest(canonical_json(trace))
    manifest["trace_digest"] = trace["digest"]
    (run / "manifest.json").write_bytes(canonical_json(manifest))
    return trace


def test_a_1_0_trace_still_parses_checks_and_replays(tmp_path):
    outcome = solve(WORKED, planner=RuleBasedPlanner())
    save_run(outcome, tmp_path)
    old = _as_trace_1_0(tmp_path)
    assert load_trace(tmp_path) == old  # validated against agent-trace-1.0.json, returned as recorded
    upgraded = upgrade_trace(old)
    assert upgraded["schemaVersion"] == "1.1" and upgraded["outcome"]["cancel"] is None
    assert (upgraded["provider"], upgraded["model"], upgraded["orbitopt_version"]) == (None, None, None)
    assert check_run(tmp_path).verified
    report = replay(tmp_path)
    assert report.equal, (report.integrity, report.differences)
    assert report.recorded_digest == old["digest"] != report.replayed_digest  # the 1.1 digest covers the 1.1 form


def test_a_cancelled_1_0_trace_replays_as_before_without_a_cancel(tmp_path):
    """A 1.0 trace does not say where the cancel fired: replay runs on as it always did, and so diverges."""
    checks = []
    outcome = solve(to_ir(P.by_id("rastrigin-d5")), planner=RuleBasedPlanner(),
                    cancel=lambda: checks.append(1) or len(checks) > 2)
    save_run(outcome, tmp_path)
    _as_trace_1_0(tmp_path)
    assert check_run(tmp_path).intact
    report = replay(tmp_path)
    assert report.integrity == () and report.differences


# ------------------------------------------------------------------------------------- tools
def quadrature_problem(tool_id="numerics.quadrature_cost.v1") -> OptimizationProblem:
    return OptimizationProblem(
        id="gauss-points-for-erf", variables=(Variable("n", kind="integer", lower=1, upper=64),),
        objectives=(Objective("points", "n"),),
        constraints=(Constraint("accuracy", ToolCall(tool_id, {"n": "n"},
                                {"integrand": "exp(-t**2)", "a": 0.0, "b": 1.0, "rule": "gauss_legendre"},
                                output="abs_error"), "<=", 1e-12),))


def test_a_tool_outside_the_allowlist_is_refused_up_front():
    view = builtin_registry().view(["math.erf.v1"])
    assert known_tool_ids(view) == {"math.erf.v1"}
    assert "numerics.quadrature_cost.v1" in known_tool_ids(builtin_registry())
    for tools in (view, {"math.erf.v1": lambda x: x}, None):
        outcome = solve(quadrature_problem(), planner=RuleBasedPlanner(), tools=tools)
        assert outcome.status is RunStatus.REFUSED and outcome.trace.usage.evaluations == 0
        assert "numerics.quadrature_cost.v1" in outcome.trace.refusal[0]
    assert len(view.log) == 0


def test_an_invalid_problem_is_refused_not_raised():
    bad = replace(WORKED, objectives=(Objective("distance_sq", "x**2 + __import__('os')"),))
    outcome = solve(bad, planner=RuleBasedPlanner())
    assert outcome.status is RunStatus.REFUSED and outcome.trace.refusal and outcome.trace.iterations == ()


def test_a_tool_based_problem_through_an_allowlisted_view(tmp_path):
    view = builtin_registry().view(["numerics.quadrature_cost.v1"])
    problem = quadrature_problem()
    outcome = solve(problem, planner=RuleBasedPlanner(), tools=view)
    from numpy.polynomial.legendre import leggauss

    exact = 0.5 * math.sqrt(math.pi) * math.erf(1.0)
    first = next(n for n in range(1, 20)
                 if abs(sum(w * math.exp(-((t + 1) / 2) ** 2) for t, w in zip(*leggauss(n))) / 2 - exact) <= 1e-12)
    assert outcome.result.x["n"] == first
    assert outcome.status is RunStatus.VERIFIED  # n takes 64 values: enumeration is the independent alternate
    assert [a.solver_id for a in outcome.trace.verifications[-1].alternates] == ["orbitopt.enumeration"]
    assert outcome.statuses["optimality_certified"] == "passed"  # and the guard enumerated the domain itself
    assert len(view.log) == outcome.trace.usage.evaluations
    save_run(outcome, tmp_path)
    assert (tmp_path / "tool-calls.json").exists()
    assert replay(tmp_path, tools=builtin_registry().view(["numerics.quadrature_cost.v1"])).equal


# ------------------------------------------------------------------------------------- the false-pass check
def _other_digest(problem):
    return replace(problem, id=problem.id + "-other", constraints=()).digest()


def _relaxed(problem):
    return replace(problem, constraints=())


def _replaced_objective(problem):
    first = problem.variables[0].name
    return replace(problem, objectives=(replace(problem.objectives[0], expression=first),))


def corrupt(fault: str, problem: OptimizationProblem, honest: OptimizationResult, inner, kwargs) -> OptimizationResult:
    """One E14-10 fault (benchmarks/e14_spec/faults.py), applied to an IR result by the solver itself."""
    f = dict(honest.objective_values)
    name = problem.objectives[0].name
    if fault == "F01":  # fabricated solver result
        return replace(honest, objective_values={name: f[name] - max(1.0, abs(f[name]))})
    if fault == "F02":  # violates a constraint but reports converged
        outside = {v.name: (v.upper if v.upper is not None else 1.0) + 1.0 for v in problem.variables}
        value = problem.evaluate(outside).objectives[name]
        return replace(honest, x=outside, objective_values={name: float(value)}, constraints_satisfied=True)
    if fault == "F04":  # missing verification evidence: honest, but nothing else can check it
        return honest
    if fault == "F05":  # swapped problem
        return replace(honest, problem_digest=_other_digest(problem))
    if fault == "F06":  # relaxed: solves without the constraints and reports for the original
        relaxed = inner.solve(_relaxed(problem), **kwargs)
        return replace(relaxed, problem_digest=problem.digest(), constraints_satisfied=True)
    if fault == "F07":  # replaced objective: solves another objective and reports its values
        other = inner.solve(_replaced_objective(problem), **kwargs)
        return replace(other, problem_digest=problem.digest(),
                       objective_values={name: other.objective_values[name]})
    if fault == "F08":  # stale cache
        return replace(honest, provenance=Provenance(cache_key=_other_digest(problem)))
    if fault == "F09":  # forged independent verification
        forged = Evidence("alternate_solver", "verified by myself", producer=honest.solver_id,
                          problem_digest=honest.problem_digest)
        return replace(honest, independently_verified=True, evidence=honest.evidence + (forged,))
    raise ValueError(fault)


def faulty_solver(fault: str, inner):
    class Faulty:
        @property
        def capability(self):
            return replace(inner.capability, solver_id=f"faulty.{fault}")

        def solve(self, problem, **kwargs):
            honest = inner.solve(problem, **kwargs)
            return replace(corrupt(fault, problem, honest, inner, kwargs), solver_id=f"faulty.{fault}")

    return Faulty()


HOSTS = {"cqp-worked-d2": SlsqpSolver, "lp-wyndor-d2": HighsSolver}
RUNTIME_FAULTS = ["F01", "F02", "F04", "F05", "F06", "F07", "F08", "F09"]


@pytest.mark.parametrize("host", sorted(HOSTS))
@pytest.mark.parametrize("fault", RUNTIME_FAULTS)
def test_no_faulty_result_is_ever_reported_verified(fault, host):
    inst = P.by_id(host)
    problem = to_ir(inst)
    faulty = faulty_solver(fault, HOSTS[host]())
    honest = [s for s in default_registry().compatible(problem)]
    registry = SolverRegistry([faulty] + ([] if fault == "F04" else honest))
    forced = RecordedPlanner([Proposal({"action": "solve", "solver_id": faulty.capability.solver_id}),
                              Proposal({"action": "stop"})])
    for planner in (forced, RuleBasedPlanner()):
        outcome = solve(problem, planner=planner, registry=registry)
        first = solves(outcome)[0]
        assert first.decision.solver_id == faulty.capability.solver_id
        if outcome.status is RunStatus.VERIFIED:  # only ever an honest solver's answer, right by the catalogue
            assert not outcome.result.solver_id.startswith("faulty."), (fault, planner.name)
            assert independently_correct(inst, problem, outcome.result)
        if fault == "F04":  # honest, but no other solver can check it
            assert outcome.status is RunStatus.UNVERIFIED_BEST
        else:  # the guard caught the lie: the solver is dropped and its point is never an answer
            assert first.diagnosis.kind is FailureKind.REJECTED_BY_GUARD, fault
            assert outcome.result is None or not outcome.result.solver_id.startswith("faulty.")
            assert faulty.capability.solver_id not in {r.decision.solver_id for r in solves(outcome)[1:]}


@pytest.mark.parametrize("target", ["sealed/result.json", "sealed/manifest.json", "trace.json"])
def test_a_tampered_run_directory_is_never_replayed_as_equal(target, tmp_path):
    """F03, tampered artifact, on the runtime's own run directory."""
    outcome = solve(WORKED, planner=RuleBasedPlanner())
    digest = save_run(outcome, tmp_path)
    assert replay(tmp_path, expected_manifest_digest=digest).equal
    path = tmp_path / target
    text = path.read_text()
    path.write_text(text.replace("0.5", "0.4", 1) if "0.5" in text else text[:-2] + "0}")
    report = replay(tmp_path, expected_manifest_digest=digest)
    assert not report.equal and report.integrity
    if target.startswith("sealed/"):
        assert report.sealed_verdict != Outcome.PASSED.value


@pytest.mark.parametrize("name", ["../outside.json", "results/../../outside.json", "ABSOLUTE", "sealed\\..\\x"])
@pytest.mark.parametrize("check", [replay, check_run])
def test_a_manifest_name_that_leaves_the_run_is_refused_before_any_file_is_read(tmp_path, name, check, monkeypatch):
    run = tmp_path / "run"
    save_run(solve(WORKED, planner=RuleBasedPlanner()), run)
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"{}")
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["files"][str(outside) if name == "ABSOLUTE" else name] = file_digest(b"{}")
    (run / "manifest.json").write_bytes(canonical_json(manifest))
    read = []
    real = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self: read.append(self.name) or real(self))
    with pytest.raises(UnsafePathError):
        check(run)
    assert read == ["manifest.json"]


@pytest.mark.parametrize("folder", ["../sealed", "/tmp/sealed"])
def test_a_sealed_folder_outside_the_run_is_refused(tmp_path, folder):
    run = tmp_path / "run"
    save_run(solve(WORKED, planner=RuleBasedPlanner()), run)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["sealed"]["dir"] = folder
    (run / "manifest.json").write_bytes(canonical_json(manifest))
    for check in (replay, check_run):
        with pytest.raises(UnsafePathError):
            check(run)


def test_a_forger_who_recomputes_every_digest_still_fails_replay(tmp_path):
    outcome = solve(WORKED, planner=RuleBasedPlanner())
    save_run(outcome, tmp_path)
    trace = json.loads((tmp_path / "trace.json").read_text())
    first = trace["iterations"][0]
    first["result"]["objective_values"]["distance_sq"] = 0.25  # a better answer that was never computed
    trace["digest"] = trace_digest(trace)
    (tmp_path / "trace.json").write_bytes(canonical_json(trace))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["files"]["trace.json"] = file_digest(canonical_json(trace))
    manifest["trace_digest"] = trace["digest"]
    (tmp_path / "manifest.json").write_bytes(canonical_json(manifest))
    report = replay(tmp_path)
    assert any("iteration-0001 is not the result in results/iteration-0001.json" in i for i in report.integrity)
    assert any("objective_values/distance_sq" in d for d in report.differences)
    assert report.status == "diverged"
    assert not check_run(tmp_path).intact  # caught without re-executing anything


class Untouchable(SlsqpSolver):
    """A solver the integrity check must never run."""

    def solve(self, *args, **kwargs):
        raise AssertionError("check_run ran a solver")


def test_check_run_regrades_the_sealed_run_with_recorded_alternates_and_runs_no_solver(tmp_path, monkeypatch):
    outcome = solve(to_ir(P.by_id("cqp-worked-d2")), planner=RuleBasedPlanner())
    assert outcome.status is RunStatus.VERIFIED and outcome.trace.verifications[-1].alternates
    digest = save_run(outcome, tmp_path)
    calls = []
    monkeypatch.setattr(DifferentialEvolutionSolver, "solve", lambda *a, **k: calls.append(a) or None)
    report = check_run(tmp_path, registry=SolverRegistry([Untouchable()]), expected_manifest_digest=digest)
    assert report.intact and report.verified and report.alternates == "recorded" and calls == []
    assert report.report.statuses["independently_verified"].outcome is Outcome.PASSED
    assert report.recorded_status == "verified" and report.trace["digest"] == outcome.trace.digest()
    assert set(report.files) >= {"manifest.json", "trace.json", "problem.json", "sealed/result.json",
                                 "sealed/manifest.json"}
    assert report.files["trace.json"] == (tmp_path / "trace.json").read_bytes()


def test_check_run_finds_a_recorded_alternate_that_was_rewritten(tmp_path):
    """The recorded alternate goes through the guard: a forged one that disagrees fails the re-grade."""
    outcome = solve(to_ir(P.by_id("cqp-worked-d2")), planner=RuleBasedPlanner())
    save_run(outcome, tmp_path)
    label = next(p.stem for p in (tmp_path / "results").glob("verify-*.json"))
    path = tmp_path / "results" / f"{label}.json"
    data = json.loads(path.read_text())
    data["x"] = {k: v + 0.5 for k, v in data["x"].items()}  # a point that is not where the alternate stopped
    path.write_bytes(canonical_json(data))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["files"][f"results/{label}.json"] = file_digest(canonical_json(data))
    (tmp_path / "manifest.json").write_bytes(canonical_json(manifest))
    report = check_run(tmp_path)
    assert not report.intact
    assert any(label in i for i in report.integrity)




# ------------------------------------------------------------------------------------- persistence and replay
def test_saved_run_layout_and_trace_schema(tmp_path):
    outcome = solve(WORKED, planner=RuleBasedPlanner())
    digest = save_run(outcome, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["format"] == "orbitopt.agent.run/1" and digest.startswith("sha256:")
    assert set(manifest["files"]) >= {"problem.json", "trace.json", "results/iteration-0001.json"}
    assert "manifest.json" not in json.loads((tmp_path / "sealed/manifest.json").read_text())["files"]
    report = verify_run(tmp_path / "sealed", problem=WORKED,
                        expected_manifest_digest=manifest["sealed"]["manifest_digest"])
    assert report.codes.isdisjoint({FindingCode.DIGEST_MISMATCH, FindingCode.MISSING_FILE})
    assert report.statuses["constraints_satisfied"].outcome is Outcome.PASSED
    trace = load_trace(tmp_path)
    assert trace["digest"] == outcome.trace.digest() and trace["outcome"]["status"] == "verified"
    (tmp_path / "trace.json").write_text(json.dumps({k: v for k, v in trace.items() if k != "limits"}))
    with pytest.raises(ValueError):
        load_trace(tmp_path)


def test_replay_of_a_rule_run_is_equal(tmp_path):
    outcome = solve(to_ir(P.by_id("rosenbrock-d5")), planner=RuleBasedPlanner(), seed=3)
    digest = save_run(outcome, tmp_path)
    report = replay(tmp_path, expected_manifest_digest=digest)
    assert report.equal, (report.integrity, report.differences)
    assert report.recorded_digest == report.replayed_digest == outcome.trace.digest()
    assert report.sealed_verdict == "passed"


def test_replay_of_an_llm_run_makes_no_llm_call(tmp_path):
    mock = MockProvider([json.dumps({"action": "solve", "solver_id": "scipy.slsqp"}), json.dumps({"action": "stop"})])
    outcome = solve(WORKED, planner=LLMPlanner(mock))
    save_run(outcome, tmp_path)
    calls = len(mock.requests)
    report = replay(tmp_path)
    assert report.equal and len(mock.requests) == calls == 2


def test_a_recorded_transcript_reproduces_the_run(tmp_path):
    cassette = tmp_path / "planner.jsonl"
    replies = [json.dumps({"action": "solve", "solver_id": "scipy.differential_evolution", "evaluations": 2000,
                           "options": {"popsize": 6}}),
               json.dumps({"action": "solve", "solver_id": "scipy.lbfgsb", "x0": {"x0": 0.1, "x1": 0.1}}),
               json.dumps({"action": "stop"})]
    problem = to_ir(P.by_id("rastrigin-d2"))
    recorded = solve(problem, planner=LLMPlanner(RecordingProvider(MockProvider(replies), cassette)), seed=5)
    again = ReplayProvider(cassette)
    replayed = solve(problem, planner=LLMPlanner(again), seed=5)
    again.assert_exhausted()
    assert replayed.trace.digest() == recorded.trace.digest()


def test_the_trace_names_the_provider_model_and_version_and_only_the_version_is_outside_the_digest(tmp_path):
    mock = MockProvider([json.dumps({"action": "solve", "solver_id": "scipy.slsqp"}), json.dumps({"action": "stop"})],
                        model="qwen-test")
    outcome = solve(WORKED, planner=LLMPlanner(mock))
    body = outcome.trace.to_dict()
    assert (body["planner"], body["provider"], body["model"]) == ("llm", "mock", "qwen-test")
    assert body["orbitopt_version"] == orbitopt_version()
    rule = solve(WORKED, planner=RuleBasedPlanner()).trace.to_dict()
    assert (rule["provider"], rule["model"]) == (None, None)
    assert trace_digest({**body, "orbitopt_version": "9.9.9"}) == body["digest"]  # environmental
    assert trace_digest({**body, "model": "another-model"}) != body["digest"]  # defines the run
    assert trace_digest({**body, "provider": "openai_compat"}) != body["digest"]
    save_run(outcome, tmp_path)
    trace = json.loads((tmp_path / "trace.json").read_text())
    trace["orbitopt_version"] = "0.0.1"  # recorded by another version: the digest and the replay still hold
    (tmp_path / "trace.json").write_bytes(canonical_json(trace))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["files"]["trace.json"] = file_digest(canonical_json(trace))
    (tmp_path / "manifest.json").write_bytes(canonical_json(manifest))
    report = replay(tmp_path)
    assert report.equal, (report.integrity, report.differences)
    assert report.check.trace["orbitopt_version"] == "0.0.1"


def test_two_runs_with_the_same_seed_have_the_same_trace_digest():
    problem = to_ir(P.by_id("rastrigin-d5"))
    a = solve(problem, planner=RuleBasedPlanner(), seed=7)
    b = solve(problem, planner=RuleBasedPlanner(), seed=7)
    assert a.trace.digest() == b.trace.digest() == a.trace.to_dict()["digest"]
    noisy = json.loads(json.dumps(a.trace.body()))
    noisy["usage"]["wall_s"] += 123.0  # another machine's clock
    x = noisy["iterations"][0]["result"]["x"]
    name = next(k for k, v in sorted(x.items()) if v)
    x[name] *= 1 + 1e-15  # another machine's last bits
    assert trace_digest(noisy) == a.trace.digest()
    noisy["iterations"][0]["decision"]["seed"] += 1
    assert trace_digest(noisy) != a.trace.digest()


# ------------------------------------------------------------------------------------- end to end on the catalogue
E2E = ["cqp-worked-d2", "lp-wyndor-d2", "knapsack-p01-n10", "rastrigin-d5", "rosenbrock-d20", "integ-exp-1e-10"]
CERTIFIED = {"knapsack-p01-n10"}
"""Instances whose final report also passes ``optimality_certified``: the guard's enumeration fits the budget."""


def llm_policy(request) -> str:
    """A scripted LLM: reads the runtime's prompt and answers as a sensible model would."""
    state = json.loads(request.messages[-1].content)
    solvers, history = state["solvers"], [h for h in state["history"] if h["kind"] == "solve"]
    local = next((s for s in solvers if s["convergence"] == "first_order"), None)
    global_ = next((s for s in solvers if s["is_global"] and s["convergence"] == "heuristic"), None)
    if not history:
        pick = solvers[0] if state["problem"]["provably_convex"] or state["problem"]["class"] == "milp" else global_
        return json.dumps({"action": "solve", "solver_id": pick["solver_id"],
                           "evaluations": state["remaining"]["evaluations"] // 2,
                           "options": {"popsize": 5} if "popsize" in pick["options"] else {}})
    if history[-1]["diagnosis"] in ("heuristic_stop", "budget") and local and state["incumbent"]:
        return json.dumps({"action": "solve", "solver_id": local["solver_id"], "x0": state["incumbent"]["x"]})
    if history[-1]["diagnosis"] == "heuristic_stop" and len(history) < 4:
        return json.dumps({"action": "solve", "solver_id": global_["solver_id"], "seed": len(history),
                           "options": {"popsize": 5}, "evaluations": state["remaining"]["evaluations"] // 3})
    return json.dumps({"action": "stop", "rationale": "converged"})


@pytest.mark.parametrize("planner_kind", ["rule", "llm"])
@pytest.mark.parametrize("instance", E2E)
def test_catalogue_instances_reach_the_known_optimum(instance, planner_kind):
    inst = P.by_id(instance)
    problem = to_ir(inst)
    planner = RuleBasedPlanner() if planner_kind == "rule" else LLMPlanner(MockProvider(llm_policy))
    outcome = solve(problem, planner=planner, tools=tools_for(inst), seed=0)
    assert correct(inst, outcome.result), (outcome.status, outcome.result.objective_values)
    assert outcome.trace.usage.evaluations <= A.BUDGET.evaluations(inst.dimension)
    assert every_result_answers(outcome, problem)
    assert outcome.status is RunStatus.VERIFIED  # the integration instance through the enumeration alternate
    assert (outcome.statuses["optimality_certified"] == "passed") == (instance in CERTIFIED)
    if planner_kind == "rule":
        assert outcome.trace.usage.tokens == 0
    else:
        assert outcome.trace.usage.tokens == sum(r.tokens for r in outcome.trace.iterations) > 0


def test_the_rule_planner_puts_a_global_solver_first_unless_the_problem_is_provably_convex():
    first = {}
    for instance in ("rosenbrock-d5", "rastrigin-d2", "cqp-worked-d2", "lp-wyndor-d2", "knapsack-p02-n5"):
        outcome = solve(to_ir(P.by_id(instance)), planner=RuleBasedPlanner(), budget=Budget(max_iterations=1))
        first[instance] = solves(outcome)[0].decision.solver_id
    assert first == {"rosenbrock-d5": "scipy.differential_evolution", "rastrigin-d2": "scipy.differential_evolution",
                     "cqp-worked-d2": "scipy.slsqp", "lp-wyndor-d2": "scipy.highs", "knapsack-p02-n5": "scipy.highs"}


def test_the_public_api_is_importable_and_imports_no_backend():
    import subprocess

    code = ("import sys; import orbitopt.agent.runtime as r; "
            "assert {'solve', 'replay', 'save_run', 'RuleBasedPlanner', 'LLMPlanner'} <= set(r.__all__); "
            "assert 'scipy' not in sys.modules and 'numpy' not in sys.modules, sorted(m for m in sys.modules "
            "if m.split('.')[0] in ('scipy', 'numpy'))")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO,
                   env={"PYTHONPATH": str(REPO / "src"), "PATH": ""})


# ------------------------------------------------------------------------------------- enumeration in verification
def test_enumeration_verifies_and_certifies_a_knapsack_that_differential_evolution_gets_wrong():
    """knapsack-p07-n15: differential evolution stops at 1456, the optimum is 1458 (2^15 points)."""
    inst = P.by_id("knapsack-p07-n15")
    problem = to_ir(inst)
    de_first = RecordedPlanner([Proposal({"action": "solve", "solver_id": "scipy.differential_evolution", "seed": 1}),
                                Proposal({"action": "stop"})])
    outcome = solve(problem, planner=de_first, budget=Budget(max_evaluations=100_000))
    first, second = outcome.trace.verifications
    assert solves(outcome)[0].result.objective_values["f0"] == 1456.0
    assert [a.solver_id for a in first.alternates] == ["orbitopt.enumeration"]
    assert first.alternates[0].allocation == first.alternates[0].result.evaluations == 1 << 15
    assert first.report.statuses["independently_verified"].outcome is Outcome.FAILED  # refuted
    assert first.report.statuses["optimality_certified"].outcome is Outcome.FAILED  # and the guard's own count agrees
    assert outcome.status is RunStatus.VERIFIED and outcome.result.objective_values["f0"] == 1458.0
    assert second.candidate == "verify-1-orbitopt.enumeration" and correct(inst, outcome.result)
    assert outcome.trace.usage.evaluations <= 100_000

    rule = solve(problem, planner=RuleBasedPlanner(), budget=Budget(max_evaluations=100_000))
    assert rule.status is RunStatus.VERIFIED and rule.statuses == {
        "solver_converged": "passed", "constraints_satisfied": "passed", "independently_verified": "passed",
        "optimality_certified": "passed"}
    (only,) = rule.trace.verifications
    assert [a.solver_id for a in only.alternates] == ["orbitopt.enumeration"]
    assert only.evaluations == 2 * (1 << 15) + 3  # the alternate's points, the guard's own and three re-evaluations


def test_enumeration_is_left_out_of_verification_when_the_budget_cannot_cover_the_domain():
    problem = to_ir(P.by_id("knapsack-p07-n15"))  # 2^15 points, default limit 2000 * 15
    outcome = solve(problem, planner=RuleBasedPlanner())
    assert outcome.trace.limits.max_evaluations == 30_000
    assert outcome.status is RunStatus.VERIFIED and outcome.statuses["optimality_certified"] == "not_assessed"
    assert "orbitopt.enumeration" not in {a.solver_id for v in outcome.trace.verifications for a in v.alternates}
    assert outcome.trace.usage.evaluations <= 30_000


def test_a_run_verified_by_enumeration_replays_and_checks(tmp_path):
    outcome = solve(to_ir(P.by_id("knapsack-p02-n5")), planner=RuleBasedPlanner())
    assert outcome.statuses["optimality_certified"] == "passed"
    digest = save_run(outcome, tmp_path)
    check = check_run(tmp_path, expected_manifest_digest=digest)
    assert check.verified and check.report.statuses["optimality_certified"].outcome is Outcome.PASSED
    assert replay(tmp_path, expected_manifest_digest=digest).equal
