"""E14-06 independent verification guard: honest passes, the E14-10 fault matrix, extra negatives, certificates.

The fault matrix applies every injector in ``benchmarks/e14_spec/faults.py`` to an honest artifact of every
catalogue instance the IR can express, maps the artifact onto the guard's inputs field by field, and grades it
twice: in memory with ``verify_result`` and from a sealed run directory with ``verify_run``. The mapping never
decides a verdict. For the one fault that is not re-sealed (F03) it applies the same injector to the guard's own
recorded digest, so the guard sees the same attack in its own format.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import math
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from orbitopt.optimization import (
    Constraint,
    Evidence,
    Expression,
    Objective,
    OptimizationProblem,
    OptimizationResult,
    Parameter,
    ToolCall,
    Variable,
    VerificationPolicy,
    validate_problem,
)
from orbitopt.optimization.verification import (
    AlternateSolverCheck,
    AnalyticalOptimum,
    FindingCode,
    LowerBoundCertificate,
    Outcome,
    Provenance,
    ReferenceSolution,
    kkt_certificate,
    result_digest,
    seal_run,
    verify_result,
    verify_run,
)

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "benchmarks/e14_spec"
ALL = ("solver_converged", "constraints_satisfied", "independently_verified", "optimality_certified")
C = FindingCode


def _load():
    if "e14_spec" not in sys.modules:
        spec = importlib.util.spec_from_file_location("e14_spec", PKG / "__init__.py",
                                                      submodule_search_locations=[str(PKG)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules["e14_spec"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["e14_spec"]


spec = _load()
P, F = spec.problems, spec.faults


# ------------------------------------------------------------------------------ neutral form -> IR
class _Inline(ast.NodeTransformer):
    """Replace ``n`` and each scalar datum by its value; ``x[i]``, ``s[i]`` and sums are IR 1.1 syntax already."""

    def __init__(self, values: dict):
        self.values = values

    def visit_Name(self, node):
        return ast.Constant(self.values[node.id]) if node.id in self.values else node


def to_ir(form: dict) -> OptimizationProblem:
    """The IR problem a neutral form states; variables are x0..x{n-1}, constraints are ``g <= 0`` / ``h == 0``.

    The neutral grammar is IR 1.1 syntax once ``n`` and scalar data are inlined; a data vector ``s`` becomes the
    parameters ``s0, s1, ...`` that ``s[i]`` reads, and ``pi`` and ``e`` are parameters too.
    """
    if form["blackboxes"]:
        raise ValueError("black boxes have no IR expression")
    data = form["data"]
    inline = _Inline({"n": form["dimension"], **{k: v for k, v in data.items()
                                                 if isinstance(v, (int, float)) and not isinstance(v, bool)}})

    def source(expr: str) -> str:
        return ast.unparse(inline.visit(ast.parse(expr, mode="eval")).body)

    vectors = tuple(Parameter(f"{name}{i}", float(v)) for name, values in sorted(data.items())
                    if isinstance(values, list) and all(isinstance(v, (int, float)) for v in values)
                    for i, v in enumerate(values))
    variables = []
    for i, (lo, hi) in enumerate(form["bounds"]):
        if form["variable_type"] == "binary" and (lo, hi) == (0.0, 1.0):
            variables.append(Variable(f"x{i}", "binary"))
        else:
            kind = "continuous" if form["variable_type"] == "continuous" else "integer"
            variables.append(Variable(f"x{i}", kind, lower=lo, upper=hi))
    return OptimizationProblem(
        id=form["id"], variables=tuple(variables),
        parameters=(Parameter("pi", math.pi), Parameter("e", math.e), *vectors),
        objectives=tuple(Objective(f"f{k}", source(e), sense=form["sense"]) for k, e in enumerate(form["objective_exprs"])),
        constraints=tuple(Constraint(c["name"], source(c["expr"]), "<=" if c["kind"] == "le" else "==", 0.0)
                          for c in form["constraints"]))


def _translate_catalogue() -> dict[str, OptimizationProblem]:
    out = {}
    for inst in P.INSTANCES:
        try:
            problem = to_ir(P.neutral_form(inst))
        except ValueError:
            continue
        if not validate_problem(problem):
            out[inst.id] = problem
    return out


IR = _translate_catalogue()
BENCH_TO_IR = {F.problem_digest(P.by_id(i)): p.digest() for i, p in IR.items()}
HOSTS = [i.id for i in P.INSTANCES if i.id in IR and len(i.objectives) == 1 and i.reference.x is not None]


def resolve(bench_digest: str) -> str:
    """The IR digest of the problem a benchmark digest names; an unknown one maps to no problem's digest."""
    return BENCH_TO_IR.get(bench_digest) or __import__("hashlib").sha256(bench_digest.encode()).hexdigest()


def test_translation_agrees_with_the_catalogue_callables():
    assert len(HOSTS) >= 15, HOSTS
    for inst_id in HOSTS:
        inst, problem = P.by_id(inst_id), IR[inst_id]
        point = [float(v) for v in inst.reference.x]
        ev = problem.evaluate({f"x{i}": v for i, v in enumerate(point)})
        assert ev.objectives["f0"] == pytest.approx(inst.objectives[0](point), abs=1e-9)
        for c in inst.constraints:
            assert ev.residuals[c.name] == pytest.approx(c.fn(point), abs=1e-9)


# ------------------------------------------------------------------------------ benchmark artifact -> guard inputs
def guard_inputs(artifact: dict, inst):
    """Map an E14-10 artifact onto (graded problem, result, evidence, provenance), field by field."""
    reference = IR[inst.id]
    ev = artifact["evidence"]
    names = [v.name for v in reference.variables]
    converged = artifact["statuses"]["solver"] == "converged"
    verified = artifact["statuses"]["verification"] == "verified"
    result = OptimizationResult(
        problem_digest=resolve(artifact["problem_digest"]), solver_id=ev["solver"],
        termination="converged" if converged else "error", solver_converged=converged,
        constraints_satisfied=verified, independently_verified=verified,
        x=dict(zip(names, artifact["x"])), objective_values={"f0": artifact["objective"]},
        constraint_residuals={c.name: r for c, r in zip(reference.constraints, artifact["constraint_residuals"])})
    evidence = []
    check = ev.get("independent_check")
    if check is not None:  # the catalogue's stored reference, run by the named verifier
        evidence.append(ReferenceSolution(
            id=check["verifier"], producer=check["verifier"], problem_digest=reference.digest(),
            result_digest=result_digest(result), value=inst.reference.value,
            tolerance=max(1e-9, inst.tolerance.objective_abs), source=inst.reference.derivation))
    provenance = Provenance(solved_problem=to_ir(ev["solved_problem"]), cache_hit=ev["cache"]["hit"],
                            cache_key=resolve(ev["cache"]["problem_digest"]))
    return reference, result, evidence, provenance


def _flip(digest: str) -> str:
    """The F03 injector itself, applied to a digest the guard recorded."""
    return F.tampered_artifact({"evidence": {"artifact_digest": digest}})["evidence"]["artifact_digest"]


def _params(fault, inst) -> dict:  # as in tests/test_e14_10_benchmark_spec.py
    other = next(i for i in P.INSTANCES if i.id != inst.id)
    stale = dict(P.neutral_form(inst), objective_exprs=["0"])
    outside = [hi + 1.0 for _, hi in inst.bounds]
    return {"F02": {"x": outside, "objective": inst.objectives[0](outside)},
            "F05": {"other_problem_digest": F.problem_digest(other)},
            "F07": {"expr": "x[0]"},
            "F08": {"previous_problem_digest": F.digest(stale)}}.get(fault.id, {})


def grade_in_memory(artifact, inst, resealed=True):
    problem, result, evidence, provenance = guard_inputs(artifact, inst)
    seal = result_digest(result)
    return verify_result(problem, result, evidence=evidence, provenance=provenance,
                         artifact_digest=seal if resealed else _flip(seal))


def grade_sealed(artifact, inst, run_dir: Path, resealed=True):
    problem, result, evidence, provenance = guard_inputs(artifact, inst)
    seal_run(provenance.solved_problem, result, evidence, run_dir, cache_hit=provenance.cache_hit,
             cache_key=provenance.cache_key)
    if not resealed:
        manifest = json.loads((run_dir / "manifest.json").read_text())
        manifest["files"]["result.json"] = _flip(manifest["files"]["result.json"])
        (run_dir / "manifest.json").write_text(json.dumps(manifest))
    return verify_run(run_dir, problem=problem)


@pytest.mark.parametrize("host", HOSTS)
def test_honest_benchmark_artifact_passes_both_paths(host, tmp_path):
    inst = P.by_id(host)
    honest = F.make_result(inst, inst.reference.x)
    for report in (grade_in_memory(honest, inst), grade_sealed(honest, inst, tmp_path)):
        assert report.verdict is Outcome.PASSED, report.to_dict()
        assert [report.statuses[s].outcome for s in ALL] == [Outcome.PASSED] * 3 + [Outcome.NOT_ASSESSED]
        assert not any(f.fatal for f in report.findings)


@pytest.mark.parametrize("host", HOSTS)
@pytest.mark.parametrize("fault", F.FAULTS, ids=lambda f: f.id)
def test_fault_is_never_a_pass_and_is_named_for_its_reason(fault, host, tmp_path):
    inst = P.by_id(host)
    faulty = fault.inject(F.make_result(inst, inst.reference.x), **_params(fault, inst))
    for path, report in (("verify_result", grade_in_memory(faulty, inst, fault.resealed)),
                         ("verify_run", grade_sealed(faulty, inst, tmp_path, fault.resealed))):
        assert report.verdict is not Outcome.PASSED, (path, report.to_dict())
        assert FindingCode(fault.reason) in report.codes, (path, sorted(report.codes))


# ------------------------------------------------------------------------------ hand-built problems
POLICY_ALL = VerificationPolicy(required=ALL)
WORKED = OptimizationProblem(
    id="worked", variables=(Variable("x", lower=0), Variable("y", lower=0)),
    objectives=(Objective("distance_sq", "x**2 + y**2"),),
    constraints=(Constraint("halfplane", "x + y", ">=", 1),))
ROSENBROCK = OptimizationProblem(
    id="rosenbrock", variables=(Variable("x", lower=-5, upper=10), Variable("y", lower=-5, upper=10)),
    objectives=(Objective("f", "100*(y - x**2)**2 + (1 - x)**2"),))
INTEGER = OptimizationProblem(
    id="integer", variables=(Variable("n", "integer", lower=0, upper=10), Variable("b", "binary"),
                             Variable("d", "discrete", choices=(0.5, 2.0))),
    objectives=(Objective("f", "n + b + d"),), constraints=(Constraint("at_least", "n + b + d", ">=", 2.5),))


def honest(problem, point, solver="solver.a", **claims):
    ev = problem.evaluate(point)
    return OptimizationResult(problem_digest=problem.digest(), solver_id=solver, termination="converged",
                              x=point, objective_values=ev.objectives, constraint_residuals=ev.residuals, **claims)


def reference(problem, result, value, **kw):
    return ReferenceSolution(id=kw.pop("id", "ref"), problem_digest=kw.pop("problem_digest", problem.digest()),
                             result_digest=kw.pop("result_digest", result_digest(result)), value=value,
                             source="worked example", **kw)


def outcomes(report):
    return {s: report.statuses[s].outcome.value for s in ALL}


def test_honest_path_passes_exactly_what_the_evidence_supports():
    result = honest(WORKED, {"x": 0.5, "y": 0.5})  # claims nothing: the guard derives every status itself
    bare = verify_result(WORKED, result, policy=POLICY_ALL)
    assert outcomes(bare) == {"solver_converged": "passed", "constraints_satisfied": "passed",
                              "independently_verified": "not_assessed", "optimality_certified": "passed"}
    assert bare.verdict is Outcome.NOT_ASSESSED and C.MISSING_EVIDENCE in bare.codes
    full = verify_result(WORKED, result, policy=POLICY_ALL, evidence=[reference(WORKED, result, 0.5)])
    assert full.verdict is Outcome.PASSED and full.statuses["independently_verified"].evidence_ids == ("ref",)
    default = verify_result(WORKED, result, evidence=[reference(WORKED, result, 0.5)])
    assert default.verdict is Outcome.PASSED
    assert default.statuses["optimality_certified"].outcome is Outcome.NOT_ASSESSED  # opt-in, not requested


def test_analytical_and_alternate_solver_evidence_validate_against_the_result():
    result = honest(WORKED, {"x": 0.5, "y": 0.5})
    pins = dict(problem_digest=WORKED.digest(), result_digest=result_digest(result))
    analytical = AnalyticalOptimum(id="qm-am", value=0.5, x={"x": 0.5, "y": 0.5},
                                   derivation="x^2 + y^2 >= (x + y)^2 / 2 >= 1/2", **pins)
    calls = []

    def alternate(problem):
        calls.append(problem.digest())
        return honest(problem, {"x": 0.5, "y": 0.5}, solver="solver.b")

    alt = AlternateSolverCheck(id="alt", solver_id="solver.b", solver=alternate, **pins)
    for item in (analytical, alt):
        report = verify_result(WORKED, result, evidence=[item])
        assert report.verdict is Outcome.PASSED, report.to_dict()
        assert report.statuses["independently_verified"].evidence_ids == (item.id,)
    assert calls == [WORKED.digest()]  # the guard ran the solver itself, on the graded problem

    wrong_x = replace(analytical, x={"x": 1.0, "y": 0.0})  # feasible but does not attain the stated value
    report = verify_result(WORKED, result, evidence=[wrong_x])
    assert report.statuses["independently_verified"].outcome is Outcome.NOT_ASSESSED and C.EVIDENCE_INVALID in report.codes
    no_proof = replace(analytical, derivation=" ")
    assert verify_result(WORKED, result, evidence=[no_proof]).verdict is not Outcome.PASSED


def test_alternate_solver_that_finds_better_refutes_and_one_that_is_the_solver_is_forged():
    result = honest(WORKED, {"x": 0.6, "y": 0.4})
    pins = dict(problem_digest=WORKED.digest(), result_digest=result_digest(result))
    better = AlternateSolverCheck(id="alt", solver_id="solver.b", **pins,
                                  solver=lambda p: honest(p, {"x": 0.5, "y": 0.5}, solver="solver.b"))
    report = verify_result(WORKED, result, evidence=[better])
    assert report.statuses["independently_verified"].outcome is Outcome.FAILED and C.EVIDENCE_DISAGREES in report.codes
    same = AlternateSolverCheck(id="alt", solver_id="solver.b", **pins,
                                solver=lambda p: honest(p, {"x": 0.6, "y": 0.4}, solver="solver.a"))
    report = verify_result(WORKED, result, evidence=[same])
    assert report.verdict is Outcome.FAILED and C.FORGED_VERIFICATION in report.codes


def test_converged_but_infeasible_result_fails_constraints():
    point = {"x": 0.2, "y": 0.2}
    result = honest(WORKED, point, solver_converged=True, constraints_satisfied=True)
    report = verify_result(WORKED, result, evidence=[reference(WORKED, result, 0.5)])
    assert report.statuses["solver_converged"].outcome is Outcome.PASSED
    assert report.statuses["constraints_satisfied"].outcome is Outcome.FAILED
    assert {C.CONSTRAINT_VIOLATION, C.CLAIM_CONTRADICTED} <= report.codes and report.verdict is Outcome.FAILED


def test_feasible_but_not_optimal_result_claiming_optimality_fails():
    result = honest(WORKED, {"x": 1.0, "y": 0.0}, optimality_certified=True)
    report = verify_result(WORKED, result, evidence=[reference(WORKED, result, 1.0)])  # even with agreeing evidence
    assert report.statuses["constraints_satisfied"].outcome is Outcome.PASSED
    assert report.statuses["optimality_certified"].outcome is Outcome.FAILED
    assert C.CLAIM_CONTRADICTED in report.codes and report.verdict is Outcome.FAILED


def test_forged_independent_claim_and_self_attestation_never_pass():
    own = Evidence("analytical", "the solver says its answer is optimal", {"value": 0.5})
    result = honest(WORKED, {"x": 0.5, "y": 0.5}, independently_verified=True, evidence=(own,))
    report = verify_result(WORKED, result)
    assert report.statuses["independently_verified"].outcome is Outcome.NOT_ASSESSED
    assert {C.SELF_ATTESTATION, C.CLAIM_CONTRADICTED} <= report.codes and report.verdict is Outcome.FAILED
    by_solver = reference(WORKED, result, 0.5, producer="solver.a")
    report = verify_result(WORKED, result, evidence=[by_solver])
    assert C.FORGED_VERIFICATION in report.codes and report.verdict is Outcome.FAILED


def test_evidence_naming_another_problem_or_result_or_a_disallowed_kind_never_counts():
    result = honest(WORKED, {"x": 0.5, "y": 0.5})
    other = replace(WORKED, constraints=(Constraint("halfplane", "x + y", ">=", 2),))
    for item in (reference(WORKED, result, 0.5, problem_digest=other.digest()),
                 reference(WORKED, result, 0.5, result_digest="0" * 64),
                 reference(WORKED, result, 0.5, problem_digest="")):
        report = verify_result(WORKED, result, evidence=[item])
        assert report.statuses["independently_verified"].outcome is Outcome.NOT_ASSESSED
        assert C.EVIDENCE_MISMATCH in report.codes and report.verdict is not Outcome.PASSED
    only_analytical = VerificationPolicy(independent_methods=("analytical",))
    report = verify_result(WORKED, result, policy=only_analytical, evidence=[reference(WORKED, result, 0.5)])
    assert C.EVIDENCE_NOT_ALLOWED in report.codes and report.verdict is not Outcome.PASSED


def test_multi_objective_result_needs_independent_evidence_for_every_objective():
    two = replace(WORKED, objectives=WORKED.objectives + (Objective("sum", "x + y"),))
    result = honest(two, {"x": 0.5, "y": 0.5})
    first = reference(two, result, 0.5, id="a", objective="distance_sq")
    report = verify_result(two, result, evidence=[first])
    assert report.statuses["independently_verified"].outcome is Outcome.NOT_ASSESSED
    both = verify_result(two, result, evidence=[first, reference(two, result, 1.0, id="b", objective="sum")])
    assert both.verdict is Outcome.PASSED and both.statuses["independently_verified"].evidence_ids == ("a", "b")
    unnamed = verify_result(two, result, evidence=[reference(two, result, 0.5)])  # ambiguous objective
    assert C.EVIDENCE_INVALID in unnamed.codes and unnamed.verdict is not Outcome.PASSED


def test_reference_tolerance_tightens_but_never_loosens_the_policy():
    result = honest(WORKED, {"x": 0.6, "y": 0.4})  # 0.52, not the optimum 0.5
    report = verify_result(WORKED, result, evidence=[reference(WORKED, result, 0.5, tolerance=1.0)])
    assert report.statuses["independently_verified"].outcome is Outcome.FAILED


def test_point_violating_integrality_binary_or_choices_fails_its_domain():
    for point in ({"n": 1.5, "b": 1.0, "d": 0.5}, {"n": 2.0, "b": 0.5, "d": 0.5}, {"n": 2.0, "b": 1.0, "d": 1.0}):
        report = verify_result(INTEGER, honest(INTEGER, point))
        assert report.statuses["constraints_satisfied"].outcome is Outcome.FAILED, point
        assert C.DOMAIN_VIOLATION in report.codes, point
    assert verify_result(INTEGER, honest(INTEGER, {"n": 2.0, "b": 0.0, "d": 0.5})) \
        .statuses["constraints_satisfied"].outcome is Outcome.PASSED


def test_point_outside_a_bound_is_a_constraint_violation():
    report = verify_result(WORKED, honest(WORKED, {"x": -0.5, "y": 2.0}))
    assert report.statuses["constraints_satisfied"].outcome is Outcome.FAILED
    assert any(f.code is C.CONSTRAINT_VIOLATION and f.path == "/x/x" for f in report.findings)


def test_nan_in_x_or_in_claims_is_never_a_pass():
    good = honest(WORKED, {"x": 0.5, "y": 0.5})
    nan_x = replace(good, x={"x": math.nan, "y": 0.5})
    report = verify_result(WORKED, nan_x, evidence=[reference(WORKED, nan_x, 0.5)])
    assert C.NON_FINITE in report.codes and report.statuses["constraints_satisfied"].outcome is Outcome.FAILED
    nan_claim = replace(good, constraint_residuals={"halfplane": math.nan}, objective_values={"distance_sq": math.inf})
    report = verify_result(WORKED, nan_claim, evidence=[reference(WORKED, nan_claim, 0.5)])
    assert C.NON_FINITE in report.codes and report.verdict is Outcome.FAILED


def test_claimed_objective_or_residual_that_disagrees_with_re_evaluation_fails():
    good = honest(WORKED, {"x": 0.5, "y": 0.5})
    for bad, code in ((replace(good, objective_values={"distance_sq": 0.25}), C.OBJECTIVE_MISMATCH),
                      (replace(good, constraint_residuals={"halfplane": -0.5}), C.RESIDUAL_MISMATCH)):
        report = verify_result(WORKED, bad, evidence=[reference(WORKED, bad, 0.5)])
        assert code in report.codes and report.verdict is Outcome.FAILED


def test_stale_problem_digest_fails():
    old = replace(WORKED, constraints=(Constraint("halfplane", "x + y", ">=", 0.9),))
    result = replace(honest(WORKED, {"x": 0.5, "y": 0.5}), problem_digest=old.digest())
    report = verify_result(WORKED, result, evidence=[reference(WORKED, result, 0.5)])
    assert C.PROBLEM_MISMATCH in report.codes and report.verdict is Outcome.FAILED


def test_solved_problem_that_weakens_a_hard_constraint_fails_and_one_that_tightens_is_noted():
    point = {"x": 0.5, "y": 0.5}
    cases = {
        "relaxed rhs": (replace(WORKED, constraints=(Constraint("halfplane", "x + y", ">=", 0.9),)),
                        C.UNAUTHORIZED_RELAXATION),
        "dropped": (replace(WORKED, constraints=()), C.UNAUTHORIZED_RELAXATION),
        "widened bound": (replace(WORKED, variables=(Variable("x", lower=-1), Variable("y", lower=0))),
                          C.UNAUTHORIZED_RELAXATION),
        "swapped objective": (replace(WORKED, objectives=(Objective("distance_sq", "x + y"),)),
                              C.FORMULATION_MISMATCH),
    }
    for name, (solved, code) in cases.items():
        result = replace(honest(WORKED, point), problem_digest=solved.digest())
        report = verify_result(WORKED, result, provenance=Provenance(solved_problem=solved),
                               evidence=[reference(WORKED, result, 0.5)])
        assert code in report.codes and report.verdict is Outcome.FAILED, name
    tightened = replace(WORKED, constraints=WORKED.constraints + (
        Constraint("agent_cut", "x - y", "<=", 0.1, origin="agent"),))
    result = replace(honest(WORKED, point), problem_digest=tightened.digest())
    report = verify_result(WORKED, result, provenance=Provenance(solved_problem=tightened),
                           evidence=[reference(WORKED, result, 0.5)])
    assert report.verdict is Outcome.PASSED and C.REVISED_PROBLEM in report.codes


def test_result_missing_fields_or_wrong_types_is_malformed():
    data = honest(WORKED, {"x": 0.5, "y": 0.5}).to_dict()
    del data["termination"]
    report = verify_result(WORKED, data)
    assert C.MALFORMED_RESULT in report.codes and report.verdict is Outcome.FAILED
    assert set(report.statuses) == set(ALL) and all(s.outcome is Outcome.NOT_ASSESSED for s in report.statuses.values())
    typed = replace(honest(WORKED, {"x": 0.5, "y": 0.5}), constraints_satisfied="yes")
    assert verify_result(WORKED, typed).verdict is Outcome.FAILED
    assert verify_result(WORKED, honest(WORKED, {"x": 0.5, "y": 0.5}).to_dict(), policy=POLICY_ALL) \
        .statuses["optimality_certified"].outcome is Outcome.PASSED  # the dict form grades like the object


def test_evidence_that_raises_is_recorded_and_never_counts():
    result = honest(WORKED, {"x": 0.5, "y": 0.5})
    pins = dict(problem_digest=WORKED.digest(), result_digest=result_digest(result))

    def crash(problem):
        raise RuntimeError("alternate solver crashed")

    for item in (AlternateSolverCheck(id="alt", solver_id="solver.b", solver=crash, **pins),
                 ReferenceSolution(id="bad", value="not a number", source="x", **pins)):
        report = verify_result(WORKED, result, evidence=[item])
        assert C.EVIDENCE_ERROR in report.codes
        assert report.statuses["independently_verified"].outcome is Outcome.NOT_ASSESSED
        assert report.verdict is not Outcome.PASSED


def test_internal_guard_errors_are_never_a_pass(monkeypatch):
    result = honest(WORKED, {"x": 0.5, "y": 0.5})
    report = verify_result(WORKED, result, policy="not a policy")
    assert report.verdict is Outcome.FAILED and C.GUARD_ERROR in report.codes
    from orbitopt.optimization.verification import certificates

    def broken(*args, **kwargs):
        raise ZeroDivisionError("certificate bug")

    monkeypatch.setattr(certificates, "kkt_certificate", broken)
    report = verify_result(WORKED, result, policy=POLICY_ALL, evidence=[reference(WORKED, result, 0.5)])
    assert report.statuses["optimality_certified"].outcome is Outcome.NOT_ASSESSED
    assert C.GUARD_ERROR in report.codes and report.verdict is Outcome.FAILED


def test_tool_objective_without_a_resolver_is_not_assessed_and_with_one_is_graded():
    problem = OptimizationProblem(id="tool", variables=(Variable("x", lower=0, upper=2),),
                                  objectives=(Objective("f", ToolCall("demo.square", {"v": "x"})),),
                                  constraints=(Constraint("cap", ToolCall("demo.square", {"v": "x"}), "<=", 1.0),))
    tools = {"demo.square": lambda v: v * v}
    result = OptimizationResult(problem_digest=problem.digest(), solver_id="s", termination="converged",
                                x={"x": 0.5}, objective_values={"f": 0.25})
    report = verify_result(problem, result)
    assert report.statuses["constraints_satisfied"].outcome is Outcome.NOT_ASSESSED and C.NOT_EVALUABLE in report.codes
    assert verify_result(problem, result, tools=tools).statuses["constraints_satisfied"].outcome is Outcome.PASSED
    assert verify_result(problem, replace(result, x={"x": 1.5}), tools=tools) \
        .statuses["constraints_satisfied"].outcome is Outcome.FAILED


def test_not_converged_solver_and_empty_requirement_never_pass():
    result = replace(honest(WORKED, {"x": 0.5, "y": 0.5}), termination="max_iterations", solver_converged=True)
    report = verify_result(WORKED, result, evidence=[reference(WORKED, result, 0.5)])
    assert report.statuses["solver_converged"].outcome is Outcome.FAILED and report.verdict is Outcome.FAILED
    nothing = verify_result(WORKED, honest(WORKED, {"x": 0.5, "y": 0.5}), policy=VerificationPolicy(required=()))
    assert nothing.verdict is Outcome.NOT_ASSESSED


# ------------------------------------------------------------------------------ sealed runs
def _sealed(tmp_path, point=None):
    result = honest(WORKED, point or {"x": 0.5, "y": 0.5})
    digest = seal_run(WORKED, result, [reference(WORKED, result, 0.5)], tmp_path)
    return result, digest


def test_sealed_run_round_trips_to_a_pass(tmp_path):
    _, digest = _sealed(tmp_path)
    assert verify_run(tmp_path, problem=WORKED, expected_manifest_digest=digest).verdict is Outcome.PASSED
    assert verify_run(tmp_path).verdict is Outcome.PASSED


def test_tampered_x_after_sealing_fails(tmp_path):
    _sealed(tmp_path)
    data = json.loads((tmp_path / "result.json").read_text())
    data["x"]["x"] = 0.5000001
    (tmp_path / "result.json").write_text(json.dumps(data, sort_keys=True, separators=(",", ":")))
    report = verify_run(tmp_path, problem=WORKED)
    assert report.verdict is Outcome.FAILED and C.DIGEST_MISMATCH in report.codes


def test_tampered_manifest_digest_or_a_resealed_rewrite_fails(tmp_path):
    _, digest = _sealed(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["files"]["evidence.json"] = _flip(manifest["files"]["evidence.json"])
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    report = verify_run(tmp_path, problem=WORKED)
    assert report.verdict is Outcome.FAILED and C.DIGEST_MISMATCH in report.codes
    rewritten = honest(WORKED, {"x": 0.5, "y": 0.5}, solver="solver.other")  # a consistent rewrite of the whole dir
    seal_run(WORKED, rewritten, [reference(WORKED, rewritten, 0.5)], tmp_path)
    assert verify_run(tmp_path, problem=WORKED).verdict is Outcome.PASSED
    report = verify_run(tmp_path, problem=WORKED, expected_manifest_digest=digest)
    assert report.verdict is Outcome.FAILED and C.DIGEST_MISMATCH in report.codes


def test_missing_or_unreadable_files_fail_without_raising(tmp_path):
    _sealed(tmp_path)
    (tmp_path / "evidence.json").unlink()
    report = verify_run(tmp_path, problem=WORKED)
    assert report.verdict is Outcome.FAILED and C.MISSING_FILE in report.codes
    assert verify_run(tmp_path / "nowhere").verdict is Outcome.FAILED
    (tmp_path / "manifest.json").write_text("{not json")
    assert C.MALFORMED_ARTIFACT in verify_run(tmp_path).codes


def test_sealed_result_missing_a_field_is_malformed(tmp_path):
    result, _ = _sealed(tmp_path)
    data = result.to_dict()
    del data["termination"]
    (tmp_path / "result.json").write_text("")  # rewrite and re-seal consistently, so only the content is wrong
    from orbitopt.optimization.verification import canonical_json, file_digest

    body = canonical_json(data)
    (tmp_path / "result.json").write_bytes(body)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["files"]["result.json"] = file_digest(body)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    report = verify_run(tmp_path, problem=WORKED)
    assert C.MALFORMED_RESULT in report.codes and report.verdict is Outcome.FAILED


def test_sealed_run_needs_the_alternate_solver_again(tmp_path):
    result = honest(WORKED, {"x": 0.5, "y": 0.5})
    alt = AlternateSolverCheck(id="alt", solver_id="solver.b", problem_digest=WORKED.digest(),
                               result_digest=result_digest(result),
                               solver=lambda p: honest(p, {"x": 0.5, "y": 0.5}, solver="solver.b"))
    seal_run(WORKED, result, [alt], tmp_path)
    assert verify_run(tmp_path, problem=WORKED).verdict is Outcome.NOT_ASSESSED
    assert verify_run(tmp_path, problem=WORKED, alternate_solvers={"solver.b": alt.solver}).verdict is Outcome.PASSED


# ------------------------------------------------------------------------------ optimality certificates
def test_kkt_certifies_the_worked_example_and_rejects_a_feasible_non_optimal_point():
    report = verify_result(WORKED, honest(WORKED, {"x": 0.5, "y": 0.5}), policy=POLICY_ALL)
    assert report.statuses["optimality_certified"].outcome is Outcome.PASSED
    assert kkt_certificate(WORKED, {"x": 0.5, "y": 0.5}, POLICY_ALL).multipliers["halfplane"] == pytest.approx(1.0)
    for point in ({"x": 1.0, "y": 0.0}, {"x": 0.6, "y": 0.4}, {"x": 2.0, "y": 2.0}):
        report = verify_result(WORKED, honest(WORKED, point), policy=POLICY_ALL)
        assert report.statuses["optimality_certified"].outcome is Outcome.FAILED, point
    infeasible = verify_result(WORKED, honest(WORKED, {"x": 0.1, "y": 0.1}), policy=POLICY_ALL)
    assert infeasible.statuses["optimality_certified"].outcome is Outcome.FAILED


def test_kkt_never_certifies_non_convex_mixed_integer_or_multi_objective_problems():
    saddle = OptimizationProblem(id="saddle", variables=(Variable("x", lower=-1, upper=1), Variable("y", lower=-1, upper=1)),
                                 objectives=(Objective("f", "x*y"),))
    two = replace(WORKED, objectives=WORKED.objectives + (Objective("g", "x"),))
    for problem, point in ((ROSENBROCK, {"x": 1.0, "y": 1.0}), (saddle, {"x": 1.0, "y": -1.0}),
                           (INTEGER, {"n": 2.0, "b": 0.0, "d": 0.5}), (two, {"x": 0.5, "y": 0.5})):
        report = verify_result(problem, honest(problem, point), policy=POLICY_ALL)
        assert report.statuses["optimality_certified"].outcome is Outcome.NOT_ASSESSED, problem.id
        assert C.CERTIFICATE_UNSUPPORTED in report.codes


CONVEX_HOSTS = [i for i in HOSTS if P.by_id(i).problem_class in ("lp", "convex_qp")]


@pytest.mark.parametrize("host", CONVEX_HOSTS)
def test_kkt_certifies_every_catalogue_lp_and_convex_qp_at_its_reference_optimum(host):
    inst, problem = P.by_id(host), IR[host]
    optimum = {f"x{i}": float(v) for i, v in enumerate(inst.reference.x)}
    assert kkt_certificate(problem, optimum, POLICY_ALL).outcome is Outcome.PASSED
    corners = [[lo for lo, _ in inst.bounds], [hi for _, hi in inst.bounds], [(lo + hi) / 2 for lo, hi in inst.bounds],
               [1.0] + [0.0] * (inst.dimension - 1)]
    feasible = [c for c in corners if F.max_violation(inst, c) <= 0
                and abs(inst.objectives[0](c) - inst.reference.value) > 1e-3]
    assert feasible, host
    point = {f"x{i}": v for i, v in enumerate(feasible[0])}
    assert kkt_certificate(problem, point, POLICY_ALL).outcome is Outcome.FAILED


def test_lower_bound_certificate_refutes_but_never_certifies_a_non_convex_problem():
    rastrigin = IR["rastrigin-d2"]
    zero = {"x0": 0.0, "x1": 0.0}
    result = honest(rastrigin, zero)
    pins = dict(problem_digest=rastrigin.digest(), result_digest=result_digest(result),
                derivation="each term x_i^2 - 10 cos(2 pi x_i) >= -10")
    true_bound = verify_result(rastrigin, result, policy=POLICY_ALL,
                               evidence=[LowerBoundCertificate(id="lb", bound=0.0, **pins)])
    assert true_bound.statuses["optimality_certified"].outcome is Outcome.NOT_ASSESSED
    for bound in (1.0, -5.0):  # above the result's own value / not attained
        report = verify_result(rastrigin, result, policy=POLICY_ALL,
                               evidence=[LowerBoundCertificate(id="lb", bound=bound, **pins)])
        assert report.statuses["optimality_certified"].outcome is Outcome.FAILED, bound
    local = honest(rastrigin, {"x0": 0.99495864, "x1": 0.0}, optimality_certified=True)  # a local minimum
    lie = LowerBoundCertificate(id="lb", bound=float(local.objective_values["f0"]), problem_digest=rastrigin.digest(),
                                result_digest=result_digest(local), derivation="a false claim")
    report = verify_result(rastrigin, local, policy=POLICY_ALL, evidence=[lie])
    # sampling misses the small basin below this bound, so it is not refuted; it is still never certified
    assert report.statuses["optimality_certified"].outcome is Outcome.NOT_ASSESSED
    assert C.CLAIM_CONTRADICTED in report.codes and report.verdict is Outcome.FAILED
    quartic = OptimizationProblem(id="quartic", variables=(Variable("x", lower=-2, upper=2),),
                                  objectives=(Objective("f", "x**4 - x**2"),))
    peak = honest(quartic, {"x": 0.0})  # a local maximum; half the box lies below the claimed bound 0
    report = verify_result(quartic, peak, policy=POLICY_ALL, evidence=[LowerBoundCertificate(
        id="lb", bound=0.0, problem_digest=quartic.digest(), result_digest=result_digest(peak), derivation="false")])
    assert report.statuses["optimality_certified"].outcome is Outcome.FAILED
    assert C.CERTIFICATE_FAILED in report.codes


def test_lower_bound_certificate_on_a_convex_problem_must_agree_with_kkt():
    result = honest(WORKED, {"x": 0.5, "y": 0.5})
    pins = dict(problem_digest=WORKED.digest(), result_digest=result_digest(result), derivation="QM-AM")
    ok = verify_result(WORKED, result, evidence=[LowerBoundCertificate(id="lb", bound=0.5, **pins)])
    assert ok.statuses["optimality_certified"].outcome is Outcome.PASSED
    wrong = verify_result(WORKED, result, evidence=[LowerBoundCertificate(id="lb", bound=0.6, **pins)])
    assert wrong.statuses["optimality_certified"].outcome is Outcome.FAILED


def test_parameters_count_as_constants_in_the_structural_check():
    problem = OptimizationProblem(id="param", variables=(Variable("x"),), parameters=(Parameter("a", 3.0),),
                                  objectives=(Objective("f", Expression("a*(x - a)**2 + sqrt(a)")),))
    report = verify_result(problem, honest(problem, {"x": 3.0}), policy=POLICY_ALL)
    assert report.statuses["optimality_certified"].outcome is Outcome.PASSED


# ------------------------------------------------------------------------------ isolation
def test_importing_the_guard_does_not_import_numpy():
    code = "import sys, orbitopt.optimization.verification; print('numpy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={"PYTHONPATH": str(REPO / "src")})
    assert out.stdout.strip() == "False"


def test_guard_imports_nothing_outside_the_ir():
    ir_modules = {"problem", "result", "expressions", "errors", "units"}
    for path in (REPO / "src/orbitopt/optimization/verification").glob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            found = re.match(r"\s*(?:from\s+(\S+)\s+import|import\s+(\S+))", line)
            if not found:
                continue
            module = found.group(1) or found.group(2)
            assert not module.startswith("orbitopt"), (path.name, line)
            if module.startswith(".."):
                assert module[2:] in ir_modules, (path.name, line)


def test_fault_inputs_are_not_mutated():
    inst = P.by_id(HOSTS[0])
    artifact = F.make_result(inst, inst.reference.x)
    before = copy.deepcopy(artifact)
    grade_in_memory(artifact, inst)
    assert artifact == before
