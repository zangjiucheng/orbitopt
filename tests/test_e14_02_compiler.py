"""E14-02 natural-language problem compiler, exercised with scripted providers.

Every model reply in this file and in ``tests/data/agent/formulation_cases.json`` is scripted. These tests show
that the compiler turns a given reply into the right outcome, never returns an invalid IR problem, never runs
model output, catches unauthorised and dropped constraints, and accounts for every token. They say nothing about
how accurately a real language model formulates problems; E14-10 measures that with a real model.
"""

from __future__ import annotations

import ast
import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.e14_spec import formulation, problems
from orbitopt.agent.compiler import (
    COMPILER_OUTPUT_SCHEMA,
    MAX_REPAIR_ROUNDS,
    CompileFailed,
    Compiled,
    NeedsClarification,
    RejectedUnauthorizedChange,
    Unsupported,
    compile_problem,
    extract_statements,
)
from orbitopt.agent.compiler.schema import draft_to_ir_dict
from orbitopt.agent.equivalence import semantically_equivalent
from orbitopt.agent.providers import (
    MockProvider,
    MockReply,
    ProviderUnavailableError,
    RecordingProvider,
    ReplayProvider,
    Usage,
)
from orbitopt.agent.providers.mock import mock_usage
from orbitopt.optimization import OptimizationProblem, parse_problem, validate_problem

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "tests" / "data" / "agent" / "formulation_cases.json").read_text(encoding="utf-8"))
CASES = DATA["cases"]
BY_ID = {case["id"]: case for case in CASES}
OUTCOMES = {Compiled: "compile", NeedsClarification: "needs_clarification", Unsupported: "unsupported",
            RejectedUnauthorizedChange: "rejected_unauthorized_change", CompileFailed: "compile_failed"}
CJK = re.compile(r"[㐀-鿿]")


def _quadrature_tool(f, exact):
    def tool(rule, n, a, b):
        q = problems.quadrature(f, a, b, rule, n)
        return math.inf if math.isnan(q) else abs(q - exact)
    return tool


def _tools():
    import numpy as np
    by_integrand = {"exp(t)": _quadrature_tool(np.exp, math.e - 1.0),
                    "1/(1+25*t**2)": _quadrature_tool(lambda t: 1.0 / (1.0 + 25.0 * t ** 2), 0.4 * math.atan(5.0))}
    return {"quadrature.abs_error": lambda integrand, **args: by_integrand[integrand](**args)}


def _run(case, replies=None, sentinel=None):
    replies = list(case["replies"] if replies is None else replies)
    if sentinel is not None:
        replies = [r.replace("{sentinel}", str(sentinel)) for r in replies]
    provider = MockProvider(replies)
    base = case.get("base_problem")
    outcome = compile_problem(case["prompt"], provider, base_problem=parse_problem(base) if base else None,
                              known_tool_ids=case.get("known_tool_ids", ()), attachments=case.get("attachments", ()),
                              unit_hints=case.get("unit_hints"))
    return outcome, provider


def _reference_x(case):
    if not case.get("reference_problem"):
        return None
    return problems.by_id(case["reference_problem"]).reference.x


def _equivalence(case, problem):
    return semantically_equivalent(problem, parse_problem(case["reference_ir"]), aliases=case.get("aliases"),
                                   reference_x=_reference_x(case), seed=1)


def _repair_turns(provider):
    """The compiler's repair messages, one per round after the first."""
    return [r.messages[-1].content for r in provider.requests if r.messages[-1].content.startswith("The compiler")]


# ------------------------------------------------------------------------------------------------ suite shape
def test_suite_has_the_required_cases_in_the_benchmark_format():
    assert len(CASES) >= 30 and len(BY_ID) == len(CASES)
    allowed = set(formulation.OUTCOMES) | {"compile_failed"}
    for case in CASES:
        assert {"id", "prompt", "expected_outcome", "reference_problem", "replies", "notes"} <= case.keys()
        assert case["expected_outcome"] in allowed, case["id"]
        assert case["replies"] and case["notes"]
        if case["expected_outcome"] == "compile":
            assert case["reference_ir"] is not None, case["id"]
    for example in formulation.EXAMPLES:
        assert BY_ID[example.id]["prompt"] == example.prompt
        assert BY_ID[example.id]["expected_outcome"] == example.expected_outcome
    assert sum(bool(CJK.search(case["prompt"])) for case in CASES) >= 6
    families = {}
    for case in CASES:
        if case["expected_outcome"] == "compile" and case["reference_problem"]:
            families.setdefault(problems.by_id(case["reference_problem"]).family, set()).add(case["reference_problem"])
    for family in ("rosenbrock", "constrained_quadratic", "linear_program", "knapsack", "integration", "bi_objective"):
        assert len(families.get(family, ())) >= 2, family
    for outcome in allowed:
        assert sum(case["expected_outcome"] == outcome for case in CASES) >= 4, outcome
    assert sum(len(case.get("negative_controls", ())) for case in CASES) >= 5


# ---------------------------------------------------------------------------------------------- every case
@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_case_outcome(case, tmp_path):
    expect = case.get("expect", {})
    sentinel = tmp_path / "pwned" if expect.get("sentinel") else None
    outcome, provider = _run(case, sentinel=sentinel)
    assert OUTCOMES[type(outcome)] == case["expected_outcome"], (case["id"], outcome)
    assert outcome.outcome == case["expected_outcome"]
    if isinstance(outcome, Compiled):
        report = _equivalence(case, outcome.problem)
        assert report.equivalent, (case["id"], report.reasons)
    if "attempts" in expect:
        assert outcome.attempts == expect["attempts"]
    if "provider_calls" in expect:
        assert len(provider.requests) == expect["provider_calls"]
    for code in expect.get("repair_codes", ()):
        assert any(f"[{code}]" in turn for turn in _repair_turns(provider)), (case["id"], code)
    if "question_kinds" in expect:
        assert {q.kind for q in outcome.questions} == set(expect["question_kinds"])
    if "question_about" in expect:
        assert [q.about for q in outcome.questions] == expect["question_about"]
    if "category" in expect:
        assert outcome.reason.category == expect["category"]
    if "violation_kinds" in expect:
        assert {v.kind for v in outcome.violations} == set(expect["violation_kinds"])
    if "error_codes" in expect:
        assert {e.code.value for e in outcome.errors} == set(expect["error_codes"])
    if sentinel is not None:
        assert str(sentinel) in provider.requests[1].messages[-2].content  # the payload really was sent
        assert not sentinel.exists()


def test_compiled_problems_record_user_and_compiler_provenance():
    outcome, _ = _run(BY_ID["FC-005"])
    assert isinstance(outcome, Compiled)
    assert [a.about for a in outcome.assumptions] == ["variables/rule", "variables/n"]
    assert outcome.problem.metadata["assumptions"][1]["about"] == "variables/n"
    assert all(c.origin == "user" and c.locked for c in outcome.problem.constraints)
    outcome, _ = _run(BY_ID["FC-001"])
    assert outcome.assumptions == () and all(o.origin == "user" for o in outcome.problem.objectives)


def test_unexplained_assumed_items_are_listed_for_the_user():
    case = BY_ID["FC-005"]
    reply = json.loads(case["replies"][0])
    reply["assumptions"] = []
    reply["problem"]["constraints"].append({"name": "cheap", "expression": "n", "sense": "<=", "rhs": 512,
                                            "source": "assumed"})
    outcome, _ = _run(case, [json.dumps(reply)])
    assert isinstance(outcome, Compiled)
    assert [a.about for a in outcome.assumptions] == ["variables/rule", "variables/n", "constraints/cheap"]
    cheap = outcome.problem.constraints[1]
    assert cheap.origin == "compiler" and not cheap.locked


# ---------------------------------------------------------------------------------- references vs catalogue
CATALOGUE_REFS = sorted({case["reference_problem"]: case for case in CASES if case["reference_ir"]
                         and case["reference_ir"]["id"] == case["reference_problem"]
                         and not case.get("base_problem")}.items())


@pytest.mark.parametrize("instance_id,case", CATALOGUE_REFS, ids=[i for i, _ in CATALOGUE_REFS])
def test_reference_ir_agrees_with_the_catalogue_callables(instance_id, case):
    """The hand-written reference IR evaluates like the benchmark's independent Python callables."""
    inst = problems.by_id(instance_id)
    ref = parse_problem(case["reference_ir"])
    names = [v.name for v in ref.variables]
    assert len(names) == inst.dimension
    for v, (lo, hi) in zip(ref.variables, inst.bounds):
        assert (v.kind == "binary" and (lo, hi) == (0.0, 1.0)) or (v.lower, v.upper) == (lo, hi)
    assert all(o.sense == inst.sense for o in ref.objectives)
    tools = _tools()
    rng = random.Random(7)
    for _ in range(40):
        x = [float(rng.randint(int(lo), int(hi))) if inst.variable_type != "continuous" else rng.uniform(lo, hi)
             for lo, hi in inst.bounds]
        ev = ref.evaluate(dict(zip(names, x)), tools)
        for o, f in zip(ref.objectives, inst.objectives):
            assert math.isclose(ev.objectives[o.name], f(x), rel_tol=1e-9, abs_tol=1e-9)
        for c in inst.constraints:
            assert math.isclose(ev.residuals[c.name], c.fn(x), rel_tol=1e-9, abs_tol=1e-9)
    if inst.reference.x is not None and inst.reference.indicator == "objective":
        ev = ref.evaluate(dict(zip(names, inst.reference.x)), tools)
        assert math.isclose(next(iter(ev.objectives.values())), inst.reference.value, rel_tol=1e-9, abs_tol=1e-9)


# ------------------------------------------------------------------------------------------ negative controls
def _patched(reply_text, patch):
    reply = json.loads(reply_text)
    for op in patch:
        *head, last = op["path"]
        target = reply
        for key in head:
            target = target[key]
        if op["op"] == "set":
            target[last] = op["value"]
        else:
            del target[last]
    return reply


CONTROLS = [(case["id"], control) for case in CASES for control in case.get("negative_controls", ())]


@pytest.mark.parametrize("case_id,control", CONTROLS, ids=[f"{i}-{c['name']}" for i, c in CONTROLS])
def test_negative_control_is_not_equivalent(case_id, control):
    case = BY_ID[case_id]
    reply = _patched(case["replies"][-1], control["patch"])
    problem = parse_problem(draft_to_ir_dict(reply["problem"], problem_id="control", assumptions=()))
    report = _equivalence(case, problem)
    assert not report.equivalent, control["name"]
    assert any(control["reason_contains"] in reason for reason in report.reasons), report.reasons
    # Through the compiler the perturbation is either refused or compiles to something not equivalent.
    outcome, _ = _run(case, [json.dumps(reply)] * (1 + MAX_REPAIR_ROUNDS))
    assert not (isinstance(outcome, Compiled) and _equivalence(case, outcome.problem).equivalent)


def test_equivalence_names_the_difference_and_never_guesses_a_rename():
    case = BY_ID["FC-016"]
    compiled, _ = _run(case)
    ref = parse_problem(case["reference_ir"])
    unmapped = semantically_equivalent(compiled.problem, ref)
    assert not unmapped.equivalent
    assert any("'food_1' is only in the compiled problem" in r for r in unmapped.reasons)
    swapped = semantically_equivalent(compiled.problem, ref, aliases={"food_1": "x1", "food_2": "x0"})
    assert not swapped.equivalent
    assert semantically_equivalent(compiled.problem, ref, aliases=case["aliases"]).equivalent
    assert semantically_equivalent(ref, ref).equivalent
    open_side = _equivalence(BY_ID["FC-001"], _run(BY_ID["FC-001"])[0].problem)
    assert open_side.equivalent and any("open in the compiled problem" in n for n in open_side.notes)


# -------------------------------------------------------------------------------------------- safety bars
def test_no_outcome_ever_carries_an_invalid_ir_problem():
    for case in CASES:
        outcome, _ = _run(case, sentinel=Path("/nonexistent/sentinel"))
        held = [value for value in vars(outcome).values() if isinstance(value, OptimizationProblem)]
        if isinstance(outcome, Compiled):
            assert held == [outcome.problem] and validate_problem(outcome.problem) == []
            assert parse_problem(outcome.problem.to_dict()) == outcome.problem
        else:
            assert held == [], case["id"]
    invalid = parse_problem(BY_ID["FC-001"]["reference_ir"])
    invalid = OptimizationProblem(id="bad", variables=invalid.variables, objectives=())
    with pytest.raises(ValueError):
        Compiled(invalid, (), Usage(0, 0, 0), 1)


def test_compiler_never_evaluates_or_imports_model_text():
    forbidden_calls = {"eval", "exec", "compile", "__import__", "getattr", "setattr", "open"}
    sources = sorted((ROOT / "src" / "orbitopt" / "agent" / "compiler").glob("*.py"))
    sources.append(ROOT / "src" / "orbitopt" / "agent" / "equivalence.py")
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_calls, (path.name, node.func.id)
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                for module in modules:
                    assert not module.startswith(("importlib", "subprocess", "os", "pickle", "marshal")), path.name


def test_compiler_imports_no_numpy_and_no_in_flight_or_mission_package():
    code = ("import sys, orbitopt.agent.compiler, orbitopt.agent.equivalence; "
            "print(sorted(m for m in sys.modules if m == 'numpy' or m.startswith(('orbitopt.mission', "
            "'orbitopt.flightlab', 'orbitopt.rendezvous', 'orbitopt.agent.tools', 'orbitopt.agent.verification', "
            "'orbitopt.optimization.solvers'))))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                            env={"PYTHONPATH": str(ROOT / "src")})
    assert result.stdout.strip() == "[]"


# ----------------------------------------------------------------------------------- budget and accounting
@pytest.mark.parametrize("case_id", ["FC-026", "FC-028", "FC-044", "FC-047"])
def test_usage_sums_every_provider_call(case_id):
    case = BY_ID[case_id]
    outcome, provider = _run(case)
    replies = case["replies"][:len(provider.requests)]
    calls = sum((mock_usage(request, MockReply(text)) for request, text in zip(provider.requests, replies)),
                Usage(0, 0, 0))
    assert outcome.usage == calls and calls.total > 0  # FC-044 ends in StructuredOutputError, which carries usage


def test_repair_rounds_are_bounded():
    case = BY_ID["FC-047"]
    outcome, provider = _run(case, case["replies"] * 3)
    assert MAX_REPAIR_ROUNDS == 2
    assert isinstance(outcome, CompileFailed) and outcome.attempts == 3 and len(provider.requests) == 3
    assert outcome.raw == case["replies"][-1]


def test_infrastructure_failures_propagate_instead_of_becoming_outcomes():
    provider = MockProvider([ProviderUnavailableError("timeout")])
    with pytest.raises(ProviderUnavailableError):
        compile_problem(BY_ID["FC-001"]["prompt"], provider)


def test_record_then_replay_gives_the_same_outcome_without_a_model(tmp_path):
    case = BY_ID["FC-026"]
    cassette = tmp_path / "compile.jsonl"
    mock = MockProvider(list(case["replies"]))
    recorded = compile_problem(case["prompt"], RecordingProvider(mock, cassette))
    calls = len(mock.requests)
    replay = ReplayProvider(cassette)
    replayed = compile_problem(case["prompt"], replay)
    assert replayed == recorded and isinstance(replayed, Compiled) and replayed.attempts == 2
    assert len(mock.requests) == calls == 2
    replay.assert_exhausted()


def test_structured_output_schema_is_what_the_provider_receives():
    outcome, provider = _run(BY_ID["FC-001"])
    assert provider.requests[0].response_schema == COMPILER_OUTPUT_SCHEMA
    assert provider.requests[0].messages[0].role == "system"


# --------------------------------------------------------------------------------- coverage check limits
def test_coverage_extraction_finds_stated_bounds_in_both_languages():
    found = {(s.direction, s.value, s.unit) for s in extract_statements(
        "x + y >= 1, 0 <= x, at most 4 km/s, no more than 5, between -2 and 3, [0, 1], nonnegative")}
    assert found >= {(">=", 1.0, None), (">=", 0.0, None), ("<=", 4.0, "km/s"), ("<=", 5.0, None),
                     (">=", -2.0, None), ("<=", 3.0, None), ("<=", 1.0, None)}
    assert ("<=", 5.0, None) in found and (">=", 5.0, None) not in found  # "no more than" is not "more than"
    chinese = {(s.direction, s.value) for s in extract_statements(BY_ID["FC-Z02"]["prompt"])}
    assert chinese == {(">=", 4.0), (">=", 6.0), ("<=", 10.0)}


def test_coverage_extraction_misses_bounds_without_a_comparator_phrase():
    """Documented limit: capacities stated as facts, and numbers in words beyond ten, are not extracted."""
    assert extract_statements("Plant 1 has 4 hours and plant 2 has 12 hours.") == []
    assert extract_statements("The bag holds twenty kilograms.") == []
