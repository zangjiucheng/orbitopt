"""E14-07 experiment memory: record, cache, resume, query and compare real E14-05 runs.

Every run here is produced by ``orbitopt.agent.runtime.solve`` with the rule-based planner, a recorded
planner or the LLM planner on a ``MockProvider``, on E14-10 catalogue instances or the runtime's worked
problem. No test needs a network. Expected values come from the runs' own outcomes and the catalogue.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import multiprocessing
import pickle
import sqlite3
import statistics
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from orbitopt.agent.experiments import (MIGRATIONS, SCHEMA_VERSION, ArtifactError, ArtifactStore, ExperimentDB,
                                        ExperimentRecord, IgnoredReason, Integrity, InvalidIdError,
                                        MalformedRecordError, RunIntegrityError, SchemaVersionError, UnknownRunError,
                                        UnsafePathError, VersionPolicy, aggregate, cache_key, compare, export_run,
                                        lookup, orbitopt_version, query, record_run, resume_plan)
from orbitopt.agent.providers import MockProvider
from orbitopt.agent.runtime import (LLMPlanner, Proposal, RecordedPlanner, RuleBasedPlanner, RunStatus, StopReason,
                                    replay, safe_relative, save_run, solve, trace_digest)
from orbitopt.optimization import (Budget, Constraint, Objective, OptimizationProblem, Parameter, ToolCall, Variable,
                                   VerificationPolicy)
from orbitopt.optimization.classify import Form
from orbitopt.optimization.solvers import DifferentialEvolutionSolver, LbfgsbSolver, SolverRegistry
from orbitopt.optimization.verification import FindingCode, canonical_json, file_digest, verify_result

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "benchmarks/e14_spec"


def _catalogue():
    if "e14_spec" not in sys.modules:
        spec = importlib.util.spec_from_file_location("e14_spec", PKG / "__init__.py",
                                                      submodule_search_locations=[str(PKG)])
        module = importlib.util.module_from_spec(spec)
        sys.modules["e14_spec"] = module
        spec.loader.exec_module(module)
    return sys.modules["e14_spec"].problems


P = _catalogue()


class _Inline(ast.NodeTransformer):
    def __init__(self, inst):
        self.values = {"n": inst.dimension, **{k: v for k, v in inst.data.items()
                                               if isinstance(v, (int, float)) and not isinstance(v, bool)}}

    def visit_Name(self, node):
        return ast.Constant(self.values[node.id]) if node.id in self.values else node


def to_ir(inst) -> OptimizationProblem:
    """A catalogue instance as IR, as the E14-05 tests build it (no tool-evaluated instances here)."""
    def source(expr: str) -> str:
        return ast.unparse(_Inline(inst).visit(ast.parse(expr, mode="eval")).body)

    vectors = tuple(Parameter(f"{name}{i}", float(v)) for name, values in sorted(inst.data.items())
                    if isinstance(values, tuple) and all(isinstance(v, (int, float)) for v in values)
                    for i, v in enumerate(values))
    return OptimizationProblem(
        id=inst.id, variables=tuple(Variable(f"x{i}", kind=inst.variable_type, lower=lo, upper=hi)
                                    for i, (lo, hi) in enumerate(inst.bounds)),
        parameters=(Parameter("pi", math.pi), Parameter("e", math.e), *vectors),
        objectives=tuple(Objective(f"f{k}", source(e), sense=inst.sense) for k, e in enumerate(inst.objective_exprs)),
        constraints=tuple(Constraint(c.name, source(c.expr), "==" if c.kind == "eq" else "<=", 0.0)
                          for c in inst.constraints),
        verification=VerificationPolicy(constraint_tol=max(inst.tolerance.constraint_abs, 1e-15)),
        metadata={"family": inst.family})


CQP, WYNDOR, RASTRIGIN = (to_ir(P.by_id(i)) for i in ("cqp-worked-d2", "lp-wyndor-d2", "rastrigin-d5"))


def forced(*proposals) -> RecordedPlanner:
    return RecordedPlanner([Proposal(p) for p in (*proposals, {"action": "stop"})])


def saved(outcome, path: Path) -> Path:
    save_run(outcome, path)
    return path


@pytest.fixture
def db(tmp_path):
    with ExperimentDB(tmp_path / "db") as database:
        yield database


def without_clock(data):
    if isinstance(data, dict):
        return {k: without_clock(v) for k, v in data.items() if k not in ("wall_s", "wall_time_s", "latency_s")}
    return [without_clock(v) for v in data] if isinstance(data, list) else data


def count(db) -> int:
    return query(db, limit=1).total


# ------------------------------------------------------------------------------------- round trip, idempotence
def test_a_recorded_run_reads_back_with_the_runs_own_numbers(db, tmp_path):
    outcome = solve(CQP, planner=RuleBasedPlanner(), seed=0)
    digest = save_run(outcome, tmp_path / "run")
    record = record_run(db, tmp_path / "run", expected_manifest_digest=digest)

    assert record.run_id == digest and record.problem_digest == CQP.digest()
    assert record.status is RunStatus.VERIFIED and record.integrity is Integrity.CHECKED
    assert record.objective == outcome.result.objective_values["f0"]
    assert abs(record.objective - P.by_id("cqp-worked-d2").reference.value) < 1e-8
    assert record.statuses == outcome.statuses
    assert record.trace_digest == outcome.trace.digest() and record.family == "constrained_quadratic"
    assert record.usage == outcome.trace.usage and record.limits == outcome.trace.limits
    assert [s.solver_id for s in record.steps][0] == outcome.trace.iterations[0].decision.solver_id
    assert record.max_violation == 0.0 and record.orbitopt_version == orbitopt_version()
    assert ExperimentRecord.from_dict(record.to_dict()) == record

    with ExperimentDB(tmp_path / "db") as reopened:
        page = query(reopened, problem_digest=CQP.digest())
        assert page.total == 1 and page.records == (record,)
        assert json.loads(json.dumps(page.to_dict()))["records"][0]["run_id"] == digest
        assert digest[:19] in page.to_text()


def test_recording_the_same_run_twice_gives_one_record(db, tmp_path):
    save_run(solve(CQP, planner=RuleBasedPlanner()), tmp_path / "run")
    first = record_run(db, tmp_path / "run")
    blobs = sorted(p.name for p in (tmp_path / "db/artifacts").rglob("*.json"))
    again = record_run(db, tmp_path / "run")
    assert again == first and again.created_at == first.created_at
    assert count(db) == 1
    assert sorted(p.name for p in (tmp_path / "db/artifacts").rglob("*.json")) == blobs


class Untouchable(LbfgsbSolver):
    """A solver that default ingest must never run."""

    CAPABILITY = replace(LbfgsbSolver.CAPABILITY, solver_id="scipy.slsqp")

    def solve(self, *args, **kwargs):
        raise AssertionError("ingest ran a solver")


def test_default_ingest_runs_no_solver_and_replay_is_opt_in(db, tmp_path):
    save_run(solve(CQP, planner=RuleBasedPlanner()), tmp_path / "run")
    nothing = SolverRegistry([Untouchable()])
    checked = record_run(db, tmp_path / "run", registry=nothing)
    assert checked.integrity is Integrity.CHECKED and checked.verified
    with ExperimentDB(tmp_path / "db2") as other:
        replayed = record_run(other, tmp_path / "run", verify="replay")
        assert replayed.integrity is Integrity.REPLAYED
        assert {k: v for k, v in replayed.to_dict().items() if k not in ("created_at", "integrity")} == \
            {k: v for k, v in checked.to_dict().items() if k not in ("created_at", "integrity")}
        with pytest.raises(ValueError):
            record_run(other, tmp_path / "run", verify="trust")


def test_a_checked_record_never_serves_the_cache(db, tmp_path):
    outcome = solve(CQP, planner=RuleBasedPlanner())
    record = record_run(db, saved(outcome, tmp_path / "run"))
    assert record.integrity is Integrity.CHECKED and record.verified
    assert not _ask(db, CQP, record.best_step).hit


def test_an_exported_run_replays_equal_from_the_store_alone(db, tmp_path):
    save_run(solve(WYNDOR, planner=RuleBasedPlanner()), tmp_path / "run")
    record = record_run(db, tmp_path / "run")
    out = export_run(db, record.run_id, tmp_path / "exported")
    assert replay(out, expected_manifest_digest=record.run_id).equal


# ------------------------------------------------------------------------------------- tampering
def _edit(path: Path) -> None:
    text = path.read_text()
    path.write_text(text.replace("0.5", "0.4", 1) if "0.5" in text else text.replace("1", "2", 1))


@pytest.mark.parametrize("target", ["trace.json", "problem.json", "sealed/result.json", "sealed/manifest.json",
                                    "results/iteration-0001.json", "manifest.json"])
def test_a_tampered_run_is_refused_and_records_nothing(db, tmp_path, target):
    digest = save_run(solve(CQP, planner=RuleBasedPlanner()), tmp_path / "run")
    _edit(tmp_path / "run" / target)
    with pytest.raises(RunIntegrityError):
        record_run(db, tmp_path / "run", expected_manifest_digest=digest)
    assert count(db) == 0


def _better_objective(trace):
    trace["iterations"][0]["result"]["objective_values"]["f0"] = 0.25  # better than the optimum, never computed
    trace["digest"] = trace_digest(trace)


def _wrong_trace_digest(trace):
    trace["digest"] = "0" * 64


@pytest.mark.parametrize("forge, found", [(_better_objective, "objective_values"), (_wrong_trace_digest, "digest")])
def test_a_forger_who_rewrites_the_manifest_is_still_refused(db, tmp_path, forge, found):
    run = tmp_path / "run"
    save_run(solve(CQP, planner=RuleBasedPlanner()), run)
    trace = json.loads((run / "trace.json").read_text())
    forge(trace)
    (run / "trace.json").write_bytes(canonical_json(trace))
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["files"]["trace.json"] = file_digest(canonical_json(trace))
    manifest["trace_digest"] = trace["digest"]
    (run / "manifest.json").write_bytes(canonical_json(manifest))
    with pytest.raises(RunIntegrityError) as caught:
        record_run(db, run)
    assert any(found in p for p in caught.value.problems)
    assert count(db) == 0


@pytest.mark.parametrize("name", ["../outside.json", "results/../../outside.json", "OUTSIDE"])
def test_a_manifest_that_names_a_file_outside_the_run_is_refused(db, tmp_path, name):
    run = tmp_path / "run"
    save_run(solve(CQP, planner=RuleBasedPlanner()), run)
    (tmp_path / "outside.json").write_bytes(b"{}")
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["files"][str(tmp_path / "outside.json") if name == "OUTSIDE" else name] = file_digest(b"{}")
    (run / "manifest.json").write_bytes(canonical_json(manifest))
    with pytest.raises(UnsafePathError):
        record_run(db, run)
    assert count(db) == 0


def test_a_directory_that_is_not_a_run_is_refused(db, tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(RunIntegrityError):
        record_run(db, tmp_path / "empty")
    save_run(solve(CQP, planner=RuleBasedPlanner()), tmp_path / "run")
    with pytest.raises(RunIntegrityError):
        record_run(db, tmp_path / "run", expected_manifest_digest="sha256:" + "0" * 64)


# ------------------------------------------------------------------------------------- cache
def _recorded(db, tmp_path, problem=CQP, **solve_args):
    outcome = solve(problem, planner=solve_args.pop("planner", RuleBasedPlanner()), **solve_args)
    record = record_run(db, saved(outcome, tmp_path / f"run-{len(list(tmp_path.iterdir()))}"), verify="replay",
                        tools=solve_args.get("tools"), registry=solve_args.get("registry"))
    return outcome, record


def _ask(db, problem, step, **kw):
    return lookup(db, problem, solver_id=kw.pop("solver_id", step.solver_id), options=kw.pop("options", step.options),
                  seed=kw.pop("seed", step.seed), x0=kw.pop("x0", step.x0),
                  evaluations=kw.pop("evaluations", step.evaluations), **kw)


def test_a_cache_hit_returns_the_verified_result_marked_as_a_hit(db, tmp_path):
    outcome, record = _recorded(db, tmp_path)
    found = _ask(db, CQP, record.best_step)
    assert found.hit and found.run_id == record.run_id and found.ignored == ()
    assert found.result.provenance.cache_key == CQP.digest()
    assert found.result.x == outcome.result.x and found.result.objective_values == outcome.result.objective_values
    assert found.evaluations == 0 and found.recorded_evaluations == outcome.trace.usage.evaluations > 0
    assert FindingCode.STALE_CACHE not in verify_result(CQP, found.result).codes
    # The same result claimed for another problem is the stale cache the guard catches.
    other = replace(CQP, id="other", constraints=())
    assert FindingCode.STALE_CACHE in verify_result(other, replace(found.result, problem_digest=other.digest())).codes


@pytest.mark.parametrize("change", [{"seed": 7}, {"evaluations": 2991}, {"options": {"ftol": 1e-3}},
                                    {"x0": {"x0": 1.0, "x1": 1.0}}, {"solver_id": "scipy.differential_evolution"}])
def test_any_other_solve_is_a_miss(db, tmp_path, change):
    _, record = _recorded(db, tmp_path)
    found = _ask(db, CQP, record.best_step, **change)
    assert not found.hit and found.result is None and found.ignored == ()


def test_options_and_start_points_are_normalised_into_the_key():
    base = cache_key("a" * 64, "s", {"popsize": 15, "tol": None}, 0, {}, 100, version="1.2.3")
    assert base == cache_key("a" * 64, "s", {"popsize": 15.0}, 0, None, 100, version="1.2.3")
    assert base != cache_key("a" * 64, "s", {"popsize": 16}, 0, None, 100, version="1.2.3")


@pytest.mark.parametrize("version, policy, hit", [
    ("9.9.9", VersionPolicy.EXACT, False), (None, VersionPolicy.EXACT, True),
    ("0.1.99", VersionPolicy.MINOR, True), ("0.2.0", VersionPolicy.MINOR, False), ("5.0", VersionPolicy.ANY, True)])
def test_the_version_policy_decides_which_versions_share_an_entry(db, tmp_path, version, policy, hit):
    assert orbitopt_version().startswith("0.1.")  # the installed version these cases are written against
    _, record = _recorded(db, tmp_path)
    assert _ask(db, CQP, record.best_step, version=version, version_policy=policy).hit is hit


def test_a_corrupt_artifact_is_ignored_not_returned(db, tmp_path):
    _, record = _recorded(db, tmp_path)
    sealed_result = record.files["sealed/result.json"]
    (blob,) = (tmp_path / "db/artifacts").rglob(sealed_result.split(":")[1] + ".json")
    blob.write_bytes(blob.read_bytes().replace(b"0.5", b"0.4", 1))
    found = _ask(db, CQP, record.best_step)
    assert not found.hit and [i.reason for i in found.ignored] == [IgnoredReason.CORRUPT]
    blob.unlink()
    found = _ask(db, CQP, record.best_step)
    assert not found.hit and [i.reason for i in found.ignored] == [IgnoredReason.CORRUPT]


def test_an_unverified_or_cancelled_run_is_never_served(db, tmp_path):
    _, record = _recorded(db, tmp_path, planner=forced({"action": "solve", "solver_id": "scipy.differential_evolution",
                                                        "evaluations": 40, "seed": 3}))
    assert record.status is RunStatus.UNVERIFIED_BEST
    found = _ask(db, CQP, record.best_step)
    assert not found.hit and found.ignored == ()  # never entered, not merely skipped
    checks = []
    _, cancelled = _recorded(db, tmp_path, cancel=lambda: checks.append(1) or len(checks) > 1)
    assert cancelled.status is RunStatus.CANCELLED and not _ask(db, CQP, cancelled.best_step).hit


class Counter:
    """An independent evaluation counter: a tool every evaluation of the problem calls once."""

    def __init__(self, shift: float = 0.0):
        self.calls, self.shift = 0, shift

    def __call__(self, x, y):
        self.calls += 1
        return (x - 1.0) ** 2 + (y + 0.5) ** 2 + self.shift


COUNTED = OptimizationProblem(
    id="counted", variables=(Variable("x", lower=-2.0, upper=2.0), Variable("y", lower=-2.0, upper=2.0)),
    objectives=(Objective("f", ToolCall("test.counted.v1", {"x": "x", "y": "y"})),))


class AnyFormLocal(LbfgsbSolver):
    CAPABILITY = replace(LbfgsbSolver.CAPABILITY, solver_id="test.lbfgsb_any", max_form=Form.BLACK_BOX)


def counted_registry() -> SolverRegistry:
    return SolverRegistry([DifferentialEvolutionSolver(), AnyFormLocal()])


def test_a_hit_spends_no_solver_evaluation_on_an_independent_counter(db, tmp_path):
    counter = Counter()
    outcome = solve(COUNTED, planner=RuleBasedPlanner(), registry=counted_registry(),
                    tools={"test.counted.v1": counter})
    assert outcome.status is RunStatus.VERIFIED
    record = record_run(db, saved(outcome, tmp_path / "run"), verify="replay", tools={"test.counted.v1": Counter()},
                        registry=counted_registry())
    counter.calls = 0
    found = _ask(db, COUNTED, record.best_step, tools={"test.counted.v1": counter})
    assert found.hit and found.evaluations == 0
    assert counter.calls == found.check_evaluations == 1  # the guard's re-evaluation of the stored point, only
    assert found.recorded_evaluations == outcome.trace.usage.evaluations > 100


def test_an_entry_whose_tool_now_answers_differently_is_stale(db, tmp_path):
    outcome = solve(COUNTED, planner=RuleBasedPlanner(), registry=counted_registry(),
                    tools={"test.counted.v1": Counter()})
    record = record_run(db, saved(outcome, tmp_path / "run"), verify="replay", tools={"test.counted.v1": Counter()},
                        registry=counted_registry())
    found = _ask(db, COUNTED, record.best_step, tools={"test.counted.v1": Counter(shift=1.0)})
    assert not found.hit and [i.reason for i in found.ignored] == [IgnoredReason.STALE]
    assert "objective_mismatch" in found.ignored[0].detail


def test_an_index_row_that_points_at_another_problems_run_is_stale(db, tmp_path):
    _, record = _recorded(db, tmp_path)
    step = record.best_step
    other = replace(CQP, id="cqp-other", objectives=(Objective("f0", "x0**2 + x1**2 + 1"),))
    key = cache_key(other.digest(), step.solver_id, step.options, step.seed, step.x0, step.evaluations,
                    version=orbitopt_version())
    with sqlite3.connect(tmp_path / "db/index.sqlite") as conn:  # an index edited behind the database's back
        conn.execute("INSERT INTO cache (cache_key, run_id) VALUES (?, ?)", (key, record.run_id))
    found = _ask(db, other, step)
    assert not found.hit and [i.reason for i in found.ignored] == [IgnoredReason.STALE]


# ------------------------------------------------------------------------------------- resume
def test_resume_plan_of_a_cancelled_run_replays_its_completed_iterations(db, tmp_path):
    checks = []
    outcome = solve(RASTRIGIN, planner=RuleBasedPlanner(), cancel=lambda: checks.append(1) or len(checks) > 2)
    assert outcome.status is RunStatus.CANCELLED and len(outcome.trace.iterations) == 2
    assert record_run(db, saved(outcome, tmp_path / "run")).integrity is Integrity.CHECKED
    with ExperimentDB(tmp_path / "db2") as other:
        record = record_run(other, tmp_path / "run", verify="replay")
    assert record.integrity is Integrity.REPLAYED  # replay cancels where the run was cancelled

    plan = resume_plan(db, record.run_id)
    assert plan.resumable and plan.stop_reason is StopReason.CANCELLED
    assert plan.proposals == tuple(r.proposal for r in outcome.trace.iterations)
    limits, used = outcome.trace.limits, outcome.trace.usage
    assert plan.used == used
    assert plan.remaining.max_iterations == limits.max_iterations - 2
    assert plan.remaining.max_evaluations == limits.max_evaluations - used.evaluations
    assert plan.remaining.max_tokens == limits.max_tokens
    assert plan.problem.digest() == RASTRIGIN.digest()
    # The runtime can pick the run up: the recorded proposals reproduce the completed iterations.
    again = solve(plan.problem, planner=RecordedPlanner(plan.proposals), budget=Budget(**vars(plan.limits)),
                  policy=plan.policy, seed=plan.seed)
    assert [without_clock(r.to_dict()) for r in again.trace.iterations[:2]] == \
        [without_clock(r.to_dict()) for r in outcome.trace.iterations]
    assert json.loads(json.dumps(plan.to_dict()))["remaining"]["max_iterations"] == limits.max_iterations - 2
    assert "max_evaluations" in plan.to_text()


def test_resume_plan_of_a_budget_exhausted_run(db, tmp_path):
    outcome = solve(CQP, planner=RuleBasedPlanner(), budget=Budget(max_evaluations=30))
    assert (outcome.status, outcome.stop_reason) == (RunStatus.BUDGET_EXHAUSTED, StopReason.MAX_EVALUATIONS)
    record = record_run(db, saved(outcome, tmp_path / "run"))
    plan = resume_plan(db, record.run_id)
    assert plan.resumable and plan.limits.max_evaluations == 30
    assert plan.remaining.max_evaluations == 30 - outcome.trace.usage.evaluations


def test_resume_plan_of_a_token_limited_llm_run_records_the_provider(db, tmp_path):
    planner = LLMPlanner(MockProvider([json.dumps({"action": "solve", "solver_id": "scipy.slsqp"}),
                                       json.dumps({"action": "stop"})]))
    outcome = solve(CQP, planner=planner, budget=Budget(max_tokens=1))
    assert outcome.stop_reason is StopReason.MAX_TOKENS and outcome.trace.usage.tokens > 1
    record = record_run(db, saved(outcome, tmp_path / "run"), provider="mock", model="mock")
    assert (record.provider, record.model, record.planner) == ("mock", "mock", "llm")
    assert record.usage.tokens == outcome.trace.usage.tokens
    plan = resume_plan(db, record.run_id)
    assert plan.resumable and plan.remaining.max_tokens == 0 and len(plan.proposals) == 1


def test_a_completed_run_is_not_resumable_and_a_resumed_run_links_back(db, tmp_path):
    _, first = _recorded(db, tmp_path)
    assert not resume_plan(db, first.run_id).resumable
    _, child = _recorded(db, tmp_path, seed=1)
    outcome = solve(CQP, planner=RuleBasedPlanner(), seed=2)
    linked = record_run(db, saved(outcome, tmp_path / "linked"), parent_id=first.run_id, resumed_from=child.run_id)
    assert (linked.parent_id, linked.resumed_from) == (first.run_id, child.run_id)
    outcome = solve(CQP, planner=RuleBasedPlanner(), seed=3)
    with pytest.raises(UnknownRunError):
        record_run(db, saved(outcome, tmp_path / "orphan"), parent_id="sha256:" + "1" * 64)
    with pytest.raises(UnknownRunError):
        resume_plan(db, "sha256:" + "2" * 64)


# ------------------------------------------------------------------------------------- query
def test_filters_sorting_and_pagination(db, tmp_path):
    records = [_recorded(db, tmp_path, seed=s)[1] for s in range(3)]
    _, unverified = _recorded(db, tmp_path, planner=forced({"action": "solve", "solver_id": "scipy.differential_evolution",
                                                           "evaluations": 40, "seed": 3}))
    _, lp = _recorded(db, tmp_path, problem=WYNDOR)

    assert query(db).total == 5
    assert {r.run_id for r in query(db, problem_digest=CQP.digest()).records} == {*(r.run_id for r in records),
                                                                                 unverified.run_id}
    assert [r.run_id for r in query(db, verified=False).records] == [unverified.run_id]
    assert query(db, status="unverified_best").records == query(db, verified=False).records
    assert query(db, seed=1).records == (records[1],)
    assert query(db, family="linear_program").records == (lp,)
    solvers = {s.solver_id for s in lp.steps}
    assert all(r.run_id != lp.run_id for r in query(db, solver_id="nope.none").records)
    assert lp in query(db, solver_id=sorted(solvers)[0]).records

    by_evals = query(db, order_by="evaluations", descending=False).records
    assert [r.usage.evaluations for r in by_evals] == sorted(r.usage.evaluations for r in by_evals)
    pages = [query(db, order_by="created_at", descending=False, limit=2, offset=o).records for o in (0, 2, 4)]
    assert [len(p) for p in pages] == [2, 2, 1]
    assert [r.run_id for p in pages for r in p] == [r.run_id for r in (*records, unverified, lp)]

    first, last = records[0].created_at, lp.created_at
    assert query(db, since=first, until=last).total == 5
    assert query(db, since=lp.created_at).records == (lp,)
    assert query(db, until=records[0].created_at).records == (records[0],)


@pytest.mark.parametrize("bad", [dict(order_by="objective; DROP TABLE experiments"), dict(status="verified' OR 1=1"),
                                 dict(limit=0), dict(limit=10_000), dict(offset=-1), dict(since="yesterday")])
def test_bad_query_arguments_raise(db, bad):
    with pytest.raises((ValueError, TypeError)):
        query(db, **bad)


# ------------------------------------------------------------------------------------- compare and aggregate
def test_compare_against_hand_computed_gaps(db, tmp_path):
    best_cqp = _recorded(db, tmp_path)
    weak_cqp = _recorded(db, tmp_path, planner=forced({"action": "solve", "solver_id": "scipy.differential_evolution",
                                                       "evaluations": 40, "seed": 3}))
    best_lp = _recorded(db, tmp_path, problem=WYNDOR)
    weak_lp = _recorded(db, tmp_path, problem=WYNDOR, planner=forced(
        {"action": "solve", "solver_id": "scipy.slsqp", "x0": {"x0": 0.0, "x1": 0.0}, "max_iterations": 1}))
    infeasible_lp = _recorded(db, tmp_path, problem=WYNDOR, planner=forced(
        {"action": "solve", "solver_id": "scipy.differential_evolution", "evaluations": 40, "seed": 3}))
    runs = [best_cqp, weak_cqp, best_lp, weak_lp, infeasible_lp]
    f = [o.result.objective_values["f0"] for o, _ in runs]
    assert infeasible_lp[0].status is RunStatus.NO_FEASIBLE and f[4] > f[2]  # better on paper, but infeasible

    rows = compare(db, [r.run_id for _, r in runs]).rows
    expected_gap = [0.0, f[1] - f[0], 0.0, f[2] - f[3], f[2] - f[4]]  # minimise CQP; maximise Wyndor
    assert [r.gap for r in rows] == pytest.approx(expected_gap, abs=1e-12)
    assert [r.best for r in rows] == pytest.approx([f[0], f[0], f[2], f[2], f[2]])
    assert rows[1].relative_gap == pytest.approx((f[1] - f[0]) / 1.0)
    assert rows[3].relative_gap == pytest.approx((f[2] - f[3]) / abs(f[2]))
    assert [r.evaluations for r in rows] == [o.trace.usage.evaluations for o, _ in runs]
    assert [r.tokens for r in rows] == [0] * 5
    assert [r.statuses for r in rows] == [o.statuses for o, _ in runs]
    table = compare(db, [best_cqp[1].run_id, weak_cqp[1].run_id]).to_text()
    assert table.splitlines()[0].split()[:4] == ["run_id", "problem", "status", "objective"]
    assert json.loads(json.dumps(compare(db, [best_lp[1].run_id]).to_dict()))["rows"][0]["gap"] == 0.0


def test_aggregate_by_family(db, tmp_path):
    runs = [_recorded(db, tmp_path, seed=s) for s in range(2)]
    runs.append(_recorded(db, tmp_path, planner=forced({"action": "solve", "solver_id": "scipy.differential_evolution",
                                                        "evaluations": 40, "seed": 3})))
    runs.append(_recorded(db, tmp_path, problem=WYNDOR))
    families = {f.family: f for f in aggregate(db).families}
    cqp = families["constrained_quadratic"]
    assert (cqp.runs, cqp.verified, cqp.unverified_best, cqp.other) == (3, 2, 1, 0)
    assert cqp.median_evaluations == statistics.median(o.trace.usage.evaluations for o, _ in runs[:3])
    lp = families["linear_program"]
    assert (lp.runs, lp.verified, lp.median_evaluations) == (1, 1, runs[3][0].trace.usage.evaluations)
    assert "constrained_quadratic" in aggregate(db).to_text()


# ------------------------------------------------------------------------------------- the index
def test_a_v0_database_migrates_to_the_current_version(tmp_path):
    root = tmp_path / "db"
    root.mkdir()
    with sqlite3.connect(root / "index.sqlite") as conn:
        conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
        conn.execute("INSERT INTO schema_version (version) VALUES (0)")
    with ExperimentDB(root) as db:
        assert db.version == SCHEMA_VERSION
        save_run(solve(CQP, planner=RuleBasedPlanner()), tmp_path / "run")
        record_run(db, tmp_path / "run")
        assert count(db) == 1


def test_a_version_1_record_of_a_cancelled_run_migrates_to_checked(tmp_path):
    root = tmp_path / "db"
    with ExperimentDB(root, migrations=MIGRATIONS[:1]) as db:  # a database written before version 2
        checks = []
        outcome = solve(RASTRIGIN, planner=RuleBasedPlanner(), cancel=lambda: checks.append(1) or len(checks) > 2)
        record = record_run(db, saved(outcome, tmp_path / "run"))
    with sqlite3.connect(root / "index.sqlite") as conn:  # what the version 1 code stored for it
        data = {**record.to_dict(), "integrity": "iterations_replayed"}
        conn.execute("UPDATE experiments SET record = ? WHERE run_id = ?", (json.dumps(data), record.run_id))
    with ExperimentDB(root) as db:
        assert db.version == SCHEMA_VERSION == 2
        assert db.get(record.run_id) == record and record.integrity is Integrity.CHECKED


def test_the_migration_hook_runs_new_steps_once_and_rolls_back_a_failing_one(tmp_path):
    root = tmp_path / "db"
    with ExperimentDB(root) as db:
        save_run(solve(CQP, planner=RuleBasedPlanner()), tmp_path / "run")
        record = record_run(db, tmp_path / "run")

    def broken(conn):
        conn.execute("CREATE TABLE half_done (x)")
        raise RuntimeError("migration failed")

    with pytest.raises(RuntimeError):
        ExperimentDB(root, migrations=(*MIGRATIONS, broken))
    with ExperimentDB(root) as db:
        assert db.version == SCHEMA_VERSION
        assert db.rows("SELECT name FROM sqlite_master WHERE name = 'half_done'") == []

    def add_note(conn):
        conn.execute("ALTER TABLE experiments ADD COLUMN note TEXT")

    with ExperimentDB(root, migrations=(*MIGRATIONS, add_note)) as db:
        assert db.version == SCHEMA_VERSION + 1 and db.get(record.run_id) == record
    with ExperimentDB(root, migrations=(*MIGRATIONS, add_note)) as db:  # already applied: nothing runs again
        assert db.version == SCHEMA_VERSION + 1
    with pytest.raises(SchemaVersionError):
        ExperimentDB(root)  # this database is newer than the default steps


def test_a_foreign_sqlite_file_is_not_taken_over(tmp_path):
    (tmp_path / "db").mkdir()
    with sqlite3.connect(tmp_path / "db/index.sqlite") as conn:
        conn.execute("CREATE TABLE something_else (x)")
    with pytest.raises(SchemaVersionError):
        ExperimentDB(tmp_path / "db")


def _record_in_process(args):
    root, run_dirs, barrier = args
    barrier.wait()
    with ExperimentDB(root) as db:
        return [record_run(db, run).run_id for run in run_dirs]


def _open_in_process(args):
    root, barrier = args
    barrier.wait()
    with ExperimentDB(root) as db:
        return db.version


def test_processes_that_create_one_database_at_once_all_open_it(tmp_path):
    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager, context.Pool(6) as pool:
        for k in range(4):
            barrier = manager.Barrier(6)
            assert pool.map(_open_in_process, [(tmp_path / f"db-{k}", barrier)] * 6) == [SCHEMA_VERSION] * 6


def test_concurrent_writers_do_not_corrupt_the_database(tmp_path):
    own = [saved(solve(CQP, planner=RuleBasedPlanner(), seed=s), tmp_path / f"run-{s}") for s in range(4)]
    shared = saved(solve(WYNDOR, planner=RuleBasedPlanner()), tmp_path / "shared")
    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager:
        barrier = manager.Barrier(4)
        with context.Pool(4) as pool:
            results = pool.map(_record_in_process, [(tmp_path / "db", [run, shared], barrier) for run in own])
    with ExperimentDB(tmp_path / "db") as db:
        assert count(db) == 5
        assert len({ids[1] for ids in results}) == 1  # every writer saw the one shared record
        for ids in results:
            for run_id in ids:
                record = db.get(run_id)
                assert all(db.store.has(d) for d in record.files.values())
        assert db.rows("PRAGMA integrity_check") == [("ok",)]


# ------------------------------------------------------------------------------------- hostile input
HOSTILE = ["sha256:' OR '1'='1", "x'; DROP TABLE experiments; --", "../../etc/passwd",
           "sha256:" + "a" * 63 + "'", "sha256:" + "A" * 64, "", "sha256:" + "a" * 64 + "\n"]


@pytest.mark.parametrize("hostile", HOSTILE)
def test_hostile_ids_are_refused_and_change_nothing(db, tmp_path, hostile):
    _, record = _recorded(db, tmp_path)
    for call in (lambda: db.get(hostile), lambda: resume_plan(db, hostile), lambda: compare(db, [hostile]),
                 lambda: export_run(db, hostile, tmp_path / "out"), lambda: db.store.get(hostile)):
        with pytest.raises(InvalidIdError):
            call()
    with pytest.raises(InvalidIdError):
        record_run(db, tmp_path / "run-0", parent_id=hostile)
    assert count(db) == 1 and db.get(record.run_id) == record


@pytest.mark.parametrize("hostile", HOSTILE + ["%", "_", "' OR 1=1 --"])
def test_hostile_filter_values_match_nothing_and_change_nothing(db, tmp_path, hostile):
    _recorded(db, tmp_path)
    for field in ("problem_digest", "family", "solver_id"):
        assert query(db, **{field: hostile}).total == 0
    assert count(db) == 1


@pytest.mark.parametrize("name", ["../x.json", "/etc/passwd", "a/../../b.json", "a\\..\\b.json", "C:/x.json",
                                  "a//b.json", "./a.json", "", "a/", "results/..", "\u2025/x.json"])
def test_file_names_that_leave_their_directory_are_refused(name):
    with pytest.raises(UnsafePathError):
        safe_relative(name)


def test_the_store_holds_json_only_and_checks_every_read(tmp_path):
    store = ArtifactStore(tmp_path)
    with pytest.raises(ArtifactError):
        store.put(pickle.dumps({"x": 1}))
    with pytest.raises(ArtifactError):
        store.put(b'{"x": NaN}')
    digest = store.put(b'{"__reduce__": "os.system", "x": 1}')
    assert store.put(b'{"__reduce__": "os.system", "x": 1}') == digest
    assert store.get_json(digest) == {"__reduce__": "os.system", "x": 1}  # data, never code
    with pytest.raises(ArtifactError):
        store.get("sha256:" + "0" * 64)


# ------------------------------------------------------------------------------------- the record schema
def test_records_validate_against_the_schema_and_malformed_ones_are_refused(db, tmp_path):
    _, record = _recorded(db, tmp_path)
    data = record.to_dict()
    for broken in ({**data, "extra": 1}, {**data, "status": "great"}, {**data, "files": {"../x.json": data["run_id"]}},
                   {**data, "usage": {**data["usage"], "evaluations": -1}}, {**data, "run_id": "1 OR 1"},
                   {k: v for k, v in data.items() if k != "statuses"}):
        with pytest.raises(MalformedRecordError):
            ExperimentRecord.from_dict(broken)


def test_importing_the_package_pulls_in_no_llm_sdk():
    code = ("import sys, orbitopt.agent.experiments; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in "
            "{'openai', 'anthropic', 'google', 'ollama', 'httpx', 'requests', 'litellm', 'transformers'}))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={"PYTHONPATH": str(REPO / "src"), "PATH": ""})
    assert out.stdout.strip() == "[]"
