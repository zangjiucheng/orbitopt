"""A limit that cuts any stage of a run short is a typed fact in the trace, whatever the final status.

The first E14-10 campaign found four ``rosenbrock-disk-d2`` cells that used 4000 of 4000 evaluations and ended
``verified`` with stop reason ``planner_stop``: the final verification's alternate stopped at its allocation, and
nothing in the run said so. These tests pin that case and the rule behind it: ``outcome.budget`` names the stage and
the resource, ``verified`` still needs the guard's pass on the evidence the cut stage produced, and a run that is not
verified after a cut is ``budget_exhausted``.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCH))

import bench_e14_agent_iteration as ERULE  # noqa: E402
import e14_common as C  # noqa: E402

from orbitopt.agent.experiments import ExperimentDB, ExperimentRecord, record_run  # noqa: E402
from orbitopt.agent.runtime import (BudgetEffect, BudgetEvent, BudgetResource, BudgetStage, Proposal,  # noqa: E402
                                    RecordedPlanner, RuleBasedPlanner, RunStatus, check_run, load_trace, replay,
                                    save_run, solve, trace_digest, upgrade_trace)
from orbitopt.optimization import (Budget, Constraint, Objective, OptimizationProblem, OptimizationResult,  # noqa: E402
                                   Termination, Variable)
from orbitopt.optimization.solvers import SlsqpSolver, SolverRegistry  # noqa: E402
from orbitopt.optimization.verification import canonical_json, file_digest  # noqa: E402

WORKED = OptimizationProblem(
    id="worked", variables=(Variable("x", lower=0.0, upper=10.0), Variable("y", lower=0.0, upper=10.0)),
    objectives=(Objective("distance_sq", "x**2 + y**2"),),
    constraints=(Constraint("halfplane", "x + y", ">=", 1.0),))


class Exhausting:
    """An alternate that spends its whole allocation and stops on it, answering with a fixed point."""

    def __init__(self, point: dict[str, float]):
        self.point = point

    @property
    def capability(self):
        return replace(SlsqpSolver.CAPABILITY, solver_id="test.exhausting", options={})

    def solve(self, problem, *, budget=None, seed=0, x0=None, options=None, tools=None, max_backend_iterations=None):
        for k in range(budget.max_evaluations):
            ev = problem.evaluate({"x": self.point["x"] + 1e-9 * k, "y": self.point["y"]}, tools)
        ev = problem.evaluate(self.point, tools) if budget.max_evaluations == 0 else ev
        return OptimizationResult(
            problem_digest=problem.digest(), solver_id="test.exhausting", termination=Termination.MAX_EVALUATIONS,
            solver_converged=False, constraints_satisfied=problem.constraints_satisfied(self.point, ev.residuals),
            x={"x": self.point["x"] + 1e-9 * (budget.max_evaluations - 1), "y": self.point["y"]},
            objective_values=dict(ev.objectives), constraint_residuals=dict(ev.residuals),
            evaluations=budget.max_evaluations)


def run_with_alternate(point):
    registry = SolverRegistry([SlsqpSolver(), Exhausting(point)])
    planner = RecordedPlanner([Proposal({"action": "solve", "solver_id": "scipy.slsqp", "evaluations": 200}),
                               Proposal({"action": "stop"})])
    return solve(WORKED, planner=planner, budget=Budget(max_evaluations=400), registry=registry)


def alternate_events(outcome):
    return [e for e in outcome.budget if e.stage is BudgetStage.ALTERNATE]


# ------------------------------------------------------------------------------------- the rule on a scripted run
def test_an_alternate_cut_by_its_allocation_is_typed_and_a_pass_on_its_evidence_stays_verified():
    outcome = run_with_alternate({"x": 0.5, "y": 0.5})
    assert outcome.status is RunStatus.VERIFIED  # the guard accepted the evidence the cut alternate produced
    (event,) = alternate_events(outcome)
    assert (event.resource, event.effect, event.label) == (BudgetResource.EVALUATIONS, BudgetEffect.STOPPED,
                                                         "verify-1-test.exhausting")
    run = outcome.trace.verifications[0].alternates[0]
    assert event.allowance == run.allocation == run.result.evaluations
    assert outcome.trace.to_dict()["outcome"]["budget"] == [e.to_dict() for e in outcome.budget]


def test_an_alternate_cut_short_without_a_pass_makes_the_run_budget_exhausted():
    outcome = run_with_alternate({"x": 0.6, "y": 0.4})  # feasible and worse: the evidence is inconclusive
    assert outcome.report.statuses["independently_verified"].outcome.value == "not_assessed"
    assert outcome.status is RunStatus.BUDGET_EXHAUSTED  # was unverified_best: the cut went unsaid
    assert alternate_events(outcome)
    assert outcome.stop_reason.value == "planner_stop"  # the loop itself was not stopped by a limit


def test_a_run_no_limit_touched_has_an_empty_budget_section():
    outcome = solve(WORKED, planner=RuleBasedPlanner())
    assert outcome.budget == ()
    assert outcome.trace.to_dict()["outcome"]["budget"] == []


def test_the_loop_stop_and_the_used_up_limit_are_typed():
    always = Proposal({"action": "solve", "solver_id": "scipy.slsqp", "evaluations": 20})
    outcome = solve(WORKED, planner=RecordedPlanner([always] * 5), budget=Budget(max_iterations=2))
    assert outcome.stop_reason.value == "max_iterations"
    stages = {(e.stage, e.resource) for e in outcome.budget}
    assert (BudgetStage.LOOP, BudgetResource.ITERATIONS) in stages
    assert (BudgetStage.RUN, BudgetResource.ITERATIONS) in stages


def test_alternates_skipped_for_lack_of_evaluations_are_typed():
    """30 evaluations leave too few for an alternate after the solve and the guard's two re-evaluations."""
    events = [BudgetEvent(BudgetStage.ALTERNATE, BudgetResource.EVALUATIONS, BudgetEffect.SKIPPED, None, 3)]
    registry = SolverRegistry([SlsqpSolver(), Exhausting({"x": 0.5, "y": 0.5})])
    planner = RecordedPlanner([Proposal({"action": "solve", "solver_id": "scipy.slsqp", "evaluations": 13}),
                               Proposal({"action": "stop"})])
    outcome = solve(WORKED, planner=planner, budget=Budget(max_evaluations=30), registry=registry)
    skipped = [e for e in outcome.budget if e.effect is BudgetEffect.SKIPPED]
    assert skipped == events
    assert outcome.trace.verifications[0].alternates == ()
    assert outcome.status is RunStatus.BUDGET_EXHAUSTED


# ------------------------------------------------------------------------------------- the campaign case
def test_rosenbrock_disk_d2_cells_say_where_the_verification_ran_out(tmp_path):
    """The E14-10 v1 cells (seeds 2, 3, 7, 8): the differential-evolution alternate stopped at its allocation, the
    run said ``planner_stop`` and ``verified``, and nothing typed the cut. In v1 they also used 4000 of 4000; the
    restoration reserve of the adapters now leaves a few evaluations unspent, but on seeds 3, 7 and 8 the
    alternate still stops at its allocation, so the case stays exercised."""
    inst = C.P.by_id("rosenbrock-disk-d2")
    cut_seeds = []
    for seed in (2, 3, 7, 8):
        row = ERULE.run_cell("E-rule", inst, seed, tmp_path / f"run-{seed}")
        trace = load_trace(tmp_path / f"run-{seed}")
        cut = [f"verify-{v['attempt']}-{a['solver_id']}" for v in trace["verifications"] for a in v["alternates"]
               if a["result"] and a["result"]["termination"] == "max_evaluations"]
        typed = [e["label"] for e in trace["outcome"]["budget"] if e["stage"] == "alternate"]
        assert typed == cut
        if row["n_evaluations_independent"] >= row["budget_evaluations"] or cut:
            assert row["budget_typed"] is True and row["stop"] == "budget_exhausted:evaluations"
        assert row["verification"]["run_status"] == "verified" and row["verification"]["verdict"] == "passed"
        assert row["replay"]["equal"]
        cut_seeds += [seed] if cut else []
    assert cut_seeds  # 3, 7 and 8 on the machines measured so far


# ------------------------------------------------------------------------------------- old traces
def _rewrite(run: Path, version: str) -> dict:
    """Rewrite a saved run's trace in an older schema and reseal the manifest."""
    trace = json.loads((run / "trace.json").read_text())
    trace["schemaVersion"] = version
    del trace["outcome"]["budget"]
    if version == "1.0":
        del trace["outcome"]["cancel"], trace["provider"], trace["model"], trace["orbitopt_version"]
    trace["digest"] = trace_digest(trace)
    (run / "trace.json").write_bytes(canonical_json(trace))
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["files"]["trace.json"] = file_digest(canonical_json(trace))
    manifest["trace_digest"] = trace["digest"]
    (run / "manifest.json").write_bytes(canonical_json(manifest))
    return trace


@pytest.mark.parametrize("version", ["1.0", "1.1"])
def test_an_old_trace_gets_the_budget_events_its_records_show(version, tmp_path):
    outcome = run_with_alternate({"x": 0.5, "y": 0.5})
    save_run(outcome, tmp_path)
    old = _rewrite(tmp_path, version)
    assert load_trace(tmp_path) == old
    upgraded = upgrade_trace(old)
    assert upgraded["schemaVersion"] == "1.2"
    assert upgraded["outcome"]["budget"] == [e.to_dict() for e in outcome.budget]
    assert check_run(tmp_path).verified
    registry = SolverRegistry([SlsqpSolver(), Exhausting({"x": 0.5, "y": 0.5})])
    report = replay(tmp_path, registry=registry)
    assert report.equal, (report.integrity, report.differences)


def test_the_studio_view_and_the_experiment_record_carry_the_budget(tmp_path):
    from orbitopt.viz.optimize.views import report_document, run_view_from_trace

    outcome = run_with_alternate({"x": 0.6, "y": 0.4})
    save_run(outcome, tmp_path / "run")
    expected = [e.to_dict() for e in outcome.budget]
    view = run_view_from_trace(load_trace(tmp_path / "run"), objective_name="distance_sq", sense="minimize")
    assert [dict(e) for e in view.budget] == expected
    assert report_document(view, problem=WORKED.to_dict(), meta={})["budget_events"] == expected
    registry = SolverRegistry([SlsqpSolver(), Exhausting({"x": 0.6, "y": 0.4})])
    with ExperimentDB(tmp_path / "db") as db:
        record = record_run(db, tmp_path / "run", registry=registry)
        assert [e.to_dict() for e in record.budget] == expected
        assert db.get(record.run_id).budget == record.budget


def test_a_stored_record_without_a_budget_still_parses(tmp_path):
    outcome = run_with_alternate({"x": 0.6, "y": 0.4})
    save_run(outcome, tmp_path / "run")
    registry = SolverRegistry([SlsqpSolver(), Exhausting({"x": 0.6, "y": 0.4})])
    with ExperimentDB(tmp_path / "db") as db:
        data = record_run(db, tmp_path / "run", registry=registry).to_dict()
    del data["budget"]
    assert ExperimentRecord.from_dict(data).budget == ()


def test_the_cli_solve_summary_lists_the_budget(tmp_path, capsys):
    from orbitopt.cli import main

    problem = tmp_path / "worked.json"
    problem.write_text(json.dumps(WORKED.to_dict()))
    with pytest.raises(SystemExit):
        main(["agent", "solve", str(problem), "--output", str(tmp_path / "run"), "--max-iterations", "1",
              "--max-evaluations", "4000", "--json"])
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "verified"  # the loop stopped on its limit; verification then passed
    assert {(e["stage"], e["resource"]) for e in summary["budget"]} >= {("loop", "iterations"), ("run", "iterations")}
    with pytest.raises(SystemExit):
        main(["agent", "solve", str(problem), "--output", str(tmp_path / "run2"), "--max-iterations", "1",
              "--max-evaluations", "4000"])
    assert "budget       the loop stopped on its iterations limit" in capsys.readouterr().out
