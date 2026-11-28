"""E14-10 arm E on the reference problem: the E14 agent runtime with its verification guard.

``E-rule`` runs ``orbitopt.agent.runtime.solve`` with the deterministic ``RuleBasedPlanner``: no model, no tokens.
It is NOT the pre-registered arm E (which compiles the natural-language statement and plans with an LLM), so
E-rule against A or B says nothing about what an LLM adds. ``E-LLM`` is the same loop with ``LLMPlanner`` on a
provider from the LLM_* variables (``e14_common.provider_from_env``); without one it is not run.

Every cell is saved with ``save_run`` and replayed with ``replay`` right away; the row records whether the replay
was EQUAL. Evaluations are counted twice: by the runtime's own meter (the trace) and by an independent counter that
wraps ``OptimizationProblem.evaluate`` for the duration of the solve.

    PYTHONPATH=src python benchmarks/bench_e14_agent_iteration.py --instance rastrigin-d5 --seed 0 --run-dir /tmp/r
"""
from __future__ import annotations

import argparse
import contextlib
import json
import tempfile
import time
from pathlib import Path

import e14_common as C


@contextlib.contextmanager
def independent_count():
    """Count every ``OptimizationProblem.evaluate`` call in this process, metered copies included."""
    from orbitopt.optimization.problem import OptimizationProblem
    original = OptimizationProblem.evaluate
    box = [0]

    def counted(self, x, tools=None):
        box[0] += 1
        return original(self, x, tools)

    OptimizationProblem.evaluate = counted
    try:
        yield box
    finally:
        OptimizationProblem.evaluate = original


def thin_front(result, limit: int) -> list[dict]:
    """At most ``limit`` members, evenly spaced along the first objective of the runtime's own values."""
    members = sorted(result.front, key=lambda p: tuple(p.objective_values.values()))
    if len(members) <= limit:
        return [dict(p.x) for p in members]
    step = (len(members) - 1) / (limit - 1)
    return [dict(members[round(k * step)].x) for k in range(limit)]


def _stop(outcome) -> str:
    """Why the cell ended: a budget cut anywhere but one planner solve is ``budget_exhausted:<resource>``, the
    first such event's resource, whatever the run status; otherwise the loop's own stop."""
    cut = [e for e in outcome.budget if e.run_level]
    if cut:
        return f"budget_exhausted:{cut[0].resource.value}"
    reason = outcome.stop_reason
    if reason.value == "planner_stop":
        return "completed"
    return {"refused": "refused", "planner_failed": "error"}.get(reason.value, reason.value)


def planner_for(arm: str, provider=None):
    from orbitopt.agent.runtime import LLMPlanner, RuleBasedPlanner
    return RuleBasedPlanner(max_restarts=3) if arm == "E-rule" else LLMPlanner(provider)


def run_cell(arm: str, inst, seed: int, run_dir: Path, *, provider=None, mode: str = "scored", problem=None,
             tools=None) -> dict:
    """One cell of E-rule or E-LLM: solve, save, replay, score the returned x with the catalogue callables.

    ``problem`` defaults to the reference IR; E-LLM passes its compiled problem (and its tool view), whose x is
    mapped back to the reference by variable order."""
    from orbitopt.agent.runtime import replay, save_run, solve
    from orbitopt.optimization import Budget
    if problem is None:
        problem, tools = C.instance_ir(inst), C.tools_for(inst)
    budget = C.A.BUDGET.evaluations(inst.dimension)
    run_dir = Path(run_dir)
    planner = planner_for(arm, provider)
    limits = Budget(max_evaluations=budget, max_wall_s=C.A.BUDGET.wall_clock_s, max_tokens=C.A.BUDGET.llm_tokens_max,
                    max_iterations=C.A.BUDGET.llm_calls_max)
    started = time.perf_counter()
    with independent_count() as counter:
        outcome = solve(problem, planner=planner, budget=limits, tools=tools, seed=seed)
    wall = time.perf_counter() - started
    manifest = save_run(outcome, run_dir)
    t_replay = time.perf_counter()
    rep = replay(run_dir, tools=tools, expected_manifest_digest=manifest)
    replay_wall = time.perf_counter() - t_replay

    result = outcome.result
    names = [v.name for v in problem.variables]
    x = front = None
    if result is not None and result.x:
        x = [float(result.x[n]) for n in names]
    if result is not None and len(inst.objectives) > 1:
        front = [[float(p[n]) for n in names] for p in thin_front(result, C.P.MO_MAX_POINTS)]
    sc = C.score(inst, x=x, front=front)
    trace = outcome.trace.to_dict()
    usage = trace["usage"]
    stop = _stop(outcome)
    formulations = sorted({r.problem_digest for r in outcome.results.values()})
    hard = len(problem.constraints) + len(problem.variables)
    preserved = formulations in ([], [problem.digest()])
    row = C.base_row(arm, inst, seed, mode)
    row.update({
        "compile_outcome": "reference", "stop": stop, "error": None,
        "solvers": [it["decision"]["solver_id"] for it in trace["iterations"]
                    if it.get("decision") and it["decision"].get("solver_id")],
        "status": C.outcome_status(sc, stop, x is not None or bool(front)),
        "x": x, "front": front,
        **{k: sc[k] for k in ("objective", "hypervolume", "max_violation", "gap", "gap_norm", "feasible", "success",
                              "alarm")},
        "n_evaluations": usage["evaluations"], "n_evaluations_independent": counter[0],
        "within_budget": (counter[0] <= budget and usage["evaluations"] == counter[0]
                          and wall <= C.A.BUDGET.wall_clock_s and usage["tokens"] <= C.A.BUDGET.llm_tokens_max),
        "budget_typed": stop.startswith("budget_exhausted") if counter[0] >= budget else True,
        "budget_events": [e.to_dict() for e in outcome.budget],
        "wall_clock_s": wall, "tokens_in": usage["tokens"] if arm == "E-LLM" else 0,
        "iterations": usage["iterations"],
        "verification": {"verdict": None if outcome.report is None else outcome.report.verdict.value,
                         "run_status": outcome.status.value, "statuses": outcome.statuses,
                         "codes": [] if outcome.report is None else sorted(c.value for c in outcome.report.codes)},
        "hard_constraints": {"items": hard, "formulations": len(formulations) or 1,
                             "preserved": hard * (len(formulations) or 1) if preserved else 0,
                             "fingerprint_equal": preserved},
        "artifact_digest": manifest, "trace_digest": trace["digest"],
        "messages": [],
    })
    row["row_digest"] = C.row_digest(row)
    row["replay"] = {"kind": "orbitopt.agent.runtime.replay", "equal": bool(rep.equal),
                     "differences": [str(d) for d in list(rep.differences)[:5]],
                     "integrity": [str(i) for i in list(rep.integrity)[:5]]}
    row["replay_wall_clock_s"] = replay_wall
    return row


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--instance", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    run_dir = args.run_dir or Path(tempfile.mkdtemp(prefix="e14-erule-"))
    row = run_cell("E-rule", C.P.by_id(args.instance), args.seed, run_dir)
    print(json.dumps(C.jsonable(row), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
