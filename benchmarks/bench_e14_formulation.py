"""E14-10 compiler arms: arm D (LLM problem compiler + the fixed solver) and the formulation-accuracy suite.

Both need a real model. With no provider configured (the LLM_* variables, ``e14_llm.context_from_env``) every
cell is recorded as ``not_run: no provider``; nothing is guessed. Each compile runs under arm D's per-cell call
cap and token budget, and every call is recorded in a cassette under the run directory. ``--mode pipeline-smoke`` runs the suite on the scripted replies of
``tests/data/agent/formulation_cases.json`` through a ``MockProvider``: that exercises the pipeline only, says
nothing about any model, and every row it writes is labelled ``pipeline-smoke`` and kept out of every statistic.

    PYTHONPATH=src python benchmarks/bench_e14_formulation.py --mode pipeline-smoke -o /tmp/formulation.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import e14_common as C
import e14_llm as L

CASES_PATH = C.REPO / "tests" / "data" / "agent" / "formulation_cases.json"


def builtin_tools():
    from orbitopt.agent.tools import builtin_registry
    registry = builtin_registry()
    return registry.view([s.tool_id for s in registry.list()])


def resolvers(problem, view):
    from orbitopt.agent.tools import Caller, bind_problem
    return bind_problem(problem, view, Caller("e14-campaign"))


def compile_statement(statement: str, provider, seed: int, problem_id: str):
    from orbitopt.agent.compiler import compile_problem
    view = builtin_tools()
    return compile_problem(statement, provider, known_tool_ids=tuple(sorted(view.allowed)), problem_id=problem_id,
                           seed=seed), view


@dataclass(frozen=True)
class _Con:
    name: str
    kind: str
    fn: Callable


@dataclass(frozen=True)
class CompiledAdapter:
    """A compiled IR problem in the catalogue's vector form, so the arm-A solver runs on it unchanged.

    Variables keep their declared order; objectives are turned to minimisation; constraints are the IR residuals
    (``g <= 0``, ``h == 0``)."""
    id: str
    dimension: int
    bounds: tuple
    variable_type: str
    sense: str
    objectives: tuple
    constraints: tuple
    tolerance: object
    blackboxes: dict = field(default_factory=dict)
    objective_exprs: tuple = ("",)


def adapt(problem, tools) -> CompiledAdapter:
    names = [v.name for v in problem.variables]
    kinds = {v.kind for v in problem.variables}
    bounds = []
    for v in problem.variables:
        if v.kind == "binary":
            bounds.append((0.0, 1.0))
        elif v.kind == "discrete" or v.lower is None or v.upper is None:
            raise ValueError(f"variable {v.name}: the fixed solver needs finite bounds")
        else:
            bounds.append((float(v.lower), float(v.upper)))
    cache: dict = {}

    def ev(x):
        key = tuple(float(a) for a in x)
        if key not in cache:
            cache.clear()
            cache[key] = problem.evaluate(dict(zip(names, key)), tools)
        return cache[key]

    objectives = tuple((lambda x, o=o, s=(1.0 if o.sense == "minimize" else -1.0): s * ev(x).objectives[o.name])
                       for o in problem.objectives)
    constraints = tuple(_Con(c.name, "eq" if c.sense == "==" else "le", lambda x, n=c.name: ev(x).residuals[n])
                        for c in problem.constraints)
    vtype = "continuous" if kinds == {"continuous"} else "binary" if kinds == {"binary"} else "integer"
    tol = type("Tol", (), {"constraint_abs": problem.verification.constraint_tol})()
    return CompiledAdapter(problem.id, len(names), tuple(bounds), vtype, "minimize", objectives, constraints, tol)


def hard_constraints(compiled, inst) -> dict:
    """Probe-point equivalence of a compiled problem with the reference, variables matched by declared order."""
    from orbitopt.agent.equivalence import semantically_equivalent
    reference = C.instance_ir(inst)
    aliases = None
    if len(compiled.variables) == len(reference.variables):
        aliases = {a.name: b.name for a, b in zip(compiled.variables, reference.variables) if a.name != b.name}
    same = semantically_equivalent(compiled, reference, tools=C.tools_for(inst), aliases=aliases).equivalent
    items = len(reference.constraints) + len(reference.variables)
    return {"items": items, "formulations": 1, "preserved": items if same else 0, "fingerprint_equal": same}


def not_run_row(arm: str, inst, seed: int, reason: str = "no provider", mode: str = "scored") -> dict:
    row = C.base_row(arm, inst, seed, mode)
    row.update({k: v for k, v in C.score(inst).items() if k != "success"})
    row.update({"status": "not_run", "stop": "not_run", "not_run": reason, "compile_outcome": None, "success": None,
                "x": None, "front": None, "n_evaluations": 0, "n_evaluations_independent": 0, "within_budget": None,
                "budget_typed": None, "wall_clock_s": 0.0, "verification": None, "solvers": [], "messages": []})
    row["row_digest"] = C.row_digest(row)
    return row


def run_cell_d(inst, seed: int, provider, mode: str = "scored") -> dict:
    """Arm D: compile the statement, solve the compiled problem with the arm-A solver, score x on the reference.

    ``provider`` carries the cell's meter: the spec allows arm D 3 calls, so a compile whose repair rounds would
    need a fourth call stops there, typed ``budget_exhausted:llm_calls``."""
    import bench_e14_solver_selection as S
    from orbitopt.agent.providers import BudgetExceededError
    started = time.perf_counter()
    try:
        outcome, view = compile_statement(inst.statement, provider, seed, inst.id)
    except BudgetExceededError:
        outcome, view = None, None
    llm_wall = time.perf_counter() - started
    row = C.base_row("D", inst, seed, mode)
    compiled = "budget_exhausted" if outcome is None else outcome.outcome
    row.update({"compile_outcome": compiled, "compile_attempts": None if outcome is None else outcome.attempts,
                "schema_valid": None if outcome is None else outcome.outcome != "compile_failed",
                "llm_wall_clock_s": llm_wall, "solvers": [], "messages": [], **L.llm_fields(provider.meter)})
    x = None
    stop, error, evaluations = ("rejected" if outcome is not None else
                                f"budget_exhausted:llm_{provider.meter.cut}"), None, 0
    if compiled == "compile":
        try:
            adapter = adapt(outcome.problem, resolvers(outcome.problem, view))
            budget = C.A.BUDGET.evaluations(adapter.dimension)
            ev = C.Evaluator(adapter, budget, C.A.BUDGET.wall_clock_s - llm_wall)
            run = S.Run(ev)
            stop = "completed"
            try:
                S.arm_a(run, adapter, seed)
            except C.BudgetExhausted as exc:
                stop = f"budget_exhausted:{exc.kind}"
            x = run.x or ev.best_x()
            evaluations = ev.count
            row["solvers"] = run.solvers
        except Exception as exc:
            stop, error = "error", f"{type(exc).__name__}: {exc}"
    if x is not None and len(x) != inst.dimension:
        error, x = f"compiled dimension {len(x)} != reference {inst.dimension}", None
    if compiled == "compile":
        row["hard_constraints"] = hard_constraints(outcome.problem, inst)
    sc = C.score(inst, x=x)
    if row["llm_over_budget"]:
        sc = {**sc, "success": False}
    status = ("rejected" if stop == "rejected" else "budget_exhausted" if outcome is None
              else C.outcome_status(sc, stop, x is not None))
    row.update({"stop": stop, "error": error, "x": x, "front": None,
                **{k: sc[k] for k in ("objective", "hypervolume", "max_violation", "gap", "gap_norm", "feasible",
                                      "success", "alarm")},
                "status": status, "n_evaluations": evaluations, "n_evaluations_independent": evaluations,
                "within_budget": (evaluations <= C.A.BUDGET.evaluations(inst.dimension)
                                  and not row["llm_over_budget"]),
                "budget_typed": True, "wall_clock_s": time.perf_counter() - started, "verification": None})
    row["row_digest"] = C.row_digest(row)
    return row


# ------------------------------------------------------------------------------------- formulation-accuracy suite
def load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text())["cases"]


def variable_matching(compiled, reference, case_aliases: dict | None) -> tuple[dict | None, str]:
    """How a compiled problem's variables are matched to the reference's: the case's alias table when every
    alias names a compiled variable (the scripted replies), else declared order when the counts match (as arm D
    maps x back), else by name. A model names its variables freely, so matching by name alone would fail a
    correct formulation."""
    names = [v.name for v in compiled.variables]
    if case_aliases and set(case_aliases) <= set(names):
        return dict(case_aliases), "case aliases"
    if len(names) == len(reference.variables):
        return {a: b.name for a, b in zip(names, reference.variables) if a != b.name}, "declared order"
    return None, "name"


D_CALLS = C.A.ARMS[C.A.ARM_IDS.index("D")].llm_calls_max


def case_row(case: dict, seed: int, provider, mode: str) -> dict:
    """One formulation case: does the outcome match, and is a compiled problem equivalent to the reference IR?

    ``provider`` carries arm D's per-cell meter; a compile it cuts is not schema-valid and not correct."""
    from orbitopt.agent.equivalence import semantically_equivalent
    from orbitopt.agent.providers import BudgetExceededError
    from orbitopt.optimization import parse_problem
    started = time.perf_counter()
    try:
        outcome, _ = compile_statement(case["prompt"], provider, seed, case["id"])
    except BudgetExceededError:
        outcome = None
    llm = L.llm_fields(provider.meter)
    row = {"case_id": case["id"], "seed": seed, "mode": mode, "language": case.get("language"),
           "expected_outcome": case["expected_outcome"],
           "outcome": "budget_exhausted" if outcome is None else outcome.outcome,
           "schema_valid": outcome is not None and outcome.outcome != "compile_failed", "equivalent": None,
           "reasons": [], "tokens": llm["tokens_in"] + llm["tokens_out"], "calls": llm["llm_calls"],
           "compile_attempts": None if outcome is None else outcome.attempts, **llm,
           "wall_clock_s": time.perf_counter() - started}
    if outcome is not None and outcome.outcome == "compile" and case.get("reference_ir"):
        reference = parse_problem(case["reference_ir"])
        aliases, row["variable_matching"] = variable_matching(outcome.problem, reference, case.get("aliases"))
        report = semantically_equivalent(outcome.problem, reference, seed=seed, aliases=aliases)
        row["equivalent"], row["reasons"] = report.equivalent, list(report.reasons)[:5]
    row["correct"] = row["outcome"] == row["expected_outcome"] and (row["equivalent"] is not False)
    return row


def case_digest(row: dict) -> str:
    return C.digest(C.jsonable({k: v for k, v in row.items() if k not in ("wall_clock_s", "replay")}))


def run_suite(llm: L.LLMContext | None, seeds, mode: str, runs_dir: Path) -> dict:
    """Every formulation case at every seed. Live, each (case, seed) has its own cassette under ``runs_dir`` and
    is replayed from it right away; in pipeline-smoke mode the case's scripted replies stand in for a model."""
    import bench_e14_end_to_end as E2E
    from orbitopt.agent.providers import MockProvider
    cases = load_cases()
    if llm is None and mode != "pipeline-smoke":
        return {"metric": "formulation_accuracy", "mode": mode, "status": "not_run", "not_run": "no provider",
                "cases": len(cases), "seeds": list(seeds), "rows": []}
    runs_dir = Path(runs_dir)
    caps = (D_CALLS, C.A.BUDGET.llm_tokens_max)
    rows = []
    for case in cases:
        for seed in seeds:
            cassette = runs_dir / case["id"] / f"seed-{seed}.cassette.jsonl"
            context = llm
            if mode == "pipeline-smoke":
                context = L.LLMContext(MockProvider(list(case.get("replies", [])) or ["{}"]), None)
                cassette.unlink(missing_ok=True)
            cassette.parent.mkdir(parents=True, exist_ok=True)
            row = case_row(case, seed, context.cell(cassette, *caps), mode)
            row["transcript_digest"] = C.file_sha256(cassette) if cassette.exists() else None
            if cassette.exists():
                again = case_row(case, seed, E2E.replay_provider(cassette, caps), mode)
                again["transcript_digest"] = row["transcript_digest"]
                row["replay"] = {"kind": "re-execution from the recorded cassette",
                                 "equal": case_digest(again) == case_digest(row)}
            rows.append(row)
    n = len(rows)
    return {"metric": "formulation_accuracy", "mode": mode, "status": "run", "cases": len(cases),
            "seeds": list(seeds), "rows": rows,
            "formulation_accuracy": sum(r["correct"] for r in rows) / n if n else math.nan,
            "ir_schema_validity": sum(r["schema_valid"] for r in rows) / n if n else math.nan}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("scored", "pipeline-smoke"), default="scored")
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args(argv)
    seeds = [0] if args.mode == "pipeline-smoke" else list(C.M.SCORING_SEEDS)
    doc = run_suite(L.context_from_env(args.out.parent), seeds, args.mode, args.out.parent / "runs" / "formulation")
    doc.update(spec_digest=C.SPEC.spec_digest(), protocol_digest=C.protocol_digest())
    C.write_json_atomic(args.out, C.sealed(C.jsonable(doc)))
    print(json.dumps({k: v for k, v in doc.items() if k != "rows"}, indent=1, default=str))


if __name__ == "__main__":
    main()
