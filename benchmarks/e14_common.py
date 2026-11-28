"""Shared pieces of the E14-10 qualification campaign: spec loading, the protocol and its freeze, the catalogue -> IR
translation, scoring with the catalogue's own callables, the budgeted evaluator of the baseline arms, and sealed JSON.

The campaign CLI is ``benchmarks/bench_e14_end_to_end.py``. The arms live in ``bench_e14_solver_selection.py`` (A, B),
``bench_e14_agent_iteration.py`` (E-rule, E-LLM) and ``bench_e14_formulation.py`` (D and the formulation suite).
"""
from __future__ import annotations

import ast
import functools
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

import e14_llm as L

REPO = Path(__file__).resolve().parents[1]
SPEC_PKG = REPO / "benchmarks" / "e14_spec"
RESULTS_DIR = REPO / "benchmarks" / "results" / "e14-v3"
"""Campaign v3: v2's rule arms plus the provider arms. v1's results and freeze stay in ``benchmarks/results/e14``
and v2's in ``benchmarks/results/e14-v2``; neither is ever rewritten."""
FREEZE_PATH = RESULTS_DIR / "freeze.json"


def load_spec():
    """The pre-registered spec package, imported by path as the E14 tests do (``benchmarks`` is not a package)."""
    if "e14_spec" not in sys.modules:
        spec = importlib.util.spec_from_file_location("e14_spec", SPEC_PKG / "__init__.py",
                                                      submodule_search_locations=[str(SPEC_PKG)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules["e14_spec"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["e14_spec"]


SPEC = load_spec()
P, A, M, F = SPEC.problems, SPEC.arms, SPEC.metrics, SPEC.faults

#: arms this campaign runs without a model, and arms that need a provider. ``E-rule`` is arm E with the
#: deterministic RuleBasedPlanner instead of an LLM: it is NOT the pre-registered arm E.
RULE_ARMS = ("A", "B", "E-rule")
PROVIDER_ARMS = ("C", "D", "E-LLM")
ALL_ARMS = RULE_ARMS + PROVIDER_ARMS
TOOL_ID = "bench.quadrature_error"
SCHEMA = "orbitopt.e14-campaign.shard/1"
#: row fields that a replay does not compare (wall clock is not reproducible)
TIMING_FIELDS = ("wall_clock_s", "llm_wall_clock_s", "replay_wall_clock_s")

def _prompt_digests() -> dict:
    import hashlib as _h

    def d(text: str) -> str:
        return "sha256:" + _h.sha256(text.encode()).hexdigest()

    from orbitopt.agent.compiler.prompts import SYSTEM_PROMPT as COMPILER
    from orbitopt.agent.compiler.schema import COMPILER_OUTPUT_SCHEMA
    from orbitopt.agent.providers.openai_compat import json_instruction
    from orbitopt.agent.runtime.planners import PROPOSAL_SCHEMA, SYSTEM_PROMPT as PLANNER
    return {"compiler_system": d(COMPILER), "compiler_schema_instruction": d(json_instruction(COMPILER_OUTPUT_SCHEMA)),
            "planner_system": d(PLANNER), "planner_schema_instruction": d(json_instruction(PROPOSAL_SCHEMA))}


LLM_PROTOCOL = {
    "provider": {"kind": "deepseek", "model": "deepseek-flash", "base_url": "https://api.deepseek.com",
                 "path": "/chat/completions", "temperature": 0.0, "thinking": "disabled",
                 "structured_output": "response_format json_object, the JSON Schema in the system instruction, the "
                                      "reply validated locally with one repair turn (DeepSeek has no json_schema)",
                 "seed": "not sent (DeepSeek documents none); the cell seed stays in each request and its digest",
                 "retries": "429, 5xx, timeouts, connection errors and finish_reason insufficient_system_resource: "
                            "at most 3 retries, backoff 1 s doubling to 8 s or Retry-After up to 60 s; 401, 402, "
                            "403 stop the campaign; other 4xx fail the call",
                 "timeout_s": 120.0,
                 "recorded": "the model each response names, its usage (cache hit and miss), latency and attempts"},
    "reproduction": "DeepSeek's outputs may differ between identical calls even at temperature 0, so the cassette "
                    "of every cell, not a re-run, is the reproduction artifact; experiment_replay reruns each cell "
                    "from its cassette with no model call",
    "per_cell": {"calls": {arm: next(a.llm_calls_max for a in A.ARMS if a.id == arm[0])
                           for arm in ("C", "D", "E-LLM")},
                 "tokens": A.BUDGET.llm_tokens_max,
                 "counted": "every call made through the cell's provider: structured-output repairs, compiler "
                            "repair rounds and planner iterations; a call past a cap is refused before it is made; "
                            "the call whose tokens cross the budget makes the cell budget_exhausted and, for C, a "
                            "failure"},
    "arm_C": {"max_tokens_per_reply": 8192, "points_per_reply": "at most 50", "feedback": "objective values, worst "
              "violation and the budget left", "stop": "the model's stop flag, or a cap", "scored": "the best "
              "evaluated point, feasible first; for two objectives the last 50 points"},
    "arm_D": {"compiler": "compile_problem, max_tokens 4096, at most 3 rounds, inside D's 3 calls",
              "solver": "the arm-A differential evolution on the compiled problem", "x": "mapped to the reference "
              "by declared variable order"},
    "arm_E_LLM": {"compiler": "as arm D, sharing the cell's 20 calls", "planner": "LLMPlanner, max_reply_tokens "
                  "512, temperature 0", "runtime": "Budget(max_evaluations = spec, max_wall_s = 600, max_tokens = "
                  "what the compile left, max_iterations = 20)"},
    "formulation": {"cases": "tests/data/agent/formulation_cases.json (58: 50 English, 8 Chinese)",
                    "seeds": "the 10 scoring seeds", "caps": "arm D's per cell"},
    "price_table": {"source": L.PRICE_SOURCE, "usd_per_mtok": L.PRICES_USD_PER_MTOK,
                    "basis": "peak (list) price; off-peak is half"},
    "hard_stop": "campaign-wide calls and estimated USD in <out>/llm_ledger.jsonl, reserved at a worst-case cost "
                 "before every network call (--max-llm-calls, --max-llm-usd)",
    "prompt_digests": _prompt_digests(),
}
"""The provider arms' choices. A scored shard refuses a provider whose kind, model or base URL differ."""


#: Every choice the spec leaves to the runner, fixed before the first scored run and digested into the freeze.
PROTOCOL = {
    "version": "e14-10-campaign/3",
    "changes_from_v2": [
        "provider arms C, D and E-LLM and the formulation suite run on the provider in llm.provider; every rule-arm "
        "choice below is v2's, and v2's frozen results stay as recorded",
        "one provider configuration (the LLM_* variables, orbitopt.llm_env) replaces the campaign's own variables",
        "arm C: the proposer may set stop, is told the budget left after each reply, and a cell whose calls passed "
        "the token budget is a failure",
        "arms D and E-LLM: every HTTP-level call is metered against the spec's per-cell call cap (D 3, C and "
        "E-LLM 20, the E-LLM compile included) and the 200,000-token budget, refused before the call",
        "formulation suite: compiled variables are matched to the reference by the case's alias table when it "
        "names compiled variables, else by declared order (v2 matched by name only, which fails a correct "
        "formulation with its own names)",
    ],
    "changes_from_v1": [
        "runtime: every budget cut of a run is typed in the trace (agent-trace-1.2 outcome.budget); a run that is "
        "not verified after a cut of the loop, the final verification or the run is budget_exhausted",
        "solver adapters: clean points (no constraint past 1e-6 of its tolerance band) rank before band points, and "
        "the best band point is restored onto the constraints before it is reported; the runtime's incumbent "
        "ranks clean candidates first; the verification guard and every tolerance are unchanged",
        "scorer: a gap below minus the tolerance is an alarm and no success (spec section 2); v1 flagged it and "
        "counted it a success",
        "E-rule rows: stop is budget_exhausted:<resource> when the run's budget section has a cut of the loop, "
        "the final verification or the run, and budget_typed reads that stop as for A and B",
    ],
    "seeds": {"scoring": list(M.SCORING_SEEDS), "tuning": list(M.TUNING_SEEDS),
              "why": "the spec's own seeds (section 5): 10 scoring seeds, 5 tuning seeds"},
    "shard": "arm x family x every seed of the split; fault-matrix shards per family",
    "jobs": "one process per family runs that family's shards for every arm in turn (A, B, E-rule, C, D, E-LLM), so "
            "the arms of one instance share a host and its load; at most four such processes at once",
    "arm_A": {"solver": "scipy.optimize.differential_evolution", "rng": "cell seed", "polish": True,
              "popsize": 15, "tol": 0.01, "init": "latinhypercube",
              "maxiter": "floor(0.9 * budget / (popsize * d)) - 1, so the polish has about a tenth of the budget",
              "integrality": "from the variable type",
              "constraints": "NonlinearConstraint over the catalogue callables (le: g <= 0, eq: h == 0)",
              "multi_objective": "Tchebycheff scalarisation over 50 weights, each by the same DE with 1/50 of the "
                                 "budget; the two extreme weights first, the ideal point is the best value seen"},
    "arm_B": {"lp": "linprog(method='highs') on coefficients probed from the callables (d + 1 evaluations)",
              "milp": "milp (HiGHS) on probed coefficients, integrality 1 on every variable",
              "convex_qp": "minimize(method='trust-constr') from the box centre, 3-point finite differences, BFGS "
                           "Hessians, TRUST_CONSTR_OPTIONS (tuning_adjustments)",
              "nlp": "minimize(method='SLSQP') from seeded uniform starts until the evaluation budget is spent",
              "global": "the arm-A differential evolution",
              "black_box": "exhaustive enumeration of the integer grid sorted by the objective expression "
                           "(the stated cost, read without a black-box call), stopping at the first feasible "
                           "point; differential evolution with integrality when the grid exceeds the budget",
              "multi_objective": "Tchebycheff over 50 weights in epigraph form, each by the nlp rule with 1/50 of "
                                 "the budget"},
    "arm_E_rule": {"runtime": "orbitopt.agent.runtime.solve", "planner": "RuleBasedPlanner(max_restarts=3)",
                   "policy": "Policy() defaults", "registry": "default_registry()",
                   "verification": "IR default VerificationPolicy, constraint_tol = max(constraint_abs, 1e-15)",
                   "budget": "max_evaluations = spec budget, max_wall_s = 600; a cut of the loop, the final "
                             "verification or the run (outcome.budget) types the row's stop as "
                             "budget_exhausted:<resource>"},
    "tuning_adjustments": [
        {"arm": "B", "rule": "convex_qp", "instance": "cqp-worked-d2", "seeds": "deterministic (no seed)",
         "before": "trust-constr scipy defaults: gtol termination at gap 1.6e-4 after 27 evaluations",
         "after": "jac and constraint jac '3-point', hess BFGS, gtol = xtol = barrier_tol = 1e-12, "
                  "initial_barrier_parameter = 1e-6: gap 8.7e-13 after 85 evaluations"},
        {"arm": "B", "rule": "nlp", "instance": "rosenbrock-d2", "seeds": [1000],
         "before": "SLSQP default ftol 1e-6: five starts end at f = 7e-9 to 2.5e-7, all within the 1e-6 tolerance",
         "after": "kept the default"},
        {"arm": "all rule arms", "rule": "budgets", "instance": "the 8 tuning instances", "seeds": "1000-1004",
         "before": "tuning dry run of code 63c30a2 (120 cells): every cell within budget and replayed EQUAL; the "
                   "budget bound only where the rule spends all of it by design (B nlp and B multi_objective)",
         "after": "no parameter changed"},
        {"arm": "all rule arms", "rule": "v2 runtime fixes", "instance": "the 8 tuning instances",
         "seeds": "1000-1004",
         "before": "tuning dry run of the v2 code (commit e46e2eb plus this protocol, 120 rule-arm cells): every "
                   "cell within budget, budget typed and replayed EQUAL, no alarm; success A 30/40, B 40/40, "
                   "E-rule 40/40",
         "after": "no parameter changed; the v1 entries above still hold"},
        {"arm": "C, D, E-LLM", "rule": "provider plumbing (v3)", "seeds": [1000],
         "instance": "D: the 8 tuning instances; E-LLM: cqp-worked-d2, lp-diet-d2, rastrigin-d2; C: cqp-worked-d2, "
                     "knapsack-p02-n5, rastrigin-d2",
         "before": "tuning dry run on deepseek-flash (2026-11-28) of this code with the protocol not yet final: 30 "
                   "calls, 83,422 tokens, $0.0181 at list price, under a 40-call hard stop; every response named "
                   "deepseek-flash; no retry, no error, no repair; every cell replayed EQUAL from its cassette; a "
                   "rerun after deleting a shard made no call",
         "after": "no parameter changed"},
    ],
    "evaluation": "one evaluation = objectives and constraints at one point; an exact repeat of a point inside a "
                  "cell is served from the cell's cache at no cost (the IR adapters' rule)",
    "scoring": "catalogue callables; violation adds |x - round(x)| for integer and binary variables; a front keeps "
               "at most 50 points and only points inside the bounds; a gap below minus the tolerance (less "
               "1e-12) is an alarm and never a success (spec section 2)",
    "fault_hosts": "the seed-0 result of every rule arm on every scoring instance, plus the stored reference "
                   "optimum where the catalogue has one; graded in memory (verify_result) and as a sealed run "
                   "(verify_run)",
    "llm_provider_env": ["LLM_KIND", "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY", "LLM_TIMEOUT_S"],
    "llm": LLM_PROTOCOL,
    "statistics": {"success": "cluster sign-flip on per-instance discordance (spec); exact McNemar on seed 0 "
                              "as a check", "cost": "exact Wilcoxon on per-instance log ERT ratio (spec)",
                   "gap": "exact Wilcoxon on per-instance log10 median gap (secondary)",
                   "ci": "cluster bootstrap, 10000 resamples, seed 1410, strata = family",
                   "holm": "primary family of the spec's 15 hypotheses, not-run hypotheses count as p = 1; "
                           "E-rule comparisons as a separate secondary family"},
}


# ------------------------------------------------------------------------------------------------- digests and JSON
def canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(obj: object) -> str:
    return "sha256:" + hashlib.sha256(canonical(obj)).hexdigest()


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_digest() -> str:
    return digest(PROTOCOL)


CODE_GLOBS = ("benchmarks/e14_common.py", "benchmarks/e14_llm.py", "benchmarks/bench_e14_*.py", "benchmarks/e14_spec/*.py",
              "src/orbitopt/**/*.py", "src/orbitopt/**/*.json")


@functools.lru_cache(maxsize=None)
def code_digest(root: Path = REPO) -> str:
    """SHA-256 over every source file a scored cell runs, by relative path: the same on any checkout of one tree."""
    entries = []
    for pattern in CODE_GLOBS:
        for path in sorted(root.glob(pattern)):
            if "__pycache__" not in path.parts:
                entries.append([path.relative_to(root).as_posix(), file_sha256(path)])
    return digest(sorted(entries))


def sealed(doc: dict) -> dict:
    out = dict(doc)
    out.pop("content_digest", None)
    out["content_digest"] = digest(out)
    return out


def seal_ok(doc: dict) -> bool:
    body = dict(doc)
    claimed = body.pop("content_digest", None)
    return claimed is not None and claimed == digest(body)


def write_json_atomic(path: Path, doc: object) -> None:
    """Write to a temporary file in the same directory and rename it into place: a reader never sees half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "wb") as fh:
        fh.write(json.dumps(doc, sort_keys=True, indent=1, allow_nan=False).encode())
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def jsonable(value):
    """Floats that JSON can hold: inf and nan become strings, numpy scalars become Python numbers."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else ("inf" if v > 0 else "-inf" if v < 0 else "nan")
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def num(value) -> float:
    """Inverse of ``jsonable`` for one number ("inf" and "nan" parse as floats)."""
    return math.nan if value is None else float(value)


# ------------------------------------------------------------------------------------------------------- the freeze
class FreezeError(RuntimeError):
    """A scored run was asked for without a matching freeze."""


def freeze_record() -> dict:
    return {"spec_version": SPEC.SPEC_VERSION, "spec_digest": SPEC.spec_digest(), "protocol": PROTOCOL,
            "protocol_digest": protocol_digest(), "code_digest": code_digest()}


def load_freeze(path: Path = FREEZE_PATH) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    if not seal_ok(doc):
        raise FreezeError(f"{path}: content digest does not match")
    return doc


def check_provider(spec: dict | None) -> None:
    """Refuse a scored provider shard whose provider is not the frozen one (``LLM_PROTOCOL["provider"]``)."""
    frozen = LLM_PROTOCOL["provider"]
    have = {k: (spec or {}).get(k, "https://api.deepseek.com" if k == "base_url" else None)
            for k in ("kind", "model", "base_url")}
    moved = sorted(k for k, v in have.items() if v != frozen[k])
    if moved:
        raise FreezeError(f"the provider is not the frozen one: {moved} differ ({have} vs {frozen['kind']}, "
                          f"{frozen['model']}, {frozen['base_url']})")


def check_freeze(path: Path = FREEZE_PATH) -> dict:
    """The freeze a scored run runs under; refuses a missing freeze or any digest that moved since it was written."""
    doc = load_freeze(path)
    if doc is None:
        raise FreezeError(f"no freeze at {path}: run `bench_e14_end_to_end.py freeze` before any scored shard")
    now = freeze_record()
    moved = [k for k in ("spec_digest", "protocol_digest", "code_digest") if doc[k] != now[k]]
    if moved:
        raise FreezeError(f"frozen digests moved since the freeze: {moved}; a change means a new freeze and a full "
                          f"re-run")
    return doc


# ----------------------------------------------------------------------------------------------- catalogue -> IR
class _Inline(ast.NodeTransformer):
    def __init__(self, values: dict):
        self.values = values

    def visit_Name(self, node):
        return ast.Constant(self.values[node.id]) if node.id in self.values else node


def to_ir(form: dict):
    """The IR problem a neutral form states (as E14-03/06 translate it); the quadrature black box is a ToolCall."""
    from orbitopt.optimization import (Constraint, Objective, OptimizationProblem, Parameter, ToolCall, Variable,
                                       VerificationPolicy)
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
    constraints = []
    for c in form["constraints"]:
        if "quadrature_error" in c["expr"]:
            constraints.append(Constraint(c["name"], ToolCall(TOOL_ID, {"rule": "x0", "n": "x1"}), "<=",
                                          float(data["tau"])))
        else:
            constraints.append(Constraint(c["name"], source(c["expr"]), "<=" if c["kind"] == "le" else "==", 0.0))
    tol = form.get("constraint_abs", 1e-6)
    return OptimizationProblem(
        id=form["id"], variables=tuple(variables),
        parameters=(Parameter("pi", math.pi), Parameter("e", math.e), *vectors),
        objectives=tuple(Objective(f"f{k}", source(e), sense=form["sense"])
                         for k, e in enumerate(form["objective_exprs"])),
        constraints=tuple(constraints), verification=VerificationPolicy(constraint_tol=max(tol, 1e-15)))


def instance_ir(inst):
    return to_ir(dict(P.neutral_form(inst), constraint_abs=inst.tolerance.constraint_abs))


def tools_for(inst):
    if "quadrature_error" not in inst.blackboxes:
        return None
    qe = inst.blackboxes["quadrature_error"]
    return {TOOL_ID: lambda rule, n: qe(rule, n)}


# ------------------------------------------------------------------------------------------------------- scoring
def violation(inst, x: Sequence[float]) -> float:
    """Worst violation over bounds, constraints and integrality, from the catalogue callables."""
    x = [float(v) for v in x]
    worst = F.max_violation(inst, x)
    if inst.variable_type in ("integer", "binary"):
        worst = max(worst, max(abs(v - round(v)) for v in x))
    return float(worst) if not math.isnan(worst) else math.inf


def hypervolume_2d(points, ref) -> float:
    front, best = [], math.inf
    for f1, f2 in sorted(p for p in points if p[0] < ref[0] and p[1] < ref[1]):
        if f2 < best:
            front.append((f1, f2))
            best = f2
    return sum(((front[i + 1][0] if i + 1 < len(front) else ref[0]) - f1) * (ref[1] - f2)
               for i, (f1, f2) in enumerate(front))


ALARM_SLACK = 1e-12
"""Rounding room below minus the tolerance before a gap is an alarm."""


def allowed_gap(inst) -> float:
    tol = inst.tolerance
    return tol.objective_abs + tol.objective_rel * abs(inst.reference.value)


def score(inst, x: Sequence[float] | None = None, front: Sequence[Sequence[float]] | None = None) -> dict:
    """Score a returned point (or front) with the reference problem's callables, never the arm's own claims.

    ``gap`` is the raw oriented gap (0 optimal, positive worse); ``gap_norm`` divides by max(1, |f*|) as
    ``metrics.optimality_gap`` does. ``alarm`` flags a gap below minus the tolerance: a wrong reference or a missed
    violation, which the spec says stops the analysis until explained. Spec section 2: such a gap "is not a win",
    so an alarm cell is never a success (``SCORER_ALARM_RULE``).
    """
    ref, tol = inst.reference, inst.tolerance
    if ref.indicator == "hypervolume":
        pts = [list(map(float, p)) for p in (front or [])][:P.MO_MAX_POINTS]
        kept = [p for p in pts if violation(inst, p) <= tol.constraint_abs]
        values = [tuple(o(p) for o in inst.objectives) for p in kept]
        hv = hypervolume_2d(values, ref.reference_point) if values else 0.0
        gap = ref.value - hv
        alarm = gap < -allowed_gap(inst) - ALARM_SLACK
        return {"objective": None, "hypervolume": hv, "front_size": len(pts), "front_feasible": len(kept),
                "max_violation": 0.0 if kept or not pts else min(violation(inst, p) for p in pts),
                "gap": gap, "gap_norm": gap / max(1.0, abs(ref.value)), "feasible": bool(kept),
                "success": bool(kept) and gap <= allowed_gap(inst) and not alarm, "alarm": alarm}
    if x is None or len(x) != inst.dimension:
        return {"objective": None, "hypervolume": None, "max_violation": math.inf, "gap": math.inf,
                "gap_norm": math.inf, "feasible": False, "success": False, "alarm": False}
    x = [float(v) for v in x]
    viol = violation(inst, x)
    f = float(inst.objectives[0](x))
    raw = (f - ref.value) if inst.sense == "minimize" else (ref.value - f)
    feasible = viol <= tol.constraint_abs
    gap = raw if feasible else math.inf
    alarm = feasible and raw < -allowed_gap(inst) - ALARM_SLACK
    return {"objective": f, "hypervolume": None, "max_violation": viol, "gap": gap,
            "gap_norm": gap / max(1.0, abs(ref.value)), "feasible": feasible,
            "success": feasible and raw <= allowed_gap(inst) and not alarm, "alarm": alarm}


# ---------------------------------------------------------------------------------------- the baseline evaluator
class BudgetExhausted(Exception):
    """Raised instead of an evaluation past the cell's evaluation or wall-clock budget."""

    def __init__(self, kind: str):
        super().__init__(kind)
        self.kind = kind


@dataclass
class CallCounter:
    """An independent counter: wraps the first objective callable, which every evaluation calls exactly once."""
    calls: int = 0

    def wrap(self, fn: Callable) -> Callable:
        def counted(x):
            self.calls += 1
            return fn(x)
        return counted


def counted_instance(inst) -> tuple[object, CallCounter]:
    counter = CallCounter()
    objectives = (counter.wrap(inst.objectives[0]),) + tuple(inst.objectives[1:])
    return replace(inst, objectives=objectives), counter


class Evaluator:
    """One cell's evaluations for a baseline arm: budget enforcement, a point cache, and the best point seen.

    ``evaluate(x)`` returns (objective values, constraint values) and costs one evaluation unless the exact point
    was evaluated before in this cell. It raises ``BudgetExhausted`` instead of evaluating past the budget.
    """

    def __init__(self, inst, max_evaluations: int, max_wall_s: float):
        self.inst, self.max_evaluations, self.max_wall_s = inst, max_evaluations, max_wall_s
        self.count = 0
        self.limit = max_evaluations  # an arm may lower it for one allocation (a Tchebycheff weight)
        self.started = time.perf_counter()
        self.cache: dict[bytes, tuple[tuple[float, ...], tuple[float, ...]]] = {}
        self.best: tuple[tuple, list[float]] | None = None  # (key, x): feasible first, then objective
        self.points: list[list[float]] = []  # every evaluated point, for multi-objective fronts
        self.exhausted: str | None = None

    def evaluate(self, x) -> tuple[tuple[float, ...], tuple[float, ...]]:
        xa = np.asarray(x, dtype=float).ravel()
        key = xa.tobytes()
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        if self.count >= self.limit:
            if self.limit < self.max_evaluations:
                raise BudgetExhausted("allocation")
            self.exhausted = "evaluations"
            raise BudgetExhausted("evaluations")
        if time.perf_counter() - self.started > self.max_wall_s:
            self.exhausted = "wall_clock"
            raise BudgetExhausted("wall_clock")
        self.count += 1
        xl = [float(v) for v in xa]
        fs = tuple(float(o(xl)) for o in self.inst.objectives)
        gs = tuple(float(c.fn(xl)) for c in self.inst.constraints)
        self.cache[key] = (fs, gs)
        self._track(xl, fs)
        return fs, gs

    def _track(self, x: list[float], fs: tuple[float, ...]) -> None:
        if len(fs) > 1:
            self.points.append(x)
            return
        viol = violation(self.inst, x)
        sign = 1.0 if self.inst.sense == "minimize" else -1.0
        f = sign * fs[0] if math.isfinite(fs[0]) else math.inf
        key = (0, f) if viol <= self.inst.tolerance.constraint_abs else (1, viol)
        if self.best is None or key < self.best[0]:
            self.best = (key, x)

    def best_x(self) -> list[float] | None:
        return None if self.best is None else self.best[1]


# --------------------------------------------------------------------------------------------------------- rows
def row_digest(row: dict) -> str:
    """The digest a replay must reproduce: the row without its timing fields and without the replay record."""
    body = {k: v for k, v in row.items() if k not in TIMING_FIELDS and k not in ("replay", "row_digest")}
    return digest(jsonable(body))


def cell_id(arm: str, inst_id: str, seed: int) -> str:
    return f"{arm}|{inst_id}|{seed}"


def instances(split: str, family: str | None = None):
    return [i for i in P.INSTANCES if i.split == split and (family is None or i.family == family)]


def families() -> list[str]:
    out: list[str] = []
    for inst in P.INSTANCES:
        if inst.family not in out:
            out.append(inst.family)
    return out


def group_of(problem_class: str) -> str:
    return next(g for g, classes in M.ANALYSIS_GROUPS.items() if problem_class in classes)



def base_row(arm: str, inst, seed: int, mode: str = "scored") -> dict:
    return {"arm_id": arm, "instance_id": inst.id, "family": inst.family, "problem_class": inst.problem_class,
            "group": group_of(inst.problem_class), "split": inst.split, "seed": seed, "well_known": inst.well_known,
            "dimension": inst.dimension, "problem_digest": F.problem_digest(inst), "mode": mode,
            "budget_evaluations": A.BUDGET.evaluations(inst.dimension), "budget_wall_s": A.BUDGET.wall_clock_s,
            "llm_wall_clock_s": 0.0, "llm_calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0,
            "transcript_digest": None}


def outcome_status(sc: dict, stop: str, has_point: bool) -> str:
    """The spec's row status; ``stop`` keeps why the cell ended (completed, budget_exhausted:<kind>, error, ...)."""
    if not has_point:
        return "error" if stop in ("error", "refused") else "infeasible"
    if sc["success"]:
        return "success"
    return "feasible_not_optimal" if sc["feasible"] else "infeasible"


def ir_result(problem, inst, x, front, solver_id: str, termination: str):
    """An IR result for a baseline arm's point, with the claims the arm makes; the guard grades it as any other."""
    from orbitopt.optimization import EvaluationError, OptimizationResult, ParetoPoint
    names = [v.name for v in problem.variables]
    tools = tools_for(inst)
    pts = list(front or []) or ([x] if x is not None else [])
    if not pts:
        return OptimizationResult(problem_digest=problem.digest(), solver_id=solver_id, termination=termination)
    members, first = [], None
    for p in pts[:P.MO_MAX_POINTS]:
        xm = {n: float(v) for n, v in zip(names, p)}
        try:
            ev = problem.evaluate(xm, tools)
        except EvaluationError:
            continue
        members.append(ParetoPoint(x=xm, objective_values=dict(ev.objectives)))
        if first is None:
            first = (xm, ev)
    if first is None:
        return OptimizationResult(problem_digest=problem.digest(), solver_id=solver_id, termination=termination)
    xm, ev = first
    return OptimizationResult(
        problem_digest=problem.digest(), solver_id=solver_id, termination=termination,
        solver_converged=termination == "converged",
        constraints_satisfied=problem.constraints_satisfied(xm, ev.residuals), x=xm,
        objective_values=dict(ev.objectives), constraint_residuals=dict(ev.residuals),
        front=tuple(members) if len(inst.objectives) > 1 else ())


def screen(problem, inst, result) -> dict:
    """The verification guard on a result with no independent evidence: what a baseline arm can show."""
    from orbitopt.optimization.verification import verify_result
    report = verify_result(problem, result, tools=tools_for(inst))
    return {"verdict": report.verdict.value, "run_status": None,
            "statuses": {k: v.outcome.value for k, v in report.statuses.items()},
            "codes": sorted(c.value for c in report.codes)}


_TERMINATION = {"completed": "converged", "budget_exhausted:evaluations": "max_evaluations",
                "budget_exhausted:wall_clock": "timeout", "error": "error"}


def baseline_row(arm: str, inst, seed: int, *, run, evaluations: int, independent: int, budget: int, wall: float,
                 error: str | None, mode: str = "scored") -> dict:
    sc = score(inst, x=run.x, front=run.front)
    problem = instance_ir(inst)
    result = ir_result(problem, inst, run.x, run.front, "+".join(run.solvers) or arm,
                       _TERMINATION.get(run.stop, "error"))
    row = base_row(arm, inst, seed, mode)
    row.update({
        "compile_outcome": "reference", "stop": run.stop, "error": error, "solvers": list(run.solvers),
        "status": outcome_status(sc, run.stop, run.x is not None or bool(run.front)),
        "x": None if run.x is None else [float(v) for v in run.x],
        "front": None if run.front is None else [[float(v) for v in p] for p in run.front[:P.MO_MAX_POINTS]],
        **{k: sc[k] for k in ("objective", "hypervolume", "max_violation", "gap", "gap_norm", "feasible", "success",
                              "alarm")},
        "n_evaluations": evaluations, "n_evaluations_independent": independent,
        "within_budget": independent <= budget and evaluations == independent and wall <= A.BUDGET.wall_clock_s,
        "budget_typed": (run.stop.startswith("budget_exhausted") if independent >= budget else True),
        "wall_clock_s": wall, "verification": screen(problem, inst, result),
        "artifact_digest": digest(jsonable(result.to_dict())), "messages": list(run.messages)[:5],
    })
    row["row_digest"] = row_digest(row)
    return row
