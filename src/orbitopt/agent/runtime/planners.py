"""Planners: what proposes the next step of a run. Every proposal is untrusted and validated by the runtime.

``RuleBasedPlanner`` is deterministic. It ranks the solvers from
``classify`` and the registry's capabilities, puts a global solver first
whenever the problem is not provably convex (multimodality cannot be read off
the syntax), and walks the remedy table of ``diagnostics`` after each solve.
``LLMPlanner`` asks a language model for the same JSON object through
``ChatProvider.generate_structured``, so a ``MockProvider`` or a
``ReplayProvider`` drives it without a network. ``RecordedPlanner`` feeds back
the proposals a trace recorded; ``replay`` uses it, so a replay makes no LLM
call.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

from orbitopt.agent.providers import ChatProvider, Message, ProviderError, StructuredOutputError
from orbitopt.optimization import OptimizationProblem
from orbitopt.optimization.classify import ProblemStructure
from orbitopt.optimization.solvers import OptionSpec, SolverCapability

from .diagnostics import CERTIFIED_CLASSES, PROVABLY_CONVEX, feasible, loosened
from .model import BudgetUse, IterationRecord, Proposal, RecordKind, Remedy


class PlannerUnavailable(Exception):
    """The planner cannot propose anything (its provider is down); the run stops with ``planner_failed``."""


@dataclass(frozen=True)
class PlannerState:
    """What a planner sees before each proposal.

    ``solvers`` are the solvers it may name, most specialised first, without
    those the run has excluded for a dishonest report or a crash.
    ``remaining.evaluations`` is what one decision may still spend.
    ``incumbent`` is the best record whose point the guard found feasible.
    """

    problem: OptimizationProblem
    structure: ProblemStructure
    solvers: tuple[SolverCapability, ...]
    history: tuple[IterationRecord, ...]
    incumbent: IterationRecord | None
    remaining: BudgetUse
    seed: int

    @property
    def solves(self) -> tuple[IterationRecord, ...]:
        return tuple(r for r in self.history if r.kind is RecordKind.SOLVE)

    @property
    def provably_convex(self) -> bool:
        return self.structure.problem_class in PROVABLY_CONVEX


class Planner(Protocol):
    """Proposes the next step. ``name``, ``provider`` and ``model`` are recorded in the trace.

    ``provider`` and ``model`` identify the LLM behind the planner, ``None``
    for a planner without one.
    """

    name: str
    provider: str | None
    model: str | None

    def propose(self, state: PlannerState) -> Proposal: ...


def ranked(state: PlannerState) -> list[SolverCapability]:
    """The rule: specialised first on a class whose optimum a solver can certify, global first otherwise."""
    if state.structure.problem_class in CERTIFIED_CLASSES:
        return list(state.solvers)
    return sorted(state.solvers, key=lambda c: not c.is_global)


REFINE, PERTURBED, LOOSEN, NEW_START = "refine", "perturbed", "loosen", "new start"
"""Rationale prefixes the rule-based planner reads back from its own earlier decisions."""
POPULATION_GENERATIONS = 150
"""Population size for a population solver is chosen so the allocation covers about this many generations."""


class RuleBasedPlanner:
    """Deterministic planning from structure, capabilities and the remedy table; no LLM, no tokens.

    Non-convex problems run rounds of a global solve and a local refinement
    from its point. Up to ``max_restarts`` further rounds use a new seed each,
    and the restarts end at the first round that does not beat the earlier
    best by more than the problem's ``objective_rtol``. A failed solve takes
    the first remedy of its diagnosis that the planner can still carry out.
    """

    name = "rule"
    provider = None
    model = None

    def __init__(self, *, max_restarts: int = 3):
        self.max_restarts = max_restarts

    def propose(self, state: PlannerState) -> Proposal:
        return Proposal(self.plan(state))

    def plan(self, state: PlannerState) -> dict:
        solvers = ranked(state)
        budget = state.remaining.evaluations
        if not solvers:
            return _stop("no solver left to try")
        solves = state.solves
        if not solves:
            return self.solve(state, solvers[0], budget, "primary: the highest-ranked solver for this class")
        last = solves[-1]
        for remedy in last.diagnosis.remedies:
            step = self.apply(remedy, state, solvers, last, budget)
            if step is not None:
                return step
        return _stop(f"no remedy left for {last.diagnosis.kind.value}")

    def apply(self, remedy: Remedy, state: PlannerState, solvers: list[SolverCapability], last: IterationRecord,
              budget: int) -> dict | None:
        by_id = {c.solver_id: c for c in solvers}
        tried = {r.decision.solver_id for r in state.solves}
        incumbent = state.incumbent.result.x if state.incumbent else None
        if remedy is Remedy.STOP:
            return _stop(f"{last.diagnosis.kind.value}: {last.diagnosis.detail}")
        if remedy is Remedy.REFINE_LOCALLY:
            source = last.result.x if feasible(last.result, last.screen) else incumbent
            local = next((c for c in solvers if c.convergence == "first_order"), None)
            if source is None or local is None or _started_from(state, local.solver_id, source):
                return None
            return self.solve(state, local, budget, f"{REFINE} the last point locally", x0=source)
        if remedy is Remedy.RESTART_GLOBAL:
            heuristic = next((c for c in solvers if c.is_global and c.convergence == "heuristic"), None)
            rounds = sum(1 for r in state.solves if heuristic and r.decision.solver_id == heuristic.solver_id)
            if heuristic is None or rounds > self.max_restarts or _stale(state):
                return None
            return self.solve(state, heuristic, budget, f"global restart {rounds} with a new seed",
                              seed=state.seed + rounds)
        if remedy is Remedy.RESTART_PERTURBED:
            local = next((c for c in solvers if c.convergence == "first_order"), None)
            restarts = sum(1 for r in state.solves if r.decision.rationale.startswith(PERTURBED))
            if local is None or incumbent is None or restarts >= self.max_restarts or _stale(state):
                return None
            x0 = _perturbed(state, incumbent, random.Random(state.seed * 1_000_003 + len(state.history)))
            return self.solve(state, local, budget, f"{PERTURBED} restart {restarts + 1} around the incumbent",
                              x0=x0)
        if remedy is Remedy.SWITCH_SOLVER:
            fresh = next((c for c in solvers if c.solver_id not in tried), None)
            if fresh is None:
                return None
            return self.solve(state, fresh, budget, f"switch from {last.decision.solver_id}", x0=incumbent)
        if remedy is Remedy.LOOSEN_SOLVER_TOLERANCE:
            capability = by_id.get(last.decision.solver_id)
            options = None if capability is None else loosened(last.decision.options, capability)
            loosenings = sum(1 for r in state.solves if r.decision.rationale.startswith(LOOSEN))
            if options is None or loosenings >= 2:
                return None
            x0 = last.result.x if feasible(last.result, last.screen) else incumbent
            return self.solve(state, capability, budget, f"{LOOSEN} the solver tolerance", x0=x0, options=options)
        if remedy is Remedy.CHANGE_INITIAL_GUESS:
            capability = by_id.get(last.decision.solver_id)
            changes = sum(1 for r in state.solves if r.decision.rationale.startswith(NEW_START))
            if capability is None or changes >= self.max_restarts:
                return None
            x0 = _sampled(state.problem, random.Random(state.seed * 1_000_003 + len(state.history)))
            return self.solve(state, capability, budget, f"{NEW_START} point inside the domain", x0=x0)
        return None

    def solve(self, state: PlannerState, capability: SolverCapability, budget: int, rationale: str, *,
              x0=None, seed: int | None = None, options: dict | None = None) -> dict:
        share = budget if capability.convergence in ("certificate", "first_order") else max(1, budget // 2)
        if capability.convergence == "exhaustive":  # exactly the domain, or the whole budget for it to refuse
            share = min(budget, state.structure.domain_size or budget)
        options = {**default_options(capability, share, len(state.problem.variables)), **(options or {})}
        return {"action": "solve", "solver_id": capability.solver_id, "options": options,
                "x0": None if x0 is None else dict(x0), "seed": state.seed if seed is None else seed,
                "evaluations": share, "rationale": rationale}


def default_options(capability: SolverCapability, evaluations: int, dimension: int) -> dict:
    """A population size that lets ``evaluations`` cover about ``POPULATION_GENERATIONS`` generations, within 5-15
    members per variable; no options for a solver without a population."""
    if "popsize" not in capability.options:
        return {}
    members = evaluations // (POPULATION_GENERATIONS * max(1, dimension))
    return {"popsize": int(min(max(members, 5), 15))}


def _stop(why: str) -> dict:
    return {"action": "stop", "rationale": why}


def _started_from(state: PlannerState, solver_id: str, x) -> bool:
    return any(r.decision.solver_id == solver_id and r.decision.x0 == dict(x) for r in state.solves)


def _stale(state: PlannerState) -> bool:
    """The latest round did not beat the earlier rounds' best by more than ``objective_rtol``.

    A round is a solve and the local refinements that follow it. Restarts end
    at the first round that does not improve, so the planner spends no more
    evaluations once a new start keeps landing on what it already has.
    """
    rounds: list[list[IterationRecord]] = []
    for record in state.solves:
        if record.decision.rationale.startswith(REFINE) and rounds:
            rounds[-1].append(record)
        else:
            rounds.append([record])
    if len(rounds) < 2:
        return False
    sign = -1.0 if state.problem.objectives[0].sense == "maximize" else 1.0
    name = state.problem.objectives[0].name

    def best(records) -> float | None:
        values = [sign * r.result.objective_values[name] for r in records if feasible(r.result, r.screen)]
        return min(values) if values else None

    before, latest = best([r for group in rounds[:-1] for r in group]), best(rounds[-1])
    if before is None:
        return False
    tolerance = state.problem.verification.objective_rtol * max(1.0, abs(before))
    return latest is None or latest >= before - tolerance


def _perturbed(state: PlannerState, x, rng: random.Random) -> dict:
    out = {}
    for v in state.problem.variables:
        value = float(x[v.name])
        if v.kind != "continuous":
            out[v.name] = value
            continue
        width = (v.upper - v.lower) if v.lower is not None and v.upper is not None else max(1.0, abs(value))
        value += rng.uniform(-0.1, 0.1) * width
        out[v.name] = min(max(value, -math.inf if v.lower is None else v.lower),
                          math.inf if v.upper is None else v.upper)
    return out


def _sampled(problem: OptimizationProblem, rng: random.Random) -> dict:
    """A seeded point inside every variable's domain: uniform where the bounds are finite, else near a bound or 0."""
    out = {}
    for v in problem.variables:
        if v.kind == "binary":
            out[v.name] = float(rng.randint(0, 1))
        elif v.kind == "discrete":
            out[v.name] = float(rng.choice(v.choices))
        else:
            lo = v.lower if v.lower is not None else (v.upper - 1.0 if v.upper is not None else -1.0)
            hi = v.upper if v.upper is not None else lo + 2.0
            value = rng.uniform(lo, hi)
            out[v.name] = float(round(value)) if v.kind == "integer" else value
    return out


# ------------------------------------------------------------------------------------------------- LLM planner
PROPOSAL_SCHEMA = {
    "type": "object",
    "required": ["action"],
    "properties": {
        "action": {"enum": ["solve", "stop"]},
        "solver_id": {"type": "string"},
        "options": {"type": "object"},
        "x0": {"type": ["object", "null"]},
        "seed": {"type": "integer"},
        "evaluations": {"type": "integer"},
        "max_iterations": {"type": "integer"},
        "rationale": {"type": "string"},
    },
}
"""The structured-output schema. It leaves extra fields open on purpose: a reply that asks to change the
problem must reach the runtime's validation, which records it as a ``problem_edit`` violation."""

SYSTEM_PROMPT = (
    "You plan the next step of a bounded numerical optimization run. Reply with one JSON object: "
    '{"action": "solve" or "stop", "solver_id": one of the listed solvers, "options": solver options within '
    'their listed ranges, "x0": a start point inside the variable bounds or null, "seed": an integer, '
    '"evaluations": at most the remaining evaluations, "rationale": one sentence}. '
    "You may choose only the solver, its options, the start point, the seed and the evaluation share. You may "
    "not change the objective, the constraints, the bounds, the parameters, the budget or the verification "
    "policy; such a request is refused and recorded. Every answer is checked independently afterwards, so "
    "solver_converged is not proof of optimality. Stop when another solve is unlikely to improve the answer."
)


class LLMPlanner:
    """Proposals from a language model through ``ChatProvider.generate_structured``.

    The prompt holds the problem's structure, the solvers with their option
    ranges, the remaining budget (no wall clock, so a recorded cassette
    replays) and a summary of every earlier iteration. A reply that fails the
    schema twice becomes a malformed proposal, which the runtime records as a
    violation. Any other provider failure raises ``PlannerUnavailable``.
    ``provider`` is the chat provider's ``name`` and ``model`` its model.
    """

    def __init__(self, provider: ChatProvider, *, max_reply_tokens: int = 512, temperature: float = 0.0,
                 name: str = "llm"):
        self.chat = provider
        self.provider = provider.name
        self.model = getattr(provider, "model", None)
        self.max_reply_tokens = max_reply_tokens
        self.temperature = temperature
        self.name = name

    def propose(self, state: PlannerState) -> Proposal:
        messages = (Message("system", SYSTEM_PROMPT), Message("user", json.dumps(prompt_state(state), sort_keys=True)))
        try:
            response = self.chat.generate_structured(
                messages, PROPOSAL_SCHEMA, max_tokens=max(1, min(self.max_reply_tokens, state.remaining.tokens)),
                temperature=self.temperature, seed=state.seed)
        except StructuredOutputError as exc:
            return Proposal(None, exc.usage.total, f"structured output failed: {'; '.join(exc.errors[:3])}")
        except ProviderError as exc:
            raise PlannerUnavailable(f"{type(exc).__name__}: {exc}") from exc
        return Proposal(response.parsed, response.usage.total)


def prompt_state(state: PlannerState) -> dict:
    """The planner's view as JSON: deterministic for a deterministic run, so a cassette replays."""
    problem = state.problem

    def option(spec: OptionSpec) -> dict:
        if spec.choices:
            return {"choices": list(spec.choices)}
        return {"type": spec.kind.__name__} if spec.kind is bool else \
            {"type": spec.kind.__name__, "min": spec.low, "max": spec.high}

    return {
        "problem": {
            "id": problem.id, "class": state.structure.problem_class.value,
            "provably_convex": state.provably_convex,
            "variables": [{"name": v.name, "kind": v.kind, "lower": v.lower, "upper": v.upper,
                           "choices": list(v.choices)} for v in problem.variables],
            "objectives": [{"name": o.name, "sense": o.sense} for o in problem.objectives],
            "constraints": [{"name": c.name, "sense": c.sense, "rhs": c.rhs} for c in problem.constraints],
        },
        "solvers": [{"solver_id": c.solver_id, "convergence": c.convergence, "is_global": c.is_global,
                     "takes_max_iterations": c.iteration_cap,
                     "options": {name: option(spec) for name, spec in sorted(c.options.items())}}
                    for c in state.solvers],
        "remaining": {"iterations": state.remaining.iterations, "evaluations": state.remaining.evaluations,
                      "tokens": state.remaining.tokens},
        "seed": state.seed,
        "incumbent": None if state.incumbent is None else {
            "iteration": state.incumbent.index, "objective_values": dict(state.incumbent.result.objective_values),
            "x": dict(state.incumbent.result.x)},
        "history": [_history_entry(r) for r in state.history],
    }


def _history_entry(record: IterationRecord) -> dict:
    entry = {"iteration": record.index, "kind": record.kind.value,
             "violations": [v.code.value for v in record.violations]}
    if record.decision is not None:
        entry["solver_id"] = record.decision.solver_id
    if record.result is not None:
        entry.update(termination=record.result.termination.value,
                     objective_values=dict(record.result.objective_values),
                     evaluations=record.evaluations, feasible=feasible(record.result, record.screen))
    if record.diagnosis is not None:
        entry.update(diagnosis=record.diagnosis.kind.value, remedies=[r.value for r in record.diagnosis.remedies])
    return entry


class RecordedPlanner:
    """Serves recorded proposals in order; running past the end raises ``PlannerUnavailable``.

    ``name``, ``provider`` and ``model`` are what the trace records, so a replay names the original planner.
    """

    def __init__(self, proposals: Iterable[Proposal], *, name: str = "recorded", provider: str | None = None,
                 model: str | None = None):
        self._proposals: Sequence[Proposal] = tuple(proposals)
        self._next = 0
        self.name, self.provider, self.model = name, provider, model

    def propose(self, state: PlannerState) -> Proposal:
        if self._next >= len(self._proposals):
            raise PlannerUnavailable(f"the recording holds only {len(self._proposals)} proposals")
        self._next += 1
        return self._proposals[self._next - 1]
