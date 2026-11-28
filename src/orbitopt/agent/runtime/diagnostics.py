"""Read one solve's outcome, as the guard graded it, and name the bounded remedies for it.

``diagnose`` never reads a status off the result. Feasibility is the guard's
``constraints_satisfied``; a dishonest claim is a guard finding; what
``solver_converged`` is worth comes from ``capability.convergence``, so a
differential-evolution convergence is a ``heuristic_stop``, not an optimum.
Every remedy changes the solver, its options, the start point or the seed.
None changes the problem: an infeasible or unbounded problem stays as asked.
"""

from __future__ import annotations

from typing import Mapping

from orbitopt.optimization import OptimizationResult, Termination
from orbitopt.optimization.classify import ProblemClass, ProblemStructure
from orbitopt.optimization.solvers import SolverCapability
from orbitopt.optimization.verification import FindingCode, Outcome, VerificationReport

from .model import Diagnosis, FailureKind, Remedy

PROVABLY_CONVEX = frozenset({ProblemClass.LP, ProblemClass.CONVEX_QP})
"""Classes whose convexity ``classify`` decides exactly: a first-order point of these is the global optimum."""
CERTIFIED_CLASSES = PROVABLY_CONVEX | {ProblemClass.MILP}
"""Classes a ``certificate`` solver solves to proven optimality (HiGHS: LP duality, MILP bound)."""

DISHONEST = frozenset({
    FindingCode.OBJECTIVE_MISMATCH, FindingCode.RESIDUAL_MISMATCH, FindingCode.PROBLEM_MISMATCH,
    FindingCode.STALE_CACHE, FindingCode.FORGED_VERIFICATION, FindingCode.SELF_ATTESTATION,
    FindingCode.DIGEST_MISMATCH, FindingCode.UNAUTHORIZED_RELAXATION, FindingCode.FORMULATION_MISMATCH,
    FindingCode.MALFORMED_RESULT, FindingCode.NON_FINITE,
})
"""Guard findings that mean the solver's report cannot be trusted, as opposed to an honest bad point.
A result that vouches for itself (``self_attestation``) is one: no solver may supply independent evidence."""
GUARD_ONLY = ("independently_verified", "optimality_certified")
"""Statuses only the guard sets; a solver result that claims one is not to be trusted."""

LOOSEN_FACTOR = 100.0

REMEDIES: dict[FailureKind, tuple[Remedy, ...]] = {
    FailureKind.VERIFIED_LOCAL: (Remedy.STOP,),
    FailureKind.LOCAL_OPTIMUM: (Remedy.RESTART_GLOBAL, Remedy.RESTART_PERTURBED, Remedy.STOP),
    FailureKind.HEURISTIC_STOP: (Remedy.REFINE_LOCALLY, Remedy.RESTART_GLOBAL),
    FailureKind.NON_CONVERGENCE: (Remedy.REFINE_LOCALLY, Remedy.LOOSEN_SOLVER_TOLERANCE, Remedy.SWITCH_SOLVER),
    FailureKind.STALLED: (Remedy.LOOSEN_SOLVER_TOLERANCE, Remedy.RESTART_PERTURBED, Remedy.SWITCH_SOLVER),
    FailureKind.INFEASIBLE: (Remedy.SWITCH_SOLVER, Remedy.CHANGE_INITIAL_GUESS, Remedy.RESTART_GLOBAL),
    FailureKind.UNBOUNDED: (Remedy.STOP,),
    FailureKind.EVALUATION_ERROR: (Remedy.CHANGE_INITIAL_GUESS, Remedy.SWITCH_SOLVER),
    FailureKind.SOLVER_ERROR: (Remedy.SWITCH_SOLVER,),
    FailureKind.BUDGET: (Remedy.REFINE_LOCALLY, Remedy.STOP),
    FailureKind.REJECTED_BY_GUARD: (Remedy.SWITCH_SOLVER,),
}
"""Each failure kind's remedies, most promising first. The table is the whole policy: a planner may pick
any remedy in the row, and the rule-based planner takes the first one it can carry out."""


def distrust(result: OptimizationResult, report: VerificationReport) -> tuple[str, ...]:
    """Why the guard's grade shows that the solver's report cannot be trusted; empty when it can.

    A claim the guard could not assess (a tool that raised) is not a lie. A
    claim it failed is, and so is any claim to a status only the guard sets.
    """
    reasons = sorted(c.value for c in report.codes & DISHONEST)
    reasons += [f"claims {s}" for s in GUARD_ONLY if getattr(result, s) is True]
    reasons += [f"claims {s}, which the guard failed" for s in ("solver_converged", "constraints_satisfied")
                if getattr(result, s) is True and report.statuses[s].outcome is Outcome.FAILED]
    return tuple(reasons)


def feasible(result: OptimizationResult | None, report: VerificationReport | None) -> bool:
    """Whether the guard found the point feasible on its own re-evaluation, from a report it can trust."""
    return result is not None and report is not None \
        and report.statuses["constraints_satisfied"].outcome is Outcome.PASSED and not distrust(result, report)


def diagnose(result: OptimizationResult | None, report: VerificationReport | None,
             capability: SolverCapability, structure: ProblemStructure, error: str = "") -> Diagnosis:
    """The failure kind of one solve and its remedies. ``error`` is set when the solver raised."""
    def kind_of() -> tuple[FailureKind, str]:
        if result is None:
            return FailureKind.SOLVER_ERROR, error or "the solver returned no result"
        if report is not None and distrust(result, report):
            return FailureKind.REJECTED_BY_GUARD, "; ".join(distrust(result, report))
        termination = result.termination
        if termination is Termination.UNBOUNDED:
            return FailureKind.UNBOUNDED, result.message
        if termination is Termination.ERROR:
            failed = "evaluations failed" in result.message or report is not None and \
                FindingCode.EVALUATION_ERROR in report.codes
            return (FailureKind.EVALUATION_ERROR if failed else FailureKind.SOLVER_ERROR), result.message
        if termination in (Termination.MAX_EVALUATIONS, Termination.TIMEOUT, Termination.CANCELLED):
            return FailureKind.BUDGET, termination.value
        if not feasible(result, report):
            return FailureKind.INFEASIBLE, termination.value
        if termination is Termination.STALLED:
            return FailureKind.STALLED, result.message
        if termination is not Termination.CONVERGED:
            return FailureKind.NON_CONVERGENCE, termination.value
        if capability.convergence == "certificate" and structure.problem_class in CERTIFIED_CLASSES:
            return FailureKind.VERIFIED_LOCAL, "converged with a certificate"
        if capability.convergence == "exhaustive":
            return FailureKind.VERIFIED_LOCAL, "evaluated every point of the finite domain"
        if capability.convergence == "first_order" and structure.problem_class in PROVABLY_CONVEX:
            return FailureKind.VERIFIED_LOCAL, "first-order point of a provably convex problem"
        if capability.convergence == "first_order":
            return FailureKind.LOCAL_OPTIMUM, "first-order point; the problem is not provably convex"
        return FailureKind.HEURISTIC_STOP, f"{capability.convergence} convergence claims no optimality"

    kind, detail = kind_of()
    return Diagnosis(kind, REMEDIES[kind], detail)


def loosened(options: Mapping[str, object], capability: SolverCapability) -> dict | None:
    """The options with every solver tolerance the solver takes loosened by ``LOOSEN_FACTOR``, within its schema.

    ``None`` when no tolerance can be loosened further. Only solver
    tolerances are touched: the solver's ``OptionSpec`` marks them, and no
    problem tolerance is a solver option.
    """
    out, changed = dict(options), False
    for name, spec in sorted(capability.options.items()):
        if not spec.tolerance:
            continue
        current = float(options.get(name, spec.default))
        value = min(current * LOOSEN_FACTOR, spec.high)
        if value > current:
            out[name], changed = value, True
    return out if changed else None
