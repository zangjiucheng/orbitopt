"""Agent iteration runtime (E14-05): a bounded, verified, replayable optimization loop over an IR problem.

``solve(problem, planner=...)`` lets a planner choose solvers, options, start
points and seeds, never the problem. Every proposal is validated against the
``Policy`` and the solver registry before it runs, every candidate is graded
by the E14-06 guard, and the final answer is verified with alternate solvers
as independent evidence. ``save_run`` writes a run directory and ``replay``
re-executes it without an LLM call. See ``docs/agent/runtime.md``.
"""

from .diagnostics import DISHONEST, REMEDIES, diagnose, distrust, feasible, loosened
from .loop import (MAX_ALTERNATES, VERIFICATION_SHARE, EvaluationBudgetExceeded, EvaluationMeter, known_tool_ids,
                   screen_cost, solve)
from .model import (TRACE_SCHEMA, TRACE_SCHEMAS, Action, AlternateRun, BudgetEffect, BudgetEvent,
                    BudgetResource, BudgetStage, BudgetUse, CancelPoint, CancelReason, Decision,
                    Diagnosis, FailureKind, IterationRecord, Limits, Policy, Proposal, RecordKind, Remedy, RunOutcome,
                    RunStatus, StopReason, Trace, VerificationRecord, Violation, ViolationCode, ENVIRONMENT_KEYS,
                    UNKNOWN_VERSION, derive_budget, orbitopt_version, trace_digest, upgrade_trace)
from .persistence import (RUN_MANIFEST_FORMAT, CheckReport, ReplayReport, UnsafePathError, check_run, load_trace,
                          replay, safe_relative, save_run)
from .planners import (PROPOSAL_SCHEMA, LLMPlanner, Planner, PlannerState, PlannerUnavailable, RecordedPlanner,
                       RuleBasedPlanner, prompt_state)
from .validation import PROBLEM_FIELDS, PROPOSAL_FIELDS, validate_proposal

__all__ = [
    "Action", "AlternateRun", "BudgetEffect", "BudgetEvent", "BudgetResource", "BudgetStage", "BudgetUse",
    "CancelPoint", "CancelReason", "CheckReport", "DISHONEST", "Decision", "Diagnosis", "ENVIRONMENT_KEYS",
    "EvaluationBudgetExceeded", "EvaluationMeter", "FailureKind", "IterationRecord", "LLMPlanner", "Limits",
    "MAX_ALTERNATES", "PROBLEM_FIELDS", "PROPOSAL_FIELDS", "PROPOSAL_SCHEMA", "Planner", "PlannerState",
    "PlannerUnavailable", "Policy", "Proposal", "REMEDIES", "RUN_MANIFEST_FORMAT", "RecordKind", "RecordedPlanner",
    "Remedy", "ReplayReport", "RuleBasedPlanner", "RunOutcome", "RunStatus", "StopReason", "TRACE_SCHEMA",
    "TRACE_SCHEMAS", "Trace", "UNKNOWN_VERSION", "UnsafePathError", "VERIFICATION_SHARE", "VerificationRecord",
    "Violation", "ViolationCode", "check_run", "derive_budget", "diagnose", "distrust", "feasible",
    "known_tool_ids", "load_trace", "loosened", "orbitopt_version", "prompt_state", "replay", "safe_relative",
    "save_run", "screen_cost", "solve", "trace_digest", "upgrade_trace", "validate_proposal",
]
