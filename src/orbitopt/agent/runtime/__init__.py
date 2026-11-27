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
from .model import (TRACE_SCHEMA, Action, AlternateRun, BudgetUse, Decision, Diagnosis, FailureKind, IterationRecord,
                    Limits, Policy, Proposal, RecordKind, Remedy, RunOutcome, RunStatus, StopReason, Trace,
                    VerificationRecord, Violation, ViolationCode, trace_digest)
from .persistence import (RUN_MANIFEST_FORMAT, RegradeReport, ReplayReport, UnsafePathError, load_trace, regrade,
                          replay, safe_relative, save_run)
from .planners import (PROPOSAL_SCHEMA, LLMPlanner, Planner, PlannerState, PlannerUnavailable, RecordedPlanner,
                       RuleBasedPlanner, prompt_state)
from .validation import PROBLEM_FIELDS, PROPOSAL_FIELDS, validate_proposal

__all__ = [
    "Action", "AlternateRun", "BudgetUse", "DISHONEST", "Decision", "Diagnosis", "EvaluationBudgetExceeded",
    "EvaluationMeter", "FailureKind", "IterationRecord", "LLMPlanner", "Limits", "MAX_ALTERNATES", "PROBLEM_FIELDS",
    "PROPOSAL_FIELDS", "PROPOSAL_SCHEMA", "Planner", "PlannerState", "PlannerUnavailable", "Policy", "Proposal",
    "REMEDIES", "RUN_MANIFEST_FORMAT", "RecordKind", "RecordedPlanner", "RegradeReport", "Remedy", "ReplayReport",
    "RuleBasedPlanner", "RunOutcome", "RunStatus", "StopReason", "TRACE_SCHEMA", "Trace", "UnsafePathError",
    "VERIFICATION_SHARE", "VerificationRecord", "Violation", "ViolationCode", "diagnose", "distrust", "feasible",
    "known_tool_ids", "load_trace", "loosened", "prompt_state", "regrade", "replay", "safe_relative", "save_run",
    "screen_cost", "solve", "trace_digest", "validate_proposal",
]
