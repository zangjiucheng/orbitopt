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
from .validation import OPTION_SPECS, PROBLEM_FIELDS, PROPOSAL_FIELDS, OptionSpec, validate_proposal
from .planners import (PROPOSAL_SCHEMA, LLMPlanner, Planner, PlannerState, PlannerUnavailable, RecordedPlanner,
                       RuleBasedPlanner, prompt_state)

__all__ = [
    "Action", "AlternateRun", "BudgetUse", "Decision", "diagnose", "Diagnosis", "DISHONEST", "distrust",
    "EvaluationBudgetExceeded", "EvaluationMeter", "FailureKind", "feasible", "IterationRecord", "known_tool_ids",
    "Limits", "LLMPlanner", "loosened", "MAX_ALTERNATES", "OPTION_SPECS", "OptionSpec", "Planner", "PlannerState",
    "PlannerUnavailable", "Policy", "PROBLEM_FIELDS", "prompt_state", "Proposal", "PROPOSAL_FIELDS",
    "PROPOSAL_SCHEMA", "RecordedPlanner", "RecordKind", "REMEDIES", "Remedy", "RuleBasedPlanner", "RunOutcome",
    "RunStatus", "screen_cost", "solve", "StopReason", "Trace", "trace_digest", "TRACE_SCHEMA", "validate_proposal",
    "VERIFICATION_SHARE", "VerificationRecord", "Violation", "ViolationCode",
]
