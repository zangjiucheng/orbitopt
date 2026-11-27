"""Agent iteration runtime (E14-05): a bounded, verified, replayable optimization loop over an IR problem.

``solve(problem, planner=...)`` lets a planner choose solvers, options, start
points and seeds, never the problem. Every proposal is validated against the
``Policy`` and the solver registry before it runs, every candidate is graded
by the E14-06 guard, and the final answer is verified with alternate solvers
as independent evidence. ``save_run`` writes a run directory and ``replay``
re-executes it without an LLM call. See ``docs/agent/runtime.md``.
"""

from .model import (TRACE_SCHEMA, Action, AlternateRun, BudgetUse, Decision, Diagnosis, FailureKind, IterationRecord,
                    Limits, Policy, Proposal, RecordKind, Remedy, RunOutcome, RunStatus, StopReason, Trace,
                    VerificationRecord, Violation, ViolationCode, trace_digest)
from .validation import OPTION_SPECS, PROBLEM_FIELDS, PROPOSAL_FIELDS, OptionSpec, validate_proposal

__all__ = [
    "Action", "AlternateRun", "BudgetUse", "Decision", "Diagnosis", "FailureKind", "IterationRecord", "Limits",
    "OPTION_SPECS", "OptionSpec", "Policy", "PROBLEM_FIELDS", "Proposal", "PROPOSAL_FIELDS", "RecordKind", "Remedy",
    "RunOutcome", "RunStatus", "StopReason", "Trace", "trace_digest", "TRACE_SCHEMA", "validate_proposal",
    "VerificationRecord", "Violation", "ViolationCode",
]
