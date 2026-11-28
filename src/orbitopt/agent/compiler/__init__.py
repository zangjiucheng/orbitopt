"""Natural-language problem compiler (E14-02): a statement in, a validated IR problem or a stated refusal out.

The model fills the small ``COMPILER_OUTPUT_SCHEMA`` through structured
output; the compiler builds and validates the IR itself and checks the result
against the user's text and, for a revision, against the base problem. See
``docs/agent/compiler.md``.
"""

from __future__ import annotations

from .authorization import PromptText, Statement, extract_statements, split_prompt, uncovered_statements
from .compiler import MAX_REPAIR_ROUNDS, UNSUPPORTED_MARKERS, compile_problem
from .outcome import (
    Assumption,
    CompileFailed,
    Compiled,
    CompileOutcome,
    CompilerErrorCode,
    CompilerIssue,
    NeedsClarification,
    Question,
    RejectedUnauthorizedChange,
    Unsupported,
    UnsupportedReason,
    Violation,
)
from .prompts import SYSTEM_PROMPT
from .schema import COMPILER_OUTPUT_SCHEMA, AuthorizedChange, CompilerOutput

__all__ = [
    "Assumption", "AuthorizedChange", "COMPILER_OUTPUT_SCHEMA", "CompileFailed", "CompileOutcome", "Compiled",
    "CompilerErrorCode", "CompilerIssue", "CompilerOutput", "MAX_REPAIR_ROUNDS", "NeedsClarification",
    "PromptText", "Question", "RejectedUnauthorizedChange", "SYSTEM_PROMPT", "Statement", "UNSUPPORTED_MARKERS",
    "Unsupported", "UnsupportedReason", "Violation", "compile_problem", "extract_statements", "split_prompt",
    "uncovered_statements",
]
