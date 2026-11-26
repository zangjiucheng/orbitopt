"""Typed validation errors shared by every part of the optimization IR."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ErrorCode(str, Enum):
    """What is wrong with a problem. Callers branch on this, never on messages."""

    SCHEMA = "schema"
    INVALID_VALUE = "invalid_value"
    INVALID_NAME = "invalid_name"
    DUPLICATE_NAME = "duplicate_name"
    UNKNOWN_NAME = "unknown_name"
    NO_VARIABLES = "no_variables"
    NO_OBJECTIVES = "no_objectives"
    NON_FINITE = "non_finite"
    BOUNDS_ORDER = "bounds_order"
    INTEGER_BOUNDS = "integer_bounds"
    DOMAIN_FIELDS = "domain_fields"
    EMPTY_CHOICES = "empty_choices"
    INITIAL_OUT_OF_DOMAIN = "initial_out_of_domain"
    MISSING_RHS = "missing_rhs"
    LOCKED_NON_USER = "locked_non_user"
    NON_POSITIVE_BUDGET = "non_positive_budget"
    INVALID_TOLERANCE = "invalid_tolerance"
    UNKNOWN_UNIT = "unknown_unit"
    UNIT_MISMATCH = "unit_mismatch"
    SYNTAX_ERROR = "syntax_error"
    UNSAFE_EXPRESSION = "unsafe_expression"
    UNKNOWN_FUNCTION = "unknown_function"
    WRONG_ARITY = "wrong_arity"
    BAD_LITERAL = "bad_literal"
    EXPONENT_TOO_LARGE = "exponent_too_large"
    EXPRESSION_TOO_COMPLEX = "expression_too_complex"


@dataclass(frozen=True)
class ProblemError:
    """One validation failure: a code, a JSON-pointer path into the dict form, a message."""

    code: ErrorCode
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path or '/'}: [{self.code.value}] {self.message}"


class InvalidProblemError(ValueError):
    """Raised with every validation failure of a problem at once, in ``errors``."""

    def __init__(self, errors):
        self.errors = tuple(errors)
        super().__init__("invalid optimization problem:\n" + "\n".join(f"  {e}" for e in self.errors))
