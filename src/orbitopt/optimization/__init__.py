"""Optimization IR (E14-01): one typed, serializable, digestable problem description.

The contract between natural-language intake (E14-02), solvers (E14-03),
scientific tools (E14-04), the agent runtime (E14-05) and the verification
guard (E14-06). See ``docs/optimization/ir.md`` and ADR optimization-0001.
"""

from .errors import ErrorCode, InvalidProblemError, ProblemError
from .expressions import (
    FUNCTIONS,
    MAX_EXPANDED_NODES,
    MAX_NODES,
    RESERVED_NAMES,
    SCALAR_FUNCTIONS,
    EvaluationError,
    Expression,
    ExpressionError,
    ToolCall,
)
from .problem import (
    INDEPENDENT_METHODS,
    PROBLEM_SCHEMA,
    PROBLEM_SCHEMAS,
    SCHEMA_VERSION,
    STATUSES,
    Budget,
    Constraint,
    Evaluation,
    Objective,
    OptimizationProblem,
    Parameter,
    Variable,
    VerificationPolicy,
    canonical_digest,
    load_schema,
    parse_problem,
    schema_errors,
    schema_for,
    validate_problem,
)
from .result import EVIDENCE_KINDS, RESULT_SCHEMA, RESULT_SCHEMAS, Evidence, OptimizationResult, Termination
from .units import SYMBOLS, Unit, UnitError, parse_unit

__all__ = [
    "Budget", "Constraint", "EVIDENCE_KINDS", "ErrorCode", "Evaluation", "EvaluationError", "Evidence",
    "Expression", "ExpressionError", "FUNCTIONS", "INDEPENDENT_METHODS", "InvalidProblemError", "MAX_EXPANDED_NODES",
    "MAX_NODES", "Objective", "RESERVED_NAMES", "SCALAR_FUNCTIONS",
    "OptimizationProblem", "OptimizationResult", "PROBLEM_SCHEMA", "PROBLEM_SCHEMAS", "Parameter", "ProblemError",
    "RESULT_SCHEMA", "RESULT_SCHEMAS", "SCHEMA_VERSION", "STATUSES", "SYMBOLS", "Termination", "ToolCall", "Unit",
    "UnitError", "Variable", "VerificationPolicy", "canonical_digest", "load_schema", "parse_problem", "parse_unit",
    "schema_errors", "schema_for", "validate_problem",
]
