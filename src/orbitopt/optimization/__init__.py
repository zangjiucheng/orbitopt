"""Optimization IR (E14-01): one typed, serializable, digestable problem description.

The contract between natural-language intake (E14-02), solvers (E14-03),
scientific tools (E14-04), the agent runtime (E14-05) and the verification
guard (E14-06). See ``docs/optimization/ir.md`` and ADR optimization-0001.

The IR is imported with the package. The classifier, the solver registry and
the verification guard are re-exported lazily (PEP 562): ``from
orbitopt.optimization import default_registry, verify_result`` works, and
loads those modules only when a name is first touched. No re-export imports
numpy or a solver backend; a solver imports its backend when it runs.
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
    ToolCallError,
)
from .problem import (
    DOMAIN_VIOLATIONS,
    INDEPENDENT_METHODS,
    MAX_ENUMERATION_POINTS,
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
    to_plain,
    tolerance_scale,
    validate_problem,
)
from .result import (
    EVIDENCE_KINDS,
    RESULT_SCHEMA,
    RESULT_SCHEMAS,
    VOLATILE_KEYS,
    Evidence,
    OptimizationResult,
    ParetoPoint,
    Provenance,
    Termination,
)
from .units import SYMBOLS, Unit, UnitError, parse_unit

__all__ = [
    "Budget", "Constraint", "DOMAIN_VIOLATIONS", "EVIDENCE_KINDS", "ErrorCode", "Evaluation", "EvaluationError",
    "Evidence", "Expression", "ExpressionError", "FUNCTIONS", "INDEPENDENT_METHODS", "InvalidProblemError",
    "MAX_ENUMERATION_POINTS", "MAX_EXPANDED_NODES", "MAX_NODES", "Objective", "OptimizationProblem",
    "OptimizationResult", "PROBLEM_SCHEMA", "PROBLEM_SCHEMAS", "Parameter", "ParetoPoint", "ProblemError",
    "Provenance", "RESERVED_NAMES", "RESULT_SCHEMA", "RESULT_SCHEMAS", "SCALAR_FUNCTIONS", "SCHEMA_VERSION",
    "STATUSES", "SYMBOLS", "Termination", "ToolCall", "ToolCallError", "Unit", "UnitError", "VOLATILE_KEYS",
    "Variable", "VerificationPolicy", "canonical_digest", "load_schema", "parse_problem", "parse_unit",
    "schema_errors", "schema_for", "to_plain", "tolerance_scale", "validate_problem",
]

_CLASSIFY = ("Form", "ProblemClass", "ProblemStructure", "classify", "linear_coefficients")
_SOLVERS = ("CONVERGENCE_MEANINGS", "OptionSpec", "Solver", "SolverCapability", "SolverRegistry", "default_registry")
_VERIFICATION = ("AlternateSolverCheck", "AnalyticalOptimum", "Finding", "FindingCode", "LowerBoundCertificate",
                 "ReferenceSolution", "StatusOutcome", "VerificationReport", "compare_formulations", "result_digest",
                 "seal_run", "verify_result", "verify_run")
__all__ += list(_CLASSIFY) + list(_SOLVERS) + list(_VERIFICATION)


def __getattr__(name):
    import importlib

    for module, names in ((".classify", _CLASSIFY), (".solvers", _SOLVERS), (".verification", _VERIFICATION)):
        if name in names:
            return getattr(importlib.import_module(module, __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
