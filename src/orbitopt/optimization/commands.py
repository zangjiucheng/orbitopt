"""What ``orbitopt optimization validate`` and ``verify`` compute: paths in, JSON-ready summaries out.

The CLI only parses arguments and prints. Nothing here imports numpy or a
solver backend until a solver actually runs, and nothing imports the agent
package. ``orbitopt.agent.commands.verify_run_dir`` adds agent run
directories on top of ``verify_sealed_run``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .classify import classify
from .errors import ErrorCode, InvalidProblemError, ProblemError
from .problem import OptimizationProblem, parse_problem
from .solvers import SolverRegistry, default_registry
from .verification import Outcome, verify_run


class CommandError(ValueError):
    """A request the command cannot act on: a missing file, a malformed argument. The CLI exits 2."""


def read_json(path: str | Path) -> Any:
    """The JSON document at ``path``; ``CommandError`` when it cannot be read or parsed."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise CommandError(f"{path}: cannot read: {exc.strerror or exc}") from None
    except ValueError as exc:
        raise CommandError(f"{path}: not valid JSON: {exc}") from None


def load_problem(path: str | Path) -> OptimizationProblem:
    """The validated problem at ``path``; ``CommandError`` listing every error otherwise."""
    try:
        return parse_problem(read_json(path))
    except InvalidProblemError as exc:
        raise CommandError(f"{path}: not a valid problem:\n" + "\n".join(f"  {e}" for e in exc.errors)) from None


def error_dict(error: ProblemError) -> dict:
    return {"code": error.code.value, "path": error.path, "message": error.message}


def validate_file(path: str | Path, registry: SolverRegistry | None = None) -> dict:
    """Validate the IR document at ``path``: its errors, or its digest, structure and the solvers that take it.

    ``valid`` is false for a file that is not JSON (one ``schema`` error at
    ``/``) as for one that fails the schema or the semantic checks.
    ``compatible_solvers`` are the installed solvers that can take it, most
    specialised first; ``unavailable_solvers`` those that could take it but
    whose backend is missing.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise CommandError(f"{path}: cannot read: {exc.strerror or exc}") from None
    except ValueError as exc:
        return {"path": str(path), "valid": False, "errors": [{"code": ErrorCode.SCHEMA.value, "path": "",
                                                               "message": f"not valid JSON: {exc}"}]}
    try:
        problem = parse_problem(data)
    except InvalidProblemError as exc:
        return {"path": str(path), "valid": False, "errors": [error_dict(e) for e in exc.errors]}
    registry = registry or default_registry()
    structure = classify(problem)
    capable = [c for c in registry.capabilities() if c.rejection(structure) is None]
    return {
        "path": str(path), "valid": True, "errors": [],
        "id": problem.id, "schemaVersion": data.get("schemaVersion"), "digest": problem.digest(),
        "structure": {"class": structure.problem_class.value, "form": structure.form.name.lower(),
                      "variables": len(problem.variables), "objectives": len(problem.objectives),
                      "constraints": len(problem.constraints), "integer": structure.integer,
                      "equality": structure.equality, "bounded": structure.bounded},
        "compatible_solvers": [s.capability.solver_id for s in registry.compatible(problem)],
        "unavailable_solvers": {c.solver_id: c.unavailable_reason for c in capable if not c.available},
    }


def verify_sealed_run(run_dir: str | Path, *, registry: SolverRegistry | None = None,
                      expected_manifest_digest: str | None = None) -> dict:
    """Re-grade a sealed run (``orbitopt.optimization.run/2``) from its files with the verification guard.

    An alternate-solver evidence item runs the registry's solver of that id
    again, on the problem's own budget and seed 0.
    """
    registry = registry or default_registry()

    def alternate(solver_id: str):
        return lambda problem: registry.get(solver_id).solve(problem)

    alternates = {c.solver_id: alternate(c.solver_id) for c in registry.capabilities()}
    report = verify_run(run_dir, alternate_solvers=alternates, expected_manifest_digest=expected_manifest_digest)
    return {"run_dir": str(run_dir), "kind": "sealed", "verified": report.verdict is Outcome.PASSED,
            "integrity": [], "recorded_status": None, "report": report.to_dict()}
