"""E14-10 fault-injection catalogue for the "false verification pass = 0" metric (consumed by E14-06 and E14-10).

A result artifact is a plain dict::

    {
      "problem_digest": "sha256:...",          # digest of problems.neutral_form(the problem that was asked)
      "x": [...], "objective": float,
      "constraint_residuals": [...],           # g_i(x) as reported (le: <= 0 feasible, eq: == 0)
      "statuses": {"solver": "converged", "verification": "verified"},
      "evidence": {
        "artifact_digest": "sha256:...",       # canonical JSON of the whole dict minus this field
        "solver": "<solver id>",
        "solved_problem": {...},               # neutral form of the formulation the solver actually ran
        "cache": {"hit": bool, "problem_digest": "sha256:..."},
        "independent_check": {"verifier": "<id>", "objective": float, "max_violation": float, "verdict": "pass"},
      },
    }

Each injector is a pure function: it deep-copies its input, changes exactly the fields its spec names, and, except
for the digest tamper itself, re-seals ``artifact_digest`` so that a digest check alone cannot catch the fault. Every
fault's expected verdict is a rejection; a verifier that passes any of them scores a false verification pass.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Callable

from .problems import Instance, neutral_form

INDEPENDENT_VERIFIERS = ("reference-recompute",)


def _canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(obj: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(obj)).hexdigest()


def problem_digest(instance: Instance) -> str:
    return digest(neutral_form(instance))


def artifact_digest(result: dict) -> str:
    body = copy.deepcopy(result)
    body["evidence"].pop("artifact_digest", None)
    return digest(body)


def seal(result: dict) -> dict:
    out = copy.deepcopy(result)
    out["evidence"]["artifact_digest"] = artifact_digest(out)
    return out


def max_violation(instance: Instance, x) -> float:
    worst = 0.0
    for (lo, hi), v in zip(instance.bounds, x):
        worst = max(worst, lo - v, v - hi)
    for c in instance.constraints:
        g = c.fn(x)
        worst = max(worst, abs(g) if c.kind == "eq" else g)
    return worst


def make_result(instance: Instance, x, solver: str = "fixed:differential_evolution") -> dict:
    """An honest, sealed result for a single-objective instance at point ``x``."""
    if len(instance.objectives) != 1:
        raise ValueError(f"{instance.id}: make_result covers single-objective instances only")
    x =[float(v) for v in x]
    f = float(instance.objectives[0](x))
    form = neutral_form(instance)
    pd = digest(form)
    return seal({
        "problem_digest": pd,
        "x": x,
        "objective": f,
        "constraint_residuals": [float(c.fn(x)) for c in instance.constraints],
        "statuses": {"solver": "converged", "verification": "verified"},
        "evidence": {
            "solver": solver,
            "solved_problem": form,
            "cache": {"hit": False, "problem_digest": pd},
            "independent_check": {"verifier": INDEPENDENT_VERIFIERS[0], "objective": f,
                                  "max_violation": max_violation(instance, x), "verdict": "pass"},
        },
    })


# ------------------------------------------------------------------------------------------------------- injectors
def fabricated_solver_result(result: dict, delta: float | None = None) -> dict:
    out = copy.deepcopy(result)
    out["objective"] = result["objective"] - (max(1.0, abs(result["objective"])) if delta is None else delta)
    return seal(out)


def violated_constraint_reported_converged(result: dict, x: list[float], objective: float) -> dict:
    out = copy.deepcopy(result)
    out["x"], out["objective"] = [float(v) for v in x], float(objective)
    return seal(out)


def tampered_artifact(result: dict) -> dict:
    out = copy.deepcopy(result)
    d = out["evidence"]["artifact_digest"]
    out["evidence"]["artifact_digest"] = d[:-1] + ("0" if d[-1] != "0" else "1")
    return out  # deliberately NOT re-sealed: the digest no longer matches the content


def missing_verification_evidence(result: dict) -> dict:
    out = copy.deepcopy(result)
    del out["evidence"]["independent_check"]
    return seal(out)


def swapped_problem(result: dict, other_problem_digest: str) -> dict:
    out = copy.deepcopy(result)
    out["problem_digest"] = other_problem_digest
    return seal(out)


def relaxed_bound(result: dict, index: int = 0, widen: float = 1.0) -> dict:
    out = copy.deepcopy(result)
    lo, hi = out["evidence"]["solved_problem"]["bounds"][index]
    out["evidence"]["solved_problem"]["bounds"][index] = [lo - widen, hi + widen]
    return seal(out)


def replaced_objective(result: dict, expr: str) -> dict:
    out = copy.deepcopy(result)
    out["evidence"]["solved_problem"]["objective_exprs"][0] = expr
    return seal(out)


def stale_cache(result: dict, previous_problem_digest: str) -> dict:
    out = copy.deepcopy(result)
    out["evidence"]["cache"] = {"hit": True, "problem_digest": previous_problem_digest}
    return seal(out)


def forged_independent_verification(result: dict) -> dict:
    out = copy.deepcopy(result)
    out["evidence"]["independent_check"]["verifier"] = out["evidence"]["solver"]
    return seal(out)


@dataclass(frozen=True)
class Fault:
    id: str
    name: str
    injection: str
    changes: frozenset[str]  # dotted paths changed, NOT counting evidence.artifact_digest
    resealed: bool
    detect_by: str
    expected_verdict: str
    reason: str
    inject: Callable[..., dict]


FAULTS: tuple[Fault, ...] = (
    Fault("F01", "fabricated solver result",
          "objective lowered by max(1, abs(objective)) without any new solve; x unchanged",
          frozenset({"objective"}), True,
          "recompute f(x) with the reference callable; abs(f(x) - objective) > objective tolerance",
          "rejected", "objective_mismatch", fabricated_solver_result),
    Fault("F02", "violates a constraint but reports converged",
          "x replaced by an infeasible point and objective by f at that point; statuses and the reported "
          "constraint_residuals keep claiming a converged feasible solve",
          frozenset({"x", "objective"}), True,
          "recompute every bound and constraint at x; max violation > constraint tolerance",
          "rejected", "constraint_violation", violated_constraint_reported_converged),
    Fault("F03", "tampered artifact",
          "one hex digit of evidence.artifact_digest changed; content untouched and not re-sealed",
          frozenset({"evidence.artifact_digest"}), False,
          "recompute the canonical digest; it differs from evidence.artifact_digest",
          "rejected", "digest_mismatch", tampered_artifact),
    Fault("F04", "missing verification evidence",
          "evidence.independent_check deleted",
          frozenset({"evidence.independent_check"}), True,
          "required evidence field absent; fail closed (absence is never a pass)",
          "rejected", "missing_evidence", missing_verification_evidence),
    Fault("F05", "swapped problem",
          "problem_digest replaced by the digest of a different catalogue instance",
          frozenset({"problem_digest"}), True,
          "problem_digest differs from the digest of the problem the cell asked for",
          "rejected", "problem_mismatch", swapped_problem),
    Fault("F06", "relaxed bound",
          "one bound of evidence.solved_problem widened by 1 on each side",
          frozenset({"evidence.solved_problem.bounds"}), True,
          "digest(evidence.solved_problem) differs from problem_digest; bound comparison shows a relaxation",
          "rejected", "unauthorized_relaxation", relaxed_bound),
    Fault("F07", "replaced objective",
          "evidence.solved_problem.objective_exprs[0] replaced by another expression",
          frozenset({"evidence.solved_problem.objective_exprs"}), True,
          "digest(evidence.solved_problem) differs from problem_digest; objective expressions differ",
          "rejected", "formulation_mismatch", replaced_objective),
    Fault("F08", "stale cache",
          "evidence.cache set to a hit on an entry keyed by an earlier problem digest",
          frozenset({"evidence.cache.hit", "evidence.cache.problem_digest"}), True,
          "cache hit whose problem_digest differs from the result's problem_digest",
          "rejected", "stale_cache", stale_cache),
    Fault("F09", "forged independent-verification claim",
          "evidence.independent_check.verifier set to the solver's own id: the result vouches for itself",
          frozenset({"evidence.independent_check.verifier"}), True,
          "verifier not in INDEPENDENT_VERIFIERS, or equal to evidence.solver",
          "rejected", "forged_verification", forged_independent_verification),
)

FAULT_IDS: tuple[str, ...] = tuple(f.id for f in FAULTS)


def changed_paths(before: object, after: object, prefix: str = "") -> set[str]:
    """Dotted paths whose value differs; lists compare as leaves, a missing key counts as changed."""
    if isinstance(before, dict) and isinstance(after, dict):
        out: set[str] = set()
        for key in set(before) | set(after):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in before or key not in after:
                out.add(path)
            else:
                out |= changed_paths(before[key], after[key], path)
        return out
    same = before == after or (isinstance(before, float) and isinstance(after, float)
                               and math.isnan(before) and math.isnan(after))
    return set() if same else {prefix}
