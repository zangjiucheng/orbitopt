"""Serve a prior verified result for a solve that was already run, and only while it still verifies.

A cache entry is the step that produced a verified run's best result, under
``cache_key``. ``lookup`` walks the entries of a key, newest first. An entry
counts only when its record still reads, every artifact still hashes to its
digest, and the guard re-grades the stored sealed run against the problem
asked with no fatal finding and its constraints satisfied. Every other
entry is skipped and named in ``ignored``; nothing stale or corrupt is
returned.

A hit is marked twice: ``CacheLookup.hit`` and the result's
``Provenance.cache_key``, which names the problem digest the entry was
stored under. A hit costs no new solver evaluations (``evaluations == 0``),
as the runtime's meter treats a documented cache hit. The guard's re-grade
evaluates the stored point once more; ``check_evaluations`` counts it.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from orbitopt.agent.runtime import EvaluationMeter, screen_cost
from orbitopt.optimization import OptimizationProblem, OptimizationResult, Provenance, parse_problem
from orbitopt.optimization.verification import Outcome, result_digest, verify_run

from .index import ExperimentDB
from .ingest import SEALED_FILES, orbitopt_version
from .model import ArtifactError, ExperimentError, ExperimentRecord, Integrity, VersionPolicy, cache_key
from .store import safe_relative


class IgnoredReason(str, Enum):
    """Why an entry under the key was not served."""

    CORRUPT = "corrupt"
    """The record or an artifact no longer reads, or no longer hashes to its digest."""
    STALE = "stale"
    """The artifacts are intact but no longer verify for this problem (another problem, a changed point)."""


@dataclass(frozen=True)
class Ignored:
    run_id: str
    reason: IgnoredReason
    detail: str


@dataclass(frozen=True)
class CacheLookup:
    """What ``lookup`` found. ``result`` is set only on a hit; ``ignored`` names every entry skipped."""

    key: str
    result: OptimizationResult | None = None
    run_id: str | None = None
    recorded_evaluations: int = 0
    check_evaluations: int = 0
    ignored: tuple[Ignored, ...] = ()

    @property
    def hit(self) -> bool:
        return self.result is not None

    @property
    def evaluations(self) -> int:
        """New solver evaluations this answer cost: none, hit or miss."""
        return 0


def lookup(db: ExperimentDB, problem: OptimizationProblem, *, solver_id: str,
           options: Mapping[str, Any] | None = None, seed: int = 0, x0: Mapping[str, float] | None = None,
           evaluations: int, version_policy: VersionPolicy = VersionPolicy.EXACT, version: str | None = None,
           tools: Any = None) -> CacheLookup:
    """The verified result of an earlier identical solve of ``problem``, if one is stored and still verifies.

    ``solver_id``, ``options``, ``seed``, ``x0`` and ``evaluations`` are the
    solve's decision. ``version`` defaults to the installed orbitopt.
    ``tools`` are the resolvers the guard needs for a ``ToolCall``.
    """
    key = cache_key(problem.digest(), solver_id, options, seed, x0, evaluations,
                    version=version or orbitopt_version(), policy=version_policy)
    ignored: list[Ignored] = []
    checked = 0
    for run_id in db.cached_runs(key):
        try:
            record = db.get(run_id)
            stored = _sealed(db, record)
        except ExperimentError as exc:
            ignored.append(Ignored(run_id, IgnoredReason.CORRUPT, str(exc)))
            continue
        meter = EvaluationMeter(problem, screen_cost(problem))
        reason = _stale(record, stored, problem, meter, tools)
        checked += meter.count
        if reason is not None:
            ignored.append(Ignored(run_id, IgnoredReason.STALE, reason))
            continue
        result = replace(stored["result"], provenance=Provenance(cache_key=problem.digest()))
        return CacheLookup(key, result, run_id, record.usage.evaluations, checked, tuple(ignored))
    return CacheLookup(key, check_evaluations=checked, ignored=tuple(ignored))


def _sealed(db: ExperimentDB, record: ExperimentRecord) -> dict[str, Any]:
    """The sealed run's files from the store, every one through its digest check."""
    if not record.verified or record.integrity is not Integrity.REPLAYED or record.sealed_digest is None:
        raise ArtifactError("the record is not a verified, replayed run with a sealed result")
    folder = next((name.rsplit("/", 1)[0] for name, digest in record.files.items()
                   if name.endswith("/manifest.json") and digest == record.sealed_digest), None)
    if folder is None:
        raise ArtifactError("the record names no sealed manifest")
    data = {}
    for name in SEALED_FILES:
        digest = record.files.get(f"{folder}/{name}")
        if digest is None:
            raise ArtifactError(f"the record names no {folder}/{name}")
        data[name] = db.store.get(digest)
    return {"files": data, "result": OptimizationResult.from_dict(json.loads(data["result.json"])),
            "problem": parse_problem(json.loads(data["problem.json"]))}


def _stale(record: ExperimentRecord, stored: Mapping[str, Any], problem: OptimizationProblem,
           meter: EvaluationMeter, tools: Any) -> str | None:
    if record.problem_digest != problem.digest() or stored["problem"].digest() != problem.digest():
        return "the entry answers another problem"
    if result_digest(stored["result"]) != record.result_digest:
        return "the stored result is not the result the record names"
    with tempfile.TemporaryDirectory(prefix="orbitopt-cache-") as tmp:
        for name, data in stored["files"].items():
            (Path(tmp) / safe_relative(name)).write_bytes(data)
        policy = replace(problem.verification, required=("constraints_satisfied",))
        report = verify_run(tmp, problem=meter.problem, policy=policy, tools=tools,
                            expected_manifest_digest=record.sealed_digest)
    if report.verdict is not Outcome.PASSED:
        codes = sorted(f.code.value for f in report.findings if f.fatal) or ["constraints_satisfied not passed"]
        return f"the guard re-grades the stored run {report.verdict.value}: {', '.join(codes)}"
    return None
