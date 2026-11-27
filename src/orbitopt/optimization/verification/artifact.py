"""Sealed run directories: write a run's problem, result and evidence with a manifest, and re-grade from files.

A run directory holds ``problem.json`` (the problem the solver ran),
``result.json``, ``evidence.json`` (evidence items and cache provenance) and
``manifest.json``. Each file is canonical JSON (sorted keys, compact, no NaN)
and the manifest records the SHA-256 of each file's exact bytes plus the
problem's and the result's IR digests. ``verify_run`` re-derives everything
from the files: a missing file, a changed byte or a digest that does not
match yields a failed report with the finding, never an exception.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..errors import InvalidProblemError
from ..problem import OptimizationProblem, VerificationPolicy, parse_problem
from ..result import OptimizationResult
from .evidence import EvidenceItem, evidence_from_dict, evidence_to_dict, result_digest
from .guard import Provenance, Tools, verify_result
from .report import Finding, FindingCode, VerificationReport, decide, unassessed_report

RUN_FORMAT = "orbitopt.optimization.run/1"
MANIFEST = "manifest.json"
FILES = ("problem.json", "result.json", "evidence.json")


def canonical_json(data: Any) -> bytes:
    """Sorted keys, compact separators, UTF-8, and no NaN or infinity (``ValueError``)."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False).encode()


def file_digest(data: bytes) -> str:
    """``sha256:<hex>`` of exact bytes."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def seal_run(problem: OptimizationProblem, result: OptimizationResult, evidence: Sequence[EvidenceItem],
             run_dir: str | Path, *, cache_hit: bool = False, cache_key: str | None = None) -> str:
    """Write a sealed run of ``result`` for ``problem`` into ``run_dir`` and return the manifest's digest.

    ``problem`` is the problem the solver ran. Keep the returned digest
    out of band: anyone who can rewrite the directory can rewrite the
    manifest too, and ``verify_run(..., expected_manifest_digest=...)``
    catches that. Raises ``ValueError`` for a non-finite value.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    documents = {
        "problem.json": problem.to_dict(),
        "result.json": result.to_dict(),
        "evidence.json": {"evidence": [evidence_to_dict(e) for e in evidence],
                          "provenance": {"cache_hit": cache_hit, "cache_key": cache_key}},
    }
    encoded = {name: canonical_json(doc) for name, doc in documents.items()}
    manifest = canonical_json({"format": RUN_FORMAT, "files": {n: file_digest(b) for n, b in encoded.items()},
                               "problem_digest": problem.digest(), "result_digest": result_digest(result)})
    for name, data in encoded.items():
        (run_dir / name).write_bytes(data)
    (run_dir / MANIFEST).write_bytes(manifest)
    return file_digest(manifest)


def verify_run(run_dir: str | Path, *, problem: OptimizationProblem | None = None,
               policy: VerificationPolicy | None = None,
               alternate_solvers: Mapping[str, Callable[[OptimizationProblem], OptimizationResult]] | None = None,
               tools: Tools | None = None, expected_manifest_digest: str | None = None,
               seed: int = 0) -> VerificationReport:
    """Grade a sealed run from its files alone.

    ``problem`` is the problem the run should answer; without it the run is
    graded against its own ``problem.json``, which checks consistency but
    lets whoever wrote the directory choose the problem. An alternate-solver
    evidence item runs the solver ``alternate_solvers`` maps its
    ``solver_id`` to. Never raises.
    """
    findings: list[Finding] = []
    required = tuple((policy or (problem.verification if problem else VerificationPolicy())).required)
    try:
        loaded = _load(Path(run_dir), expected_manifest_digest, findings)
        if loaded is None:
            return unassessed_report("the sealed run is incomplete or was modified", findings, required)
        manifest, docs = loaded
        try:
            solved = parse_problem(docs["problem.json"])
        except InvalidProblemError as exc:
            findings += [Finding(FindingCode.INVALID_PROBLEM, f"/problem.json{e.path}", e.message) for e in exc.errors]
            return unassessed_report("problem.json is not a valid problem", findings, required)
        if manifest.get("problem_digest") != solved.digest():
            findings.append(Finding(FindingCode.DIGEST_MISMATCH, "/manifest.json/problem_digest",
                                    "the manifest's problem digest does not match problem.json"))
        evidence_doc = docs["evidence.json"]
        items, provenance = _evidence(evidence_doc, alternate_solvers, findings)
        graded = solved if problem is None else problem
        report = verify_result(graded, docs["result.json"], policy=policy, evidence=items,
                               artifact_digest=manifest.get("result_digest") or "missing",
                               provenance=replace(provenance, solved_problem=solved), tools=tools, seed=seed)
        found = tuple(findings) + report.findings
        return replace(report, findings=found, verdict=decide(report.statuses, report.required, found))
    except Exception as exc:  # reading a run must never turn into a pass
        findings.append(Finding(FindingCode.GUARD_ERROR, "/", repr(exc)))
        return unassessed_report(f"guard error: {exc!r}", findings, required)


def _load(run_dir: Path, expected: str | None, findings: list[Finding]):
    def read(name: str) -> bytes | None:
        try:
            return (run_dir / name).read_bytes()
        except OSError as exc:
            findings.append(Finding(FindingCode.MISSING_FILE, f"/{name}", f"cannot read: {exc.strerror or exc}"))
            return None

    raw = read(MANIFEST)
    if raw is None:
        return None
    if expected is not None and file_digest(raw) != expected:
        findings.append(Finding(FindingCode.DIGEST_MISMATCH, f"/{MANIFEST}",
                                f"manifest digest {file_digest(raw)!r} is not the expected {expected!r}"))
    manifest = _json(raw, MANIFEST, findings)
    if not isinstance(manifest, dict) or manifest.get("format") != RUN_FORMAT \
            or not isinstance(manifest.get("files"), dict):
        findings.append(Finding(FindingCode.MALFORMED_ARTIFACT, f"/{MANIFEST}", f"not a {RUN_FORMAT} manifest"))
        return None
    docs = {}
    for name in FILES:
        data = read(name)
        if data is None:
            continue
        recorded = manifest["files"].get(name)
        if recorded != file_digest(data):
            findings.append(Finding(FindingCode.DIGEST_MISMATCH, f"/{name}",
                                    f"content digest {file_digest(data)!r} does not match the manifest's {recorded!r}"))
            continue
        docs[name] = _json(data, name, findings)
    if findings or any(not isinstance(docs.get(n), dict) for n in FILES):
        if not findings:
            findings.append(Finding(FindingCode.MALFORMED_ARTIFACT, "/", "a run file is not a JSON object"))
        return None
    return manifest, docs


def _json(data: bytes, name: str, findings: list[Finding]) -> Any:
    def reject(token: str) -> Any:
        raise ValueError(f"non-finite number {token}")

    try:
        return json.loads(data.decode("utf-8"), parse_constant=reject)
    except (UnicodeDecodeError, ValueError) as exc:
        findings.append(Finding(FindingCode.MALFORMED_ARTIFACT, f"/{name}", f"not valid JSON: {exc}"))
        return None


def _evidence(doc: dict, alternate_solvers, findings: list[Finding]) -> tuple[list[EvidenceItem], Provenance]:
    items = []
    raw_items = doc.get("evidence", [])
    for i, data in enumerate(raw_items if isinstance(raw_items, list) else []):
        try:
            items.append(evidence_from_dict(data, alternate_solvers))
        except ValueError as exc:
            findings.append(Finding(FindingCode.MALFORMED_ARTIFACT, f"/evidence.json/evidence/{i}", str(exc)))
    if not isinstance(raw_items, list):
        findings.append(Finding(FindingCode.MALFORMED_ARTIFACT, "/evidence.json/evidence", "not a list"))
    prov = doc.get("provenance", {})
    hit, key = (prov.get("cache_hit", False), prov.get("cache_key")) if isinstance(prov, dict) else (None, None)
    if not isinstance(hit, bool) or not (key is None or isinstance(key, str)):
        findings.append(Finding(FindingCode.MALFORMED_ARTIFACT, "/evidence.json/provenance", "malformed provenance"))
        hit, key = True, None  # an unreadable cache record is treated as a hit on an unknown entry
    return items, Provenance(cache_hit=hit, cache_key=key)
