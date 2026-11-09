"""Candidate Selection and the Verification Gate (task #171).

Two explicit steps of the unified mission pipeline (orbitopt.pipeline) that
sit between the cheap search/optimize stages and the high-fidelity verify
stage:

* **Candidate Selection** (``select_candidates``) -- turns the optimizer's
  raw candidate list into the short list high-fidelity refinement walks
  through: drops infeasible (non-finite cost) candidates, removes
  near-duplicates (within ``tolerance`` of a better one in every
  coordinate), ranks by cost and, when the list must be cut to
  ``max_candidates``, keeps it *diverse* -- candidates at least
  ``min_separation`` away from every better pick are taken first, so a
  failed verification of the best one isn't retried on a near-identical
  trajectory. Every input candidate gets a record with its status and the
  reason it was kept or dropped.

* **Verification Gate** (``VerificationGate``) -- records, per candidate,
  what happened downstream: ``rejected`` by refinement (with the reason),
  ``accepted`` (its tri-state verification report result is ``passed``),
  or ``degraded`` (flown, but verification ``failed`` or ``not_checked`` --
  e.g. a failed check, or verification disabled). Only an ``accepted``
  candidate may be reported as a success. When nothing passed and a
  candidate is exported anyway, the export is ``best_effort`` and carries
  a prominent warning (``BEST_EFFORT_WARNING``) in the gate document, the
  scene (``warnings`` + first subtitle line), the CLI output and the log.

Pure stdlib (plus orbitopt.verify.report's tri-state helpers), so the
bookkeeping is testable without the compute stack.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Candidate Selection record statuses.
SEL_SELECTED = "selected"
SEL_DUPLICATE = "duplicate"
SEL_INFEASIBLE = "infeasible"
SEL_TRUNCATED = "truncated"

# Verification Gate verdicts.
VERDICT_ACCEPTED = "accepted"  # verification passed
VERDICT_DEGRADED = "degraded"  # flown, but verification failed / not_checked
VERDICT_REJECTED = "rejected"  # refinement / high-fidelity flight did not hold up
VERDICT_NOT_EVALUATED = "not_evaluated"  # selected, but never flown (an earlier one won)
VERDICTS = (VERDICT_ACCEPTED, VERDICT_DEGRADED, VERDICT_REJECTED, VERDICT_NOT_EVALUATED)

BEST_EFFORT_WARNING = ("BEST-EFFORT EXPORT: no candidate passed verification -- this trajectory is "
                       "NOT verified and must not be used as a validated solution.")


def _finite(v) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


@dataclass
class CandidateSelection:
    """Outcome of ``select_candidates``."""

    selected: list  # the kept items, best (lowest cost) first
    selected_indices: list  # their indices in the input list
    records: list  # one dict per input candidate, in input order

    def to_candidates(self) -> list:
        """Records ordered for a StageResult.candidates list (selected first)."""
        order = {SEL_SELECTED: 0, SEL_TRUNCATED: 1, SEL_DUPLICATE: 2, SEL_INFEASIBLE: 3}
        return sorted(self.records, key=lambda r: (order[r["status"]], r.get("rank") if r.get("rank") is not None
                                                   else math.inf, r["index"]))

    def counts(self) -> dict:
        out = {s: 0 for s in (SEL_SELECTED, SEL_DUPLICATE, SEL_INFEASIBLE, SEL_TRUNCATED)}
        for r in self.records:
            out[r["status"]] += 1
        return out


def select_candidates(
    items: Sequence,
    *,
    cost: Callable,
    coords: Callable,
    tolerance: Sequence[float],
    min_separation: Sequence[float] | None = None,
    max_candidates: int | None = None,
    describe: Callable | None = None,
) -> CandidateSelection:
    """Deduplicate, rank and diversity-filter ``items``.

    ``cost(item)`` -> float (lower is better; non-finite = infeasible).
    ``coords(item)`` -> tuple of floats the distance tests use.
    ``tolerance`` -> per-coordinate duplicate tolerance: an item within it
    (in every coordinate) of a better kept item is a duplicate.
    ``min_separation`` -> per-coordinate diversity scale: an item is
    *diverse* when it is at least one separation ellipse away from every
    better kept item. With ``max_candidates``, diverse items fill the list
    first, the remaining slots go to the best non-diverse ones.
    ``describe(item)`` -> dict of extra fields copied into each record.
    """
    tol = [float(t) for t in tolerance]
    sep = None if min_separation is None else [float(s) for s in min_separation]
    records = []
    for i, item in enumerate(items):
        rec = {"index": i, "cost": float(cost(item)) if _finite(cost(item)) else None,
               "coords": [float(c) for c in coords(item)], "rank": None, "status": None,
               "reason": "", "selected": False}
        if describe is not None:
            rec.update(describe(item))
        records.append(rec)

    feasible = []
    for rec in records:
        if rec["cost"] is None:
            rec["status"], rec["reason"] = SEL_INFEASIBLE, "non-finite cost (optimizer found no feasible solution)"
        else:
            feasible.append(rec)
    feasible.sort(key=lambda r: (r["cost"], r["index"]))

    def within(a, b, scale):
        return all(abs(x - y) <= s for x, y, s in zip(a["coords"], b["coords"], scale))

    def sep_dist(a, b):
        return math.sqrt(sum(((x - y) / s) ** 2 for x, y, s in zip(a["coords"], b["coords"], sep) if s > 0))

    unique = []
    for rec in feasible:
        dup = next((u for u in unique if within(rec, u, tol)), None)
        if dup is not None:
            rec["status"], rec["duplicate_of"] = SEL_DUPLICATE, dup["index"]
            rec["reason"] = (f"duplicate of candidate #{dup['index']} (within tolerance "
                             f"{tol}, cost {rec['cost']:.6g} >= {dup['cost']:.6g})")
        else:
            unique.append(rec)
    for rank, rec in enumerate(unique):
        rec["rank"] = rank
        rec["diverse"] = sep is None or all(sep_dist(rec, u) >= 1.0 for u in unique[:rank])

    n_keep = len(unique) if max_candidates is None else max(0, int(max_candidates))
    keep = [r for r in unique if r["diverse"]][:n_keep]
    keep += [r for r in unique if not r["diverse"]][:n_keep - len(keep)]
    keep_ids = {id(r) for r in keep}
    for rec in unique:
        if id(rec) in keep_ids:
            rec["status"], rec["selected"] = SEL_SELECTED, True
            rec["reason"] = (f"rank {rec['rank']} by cost" + ("" if rec["diverse"] else
                             "; close to a better candidate but kept to fill the list"))
        else:
            rec["status"] = SEL_TRUNCATED
            rec["reason"] = (f"rank {rec['rank']} by cost, beyond max_candidates={n_keep}"
                             + ("" if rec["diverse"] else
                                " (within min_separation of a better candidate -- diversity kept others)"))
    keep.sort(key=lambda r: r["rank"])
    return CandidateSelection(
        selected=[items[r["index"]] for r in keep],
        selected_indices=[r["index"] for r in keep],
        records=records,
    )


def verification_outcome(reports: Sequence) -> tuple[str, list]:
    """Tri-state result of a candidate's verification reports plus the
    reasons it isn't ``passed`` (failed checks, not-checked reports)."""
    from orbitopt.verify.report import RESULT_FAILED, RESULT_NOT_CHECKED, RESULT_PASSED, status_result

    if not reports:
        return RESULT_NOT_CHECKED, ["no verification report was produced"]
    results, reasons = [], []
    for r in reports:
        status = r.get("status") or {}
        res = status_result(status)
        results.append(res)
        name = f"{r.get('kind', '?')}:{r.get('label', '?')}"
        if res == RESULT_FAILED:
            failed = [k for k, ok in (status.get("checks") or {}).items() if not ok]
            reasons.append(f"{name} FAILED" + (f" ({', '.join(failed)})" if failed else ""))
        elif res == RESULT_NOT_CHECKED:
            reasons.append(f"{name} NOT CHECKED")
    if RESULT_FAILED in results:
        return RESULT_FAILED, reasons
    if RESULT_NOT_CHECKED in results:
        return RESULT_NOT_CHECKED, reasons
    return RESULT_PASSED, reasons


@dataclass
class VerificationGate:
    """Per-candidate verdicts between refinement/verification and export."""

    candidates: list = field(default_factory=list)
    exported: int | None = None
    best_effort: bool = False
    notes: list = field(default_factory=list)

    def _entry(self, index, info) -> dict:
        for c in self.candidates:
            if c["index"] == index:
                c.update(info or {})
                return c
        c = {"index": index, **(info or {}), "verdict": VERDICT_NOT_EVALUATED,
             "verification": None, "reasons": []}
        self.candidates.append(c)
        return c

    def register(self, index, **info) -> dict:
        """Declare a candidate that reached the gate's short list."""
        return self._entry(index, info)

    def reject(self, index, reason: str, **info) -> dict:
        c = self._entry(index, info)
        c["verdict"] = VERDICT_REJECTED
        c["reasons"].append(str(reason))
        return c

    def evaluate(self, index, reports: Sequence, **info) -> dict:
        """Gate a flown candidate on its verification reports."""
        from orbitopt.verify.report import RESULT_PASSED

        c = self._entry(index, info)
        result, reasons = verification_outcome(reports)
        c["verification"] = result
        c["reports"] = [f"{r.get('kind', '?')}:{r.get('label', '?')}" for r in reports]
        c["verdict"] = VERDICT_ACCEPTED if result == RESULT_PASSED else VERDICT_DEGRADED
        c["reasons"].extend(reasons)
        return c

    def accepted(self) -> list:
        return [c for c in self.candidates if c["verdict"] == VERDICT_ACCEPTED]

    def export(self, index, *, best_effort: bool | None = None) -> dict:
        """Mark the candidate that is exported. ``best_effort`` defaults to
        "its verdict is not accepted"."""
        c = self._entry(index, None)
        self.exported = index
        self.best_effort = (c["verdict"] != VERDICT_ACCEPTED) if best_effort is None else bool(best_effort)
        if self.best_effort:
            c["best_effort"] = True
            logger.warning("%s (exported candidate #%s: %s)", BEST_EFFORT_WARNING, index,
                           "; ".join(c["reasons"]) or c["verdict"])
        return c

    @property
    def success(self) -> bool:
        if self.exported is None or self.best_effort:
            return False
        return any(c["index"] == self.exported and c["verdict"] == VERDICT_ACCEPTED for c in self.candidates)

    @property
    def result(self) -> str:
        """Tri-state result of the exported candidate."""
        from orbitopt.verify.report import RESULT_NOT_CHECKED

        for c in self.candidates:
            if c["index"] == self.exported:
                return c.get("verification") or RESULT_NOT_CHECKED
        return RESULT_NOT_CHECKED

    def warnings(self) -> list:
        if self.exported is None:
            return []
        if self.best_effort:
            return [BEST_EFFORT_WARNING]
        if not self.success:
            return [f"exported candidate #{self.exported} is not verified (verification: {self.result})"]
        return []

    def to_dict(self) -> dict:
        from orbitopt.pipeline import _jsonable

        counts = {v: 0 for v in VERDICTS}
        for c in self.candidates:
            counts[c["verdict"]] += 1
        return _jsonable({
            "result": self.result,
            "success": self.success,
            "best_effort": self.best_effort,
            "exported_candidate": self.exported,
            "counts": counts,
            "candidates": self.candidates,
            "warnings": self.warnings(),
            "notes": list(self.notes),
        })


def mark_scene_best_effort(scene: dict, gate: VerificationGate) -> dict:
    """Stamp a gate's warnings onto a scene: a top-level ``warnings`` list
    and the warning as the *first* subtitle line the GUI shows."""
    warnings = gate.warnings()
    if not warnings:
        return scene
    existing = scene.setdefault("warnings", [])
    for w in warnings:
        if w not in existing:
            existing.append(w)
    banner = "WARNING: " + warnings[0]
    subtitle = scene.get("subtitle") or ""
    if not subtitle.startswith(banner):
        scene["subtitle"] = banner + ("\n" + subtitle if subtitle else "")
    return scene


@dataclass
class GateOutcome:
    """What ``walk_candidates`` exported."""

    index: int
    payload: object
    result: object
    best_effort: bool


def walk_candidates(
    shortlist: Sequence,
    fly: Callable,
    *,
    gate: VerificationGate,
    reports_of: Callable,
    fallback: bool = True,
) -> GateOutcome:
    """Fly the selected candidates through refinement + verification and
    gate them.

    ``shortlist`` is ``[(index, payload), ...]`` best first.
    ``fly(index, payload, strict=..., rejected=[...])`` refines/verifies one
    candidate and returns its result, or raises ``RuntimeError`` to reject it
    (``rejected`` lists the indices rejected so far). ``reports_of(result)``
    returns the candidate's verification reports.

    The first candidate whose reports pass is exported as a success. A
    candidate that flies but is failed / not_checked is *degraded*: the walk
    goes on looking for one that passes. If none does, the best-ranked
    degraded candidate is exported best-effort; if every one was rejected,
    the best-ranked one is re-flown with ``strict=False`` (when
    ``fallback``) and exported best-effort. Best-effort exports carry
    ``BEST_EFFORT_WARNING``.
    """
    rejected, degraded = [], None
    for index, payload in shortlist:
        try:
            result = fly(index, payload, strict=True, rejected=list(rejected))
        except RuntimeError as exc:
            gate.reject(index, str(exc))
            rejected.append(index)
            logger.warning("candidate #%s rejected: %s", index, exc)
            continue
        verdict = gate.evaluate(index, reports_of(result))
        if verdict["verdict"] == VERDICT_ACCEPTED:
            gate.export(index)
            return GateOutcome(index, payload, result, best_effort=False)
        logger.warning("candidate #%s flown but not verified (%s) -- degraded: %s",
                       index, verdict["verification"], "; ".join(verdict["reasons"]))
        if degraded is None:
            degraded = (index, payload, result)
    if degraded is not None:
        index, payload, result = degraded
        gate.notes.append(f"no candidate passed verification; best-ranked flown candidate #{index} "
                          "exported best-effort")
        gate.export(index, best_effort=True)
        return GateOutcome(index, payload, result, best_effort=True)
    if not fallback or not shortlist:
        raise RuntimeError("every selected candidate was rejected: "
                           + "; ".join(f"#{c['index']}: {', '.join(c['reasons'])}" for c in gate.candidates))
    index, payload = shortlist[0]
    gate.notes.append(f"every selected candidate was rejected; best-ranked candidate #{index} re-flown "
                      "with relaxed acceptance and exported best-effort")
    try:
        result = fly(index, payload, strict=False, rejected=list(rejected))
    except RuntimeError as exc:
        gate.reject(index, f"best-effort re-flight: {exc}")
        raise
    gate.evaluate(index, reports_of(result))
    gate.export(index, best_effort=True)
    return GateOutcome(index, payload, result, best_effort=True)
