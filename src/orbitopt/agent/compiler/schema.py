"""The LLM-facing ``CompilerOutput`` schema and its deterministic conversion to an IR dict.

The model fills a small, flat JSON object; it never writes IR. The compiler
parses that object into ``CompilerOutput`` and converts the draft problem
itself, index for index, so an IR error path such as ``/constraints/1/unit``
points at the same item in the draft. Every item carries ``source``: ``user``
when the problem owner stated it, ``assumed`` when the model added it.
Assumed objectives and constraints become IR ``origin="compiler"`` (and so are
not locked); assumed variables and parameters, which have no origin in the IR,
are recorded in ``metadata["assumptions"]``.

Variable ``kind`` and constraint ``sense`` are free strings here, not
enumerations, so that a draft asking for a kind the IR lacks reaches the
compiler, which reports it as unsupported instead of the schema forcing the
model to pick a wrong kind.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from orbitopt.optimization import SCHEMA_VERSION

from .outcome import Assumption, Question, UnsupportedReason

SOURCES = ("user", "assumed")

_NUMBER_OR_NULL = {"type": ["number", "null"]}
_SOURCE = {"enum": list(SOURCES)}
_EXPRESSION = {"anyOf": [
    {"type": "string"},
    {"type": "object", "additionalProperties": False, "required": ["tool_id", "args"],
     "properties": {"tool_id": {"type": "string"},
                    "args": {"type": "object", "additionalProperties": {"type": "string"}}}},
]}

DRAFT_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["variables", "parameters", "objectives", "constraints"],
    "properties": {
        "title": {"type": "string"},
        "variables": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["name", "kind", "source"],
            "properties": {"name": {"type": "string"}, "kind": {"type": "string"},
                           "lower": _NUMBER_OR_NULL, "upper": _NUMBER_OR_NULL,
                           "choices": {"type": "array", "items": {"type": "number"}},
                           "unit": {"type": "string"}, "source": _SOURCE}}},
        "parameters": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["name", "value", "source"],
            "properties": {"name": {"type": "string"}, "value": _NUMBER_OR_NULL,
                           "unit": {"type": "string"}, "source": _SOURCE}}},
        "objectives": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["name", "expression", "sense", "source"],
            "properties": {"name": {"type": "string"}, "expression": _EXPRESSION,
                           "sense": {"enum": ["minimize", "maximize"]},
                           "unit": {"type": "string"}, "source": _SOURCE}}},
        "constraints": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "expression", "sense", "rhs", "source"],
            "properties": {"name": {"type": "string"}, "expression": _EXPRESSION,
                           "sense": {"type": "string"}, "rhs": _NUMBER_OR_NULL,
                           "unit": {"type": "string"}, "source": _SOURCE,
                           "soft": {"type": "boolean"}}}},
        "requested_models": {"type": "array", "items": {"type": "string"}},
    },
}

COMPILER_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "problem", "questions", "assumptions", "unsupported_reason",
                 "authorized_changes", "tradeoff_quote"],
    "properties": {
        "status": {"enum": ["compiled", "needs_clarification", "unsupported"]},
        "problem": {"anyOf": [{"type": "null"}, DRAFT_SCHEMA]},
        "questions": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["kind", "text"],
            "properties": {"kind": {"enum": ["missing_parameter", "ambiguity", "conflicting_objectives",
                                             "unsupported_feature", "other"]},
                           "text": {"type": "string"}, "about": {"type": "string"}}}},
        "assumptions": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["about", "text"],
            "properties": {"about": {"type": "string"}, "text": {"type": "string"}}}},
        "unsupported_reason": {"anyOf": [{"type": "null"}, {
            "type": "object", "additionalProperties": False, "required": ["category", "explanation"],
            "properties": {"category": {"enum": ["unsupported_model", "stochastic", "dynamic", "soft_constraint",
                                                 "variable_kind", "semi_infinite", "other"]},
                           "explanation": {"type": "string"}}}]},
        "authorized_changes": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["item", "quote"],
            "properties": {"item": {"type": "string"}, "quote": {"type": "string"}}}},
        "tradeoff_quote": {"type": ["string", "null"]},
    },
}
"""What the model must return. ``authorized_changes[].item`` is ``<group>:<name>``, e.g. ``constraint:budget``."""


@dataclass(frozen=True)
class AuthorizedChange:
    """The model's claim that the user's own words allow changing ``item``; ``quote`` must be those words."""

    item: str
    quote: str


@dataclass(frozen=True)
class CompilerOutput:
    """A schema-validated model reply, parsed into typed fields; ``problem`` stays a draft dict."""

    status: str
    problem: Mapping[str, Any] | None
    questions: tuple[Question, ...]
    assumptions: tuple[Assumption, ...]
    unsupported_reason: UnsupportedReason | None
    authorized_changes: tuple[AuthorizedChange, ...]
    tradeoff_quote: str | None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CompilerOutput":
        """Parse a value that already validated against ``COMPILER_OUTPUT_SCHEMA``."""
        reason = data["unsupported_reason"]
        return cls(
            status=data["status"],
            problem=data["problem"],
            questions=tuple(Question(q["kind"], q["text"], q.get("about", "")) for q in data["questions"]),
            assumptions=tuple(Assumption(a["about"], a["text"]) for a in data["assumptions"]),
            unsupported_reason=UnsupportedReason(reason["category"], reason["explanation"]) if reason else None,
            authorized_changes=tuple(AuthorizedChange(c["item"], c["quote"]) for c in data["authorized_changes"]),
            tradeoff_quote=data["tradeoff_quote"],
        )


def assumed_items(draft: Mapping[str, Any]) -> list[str]:
    """``<group>/<name>`` of every draft item whose source is ``assumed``, in draft order."""
    return [f"{group}/{item['name']}" for group in ("variables", "parameters", "objectives", "constraints")
            for item in draft.get(group, ()) if item.get("source") == "assumed"]


def draft_to_ir_dict(draft: Mapping[str, Any], *, problem_id: str, assumptions: tuple[Assumption, ...],
                     verification: Mapping[str, Any] | None = None) -> dict:
    """Build the IR dict for ``parse_problem`` from a draft whose parameters and rhs are all known.

    Index-preserving: item ``i`` of each draft list is item ``i`` of the IR
    list. Nothing in the model's text is copied as anything but a field value;
    the IR's own validation decides whether it is acceptable.
    """

    def origin(item):
        return "compiler" if item.get("source") == "assumed" else "user"

    def variable(v):
        out = {"name": v["name"], "kind": v["kind"], "unit": v.get("unit", "")}
        for side in ("lower", "upper"):
            if v.get(side) is not None:
                out[side] = v[side]
        if v.get("choices"):
            out["choices"] = list(v["choices"])
        return out

    data = {
        "schemaVersion": SCHEMA_VERSION,
        "id": problem_id,
        "title": draft.get("title", ""),
        "variables": [variable(v) for v in draft["variables"]],
        "parameters": [{"name": p["name"], "value": p["value"], "unit": p.get("unit", "")}
                       for p in draft["parameters"]],
        "objectives": [{"name": o["name"], "expression": o["expression"], "sense": o["sense"],
                        "unit": o.get("unit", ""), "origin": origin(o)} for o in draft["objectives"]],
        "constraints": [{"name": c["name"], "expression": c["expression"], "sense": c["sense"], "rhs": c["rhs"],
                         "unit": c.get("unit", ""), "origin": origin(c)} for c in draft["constraints"]],
        "metadata": {"assumptions": [{"about": a.about, "text": a.text} for a in assumptions]},
    }
    if verification is not None:
        data["verification"] = dict(verification)
    return data
