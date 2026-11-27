"""The structured IR editor's model: edits are data, the IR validator is the judge.

The editor never holds a second notion of a problem. It holds the IR's JSON
form (``doc``), the document the user started from (``base``: a compiled
problem, the compiler's draft, or a blank template) and the ordered list of
:class:`Edit` operations applied since. Every edit re-runs ``parse_problem``
on the new document, so the validation errors the user sees are the IR's own
(code, JSON-pointer path, message), and the digest shown is the IR's digest.

Edits are replayable: ``apply_edits(base, edits)`` rebuilds the exact
document, so a saved experiment can prove that the problem it ran is the
compiled problem plus the user's recorded edits.

An edit to an objective or a constraint makes the item the user's:
``origin`` becomes ``user`` and a constraint becomes ``locked``. Variables
and parameters carry no origin in the IR. ``protected_changes`` lists, with
the compiler's own ``revision_diff``, every change to what
``hard_constraint_fingerprint`` protects, so the user sees what a later
compile will have to leave alone.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

SECTIONS = ("variables", "parameters", "objectives", "constraints")
#: the editable columns of each section, in display order
FIELDS = {
    "variables": ("name", "kind", "lower", "upper", "unit", "initial"),
    "parameters": ("name", "value", "unit"),
    "objectives": ("name", "expression", "sense", "unit"),
    "constraints": ("name", "expression", "sense", "rhs", "unit"),
}
NUMERIC = frozenset({"lower", "upper", "initial", "value", "rhs"})
OWNED = frozenset({"objectives", "constraints"})
"""Sections whose items record an ``origin``; an edit there makes the item the user's."""


@dataclass(frozen=True)
class Edit:
    """One structured change: ``set`` a field of item ``index``, ``add`` an item, or ``remove`` item ``index``."""

    op: Literal["set", "add", "remove"]
    section: str
    index: int = -1
    field: str = ""
    value: Any = None

    def __post_init__(self):
        if self.section not in SECTIONS:
            raise ValueError(f"unknown section {self.section!r}")
        if self.op == "set" and self.field not in FIELDS[self.section]:
            raise ValueError(f"{self.section} has no editable field {self.field!r}")
        if self.op not in ("set", "add", "remove"):
            raise ValueError(f"unknown edit op {self.op!r}")

    def to_dict(self) -> dict:
        return {"op": self.op, "section": self.section, "index": self.index, "field": self.field,
                "value": self.value}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Edit":
        return cls(data["op"], data["section"], int(data["index"]), data.get("field", ""), data.get("value"))


def blank_problem(problem_id: str = "studio-problem") -> dict:
    """The template a hand-written problem starts from: one variable, one objective, valid as it stands."""
    return {"schemaVersion": "1.1", "id": problem_id, "title": "",
            "variables": [{"name": "x", "kind": "continuous", "lower": 0.0, "upper": 1.0, "unit": ""}],
            "parameters": [],
            "objectives": [{"name": "f", "expression": "x**2", "sense": "minimize", "unit": "", "origin": "user"}],
            "constraints": []}


def new_item(section: str, doc: Mapping[str, Any]) -> dict:
    """A fresh item for ``section`` with a name not yet used in it."""
    used = {item.get("name") for item in doc.get(section) or []}
    stem = {"variables": "x", "parameters": "p", "objectives": "f", "constraints": "g"}[section]
    name = next(f"{stem}{i}" for i in range(1, 10_000) if f"{stem}{i}" not in used)
    return {
        "variables": {"name": name, "kind": "continuous", "lower": 0.0, "upper": 1.0, "unit": ""},
        "parameters": {"name": name, "value": 0.0, "unit": ""},
        "objectives": {"name": name, "expression": "0", "sense": "minimize", "unit": "", "origin": "user"},
        "constraints": {"name": name, "expression": "0", "sense": "<=", "rhs": 0.0, "unit": "", "origin": "user",
                        "locked": True},
    }[section]


def coerce(field_name: str, text: str) -> Any:
    """A table cell's text as the value an edit stores.

    Numeric fields take a float, and an empty cell means "none" (an open
    bound). Text that is not a number is stored as typed, so the IR
    validator reports it at its path instead of the editor guessing. An
    expression that looks like JSON (a tool call) is parsed as JSON.
    """
    text = text.strip()
    if field_name in NUMERIC:
        if text == "":
            return None
        try:
            return float(text)
        except ValueError:
            return text
    if field_name == "expression" and text.startswith("{"):
        try:
            return json.loads(text)
        except ValueError:
            return text
    return text


def _own(item: dict, section: str) -> None:
    if section in OWNED:
        item["origin"] = "user"
        if section == "constraints":
            item["locked"] = True


def apply_edit(doc: Mapping[str, Any], edit: Edit) -> dict:
    """The document after ``edit`` (a new dict; ``doc`` is unchanged). ``IndexError`` for a missing item."""
    out = copy.deepcopy(dict(doc))
    items = out.setdefault(edit.section, [])
    if edit.op == "add":
        item = copy.deepcopy(edit.value) if isinstance(edit.value, Mapping) else new_item(edit.section, out)
        _own(item, edit.section)
        items.append(item)
        return out
    if not 0 <= edit.index < len(items):
        raise IndexError(f"{edit.section} has no item {edit.index}")
    if edit.op == "remove":
        del items[edit.index]
        return out
    item = items[edit.index]
    item[edit.field] = edit.value
    _own(item, edit.section)
    return out


def apply_edits(base: Mapping[str, Any], edits) -> dict:
    doc = copy.deepcopy(dict(base))
    for edit in edits:
        doc = apply_edit(doc, edit)
    return doc


@dataclass(frozen=True)
class Issue:
    """One validation error, shaped like the IR's ``ProblemError``."""

    path: str
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.path or '/'}: [{self.code}] {self.message}"


def validate_doc(doc: Mapping[str, Any]):
    """``(problem, ())`` for a valid document, ``(None, issues)`` otherwise; never raises."""
    from orbitopt.optimization import InvalidProblemError, parse_problem

    try:
        return parse_problem(doc), ()
    except InvalidProblemError as exc:
        return None, tuple(Issue(e.path, getattr(e.code, "value", str(e.code)), e.message) for e in exc.errors)
    except Exception as exc:  # noqa: BLE001 -- user input is a boundary: any failure is a shown error
        return None, (Issue("", "invalid_problem", f"{type(exc).__name__}: {exc}"),)


@dataclass(frozen=True)
class IrDraft:
    """The editor's state: where it started, what was changed, and what the IR validator says now."""

    base: Mapping[str, Any]
    edits: tuple[Edit, ...] = ()
    doc: Mapping[str, Any] = field(default=None)  # type: ignore[assignment]
    problem: Any = None
    issues: tuple[Issue, ...] = ()

    @classmethod
    def start(cls, base: Mapping[str, Any]) -> "IrDraft":
        problem, issues = validate_doc(base)
        return cls(copy.deepcopy(dict(base)), (), copy.deepcopy(dict(base)), problem, issues)

    def apply(self, edit: Edit) -> "IrDraft":
        doc = apply_edit(self.doc, edit)
        problem, issues = validate_doc(doc)
        return IrDraft(self.base, self.edits + (edit,), doc, problem, issues)

    @property
    def valid(self) -> bool:
        return self.problem is not None

    @property
    def digest(self) -> str | None:
        return None if self.problem is None else self.problem.digest()

    def issues_for(self, section: str, index: int) -> tuple[Issue, ...]:
        prefix = f"/{section}/{index}"
        return tuple(i for i in self.issues if i.path == prefix or i.path.startswith(prefix + "/"))

    def protected_changes(self) -> tuple[Any, ...]:
        """The compiler's ``revision_diff`` from the base to the current problem (both must be valid)."""
        base_problem, _ = validate_doc(self.base)
        if base_problem is None or self.problem is None:
            return ()
        from orbitopt.agent.compiler.authorization import revision_diff

        return tuple(revision_diff(base_problem, self.problem))
