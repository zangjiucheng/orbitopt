"""Deterministic checks of a compiled problem against what the user said and fixed.

None of this asks the model. Three checks:

* ``split_prompt`` separates the problem owner's words from attached material
  (tool output, logs, pasted text). Only the owner's words can authorise a
  change; an instruction inside an attachment never can.
* ``extract_statements`` and ``uncovered_statements`` form the coverage check.
  The extractor finds numeric bounds in the owner's text by pattern
  (``>= 1``, ``at most 26``, ``between -5 and 10``, ``[0, 1]``, ``0 <= x``,
  "nonnegative", and Chinese equivalents) and the check requires each number
  to reappear in the compiled problem with a compatible direction, or in an
  assumption the model wrote. It is a number-presence check: it does not know
  which quantity a number belongs to, so a coincidental equal number elsewhere
  satisfies it, and constraints stated without a comparator phrase ("plant 1
  has 4 hours") are not extracted at all. See ``docs/agent/compiler.md``.
* ``authorization_violations`` compares a revised problem with its base item
  by item (locked constraints, user objectives, every variable domain, every
  parameter) and allows a change only when the model listed it in
  ``authorized_changes`` with a quote that is literally in the owner's text
  and, for a changed number, contains the new number.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Literal, Sequence

from orbitopt.optimization import Expression, OptimizationProblem, ToolCall, UnitError, parse_unit

from .outcome import Violation
from .schema import AuthorizedChange

Direction = Literal[">=", "<=", "=="]

MIN_QUOTE_CHARS = 4
"""An authorising quote shorter than this (ignoring spaces) is never trusted."""

_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_NUMBER = re.compile(_NUM)

# Inline attachments: "[Attached ...]", "[Tool output: ...]", "[Log ...]" and the Chinese
# "attachment", "log", "tool output" prefixes, in ASCII or full-width brackets.
_ATTACHMENT = re.compile(
    r"[\[\u3010]\s*(?:attached|attachment|tool output|optimizer log|solver log|log"
    r"|\u9644\u4ef6|\u65e5\u5fd7|\u5de5\u5177\u8f93\u51fa)[^\]\u3011]*[\]\u3011]",
    re.IGNORECASE)

_GE_WORDS = ("at least", "no less than", "not less than", "no fewer than", "not fewer than",
             "greater than or equal to", "greater than", "more than", "above", "a minimum of", "minimum of")
_LE_WORDS = ("at most", "no more than", "not more than", "no greater than", "not greater than", "up to",
             "not to exceed", "must not exceed", "does not exceed", "do not exceed", "not exceed",
             "less than or equal to", "less than", "smaller than", "below", "a maximum of", "maximum of")
_EQ_WORDS = ("equal to", "equals", "exactly")
# Chinese: at least, at least (min), no less than, not below, not smaller than, greater-or-equal, greater than, above
_GE_ZH = ("\u81f3\u5c11", "\u6700\u5c11", "\u4e0d\u5c11\u4e8e", "\u4e0d\u4f4e\u4e8e", "\u4e0d\u5c0f\u4e8e",
          "\u5927\u4e8e\u7b49\u4e8e", "\u5927\u4e8e", "\u9ad8\u4e8e")
# Chinese: at most, at most (max), not exceed, no more than, not greater than, less-or-equal, not above, less than, below
_LE_ZH = ("\u81f3\u591a", "\u6700\u591a", "\u4e0d\u8d85\u8fc7", "\u4e0d\u591a\u4e8e", "\u4e0d\u5927\u4e8e",
          "\u5c0f\u4e8e\u7b49\u4e8e", "\u4e0d\u9ad8\u4e8e", "\u5c0f\u4e8e", "\u4f4e\u4e8e")
_EQ_ZH = ("\u7b49\u4e8e",)  # equal to
_NONNEGATIVE = re.compile(r"\bnon-?negative\b|\u975e\u8d1f", re.IGNORECASE)
# Chinese unit words -> IR unit symbols: kilogram (x2), gram, tonne, kilometre (x2), metre, second, minute, hour, day
_ZH_UNITS = {"\u5343\u514b": "kg", "\u516c\u65a4": "kg", "\u514b": "g", "\u5428": "t", "\u516c\u91cc": "km",
             "\u5343\u7c73": "km", "\u7c73": "m", "\u79d2": "s", "\u5206\u949f": "min", "\u5c0f\u65f6": "h",
             "\u5929": "day"}
_ASCII_UNIT = re.compile(r"\s*([A-Za-z][A-Za-z0-9*/^]*)")
_WORD_NUMBERS = {word: float(i) for i, word in enumerate(
    ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"))}
# Chinese numerals zero to ten, accepted only as a single character ("er shi", twenty, is not read)
_ZH_NUMBERS = {c: float(i) for i, c in enumerate("\u96f6\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341")}
_ZH_DIGITS = "".join(_ZH_NUMBERS)
_COMPARED = (rf"{_NUM}|(?:{'|'.join(_WORD_NUMBERS)})\b|[{_ZH_DIGITS}](?![{_ZH_DIGITS}\u767e\u5343\u4e07])")
"""What may follow a comparator: a numeral, an English number word, or one Chinese numeral."""


def _value(token: str) -> float:
    token = token.casefold()
    return _WORD_NUMBERS.get(token, _ZH_NUMBERS.get(token)) if not _NUMBER.fullmatch(token) else float(token)


def _alternation(words: Sequence[str], english: bool) -> str:
    """Longest first, and never inside a negation: "more than" does not match in "no more than"."""
    ordered = sorted(words, key=len, reverse=True)
    body = "|".join(re.escape(w).replace(r"\ ", r"\s+") for w in ordered)
    if english:
        return rf"(?<!no )(?<!not )\b(?:{body})\b"
    return rf"(?<![\u4e0d\u4e8e])(?:{body})"  # not after "not" (bu) or inside "greater-or-equal" (yu dengyu)


_COMPARATORS: tuple[tuple[re.Pattern, str], ...] = tuple(
    # "at least of 4", and up to two Chinese characters of filler: "at least is 1", "at least need 4"
    (re.compile(rf"(?:{pattern})\s*(?:of\s+|[\u4e00-\u9fff]{{1,2}}\s*)?(?P<num>{_COMPARED})", re.IGNORECASE),
     direction)
    for pattern, direction in (
        (r">=|\u2265|=>", ">="), (r"<=|\u2264|=<", "<="),
        (_alternation(_GE_WORDS, True), ">="), (_alternation(_LE_WORDS, True), "<="),
        (_alternation(_EQ_WORDS, True), "=="),
        (_alternation(_GE_ZH, False), ">="), (_alternation(_LE_ZH, False), "<="),
        (_alternation(_EQ_ZH, False), "=="),
        (r"(?<![<>=!])==?(?!=)", "=="), (r"(?<![<>=\-])>(?!=)", ">="), (r"(?<![<>=])<(?![=<])", "<="),
    ))
# "0 <= x": a number standing alone before a comparator bounds what follows it.
_REVERSED = re.compile(rf"(?:^|[(,;:\uff0c\uff1b]|\band\b|\bwith\b|\bwhere\b)\s*(?P<num>{_NUM})\s*"
                       r"(?P<cmp><=|\u2264|>=|\u2265|<(?!=)|>(?!=))", re.IGNORECASE)
_INTERVALS = (
    re.compile(rf"\bbetween\s+(?P<lo>{_NUM})\s+and\s+(?P<hi>{_NUM})", re.IGNORECASE),
    re.compile(rf"\[\s*(?P<lo>{_NUM})\s*,\s*(?P<hi>{_NUM})\s*\]"),
    # Chinese "between A and B": jieyu A he/yu/dao/zhi B, or zai A dao/zhi B zhijian
    re.compile(rf"\u4ecb\u4e8e\s*(?P<lo>{_NUM})\s*(?:\u548c|\u4e0e|\u5230|\u81f3)\s*(?P<hi>{_NUM})"),
    re.compile(rf"(?P<lo>{_NUM})\s*(?:\u5230|\u81f3)\s*(?P<hi>{_NUM})\s*(?:\S{{0,4}}?)\u4e4b\u95f4"),
)


@dataclass(frozen=True)
class PromptText:
    """The problem owner's words, and attached material that may inform but never authorise."""

    owner: str
    untrusted: tuple[str, ...] = ()


def split_prompt(prompt: str, attachments: Sequence[str] = ()) -> PromptText:
    """Move bracketed inline attachments (``[Attached log: ...]``) out of the owner's text."""
    inline = tuple(m.group(0) for m in _ATTACHMENT.finditer(prompt))
    owner = _ATTACHMENT.sub(" ", prompt).strip()
    return PromptText(owner, inline + tuple(attachments))


@dataclass(frozen=True)
class Statement:
    """A numeric bound found in the owner's text: ``direction value unit`` and where it was found."""

    direction: Direction
    value: float
    unit: str | None
    span: tuple[int, int]
    text: str


def _unit_after(text: str, end: int) -> tuple[str | None, int]:
    rest = text[end:]
    stripped = rest.lstrip()
    for word in sorted(_ZH_UNITS, key=len, reverse=True):
        if stripped.startswith(word):
            return _ZH_UNITS[word], end + len(rest) - len(stripped) + len(word)
    m = _ASCII_UNIT.match(rest)
    if m:
        token = m.group(1).rstrip("*/^")
        try:
            parse_unit(token)
        except UnitError:
            return None, end
        return token, end + m.start(1) + len(token)
    return None, end


def extract_statements(text: str) -> list[Statement]:
    """Every numeric bound the patterns find in ``text``, in order of position."""
    found: list[Statement] = []

    def add(direction, number, start, end):
        unit, stop = _unit_after(text, end)
        found.append(Statement(direction, _value(number), unit, (start, stop), text[start:stop].strip()))

    for pattern, direction in _COMPARATORS:
        for m in pattern.finditer(text):
            add(direction, m.group("num"), m.start(), m.end("num"))
    for m in _REVERSED.finditer(text):
        direction = ">=" if m.group("cmp") in ("<=", "\u2264", "<") else "<="
        found.append(Statement(direction, float(m.group("num")), None, (m.start("num"), m.end()),
                               text[m.start("num"):m.end()]))
    for pattern in _INTERVALS:
        for m in pattern.finditer(text):
            unit, stop = _unit_after(text, m.end("hi"))
            stop = stop if unit else m.end()
            for direction, group in ((">=", "lo"), ("<=", "hi")):
                found.append(Statement(direction, float(m.group(group)), unit, (m.start(), stop),
                                       text[m.start():stop]))
    for m in _NONNEGATIVE.finditer(text):
        found.append(Statement(">=", 0.0, None, m.span(), m.group(0)))
    unique = {(s.direction, s.value, s.unit, s.span): s for s in found}
    return sorted(unique.values(), key=lambda s: (s.span, s.direction))


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def _numbers(text: str) -> list[float]:
    return [float(n) for n in _NUMBER.findall(text)]


def _literals(source: str) -> list[float]:
    """Numeric literals of a validated expression as written, excluding exponents and index arithmetic.

    ``x**2`` contributes nothing, and neither do the indices and range bounds
    of ``sum(x[i+1] for i in range(19))``: they count terms, not quantities.
    """
    tree = ast.parse(source.strip(), mode="eval")
    skipped = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            skipped.add(id(node.right))
        elif isinstance(node, ast.Subscript):
            skipped |= {id(sub) for sub in ast.walk(node.slice)}
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
            skipped |= {id(sub) for arg in node.args for sub in ast.walk(arg)}
    return [float(node.value) for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and type(node.value) in (int, float) and id(node) not in skipped]


@dataclass(frozen=True)
class _Number:
    direction: str  # ">=", "<=", "==", or "any"
    value: float
    unit: str | None


def _problem_numbers(problem: OptimizationProblem) -> list[_Number]:
    out = []
    for v in problem.variables:
        if v.kind == "binary":
            out += [_Number(">=", 0.0, v.unit), _Number("<=", 1.0, v.unit)]
        out += [_Number(side, bound, v.unit) for side, bound in ((">=", v.lower), ("<=", v.upper))
                if bound is not None]
        out += [_Number("any", c, v.unit) for c in v.choices]
    out += [_Number("any", p.value, p.unit) for p in problem.parameters]
    for item in (*problem.objectives, *problem.constraints):
        sources = list(item.expression.args.values()) if isinstance(item.expression, ToolCall) \
            else [item.expression.source]
        out += [_Number("any", lit, None) for src in sources for lit in _literals(src)]
    out += [_Number(c.sense, c.rhs, c.unit) for c in problem.constraints]
    return out


def _covers(number: _Number, st: Statement) -> bool:
    signs = {">=": {">=": 1, "<=": -1}, "<=": {"<=": 1, ">=": -1}}.get(st.direction, {})
    if number.direction in ("any", "==") or st.direction == "==":
        allowed = (1, -1)
    elif number.direction in signs:
        allowed = (signs[number.direction],)
    else:
        return False
    candidates = [st.value]
    if st.unit is not None:
        stated = parse_unit(st.unit)
        if number.unit is None:
            candidates.append(st.value * stated.scale)
        elif parse_unit(number.unit).dims == stated.dims:
            candidates = [st.value * stated.scale / parse_unit(number.unit).scale]
    return any(_close(number.value, sign * c) for sign in allowed for c in candidates)


def uncovered_statements(statements: Sequence[Statement], problem: OptimizationProblem,
                         explanations: Sequence[str] = (), authorized_quotes: Sequence[str] = ()) -> list[Statement]:
    """Statements whose number appears nowhere compatible in ``problem`` and in no explanation.

    A statement is covered by a variable bound or constraint rhs of the same
    direction (or the opposite direction with the opposite sign), by a
    parameter value, discrete choice or expression literal in either sign, by
    a number in one of ``explanations`` (assumption texts), or by lying
    inside one of ``authorized_quotes`` (the user describing a change).
    When the statement has a unit and the compiled item a unit of the same
    dimension, the comparison is in that item's unit.
    """
    numbers = _problem_numbers(problem)
    explained = [n for text in explanations for n in _numbers(text)]
    quotes = [_normalise(q) for q in authorized_quotes]
    missing = []
    for st in statements:
        if any(_covers(n, st) for n in numbers) or any(_close(abs(e), abs(st.value)) for e in explained):
            continue
        if any(_normalise(st.text) in q for q in quotes):
            continue
        missing.append(st)
    return missing


def _normalise(text: str) -> str:
    return " ".join(text.casefold().split())


def quote_in(quote: str, text: str) -> bool:
    """Whether ``quote`` is literally in ``text`` (case and spacing ignored) and long enough to mean something."""
    q = _normalise(quote)
    return len(q.replace(" ", "")) >= MIN_QUOTE_CHARS and q in _normalise(text)


@dataclass(frozen=True)
class _Diff:
    kind: str
    item: str
    message: str
    new_values: tuple[float, ...] = ()


def _expr_key(expr: Expression | ToolCall):
    if isinstance(expr, ToolCall):
        return ("tool", expr.tool_id, tuple(sorted((k, "".join(v.split())) for k, v in expr.args.items())))
    return ("expr", "".join(expr.source.split()))


def _same_unit(a: str, b: str) -> bool:
    ua, ub = parse_unit(a), parse_unit(b)
    return ua.dims == ub.dims and _close(ua.scale, ub.scale)


def _same_number(a: float | None, b: float | None) -> bool:
    return (a is None) == (b is None) and (a is None or _close(a, b))


def revision_diff(base: OptimizationProblem, revised: OptimizationProblem) -> list[_Diff]:
    """Every change to what ``hard_constraint_fingerprint`` protects, per item."""
    if base.hard_constraint_fingerprint() == revised.hard_constraint_fingerprint():
        return []
    diffs = []
    new_constraints = {c.name: c for c in revised.constraints}
    for c in base.constraints:
        if not c.locked:
            continue
        item = f"constraint:{c.name}"
        new = new_constraints.get(c.name)
        if new is None or not new.locked:
            diffs.append(_Diff("removed_locked_constraint", item, f"locked constraint {c.name!r} was removed"
                               + (" or unlocked" if new is not None else "")))
        elif not (_expr_key(c.expression) == _expr_key(new.expression) and c.sense == new.sense
                  and _same_number(c.rhs, new.rhs) and _same_unit(c.unit, new.unit)):
            diffs.append(_Diff("changed_locked_constraint", item,
                               f"locked constraint {c.name!r} changed", () if _close(c.rhs, new.rhs) else (new.rhs,)))
    base_objectives = {o.name: o for o in base.objectives if o.origin == "user"}
    new_objectives = {o.name: o for o in revised.objectives if o.origin == "user"}
    for name, o in base_objectives.items():
        new = new_objectives.get(name)
        if new is None:
            diffs.append(_Diff("removed_user_objective", f"objective:{name}", f"user objective {name!r} was removed"))
        elif not (_expr_key(o.expression) == _expr_key(new.expression) and o.sense == new.sense
                  and _same_unit(o.unit, new.unit)):
            diffs.append(_Diff("changed_user_objective", f"objective:{name}", f"user objective {name!r} changed"))
    for name in new_objectives.keys() - base_objectives.keys():
        diffs.append(_Diff("added_user_objective", f"objective:{name}", f"user objective {name!r} was added"))
    new_variables = {v.name: v for v in revised.variables}
    for v in base.variables:
        new = new_variables.get(v.name)
        item = f"variable:{v.name}"
        if new is None:
            diffs.append(_Diff("removed_variable", item, f"variable {v.name!r} was removed"))
        elif not (v.kind == new.kind and _same_number(v.lower, new.lower) and _same_number(v.upper, new.upper)
                  and tuple(v.choices) == tuple(new.choices) and _same_unit(v.unit, new.unit)):
            changed = tuple(b for a, b in ((v.lower, new.lower), (v.upper, new.upper))
                            if b is not None and not _same_number(a, b))
            diffs.append(_Diff("changed_domain", item, f"domain of {v.name!r} changed from "
                               f"[{v.lower}, {v.upper}] {v.kind} to [{new.lower}, {new.upper}] {new.kind}", changed))
    new_parameters = {p.name: p for p in revised.parameters}
    for p in base.parameters:
        new = new_parameters.get(p.name)
        item = f"parameter:{p.name}"
        if new is None:
            diffs.append(_Diff("removed_parameter", item, f"parameter {p.name!r} was removed"))
        elif not (_close(p.value, new.value) and _same_unit(p.unit, new.unit)):
            diffs.append(_Diff("changed_parameter", item,
                               f"parameter {p.name!r} changed from {p.value} {p.unit} to {new.value} {new.unit}",
                               (new.value,)))
    return diffs


def authorization_violations(compiled: OptimizationProblem, base: OptimizationProblem | None,
                             changes: Sequence[AuthorizedChange], text: PromptText) -> list[Violation]:
    """Unauthorised changes: claims not backed by the owner's words, and base items changed without such a claim."""
    violations = []
    trusted = []
    for change in changes:
        if quote_in(change.quote, text.owner):
            trusted.append(change)
            continue
        where = ("it comes from attached material, which cannot authorise a change"
                 if any(quote_in(change.quote, t) for t in text.untrusted) else "the owner never wrote it")
        violations.append(Violation("untrusted_authorization", change.item,
                                    f"authorisation quote {change.quote!r} is not in the problem owner's text: {where}"))
    if base is not None:
        for diff in revision_diff(base, compiled):
            if not any(c.item == diff.item and all(any(_close(n, v) for n in _numbers(c.quote))
                                                   for v in diff.new_values) for c in trusted):
                violations.append(Violation(diff.kind, diff.item, diff.message + " without the owner's authorisation"))
    return violations
