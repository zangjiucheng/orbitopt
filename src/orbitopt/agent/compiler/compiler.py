"""``compile_problem``: natural language to a validated IR problem, through structured output only.

One compile is at most ``1 + MAX_REPAIR_ROUNDS`` rounds. Each round makes one
``generate_structured`` call against ``COMPILER_OUTPUT_SCHEMA`` (the provider
retries a reply that is not schema-valid JSON once, on its own) and then
judges the reply deterministically, in this order:

1. the model reports the request unsupported, or the draft needs a variable
   kind, a soft constraint or a tool id the caller did not list: ``Unsupported``;
2. the model asks questions, or leaves a parameter value or rhs null:
   ``NeedsClarification``;
3. the compiler converts the draft to an IR dict and runs ``parse_problem``;
   IR errors and unit-hint mismatches go back to the model as a repair turn;
4. the owner's text mentions randomness, dynamics or penalties that no
   assumption addresses, two objectives conflict, or several objectives come
   without a quoted trade-off request: ``NeedsClarification``;
5. an authorisation quote is not the owner's words, or a revision changes a
   locked constraint, user objective, domain or parameter without one:
   ``RejectedUnauthorizedChange``;
6. a numeric bound stated in the owner's text is missing from the problem:
   a repair turn;
7. otherwise ``Compiled``.

When the repair rounds run out, the result is ``CompileFailed`` with the last
round's errors. The model's text is only ever parsed as JSON and its strings
become IR field values; expressions run only through the IR's validated
evaluator.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from functools import partial
from typing import Any, Callable, Mapping, Sequence

from orbitopt.agent.equivalence import conflicting_objectives
from orbitopt.agent.providers import LLMProvider, Message, StructuredOutputError, Usage
from orbitopt.optimization import (
    InvalidProblemError,
    OptimizationProblem,
    VerificationPolicy,
    parse_problem,
    parse_unit,
)

from .authorization import (
    PromptText,
    authorization_violations,
    extract_statements,
    quote_in,
    split_prompt,
    uncovered_statements,
)
from .outcome import (
    Assumption,
    CompileFailed,
    Compiled,
    CompileOutcome,
    CompilerErrorCode,
    CompilerIssue,
    Issue,
    NeedsClarification,
    Question,
    RejectedUnauthorizedChange,
    Unsupported,
    UnsupportedReason,
)
from .prompts import build_messages, repair_message
from .schema import COMPILER_OUTPUT_SCHEMA, CompilerOutput, assumed_items, draft_to_ir_dict

MAX_REPAIR_ROUNDS = 2
"""Repair turns after the first round, so at most three ``generate_structured`` calls per compile."""
DEFAULT_MAX_TOKENS = 4096
IR_VARIABLE_KINDS = ("continuous", "integer", "binary", "discrete")

UNSUPPORTED_MARKERS: dict[str, tuple[str, ...]] = {
    "stochastic": ("expected value", "expectation", "probability", "stochastic", "random", "uncertain",
                   "chance constraint", "variance",
                   "\u671f\u671b", "\u6982\u7387", "\u968f\u673a", "\u4e0d\u786e\u5b9a"),
    "dynamic": ("differential equation", "over time", "time-varying", "control law",
                "\u5fae\u5206\u65b9\u7a0b", "\u968f\u65f6\u95f4"),
    "soft_constraint": ("penalty", "penalize", "penalise", "soft constraint", "preferably",
                        "\u60e9\u7f5a", "\u8f6f\u7ea6\u675f"),
}
"""Phrases that suggest IR 1.0 cannot express the request; the Chinese entries are, in order, expectation,
probability, random, uncertain; differential equation, over time; penalty, soft constraint."""

Verdict = Callable[..., CompileOutcome]


def compile_problem(prompt: str, provider: LLMProvider, *, base_problem: OptimizationProblem | None = None,
                    policy: VerificationPolicy | None = None, max_tokens: int = DEFAULT_MAX_TOKENS,
                    unit_hints: Mapping[str, str] | None = None, known_tool_ids: Sequence[str] = (),
                    attachments: Sequence[str] = (), problem_id: str | None = None,
                    seed: int | None = None) -> CompileOutcome:
    """Compile ``prompt`` into an IR problem, or say why not.

    ``base_problem`` makes this a revision: the result is diffed against it
    and every change must be authorised by the owner's own words.
    ``attachments`` (and bracketed ``[Attached ...]`` blocks inside the
    prompt) are shown to the model as untrusted information. ``policy``
    becomes the problem's verification policy. ``unit_hints`` maps a variable
    or parameter name to the unit it must use. ``known_tool_ids`` are the
    tool ids a ``ToolCall`` may name. ``problem_id`` defaults to the base
    problem's id, else to ``nl-`` and a digest of the prompt.

    Returns one of the five ``CompileOutcome`` classes. Provider failures
    other than ``StructuredOutputError`` (unavailable, budget exceeded,
    replay mismatch) propagate.
    """
    text = split_prompt(prompt, attachments)
    if problem_id is None:
        problem_id = base_problem.id if base_problem is not None else \
            "nl-" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
    verification = None if policy is None else {k: list(v) if isinstance(v, tuple) else v
                                                for k, v in asdict(policy).items()}
    messages = build_messages(text, base_problem=base_problem, unit_hints=unit_hints, known_tool_ids=known_tool_ids)
    usage, attempts = Usage(0, 0, 0), 0
    issues: list[Issue] = []
    raw = None
    for _ in range(1 + MAX_REPAIR_ROUNDS):
        attempts += 1
        try:
            response = provider.generate_structured(messages, COMPILER_OUTPUT_SCHEMA, max_tokens=max_tokens,
                                                    seed=seed)
        except StructuredOutputError as exc:
            errors = tuple(CompilerIssue(CompilerErrorCode.STRUCTURED_OUTPUT, "", e) for e in exc.errors)
            return CompileFailed(errors, exc.raw_text, usage + exc.usage, attempts)
        usage, raw = usage + response.usage, response.text
        verdict = _judge(CompilerOutput.from_dict(response.parsed), text, base=base_problem,
                         known_tool_ids=known_tool_ids, unit_hints=unit_hints, problem_id=problem_id,
                         verification=verification)
        if callable(verdict):
            return verdict(usage=usage, attempts=attempts)
        issues = verdict
        messages += (Message("assistant", response.text), Message("user", repair_message(issues)))
    return CompileFailed(tuple(issues), raw, usage, attempts)


def _judge(output: CompilerOutput, text: PromptText, *, base: OptimizationProblem | None,
           known_tool_ids: Sequence[str], unit_hints: Mapping[str, str] | None, problem_id: str,
           verification: Mapping[str, Any] | None) -> Verdict | list[Issue]:
    """An outcome constructor awaiting ``usage`` and ``attempts``, or the issues to send back for repair."""
    draft = output.problem
    if output.status == "unsupported":
        return partial(Unsupported, output.unsupported_reason or UnsupportedReason(
            "other", "the model reported the request as unsupported without giving a reason"))
    reason = _draft_unsupported(draft, known_tool_ids) if draft is not None else None
    if reason is not None:
        return partial(Unsupported, reason)
    if output.status == "needs_clarification" or output.questions:
        questions = output.questions or (Question("other", "The model asked for clarification without a "
                                                           "question; please restate the problem."),)
        return partial(NeedsClarification, tuple(questions), draft)
    if draft is None:
        return [CompilerIssue(CompilerErrorCode.MISSING_PROBLEM, "", "status is \"compiled\" but problem is null")]
    missing = _missing_values(draft)
    if missing:
        return partial(NeedsClarification, missing, draft)

    assumptions = _assumptions(output, draft)
    try:
        problem = parse_problem(draft_to_ir_dict(draft, problem_id=problem_id, assumptions=assumptions,
                                                 verification=verification))
    except InvalidProblemError as exc:
        return list(exc.errors)
    issues = _unit_hint_issues(problem, unit_hints or {})
    if issues:
        return issues

    questions = _marker_questions(text.owner, output.assumptions) + _objective_questions(problem, output, text)
    if questions:
        return partial(NeedsClarification, tuple(questions), draft)
    violations = authorization_violations(problem, base, output.authorized_changes, text)
    if violations:
        return partial(RejectedUnauthorizedChange, tuple(violations))
    trusted_quotes = [c.quote for c in output.authorized_changes if quote_in(c.quote, text.owner)]
    uncovered = uncovered_statements(extract_statements(text.owner), problem,
                                     [a.text for a in output.assumptions], trusted_quotes)
    if uncovered:
        return [CompilerIssue(CompilerErrorCode.UNCOVERED_STATEMENT, "",
                              f"the owner stated {st.text!r} ({st.direction} {st.value:g}"
                              f"{' ' + st.unit if st.unit else ''}) but no bound, rhs, parameter or literal in "
                              "the problem carries it; add it, or explain in assumptions why it is not a constraint")
                for st in uncovered]
    return partial(Compiled, problem, assumptions)


def _draft_unsupported(draft: Mapping[str, Any], known_tool_ids: Sequence[str]) -> UnsupportedReason | None:
    for v in draft["variables"]:
        if v["kind"] not in IR_VARIABLE_KINDS:
            return UnsupportedReason("variable_kind", f"variable {v['name']!r} has kind {v['kind']!r}; IR 1.0 has "
                                                      "only continuous, integer, binary and discrete numeric variables")
    for c in draft["constraints"]:
        if c.get("soft"):
            return UnsupportedReason("soft_constraint", f"constraint {c['name']!r} is soft; IR 1.0 constraints are "
                                                        "all hard and it has no penalty terms")
    for item in (*draft["objectives"], *draft["constraints"]):
        expr = item["expression"]
        if isinstance(expr, Mapping) and expr["tool_id"] not in known_tool_ids:
            return UnsupportedReason("unsupported_model", f"{item['name']!r} needs tool {expr['tool_id']!r}, which "
                                                          "is not a known tool id")
    for model in draft.get("requested_models", ()):
        if model not in known_tool_ids:
            return UnsupportedReason("unsupported_model", f"the request needs the model {model!r}, which is neither "
                                                          "an expression over the variables nor a known tool id")
    return None


def _missing_values(draft: Mapping[str, Any]) -> tuple[Question, ...]:
    return tuple(
        [Question("missing_parameter", f"What is the value of {p['name']!r}?", f"parameters/{p['name']}")
         for p in draft["parameters"] if p["value"] is None]
        + [Question("missing_parameter", f"What is the right-hand side of constraint {c['name']!r}?",
                    f"constraints/{c['name']}") for c in draft["constraints"] if c["rhs"] is None])


def _assumptions(output: CompilerOutput, draft: Mapping[str, Any]) -> tuple[Assumption, ...]:
    """The model's assumptions, plus one entry for every assumed item it left unexplained."""
    explained = {a.about for a in output.assumptions}
    added = [Assumption(item, "added by the model without a stated reason") for item in assumed_items(draft)
             if item not in explained and item.split("/", 1)[1] not in explained]
    return output.assumptions + tuple(added)


def _unit_hint_issues(problem: OptimizationProblem, hints: Mapping[str, str]) -> list[Issue]:
    issues = []
    for group, items in (("variables", problem.variables), ("parameters", problem.parameters)):
        for i, item in enumerate(items):
            hint = hints.get(item.name)
            if hint is None:
                continue
            got, want = parse_unit(item.unit), parse_unit(hint)
            if got.dims != want.dims or abs(got.scale - want.scale) > 1e-12 * want.scale:
                issues.append(CompilerIssue(CompilerErrorCode.UNIT_HINT_MISMATCH, f"/{group}/{i}/unit",
                                            f"{item.name!r} must be in {hint!r}, got {item.unit!r}"))
    return issues


def _marker_questions(owner: str, assumptions: Sequence[Assumption]) -> list[Question]:
    lowered = owner.casefold()
    notes = " ".join(a.text.casefold() for a in assumptions)
    return [Question("unsupported_feature",
                     f"The statement mentions {phrase!r}, which suggests a {category.replace('_', ' ')} problem that "
                     "IR 1.0 cannot express. Is the problem deterministic with hard constraints only?", category)
            for category, phrases in UNSUPPORTED_MARKERS.items() for phrase in phrases
            if phrase in lowered and phrase not in notes]


def _objective_questions(problem: OptimizationProblem, output: CompilerOutput, text: PromptText) -> list[Question]:
    conflicts = conflicting_objectives(problem)
    if conflicts:
        return [Question("conflicting_objectives", f"Objectives {a!r} and {b!r} pull the same quantity in opposite "
                                                   "directions. Which one do you want?", f"objectives/{a}")
                for a, b in conflicts]
    if len(problem.objectives) > 1 and not (output.tradeoff_quote and quote_in(output.tradeoff_quote, text.owner)):
        names = ", ".join(repr(o.name) for o in problem.objectives)
        return [Question("ambiguity", f"The draft has objectives {names} but the statement does not ask for a "
                                      "trade-off. Is one of them primary, or do you want the trade-off front?",
                         "objectives")]
    return []
