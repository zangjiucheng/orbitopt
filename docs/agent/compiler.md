# Natural-language problem compiler

`orbitopt.agent.compiler` turns a problem statement into an IR problem
(`orbitopt.optimization.OptimizationProblem`). The model never writes IR and
never writes code. It fills a small JSON object through structured output,
and the compiler builds the IR from that object, validates it, and checks it
against the user's own words. When the compiler cannot produce a faithful
problem, it says why instead of guessing.

```python
from orbitopt.agent.compiler import Compiled, compile_problem
from orbitopt.agent.providers import LocalProvider

outcome = compile_problem("Minimize x^2 + y^2 subject to x + y >= 1 and x, y >= 0.",
                          LocalProvider("qwen2.5:7b"))
if isinstance(outcome, Compiled):
    problem = outcome.problem          # valid IR, user items locked
    outcome.assumptions                # everything the model added
    outcome.usage, outcome.attempts    # tokens over all rounds, rounds used
```

## Outcomes

`compile_problem` returns exactly one of five values. A refusal is a value,
not an exception.

| Outcome | `outcome` | Carries | When |
|---|---|---|---|
| `Compiled` | `compile` | `problem`, `assumptions` | The draft is valid IR and passes every check below. |
| `NeedsClarification` | `needs_clarification` | `questions`, `partial` (the raw draft, not IR) | A value is missing, the wording is ambiguous, objectives conflict, or the text mentions something IR 1.0 may not express. |
| `Unsupported` | `unsupported` | `reason` (`category`, `explanation`) | The request needs an unknown tool or model, randomness, dynamics, soft constraints, a min-max over infinitely many cases, or a variable kind outside the IR. |
| `RejectedUnauthorizedChange` | `rejected_unauthorized_change` | `violations` | The draft changes something the user fixed, and the user's own words do not authorise it. |
| `CompileFailed` | `compile_failed` | `errors`, `raw` | No valid draft within the repair budget. |

Every outcome also carries `usage` and `attempts`. `Compiled` refuses to hold
a problem that `validate_problem` rejects, so no outcome can carry invalid IR.
Provider failures other than `StructuredOutputError` propagate:
`ProviderUnavailableError`, `BudgetExceededError` and `ReplayMismatchError`
are infrastructure faults, not compile results.

## Two schemas, kept apart

The model fills `COMPILER_OUTPUT_SCHEMA`:

- `status` is `compiled`, `needs_clarification` or `unsupported`.
- `problem` holds flat lists of variables, parameters, objectives and
  constraints. Each item has `source`, which is `user` when the owner stated it
  and `assumed` when the model added it. An expression is an IR expression
  string, which may use indexed sums such as
  `sum(x[i]**2 for i in range(20))`, or a tool call
  `{"tool_id", "args", "constants", "output"}`. The integration cases pass
  the integrand as the constant `integrand` of one `quadrature.abs_error`
  tool.
- `questions`, `assumptions` and `unsupported_reason` let the model refuse
  or explain.
- `authorized_changes` lists `{"item": "constraint:<name>", "quote": ...}`
  for each item a revision changes.
- `tradeoff_quote` holds the owner's words that ask for a trade-off.

`draft_to_ir_dict` converts the draft index for index, so an IR error path
such as `/constraints/1/unit` names the same draft item. An assumed objective
or constraint becomes `origin="compiler"` and is not locked. Every user item
is `origin="user"` and locked. Variables and parameters have no origin in the
IR, so the compiler lists assumed ones in `metadata["assumptions"]`. Every
assumed item appears in `Compiled.assumptions`. When the model gave no reason
for an item, the compiler adds the entry itself.

## What the compiler checks

A compile runs at most `1 + MAX_REPAIR_ROUNDS` rounds (`MAX_REPAIR_ROUNDS = 2`).
Each round makes one `generate_structured` call. The provider retries a reply
that is not schema-valid JSON once, on its own. The compiler then judges the
reply in this order.

1. The model reports `unsupported`, or the draft has a variable kind outside
   the IR, a constraint marked `soft`, or a tool id that the caller did not
   pass in `known_tool_ids`. The result is `Unsupported`.
2. The model asks questions, or leaves a parameter value or a constraint
   `rhs` null. The result is `NeedsClarification`.
3. `parse_problem` rejects the IR, or a variable breaks a `unit_hints` entry.
   The compiler sends the typed errors back as a repair turn, each as
   `path: [code] message`.
4. The owner's text contains a phrase from `UNSUPPORTED_MARKERS` ("random",
   "expected value", "penalty", and others in English and Chinese) and no
   assumption mentions it, or two objectives conflict, or the draft has
   several objectives without a quoted trade-off request. The result is
   `NeedsClarification`. Two objectives conflict when their values agree at
   every sample point and their senses differ, or when they are exact
   negatives with the same sense.
5. An authorisation quote is not in the owner's text, or a revision changes a
   protected item without one. The result is `RejectedUnauthorizedChange`.
6. A numeric bound in the owner's text is missing from the problem. The
   compiler sends it back as a repair turn with code `uncovered_statement`.
7. Otherwise the result is `Compiled`.

After the last round the result is `CompileFailed` with that round's errors.
Its `usage` sums every provider call, including one that ended in
`StructuredOutputError`.
The model's text only reaches `json.loads`. Its strings become IR field
values, and the IR's whitelisted evaluator is the only code that runs an
expression. A test walks the compiler's syntax trees and fails on any call to
`eval`, `exec`, `compile` or `__import__`.

## Authorisation

`split_prompt` moves attached material out of the owner's text. That
material is the `attachments` argument plus any bracketed block in the
prompt that starts with "Attached", "Attachment", "Tool output", "Log", or
the Chinese words for attachment, log and tool output. The model sees the
material, labelled untrusted. A quote from it never authorises a change.

When you pass `base_problem`, the compile is a revision. `revision_diff`
compares the result with the base item by item. It covers the items that
`hard_constraint_fingerprint` protects. These are the locked constraints, the
user objectives (removed, changed or added), and every variable domain and
parameter. Each difference needs an `authorized_changes` entry for that item.
The entry's quote must appear literally in the owner's text, ignoring case
and spacing, and must be at least four characters long. When the change sets
a new number, the quote must contain that number. "Make it solve faster"
therefore does not authorise relaxing `x + y >= 1` to `x + y >= 0.9`, and
"Relax the constraint to x + y >= 0.8" does authorise the move to 0.8.

## Coverage check and its limits

The coverage check stops the compiler from silently dropping a stated
constraint. `extract_statements` finds numeric bounds in the owner's text by
pattern:

- comparators followed by a number, such as `>=`, `<=`, `=`, "at least",
  "no more than" and "up to", and the Chinese words for at least, at most,
  not exceed and not less than;
- a number before a comparator, as in `0 <= x`;
- intervals, such as "between A and B", `[A, B]`, and the Chinese forms;
- "nonnegative", read as `>= 0`;
- the number words zero to ten, and one Chinese numeral, after a comparator.

A unit after the number is read when `parse_unit` accepts it, or when it is
one of a few Chinese unit words. `uncovered_statements` then needs each
number in the problem. A variable bound or a constraint `rhs` counts when it
has the same direction, or the opposite direction with the opposite sign. A
parameter value, a discrete choice, or an expression literal counts in either
sign. An exponent does not count. A number in one of the model's assumptions
also counts, and so does text inside an authorised quote. When both sides
have units of the same dimension, the check compares them in one unit, so a
stated 4 km/s does not match a bound of 4 m/s.

The check is a number-presence check, and it misses things.

- It does not know which quantity a number belongs to. A coincidental equal
  number anywhere in the problem satisfies it.
- It finds only bounds that have a comparator phrase. "Plant 1 has 4 hours"
  and "the bag holds 20 kg" are not extracted, so a draft that drops those
  constraints passes.
- It reads numbers written as digits, the words zero to ten, and single
  Chinese numerals. "Twenty" is not read.
- A stated number without a unit matches a compiled number in any unit.
- It checks the owner's text against the draft, not the reverse. A user
  item that the model invented is caught only when the reference comparison
  or a revision diff sees it.

The equivalence check below, and E14-10's scoring against reference
problems, are the backstops.

## Equivalence with a reference problem

`orbitopt.agent.equivalence.semantically_equivalent(compiled, reference)`
probes both problems at the same seeded points. These are 256 samples, up to
64 box corners, and the reference optimum when you pass `reference_x`. The
result is an `EquivalenceReport`. Its `reasons` list every disagreement, and
its `notes` record observations that do not count as a difference.

- Variables must match by name. A renamed variable needs an explicit
  `aliases` entry. The check never guesses a permutation. Kinds, unit
  dimensions and finite bounds must agree, and units of different scale are
  converted.
- Objectives must have the same count and senses. Their values must agree in
  SI at every probe point inside both domains.
- The feasibility verdict of `OptimizationProblem.constraints_satisfied` must
  agree at every probe point. For each reference constraint or finite bound
  that cuts the probe box, a compiled constraint or bound must hold at exactly
  the same points, and the reverse.
- A tool call must match on tool id, arguments (compared numerically), sense,
  `rhs` and unit. A tool without an implementation is replaced by the same
  deterministic opaque function on both sides.

The probes come from the union of both boxes, so a widened finite bound shows
up. A side that is open in one problem and finite in the other is compared
only up to the finite side, and the report records it in `notes`. This
follows the benchmark rule that FC-001 is judged inside the instance box.
Budgets and verification policies are not compared. The check samples. It
reports two problems equivalent when they differ only on a set the probes
miss. It reports an equality split into two inequalities as different.

## Tests

`tests/test_e14_02_compiler.py` runs the 57 cases in
`tests/data/agent/formulation_cases.json`, eight of them in Chinese. The
cases use the E14-10 formulation case format with one extra outcome,
`compile_failed`. Each case scripts a `MockProvider` with realistic replies.
The test compares each outcome with `expected_outcome`, and each compiled
problem with its reference IR through `semantically_equivalent`. Every
catalogue reference IR is also checked against the benchmark's own Python
callables.

These tests exercise the compiler with scripted replies. They say nothing
about how accurately a real language model formulates problems. E14-10
measures that with a real model.
