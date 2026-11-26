# ADR optimization-0001: one typed IR for LLM-assisted optimization (E14-01)

- **Status:** Accepted. The acceptance thresholds below are proposed and frozen here.
- **Date:** 2026-11-26
- **Deciders:** E14 LLM-assisted optimization, E14-01
- **Code:** `src/orbitopt/optimization/`, `src/orbitopt/schemas/optimization-problem-1.0.json`,
  `src/orbitopt/schemas/optimization-result-1.0.json`
- **Evidence:** `tests/test_e14_01_problem_ir.py`, `docs/optimization/ir.md`
- **Consumers:** E14-02 natural-language intake, E14-03 solvers, E14-04 tool registry,
  E14-05 agent runtime, E14-06 verification guard

## Context and problem

Five E14 workers have to agree on what a problem is. An LLM proposes it, a
solver consumes it, tools evaluate parts of it, an agent revises it and a
guard checks the answer against it. If each pair of workers agrees on a shape
of its own, every one of those boundaries becomes a translation layer, and an
agent edit that weakens the user's constraint can hide in a translation.

The contract also has to be safe to evaluate when an LLM wrote it, so
expressions cannot be Python. It has to identify a problem the same way on
every machine. The E12 rendezvous plan digests hashed exact float reprs, and
last-bit noise gave a different digest on each machine. It has to let a guard
tell whether the user's hard constraints survived an agent's revision. And it
has to keep apart four claims that are easy to blur: the solver converged, the
constraints hold, someone else checked the answer, and the answer is provably
optimal.

## Decision

**Frozen dataclasses with a JSON form.** `OptimizationProblem` holds `id`,
`title`, `description`, `variables`, `parameters`, `objectives`,
`constraints`, `budget`, `verification` and `metadata`. `to_dict` and
`from_dict` round-trip exactly. The JSON Schema checks structure only: types,
required fields and enumerations. `validate_problem` owns every value rule and
returns typed `ProblemError(code, path, message)` values. `parse_problem` is
the boundary for untrusted dicts. It raises one `InvalidProblemError` that
carries all errors from the first failing stage. Construction never
validates, so an LLM's malformed proposal can still be built and asked for
its full error list.

**The IR stores what the user said.** The canonical mathematics is
`min f(x; theta)` subject to `g_i <= 0` and `h_j = 0`. A constraint still keeps
`expression <sense> rhs` with sense `<=`, `>=` or `==`, and an objective
keeps `minimize` or `maximize`. `Constraint.residual` maps a constraint onto
the canonical form: `lhs - rhs` for `<=` and `==`, `rhs - lhs` for `>=`. All
constraints are hard. There are no soft penalties in 1.0.

**Variables.** A variable has a `kind`: `continuous`, `integer`, `binary` or
`discrete`. Continuous and integer variables take `lower`/`upper`, and `None`
leaves a side open. Integer bounds must be integers. A binary variable sets
neither. A discrete variable takes a non-empty tuple of distinct numeric
`choices`. Every variable has a `unit` and an optional `initial` that must lie
in its domain. A `Parameter` is a named, finite constant with a unit.

**Expressions are a whitelisted subset, not Python.** A source string must
parse to numeric literals, names, `+ - * / **`, unary `-`/`+` and calls to 17
fixed math functions. Calls are positional only, and each function has a
fixed arity. Attributes, subscripts, strings, booleans, comparisons, lambdas,
comprehensions, keywords and names starting with `_` are rejected with a
specific `ErrorCode`. The limits are 2000 characters, 500 nodes, nesting
depth 200 and literal exponents of at most 100. Integer literals become floats
before compilation, so `9**9**9` overflows at once and cannot spin up a huge
integer. Only the validated tree is compiled, and it runs with no builtins.
The raw text never reaches `eval`. The same code path evaluates plain floats
and numpy arrays, and numpy is imported only when an array is passed in. The
alternative is a `ToolCall(tool_id, args)`. It references a registered tool,
and each argument is itself a safe expression.

**Units: an in-repo SI table, dimension-checked.** astropy is present in the
conda compute environment but is not a base dependency. A validator that
behaved differently with and without it would repeat the E12 mistake in
another place. `units.py` parses unit strings with powers and parentheses
over 27 symbols and tracks the six SI base dimensions with rational exponents.
Angles are dimensionless, as in SI. The test suite cross-checks every symbol
against astropy where astropy is installed. Validation rejects a sum of
different dimensions, a transcendental function of a dimensioned argument, and
an objective or constraint whose expression dimension differs from its
declared `unit`. Evaluation converts names to SI-coherent values first, so
`x [km] + y [m]` is correct, then converts each result back to its declared
unit.

**Provenance and locking.** Every objective and constraint has an `origin`:
`user`, `compiler` or `agent`. `locked` defaults to `origin == "user"`. Only
user constraints may be locked. A user may unlock one of theirs, and an agent
may then revise it. `hard_constraint_fingerprint()` digests the user
objectives and the locked constraints. It also digests every variable domain
(kind, bounds, choices, unit) and every parameter (value, unit). The task
asked only for the constraints and the objective. That is not enough: an agent
that raises `dv_max` from 4000 to 9000 m/s weakens `dv <= dv_max` without
touching its text, and widening a bound is a weakened hard constraint too. The
fingerprint is order-insensitive and ignores initial guesses, budget,
verification policy, descriptions, metadata and non-user items. An agent can
therefore add a constraint that tightens the problem and keep the fingerprint.

**Digest.** The digest is the SHA-256 of canonical JSON: sorted keys, compact
separators, `allow_nan=False`. Every number is rounded to 12 significant
digits first, and `-0.0` becomes `0.0`. `problem.digest()` covers every
semantic field and leaves out `title`, the descriptions and `metadata`.
Variable, objective and constraint order is part of the digest because it
fixes the solver's vector layout. A pinned digest in the test suite holds the
canonical form to this ADR. The worked example's digest is the same on
macOS arm64 and Linux x86_64. The residual risk: noise that straddles a
12th-digit rounding boundary still splits two digests. Compare the numbers
before calling such a split a real change.

**Results keep four independent statuses.** `OptimizationResult` carries
`problem_digest` and `solver_id`, a `Termination` (`converged`,
`max_iterations`, `max_evaluations`, `timeout`, `infeasible`, `error`,
`cancelled`), and the booleans `solver_converged`, `constraints_satisfied`,
`independently_verified` and `optimality_certified`. All four default to false
and nothing derives one from another. It also records the best `x`, objective
values, canonical residuals, evaluations, iterations, wall time, a message and
an `evidence` list. Each evidence entry has a kind (`solver`, `analytical`,
`reference_solution`, `alternate_solver` or `optimality_certificate`), a
summary and data.

## Acceptance thresholds (proposed, frozen here)

`VerificationPolicy` defaults are the E14 acceptance thresholds unless a
problem states its own:

| Threshold | Default | Meaning |
|---|---|---|
| `required` | `solver_converged`, `constraints_satisfied`, `independently_verified` | Statuses that must be true for acceptance. `optimality_certified` is opt-in because most solvers cannot prove it. |
| `constraint_tol` | `1e-6` | A constraint or bound holds when its violation is at most `1e-6 * max(1, abs(rhs or bound))`. An integer must be within `1e-6` of an integer. `OptimizationProblem.constraints_satisfied` is the single implementation. |
| `objective_rtol` | `1e-6` | An independent check agrees when `abs(f - f_ref) <= 1e-6 * max(1, abs(f_ref))`. |
| `optimality_gap` | `1e-4` | The relative gap an optimality certificate must close. |
| `independent_methods` | `analytical`, `reference_solution`, `alternate_solver` | The methods allowed to set `independently_verified`. |
| Digest quantisation | 12 significant digits | Changing it is a new IR version. |

## Consequences

- The IR imports nothing else from orbitopt and needs nothing beyond
  jsonschema. A test asserts this.
- A rule appears twice only where both layers need it: enumerations live in
  the schema and in `validate_problem`, because Python-built problems never
  pass through the schema. Every other rule lives in `validate_problem` alone.
- A tool's output unit cannot be checked here. The IR trusts the declared
  `unit` of a `ToolCall` objective or constraint, and the E14-04 registry has
  to check it against the tool's declared output.
- `rhs` is a number with the constraint's unit. A constraint against a
  parameter is written with the parameter on the left, as in `dv - dv_max <= 0`.
- Discrete choices are numeric. Categorical choices would need string-valued
  variables, which arithmetic expressions cannot use. They are out of scope
  for 1.0.
