# Optimization IR

`orbitopt.optimization` describes an optimization problem as typed, frozen
data. The same description serves the natural-language front end, the solvers,
the scientific tools, the agent runtime and the verification guard. Each
problem has a JSON form (schema `optimization-problem-1.1.json`) and a digest
that comes out the same on every machine. The decisions behind the shape are
in [ADR optimization-0001](../adr/optimization-0001-optimization-ir.md), and
its IR 1.1 addendum records what changed from 1.0.

## What a problem holds

| Part | Meaning |
|---|---|
| `variables` | `continuous` and `integer` take `lower`/`upper` (`null` leaves that side open), `binary` is {0, 1}, `discrete` takes one of `choices`. Each has a `unit` and an optional `initial` guess. |
| `parameters` | Fixed named constants with units. |
| `objectives` | One or more. Each has an expression, a `sense` (`minimize` or `maximize`) and a `unit`. |
| `constraints` | Hard constraints `expression <sense> rhs` with `sense` one of `<=`, `>=`, `==`. The `unit` covers both sides. |
| `budget` | Optional limits: `max_evaluations` and `max_wall_s`, which every solver enforces, and `max_iterations` (agent loop iterations) and `max_tokens` (LLM tokens), which only the agent runtime reads. No budget field sets a backend's own iteration limit. |
| `verification` | Which result statuses acceptance requires, plus the thresholds behind them. |

## Expressions

An expression is either a safe arithmetic string over declared names, such as
`"x**2 + y**2"`, or a tool reference. The string may use numbers, names,
`+ - * / **`, unary minus and the functions `abs sqrt exp log log10 sin cos
tan asin acos atan atan2 sinh cosh tanh min max`. Anything else is rejected
before it runs: attributes, strings, comparisons, lambdas, keyword arguments,
names starting with `_`, and literal exponents over 100.

Two forms state a long sum without writing it out:

- `x[k]` reads the declared scalar `x<k>`, so `x[3]` is the variable `x3`.
- `sum(<term> for i in range(a, b))` adds `<term>` for each integer `i` from
  `a` to `b - 1`. `range(n)` starts at 0. Sums nest, and a sum index used as a
  value is that number.

An index or range bound is integer arithmetic (`+ - *`) over integer literals
and enclosing sum indices. The 20-dimensional Rosenbrock function is one
expression:

```text
sum(100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(19))
```

A parse expands every sum and index into plain syntax and checks the result
with the same whitelist, so evaluation, units and `classify` never see a
sum. Two limits keep the work bounded, and both count every node of Python's
syntax tree, operators and name contexts included:

| Limit | Value | What it counts |
|---|---|---|
| `MAX_NODES` | 500 | The source tree as written, before expansion. Its nesting depth is at most 200. |
| `MAX_EXPANDED_NODES` | 20000 | The tree after every sum and index is expanded. Expansion counts as it builds and stops at the limit, so `range(10**9)` costs no more than the limit. |

`sum` and `range` are reserved: no variable or parameter may take either name.
`Expression.tree` returns the expanded tree, for code that reads expression
structure.

A tool reference is `{"tool_id", "args", "constants", "output"}`:

| Field | Meaning |
|---|---|
| `tool_id` | A registered tool, such as `numerics.quadrature_cost.v1`. |
| `args` | Tool input to an expression source, evaluated in SI units. |
| `constants` | Tool input to a JSON constant passed unchanged: a string, a boolean, a finite number or an array of these. It shares no key with `args` and is part of the digest. |
| `output` | The output field read from a tool that returns an object. `null` for a tool that returns a number. |

Every failure in or around a tool reaches the caller as `ToolCallError`, a
subclass of `EvaluationError`, with the tool's own exception as its cause.

Units are checked by dimension. A sum of a length and a time is rejected, and
so is a constraint whose expression is a length when its `unit` is a time.
Values are converted to SI before evaluation, so `x [km] + y [m]` evaluates
correctly.

## Tolerances

A constraint holds when its violation is at most
`constraint_tol * tolerance_scale(rhs)`. The scale is `abs(rhs)`, or 1 when
`rhs` is 0. Bounds and discrete choices use the same rule, and an integer
must lie within `constraint_tol` of an integer. With the default
`constraint_tol = 1e-6`, `error <= 1e-10` accepts errors up to
`1e-10 * (1 + 1e-6)`, and `g <= 0` accepts `g` up to `1e-6`.
`Constraint.allowance(tol)` and `Variable.domain_violations(value, tol)` state
the rule once. `domain_violations` returns why a value is outside a domain:
`below_lower`, `above_upper`, `not_integer` or `not_a_choice`.

## Origin and the fingerprint

Every objective and constraint records its `origin`: `user`, `compiler` or
`agent`. User constraints are `locked` by default.
`hard_constraint_fingerprint()` changes whenever a user objective, a locked
constraint, a variable domain or a parameter changes, so a guard can catch an
agent that weakens what the user asked for.

## Worked example

Minimise x² + y² subject to x + y ≥ 1 and x, y ≥ 0. The optimum is 0.5, at
x = y = 0.5.

```json
{
  "schemaVersion": "1.1",
  "id": "doc-quadratic-halfplane",
  "title": "Closest point to the origin on x + y >= 1",
  "variables": [
    {"name": "x", "lower": 0},
    {"name": "y", "lower": 0}
  ],
  "objectives": [
    {"name": "distance_sq", "expression": "x**2 + y**2"}
  ],
  "constraints": [
    {"name": "halfplane", "expression": "x + y", "sense": ">=", "rhs": 1}
  ]
}
```

```python
import json
from orbitopt.optimization import parse_problem

problem = parse_problem(json.load(open("problem.json")))   # raises InvalidProblemError with every error
point = {"x": 0.5, "y": 0.5}
ev = problem.evaluate(point)
ev.objectives                                     # {'distance_sq': 0.5}
ev.residuals                                      # {'halfplane': 0.0}  (g = rhs - lhs <= 0 holds)
problem.constraints_satisfied(point, ev.residuals)  # True
problem.digest()                                  # identical on every machine
```

`validate_problem(problem)` returns the error list without raising. Each error
has an `ErrorCode`, a JSON-pointer `path` and a message.

## Results

A solver reports its answer as an `OptimizationResult`. That result carries
the problem's digest and four separate status booleans: `solver_converged`,
`constraints_satisfied`, `independently_verified` and `optimality_certified`.
All four start out false. It also holds:

| Field | Meaning |
|---|---|
| `termination` | `converged`, `max_iterations`, `max_evaluations`, `timeout`, `infeasible`, `error`, `cancelled`, `unbounded` (the backend reports the objective improves without bound) or `stalled` (the backend stopped making progress before its convergence test passed). |
| `front` | The non-dominated feasible points of a multi-objective solve, each a `ParetoPoint(x, objective_values)`. Empty otherwise. |
| `provenance` | `Provenance(cache_key)`: the problem digest a cache stored the result under, or `None` for a fresh solve. It is part of the result, so the result digest pins it. |
| `wall_time_s` | Wall-clock seconds of the solve. It is volatile, so the result digest leaves it out (`VOLATILE_KEYS`). |
| `evidence` | Each item has a `kind`, a `summary`, `data`, its `producer` (a solver id, tool id or method) and the `problem_digest` it was produced for. |

## Versions and digests

`to_dict` writes `schemaVersion` `"1.1"`. `parse_problem` checks a document
against the schema its `schemaVersion` names (`PROBLEM_SCHEMAS`), so a 1.0
document still parses. Every 1.1 field it lacks takes its default: no tool
constants, no output, an empty front and no cache key. The result side works
the same way through `RESULT_SCHEMAS`. A 1.0 document that names a variable
`sum` or `range` no longer validates.

Two 1.1 rules change what an unchanged 1.0 document means:

- Constraints and bounds with `0 < abs(rhs) < 1` are graded more strictly,
  because the tolerance scale is `abs(rhs)` and no longer `max(1, abs(rhs))`.
- The digest hashes the 1.1 form, which includes `schemaVersion`. A problem's
  1.1 digest therefore differs from the digest the same document had under
  1.0, and the test suite pins the 1.1 value. Digests still round every number
  to 12 significant digits, so they agree across machines.
