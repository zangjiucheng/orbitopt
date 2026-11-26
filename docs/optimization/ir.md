# Optimization IR

`orbitopt.optimization` describes an optimization problem as typed, frozen
data. The same description serves the natural-language front end, the solvers,
the scientific tools, the agent runtime and the verification guard. Each
problem has a JSON form (schema `optimization-problem-1.0.json`) and a digest
that comes out the same on every machine. The decisions behind the shape are
in [ADR optimization-0001](../adr/optimization-0001-optimization-ir.md).

## What a problem holds

| Part | Meaning |
|---|---|
| `variables` | `continuous` and `integer` take `lower`/`upper` (`null` leaves that side open), `binary` is {0, 1}, `discrete` takes one of `choices`. Each has a `unit` and an optional `initial` guess. |
| `parameters` | Fixed named constants with units. |
| `objectives` | One or more. Each has an expression, a `sense` (`minimize` or `maximize`) and a `unit`. |
| `constraints` | Hard constraints `expression <sense> rhs` with `sense` one of `<=`, `>=`, `==`. The `unit` covers both sides. |
| `budget` | Optional limits on evaluations, iterations, wall time and tokens. |
| `verification` | Which result statuses acceptance requires, plus the thresholds behind them. |

An expression is either a safe arithmetic string over declared names, such as
`"x**2 + y**2"`, or a tool reference `{"tool_id": "...", "args": {...}}`. The
string may use numbers, names, `+ - * / **`, unary minus and the functions
`abs sqrt exp log log10 sin cos tan asin acos atan atan2 sinh cosh tanh min
max`. Anything else is rejected before it runs: attributes, subscripts,
strings, comparisons, lambdas, keyword arguments, names starting with `_`,
and literal exponents over 100.

Units are checked by dimension. A sum of a length and a time is rejected, and
so is a constraint whose expression is a length when its `unit` is a time.
Values are converted to SI before evaluation, so `x [km] + y [m]` evaluates
correctly.

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
  "schemaVersion": "1.0",
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
has an `ErrorCode`, a JSON-pointer `path` and a message. A solver reports its
answer as an `OptimizationResult`. That result carries the problem's digest
and four separate status booleans: `solver_converged`,
`constraints_satisfied`, `independently_verified` and `optimality_certified`.
All four start out false.
