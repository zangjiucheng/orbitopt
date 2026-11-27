# Solvers

`orbitopt.optimization.solvers` puts every solver behind one call. You hand
the same [IR problem](ir.md) to any compatible solver and always get back an
`OptimizationResult`. Each solver declares what it can take, and
`orbitopt.optimization.classify` derives the problem's structure from the IR,
so the agent runtime and a rule-based selector can choose among solvers
without running anything.

```python
from orbitopt.optimization import Budget, parse_problem
from orbitopt.optimization.solvers import default_registry

registry = default_registry()
problem = parse_problem(data)
for solver in registry.compatible(problem):            # installed and capable, most specialised first
    result = solver.solve(problem, budget=Budget(max_evaluations=4000), seed=1)
    print(solver.capability.solver_id, result.termination.value, result.objective_values)

registry.solve_with(problem, "scipy.highs")            # the same call, by id
```

`solve(problem, *, budget=None, seed=0, x0=None, options=None, tools=None)`
takes `problem.budget` when `budget` is `None`. `x0` maps variable names to a
start, and `tools` maps `ToolCall` ids to callables, as
`OptimizationProblem.evaluate` takes them.

## Structure: what `classify` decides

`classify(problem)` returns a `ProblemStructure`: one `Form` per objective,
the most general constraint form, and whether the problem has integer,
binary or discrete variables, equality constraints and finite bounds on every
variable. The decision is exact on the expression syntax tree. Each
expression is expanded into a polynomial with rational coefficients. Any
subtree that is not a polynomial in the variables stays an opaque atom. No
function is ever sampled.

| Form | Meaning | Examples |
|---|---|---|
| `LINEAR` | degree at most 1 after expansion | `x/2`, `(x+1)**2 - x**2`, `sin(x) - sin(x)` |
| `QUADRATIC` | degree 2 | `x*y`, `x**k` with parameter `k = 2` |
| `SMOOTH` | higher-degree polynomial, or atoms built only from functions smooth on the interior of their domain | `x**3`, `1/x`, `x**0.5`, `exp(x)` |
| `NONSMOOTH` | an atom applies `abs`, `min` or `max` to a variable | `abs(x)`, `max(x, 1)` |
| `BLACK_BOX` | a `ToolCall` | |

The headline `ProblemClass` follows this precedence: `multi_objective`,
`black_box`, then `milp` or `minlp` for integer problems, then `nonsmooth`,
`lp`, `convex_qp` (a quadratic objective whose Hessian passes an exact
rational positive-semidefinite test in its sense, with linear constraints),
`box_smooth` (no constraints besides bounds) and `nlp`.

Decidable, and reported: linearity, quadraticity, convexity of a quadratic
objective, variable kinds, bounds, constraint senses and the objective count.
Not decidable, and never claimed: whether a nonlinear problem is convex or
multimodal, and what a tool computes. There is no "global" class, so the
benchmark's Rastrigin and Ackley instances classify as `box_smooth`. Two
limits keep the answer conservative but never wrong. Transcendental
identities are not used, so `sin(x)**2 + cos(x)**2` reads as `SMOOTH`. An
expansion beyond `MAX_TERMS` (2000) terms becomes an atom. `SMOOTH` does not
promise a derivative where an argument reaches a domain boundary:
`sqrt(x**2)` reads as smooth although it equals `abs(x)`.

## Built-in solvers

| `solver_id` | Takes | Integers | Constraints | Bounds needed | Convergence means |
|---|---|---|---|---|---|
| `scipy.highs` | linear, one objective | yes (discrete expanded to binaries) | yes, with equalities | no | `certificate` |
| `scipy.slsqp` | smooth, one objective | no | yes, with equalities | no | `first_order` |
| `scipy.lbfgsb` | smooth, one objective | no | bounds only | no | `first_order` |
| `scipy.differential_evolution` | anything, one objective | yes | yes, with equalities | yes | `heuristic` |
| `cma.cma_es` | anything, one objective | no | bounds only | yes | `heuristic` |
| `pygmo.nsga2` | anything, several objectives | yes | bounds only | yes | `none` |

`SolverCapability` carries these columns plus `needs_gradients`, `is_global`,
`deterministic_given_seed`, `honors` (the `Budget` fields the backend is
configured from), `options` (the option names `solve` accepts) and
`unavailable_reason`. A solver whose backend module cannot be found stays
registered. Its `unavailable_reason` names the module, `compatible` leaves it
out, and calling it returns `Termination.ERROR` rather than raising. `cma` is
not vendored, so `cma.cma_es` is unavailable until `pip install cma`.
`rejection(structure)` says in words why a solver cannot take a problem.

`compatible` ranks the capable solvers from least to most general by
`max_form`, constraint, equality and integer support, `is_global`, then
`solver_id`. The order does not depend on the order of registration. A
multimodal problem cannot be detected, so for one the first-ranked solver
may be a local method. Choosing a global solver there is the caller's job.

## The contract every adapter keeps

**One evaluation is one point.** Every point a backend asks about goes
through the adapter's `Evaluator`. It refuses the call that would exceed
`max_evaluations` or `max_wall_s` and raises before evaluating, so a budget
is never exceeded by even one evaluation. Finite-difference points count.
The objective and the constraints at the same point count once. HiGHS reads
coefficients, so it spends exactly one evaluation: the IR's evaluation of its
answer.

**Values come from the IR.** A maximised objective is negated for the
backend and reported un-negated. `objective_values`, `constraint_residuals`
and `x` are the IR's own evaluation of the reported point. For one objective
that point is the best evaluated point: feasible first, then lowest
objective, else least violation. `constraints_satisfied` comes only from
`OptimizationProblem.constraints_satisfied`. `independently_verified` and
`optimality_certified` stay false; the verification guard sets them.

**Termination, in order:** `error` (a backend exception, a backend that
reports failure, or every evaluation failing), then `unbounded` (HiGHS
proved the objective unbounded), then the adapter's budget
(`max_evaluations`, `timeout`), then `infeasible` (no feasible point found),
then `converged` (the backend's own flag), then the backend's own iteration
limit or `stalled` (an SLSQP or L-BFGS-B line search that failed, or a
CMA-ES stop that is neither convergence nor a limit), and otherwise `error`
with the backend's message. `solver_converged` is true exactly when the
termination is `converged`. A backend that claims success at an infeasible
point is reported `infeasible`. From HiGHS, `infeasible` is a proof. From
the other solvers it means they stopped without finding a feasible point.

**What `solver_converged` is worth** depends on `capability.convergence`.
`certificate` means HiGHS proved optimality within its gap (default
`mip_rel_gap` 1e-9). `first_order` means a KKT point, which is the optimum
only on a convex problem. `heuristic` means the population stagnated, so the
point can be wrong: with seed 1, differential evolution converges at 1456 on
`knapsack-p07-n15`, whose optimum is 1458. `none` means NSGA-II has no convergence test and
never reports converged.

**Seeds.** The same problem, seed and budget give the identical result for
every built-in solver. The deterministic ones ignore the seed.

**Malformed requests raise.** An invalid problem raises
`InvalidProblemError`, and an unknown option or `x0` name raises
`ValueError`. A well-formed request that cannot be carried out (backend
missing, structure outside the capability) returns `Termination.ERROR` with
the reason and zero evaluations.

**Multi-objective results.** NSGA-II reports the front member with the
lexicographically smallest objective vector as `x`. The whole non-dominated,
feasible front is `OptimizationResult.front`, one `ParetoPoint(x,
objective_values)` per member.

**Evaluation failures.** A point where the problem is undefined, or where a
tool raised (`ToolCallError`), counts as an evaluation and is infeasible, so
the backend moves on. The solver evidence names its `producer` (the solver
id) and the `problem_digest` it solved.

## Acceptance corpus

`tests/test_e14_03_solver_registry.py` translates the 37 E14-10 catalogue
instances into the IR. It runs every compatible available solver on each one
under the catalogue's budget (2000 evaluations per dimension) and scores the
answer with the catalogue's own callables and tolerances. The translation
keeps the catalogue's indexed sums: it inlines `n` and the scalar data and
declares each data vector `s` as parameters `s0, s1, ...`. All 37 instances
are valid IR, `rosenbrock-d20` among them as one indexed sum.
