# Verification guard

`orbitopt.optimization.verification` grades an `OptimizationResult` against
the problem it should answer. It never trusts what the result says about
itself. The four booleans on a result (`solver_converged`,
`constraints_satisfied`, `independently_verified`, `optimality_certified`)
are claims. The guard derives each status on its own, and a claim that it
cannot confirm is itself a finding. The problem IR is described in
[Optimization IR](ir.md).

## Grade a result

```python
from orbitopt.optimization.verification import ReferenceSolution, result_digest, verify_result

evidence = ReferenceSolution(id="doc", value=0.5, source="worked example",
                             problem_digest=problem.digest(), result_digest=result_digest(result))
report = verify_result(problem, result, evidence=[evidence])
report.verdict                                      # Outcome.PASSED
report.statuses["independently_verified"]           # StatusOutcome(passed, reason, evidence_ids=("doc",))
report.codes                                        # finding codes, empty here
```

`verify_result(problem, result, *, policy=None, evidence=(), artifact_digest=None, solved_problem=None, tools=None, seed=0)`
never raises. An error inside the guard becomes a `guard_error` finding and a
failed verdict.

| Argument | Meaning |
|---|---|
| `policy` | Defaults to `problem.verification`. It names the required statuses and the tolerances. |
| `evidence` | Evidence items for `independently_verified` and lower-bound certificates. The guard checks each item itself. |
| `artifact_digest` | The digest recorded for the result when it was stored. It must equal `result_digest(result)`. |
| `solved_problem` | The problem the solver actually ran, when it differs from the graded one. A cache hit is not an argument: it is `result.provenance.cache_key`, which the result digest covers. |
| `tools` | Resolvers for `ToolCall` objectives and constraints, by tool id. |
| `seed` | The seed for the samples that check a lower-bound certificate. |

`result` may also be the dict form. The guard checks it against
the result schema its `schemaVersion` names (`optimization-result-1.1.json`,
or 1.0 for an older document) first, so a missing field gives
`malformed_result`.

## How each status is derived

Each status is `passed`, `failed` or `not_assessed`, with a reason and the
evidence ids it rests on.

| Status | `passed` when | `failed` when | `not_assessed` when |
|---|---|---|---|
| `solver_converged` | `termination` is `converged`. | Any other termination. | Never. |
| `constraints_satisfied` | The guard re-evaluates `result.x` through `OptimizationProblem.evaluate`, and every bound, integrality rule, discrete choice and hard constraint holds within `constraint_tol`. `OptimizationProblem.constraints_satisfied` must agree. | A value is missing, not a number or not finite, or a domain or constraint is violated, or the problem is not defined at the point. | A `ToolCall` has no resolver, or a tool raised. |
| `independently_verified` | At least one allowed evidence item validates against this result and agrees with the re-evaluated objective. No validated item disagrees. | A validated item disagrees. | No evidence item validates. |
| `optimality_certified` | The KKT certificate passes and no lower-bound certificate is refuted. | The point is infeasible, the KKT check fails, or a lower-bound certificate is refuted. | Not requested, the problem is outside the certifiable class, or the point could not be evaluated. |

The guard attempts `optimality_certified` when the policy requires it, when
the result claims it, or when a lower-bound certificate is supplied.

## Verdict

The verdict is `passed` only when every status in `policy.required` passed
and no finding is fatal. It is `failed` when a required status failed or any
finding is fatal. Otherwise it is `not_assessed`. A policy that requires no
status never passes.

These checks give fatal findings, whatever the statuses say:

- `problem_mismatch`: `result.problem_digest` is not the digest of the problem
  that the result claims to answer. That problem is `solved_problem` when
  given, and the graded problem otherwise.
- `unauthorized_relaxation` and `formulation_mismatch`: the solved problem's
  `hard_constraint_fingerprint()` differs from the graded problem's. A widened
  domain, a dropped or unlocked user constraint, or a loosened right-hand side
  is a relaxation. Any other change to a user objective, locked constraint,
  variable or parameter is a formulation mismatch. A revision that keeps the
  fingerprint, such as an added agent constraint, is noted as
  `revised_problem`. The point is then graded against the original problem.
- `stale_cache`: a cache hit on an entry stored under another problem digest.
- `digest_mismatch`: `artifact_digest` or a sealed file does not match.
- `objective_mismatch`, `residual_mismatch` and `non_finite`: a claimed value
  differs from the guard's re-evaluation, or is NaN or infinite.
- `constraint_violation` and `domain_violation`: the point breaks a bound or
  constraint, or an integrality rule or discrete choice.
- `claim_contradicted`: the result claims a status that the guard did not pass.
- `forged_verification`: an evidence item comes from the result's own solver.
- `malformed_result`, `malformed_artifact`, `missing_file`, `invalid_problem`
  and `guard_error`.

Notes such as `missing_evidence`, `evidence_mismatch`, `evidence_disagrees` or
`certificate_unsupported` explain a status. They do not fail the verdict by
themselves. They still block a pass when the status they explain is required.
`FindingCode.fatal` tells the two kinds apart.

## Evidence

Every evidence item names `problem_digest` (`problem.digest()`) and
`result_digest` (`result_digest(result)`). An item with a missing digest, a
digest of another problem or result, or a kind that
`policy.independent_methods` does not allow never counts. Evidence inside
`result.evidence` is the solver vouching for itself. The guard ignores it and
records `self_attestation`.

| Type | Kind | What the guard checks |
|---|---|---|
| `AnalyticalOptimum` | `analytical` | A non-empty `derivation`. When `x` is given, the guard checks that `x` is feasible and attains `value`. Then it compares `value` with the re-evaluated objective within `objective_rtol * max(1, abs(value))`. |
| `ReferenceSolution` | `reference_solution` | A non-empty `source`. The tolerance is `min(tolerance, objective_rtol * max(1, abs(value)))`, so a reference can tighten the policy but never loosen it. |
| `AlternateSolverCheck` | `alternate_solver` | The guard calls `solver(problem)` itself. The returned result must answer the graded problem and come from the declared `solver_id`, which must differ from the graded result's. Its point must be feasible on re-evaluation. A better objective than the result's refutes the result. A worse one is inconclusive. |
| `LowerBoundCertificate` | `optimality_certificate` | See the next section. |

`AlternateSolver` is a protocol: any callable that takes an
`OptimizationProblem` and returns an `OptimizationResult`. The guard ships no
solver of its own.

## What the guard can certify

`optimality_certified` passes only through the KKT certificate
(`kkt_certificate`). The certificate applies to a problem with these
properties:

- It has exactly one objective, and every variable is continuous.
- The objective is structurally a polynomial of degree 2 or less in the
  variables. Parameters and literals count as constants. The Hessian of the
  objective is positive semidefinite when minimising and negative
  semidefinite when maximising. Its smallest eigenvalue must be at least
  `-1e-8` times its largest entry.
- Every constraint is structurally affine.
- Nothing is evaluated through a `ToolCall`.

This class covers linear programs and convex quadratic programs. For them,
the KKT conditions are sufficient for global optimality. The guard computes
gradients by central differences with step `max(1, abs(x_i))`, which is exact
for quadratics up to rounding. An inequality is active when its residual is
within `10 * constraint_tol * tolerance_scale(rhs)` of zero. The multipliers come
from a least-squares solve on the active set. While an inequality multiplier
is negative, that constraint leaves the set and the solve repeats. The point
passes when the stationarity residual is at most
`optimality_gap * max(1, max abs(grad f))` and complementarity is at most
`optimality_gap * max(1, abs(f))`. Primal feasibility comes from
`constraints_satisfied`.

Every other problem is `not_assessed`: non-convex objectives (Rosenbrock,
Rastrigin, an indefinite quadratic), mixed-integer and discrete problems,
multi-objective problems and tool-evaluated problems. **This guard never
certifies global optimality of a non-convex problem**, even when the point is
the known optimum.

A `LowerBoundCertificate` states a bound and its derivation. The guard checks
that the result attains the bound within `optimality_gap`. It also tries to
refute the bound on seeded samples of the feasible set: half of the samples
cover the variable box, and half are small perturbations of the result's
point. A refuted bound or a bound that is not attained fails the status.
Sampling cannot prove a bound, so a bound that survives does not pass the
status. On a convex problem the KKT certificate decides. On any other problem
the status stays `not_assessed`. Sampling can also miss a small basin, as for
a local minimum of Rastrigin. The certificate is therefore a refutation
check, not a proof.

## Sealed runs

`seal_run(problem, result, evidence, run_dir)`
writes four files and returns the manifest digest:

| File | Content |
|---|---|
| `problem.json` | `problem.to_dict()` for the problem the solver ran. |
| `result.json` | `result.to_dict()`, whose `provenance` records a cache hit. |
| `evidence.json` | The evidence items. An `AlternateSolverCheck` is stored without its callable. |
| `manifest.json` | Format `orbitopt.optimization.run/2`, the `sha256:` digest of each file's exact bytes, `problem.digest()` and `result_digest(result)`. Format 1 kept the cache record in `evidence.json`; a format 1 run is refused as malformed. |

Every file is canonical JSON: sorted keys, compact separators, UTF-8, and no
NaN or infinity. `verify_run(run_dir, *, problem=None, policy=None, alternate_solvers=None, tools=None, expected_manifest_digest=None, seed=0)`
re-derives everything from the files. A missing file, a changed byte or a
digest mismatch gives a failed report with the finding, and the guard grades
nothing from a modified run. Pass `problem` to grade against the problem that
was asked. Without it, the run is graded against its own `problem.json`, so
whoever wrote the directory chooses the problem. Anyone who can rewrite the
directory can also rewrite the manifest. Keep the digest that `seal_run`
returns, and pass it as `expected_manifest_digest`.

The manifest records the SHA-256 of exact bytes, not the 12-digit quantised
digest that the IR uses. A tamper seal must catch a change in the 13th digit
of `x`. The quantised `result_digest` identifies a result for evidence
pinning, and it is not a tamper seal.

## Fault matrix

`tests/test_e14_06_verification_guard.py` applies all nine injectors of
`benchmarks/e14_spec/faults.py` to an honest artifact of each of the 25
catalogue instances that the IR can express. The other instances use black
boxes, have several objectives or store no optimiser. Each faulty artifact
is graded twice, once with `verify_result` and once from a sealed run with
`verify_run`. Of the 450 graded cells, none passes, and each one names the injector's reason
among its findings. Rerun the matrix with this command:

```bash
PYTHONPATH=src python -m pytest -q tests/test_e14_06_verification_guard.py -k "fault_is_never or honest_benchmark"
```
