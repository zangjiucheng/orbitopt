# Agent iteration runtime

`orbitopt.agent.runtime` runs a bounded optimization loop on an
[IR problem](../optimization/ir.md). A planner picks the solver, its options,
the start point, the seed and the evaluation share of each step. It cannot
touch the problem. The runtime checks every proposal before it runs, grades
every candidate with the [verification guard](../optimization/verification.md),
verifies the final answer with alternate solvers, and writes a trace that
`replay` re-executes without an LLM call.

```python
from orbitopt.agent.providers import LocalProvider
from orbitopt.agent.runtime import LLMPlanner, RuleBasedPlanner, replay, save_run, solve

outcome = solve(problem, planner=RuleBasedPlanner(), seed=0)
outcome.status              # RunStatus.VERIFIED, UNVERIFIED_BEST, NO_FEASIBLE, ...
outcome.result              # the best OptimizationResult, or None
outcome.statuses            # the final report's four statuses, e.g. {"solver_converged": "passed", ...}
digest = save_run(outcome, "runs/worked")      # keep this digest out of band
replay("runs/worked", expected_manifest_digest=digest).equal

outcome = solve(problem, planner=LLMPlanner(LocalProvider("qwen2.5:7b")), seed=0)
```

`solve(problem, *, planner, budget=None, policy=None, tools=None, seed=0, cancel=None, registry=None)`
never raises for a bad proposal or a failing solver. A problem it cannot
take faithfully comes back `refused`.

| Argument | Meaning |
|---|---|
| `budget` | An IR `Budget` that overrides `problem.budget` field by field. |
| `policy` | What the planner may change. See [The policy](#the-policy). |
| `tools` | A `ToolView` or a mapping of resolvers by tool id. |
| `cancel` | A `threading.Event` or a callable, checked before each iteration. |
| `registry` | The solvers. Defaults to `default_registry()`. |

The verification policy is always `problem.verification`. No argument loosens it.

## Limits

Every limit is finite, so a run can never retry without bound. `Limits.resolve`
takes each limit from `budget`, else from `problem.budget`, else the E14-10
default.

| Limit | Default | What counts |
|---|---|---|
| `max_iterations` | 20 | Planner proposals, accepted or refused. |
| `max_evaluations` | 2000 × max(n, 2) | Every evaluation of the problem made for the run: solvers, the guard and alternate solvers. |
| `max_wall_s` | 600 s | Wall clock from the start of `solve`. |
| `max_tokens` | 200,000 | LLM tokens, prompt and completion. |

`max_iterations` limits the loop, and it is the only meaning
`Budget.max_iterations` has. The runtime does not pass it to a solver. A
decision sets a backend's own iteration cap itself, as the proposal field
`max_iterations`, which the runtime hands to the solver as
`solve(..., max_backend_iterations=...)`.

Every evaluation goes through one `EvaluationMeter`. The meter is a copy of
the problem whose `evaluate` counts each call, so solvers and the guard use
it as the problem itself. It refuses an evaluation past the current solve's
allowance by raising `EvaluationBudgetExceeded`, which is not an
`EvaluationError`, so a solver that ignores its `Budget` stops instead of
skipping the point. The adapters' own cache of a point's objective and
constraints is the one documented cache: a cache hit costs nothing, and any
other repeated point counts again.

A quarter of `max_evaluations` (`VERIFICATION_SHARE`) is held back for the
final verification. The loop ends with `max_evaluations` once a decision
could spend fewer than 8 evaluations. Tokens are checked before each planner
call, so the last call can overrun the token limit by its own size. The
runtime caps that call's reply at the remaining tokens, but it cannot know the
prompt's size in advance. The wall clock stops a solver at its next
evaluation, and the loop between iterations. A run that runs out of wall
clock skips the final verification.

## The policy

`Policy` lists what the planner may change. Nothing on it reaches the problem.

| Field | Default | Meaning |
|---|---|---|
| `allowed_solvers` | `None` | Solver ids a decision may name; `None` allows every available, compatible solver. |
| `allow_options` | `True` | Whether a decision may set solver options. |
| `allow_x0` | `True` | Whether a decision may set a start point. |
| `on_violation` | `"continue"` | `"stop"` ends the run at the first refused proposal. |
| `max_violations` | 3 | The run ends after this many refused proposals. |

## Proposals and decisions

A planner returns a `Proposal`: the raw decision object, the tokens it cost,
or the reason it has none. Every proposal is untrusted, the rule-based
planner's included. `validate_proposal` turns it into a `Decision` or into
the list of reasons it is refused. A refused proposal becomes a `rejected`
record in the trace and is never applied. No field of a `Decision` can carry
a problem change, so a change that slips past validation has nowhere to go.

A proposal may name these fields only: `action` (`solve` or `stop`),
`solver_id`, `options`, `x0`, `seed`, `evaluations`, `max_iterations` and
`rationale`.

| Violation | When |
|---|---|
| `problem_edit` | A field or an option names part of the problem, its budget or its verification policy: `objective`, `constraints`, `bounds`, `variables`, `parameters`, `budget`, `max_evaluations`, `verification`, `constraint_tol` and the rest of `PROBLEM_FIELDS`. |
| `unknown_field` | Any other field outside the list. |
| `malformed` | Not an object, a bad `action`, a non-integer seed, or an LLM reply that failed the schema twice. |
| `unknown_solver` | No registered solver has this id. |
| `solver_not_allowed` | The solver is registered but not available, not compatible with the problem, excluded by the run, or not allowed by the policy. |
| `invalid_option` | The solver does not declare the option, the value is outside the option's `OptionSpec`, or the policy allows no options. |
| `x0_unknown_variable`, `x0_out_of_domain` | A start point names an unknown variable or lies outside a variable's domain (bounds, integrality, choices). |
| `budget_overreach` | `evaluations` beyond what one decision may spend, or a backend `max_iterations` above 100,000 (refused as `invalid_option` for a solver without `iteration_cap`). |

Each solver declares its options in `SolverCapability.options`, an
`OptionSpec` per name with a type, a closed range or choices, and a
`tolerance` flag (see [solvers](../optimization/solvers.md)). The runtime
validates against those specs and keeps no table of its own. The solver
tolerances (`ftol`, `gtol`, `tol`, `tolfun`, `tolx`, `mip_rel_gap`) are the
only values a remedy may loosen.

## One iteration

1. Check cancellation and every limit.
2. Ask the planner for a proposal and add its tokens.
3. Validate it. A refused proposal is recorded and the loop goes on, unless the policy stops it.
4. A `stop` decision ends the loop.
5. Run the solver on the metered problem with the decision's allowance, seed, options and start point. A solver that raises is a failed iteration.
6. Screen the result: `verify_result(problem, result, solved_problem=problem)` with no independent evidence.
7. Diagnose the result. A solver whose report the guard caught lying, or that crashed, is excluded for the rest of the run.

## Verification

The screen reads the four statuses separately and never trusts the result's
own booleans. A candidate is eligible when the guard's `constraints_satisfied`
passed and nothing in the report shows the solver lied. `distrust` names a
lie: a finding in `DISHONEST` (a mismatched objective or residual, another
problem's digest, a stale cache, a result that vouches for itself), a claim
the guard failed, or any claim to `independently_verified` or
`optimality_certified`, which only the guard sets. A claim the guard could
not assess, such as a tool that raised, is not a lie.

The best eligible candidate has the best objective in its sense, then a
passed `solver_converged`, then the earliest iteration. After the loop the
runtime verifies it with the whole guard:

- Up to two alternate solvers, every compatible solver except the candidate's
  own and the excluded ones, go in as `AlternateSolverCheck` evidence pinned
  to `result_digest(result)`. The guard runs them itself, each with an equal
  share of the evaluations left. The policy's solver list does not apply to
  them, since choosing the verification is not the planner's job.
- When an alternate finds a better feasible point, it refutes the candidate.
  Its point is screened and verified once more (`MAX_VERIFICATION_ATTEMPTS = 2`).
- No alternates run when the policy does not require `independently_verified`
  or does not allow `alternate_solver` evidence.

`capability.convergence` decides what a convergence is worth. A
differential-evolution convergence is a `heuristic_stop`, never an optimum.
A first-order point is final only on a class that `classify` proves convex
(`lp`, `convex_qp`). A `certificate` convergence is final on `lp`, `convex_qp`
and `milp`.

| Status | When |
|---|---|
| `verified` | The final guard report passed. |
| `unverified_best` | A feasible best exists, but the guard did not pass it. |
| `no_feasible` | No eligible candidate. `result` is the honest result with the least violation, if any. |
| `budget_exhausted` | A limit ended the loop and the best is not verified. |
| `cancelled` | The cancel check fired. The final verification is skipped. |
| `policy_violation` | The policy ended the run on a refused proposal. The final verification is skipped. |
| `refused` | Invalid IR, a tool id outside `known_tool_ids(tools)`, a tool binding that fails its unit check, or no solver that can take the problem. Nothing is evaluated. |

`outcome.report` is the final verification of `outcome.result`, or its
screen when no final verification ran. `outcome.statuses` gives its four
statuses.

## Diagnostics and remedies

`diagnose(result, report, capability, structure)` names a `FailureKind`.
`REMEDIES` maps each kind to bounded remedies, most promising first. No
remedy touches the problem.

| Failure kind | When | Remedies |
|---|---|---|
| `verified_local` | A certificate, or a first-order point of a provably convex problem. | stop |
| `local_optimum` | A first-order point of a problem that is not provably convex. | restart global, restart perturbed, stop |
| `heuristic_stop` | A heuristic converged. | refine locally, restart global |
| `non_convergence` | The backend's iteration limit. | refine locally, loosen solver tolerance, switch solver |
| `stalled` | A failed line search or a stagnation. | loosen solver tolerance, restart perturbed, switch solver |
| `infeasible` | No feasible point found, or a proof of infeasibility. | switch solver, change initial guess, restart global |
| `unbounded` | HiGHS proved the objective unbounded. | stop |
| `evaluation_error` | Every evaluation failed. | change initial guess, switch solver |
| `solver_error` | The solver raised, or reported an error. | switch solver |
| `budget` | The solve ran out of evaluations or time. | refine locally, stop |
| `rejected_by_guard` | `distrust` found a lie. | switch solver |

Loosening multiplies each solver tolerance by 100, from the option's value or
its `OptionSpec` default, and stops at the spec's upper bound. The
rule-based planner loosens at most twice per run.

## Planners

`RuleBasedPlanner` is deterministic and uses no tokens. Its ranking is the
registry's, most specialised first, on `lp`, `convex_qp` and `milp`. On every
other class it puts a global solver first, because no syntax test can rule
out several minima. A global solve is followed by a local refinement from its
point. Up to `max_restarts` (3) further rounds use a new seed each, and the
restarts end at the first round that does not beat the earlier best by more
than `objective_rtol`. A failed solve takes the first remedy of its diagnosis
that the planner can still carry out. A population solver gets a population
that covers about 150 generations of its allocation, 5 to 15 members per
variable.

`LLMPlanner(provider)` asks a model for the same JSON object through
`generate_structured`. The prompt holds the problem's class and domains, the
solvers with their option ranges, the remaining iterations, evaluations and
tokens, the incumbent and a summary of every earlier iteration. It holds no
wall-clock value, so a `ReplayProvider` cassette replays it. `PROPOSAL_SCHEMA`
leaves extra fields open on purpose, so that a reply that asks to relax a
constraint reaches the runtime's validation and is recorded as a
`problem_edit`. A reply that fails the schema twice is a `malformed`
proposal. Any other provider failure ends the run with `planner_failed`.
`MockProvider` and `ReplayProvider` run it with no network.

`RecordedPlanner(proposals)` serves recorded proposals in order. `replay`
uses it.

## Run directories and replay

`save_run(outcome, run_dir)` writes:

| File | Content |
|---|---|
| `problem.json` | The problem asked. |
| `trace.json` | The trace, schema `agent-trace-1.0.json`. |
| `results/<label>.json` | Every result of the run, full, by label (`iteration-0003`, `verify-1-scipy.slsqp`). |
| `sealed/` | `seal_run` of the best result with its final evidence, format `orbitopt.optimization.run/2`. |
| `tool-calls.json` | The tool call log, when the run used a `ToolView`. |
| `manifest.json` | Format `orbitopt.agent.run/1`: the SHA-256 of every file above, the trace digest, and the sealed run's manifest digest. |

The sealed run's manifest digest lives in `manifest.json`, outside `sealed/`,
as `seal_run` asks. `save_run` returns the digest of `manifest.json` itself.
Keep it out of band, since whoever can rewrite the directory can rewrite the
manifest too.

`check_run(run_dir, *, tools=None, registry=None, expected_manifest_digest=None, alternates="recorded")`
is the integrity check. It re-executes nothing.

1. Every file name the manifest gives, and the sealed run's folder, go
   through `safe_relative`. A name with `..`, an absolute path, a backslash
   or a character outside `[A-Za-z0-9_.-]` raises `UnsafePathError` before
   any file besides `manifest.json` is read.
2. Every file must hash to the digest the manifest records, and the
   manifest must seal `problem.json` and `trace.json`. With
   `expected_manifest_digest`, the manifest itself must hash to it.
3. The trace must match its own digest, name the problem in
   `problem.json`, and match the manifest's trace digest.
4. Each result summary in the trace must be the summary of the full result
   saved under its label, and `sealed/result.json` must be byte for byte the
   result the trace names as best.
5. The guard re-grades `sealed/` with `verify_run`, pinned to the sealed
   manifest digest. A verdict that contradicts the trace's status is a
   finding.

With `alternates="recorded"` the guard takes each final-verification
alternate's saved result as that alternate's answer. It re-checks the
alternate's point and the comparison, and runs no solver. It cannot show
that the solver produces that point. `alternates="rerun"` runs each
alternate again with its recorded allocation, seed and options from
`registry`. The guard re-evaluates points either way, so pass `tools` as the
run had them.

`CheckReport` holds `integrity` (one line per finding), the guard's
`report`, the trace's `recorded_status`, the trace, the problem and the
bytes of every file it checked. `intact` is true with no finding.
`verified` is true when it is also intact and the guard passes the sealed
run again. `orbitopt agent verify` runs `check_run` with the alternates run
again.

What `check_run` catches, and what it does not:

- A changed byte, unless the manifest was rewritten to match. Pin the
  manifest with `expected_manifest_digest` to catch that too.
- A forger who rewrites the trace and every digest: the trace no longer
  summarises the saved results, or the sealed result no longer re-evaluates
  to its claims.
- A forger who rewrites a result that is not the best, its trace entry and
  every digest is not caught, since nothing re-runs that solve. `replay` is.

`replay(run_dir, *, tools=None, registry=None, expected_manifest_digest=None)`
runs `check_run` with the alternates run again. It then re-executes the
recorded proposals with a `RecordedPlanner`, on the problem in `problem.json`,
with the recorded seed, limits and policy. It compares the new trace with
the recorded one at 12 significant digits, wall-clock fields left out.
`ReplayReport.check` is that check, and `integrity` and `sealed_verdict`
read from it. `ReplayReport.equal` is true when no file changed and nothing
differs. A
forger who edits a result and recomputes every digest still fails, because
the solvers recompute the result. A solver whose capability sets
`deterministic_given_seed` to false shows up as a difference unless it
reproduces its result.

## The trace

`Trace.to_dict()` validates against `agent-trace-1.0.json`. Each iteration
record holds the raw proposal, the decision or the violations, a summary of
the result, the screen (verdict, the four statuses with reasons, the finding
codes), the diagnosis, and the evaluations and tokens it cost. Each
verification record holds the candidate's label, the alternate runs and the
guard's report. The digest is the IR's `canonical_digest` (12 significant
digits) of the trace without `wall_s`, `wall_time_s` and `latency_s`
(`orbitopt.optimization.VOLATILE_KEYS`, which `result_digest` leaves out too). Two
runs with the same problem, planner, seed and solvers share it. A number that
straddles a 12th-digit rounding boundary still splits it.

## Measured runs

`tests/test_e14_05_runtime.py` runs six catalogue instances with each
planner. The scripted LLM is a `MockProvider` whose function reads the
runtime's prompt and answers: a global or specialised solve first, a local
refinement from the incumbent, restarts with new seeds for a heuristic, then
stop. Measured with seed 0 on one machine (macOS arm64, Python 3.12, scipy
from the `orbitopt-py312` environment):

| Instance | Planner | Status | Objective (reference) | Solves | Evaluations, loop | Evaluations, verification | Limit | Tokens |
|---|---|---|---|---|---|---|---|---|
| cqp-worked-d2 | rule | verified | 0.5 (0.5) | 1 | 7 | 800 | 4000 | 0 |
| cqp-worked-d2 | LLM | verified | 0.5 (0.5) | 1 | 7 | 800 | 4000 | 555 |
| lp-wyndor-d2 | rule | verified | 36 (36) | 1 | 2 | 393 | 4000 | 0 |
| lp-wyndor-d2 | LLM | verified | 36 (36) | 1 | 2 | 393 | 4000 | 617 |
| knapsack-p01-n10 | rule | verified | 309 (309) | 1 | 2 | 1555 | 20000 | 0 |
| knapsack-p01-n10 | LLM | verified | 309 (309) | 1 | 2 | 1555 | 20000 | 737 |
| rastrigin-d5 | rule | verified | 1.4e-14 (0) | 4 | 5678 | 2170 | 10000 | 0 |
| rastrigin-d5 | LLM | verified | 1.4e-14 (0) | 2 | 3757 | 3131 | 10000 | 1038 |
| rosenbrock-d20 | rule | verified | 4.1e-10 (0) | 4 | 26271 | 7926 | 40000 | 0 |
| rosenbrock-d20 | LLM | verified | 1.5e-9 (0) | 2 | 17500 | 12311 | 40000 | 1578 |
| integ-exp-1e-10 | rule | unverified_best | 5 (5) | 3 | 479 | 1 | 4000 | 0 |
| integ-exp-1e-10 | LLM | unverified_best | 5 (5) | 4 | 646 | 1 | 4000 | 1503 |

What limits these numbers:

- On the convex and linear instances the loop costs one solve, and
  verification costs the most. That cost is the alternate differential
  evolution run converging, not the budget: it stops well below its share.
- On Rastrigin and Rosenbrock the loop spends what the planner allots to
  differential evolution, half of what is left per global round, because
  the solve ends on the evaluation limit. The rule-based planner's second
  round does not improve, so its restarts end there.
- The integration instance has one compatible solver, differential
  evolution, so no alternate solver exists and nothing independent can check
  the answer. The runtime reports `unverified_best` with the right value.
- Rastrigin's unshifted optimum is the box centre, where SLSQP starts by
  default. The alternate check there agrees trivially. The shifted variants
  have no such shortcut.
- Tokens are the mock's word counts, prompt plus reply, so they show how the
  prompt grows with the history, not what a real model would bill.

## Tests

`tests/test_e14_05_runtime.py` covers the budgets against an independent
counter (a tool that every evaluation must call), a solver that ignores its
budget, the iteration, token and wall-clock limits, a scripted LLM that asks
to relax a constraint, change the objective, loosen the verification policy,
name an unknown solver, ask for 10^12 evaluations or start outside the
domain, every failure kind with fault-injecting solvers and problems,
cancellation, replay and tamper detection, determinism, and the E14-10 fault
injectors through the runtime. In that check, a faulty solver returns each
of the eight result faults of `benchmarks/e14_spec/faults.py` that a solver
can commit, on a convex QP and an LP, with a scripted and with the
rule-based planner. The ninth, a tampered artifact, is applied to the run
directory. No faulty result is ever reported verified.

```bash
PYTHONPATH=src python -m pytest -q tests/test_e14_05_runtime.py
```
