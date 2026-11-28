# E14-10 qualification report: first campaign

> Superseded by [campaign v2](e14_qualification_report_v2.md), which reran these arms after the runtime fixes for budget typing and tolerance-edge selection. This report and its results stay as recorded.

This is the first scored run of the [E14-10 benchmark specification](e14_benchmark_spec.md). It ran the three arms
that need no language model on all 29 scoring instances with the spec's 10 seeds, and checked the safety metrics.
The four arms and metrics that need a model were not run, because no provider was configured.

**Read this first.** No LLM was called. Arm `E-rule` is the E14 runtime with its deterministic rule-based planner and
its verification guard, run on the hand-written reference problem. It is not the pre-registered arm E, which compiles
the natural-language statement and plans with a model. `E-rule` against A or B measures the runtime's solver
selection, restarts and verification. **It says nothing about whether an LLM helps.**

## 1. The question and the short answer

The spec asks, per analysis group of problem classes, whether an LLM in the loop changes success rate, evaluation
cost or wall clock against the traditional arms, and whether the verification layer ever passes a faulty result.

What this campaign can answer:

- **LLM value: not answered.** Arms C, D and E, and formulation accuracy, were not run (section 3).
- **B against A (primary).** The rule table beats the fixed differential-evolution solver overall, on success
  (+0.255 success rate, 95 % CI [0.19, 0.33], Holm p = 0.0004 of 15), evaluations (ERT ratio 0.23, CI [0.17, 0.30])
  and wall clock (ERT-time ratio 0.35, CI [0.29, 0.40]). The whole effect sits in the algebraic group (LP, convex
  QP, MILP), where exact solvers exist. The spec predicted this as a harness sanity check, not a finding.
- **E-rule against B (secondary).** Not shown different on success (-0.04, CI [-0.13, 0.05]). E-rule is inferior
  on evaluations (11.3x, CI [9.0, 14.2]) and wall clock (4.9x, CI [3.7, 6.5]), because its final verification costs
  far more evaluations than the solve. On the continuous nonconvex group (Rosenbrock, Rastrigin, Ackley) it is
  superior on evaluations (0.80x, CI [0.74, 0.86], Holm p = 0.035 of 9).
- **Safety.** No faulty artifact passed the guard: 0 of 1,008 in memory and 0 of 1,008 as sealed runs (one-sided
  95 % upper bound 0.30 %). No scored run was verified while wrong. Replay was EQUAL for all 870 scored rule-arm
  cells. Budget enforcement is 290/290 for A and B but **286/290 for E-rule**: four cells used exactly their
  evaluation budget and reported `planner_stop` (section 6.2). Hard constraints were preserved in every E-rule
  formulation (2,400 of 2,400).

## 2. Setup

### Frozen digests

The freeze was written and committed before any scored shard ran (commit `8c06de7`, file
[`benchmarks/results/e14/freeze.json`](../../benchmarks/results/e14/freeze.json)). Every scored shard refuses to run
unless all three digests still match, and records them in its file. `analyze` refuses shards whose digests differ.

| digest | value |
|---|---|
| spec (`e14_spec.spec_digest()`, as quoted in the spec) | `sha256:e5104c340d419f388643bed2ad1de32662f8384f7b1c4adb7a05b90ad877db47` |
| protocol (the runner's own choices, `e14_common.PROTOCOL`) | `sha256:ecefd6a0a83ffb8b4abd637c6cb218f88ea92ff4c499974f8f35bdd058119e46` |
| code (every runner and `src/orbitopt` source file) | `sha256:d4b73f6935d150d60bf763f95d4bdaf125b5e3c4ea0cd32ac97e0a01dff483b3` |
| freeze file content | `sha256:7c7fde0ca571983eee48cd20467eecdeb59a1a0a93925b3bf676724b86cba47f` |

The laptop and the server computed the same three digests from the same commit.

### Arms

| arm | problem | solver | notes |
|---|---|---|---|
| A | reference callables | `scipy.optimize.differential_evolution`, cell seed, `polish=True`, popsize 15, generations fitted to 90 % of the budget, integrality from the variable type | multi-objective: Tchebycheff over 50 weights, each by the same DE with 1/50 of the budget |
| B | reference callables | the spec's rule table: HiGHS `linprog` / `milp` on coefficients probed from the callables, `trust-constr` from the box centre, SLSQP multistart until the budget, arm-A DE for `global`, enumeration in increasing stated cost for `black_box`, Tchebycheff + SLSQP for multi-objective | `global` is the same DE as A, so B = A there by construction |
| E-rule | reference IR | `orbitopt.agent.runtime.solve` with `RuleBasedPlanner(max_restarts=3)`, default policy and registry, the IR's default verification policy | every cell saved with `save_run` and replayed with `replay` |

A and B call scipy directly on the catalogue's Python callables through their own budgeted evaluator, so they share
no code with the runtime. The budget is the spec's: 2000 x max(d, 2) point evaluations and 600 s per cell.

### Seeds, tuning and hold-out

The spec fixes the seeds: scoring seeds 0-9 and tuning seeds 1000-1004. The runner uses those.

The tuning split was used once, before the freeze, as a dry run (120 cells). It changed one parameter: B's
`trust-constr` rule stopped 1.6e-4 short of the QP optimum on `cqp-worked-d2` with scipy defaults, and reaches
8.7e-13 with 3-point differences, BFGS Hessians, tolerances 1e-12 and an initial barrier parameter of 1e-6. Nothing
else changed. No scoring instance was run before the freeze. All three entries are in `PROTOCOL["tuning_adjustments"]`.

### Where it ran

| item | value |
|---|---|
| server | `j7zang` (`linux.student.cs.uwaterloo.ca`, a load balancer; every scored shard ran on host `ubuntu2404-002`) |
| environment | `~/e11env`: Python 3.12.13, numpy 2.5.5, scipy 1.18.1, pygmo 2.20; nothing installed |
| workspace | `~/e14-10-campaign/` (code in `code-8c06de7/`, results in `out/`, logs in `logs/`, launchers `e14run.sh`, `e14jobs.sh`), left in place |
| jobs | 4 processes at once, staggered by about 20 s, `OMP/OPENBLAS/MKL_NUM_THREADS=1`, `nice 10` |
| wall time | 9 min 10 s for all 105 shards (03:13:01 to 03:22:11 UTC). The slowest shard, `cells.scoring.E-rule.knapsack`, took 180 s including replays. No shard came near the 30-minute cap. |
| failures and reruns | none. One earlier tuning dry run ran under code `63c30a2`; its shards were re-run under the frozen code. |

One process ran every arm of one family in turn, so an instance's arms shared a host.

## 3. What was not run, and why

| item | why | how to run it |
|---|---|---|
| arm C (the model proposes points) | no provider | set the variables below and rerun the `cells.*.C.*` shards |
| arm D (compiler + fixed solver) | no provider | same, `cells.*.D.*` |
| arm E, recorded as `E-LLM` (compiler + LLM planner) | no provider | same, `cells.*.E-LLM.*` |
| `formulation_accuracy` (58 cases x 10 seeds) | needs a real model | `formulation.scoring` |
| `ir_schema_validity` of the compiler | needs a real model | arms D and E-LLM and the formulation shard |
| token cost | 0 tokens and $0 for A, B and E-rule by construction | arms C, D, E-LLM |
| primary comparisons C-A, D-A, E-D, E-B | an arm of each was not run | counted as p = 1 in the Holm family of 15 |
| external CUTEst and COCO instances | not vendored (spec section 3) | |

Each not-run cell is still in the results, as a row with status `not_run` and reason `no provider`: C, D and
E-LLM have 290 scored rows each. No mock or cassette answered for a model in any scored row.

A provider is configured only from the environment:

| variable | meaning |
|---|---|
| `E14_LLM_KIND` | `openai_compat` (default) or `local` |
| `E14_LLM_BASE_URL` | the endpoint, for example `https://api.openai.com/v1` or `http://127.0.0.1:11434/v1` |
| `E14_LLM_MODEL` | the model id, recorded in each shard's provenance |
| `E14_LLM_API_KEY` | the key for `openai_compat`; the provider reads it at call time and never stores it |

Historical names: these are the variables v1 read. From campaign v3 on, the campaign, the CLI and Mission
Studio read one set, `LLM_KIND`, `LLM_MODEL`, `LLM_BASE_URL` and `LLM_API_KEY`
([providers](../agent/providers.md#configuration-from-the-environment)); the `E14_LLM_*` names are gone.

The provider arms record every model call to a cassette, and their replay re-runs the cell from it. The pipeline is
tested in `tests/test_e14_10_campaign.py` with mock providers, labelled `pipeline-smoke`. Analysis refuses those rows
unless told it is a pipeline test, and then labels every number "not evidence".

## 4. Results per class

Success counts cells (instances x 10 seeds). The median gap is the normalised gap of section 4 of the spec. "Alarms"
are cells whose gap is below minus the tolerance (section 6.1); the spec says such a cell is not a win, so it is
counted as no success.

| class | arm | cells | success | feasible | alarms | median gap | budget exhausted | median evaluations | median wall s |
|---|---|---|---|---|---|---|---|---|---|
| lp | A | 20 | 0 | 20 | 0 | 5.18e-06 | 0 | 9.22e+03 | 0.489 |
| lp | B | 20 | 20 | 20 | 0 | 0 | 0 | 8 | 0.00146 |
| lp | E-rule | 20 | 20 | 20 | 0 | 0 | 0 | 3e+03 | 0.284 |
| convex_qp | A | 40 | 5 | 40 | 0 | 5.12e-08 | 5 | 4.78e+03 | 0.264 |
| convex_qp | B | 40 | 40 | 40 | 0 | 1.37e-11 | 0 | 160 | 0.0253 |
| convex_qp | E-rule | 40 | 21 | 40 | 19 | 1.05e-15 | 0 | 3.97e+03 | 0.281 |
| milp | A | 40 | 33 | 40 | 0 | 0 | 0 | 4.02e+03 | 0.81 |
| milp | B | 40 | 40 | 40 | 0 | 0 | 0 | 15.5 | 0.013 |
| milp | E-rule | 40 | 40 | 40 | 0 | 0 | 0 | 9.8e+03 | 0.919 |
| nlp | A | 40 | 36 | 40 | 0 | 1.04e-10 | 0 | 1.38e+04 | 0.562 |
| nlp | B | 40 | 40 | 40 | 0 | 6.43e-09 | 40 | 1.5e+04 | 0.472 |
| nlp | E-rule | 40 | 40 | 40 | 0 | 3.41e-11 | 3 | 1.25e+04 | 0.825 |
| global | A | 60 | 33 | 60 | 0 | 2.31e-08 | 0 | 1.82e+04 | 0.794 |
| global | B | 60 | 33 | 60 | 0 | 2.31e-08 | 0 | 1.82e+04 | 0.787 |
| global | E-rule | 60 | 50 | 60 | 0 | 6.53e-09 | 0 | 1.65e+04 | 1.05 |
| black_box | A | 60 | 53 | 60 | 0 | 0 | 0 | 319 | 0.0424 |
| black_box | B | 60 | 60 | 60 | 0 | 0 | 0 | 104 | 0.0197 |
| black_box | E-rule | 60 | 60 | 60 | 0 | 0 | 0 | 2.38e+03 | 0.109 |
| multi_objective | A | 30 | 29 | 30 | 0 | 0.0178 | 0 | 9.41e+03 | 0.599 |
| multi_objective | B | 30 | 30 | 30 | 0 | 0.0105 | 30 | 1e+04 | 0.762 |
| multi_objective | E-rule | 30 | 21 | 30 | 0 | 0.0151 | 0 | 3.41e+03 | 0.0944 |

The spec's target for A, B, D and E is success = 1.00 on lp, convex_qp and milp, "a miss is a defect". B meets it.
A misses it, as the spec expected of differential evolution (its QP gaps of about 5e-8 sit just above the 1e-8
tolerance, and its LP gaps near 5e-6). E-rule meets it on lp and milp and misses it on convex_qp: that is the alarm
of section 6.1, a defect in how the runtime picks its best candidate.

### Paired comparisons

Every test pairs arms by instance and seed, aggregates seeds within each instance first, and treats instances as the
units: the exact cluster sign-flip test on success, the exact Wilcoxon signed-rank test on per-instance log ERT
ratios, a cluster bootstrap CI (10,000 resamples, seed 1410, strata = family), Holm within each family, and the
spec's verdict rule. *Superior* needs at least 6 informative instances, Holm p < 0.05 and a CI that excludes 0 on
the favourable side. A ratio below 1 means the treatment is cheaper. The gap endpoint is secondary: the mean
log10 ratio of median gaps (gaps floored at 1e-12), negative means the treatment gets closer.

**Overall (29 instances).** Primary family: Holm over the spec's 15 hypotheses, the 12 not run counted as p = 1.
Secondary family (E-rule): Holm over 6.

| comparison | endpoint | effect | 95 % CI | informative | p | p Holm (m) | verdict |
|---|---|---|---|---|---|---|---|
| B vs A | success-rate difference | +0.255 | [0.19, 0.328] | 16 | 3.05e-05 | 0.000397 (15) | superior |
| B vs A | ERT ratio | 0.225 | [0.168, 0.3] | 23 | 2.62e-05 | 0.000387 (15) | superior |
| B vs A | ERT-time ratio | 0.345 | [0.292, 0.401] | 27 | 2.58e-05 | 0.000387 (15) | superior |
| B vs A | log10 gap ratio | -1.21 | [-1.88, -0.629] | 14 | 0.0245 | 0.0245 (3) | superior |
| E-rule vs A | success-rate difference | +0.217 | [0.1, 0.338] | 20 | 0.0108 | 0.0431 (6) | superior |
| E-rule vs A | ERT ratio | 1.59 | [1.32, 1.91] | 27 | 0.786 | 1 (6) | not shown different |
| E-rule vs A | ERT-time ratio | 1.07 | [0.899, 1.29] | 27 | 0.53 | 1 (6) | not shown different |
| E-rule vs A | log10 gap ratio | -2.2 | [-3.24, -1.3] | 18 | 0.000145 | 0.000435 (3) | superior |
| E-rule vs B | success-rate difference | -0.0379 | [-0.128, 0.0517] | 8 | 0.617 | 1 (6) | not shown different |
| E-rule vs B | ERT ratio | 11.3 | [9, 14.2] | 28 | 0.00611 | 0.0305 (6) | inferior |
| E-rule vs B | ERT-time ratio | 4.92 | [3.68, 6.47] | 28 | 0.00225 | 0.0135 (6) | inferior |
| E-rule vs B | log10 gap ratio | -0.642 | [-1.52, 0.127] | 16 | 0.0739 | 0.0739 (3) | not shown different |

The success rates behind the first row are 0.907 (B) and 0.652 (A). As checks, an exact McNemar test on seed 0 only
(one pair per instance, where the spec's cluster test reduces to McNemar) gives 8 instances where only B succeeded
and 0 the other way, p = 0.0078. A McNemar test pooled over seeds gives 74 to 0 and p = 1e-22. That pooled p
ignores clustering and is not evidence: it treats 10 seeds of one instance as 10 independent problems.

The ERT CIs cover only instances that both arms solved at least once (24 for B vs A); instances that only one arm
solved enter the Wilcoxon test as an infinite ratio (5 for B vs A, all in B's favour) and are counted by the
success endpoint.

**Per analysis group.** Holm runs over the 9 tests in each group.

| group (instances) | comparison | success diff [CI], verdict | ERT ratio [CI], verdict | ERT-time ratio [CI], verdict |
|---|---|---|---|---|
| algebraic (10) | B vs A | +0.62 [0.44, 0.81], superior | 0.005 [0.0019, 0.013], superior | 0.019 [0.0087, 0.037], superior |
| algebraic (10) | E-rule vs A | +0.43 [0.14, 0.72], not shown different | 1.85 [1.05, 3.14], not shown different | 1.03 [0.61, 1.73], not shown different |
| algebraic (10) | E-rule vs B | -0.19 [-0.38, 0], not testable (2 informative) | 225 [148, 328], inferior | 53 [23.5, 109], inferior |
| continuous_nonconvex (10) | B vs A | +0.04 [0, 0.09], not testable | 0.99 [0.96, 1.03], not testable | 0.92 [0.84, 1.04], not shown different |
| continuous_nonconvex (10) | E-rule vs A | +0.21 [0.07, 0.36], not shown different (Holm p 0.22) | 0.79 [0.72, 0.88], superior | 1.13 [1.05, 1.21], not shown different |
| continuous_nonconvex (10) | E-rule vs B | +0.17 [0.04, 0.33], not testable (4 informative) | 0.80 [0.74, 0.86], superior | 1.24 [1.09, 1.40], not shown different |
| black_box (6) | B vs A | +0.12 [0.03, 0.23], not testable | 0.22 [0.09, 0.45], not shown different | 0.40 [0.31, 0.49], not shown different |
| black_box (6) | E-rule vs B | 0 [0, 0], not testable | 30 [15.8, 58.7], not shown different | 5.6 [5.1, 6.2], not shown different |
| multi_objective (3) | all | descriptive only (spec: 3 instances cannot reach p < 0.05) | | |

The black-box group cannot reach p < 0.05 unless all 6 instances are discordant in one direction. Its cost ratios are
6 of 6 in one direction with exact p = 0.031, but Holm over 9 tests lifts that to 0.28, so the verdict stays "not
shown different".

**Sensitivity.** The frozen scorer counts the 19 alarm cells as successes. With that reading, E-rule vs B is 0.0 on
algebraic success (no informative instance) and +0.028 [-0.035, 0.09] overall (not shown different), and E-rule vs A
is +0.283 [0.19, 0.38] overall (superior, Holm p 0.008). No verdict on B vs A changes. The full table is in
[`tables.md`](../../benchmarks/results/e14/tables.md).

### What limits these numbers

- **E-rule's evaluation cost is its verification.** From the saved run directories
  ([`erule_evaluation_split.jsonl`](../../benchmarks/results/e14/erule_evaluation_split.jsonl), descriptive, added
  after the run): on the LPs and knapsacks the planner loop spends 2 evaluations (one HiGHS solve), and the final
  verification spends 418 to 24,526 (alternate solvers and enumeration). On the integration instances the loop
  spends about 300 and verification 2,051, which is the enumeration of the 2,048-point domain. A and B verify
  nothing, so the ERT comparison charges E-rule for work the baselines do not do. On the nonconvex group the loop is
  59 to 77 % of the total. ERT divides by the successful seeds, so part of E-rule's lower ERT there comes from its
  higher success rate (inferred from the definition, not measured separately).
- **B's `nlp` and multi-objective rules spend the whole budget by design** ("SLSQP multistart until the budget"), so
  their 70 `budget_exhausted` cells are the rule, not a failure, and their ERT is the budget divided by the success
  rate.
- **B = A on `global`.** The rule table names the same DE, so 6 of the 10 continuous-nonconvex instances cannot be
  discordant for B vs A.
- **Wall clock is short and shared.** Every cell finished in under 3.5 s against the 600 s cap. All shards ran on one
  shared host at a load of about 2 to 7, so the wall-clock ratios carry that host's noise. Wall clock covers the
  solve only, not saving or replaying a run. The Gauss-Legendre rule cache is built before the first cell of each
  shard, so no arm pays for it.
- **Most group-level tests are not testable.** Ten instances per group is the minimum the spec's exact test can use,
  and several comparisons have fewer than 6 discordant instances.

## 5. Verification status (E-rule)

| run status | cells | where |
|---|---|---|
| verified | 214 | |
| unverified_best | 76 | `rastrigin-d10-shifted` (10, all wrong), the three scoring MO instances (30: NSGA-II has no alternate), `cqp-projection-d2` and `cqp-equality-d3` (20, the alarm cells), `ackley-d10-shifted` (10, all right: the alternates did not reach its basin), `integ-nearsing-1e-6`, `knapsack-p07-n15` and `knapsack-correlated-n20` (2 each) |

`optimality_certified` passed in 20 cells, all on `knapsack-p01-n10` and `knapsack-2d-n12`, whose domains (1,024 and
4,096 points) fit the evaluations left for the guard's own enumeration. It was not assessed in the other 270:
continuous nonconvex problems are never certified, and the larger domains did not fit. The guard's verdict failed in
30 cells (the MO instances: no converged claim and no evidence). Every verified cell succeeded: no cell was verified
while infeasible or outside tolerance.

A and B carry only the guard's screen (no independent evidence), so their verdicts are `not_assessed` or `failed`
(`failed` when the solver stopped on the budget and so did not converge). Their verdicts cannot pass by design.

## 6. Safety metrics

| metric | A | B | E-rule | target |
|---|---|---|---|---|
| budget enforcement | 290/290 | 290/290 | **286/290** | 1.00 |
| independent counter = reported evaluations | 290/290 | 290/290 | 290/290 | |
| most evaluations used / budget | 1.0 | 1.0 | 1.0 | <= 1 |
| experiment replay EQUAL | 290/290 | 290/290 | 290/290 | 1.00 |
| hard constraints preserved | n/a (no formulation) | n/a | 2,400/2,400 | 1.00 |
| scored runs verified while wrong | 0 | 0 | 0 of 214 verified | 0 |
| tokens, USD | 0, 0 | 0, 0 | 0, 0 | report only |

### 6.1 The alarm cells

19 E-rule cells have a gap below minus the tolerance: all 10 seeds of `cqp-projection-d2` (gap -2.98e-8, violation
1.49e-8) and 9 seeds of `cqp-equality-d3` (gaps -1.04e-6 to -1.09e-6, violations 9.5e-7 to 9.99e-7). Each point is
feasible within the 1e-6 constraint tolerance and "beats" the optimum by buying violation. The arithmetic confirms
it: the constraint's Lagrange multiplier times the violation equals the gap (2 x 1.49e-8 = 2.98e-8 at (1, 0);
12/11 x 9.9e-7 = 1.08e-6 at (6, 3, 2)/11). On `cqp-projection-d2` the SLSQP solve itself stops 1.5e-8 outside the
half-plane. On `cqp-equality-d3` the SLSQP point is exact (6/11), but the differential-evolution alternate in the
final verification finds a point at the tolerance edge with a lower objective, refutes the exact point, and becomes
the run's best. The guard verified none of the 19 (`unverified_best`). The references are right; the runtime's
best-candidate rule ranks points that are feasible within tolerance by raw objective. Per the spec these cells count
as no success. The spec's own text records the same case for SLSQP on the LP relaxation.

### 6.2 E-rule budget enforcement: 286/290

Four cells of `rosenbrock-disk-d2` (seeds 2, 3, 7, 8) used exactly 4,000 of 4,000 evaluations and ended with
`planner_stop` and status `verified`. None overran: the meter and the independent counter agree in all 290 cells.
The last evaluations went to the final verification, whose differential-evolution alternate stopped on its
allocation (termination `max_evaluations`) and agreed with the candidate. The exhaustion is typed on the alternate's
result in the trace, but not on the run, so under the spec's rule ("stopped with a typed budget_exhausted status
instead of a silent truncation") these cells fail. The fix belongs in the runtime: a run whose verification ran out
of budget should say so in its status.

### 6.3 False verification pass: fault injection

Every scoring instance got every fault on up to four hosts: the seed-0 point of A, of B and of E-rule, and the stored
optimum where the catalogue has one (25 instances). That is 112 hosts x 9 faults = 1,008 faulty artifacts, each
graded in memory (`verify_result`) and as a sealed run (`verify_run`).

| path | graded | false pass | 95 % upper bound | reason named | honest host passed | false pass on those | reason named on those |
|---|---|---|---|---|---|---|---|
| verify_result | 1,008 | 0 | 0.30 % | 993 | 792 | 0 | 792 |
| verify_run | 1,008 | 0 | 0.30 % | 993 | 792 | 0 | 792 |

A fault on a host whose honest artifact already fails proves little, so the informative count is the 792 artifacts
whose honest host passed (88 hosts). On those, every fault was rejected for its own reason (one-sided 95 % upper
bound on the false-pass rate 0.38 %). The honest host fails for 24 hosts: 15 whose point is not optimal (A on the QPs,
the LPs and Rastrigin, B and E-rule on `rastrigin-d10-shifted`, E-rule on the two alarm QPs, and so on) and the 9 MO
hosts, which have no reference objective to give as independent evidence. The 15 cells per path where the fault's own reason was not
named are all MO hosts: F09 (a forged verifier id) cannot be named when there is no evidence item, and F07 replaces
the first objective with `x[0]`, which on ZDT1 and ZDT2 is the objective already, so the injection changes nothing.
They were all rejected.

With 870 scored rule-arm runs added, the metric's denominator is 1,878 per path; the false-pass count is 0.

## 7. Failures

Every instance that an arm did not solve on all 10 seeds, from the generated table:

| arm | instance | success | median gap | worst gap |
|---|---|---|---|---|
| A | rastrigin-d20 | 0/10 | 36.8 | 72.6 |
| A, B | rastrigin-d10-shifted | 0/10 | 10.9 | 15.9 |
| E-rule | rastrigin-d10-shifted | 0/10 | 6.96 | 10.9 |
| A, B | rastrigin-d5 | 6/10 | 3.9e-14 | 0.995 |
| A | cqp-projection-d2, cqp-simplex-d5, cqp-simplex-d10 | 0/10 each | 5e-8 to 6e-6 | |
| A | cqp-equality-d3 | 5/10 (5 cells on the budget) | 4e-7 | 0.396 |
| A | lp-wyndor-d2, knapsack-p01-lp-relaxation | 0/10 each | 9e-6, 7e-7 | |
| A | knapsack-correlated-n20 | 5/10 | 0.0018 | 0.0036 |
| A | integration (4 of 6 instances) | 6/10 to 9/10 | 0 | up to 1.62 |
| A | rosenbrock-d5, rosenbrock-d20, ackley-d20, ackley-d10-shifted, knapsack-2d-n12, knapsack-p07-n15, mo-zdt2-d5 | 7/10 to 9/10 | | |
| B | ackley-d20 (9/10), ackley-d10-shifted (8/10) | | | |
| E-rule | cqp-projection-d2 (0/10), cqp-equality-d3 (1/10) | alarm cells, section 6.1 | | |
| E-rule | mo-zdt2-d5 | 2/10 | 0.020 | 0.333 |
| E-rule | mo-zdt1-d5 | 9/10 | 0.017 | 0.043 |

`rastrigin-d10-shifted` and `rastrigin-d20` defeat every arm here; the shifted variant is the one the spec added
because no textbook optimum points at it. E-rule's NSGA-II front on ZDT2 (non-convex) misses the 5 % hypervolume
tolerance in 8 of 10 seeds, where B's 50-weight Tchebycheff passes all 10.

## 8. Threats to validity

- **No LLM.** Nothing here measures what a model adds. The comparisons that answer the spec's question (C-A, D-A,
  E-D, E-B) have no data.
- **E-rule is not arm E.** It runs on the hand-written problem, so it cannot fail at formulation, and its planner is
  fixed code. Its results bound what the runtime and guard do with a perfect formulation and a cautious planner.
- **Verification is charged to E-rule only.** A and B return unverified points; E-rule pays thousands of evaluations
  to verify. The cost endpoints compare an answer with a verified answer.
- **Runner choices the spec left open** (A's generation count, A on multi-objective, the ideal point of the
  Tchebycheff weights, B's ordering of the black-box grid by the stated cost, the integrality term in the scorer) are
  fixed in the protocol digest before the run. Other reasonable choices could move A and B.
- **One tuning change** (B's `trust-constr` options) was set on the tuning split. Without it B's convex-QP rule
  would miss on the tuning instance, and likely on the scoring QPs.
- **Tolerance edges.** Several A failures are gaps a few times the tolerance (QPs at about 5e-8 against 1e-8), and
  the E-rule alarm cells sit at the constraint tolerance. Success near a tolerance is sensitive to solver settings.
- **Small groups.** Ten instances per group is the floor of the exact test; black box and multi-objective cannot
  support a claim; seeds are not independent units, so they add little power.
- **Memorisation does not apply** without an LLM, so the `well_known` split was not analysed.
- **One shared host** for all wall-clock numbers, under other users' load.
- **Scorer and analysis.** The frozen scorer did not apply the spec's alarm rule; the analysis applies it, and the
  sensitivity table shows the scorer's reading. The analysis module is outside the code digest, and its output is
  reproducible: the laptop recomputed the server's `analysis.json` to the same content digest before this change.

## 9. Data and integrity

| file | where | digest |
|---|---|---|
| 105 shard files (4.5 MB) | server, `~/e14-10-campaign/out/shards/` | each in [`manifest.json`](../../benchmarks/results/e14/manifest.json) (sha256 and content digest) |
| E-rule run directories (7.9 MB) | server, `~/e14-10-campaign/out/runs/`, and `~/e14-10-campaign/runs-8c06de7.tar` | tar `sha256:187fe9d02033cb53e5ac6ec0b44f7fa1bd947e4e216541457919fc6ca1a86b09`; each run's manifest digest is the row's `artifact_digest` |
| per-cell CSV | [`benchmarks/results/e14/cells.csv`](../../benchmarks/results/e14/cells.csv) | in the manifest |
| analysis | [`benchmarks/results/e14/analysis.json`](../../benchmarks/results/e14/analysis.json), [`tables.md`](../../benchmarks/results/e14/tables.md) | content digest `sha256:26c076f83d7ad9c90d335fd61020139fadf1b05062cbaee3ef0fbd1b4cd52834` |
| freeze | [`benchmarks/results/e14/freeze.json`](../../benchmarks/results/e14/freeze.json) | section 2 |

Every shard file seals its own content digest. The copy-back was `rsync` of `out/` without `runs/` to the laptop,
then `verify`, which re-hashed every file the manifest names: all matched. Nothing on the server was deleted.

## 10. Reproduction

On the server (or any machine with numpy, scipy, jsonschema and pygmo), from a checkout of the frozen commit:

```bash
git archive 8c06de7 | tar -x -C code-8c06de7
cd code-8c06de7
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python benchmarks/bench_e14_end_to_end.py plan                    # digests and the 105 shards
for f in rosenbrock rastrigin ackley constrained_quadratic linear_program knapsack integration bi_objective; do
  python benchmarks/bench_e14_end_to_end.py run --all --family $f --out ../out   # tuning, scoring, faults
done
python benchmarks/bench_e14_end_to_end.py run --shard formulation.scoring --out ../out
python benchmarks/bench_e14_end_to_end.py collect --out ../out
python benchmarks/bench_e14_end_to_end.py verify --out ../out
```

`run` skips every shard already present under the same digests, so an interrupted job resumes by running the same
command. The analysis and tables come from the current branch, on the copied shard files:

```bash
python benchmarks/bench_e14_end_to_end.py analyze --out <copy>
python benchmarks/bench_e14_end_to_end.py report --out <copy>
python benchmarks/e14_analysis.py <server out dir>                # the post-hoc E-rule evaluation split
```

The pipeline test, with no server and no model:

```bash
PYTHONPATH=src python -m pytest -q tests/test_e14_10_campaign.py
```
