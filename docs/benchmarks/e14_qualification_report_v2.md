# E14-10 qualification report: campaign v2

This is the second scored run of the [E14-10 benchmark specification](e14_benchmark_spec.md). It reran the three
arms that need no language model, on the same 29 scoring instances, seeds, runner, shards and server as
[campaign v1](e14_qualification_report.md), after two runtime fixes that v1 exposed. v1 and its results stay as
recorded; v2 is a separate freeze with its own results in
[`benchmarks/results/e14-v2/`](../../benchmarks/results/e14-v2/).

**Read this first.** No LLM was called. Arm `E-rule` is the E14 runtime with its deterministic rule-based planner on
the hand-written reference problem. It is not the pre-registered arm E. `E-rule` against A or B measures the
runtime's solver selection, restarts and verification. **It says nothing about whether an LLM helps.**

## 1. The short answer

- **The two v1 defects are gone.** No cell of any arm has a gap below minus the tolerance (v1: 19 E-rule cells).
  E-rule now succeeds on all 40 convex-QP cells (v1: 21 under the spec's rule) and so meets the spec's
  `success = 1.00` target on lp, convex_qp and milp. Every E-rule cell whose run was cut by a limit says so in a
  typed field (v1: 4 cells used their whole budget and said nothing).
- **Safety holds.** 0 false passes in 1,008 injected faults per path, 0 scored runs verified while wrong (223
  verified), replay EQUAL 870/870, budget enforcement 290/290 for every rule arm, hard constraints 2,400/2,400.
- **A and B did not move.** Every one of their 580 scored rows has the same row digest as in v1 (timing fields
  excluded). The scorer change touched no A or B cell, and the two arms reproduced exactly on another host.
- **E-rule against B** is still not shown different on success (+0.028, CI [-0.035, 0.090]) and still inferior on
  evaluations (10.4x) and wall clock (4.5x), because its final verification costs far more than B's solve. Its gap
  endpoint is now superior (-0.99 in log10, Holm p 0.005 of 3); in v1 the alarm cells hid that.
- **LLM value: not answered.** Arms C, D and E-LLM were not run (no provider).

## 2. What changed from v1

| change | where | effect on the run |
|---|---|---|
| Every budget cut is typed: `outcome.budget` in the trace (agent-trace-1.2) names the stage (loop, solve, alternate, certificate, verification, run), the resource and the allowance | runtime ([runtime docs](../agent/runtime.md#where-a-limit-cut-the-run)) | A run that is not verified after a cut of the loop, the final verification or the run is `budget_exhausted`, not `unverified_best`. `verified` still needs the guard's pass on the evidence the cut stage produced. |
| Clean points first, and restoration onto the constraints | solver adapters and the runtime's incumbent ([solvers docs](../optimization/solvers.md#the-tolerance-band-is-not-room-to-optimise)) | A point that uses more than 1e-6 of a constraint's tolerance band never outranks a clean point on objective; the adapter moves its best band point onto the constraints before it reports it. The guard and every tolerance are unchanged. |
| The scorer applies spec section 2: a gap below minus the tolerance is an alarm and no success | `e14_common.score` | No analysis-side correction and no sensitivity table is needed: v1's analysis had to apply the rule after the fact. |
| E-rule rows type a cut in `stop` (`budget_exhausted:<resource>`), and `budget_typed` reads it as for A and B | `bench_e14_agent_iteration.py` | The per-class "budget exhausted" column now counts E-rule cells whose verification alternate stopped at its allocation. |

The spec is unchanged (its digest is v1's). The scorer now does what section 2 already says, so no addendum was
needed.

### Root causes, in short

**Budget typing.** The runtime typed a limit only when it ended the planner loop (`stop_reason`). The final
verification's alternates get what the loop leaves, and on `rosenbrock-disk-d2` the differential-evolution
alternate spent the last evaluations and stopped on its allocation. Its own result said `max_evaluations`; the run
said `planner_stop` and `verified`.

**Negative gaps.** Two places ranked tolerance-feasible points by raw objective: the adapters' best evaluated point
and the runtime's incumbent. A point inside the band beats the exact optimum by its multiplier times its violation:

| instance | point | violation | multiplier | multiplier x violation | v1 gap |
|---|---|---|---|---|---|
| `cqp-projection-d2` | SLSQP, (1 + 2^-26, 0) | 1.49e-8 | 2 | 2.98e-8 | -2.98e-8 |
| `cqp-equality-d3` | differential evolution | 9.54e-7 | 12/11 | 1.04e-6 | -1.04e-6 |

Differential evolution sees the band as its constraint, so it presses against its edge. On `cqp-equality-d3` that
gain (1.04e-6) exceeds the guard's agreement tolerance (1e-6 x max(1, 6/11)), so the alternate refuted SLSQP's exact
6/11 and its band point became the run's best. The guard's refutation rule was left alone: refutation can only fail
a correct result, never pass a wrong one, and telling a bought gain from a real one needs the multiplier. With a
clean threshold of 1e-6 of the band, a clean point can buy at most 2.98e-14 and 1.04e-12 on these two instances,
below the spec's 1e-8 QP tolerance. Restoring the band point (least-norm Gauss-Newton on the active constraints,
bounds held) makes it compete on what feasibility costs: the restored alternate answers 0.54545454545465, within
1e-13 of 6/11, and agrees.

## 3. Setup

| item | value |
|---|---|
| spec digest (unchanged) | `sha256:e5104c340d419f388643bed2ad1de32662f8384f7b1c4adb7a05b90ad877db47` |
| protocol (`e14-10-campaign/2`) | `sha256:9cd233d3dc127da9765bc2b322d2480e5a49f2c5b8b61ff3652826ce4464ed8f` |
| code | `sha256:d7a8ed51c21c5ba57e1ba08eb53b3ba2121850e706acbd90f21cd0354e6af5bb` |
| freeze file content | `sha256:4c4560dc73a5b72f8252aeaca0227c39c777e3bd82098d6a5239a73966c44841` |
| freeze commit | `3311d09`, committed before any scored shard ran; [`freeze.json`](../../benchmarks/results/e14-v2/freeze.json) |
| tuning dry run | before the freeze, on the v2 code with the protocol not yet final: 120 rule-arm tuning cells, all within budget, typed and replayed EQUAL, no alarm; no parameter changed (`PROTOCOL["tuning_adjustments"]`) |
| server | `j7zang`; every scored shard ran on host `ubuntu2404-012` (v1: `ubuntu2404-002`) |
| environment | `~/e11env` (Python 3.12, numpy, scipy, pygmo as in v1); nothing installed |
| workspace | `~/e14-10-campaign-v2/` (code in `code-3311d09/`, results in `out/`, logs in `logs/`, launchers `e14run.sh`, `e14jobs.sh`); v1's `~/e14-10-campaign/` untouched |
| jobs | 4 family processes plus the formulation shard, staggered by 20 s, threads pinned to 1, `nice 10`, as in v1 |
| wall time | 7 min 42 s for 105 shards (05:13:46 to 05:21:28 UTC; v1: 9 min 10 s). The slowest shard, `cells.scoring.E-rule.knapsack`, took 155 s (v1: 180 s). The host's load average was 3 to 18, from other users. |
| failures and reruns | none |

The laptop and the server computed the same three digests from the same commit, and the same `analysis.json`
content digest (`sha256:2fe70d6e85bb46b8c03380e77584a3b4407ad9a49156d3e0900df48ef86326f1`).

## 4. v1 against v2, the cells that moved

E-rule rows differ from v1 in every cell, because each now carries its budget events. 9 of the 29 instances moved in
success, verification, typing or evaluations; the other 20 are equal on all five. A and B: 0 of 580 rows moved.

| instance (E-rule, 10 seeds) | success (spec rule) v1 to v2 | alarms | verified | budget typed | median evaluations | lowest gap |
|---|---|---|---|---|---|---|
| `cqp-projection-d2` | 0 to 10 | 10 to 0 | 0 to 0 | 10 to 10 | 725 to 735 | -2.98e-8 to 0 |
| `cqp-equality-d3` | 1 to 10 | 9 to 0 | 0 to 9 | 10 to 10 | 4816 to 4780 | -1.09e-6 to 2.2e-16 |
| `rosenbrock-disk-d2` | 10 to 10 | 0 to 0 | 10 to 10 | 6 to 10 | 3993 to 3842 | 0 to 0 |
| `ackley-d10-shifted` | 10 to 10 | 0 | 0 to 0 (now `budget_exhausted`) | 10 to 10 | 17048 to 17048 | 1.2e-8 |
| `rastrigin-d10-shifted` | 0 to 0 | 0 | 0 to 0 (now `budget_exhausted`) | 10 to 10 | 15946 to 15946 | 3.98 |
| `cqp-simplex-d5`, `cqp-simplex-d10`, `lp-wyndor-d2`, `knapsack-p01-lp-relaxation` | 10 each, unchanged | 0 | 10 each, unchanged | 10 | +1, +1, +3, +18 | unchanged |

What limits these numbers:

- `cqp-projection-d2` succeeds but stays unverified. Its only alternate, differential evolution, converges at
  2.00001 (`tol = 0.01`), a feasible point 1e-5 worse, which the guard rightly calls inconclusive.
- The tenth `cqp-equality-d3` seed is `budget_exhausted`: its alternate stopped at its allocation at 0.54552, worse
  than 6/11, and the run says so.
- `rosenbrock-disk-d2` no longer uses its whole budget (at most 3992 of 4000): the adapters keep 3 (n + 1)
  evaluations for restoration and leave them unspent when the point is clean. Its verification alternate still stops
  at its allocation on 3 seeds, and those carry an `alternate` event with status `verified`, since the guard
  accepted the evidence the cut alternate produced.
- The shifted Ackley and Rastrigin instances moved to `budget_exhausted` because both their alternates stopped at
  their allocations. That is the new rule, not a change in what was found.
- The extra evaluations (+1 to +18) are restoration probes: difference points and steps on solver points that used
  more than 1e-6 of a band.

## 5. Results per class

Success counts cells (instances x 10 seeds). The scorer now counts an alarm as no success, so this table needs no
correction. "Budget exhausted" counts cells whose `stop` is `budget_exhausted:*`; for E-rule that includes cells
whose verification alternate stopped at its allocation (section 6.2).

| class | arm | cells | success | feasible | alarms | median gap | budget exhausted | median evaluations | median wall s |
|---|---|---|---|---|---|---|---|---|---|
| lp | A | 20 | 0 | 20 | 0 | 5.18e-06 | 0 | 9.22e+03 | 0.409 |
| lp | B | 20 | 20 | 20 | 0 | 0 | 0 | 8 | 0.00127 |
| lp | E-rule | 20 | 20 | 20 | 0 | 0 | 0 | 3.01e+03 | 0.246 |
| convex_qp | A | 40 | 5 | 40 | 0 | 5.12e-08 | 5 | 4.78e+03 | 0.225 |
| convex_qp | B | 40 | 40 | 40 | 0 | 1.37e-11 | 0 | 160 | 0.0207 |
| convex_qp | E-rule | 40 | **40** (v1: 21) | 40 | **0** (v1: 19) | 2.78e-16 | 1 | 3.95e+03 | 0.245 |
| milp | A | 40 | 33 | 40 | 0 | 0 | 0 | 4.02e+03 | 0.681 |
| milp | B | 40 | 40 | 40 | 0 | 0 | 0 | 15.5 | 0.011 |
| milp | E-rule | 40 | 40 | 40 | 0 | 0 | 0 | 9.8e+03 | 0.782 |
| nlp | A | 40 | 36 | 40 | 0 | 1.04e-10 | 0 | 1.38e+04 | 0.451 |
| nlp | B | 40 | 40 | 40 | 0 | 6.43e-09 | 40 | 1.5e+04 | 0.386 |
| nlp | E-rule | 40 | 40 | 40 | 0 | 3.41e-11 | 32 (v1: 3) | 1.25e+04 | 0.627 |
| global | A | 60 | 33 | 60 | 0 | 2.31e-08 | 0 | 1.82e+04 | 0.644 |
| global | B | 60 | 33 | 60 | 0 | 2.31e-08 | 0 | 1.82e+04 | 0.687 |
| global | E-rule | 60 | 50 | 60 | 0 | 6.53e-09 | 56 (v1: 0) | 1.65e+04 | 0.900 |
| black_box | A | 60 | 53 | 60 | 0 | 0 | 0 | 319 | 0.0342 |
| black_box | B | 60 | 60 | 60 | 0 | 0 | 0 | 104 | 0.0164 |
| black_box | E-rule | 60 | 60 | 60 | 0 | 0 | 0 | 2.38e+03 | 0.092 |
| multi_objective | A | 30 | 29 | 30 | 0 | 0.0178 | 0 | 9.41e+03 | 0.534 |
| multi_objective | B | 30 | 30 | 30 | 0 | 0.0105 | 30 | 1e+04 | 0.668 |
| multi_objective | E-rule | 30 | 21 | 30 | 0 | 0.0151 | 0 | 3.41e+03 | 0.081 |

The spec's target for A, B, D and E is success = 1.00 on lp, convex_qp and milp. B and E-rule meet it; A misses it as
in v1 (differential evolution stops just above the QP and LP tolerances).

Wall clock is 10 to 20 % lower than v1 for every arm, A and B included, whose work is bit for bit the same. That is
the host, not the code: v2 ran on another machine of the same pool.

### Paired comparisons

The statistics are v1's (cluster sign-flip on success, exact Wilcoxon on log ERT ratios, cluster bootstrap CI, Holm
per family, the spec's verdict rule). The full table is in [`tables.md`](../../benchmarks/results/e14-v2/tables.md).

**Overall (29 instances).**

| comparison | endpoint | v1 | v2 effect | 95 % CI | informative | p Holm (m) | v2 verdict |
|---|---|---|---|---|---|---|---|
| B vs A | success-rate difference | +0.255, superior | +0.255 | [0.19, 0.328] | 16 | 0.00043 (15) | superior |
| B vs A | ERT ratio | 0.225, superior | 0.225 | [0.168, 0.30] | 23 | 0.00039 (15) | superior |
| B vs A | ERT-time ratio | 0.345, superior | 0.346 | [0.294, 0.402] | 27 | 0.00053 (15) | superior |
| E-rule vs A | success-rate difference | +0.217, superior | +0.283 | [0.193, 0.376] | 21 | 0.0083 (6) | superior |
| E-rule vs A | ERT ratio | 1.59, not shown different | 1.46 | [1.23, 1.75] | 28 | 0.74 (6) | not shown different |
| E-rule vs A | ERT-time ratio | 1.07, not shown different | 0.993 | [0.838, 1.19] | 28 | 0.62 (6) | not shown different |
| E-rule vs B | success-rate difference | -0.038, not shown different | +0.028 | [-0.035, 0.090] | 6 | 0.74 (6) | not shown different |
| E-rule vs B | ERT ratio | 11.3, inferior | 10.4 | [8.28, 13.1] | 28 | 0.026 (6) | inferior |
| E-rule vs B | ERT-time ratio | 4.92, inferior | 4.54 | [3.38, 6.02] | 28 | 0.017 (6) | inferior |
| E-rule vs B | log10 gap ratio | -0.64, not shown different | -0.994 | [-1.89, -0.385] | 15 | 0.0053 (3) | superior |

The primary Holm family still holds the spec's 15 hypotheses, 12 of them not run (p = 1). E-rule comparisons are a
secondary family of 6, the gap endpoint a family of 3, as in v1. v2's E-rule vs B success effect equals v1's
sensitivity reading (+0.028), which is what the scorer counted then; the difference is that the alarm cells now
really are successes.

**Per analysis group** (Holm over the 9 tests of each group):

| group | comparison | success diff, verdict | ERT ratio, verdict | ERT-time ratio, verdict |
|---|---|---|---|---|
| algebraic (10) | B vs A | +0.62, superior | 0.005, superior | 0.019, superior |
| algebraic (10) | E-rule vs A | +0.62 [0.44, 0.81], superior (v1: +0.43, not shown different) | 1.32, not shown different | 0.655, not shown different |
| algebraic (10) | E-rule vs B | 0, not testable (0 informative) | 132 [90, 191], inferior (v1: 225) | 32 [15, 67], inferior |
| continuous_nonconvex (10) | E-rule vs A | +0.21, not shown different | 0.792 [0.716, 0.875], superior | 1.15, not shown different |
| continuous_nonconvex (10) | E-rule vs B | +0.17, not testable | 0.798 [0.739, 0.862], superior | 1.27, not shown different |
| black_box (6), multi_objective (3) | all | as in v1: not testable or not shown different | | |

The algebraic E-rule vs B ERT ratio fell from 225 to 132 because ERT divides evaluations by the success rate, and
two QPs went from 0 and 0.1 to 1. E-rule still spends 132 times B's evaluations there: B solves an LP or QP in 8 to
160 evaluations and verifies nothing; E-rule's final verification runs alternates (section 7).

## 6. Safety metrics

| metric | A | B | E-rule (v1) | target |
|---|---|---|---|---|
| budget enforcement | 290/290 | 290/290 | **290/290** (286/290) | 1.00 |
| independent counter = reported evaluations | 290/290 | 290/290 | 290/290 | |
| most evaluations used / budget | 1.0 | 1.0 | 0.998 (1.0) | <= 1 |
| experiment replay EQUAL | 290/290 | 290/290 | 290/290 | 1.00 |
| hard constraints preserved | n/a | n/a | 2,400/2,400 | 1.00 |
| scored runs verified while wrong | 0 | 0 | 0 of 223 verified (0 of 214) | 0 |
| cells with a gap below minus the tolerance | 0 | 0 | **0** (19) | 0 |
| tokens, USD | 0, 0 | 0, 0 | 0, 0 | report only |

### 6.1 Budget typing, and what the 290/290 rests on

The spec's rule applies to a cell that used its budget. No E-rule cell did in v2 (the most was 0.998 of it), so for
E-rule the 290/290 is met without the rule's condition ever firing: that number alone is weak evidence. The stronger
fact is the typed field. 89 E-rule cells had a final-verification alternate stop at its allocation, and all 89 carry
an `alternate` event and the stop `budget_exhausted:evaluations`. 68 of them are still `verified` (the guard accepted
the evidence the cut alternate produced), 21 are `budget_exhausted`. 127 cells also have a planner solve that stopped
at its allowance, which is typed but leaves the status to the final verification. In v1 the same cuts happened and
only the alternate's own result recorded them. The unit tests pin the exhausted case directly
([`test_e14_05_budget_typing.py`](../../tests/test_e14_05_budget_typing.py)), including an alternate that spends its
whole allocation, stages skipped for lack of evaluations, and a run that uses its whole iteration limit.

### 6.2 False verification pass: fault injection

| path | graded | false pass | 95 % upper bound | reason named | honest host passed | false pass on those | reason named on those |
|---|---|---|---|---|---|---|---|
| verify_result | 1,008 | 0 | 0.30 % | 993 | 810 (v1: 792) | 0 | 810 |
| verify_run | 1,008 | 0 | 0.30 % | 993 | 810 (v1: 792) | 0 | 810 |

The 18 extra honest artifacts are on E-rule hosts (207 to 225 of 261): two E-rule seed-0 hosts that now pass, times
9 faults (inferred: the two QPs are the only E-rule hosts whose point changed from failing to correct). The one-sided
95 % upper bound on the false-pass rate over the 810 informative artifacts is 0.37 %.
The 15 cells per path where the fault's own reason was not named are the MO hosts, as in v1. Every fault was
rejected. With 870 scored rule-arm runs added, the denominator is 1,878 per path and the false-pass count is 0.

## 7. Verification status (E-rule)

| run status | cells | where |
|---|---|---|
| verified | 223 (v1: 214) | |
| budget_exhausted | 21 (v1: 0) | `rastrigin-d10-shifted` (10, all wrong), `ackley-d10-shifted` (10, all right), `cqp-equality-d3` seed 1 |
| unverified_best | 46 (v1: 76) | the three scoring MO instances (30: NSGA-II has no alternate), `cqp-projection-d2` (10, all right), `integ-nearsing-1e-6`, `knapsack-p07-n15`, `knapsack-correlated-n20` (2 each) |

`optimality_certified` passed in 20 cells, as in v1. Every verified cell succeeded.

E-rule's evaluation cost is still its verification
([`erule_evaluation_split.jsonl`](../../benchmarks/results/e14-v2/erule_evaluation_split.jsonl), descriptive): on
`cqp-equality-d3` the loop spends 37 evaluations and the final verification 4,743 (median); on
`cqp-projection-d2` 12 and 723. A and B verify nothing.

## 8. Remaining failures

| arm | instance | success | note |
|---|---|---|---|
| A, B, E-rule | `rastrigin-d10-shifted` | 0/10 each | no arm finds the shifted basin; E-rule's median gap 6.96 |
| A | `rastrigin-d20` | 0/10 | |
| A, B | `rastrigin-d5` | 6/10 | |
| A | `cqp-projection-d2`, `cqp-simplex-d5`, `cqp-simplex-d10`, `lp-wyndor-d2`, `knapsack-p01-lp-relaxation` | 0/10 each | gaps a few times the tolerance |
| A | `cqp-equality-d3`, `knapsack-correlated-n20` | 5/10 each | |
| A | integration (4 instances), `rosenbrock-d5`, `rosenbrock-d20`, `ackley-d20`, `ackley-d10-shifted`, `knapsack-2d-n12`, `knapsack-p07-n15`, `mo-zdt2-d5` | 6/10 to 9/10 | |
| B | `ackley-d20` 9/10, `ackley-d10-shifted` 8/10 | | |
| E-rule | `mo-zdt2-d5` 2/10, `mo-zdt1-d5` 9/10 | | NSGA-II's front misses the 5 % hypervolume tolerance on the non-convex ZDT2 |

Every A and B failure is v1's, row for row. E-rule lost no cell it had in v1.

## 9. What was not run

| item | why |
|---|---|
| arms C, D, E (as `E-LLM`), `formulation_accuracy`, compiler `ir_schema_validity`, token cost | no provider; 290 `not_run` rows each for C, D and E-LLM, reason `no provider` |
| primary comparisons C-A, D-A, E-D, E-B | an arm of each was not run; p = 1 in the Holm family of 15 |
| external CUTEst and COCO instances | not vendored (spec section 3) |

A provider comes only from the environment. The OpenAI-compatible path through these variables is tested against a
fake local endpoint with no network (`tests/test_e14_10_campaign.py`).

| variable | meaning | example for DeepSeek |
|---|---|---|
| `E14_LLM_KIND` | `openai_compat` (default) or `local` | `openai_compat` |
| `E14_LLM_BASE_URL` | the endpoint | `https://api.deepseek.com/v1` |
| `E14_LLM_MODEL` | the model id, recorded in each shard's provenance | `deepseek-chat` |
| `E14_LLM_API_KEY` | the key for `openai_compat`; read at call time, never stored | the key |

Historical names: these are the variables v2 read. From campaign v3 on, the campaign, the CLI and Mission
Studio read one set, `LLM_KIND`, `LLM_MODEL`, `LLM_BASE_URL` and `LLM_API_KEY`
([providers](../agent/providers.md#configuration-from-the-environment)); the `E14_LLM_*` names are gone.

Running the provider arms means rerunning the `cells.*.C.*`, `cells.*.D.*`, `cells.*.E-LLM.*` and
`formulation.scoring` shards. A provider changes no frozen digest, but the shards already written as `not_run`
would have to be removed by hand or written to a new output directory.

## 10. Threats to validity

- **No LLM.** Nothing here measures what a model adds.
- **E-rule is not arm E.** It runs on the hand-written problem and cannot fail at formulation.
- **Verification is charged to E-rule only.** The cost endpoints compare an answer with a verified answer.
- **Restoration is a change of the solvers, not only of the ranking.** Every E-rule cell on a constrained continuous
  problem ran slightly different solver budgets (the reserve). Measured effect: 9 instances moved, none for the
  worse in success or verification; evaluations rose by 1 to 18 on four instances and fell on `rosenbrock-disk-d2`.
- **The clean threshold is a choice.** `SELECTION_FRACTION = 1e-6` of the band is set so that a clean point's bought
  gain is far below every tolerance in the catalogue for multipliers up to about 1e3. A problem with a much larger
  multiplier could still let a clean point buy more than its objective tolerance.
- **The guard can still refute an exact result with a band point it cannot restore** (an alternate with no
  evaluations left). That is a false fail, never a false pass; none occurred here.
- **Budget typing for E-rule is partly vacuous on the spec's own condition** (section 6.1).
- **Runner choices, the one v1 tuning change, tolerance edges, small groups, memorisation and one shared host**:
  as in [v1, section 8](e14_qualification_report.md#8-threats-to-validity). The host was another machine of the same
  pool, under 3 to 18 load from other users.

## 11. Data and integrity

| file | where | digest |
|---|---|---|
| 105 shard files (4.5 MB) | server, `~/e14-10-campaign-v2/out/shards/` | each in [`manifest.json`](../../benchmarks/results/e14-v2/manifest.json) |
| E-rule run directories (7.9 MB) | server, `~/e14-10-campaign-v2/out/runs/` and `~/e14-10-campaign-v2/runs-3311d09.tar` | tar `sha256:d977345fe1c78dc8c4276adaeb03d16ee615623dfab2599a6d60c9c4d8514492` |
| per-cell CSV | [`cells.csv`](../../benchmarks/results/e14-v2/cells.csv) | `sha256:1201b57cc025f5f4cef73323ccc05bb0e3b112b5d71587ac0cde71af9f9bd053` (in the manifest) |
| analysis | [`analysis.json`](../../benchmarks/results/e14-v2/analysis.json), [`tables.md`](../../benchmarks/results/e14-v2/tables.md) | content digest `sha256:2fe70d6e85bb46b8c03380e77584a3b4407ad9a49156d3e0900df48ef86326f1` |
| freeze | [`freeze.json`](../../benchmarks/results/e14-v2/freeze.json) | section 3 |
| tuning dry run (not scored) | server, `~/e14-10-campaign-v2/out-dryrun/` (code in `code-dryrun/`) | |

The copy-back was `rsync` of `out/` without `runs/` to the laptop, then `verify`, which re-hashed every file the
manifest names: all matched. Nothing on the server was deleted.

## 12. Reproduction

```bash
git archive 3311d09 | tar -x -C code-3311d09
cd code-3311d09
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python benchmarks/bench_e14_end_to_end.py plan                    # digests and the 105 shards
for f in rosenbrock rastrigin ackley constrained_quadratic linear_program knapsack integration bi_objective; do
  python benchmarks/bench_e14_end_to_end.py run --all --family $f --out ../out
done
python benchmarks/bench_e14_end_to_end.py run --shard formulation.scoring --out ../out
python benchmarks/bench_e14_end_to_end.py collect --out ../out
python benchmarks/bench_e14_end_to_end.py verify --out ../out
python benchmarks/bench_e14_end_to_end.py analyze --out ../out
python benchmarks/bench_e14_end_to_end.py report --out ../out
python benchmarks/e14_analysis.py ../out > erule_evaluation_split.jsonl
```

The regression tests for the two fixes, with no server and no model:

```bash
PYTHONPATH=src python -m pytest -q tests/test_e14_05_budget_typing.py tests/test_e14_05_candidate_selection.py
```
