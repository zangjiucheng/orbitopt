# E14-10 qualification report: campaign v3

This is the first scored run of the [E14-10 benchmark specification](e14_benchmark_spec.md) with a language model.
It ran the three provider arms against DeepSeek's `deepseek-flash`: arm C (the model proposes points, no solver),
arm D (the LLM compiler plus the fixed arm-A solver) and arm E as `E-LLM` (the LLM compiler plus the LLM planner in
the runtime). It also ran the formulation suite and, in the same freeze, the three arms that need no model (A, B,
`E-rule`). The plan it followed is the [campaign v3 plan](e14_llm_campaign_plan.md). [Campaign v1](e14_qualification_report.md)
and [campaign v2](e14_qualification_report_v2.md) stay as recorded. The results are in
[`benchmarks/results/e14-v3/`](../../benchmarks/results/e14-v3/).

**Read this first.** One model, one provider, one day, temperature 0. The 10 seeds of a provider cell send the same
request, so they measure the provider's run-to-run variation and the solver's seed, not 10 independent samples of a
model. The unit of every claim is the instance (29) or the formulation case (58), not the cell.

## 1. The question and the short answer

The spec asks whether the LLM problem compiler and the iteration agent help over traditional baselines, per problem
class (spec section 0). The five primary comparisons are B-A, C-A, D-A, E-D and E-B, with E as E-LLM.

- **The compiler never produced a wrong problem that it accepted on a scored instance.** Each of the 400 compiles of
  arms D and E-LLM was equivalent to the reference instance at the probe points, so every hard constraint and bound
  was preserved (2,100 of 2,100 each). All 280 formulation compiles that have a reference were equivalent to it.
  Schema validity was 1.00 (870 of 870 for D, 290 of 290 for E-LLM).
- **It refused two whole classes.** It called every integration statement (6 instances) and every bi-objective
  statement (3 instances) unsupported, on every seed, in D and in E-LLM: 90 of 290 cells per arm. The formulation
  suite shows the same refusals.
- **Formulation accuracy is 0.690 (95 % CI 0.603 to 0.776), below the spec's 0.95.** Nine of the 58 cases expect an
  outcome that only the scripted test replies can produce. Without them it is 0.816 (0.714 to 0.898), still below.
- **D vs A: inferior on success** (-0.283, Holm p 0.031). All of it comes from the compile step: 90 refusals and 10
  cells where the compiled problem had an unbounded variable that arm D's solver cannot take. On the 19 instances
  that compiled and solved, D equals A cell for cell in success (190 of 190 cells), with an ERT ratio of 1.00.
- **E-LLM vs D: superior on success** (+0.252, CI 0.159 to 0.345, Holm p 0.037). Both use the same compiler, so this
  is the runtime's exact solvers beating differential evolution, not the model.
- **E-LLM vs B: not shown different on success** (-0.286, Holm p 0.070), **inferior on evaluations (11.1 times) and
  wall clock (91.7 times).**
- **The LLM planner against the rule planner (E-LLM vs E-rule, added after the freeze)** is inferior on success
  (-0.314) because of the 90 refusals and one instance, `ackley-d10-shifted` (E-LLM 0 of 10, E-rule 10 of 10). On the
  20 compiled instances the success difference is -0.05 with 1 informative instance (not testable), and E-LLM is
  11.8 times slower in wall clock. The planner did not help.
- **C vs A: not shown different on success** (-0.134, Holm p 0.80). C wins the textbook global functions and loses
  knapsack, integration and bi-objective. Its wins are not search: on the shifted 10-D Rastrigin and Ackley it
  proposes the shift vector from the statement as its first point and stops after 1 evaluation.
- **Safety mostly holds.** 0 false passes in 1,008 injected faults per verifier path. Replay EQUAL on all 1,740
  scored cells and all 580 compiles. Two misses: **3 E-LLM runs were verified while 36.8 from the optimum** (a local
  minimum of the shifted Rastrigin), and **72 arm-C cells passed the 200,000-token budget** (by up to 19.7 %), so C's
  budget enforcement is 218 of 290.
- **Spend: $3.33 at list price** for 3,786 calls and 23.5 M tokens, under the approved $5.00 hard stop. Every
  response named `deepseek-flash`.

Where the LLM helped: nowhere in a pre-registered comparison. Where it did not: every LLM arm had a lower success
rate than the non-LLM arm it was compared with (C and D against A, E-LLM against B and E-rule), through refusals for
D and E-LLM, and the LLM arms took 3 to 92 times the non-LLM arms' wall clock. The compiler's accepted output was
always right, which is the strongest positive result here.

## 2. Setup

| item | value |
|---|---|
| spec digest (unchanged since v1) | `sha256:e5104c340d419f388643bed2ad1de32662f8384f7b1c4adb7a05b90ad877db47` |
| protocol (`e14-10-campaign/3`) | `sha256:ad381579aafcf84a55ac525052406cbff5005e22a55bd14ebc493dab8b72518a` |
| code | `sha256:9f4ed53b1c00380a372f1aa93854bada820c2929d12f0a99f5e2dc271ddd42f1` |
| freeze file content | `sha256:7caa7d0255edc82202b4d7e6e22a378e5ce30ad269c97e71fd961b597d3699fd`, [`freeze.json`](../../benchmarks/results/e14-v3/freeze.json), commit `2c34539` |
| shipped tree | `git archive 7661026` (the freeze plus the plan document); `plan` on the server printed the four digests above |
| prompt digests | compiler system `sha256:8c5290c3...`, compiler schema instruction `sha256:1722591e...`, planner system `sha256:955502fd...`, planner schema instruction `sha256:1ed30d7d...`; equal to `freeze.json`'s `protocol.llm.prompt_digests`, checked on the server before the first call |
| provider | `deepseek`, model `deepseek-flash`, `https://api.deepseek.com/chat/completions`, temperature 0, thinking disabled, no seed sent, `json_object` with the schema in the system instruction and one repair turn |
| per-cell caps | calls C 20, D 3, E-LLM 20; 200,000 tokens per cell |
| price table | per million tokens: cache hit $0.006, cache miss $0.30, output $1.20 (peak list price, read 2026-11-28) |
| hard stop | `--max-llm-usd 5 --max-llm-calls 14210`, one ledger for every process |
| server | `j7zang`, every shard on host `ubuntu2404-008`, `~/e11env` (Python 3.12.13, scipy 1.18.1, numpy 2.5.5) |
| workspace | `~/e14-10-campaign-v3/` (code `code-7661026/`, results `out/`, logs `logs/`); v1 and v2 workspaces untouched |
| launchers | [`launchers/`](../../benchmarks/results/e14-v3/launchers/): `e14run.sh` (the plan's, plus a refusal without the key file and a stop marker), `e14jobs.sh` (the plan's, with the hard stop as an argument and a retry of a killed run), `e14watch.sh` (the model-name watchdog), `e14status.py` and `e14project.py` (ledger reading and the pilot projection) |
| jobs | formulation shard alone; then the pilot (3 processes); then 4 family processes, staggered by 20 s, threads pinned to 1, `nice 10` |
| wall time | 06:45:04 to 08:07:03 UTC on 2026-11-28, 1 h 22 min. Model latency summed over calls was 2 h 13 min. |

**The key.** It lived in the laptop's `.env` and reached the server only as `~/.llm_env`, written with `umask 077`
(mode 600, checked), and was sourced inside the launcher. The launcher then exported the frozen kind, model and base
URL over the file's own. It refused to start without the file, so no provider shard could be written as
`not_run: no provider` by accident. After the last provider shard the file was removed with `shred -u` and checked
absent. No file in the server workspace or in the committed results contains the key value; the check ran `grep -F`
with the key read from standard input. The 1,450 cassettes contain no path, host or user name; their prompts are
benchmark problem text only.

## 3. How it ran: stages and the ledger

The run went in three stages, with the ledger read after each. A stage cap below $5 bounded the first two.

| stage | what ran | calls | tokens | cache-hit share | USD (list) | ledger after |
|---|---|---|---|---|---|---|
| 1, cap $1.00 | formulation suite: 58 cases x 10 seeds | 586 | 1,210,333 | 0.883 | 0.2241 | $0.2241 |
| 2, cap $0.85 | pilot: the Rosenbrock shards of C, D and E-LLM, 120 cells | 440 | 2,657,324 | 0.853 | 0.3957 | $0.6198 |
| 3, cap $5.00 | every other scoring shard, rule arms and faults included | 2,760 | 19,657,463 | 0.873 | 2.7144 | $3.3342 |
| **total** | 57 scoring shards | **3,786** | **23,525,120** | 0.871 | **3.3342** | |

The cache-hit share is cache-hit prompt tokens over all prompt tokens. The ledger's settled totals equal the sum of
the rows' own usage to the call, the token and the cent.

- **The pilot gate.** From the pilot's finished cells the projection was $3.76 (C's 250 remaining cells at $0.0102
  each, the mean of its cells of dimension 5 or less, the higher of its two rates; D at $0.00104 and E-LLM at
  $0.00264 per cell, their dearest instance). That was under the $4.00 gate, so stage 3 ran. The outcome was $3.33.
  C on knapsack and bi-objective cost about $0.02 per cell, three times its Rosenbrock rate, and the conservative D
  and E-LLM rates made up for it.
- **Against the Phase A estimate.** The plan's expected scenario was $1.63, its high one $11.88. C cost $2.45
  against an expected $0.80: it used 2,012 calls (6.9 per cell) where the dry run saw 2 to 7.
- **Billing.** The ledger prices every token at the peak list price. Every call was on a Saturday, which the
  pricing page read in Phase A puts off peak at half price, so the bill is probably about $1.67. That is inferred;
  the account's bill was not read.
- **Calls and attempts.** 3,786 calls took 3,786 HTTP attempts: no 429, no 5xx, no retry, no account error, no
  timeout. Every one of the 3,786 responses named `deepseek-flash`; a watchdog that would have stopped the run on any
  other name never fired.
- **Things that went wrong.** The first launch of stage 3 passed each job's two family names as one word (the
  server's login shell is zsh, which does not split an unquoted variable). The four jobs matched no shard, wrote
  nothing, made no call and exited 0; the relaunch split the names. Late in the run my ssh sessions started landing
  on another host of the pool (`ubuntu2404-004`), so the processes on `ubuntu2404-008` were followed through the logs
  on the shared home. No shard failed, no job retried, no hard stop fired.
- **Order.** The protocol's jobs rule runs one family's arms in turn (A, B, E-rule, C, D, E-LLM). The staging ran the
  formulation shard and the three Rosenbrock provider shards first, then the four family jobs, which skipped those
  shards. Only wall clock can depend on that order.

## 4. What ran and what did not

| item | cells | note |
|---|---|---|
| A, B, E-rule | 290 each | A and B: all 580 row digests equal v2's. E-rule: all 290 rows equal v2's in every field but the timing fields and `artifact_digest`, which hashes a run manifest holding wall-clock times |
| C | 290 | 200 ended on the model's stop flag, 72 were cut by the token budget, 18 ended on a reply that failed validation after the repair turn (4 of them with no evaluated point) |
| D | 290 | 190 compiled and solved, 10 compiled but errored in the solver (`lp-wyndor-d2`, section 8), 90 refused as unsupported; 1 compile took a second call |
| E-LLM | 290 | 200 compiled and ran the runtime, 90 refused as unsupported |
| formulation | 580 compiles | 5 took more than one call (6 extra calls) |
| faults | 1,008 per verifier path | as in v2 |
| tuning shards | not run | by the plan: `--split scoring` leaves out 120 tuning cells that would call the model |

No cell is `not_run`. No cell hit the campaign hard stop. The 72 C cells cut by the per-cell token budget are typed
`budget_exhausted:llm_tokens` and scored as failures, as the protocol says.

## 5. Results per class

Success counts cells. Gap is normalised. "Budget exhausted" counts cells whose `stop` is `budget_exhausted:*`. Wall
clock includes LLM latency; for E-LLM it adds the compile latency the row leaves out (section 11).

| class | arm | cells | success | feasible | median gap | budget exhausted | median evaluations | median wall s |
|---|---|---|---|---|---|---|---|---|
| lp | A | 20 | 0 | 20 | 5.18e-06 | 0 | 9,220 | 0.408 |
| lp | B | 20 | 20 | 20 | 0 | 0 | 8 | 0.00125 |
| lp | C | 20 | 5 | 20 | 1.1e-07 | 6 | 118 | 26.7 |
| lp | D | 20 | 0 | 10 | 3.46e-07 | 0 | 8,660 | 2.78 |
| lp | E-LLM | 20 | **20** | 20 | 0 | 0 | 2,750 | 4.48 |
| lp | E-rule | 20 | 20 | 20 | 0 | 0 | 3,010 | 0.24 |
| convex_qp | A | 40 | 5 | 40 | 5.12e-08 | 5 | 4,780 | 0.219 |
| convex_qp | B | 40 | 40 | 40 | 1.37e-11 | 0 | 160 | 0.0207 |
| convex_qp | C | 40 | 38 | 40 | 2.78e-17 | 0 | 16.5 | 3.35 |
| convex_qp | D | 40 | 5 | 40 | 5.12e-08 | 5 | 4,780 | 1.8 |
| convex_qp | E-LLM | 40 | **40** | 40 | 1.39e-17 | 1 | 3,940 | 3.78 |
| convex_qp | E-rule | 40 | 40 | 40 | 2.78e-16 | 1 | 3,950 | 0.231 |
| milp | A | 40 | 33 | 40 | 0 | 0 | 4,020 | 0.685 |
| milp | B | 40 | 40 | 40 | 0 | 0 | 15.5 | 0.0116 |
| milp | C | 40 | 5 | 35 | 0.0981 | 28 | 48.5 | 37.6 |
| milp | D | 40 | 33 | 40 | 0 | 0 | 4,020 | 4.34 |
| milp | E-LLM | 40 | **40** | 40 | 0 | 0 | 9,800 | 6.23 |
| milp | E-rule | 40 | 40 | 40 | 0 | 0 | 9,800 | 0.768 |
| nlp | A | 40 | 36 | 40 | 1.04e-10 | 0 | 13,800 | 0.475 |
| nlp | B | 40 | 40 | 40 | 6.43e-09 | 40 | 15,000 | 0.39 |
| nlp | C | 40 | 32 | 40 | 0 | 8 | 40 | 7.12 |
| nlp | D | 40 | 36 | 40 | 1.31e-10 | 0 | 13,800 | 2.32 |
| nlp | E-LLM | 40 | 40 | 40 | 2.38e-11 | 30 | 8,170 | 7.28 |
| nlp | E-rule | 40 | 40 | 40 | 3.41e-11 | 32 | 12,500 | 0.629 |
| global | A | 60 | 33 | 60 | 2.31e-08 | 0 | 18,200 | 0.643 |
| global | B | 60 | 33 | 60 | 2.31e-08 | 0 | 18,200 | 0.645 |
| global | C | 60 | **56** | 59 | 4.44e-16 | 3 | 21 | 9.03 |
| global | D | 60 | 33 | 60 | 2.23e-08 | 0 | 18,200 | 3.91 |
| global | E-LLM | 60 | 40 | 60 | 4.44e-16 | 57 | 19,100 | 10.1 |
| global | E-rule | 60 | 50 | 60 | 6.53e-09 | 56 | 16,500 | 0.891 |
| black_box | A | 60 | 53 | 60 | 0 | 0 | 319 | 0.0363 |
| black_box | B | 60 | 60 | 60 | 0 | 0 | 104 | 0.0167 |
| black_box | C | 60 | 10 | 16 | 0 | 1 | 21.5 | 6.49 |
| black_box | D | 60 | 0 | 0 | inf | 0 | 0 | 2.14 |
| black_box | E-LLM | 60 | 0 | 0 | inf | 0 | 0 | 2.09 |
| black_box | E-rule | 60 | 60 | 60 | 0 | 0 | 2,380 | 0.095 |
| multi_objective | A | 30 | 29 | 30 | 0.0178 | 0 | 9,410 | 0.504 |
| multi_objective | B | 30 | 30 | 30 | 0.0105 | 30 | 10,000 | 0.633 |
| multi_objective | C | 30 | 4 | 30 | 0.2 | 26 | 505 | 35.7 |
| multi_objective | D | 30 | 0 | 0 | 0.667 | 0 | 0 | 1.26 |
| multi_objective | E-LLM | 30 | 0 | 0 | 0.667 | 0 | 0 | 1.27 |
| multi_objective | E-rule | 30 | 21 | 30 | 0.0151 | 0 | 3,410 | 0.0808 |

No cell of any arm raised an alarm (a gap below minus the tolerance). The full table, with every comparison, is in
[`tables.md`](../../benchmarks/results/e14-v3/tables.md).

**The spec's absolute target** is success 1.00 on lp, convex_qp and milp for A, B, D and E. **E-LLM meets it**
(100 of 100), like B and E-rule. **D misses it** (38 of 100), for the same reason as A: its solver is A's
differential evolution, which stops just above the LP and QP tolerances. In D's lp row, 10 of the 20 cells are the
`lp-wyndor-d2` solver errors.

What limits these rows:

- D's and E-LLM's black_box and multi_objective rows are all refusals. They measure the compiler's coverage, not a
  solver.
- C's global row is analytic knowledge, not optimisation (section 8.1).
- Each class has 1 to 6 instances. The per-class claims below are made per analysis group, as in v1 and v2.

## 6. Paired comparisons

Statistics as in v1 and v2: success by the cluster sign-flip test on per-instance discordance, cost endpoints by the
exact Wilcoxon test on per-instance log ERT ratios, cluster bootstrap CIs (10,000 resamples, seed 1410, strata =
family), Holm per family, and the spec's verdict rule. Effects are treatment minus control (success) or
treatment over control (ERT, ERT-time); the gap effect is the mean log10 ratio of median gaps.

### 6.1 Primary family (the spec's 15 hypotheses, all run)

| comparison | success diff [95 % CI] | p Holm | verdict | ERT ratio [CI] | p Holm | verdict | ERT-time ratio [CI] | p Holm | verdict |
|---|---|---|---|---|---|---|---|---|---|
| B vs A | +0.255 [0.190, 0.328] | 0.00037 | superior | 0.225 [0.168, 0.30] | 0.00034 | superior | 0.346 [0.293, 0.402] | 0.0011 | superior |
| C vs A | -0.134 [-0.238, -0.038] | 0.80 | not shown different | 0.0284 [0.0138, 0.0646] | 0.80 | not shown different | 63.2 [28.8, 136] | 0.42 | not shown different |
| D vs A | -0.283 [-0.300, -0.259] | 0.031 | **inferior** | 1.00 [1.00, 1.00] | 0.022 | not shown different | 3.20 [2.37, 4.36] | 7.2e-06 | **inferior** |
| E-LLM vs D | +0.252 [0.159, 0.345] | 0.037 | **superior** | 0.954 [0.827, 1.09] | 0.20 | not shown different | 1.39 [1.24, 1.57] | 0.83 | not shown different |
| E-LLM vs B | -0.286 [-0.352, -0.221] | 0.070 | not shown different | 11.1 [8.86, 13.7] | 0.0023 | **inferior** | 91.7 [65.7, 131] | 6.5e-05 | **inferior** |

Reading the rows:

- **C vs A.** The CI on success excludes 0, but the sign-flip test does not reject (raw p 0.30). C's discordance is
  large in both directions: it wins every textbook global instance and loses knapsack, integration and
  bi-objective. The spec's rule needs both, so "not shown different". The ERT ratio of 0.028 counts C's 1 to 20
  evaluations on the instances it reads off the statement.
- **D vs A, ERT.** The Holm p (0.022) is below 0.05 but comes from the 9 instances only A solved; on the instances
  both solved the ratio is 1.00 to three figures (3 of them differ at all). The CI is [1, 1], so the verdict is "not
  shown different".
- **E-LLM vs D** isolates the runtime: both arms compile with the same prompt, then D runs differential evolution and
  E-LLM runs the planner over the solver registry. Of E-LLM's 73 extra successes, 62 are on lp, convex_qp and milp,
  where the runtime has exact solvers; the other 11 are on nlp and global.

The gap endpoint is a secondary family of the 7 tested overall comparisons: D vs A inferior (+0.202 in log10, Holm
p 0.040), E-LLM vs D superior (-1.95, Holm p 0.031), E-rule vs A and E-rule vs B superior as in v2; C vs A, E-LLM vs B
and B vs A not shown different.

### 6.2 Secondary and exploratory families

| comparison | family | success diff | verdict | ERT ratio | verdict | ERT-time ratio | verdict |
|---|---|---|---|---|---|---|---|
| E-rule vs A | secondary (6) | +0.283 [0.193, 0.376] | superior | 1.46 | not shown different | 1.04 | not shown different |
| E-rule vs B | secondary (6) | +0.028 [-0.035, 0.090] | not shown different | 10.4 | inferior | 4.63 | inferior |
| E-LLM vs E-rule | exploratory (4), added after the freeze | -0.314 [-0.379, -0.252] | **inferior** | 0.819 [0.788, 0.863] | "superior", see below | 11.8 [7.87, 17.7] | **inferior** |

The E-rule rows equal v2's except the wall clock, which is the host's (v2: 0.993 and 4.54).

**The E-LLM vs E-rule ERT verdict is an artifact.** The frozen verdict joins a Wilcoxon p over all 29 instances with
a CI over the finite ratios only. Here the p (raw 0.013) comes from the 10 instances where E-LLM never succeeded and
E-rule did: they rank as infinitely worse for E-LLM. The CI comes from the 19 others, where E-LLM used fewer
evaluations (geometric mean 0.82, most of it `lp-wyndor-d2` at 0.04). The two halves point in opposite directions and
the rule reads "superior". The analysis now records which side the signed ranks favour
(`wilcoxon_direction`, "treatment worse" here). It is the only decided verdict in the campaign where the side and the
verdict disagree. On the 20 compiled instances alone the same endpoint is not shown different (Holm p 1).

### 6.3 Per analysis group

Holm over the 21 tests of each group (7 comparisons x 3 endpoints); the exploratory comparison is its own family.

| group | comparison | result |
|---|---|---|
| algebraic (10) | E-LLM vs D | success +0.62 [0.44, 0.81], not shown different after Holm; ERT 1.32 and ERT-time 1.10 not shown different |
| algebraic (10) | E-LLM vs B | success 0 (no informative instance); ERT 95.9 and ERT-time 469, inferior |
| algebraic (10) | E-LLM vs E-rule | success 0; ERT-time 15.6, inferior |
| continuous_nonconvex (10) | C vs A | ERT 0.0044, superior; gap -4.9, superior; success +0.19, not shown different |
| continuous_nonconvex (10) | E-LLM vs E-rule | success -0.10 (1 informative instance, not testable); ERT-time 8.7, inferior |
| black_box (6), multi_objective (3) | every LLM arm | not shown different or not testable; every D and E-LLM cell is a refusal |

C's two decided wins in continuous_nonconvex are the contamination case of section 8.1. They are not evidence that
proposing points beats differential evolution.

### 6.4 Post hoc: the 20 instances every D and E-LLM cell compiled

Descriptive, one Holm family of 16 tests. This separates what the compiler refused from what happened after it.

| comparison | success diff | ERT ratio | ERT-time ratio |
|---|---|---|---|
| D vs A | 0 (0 informative) | 1.00 (3 informative) | 3.20, inferior |
| E-LLM vs D | +0.365 [0.235, 0.500], Holm p 0.068 | 0.924 | 1.75 |
| E-LLM vs B | +0.035 (4 informative, not testable) | 11.1 | 91.7, inferior |
| E-LLM vs E-rule | -0.05 (1 informative, not testable) | 0.819, not shown different | 11.8, inferior |

On what it compiled, D is A plus about 2 s of compile latency per cell, and E-LLM is E-rule with fewer evaluations
on some instances, one instance lost, and an order of magnitude more wall clock.

## 7. Formulation accuracy

580 compiles: 58 cases x 10 seeds, under arm D's caps. A compile is correct when its outcome equals the case's
expected outcome and, for a compile, the problem is equivalent to the reference (same optimum within tolerance and
the same feasibility verdict at 1,000 probe points). CIs are cluster bootstrap over cases, strata language x expected
outcome.

| slice | cases | correct compiles | accuracy | 95 % CI |
|---|---|---|---|---|
| **overall** | 58 | 400 of 580 | **0.690** | [0.603, 0.776] |
| English | 50 | 340 of 500 | 0.680 | [0.58, 0.76] |
| Chinese | 8 | 60 of 80 | 0.750 | [0.50, 0.875] |
| supported (expected `compile`) | 34 | 280 of 340 | 0.824 | [0.676, 0.941] |
| unsupported | 6 | 50 of 60 | 0.833 | [0.50, 1.00] |
| ambiguous (expected `needs_clarification`) | 8 | 70 of 80 | 0.875 | [0.625, 1.00] |
| expected `rejected_unauthorized_change` | 5 | 0 of 50 | 0 | |
| expected `compile_failed` | 5 | 0 of 50 | 0 | |
| English supported / unsupported / ambiguous | 29 / 5 / 7 | 240 / 40 / 60 | 0.828 / 0.80 / 0.857 | |
| Chinese supported / unsupported / ambiguous | 5 / 1 / 1 | 40 / 10 / 10 | 0.80 / 1.00 / 1.00 | |
| without the 9 reply-determined cases (post hoc) | 49 | 400 of 490 | 0.816 | [0.714, 0.898] |

**Against the 0.95 target: missed.** The CI's upper end (0.776) is below it, and stays below (0.898) without the
reply-determined cases.

**Every compiled problem that has a reference was right.** 280 compiles of the 34 "compile" cases had a reference
and all 280 were equivalent to it. The 60 misses among those cases are refusals: the model never accepted a wrong
problem there.

**The reply-determined cases.** The suite was written for scripted replies. Nine cases expect what only those replies
can produce: FC-044 to FC-048 expect `compile_failed` because their scripted replies are malformed JSON, unsafe
expressions, a dropped constraint, invalid IR or a silent relaxation. FC-031 and FC-041 to FC-043 are revisions of a
base problem that the live harness does not send. A model that compiles "Minimize x^2 for x between -1 and 1"
correctly fails FC-045. The nine were listed from the case notes before any v3 outcome was read; the frozen metric
keeps them, and the post hoc row leaves them out.

**What the model got wrong** (each case behaves the same on all 10 seeds except FC-039):

| case | expected | the model's outcome | what it said (quoted, shortened) | the checker |
|---|---|---|---|---|
| FC-005, FC-022 (integration) | compile | unsupported, 10 of 10 each | "Choosing the quadrature rule and the number of points ... requires comparing error estimates across candidate rules ... (a discrete selec[tion])"; "the available tool ids (numerics.gauss_legendre.v1, numerics.adaptive_simpson.v1, numerics.quadrature_cost.v1) do not provid[e] ..." | outcome differs from expected |
| FC-023, FC-024, FC-Z04 (bi-objective) | compile | unsupported, 10 of 10 each | "The request asks for up to 50 non-dominated points (a Pareto front / set of solutions), which is a multi-solution enumeration that this single-solution optimization format cannot express" | outcome differs; the IR does hold two objectives |
| FC-004, FC-Z08 (relaxation in an attached log) | rejected_unauthorized_change | compile, 10 of 10 each | every one of the 20 compiles kept `x + y >= 1` as stated, with no authorised change: the attachment was ignored, not flagged | outcome differs; the compiled problem is the safe one |
| FC-036 (random demand) | needs_clarification | unsupported, 10 of 10 | "The demand is random, which requires a stochastic model; this format cannot express randomness or expectations. Should the demand be replaced by a deterministic value?" | outcome differs; the model asked the right question under another status |
| FC-039 (paint colour) | unsupported | needs_clarification 6, compile 4 | on all 10 seeds the reply compiled `colour` as a discrete variable with choices [0, 1, 2]: "Encoded the categorical colours red, green, blue as the discrete choices 0, 1, 2 since the format requires numeric choices"; on 6 it also asked how to encode the colour, which the compiler turns into `needs_clarification` | 4 compiles accepted with an invented encoding |
| FC-031, FC-041 to FC-043 (revisions) | compile or rejected | needs_clarification, 10 of 10 each | "There is no current problem attached, so I cannot tell what should solve faster" | reply-determined |
| FC-044 to FC-048 | compile_failed | compile, 10 of 10 each | compiles of the stated problems; the cases have no reference IR, so they were not checked for equivalence (FC-044's, read by hand, matches its statement) | reply-determined |

FC-039's four compiles are the only accepted output that a stricter reading would call wrong. The colour is not in
the cost (2 t), so their optimum is right; the encoding was stated as an assumption.

**Seeds.** The 10 seeds of a case send the same HTTP body (the seed is recorded but not sent). The reply text
differed between seeds on 57 of the 58 cases, yet the outcome was the same on all 10 seeds for 57 of 58. The same
holds for the scored arms: the first reply's text differed across seeds on 28 of 29 instances for C and D and 27 of
29 for E-LLM. Temperature 0 is not deterministic on this provider; the outcomes mostly are.

## 8. Failures, with examples

### 8.1 Arm C knows the textbook functions

C succeeded on 10 of 10 seeds of `rastrigin-d10-shifted` and of `ackley-d10-shifted`. Every other arm scored 0 of 10
on the Rastrigin one. The statement gives the shift:

> Find the global minimum of the Rastrigin function 10n + sum_i (z_i^2 - 10 cos(2 pi z_i)) in 10 dimensions, where
> z_i = x_i - s_i with s = [0.7, -1.3, 2.1, -0.4, 1.8, -2.6, 0.3, 1.1, -0.9, 2.4], each x_i between -5.12 and 5.12.

The first reply proposes `[0.7, -1.3, 2.1, -0.4, 1.8, -2.6, 0.3, 1.1, -0.9, 2.4]`, the second stops. The median is 1
evaluation on both shifted 10-D instances and 19.5 to 21 on the three 20-D ones. This is the minimiser of a textbook function read
from its definition. The cassette is in
[`cassettes-sample/`](../../benchmarks/results/e14-v3/cassettes-sample/cells.scoring.C.rastrigin/C/rastrigin-d10-shifted/seed-0.cassette.jsonl).

Where C had to search it did badly: knapsack 5 of 50, bi-objective 4 of 30, integration 10 of 60 (in 44 of the 60
integration cells its best point was infeasible). 72 of its cells used more than the 200,000-token budget, 18 ended on
a reply that failed schema validation twice.

### 8.2 Arm D

- **Refusals** (90 cells): the six integration and three bi-objective statements, every seed, as in section 7.
- **`lp-wyndor-d2`, 10 of 10 errors**: `ValueError: variable x1: the fixed solver needs finite bounds`. The statement
  states nonnegativity and no upper bound; the model compiled exactly that (the reference adds a box [0, 100]). The
  compile is right; arm D's fixed solver, differential evolution, needs a finite box. E-LLM, on the same compile,
  solved all 10 with an exact LP solver.
- Everywhere else D equals A in success, cell for cell (190 of 190). The evaluation count differs on 25 of those
  cells, on 6 instances, and the returned point on 106, by amounts that change no success. Inferred: the compiled
  expressions are the reference's in another floating-point order, so differential evolution under the same seed
  drifts slightly.

### 8.3 Arm E-LLM

- **Refusals**: the same 90 cells as D.
- **`ackley-d10-shifted`, 0 of 10** (E-rule 10 of 10): the planner asked for a second differential-evolution run where
  the rule planner asked for a second local polish (solver lists of the trace, e.g. seed 0: DE, L-BFGS-B, DE, L-BFGS-B
  against DE, L-BFGS-B, L-BFGS-B). Every seed ended 1.6 to 1.7 from the optimum with the evaluation budget spent.
- **`rastrigin-d10-shifted`, 0 of 10**, as E-rule. Three of these runs were verified (section 9).
- Elsewhere 10 of 10 on every compiled instance.

## 9. Safety metrics

| metric | A | B | C | D | E-LLM | E-rule | target |
|---|---|---|---|---|---|---|---|
| budget enforcement | 290/290 | 290/290 | **218/290** | 290/290 | 290/290 | 290/290 | 1.00 |
| independent counter = reported evaluations | 290/290 | 290/290 | 290/290 | 290/290 | 290/290 | 290/290 | |
| experiment replay EQUAL | 290/290 | 290/290 | 290/290 | 290/290 | 290/290 | 290/290 | 1.00 |
| hard constraints preserved | n/a | n/a | n/a | 2,100/2,100 | 2,100/2,100 | 2,400/2,400 | 1.00 |
| IR schema validity | | | | 870/870 | 290/290 | | 1.00 |
| scored runs the guard verified | 0 | 0 | 0 | 0 | 165 | 223 | |
| verified and infeasible (spec's false pass) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **verified but not within tolerance** | | | | | **3** | 0 | |

The formulation suite replayed EQUAL on 580 of 580 compiles. Replay reruns each cell from its cassette with no model
call; it shows that the recorded run is reproducible from the artifact, not that DeepSeek would answer the same way
again (section 7, Seeds).

**Fault injection** was rerun on its hosts, the rule arms' seed-0 results and the stored optima, which are v2's (the
fault injectors do not depend on a model): 0 false passes of 1,008 per path, both paths (95 % upper bound 0.30 %); 0
of the 810 whose honest host passed (upper bound 0.37 %); the fault's own reason named on 993. The counts equal v2's.

**Verified while wrong: 3 of 165 E-LLM verified runs.** Seeds 5, 6 and 8 of `rastrigin-d10-shifted` ended at the same
local minimum, 36.81 above the optimum, feasible, and the guard passed them: `solver_converged`,
`constraints_satisfied` and `independently_verified` passed, `optimality_certified` was not assessed. The spec's false
verification pass counts a verified run whose recomputation with the reference callables fails; these recompute
correctly, so the spec's count stays 0. In plain terms the verifier labelled 3 wrong answers `verified`. The rate is
3 of 165 (1.8 %, 95 % Clopper-Pearson interval 0.38 % to 5.2 %), all on one instance. E-rule never reached `verified`
on this instance (its alternates stopped at their allocation, v2 section 7); E-LLM's runs left the alternate enough
evaluations to converge to the same basin and agree (inferred from the statuses; not traced call by call).

**Budget enforcement, arm C.** The meter checks the token budget before each call, but a call's size is known only
after it. 72 C cells crossed 200,000 tokens on their last call (median 215,112, at most 239,417, 19.7 % over). Each is
typed `budget_exhausted:llm_tokens` and scored as a failure, as the protocol says, so no answer was bought with the
overrun, but the spec's metric counts them out of budget. The campaign hard stop is not affected: it reserves a call's
worst case before the call. D and E-LLM never came near the token budget.

## 10. Cost, tokens, latency and models

| arm | cells | calls | tokens in | of which cache hits | tokens out | USD (list) | USD per cell | USD per success | median LLM latency s | median wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| C | 290 | 2,012 | 18,572,844 | 0.893 | 1,462,249 | 2.4483 | 0.0084 | 0.0163 (150 successes) | 9.62 | 9.66 |
| D | 290 | 291 | 537,354 | 0.896 | 143,724 | 0.1921 | 0.00066 | 0.0018 (107) | 2.06 | 2.30 |
| E-LLM | 290 | 897 | 1,364,302 | 0.551 | 234,314 | 0.4696 | 0.0016 | 0.0026 (180); **0.0029 per verified success** (162) | 4.40 | 4.88 |
| formulation | 580 | 586 | 1,059,298 | 0.883 | 151,035 | 0.2241 | 0.00039 | 0.00056 (400 correct) | | 1.33 |

- **Models returned**: `deepseek-flash` on all 3,786 responses (rows' `llm_models`, the ledger and the cassettes
  agree).
- **Where the money went**: C took 73 % of the spend. It resends the whole conversation each turn; its cache share
  is high because the prefix repeats.
- **Most calls in a cell**: C 20 (the cap), E-LLM 8 (compile plus planner), D 2.
- **Model latency summed over calls**: C 5,090 s, E-LLM 1,390 s, formulation 833 s, D 644 s.
- **Wall clock is model latency.** C's median wall clock (9.66 s) is its LLM latency (9.62 s). B solves an lp in
  1.3 ms; D spends about 2 s compiling it.

## 11. Defects found, not fixed

Each fix would change digested code, so none was made: a fix means a refreeze, which is the user's call. None of them
needs a rerun to be read correctly, because the cassettes and rows hold what is needed.

| defect | where | effect here | how this report handles it |
|---|---|---|---|
| E-LLM's `wall_clock_s` times the runtime only and leaves out the compile calls | `bench_e14_end_to_end.run_cell_e_llm` | under-states E-LLM's wall clock by its compile latency (median 2.0 s per cell) | the analysis adds each cell's compile latency from its cassette (`llm_latency.jsonl`). With the row's own value: E-LLM vs B 60.0 instead of 91.7, vs E-rule 7.81 instead of 11.8, vs D 1.10 instead of 1.39; no verdict changes |
| C's token budget is checked before a call and enforced after it | `CellMeter` through `UsageMeter` | 72 cells over budget, up to 19.7 % | reported; the cells are failures as the protocol says |
| D's fixed solver needs a finite box | `bench_e14_formulation.adapt` | 10 errors on a correct compile | reported as a D failure; arguably a limit of the arm's definition, not of the model |
| nine formulation cases expect outcomes only scripted replies produce | `tests/data/agent/formulation_cases.json` with the live harness | caps the frozen accuracy at 0.845 for a perfect model | the post hoc row without them |
| the guard verified a local minimum | runtime verification on a multimodal problem | 3 verified-while-wrong E-LLM runs | section 9 |
| the verdict rule can join a p and a CI of opposite sides | `e14_analysis.compare` (outside the digest) | one decided verdict misleads | the side is now recorded next to each verdict (section 6.2) |

The compiler's refusals of integration and bi-objective problems are a finding about this model with this prompt, not
a defect found in code. Whether a prompt change fixes them is untested.

## 12. Threats to validity

- **One model, one provider, one day.** Nothing here generalises to another model, or to `deepseek-flash` on another
  day. The provider changed no visible behaviour during the run, but its outputs vary between identical calls.
- **API nondeterminism, and what the seeds mean.** Identical requests got different reply text on nearly every
  instance. The outcomes were stable, but the 10 seeds of a provider cell are repeated draws of one prompt, not 10
  prompts. Every test clusters by instance or case for this reason.
- **Cassettes are the reproduction artifact.** Replay EQUAL means a cell is reproducible from its recording. A rerun
  against the API can differ; the protocol says so in advance.
- **Contamination.** Rastrigin, Rosenbrock and Ackley are public, with known closed-form minimisers, and the shifted
  variants state the shift. C exploited this directly (section 8.1). The compiler may also be helped by knowing these
  functions; its compiles were checked for equivalence, so that help cannot have produced a wrong accepted problem.
  The knapsack instances (Burkardt P01, P07) are public too; C did badly on them anyway.
- **Tuning on 8 instances.** The rule arms' choices were tuned on the 8 tuning instances in v1. The provider arms'
  prompts and caps were exercised by a 30-call dry run on those instances, seed 1000, and nothing was changed after
  it. The refusals were already visible there (`mo-schaffer-d1`, `integ-exp-1e-10`). Nothing was tuned after a scored
  call. Eight instances cannot show how a prompt generalises; the scoring set is where the refusals were measured.
- **The price snapshot.** USD is at the list price read on 2026-11-28 from a page with no date. The bill is probably
  half (off-peak), inferred and not checked.
- **Order effects.** The formulation and pilot shards ran before the family jobs, against the protocol's jobs order.
  Only wall clock can depend on order; rows other than timing cannot.
- **The formulation suite was written for scripted replies.** Nine of its cases cannot be met live (section 7); its
  expected outcome for an ignored attachment (FC-004, FC-Z08) penalises a compile that is safe.
- **The verifier's "verified" is not optimality.** Section 9's three runs show it can certify a local minimum.
- **Hard-constraint preservation is measured on compiled formulations only.** A refused statement preserves nothing
  and violates nothing; 90 cells per arm add no evidence either way.
- **Wall clock compares a network service with local solvers**, on a shared host with other users' load.
- **The pilot was the Rosenbrock family**, the cheapest for C. The gate held, but its projection under-stated C on
  knapsack and bi-objective by about three times.

## 13. Conclusion

On these 29 instances, with `deepseek-flash` at temperature 0:

- **The LLM compiler is safe and narrow.** It accepted no wrong problem on a scored instance and lost no constraint,
  and its output always validated. It refused integration and bi-objective statements wholesale, and that refusal,
  not wrong compilation, is what made D inferior to A and E-LLM inferior to E-rule on success. Formulation accuracy
  (0.690) is well below the spec's 0.95 target, and below it even without the nine cases a live model cannot meet.
- **The LLM planner did not help.** On the instances that compiled it matched the rule planner on success except
  for one instance it lost, used somewhat fewer evaluations, cost 8 to 12 times the wall clock, and produced the
  campaign's only verified-while-wrong runs.
- **The LLM alone (arm C) is not an optimiser here.** It looks strong only where it can read the answer off a
  textbook definition, and it fails knapsack, integration and multi-objective problems.
- **The one superior primary result with an LLM arm, E-LLM over D, measures the runtime, not the model.**
- **B, the classical exact-solver baseline, remains the arm to beat**, and no LLM arm beat it on any endpoint.

What the sample does not support: any claim about other models or prompts, any per-class claim for black_box or
multi_objective beyond "the compiler refused them", and any claim that the planner is worse in general (one
instance carries its success deficit).

## 14. Data and integrity

| file | where | digest |
|---|---|---|
| 57 scoring shards (5.5 MB) | server `~/e14-10-campaign-v3/out/shards/` | each in [`manifest.json`](../../benchmarks/results/e14-v3/manifest.json) |
| formulation shard | [`formulation.scoring.json`](../../benchmarks/results/e14-v3/formulation.scoring.json) | `sha256:e1210203...d2289b`, equal to the manifest |
| every cassette (1,450) and run directory | server `~/e14-10-campaign-v3/runs-7661026.tar` (69.5 MB) | `sha256:8589fc41f00868ba384f5340e62559253aac930e0d8180de0eca864ded7f4fe7` |
| eight sample cassettes | [`cassettes-sample/`](../../benchmarks/results/e14-v3/cassettes-sample/) | each in [`artifacts.json`](../../benchmarks/results/e14-v3/artifacts.json), checked against the server copies |
| ledger (7,572 lines) | server `out/llm_ledger.jsonl`; committed gzipped | `sha256:60800a8a...880042` |
| per-cell LLM latency | [`llm_latency.jsonl`](../../benchmarks/results/e14-v3/llm_latency.jsonl) | in `artifacts.json` |
| per-cell CSV | [`cells.csv`](../../benchmarks/results/e14-v3/cells.csv) | in the manifest |
| analysis | [`analysis.json`](../../benchmarks/results/e14-v3/analysis.json), [`tables.md`](../../benchmarks/results/e14-v3/tables.md) | content digest `sha256:f65eed7833ac076c818f6475cb589781ea97fa8fcf371889fbd1a4740c022b01`, the same on the laptop and the server |

The copy-back was `rsync` of `out/` without `runs/`, then `verify`, which re-hashed every file the manifest names:
all matched. Nothing on the server was deleted except the key file.

**Analysis added after the freeze** (in [`e14_analysis.py`](../../benchmarks/e14_analysis.py), outside the code
digest, changing no row): the E-LLM vs E-rule comparison as its own family; IR schema validity over the spec's full
denominator (arm D's cell compiles added to the suite's); the formulation breakdown with case-clustered CIs and the
post hoc row without reply-determined cases; per-arm LLM accounting; the E-LLM wall-clock correction; the Wilcoxon
side; the post hoc comparisons of section 6.4.

## 15. Reproduction

The analysis, with no model. `v3/` is the server's `out/` copied without `runs/` (the 57 shards, `manifest.json`,
`cells.csv`), plus `llm_latency.jsonl` from `benchmarks/results/e14-v3/`:

```bash
PYTHONPATH=src python benchmarks/bench_e14_end_to_end.py verify --out v3     # re-hash against the manifest
PYTHONPATH=src python benchmarks/bench_e14_end_to_end.py analyze --out v3    # analysis.json, content digest above
PYTHONPATH=src python benchmarks/bench_e14_end_to_end.py report --out v3     # tables.md
```

The provider cells, from the cassettes, with no model call. Untar `runs-7661026.tar` into a fresh `repro/`, keep the
`runs/` of the shards to rerun, and run them with a placeholder key and a hard stop of zero calls, so that any request
a cassette does not hold stops the run instead of reaching the network:

```bash
LLM_KIND=deepseek LLM_MODEL=deepseek-flash LLM_BASE_URL=https://api.deepseek.com LLM_API_KEY=placeholder \
PYTHONPATH=src python benchmarks/bench_e14_end_to_end.py run --shard cells.scoring.D.linear_program \
    --shard cells.scoring.E-LLM.linear_program --out repro --max-llm-calls 0 --max-llm-usd 0
```

Done on the server for those two shards: 0 calls in the ledger, the 10 D rows equal to the scored rows digest for
digest, the 10 E-LLM rows equal outside `artifact_digest` (a manifest of files that hold wall-clock times).

The live run, on the server, as the [plan](e14_llm_campaign_plan.md#5-phase-b-after-approval) says, with this
campaign's caps and the shell's word splitting made explicit:

```bash
C=code-7661026
./e14run.sh $C out formulation run --shard formulation.scoring --max-llm-usd 1.00 --max-llm-calls 14210
for a in D E-LLM C; do ./e14run.sh $C out pilot-$a run --shard cells.scoring.$a.rosenbrock --max-llm-usd 0.85 --max-llm-calls 14210 & sleep 20; done; wait
for g in "rosenbrock rastrigin" "ackley constrained_quadratic" "linear_program knapsack" "integration bi_objective"; do
  set -- $g   # bash; in zsh use ${=g}
  nohup ./e14jobs.sh $C out jobs-$1 5 "--split scoring" "$@" > /dev/null 2>&1 &
  sleep 20
done
```

Then `collect` and `verify` as in v2, and the latency table from the cassettes, with the analysis of this report's
commit (the shipped tree predates it):

```bash
PYTHONPATH=src python benchmarks/e14_analysis.py --llm-latency ../out > ../out/llm_latency.jsonl
```
