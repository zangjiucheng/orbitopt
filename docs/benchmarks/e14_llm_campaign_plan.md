# E14-10 LLM campaign plan: campaign v3

Campaign v3 runs the pre-registered provider arms of the [E14-10 benchmark specification](e14_benchmark_spec.md)
against DeepSeek: arm C (the model proposes points, no solver), arm D (LLM compiler plus the fixed solver) and arm
E as `E-LLM` (LLM compiler plus the LLM planner in the runtime), and the formulation suite behind arm D's
`formulation_accuracy` and the compiler's `ir_schema_validity`. The rule arms A, B and E-rule run again as in
[campaign v2](e14_qualification_report_v2.md), because the code digest moved.

**Status: Phase A and Phase B done.** Phase A prepared and froze the campaign and ran a 30-call tuning dry run.
Phase B ran on 2026-11-28 under a $5 hard stop in three stages and cost $3.33 at list price; its results are in the
[campaign v3 report](e14_qualification_report_v3.md).

## 1. What is frozen

| item | value |
|---|---|
| protocol | `e14-10-campaign/3`, `sha256:ad381579aafcf84a55ac525052406cbff5005e22a55bd14ebc493dab8b72518a` |
| spec (unchanged since v1) | `sha256:e5104c340d419f388643bed2ad1de32662f8384f7b1c4adb7a05b90ad877db47` |
| code | `sha256:9f4ed53b1c00380a372f1aa93854bada820c2929d12f0a99f5e2dc271ddd42f1` |
| freeze file | [`benchmarks/results/e14-v3/freeze.json`](../../benchmarks/results/e14-v3/freeze.json), content `sha256:7caa7d0255edc82202b4d7e6e22a378e5ce30ad269c97e71fd961b597d3699fd` |
| provider | `deepseek`, model `deepseek-flash`, `https://api.deepseek.com/chat/completions`; a scored provider shard refuses any other kind, model or base URL |
| sampling | temperature 0, thinking disabled, no seed sent (DeepSeek documents none) |
| structured output | `response_format` `json_object`, the JSON Schema in the system instruction, validated locally with one repair turn |
| per-cell caps | calls: C 20, D 3, E-LLM 20 (compile and planner together); 200,000 tokens; checked before every call |
| price table | `deepseek-flash` per million tokens: cache hit $0.006, cache miss $0.30, output $1.20 (peak list price; off-peak is half), from the [DeepSeek pricing page](https://api-docs.deepseek.com/quick_start/pricing), read 2026-11-28; the page shows no date |

The v1 and v2 freezes and results are unchanged. `PROTOCOL["changes_from_v2"]` in
[`e14_common.py`](../../benchmarks/e14_common.py) lists what moved; every rule-arm choice is v2's.

## 2. What Phase A changed

**One provider configuration.** The campaign, the CLI (`orbitopt agent ... --provider env`) and Mission Studio
read `LLM_KIND`, `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_TIMEOUT_S` through `orbitopt.llm_env`. The
campaign-only and Studio-only names of v1 and v2 are gone. The key comes only from the environment.
`--env-file FILE` reads `KEY=VALUE` lines without expanding anything, never overrides a variable that is set, and never prints a value
([providers](../agent/providers.md#configuration-from-the-environment)).

**DeepSeek.** `DeepSeekProvider` (`LLM_KIND=deepseek`) follows DeepSeek's documentation, read 2026-11-28
([providers](../agent/providers.md#deepseek)). Every response records the model the server named, its usage with
cache hits and misses, its latency and its HTTP attempts. 429 and 5xx are retried at most 3 times, honouring
`Retry-After` up to 60 s. 401 and 402 stop the campaign.

**The provider arms.** Each cell talks to the model through its cassette, its meter and the campaign ledger
([`e14_llm.py`](../../benchmarks/e14_llm.py)):

- The cassette records every call under `--runs`: the request, the response, the model returned, usage and
  latency. A rerun serves every call the cassette holds, so an interrupted shard never pays for a call twice.
- The meter enforces the spec's per-cell call cap and token budget before every call. Structured-output repairs,
  compiler repair rounds and planner iterations all count. An arm-C cell whose calls passed the token budget is a
  failure.
- The ledger, `<out>/llm_ledger.jsonl`, is the campaign-wide hard stop. It is shared by every process writing one
  output directory, under an exclusive lock. Before every network call it reserves a worst-case cost: one token
  per two characters of the request at the cache-miss price, plus `max_tokens` at the output price. A call that
  would pass `--max-llm-calls` or `--max-llm-usd` is refused. The shard is then left unwritten and a rerun
  resumes it.
- Each cell is replayed from its cassette with `ReplayProvider` and no model call, and the row records whether the
  replay reproduced it. That is the `experiment_replay` metric for the provider arms. DeepSeek's outputs can vary
  between identical calls even at temperature 0, so the cassette is the reproduction artifact, not a re-run.

**A scorer fix found by the dry-run tests.** The formulation suite matched compiled variables to the reference by
name only, so a correct compile that names its own variables (`food_1` for `x0`) failed. It now uses the case's
alias table when that names compiled variables, else the declared order, which is how arm D maps x back.

## 3. Tuning dry run (Phase A, real API)

Run on 2026-11-28 from the laptop, on the v3 code before the protocol was final, under a 40-call hard stop. Only
tuning instances and tuning seed 1000 were used. The prompts hold benchmark problem text only.

| arm | cells | calls | tokens in | of which cache hits | tokens out | USD (list) | replay EQUAL |
|---|---|---|---|---|---|---|---|
| D | 8 (every tuning instance) | 8 | 14,495 | 9,856 | 2,103 | 0.0040 | 8/8 |
| E-LLM | 3 (`cqp-worked-d2`, `lp-diet-d2`, `rastrigin-d2`) | 10 | 12,525 | 5,760 | 1,539 | 0.0039 | 3/3 |
| C | 3 (`cqp-worked-d2`, `knapsack-p02-n5`, `rastrigin-d2`) | 12 | 47,707 | 34,688 | 5,053 | 0.0102 | 3/3 |
| **total** | 14 | **30** | 74,727 | 50,304 | 8,695 | **0.0181** | 14/14 |

- Every response named `deepseek-flash`. No retry, no HTTP error, no blank JSON-mode reply, no repair round.
  Median latency was 1.3 s and the maximum 2.2 s.
- JSON mode accepted the schema instruction as the "json" the API requires. `reasoning_tokens` never appeared, as
  expected with thinking off.
- The compiler's 1,408-token system-and-instruction prefix was a cache hit from the second compile on.
- Deleting a finished shard and rerunning it made no call: the ledger stayed at 30. The rerun reproduced the
  row digest.
- What the models did is a tuning observation, not evidence. D compiled 6 of 8 and called `mo-schaffer-d1` and
  `integ-exp-1e-10` unsupported. E-LLM succeeded on 3 of 3 and C on 2 of 3.

No parameter changed after the dry run (`PROTOCOL["tuning_adjustments"]`, last entry). Its 14 cassettes and its
ledger are in [`benchmarks/results/e14-v3/dryrun/`](../../benchmarks/results/e14-v3/dryrun/); they hold the
prompts, the replies and no credential.

## 4. Estimate for the scored campaign

**Structure.** 29 scoring instances x 10 seeds = 290 cells for each of C, D and E-LLM, plus 58 formulation cases
(50 English, 8 Chinese) x 10 seeds = 580 compiles under arm D's caps. The structural maximum is 870 + 5,800 +
5,800 + 1,740 = **14,210 calls**.

**Method** ([`e14_llm_estimate.py`](../../benchmarks/e14_llm_estimate.py); its output is
[`estimate.json`](../../benchmarks/results/e14-v3/estimate.json); rerun it with
`PYTHONPATH=src python benchmarks/e14_llm_estimate.py --dry-run benchmarks/results/e14-v3/dryrun`). For every scoring instance and case it
builds the exact first request each arm sends and measures it in characters on the wire. For the planner it uses
every iteration's request, taken from a run of the runtime with a planner that always asks for another solve. The
dry run's cassettes then give tokens per character (compile 0.270, planner 0.336, proposer 0.433), reply length
(compile mean 272 tokens, planner 92, proposer 421) and cache-hit share (compile 0.74, planner 0.13, proposer
0.73). They also give calls per cell: 1 per compile, planner 2 to 3, proposer 2 to 7.

| scenario | assumption | calls | tokens in | tokens out | USD peak | USD off-peak |
|---|---|---|---|---|---|---|
| low | 1 call per compile, the fewest planner and proposer calls seen, measured cache share | 2,320 | 3.4 M | 0.6 M | **1.14** | 0.57 |
| expected | the dry run's means | 3,000 | 5.3 M | 0.9 M | **1.63** | 0.81 |
| high | as expected, but every C and E-LLM cell uses every call its cap allows, each stopped at 200,000 tokens | 12,190 | 72.6 M | 3.2 M | **11.88** | 5.94 |
| worst | every cap, every reply at `max_tokens`, every prompt token a cache miss, each cell stopped at 200,000 tokens | 9,860 | 90.5 M | 28.7 M | **61.53** | 30.77 |

Expected by part: D $0.14, the E-LLM compile $0.14, the E-LLM planner $0.27, C $0.80, formulation $0.28. C
dominates every scenario because it resends the whole conversation on each turn. The worst case has fewer calls
than the high one because 8,192-token replies make a C cell reach 200,000 tokens in about 5 calls.

**What limits the estimate.**

- The call counts of C and E-LLM come from 3 tuning cells each, all of dimension 2 to 5. The scoring set has 10-D
  and 20-D instances, where C will likely use more calls and longer replies. The expected figure probably
  understates C, and the high scenario bounds it at the cap.
- Repair rounds: all 11 dry-run compiles took 1 call. The ambiguous, unsupported and Chinese formulation cases may
  need repairs; the worst case charges every compile 3 calls.
- Cache hits are best effort on DeepSeek's side. The worst case assumes none.
- The tokens-per-character ratios rest on 11, 7 and 12 calls.
- Spend is at the peak price. A run entirely in off-peak hours costs half. Peak hours are 01:00 to 04:00 and 06:00
  to 10:00 UTC, Monday to Friday.
- The ledger over-states while calls are in flight: each open reservation counts at its worst case, about $0.01
  for a C call, so 4 processes add at most about $0.04.

**Recommended hard stop: `--max-llm-usd 15` and `--max-llm-calls 14210`.** $15 covers the high scenario ($11.88)
with 25 % margin, and stays a quarter of the worst case. If it binds, the run has gone well past every calibrated
scenario, and stopping to look is the right response. The call limit equals the structural maximum the per-cell
caps already enforce, so it binds only if something is wrong. Expected wall time is about 3,000 calls x 1.3 s,
roughly 1 h 05 min of model latency split over 4 processes, plus the solver time of v2 (8 min). The high scenario
is about 4.5 h of latency, about 70 min over 4 processes.

## 5. Phase B (after approval)

**Ship the code and the key.** From the laptop, with `<sha>` the freeze commit:

```bash
ssh j7zang 'mkdir -p ~/e14-10-campaign-v3/code-<sha> ~/e14-10-campaign-v3/logs'
git -C ~/Dev/orbitopt archive <sha> | ssh j7zang 'tar -x -C ~/e14-10-campaign-v3/code-<sha>'
echo '{"commit": "<sha>", "dirty": false}' | ssh j7zang 'cat > ~/e14-10-campaign-v3/code-<sha>/.e14_ship.json'
# the key: created mode 600 (umask 077), never on a command line or in a process list
ssh j7zang 'umask 077; cat > ~/.llm_env' < ~/Dev/orbitopt/.env
ssh j7zang 'stat -c "%a %n" ~/.llm_env'       # expect 600
```

This is `scp` to `~/.llm_env` with mode 600, without the moment `scp` leaves the file at the default mode. The
server reaches `api.deepseek.com` (checked 2026-11-28 from `ubuntu2404-002`: HTTP 401 without a key). Its
`~/e11env` has jsonschema 4.26, scipy 1.18.1 and Python 3.12.13.

**Launcher** `~/e14-10-campaign-v3/e14run.sh`, v2's launcher with the key loaded inside it:

```bash
#!/bin/bash
# usage: e14run.sh <code-dir> <out-dir> <label> <bench_e14_end_to_end.py args...>
set -u
CODE="$1"; OUT="$2"; LABEL="$3"; shift 3
ROOT="$HOME/e14-10-campaign-v3"
set -a; . "$HOME/.llm_env"; set +a                 # the key, never echoed
export LLM_KIND=deepseek LLM_MODEL=deepseek-flash LLM_BASE_URL=https://api.deepseek.com
export LLM_CAMPAIGN_MAX_CALLS=14210 LLM_CAMPAIGN_MAX_USD=15
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
cd "$ROOT/$CODE" || exit 1
{
  echo "start $(date -u +%FT%TZ) host $(hostname) code $CODE args $*"
  PYTHONPATH=src nice -n 10 "$HOME/e11env/bin/python" benchmarks/bench_e14_end_to_end.py "$@" --out "$ROOT/$OUT"
  echo "exit $? $(date -u +%FT%TZ)"
} >> "$ROOT/logs/$LABEL.log" 2>&1
```

The run prints the provider spec (kind, model, base URL and the key's variable name, never the key), the hard stop
and the ledger totals. `e14jobs.sh` is v2's with `v3` in its paths. It runs one family's shards in turn, every arm
in plan order (A, B, E-rule, C, D, E-LLM), then the family's faults, as `PROTOCOL["jobs"]` fixes.

**Run.** On the server, in `~/e14-10-campaign-v3`:

```bash
C=code-<sha>
./e14run.sh $C out plan plan                       # logs/plan.log: the digests must equal section 1
for g in "rosenbrock rastrigin" "ackley constrained_quadratic" "linear_program knapsack" "integration bi_objective"; do
  nohup ./e14jobs.sh $C out jobs-${g%% *} "--split scoring" $g > /dev/null 2>&1 &
  sleep 20
done
./e14run.sh $C out formulation run --shard formulation.scoring
```

Then `collect`, `verify`, `analyze` and `report` as in [v2 section 12](e14_qualification_report_v2.md#12-reproduction).
`--split scoring` leaves out the tuning shards, which would otherwise call the model for 120 more cells.

**If the hard stop or an account error fires** (exit 3, `STOPPED` in the log): read `out/llm_ledger.jsonl` and
the log, fix the cause or raise the limit, and rerun the same command. Finished shards are skipped, and every call
a cassette holds is served from it. Exit 3 on HTTP 402 means the balance ran out.

**Copy back and clean up.** `rsync` `out/` without `runs/` as in v2. Then a tar of every
`runs/**/*.cassette.jsonl`, the reproduction artifacts (inferred size 50 to 400 MB), with its digest in the
manifest. Then remove the key: `ssh j7zang 'shred -u ~/.llm_env'`.

## 6. Not verified in Phase A

- No scored shard ran. Scoring-instance prompts, 10-D and 20-D proposer conversations, and Chinese formulation
  cases have not been sent to DeepSeek.
- Behaviour under 429, 5xx, blank JSON-mode content and `insufficient_system_resource` is tested only against the
  local look-alike server ([`tests/fake_deepseek.py`](../../tests/fake_deepseek.py)). The dry run met none of them.
- Whether DeepSeek bills a failed attempt. The ledger charges a failed call nothing beyond its released
  reservation.
- The analysis reports `ir_schema_validity` from the formulation suite only. The spec's denominator also counts
  the 290 arm-D compiles; their rows carry `schema_valid` for the analysis to add (the analysis is outside the code
  digest).
- Arms D and E-LLM map the compiled x to the reference by declared variable order. A model that declares the
  variables in another order is scored on a permuted point. That is v1's and v2's rule, kept.
