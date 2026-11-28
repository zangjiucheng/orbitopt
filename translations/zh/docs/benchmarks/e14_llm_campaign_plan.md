# E14-10 LLM campaign plan: campaign v3

Campaign v 3运行[E14-10 benchmark specification](e14_benchmark_spec.md)对DeepSeek的预先注册的提供者武器:臂C(模型提出点数,无解题),臂D(LLM编译器加固定解题)和臂E为`E-LLM`(运行时间LLM编译器加LLM规划器),臂D的`formulation_accuracy`和编译器的`ir_schema_validity`后面的配方套装. 规则臂A,B和E-规则再次像[campaign v2](e14_qualification_report_v2.md)一样运行,因为代码摘要移动了.

**Status: Phase A and Phase B done.** Phase A prepared and froze the campaign and ran a 30-call tuning dry run.
Phase B ran on 2026-11-28 under a $5 hard stop in three stages and cost $3.33 at list price; its results are in the
[campaign v3 report](e14_qualification_report_v3.md).

## 1. 冻什么?

| 项目 | 价值 |
|---|---|
| 协议 | `e14-10-campaign/3`, `sha256:ad381579aafcf84a55ac525052406cbff5005e22a55bd14ebc493dab8b72518a` |
| spec (自 v 1 起不变) | `sha256:e5104c340d419f388643bed2ad1de32662f8384f7b1c4adb7a05b90ad877db47` |
| 代码 | `sha256:9f4ed53b1c00380a372f1aa93854bada820c2929d12f0a99f5e2dc271ddd42f1` |
| 冻结文件 | [`benchmarks/results/e14-v3/freeze.json`](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/results/e14-v3/freeze.json),内容`sha256:7caa7d0255edc82202b4d7e6e22a378e5ce30ad269c97e71fd961b597d3699fd` |
| 提供者 | `deepseek`, model `deepseek-flash`, `https://api.deepseek.com/chat/completions`; a scored provider shard refuses any other kind, model or base URL |
| 取样 | 温度 0, 思维已禁用, 没有发送种子( DeepSeek 文档无) |
| 结构化产出 | `response_format` `json_object`, the JSON Schema in the system instruction, validated locally with one repair turn |
| 每个细胞盖 | 电话: C 20, D 3, E-LLM 20 (编译和计划员一起) ; 200,000令牌; 每次通话前检查 |
| 价格表 | `deepseek-flash` per million tokens: cache hit $0.006, cache miss $0.30, output $1.20 (peak list price; off-peak is half), from the [DeepSeek pricing page](https://api-docs.deepseek.com/quick_start/pricing), read 2026-11-28; the page shows no date |

The v1 and v2 freezes and results are unchanged. `PROTOCOL["changes_from_v2"]` in
[`e14_common.py`](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/e14_common.py) lists what moved; every rule-arm choice is v2's.

## 2. A阶段的变化

**One provider configuration.** The campaign, the CLI (`orbitopt agent ... --provider env`) and Mission Studio
read `LLM_KIND`, `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_TIMEOUT_S` through `orbitopt.llm_env`. The
campaign-only and Studio-only names of v1 and v2 are gone. The key comes only from the environment.
`--env-file FILE` reads `KEY=VALUE` lines without expanding anything, never overrides a variable that is set, and never prints a value
([providers](https://github.com/zangjiucheng/orbitopt/blob/main/docs/agent/providers.md#configuration-from-the-environment)).

**DeepSeek.** `DeepSeekProvider` (`LLM_KIND=deepseek`) follows DeepSeek's documentation, read 2026-11-28
([providers](https://github.com/zangjiucheng/orbitopt/blob/main/docs/agent/providers.md#deepseek)). Every response records the model the server named, its usage with
cache hits and misses, its latency and its HTTP attempts. 429 and 5xx are retried at most 3 times, honouring
`Retry-After` up to 60 s. 401 and 402 stop the campaign.

**The provider arms.** Each cell talks to the model through its cassette, its meter and the campaign ledger
([`e14_llm.py`](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/e14_llm.py)):

- 磁带记录`--runs`的每次通话:请求、答复、模型返回、使用和
  长眠. 重播时每打一次磁带, 中断的shard从不支付两次。
- 计数器在每个调用前执行Spec每个细胞的调用上限和符号预算. 结构化产出维修,
  编译器修复弹和规划器重排全部计数. 一个C股的电话通过 象征性的预算是
  失败。
- `<out>/llm_ledger.jsonl`是整个竞选的硬站。 每一个过程都写一个
  输出目录, 在独锁下 。 在每个网络称它之前 都保留着最坏的情况:一个标志
  按缓存差价,每请求的两个字符,加上按输出价的`max_tokens`。 一个叫
  会通过`--max-llm-calls`或`--max-llm-usd`被拒绝. 硬的就留下了未写和重播
  恢复它。
- 每个单元格都用`ReplayProvider`的磁带重播,没有模型的呼号,行记录是否
  重播重现. 这是武器供应商的`experiment_replay`度量衡。 DeepSeek 的输出会有所不同
  between identical calls even at temperature 0, so the cassette is the reproduction artifact, not a re-run.

**A scorer fix found by the dry-run tests.** The formulation suite matched compiled variables to the reference by
name only, so a correct compile that names its own variables (`food_1` for `x0`) failed. It now uses the case's
alias table when that names compiled variables, else the declared order, which is how arm D maps x back.

## 3. Tuning干线运行(A阶段,真API).

Run on 2026-11-28 from the laptop, on the v3 code before the protocol was final, under a 40-call hard stop. Only
tuning instances and tuning seed 1000 were used. The prompts hold benchmark problem text only.

| 手臂 | 单元格 | 电话 | 符号在 | 其中缓存点击数 | 发号施令 | USD(列表) | 重播 EQUAL |
|---|---|---|---|---|---|---|---|
| D级 | 8(每个调音实例) | 8 | 14,495 | 9,856 | 2,103 | 0.0040 | 8/8 |
| E-LLM | 3 (`cqp-worked-d2`, `lp-diet-d2`, `rastrigin-d2`) | 10 | 12,525 | 5,760 | 1,539 | 0.0039 | 3/3 |
| C级 | 3 (`cqp-worked-d2`, `knapsack-p02-n5`, `rastrigin-d2`) | 12 | 47,707 | 34,688 | 5,053 | 0.0102 | 3/3 |
| **total** | 14 | **30** | 74,727 | 50,304 | 8,695 | **0.0181** | 14/14 |

- 每个反应都叫`deepseek-flash`. 没有重试,没有HTTP错误,没有空白的JSON-模式回复,没有修复回合.
  中间空档期为1.3,最大空档期为2.2.
- JSON模式接受计划指令为API要求的"json". `reasoning_tokens`从未出现,因为
  期待与思考关闭。
- 编译器的 1,408 -token 系统与指令前缀是第二编译器上的一个缓存命中.
- 删除一个完成的硬盘并重新运行它没有打电话:分类账留在30. 重播时复制了
  排行文摘.
- What the models did is a tuning observation, not evidence. D compiled 6 of 8 and called `mo-schaffer-d1` and
  `integ-exp-1e-10` unsupported. E-LLM succeeded on 3 of 3 and C on 2 of 3.

干运行后没有改变参数(`PROTOCOL["tuning_adjustments"]`,最后一个条目). 它的14磁带和分类账在[`benchmarks/results/e14-v3/dryrun/`](https://github.com/zangjiucheng/orbitopt/tree/main/benchmarks/results/e14-v3/dryrun/);它们持有提示,答复,没有证书.

## 4. 得分战绩估计

**结构.** 29 评分例×10 种子=290 各个C,D和E-LLM的细胞,加上58 配体例(50英语,8中文)×10 种子=580 在臂D的帽下编译. 结构上限为870 + 5,800 + 5,800 + 1,740 = **14,210电话** .

**Method** ([`e14_llm_estimate.py`](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/e14_llm_estimate.py); its output is
[`estimate.json`](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/results/e14-v3/estimate.json); rerun it with
`PYTHONPATH=src python benchmarks/e14_llm_estimate.py --dry-run benchmarks/results/e14-v3/dryrun`). For every scoring instance and case it
builds the exact first request each arm sends and measures it in characters on the wire. For the planner it uses
every iteration's request, taken from a run of the runtime with a planner that always asks for another solve. The
dry run's cassettes then give tokens per character (compile 0.270, planner 0.336, proposer 0.433), reply length
(compile mean 272 tokens, planner 92, proposer 421) and cache-hit share (compile 0.74, planner 0.13, proposer
0.73). They also give calls per cell: 1 per compile, planner 2 to 3, proposer 2 to 7.

| 假设 | 假设 | 电话 | 符号在 | 发号施令 | USD峰值 | USD 离峰 |
|---|---|---|---|---|---|---|
| 低级 | 1 调用每个编译, 最少的规划和提议者调用, | 2,320 | 3.4 M | 0.6 M | **1.14** | 0.57 |
| 预期 | 干燥意味着 | 3,000 | 5.3 M | 0.9 M | **1.63** | 0.81 |
| 高级 | 但每个C和E-LLM的细胞 都使用其顶部允许的呼号 每一个都停在200,000的标志上 | 12,190 | 72.6 M | 3.2 M | **11.88** | 5.94 |
| 差一点 | 每一个封顶,`max_tokens`的每个回覆,每个提示,一个缓存丢失,每个单元格停止在200,000指示, | 9,860 | 90.5 M | 28.7 M | **61.53** | 30.77 |

预期部分为:D $0.14,E-LLM 汇编 $0.14,E-LLM 规划者 $0.27,C $0.80,配音:0.28. C主导了每个场景,因为它在每个回合重播整个对话. 最糟糕的情况是电话比高的要少,因为8,192-token的回复使一个C细胞在大约5的通话中到达200,000的令牌.

**What limits the estimate.**

- C和E-LLM的调用计数来自3调制单元格,每个调制单元格都是2到5. 得分集有10-D
  和 20-D 实例,其中 C 可能使用更多呼叫和更长的答复。 预期数字可能
  低调的C, 和高的情景 约束它在顶端。
- 修复回合:11干线编译全部使用1调用. 模棱两可、没有支持和中国表述的案例可能
  需要修理;每一次编译的3电话,最差的收费.
- 缓存命中是DeepSeek方面的最佳努力. 最坏的情况是没有。
- 11,7和12的调用是每个特征的代号比率.
- 花费在最高价格。 跑步完全在高峰时间 花费一半。 高峰时数为01:00到04:00和06:00.
  to 10:00 UTC, Monday to Friday.
- 电话通话时的分类账超标:每一次开放的预订都算在最坏的情况下,大约为0.01美元.
  4进程最多增加约$0.04。

**Recommended hard stop: `--max-llm-usd 15` and `--max-llm-calls 14210`.** $15 covers the high scenario ($11.88)
with 25 % margin, and stays a quarter of the worst case. If it binds, the run has gone well past every calibrated
scenario, and stopping to look is the right response. The call limit equals the structural maximum the per-cell
caps already enforce, so it binds only if something is wrong. Expected wall time is about 3,000 calls x 1.3 s,
roughly 1 h 05 min of model latency split over 4 processes, plus the solver time of v2 (8 min). The high scenario
is about 4.5 h of latency, about 70 min over 4 processes.

## 5 B阶段(核准后)

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

运行会打印提供者的 spec(kind, 模型, Base URL)和键的可变名称,永不键),硬站和分类账总计. `e14jobs.sh`是2's与`v3`在它的路径. 它依次运行一个家族的硬盘,每个臂部按计划顺序排列(A,B,E-规则,C,D,E-LLM),然后作为`PROTOCOL["jobs"]`修正的断层.

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

然后`collect`,`verify`,`analyze`和`report`,如[v2 section 12](https://github.com/zangjiucheng/orbitopt/blob/main/docs/benchmarks/e14_qualification_report_v2.md#12-reproduction). `--split scoring`将调音软体排除在外,否则会将模型称为120更多细胞.

**If the hard stop or an account error fires** (exit 3, `STOPPED` in the log): read `out/llm_ledger.jsonl` and
the log, fix the cause or raise the limit, and rerun the same command. Finished shards are skipped, and every call
a cassette holds is served from it. Exit 3 on HTTP 402 means the balance ran out.

**Copy back and clean up.** `rsync` `out/` without `runs/` as in v2. Then a tar of every
`runs/**/*.cassette.jsonl`, the reproduction artifacts (inferred size 50 to 400 MB), with its digest in the
manifest. Then remove the key: `ssh j7zang 'shred -u ~/.llm_env'`.

## 6. A阶段未核实

- 没有得分的Shard跑。 10-D和20-D求婚者对话,以及中文配方
  案件尚未发送 DeepSeek。
- 429,5 xxx,空白的JSON-模式内容和`insufficient_system_resource`下的行为只针对:
  本地外观类服务器( [`tests/fake_deepseek.py`](https://github.com/zangjiucheng/orbitopt/blob/main/tests/fake_deepseek.py)) 。 干燥没有遇到他们。
- DeepSeek 是否尝试失败 。 分类账上充斥了一次失败的呼叫 仅此而已
  保留。
- 分析报告`ir_schema_validity`只来自配方套件. 光谱分母也算数
  290臂-D编译;他们的行携带`schema_valid`以供分析添加(分析在代码之外).
  文摘).
- 武器D和E-LLM按已宣布的可变顺序将汇编的x与参考物映射. 一个模型宣布
  变量,按另一个顺序,在固定点上得分。 这是1's和2's规则,保留.
