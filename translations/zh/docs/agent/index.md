# LLM- 协助优化

`orbitopt.agent`将问题说明变成[optimization IR](../optimization/ir.md)问题,用一个有界限的循环解决,每个考生的[verification guard](../optimization/verification.md)成绩,并省下一个没有LLM呼叫的重播跑. 一个语言模型可能写出问题并选择解析器. 它从来不给一个答案:只有警卫从证据中得出答案。

| 页次 | |
|---|---|
| [Providers](providers.md) | 一个与OpenAI兼容的接口,本地化,模拟和重播模型;磁带. |
| [Compiler](compiler.md) | 声明,确认IR问题或声明拒绝退出. |
| [Tools](tools.md) | IR `ToolCall`的科研工具注册,可以到达,登录和重播. |
| [Runtime](runtime.md) | 限定循环:限制,政策,规划,诊断,运行目录和重播. |
| [Optimize workspace](studio.md) | Mission Control的页面:单词在,IR作为可编辑的数学,一个被监视的运行,四个状态,重播. |
| [Qualification results](../benchmarks/e14_qualification_report_v2.md) | 决定性解析器和运行时间比较,包括验证,预算执行和重播. |
| [Provider campaign plan](../benchmarks/e14_llm_campaign_plan.md) | E14-10 v 3干线运行和对得分模型比较的冷冻设计. |

The recorded v2 campaign used the rule-based planner. The v3 provider
campaign has a tuning dry run and no scored provider results in the archive.

## 安装

```bash
pip install -e '.[agent]'
```

提供者层只使用标准库和`jsonschema`,因此没有安装LLM SDK. `[agent]`是`[solvers]`(NumPy,scipy,`cma`),运行时间和内置工具需要这些. 该基地仍然安装进口和运行其他所有设备,`orbitopt --help`没有装载这些设备.

## 五个命令

下面的会话像测试一样使用带有录制编译器回复的模拟提供者. 将 `--provider mock --mock-replies replies.json`替换为真正的提供者(下一节),以使用 模型 。

```text
$ cat prompt.txt
A plant makes two products with profit 3 and 5 per batch. Plant 1 has 4 hours
(product 1 uses 1 hour), plant 2 has 12 hours (product 2 uses 2 hours), plant 3
has 18 hours (product 1 uses 3, product 2 uses 2). Batches are continuous and
nonnegative. Maximize profit.

$ orbitopt agent compile --prompt-file prompt.txt -o problem.json \
      --provider mock --mock-replies replies.json --cassette compile.jsonl
COMPILE after 1 round(s), 624 tokens
  problem      problem.json  (338cb4c5222376bcc428220cb56e7725d7b0249f7f3658e8e5f02440d3b4a92a)
  report       problem.compile.json

$ orbitopt optimization validate problem.json
VALID nl-74a320afd4c5 (IR 1.1)
  digest       338cb4c5222376bcc428220cb56e7725d7b0249f7f3658e8e5f02440d3b4a92a
  structure    lp (linear): 2 variable(s), 1 objective(s), 3 constraint(s)
  solvers      scipy.highs, scipy.slsqp

$ orbitopt agent solve problem.json --max-iterations 20 --max-evaluations 4000 \
      -o runs/experiment-001
VERIFIED (planner_stop) after 2 iteration(s), 14 evaluations, 0 tokens
  best         scipy.highs: {'profit': 36.0} at {'product_1': 2.0, 'product_2': 6.0}
  solver_converged         passed
  constraints_satisfied    passed
  independently_verified   passed
  optimality_certified     not_assessed
  run          runs/experiment-001
  manifest     sha256:7534331142fbac893c49fa69452193b59bc9c4b6ffe8044157876efa04b01dd0  (keep it; ...)

$ orbitopt optimization verify runs/experiment-001
VERIFIED runs/experiment-001 (agent run, recorded status verified)
  verdict      passed  (required: solver_converged, constraints_satisfied, independently_verified)
  solver_converged         passed        the solver reports termination 'converged'
  constraints_satisfied    passed        every domain and hard constraint holds on re-evaluation
  independently_verified   passed        independent evidence agrees with the re-evaluated result
  optimality_certified     not_assessed  not requested by the policy, claimed or evidenced

$ orbitopt agent replay runs/experiment-001
EQUAL runs/experiment-001 (sealed verdict: passed)
  trace        recorded 7bdc6c28..., replayed 7bdc6c28...
```

| 命令 | 对 | 0 当 |
|---|---|---|
| `agent compile --prompt-file P -o problem.json` | Runs the [compiler](compiler.md). Writes the problem only when the compile is accepted, and always writes the report (`problem.compile.json`, or `--report`) with the assumptions, missing parameters, ambiguities, other questions, refusal reason, unauthorised changes or errors, and the tokens spent. `--allow-draft` writes the model's raw draft to `-o` on a refusal, marked `"draft": true`; it is not IR and `validate` rejects it. | 已接受编译 |
| `optimization validate problem.json` | 打印每个错误,或者摘要,结构类和安装的兼容解析器. | 问题是合法的 |
| `agent solve problem.json --max-iterations N --max-evaluations N -o DIR` | 运行 [runtime](runtime.md) 循环并保存运行目录. `--max-seconds`,`--max-tokens`和`--seed`是可选的;`--planner rule`(默认)不问模型,`--planner llm`问`--provider`. 拒绝已经持有运行的目录 。 | 答案已经验证 |
| `optimization verify DIR` | 检查每个字节和跟踪,然后与守卫和最终验证的替代解析器重新划分封存的最佳答案. 临ΤΩ bare | 又被验证了 |
| `agent replay DIR` | 重新执行记录的建议书,没有规划人员或模型,并比较痕迹. | 没有区别 |

exit 1表示命令运行,答案为否;exit 2表示请求本身是错误的(缺少文件,使用运行目录,无法到达或没有密钥的提供者). 每个命令都使用 `--json`. `verify` 和 `replay` 将 `--expect-digest` 与显示的文摘 `solve` 一起打印: 使该文摘不带乐队,因为谁可以重写运行的目录,也可以重写它的表.

## 命令行上的提供者

`compile`和`solve --planner llm`采用相同的提供者选项.

| `--provider` | 需求 | Notes |
|---|---|---|
| `env` | `LLM_*`变量 | 提供者环境配置 ([providers](https://github.com/zangjiucheng/orbitopt/blob/main/docs/agent/providers.md#configuration-from-the-environment)); `orbitopt --env-file .env agent ...`从文件中加载它们 。 |
| `openai-compatible` | `--base-url`, `--model` | 从环境变量 `--api-key-env` 名称(默认 `LLM_API_KEY` ) 读取 API 键,在调用时使用. |
| `local` | `--model` | 默认的 Ollama; `--base-url` for lama.cpp 或 vLLM. 没有钥匙。 |
| `mock` | `--mock-replies FILE` | JSON回复列表,每个字符串或JSON对象作为JSON文本发送. |
| `replay` | `--cassette FILE` | 播放录制的磁带; 没有录制的任何请求都是错误 。 |

No option takes a key, so a key never lands in shell history, a cassette, a
report or the output. `--cassette FILE` with any other provider records every
model call there, and `--provider replay --cassette FILE` serves the same
calls back later:

```bash
export LLM_API_KEY=...           # set in the environment, never on the command line
orbitopt agent compile --prompt-file prompt.txt -o problem.json \
    --provider openai-compatible --base-url https://api.openai.com/v1 --model gpt-4o-mini \
    --cassette compile.jsonl
orbitopt agent compile --prompt-file prompt.txt -o problem.json \
    --provider replay --cassette compile.jsonl          # same outcome, no model call
```

`agent replay` needs no cassette at all: the trace holds every proposal, so
an LLM-planned run re-executes without the model.

## 从Python

The package root re-exports each part's entry points lazily, so
`import orbitopt.agent` loads nothing:

```python
from orbitopt.agent import LocalProvider, RuleBasedPlanner, compile_problem, replay, save_run, solve

outcome = compile_problem(statement, LocalProvider("qwen2.5:7b"))
run = solve(outcome.problem, planner=RuleBasedPlanner(), seed=0)
digest = save_run(run, "runs/experiment-001")
replay("runs/experiment-001", expected_manifest_digest=digest).equal
```

两部分定义的名称(`ToolSpec`,`ToolCall`,`Violation`,工具日志的`replay`)没有再出口;从它们的部分导入. `orbitopt.agent.commands`持有每个命令的计算,作为带有路径和摘要的函数.
