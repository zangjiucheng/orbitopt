# 代理复制运行时间

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

`solve(problem, *, planner, budget=None, policy=None, tools=None, seed=0, cancel=None, registry=None)`从不提出坏建议或失败解决者. `refused`不能忠实地回答问题。

| 参数 | 含义 |
|---|---|
| `budget` | 一个IR `Budget`,逐字覆盖`problem.budget`字段. |
| `policy` | 计划者可能改变什么 见[The policy](https://github.com/zangjiucheng/orbitopt/blob/main/docs/agent/runtime.md#the-policy). |
| `tools` | `ToolView`或用工具id绘制解析器的映射. |
| `cancel` | A `threading.Event` or a callable, checked before each planner proposal and before the final verification. A `CancelPoint` cancels at exactly that point. |
| `registry` | 解脱者. 默认为 `default_registry()` 。 |

验证政策始终是`problem.verification`. 没有争论可以放松它。

## 限制

Every limit is finite, so a run can never retry without bound. `Limits.resolve`
takes each limit from `budget`, else from `problem.budget`, else the E14-10
default.

| 限制 | 默认 | 关键是 |
|---|---|---|
| `max_iterations` | 20 | 计划提案,接受或拒绝. |
| `max_evaluations` | 2000×最大(n,2) | 每为跑步而作的问题评价:解者,守者及候补解者. |
| `max_wall_s` | 600 s | `solve`开头的墙钟. |
| `max_tokens` | 200,000 | LLM符,即时完成. |

`max_iterations`限制循环,这是`Budget.max_iterations`唯一的含义. 运行时间不会传递给解决者. 一个决定设定了后端自身的迭代封顶,作为建议字段`max_iterations`,运行时间将它交给解析器作为`solve(..., max_backend_iterations=...)`.

每一次评估都要经过一次`EvaluationMeter`. 计数器是问题的副本,`evaluate`每拨一次,所以解码器和警卫用它作为问题本身. 它通过提高`EvaluationBudgetExceeded`(不是`EvaluationError`)来拒绝超过当前解决方案的允许性的评价,因此一个无视其`Budget`的解决方案停止而不是跳过点. 适配器自身缓存一个点的目标和限制是被记录的缓存:缓存命中不值钱,任何其他重复的点再次计数.

`max_evaluations`(`VERIFICATION_SHARE`)的四分之一被挡住了最终的验证. 一旦决定花费少于8的评价,循环就以`max_evaluations`结束. 每次计划员打电话前都要检查托肯斯,所以最后的呼叫可以以自己的大小超越符号限制. 运行时间上限在剩余符号上调用回复,但它无法提前知道提示大小. 壁钟在下一次评价时会停止一个解题,以及迭代之间的循环. 跑出墙上的时钟 跳过最后的验证

## 政策

`Policy`列出计划者可能改变什么. 上面没有问题

| 外地 | 默认 | 含义 |
|---|---|---|
| `allowed_solvers` | `None` | solver ids 一个决定可能命名; `None`允许每个可用的,兼容的解析器. |
| `allow_options` | `True` | 决定是否设置解析器选项 。 |
| `allow_x0` | `True` | 决定是否设定起点 。 |
| `on_violation` | `"continue"` | `"stop"`在第一次被拒绝的提案中结束运行. |
| `max_violations` | 3 | 跑步在这么多被拒绝的提案之后结束. |

## 提案和决定

一个计划员返回一个 `Proposal` : 原始的决定对象, 它的成本, 或者它没有的原因 。 每条建议都是不可信的,包括了按章办事的规划者. `validate_proposal`将其变为`Decision`或列为被拒绝的理由清单. 被拒绝的提案成为`rejected`记录,从未适用。 `Decision`的任何字段都无法进行问题更改,所以滑过验证的更改无处可去.

提案可能只列出这些字段:`action`(`solve`或`stop`),`solver_id`,`options`,`x0`,`seed`,`evaluations`,`max_iterations`和`rationale`.

| 违反 | 何时 |
|---|---|
| `problem_edit` | 字段或选项名称 部分问题,预算或验证政策:`objective`,`constraints`,`bounds`,`variables`,`parameters`,`budget`,`max_evaluations`,`verification`,`constraint_tol`和其余`PROBLEM_FIELDS`. |
| `unknown_field` | 列表之外的任何其他字段. |
| `malformed` | 不是一个对象,一个坏的`action`,一个非整数种子,或者一个LLM的回复两次使计划失败. |
| `unknown_solver` | 没有注册的解析器有此 ID 。 |
| `solver_not_allowed` | 解析器虽有注册但无法获取,与问题不符,被运行排除,或政策不允许. |
| `invalid_option` | 解析器不声明选项,该值在选项的`OptionSpec`之外,否则策略不允许选项. |
| `x0_unknown_variable`, `x0_out_of_domain` | 一个起始点命名一个未知变量或位于变量域外(捆绑,集成,选择). |
| `budget_overreach` | `evaluations`超出一个决定可能花费多少,或者100,000以上后置器`max_iterations`(对于一个没有`iteration_cap`的解题者,改称为`invalid_option`). |

每个解析器在 `SolverCapability.options` 中声明其选项,每个名称有一个`OptionSpec`,一个类型,一个封闭范围或选择,以及一个`tolerance`旗(参见 [solvers](../optimization/solvers.md) ). 运行时间与这些规格相对应,没有自己的表格。 解析器的容忍度(`ftol`,`gtol`,`tol`,`tolfun`,`tolx`,`mip_rel_gap`)是解析的唯一值.

## 一个迭代

1. 请检查date=中的日期值 (帮助) 取消(`before_iteration`)和每个限制.
2. 向计划者请示并加其符.
3. 验证它。 一个被拒绝的建议被记录下来,循环继续,除非政策阻止.
4. `stop`决定结束循环.
5. 用决定的允许性、种子、选项和起始点运行解析器。 升起的解决者是失败的迭代.
6. 检查结果:`verify_result(problem, result, solved_problem=problem)`没有独立证据。
7. 诊断结果. 一个被看守抓到说谎或坠毁的解密者 其余时间都不算

## 验证

屏幕分别阅读四个状态,从不相信结果本身的布尔. 一个候选人有资格当警卫的`constraints_satisfied`通过,报告中没有任何内容显示解题者撒谎. `distrust`命名了一个谎言:`DISHONEST`的发现(一个不匹配的目标或剩余,另一个问题的文摘,一个 stale 缓存,一个能保证自己的结果),一个声称的警卫失败,或者对`independently_verified`或`optimality_certified`的任何主张,只有警卫设置. 警卫不能评估的说法,例如提出的工具,不是谎言。

最佳的合格候选人首先是干净的(没有限制使用超过其容忍波段的`SELECTION_FRACTION`),然后有其意义上的最佳目标,然后是经过的`solver_converged`,然后是最早的重复. 更深层的乐队中的候选人可以通过其违规买到的东西来击败准确的优化,因此在客观上从未比干净的好过. 解析器适配器在报告前恢复其最佳的波段点,所以这里的一个波段候选人是无法恢复的([solvers](https://github.com/zangjiucheng/orbitopt/blob/main/docs/optimization/solvers.md#the-tolerance-band-is-not-room-to-optimise)). 循环后,运行时间与整个守卫核实:

- 最多两个替代解析器,每个兼容解析器除了候选人的
  自己人和被排除在外的人, 以`AlternateSolverCheck` 证据封存
  到`result_digest(result)`. 守卫自己管理着他们,每个都有平等的
  所剩评价份额。 策略的解析器列表不适用于
  他们,因为选择验证不是规划者的工作.
- 当一名候补选手发现一个更可行的点时,会反驳该候选人.
  它的要点经过再次筛选和核实(`MAX_VERIFICATION_ATTEMPTS = 2` ).
- No alternates run when the policy does not require `independently_verified`
  或者不允许`alternate_solver`证据.

On a finite domain (`ProblemStructure.domain_size`), enumeration joins the
final verification:

- 详尽的解析器 (`orbitopt.enumeration`) 是当
  剩下的评价涵盖域。 它的分配正是域的
  大小。 这是准确的,所以它单独运行,没有增加第二个候补。 A级
  域名中不匹配的域名除外,其他替代域以
  从前
- 关于一个单一目标问题,其领域仍然适合:
  替补,跑步时间加一个`EnumerationCertificate`,和警卫
  列出域本身。 这可以通过`optimality_certified`.
  候补者先到:证书只收他们留下的东西。
- 两人通过计时器对`max_evaluations`进行计数. N的一个域
  有关2 N评价的点数成本以核实和认证. 一个被反驳的人
  候选人的第二次尝试需要更多的认证。

`capability.convergence`决定何为趋同. 差异式演变的趋同是`heuristic_stop`,从来不是最佳的. 一阶点只在`classify`证明对流的类(`lp`,`convex_qp`)上才最终确定. `certificate`的汇合在`lp`,`convex_qp`和`milp`,`exhaustive`的汇合在任何等级上都是最终的.

| 状态 | 何时 |
|---|---|
| `verified` | 最终警卫报告通过. |
| `unverified_best` | 一个可行的最佳方法存在,但警卫没有通过. |
| `no_feasible` | 无合格候选人。 `result`是诚实的结果, 最少的违反,如果有的话。 |
| `budget_exhausted` | The best is not verified, and a limit ended the loop, cut or skipped part of the final verification, or was used up by the run (`outcome.budget` says where). |
| `cancelled` | 取消了检查 最后的验证跳过了. 追踪是`outcome.cancel`说在哪里 |
| `policy_violation` | 政策以一个被拒绝的建议书结束了运行. 最后的验证跳过了. |
| `refused` | 无效的 IR, `known_tool_ids(tools)`之外的工具 id, 绑定工具失败其单位检查, 或没有可以接受问题的解析器 。 没有评估。 |

`outcome.report`是`outcome.result`的终极验证,或者它的屏幕在无最终验证运行时. `outcome.statuses`给出了四个状态.

### 限制会缩短时间

`outcome.budget`(追踪的`outcome.budget`)列出了每个地方的极限切短,无论状态如何. `BudgetEvent`各命名舞台,资源(`iterations`,`evaluations`,`tokens`,`wall_s`),舞台是否耗尽(`stopped`)或不运行(`skipped`),舞台结果的标签,以及它拥有的津贴.

| 阶段 | 何时 |
|---|---|
| `loop` | 规划员的环路以一个极限结束. 止因缘名. |
| `solve` | A planner's solve stopped at its allowance: termination `max_evaluations` or `timeout`. |
| `alternate` | 最终-验证在分配时停止,否则剩下的评价不能给任何候补`MIN_ALTERNATE_EVALUATIONS`(`skipped`). |
| `certificate` | 该政策要求`optimality_certified`,而列举不符合候补者留下的(`skipped`). |
| `verification` | 验证尝试被跳过:剩下的评价没有覆盖警卫的重新评价. |
| `run` | 该运行使用了其所有迭代,评价或符号. |

A cut never decides `verified` on its own. The guard grades the evidence the
cut stage did produce: an alternate that stopped at its allocation and still
agrees with the candidate is independent evidence, and the run is `verified`
with the cut listed. A run that is not verified after a cut of the loop, the
final verification or the run is `budget_exhausted`, not
`unverified_best`. A `solve` cut alone leaves the status to the final
verification, since the planner chose that allowance and went on.

第一次E14-10运动显示为何需要该名单:四个`rosenbrock-disk-d2`细胞使用4000评价的4000,以停止理由`planner_stop`结束`verified`,因为差别进化的替补使用上次评价并停止分配. 追踪到的只是替补的结果 运行中现在有`alternate`和`run`事件.

`loop`,`solve`,`alternate`和`run`事件跟随着微量自己的记录(`derive_budget`),所以一个老微量在升级时得到它们. A `skipped`事件没有留下其他记录,运行时间在衍生事件之后添加. 运行级的墙钟事件并不产生,因为`usage.wall_s`被排除在文摘之外;一个墙钟站显示为`timeout`终止或`max_wall_s`停止原因. `orbitopt agent solve`每件事件打印一款`budget`线,工作室运行面板将它们列在状态下,实验记录保存它们.

## 诊断和补救办法

`diagnose(result, report, capability, structure)`命名为`FailureKind`. `REMEDIES`每类图示有界限的补救方法,最有希望的第一. 没有补救办法触及问题。

| 失败类型 | 何时 | 补救措施 |
|---|---|---|
| `verified_local` | 证书,详尽列举,或证明对流问题的第一顺序点. | 停下来 |
| `local_optimum` | 一个无法证明对流的问题的第一顺序点. | 重启全局, 重新启动扰动, 停止 |
| `heuristic_stop` | 一个休眠症凝聚。 | 精炼本地, 重启全球 |
| `non_convergence` | 后端的迭代极限. | 本地精细化, 放松解析器容忍度, 切换解析器 |
| `stalled` | 搜索失败或停滞 。 | 放松解析器容忍度, 重新启动扰动, 切换解析器 |
| `infeasible` | 没有找到可行的点,也没有证据证明不可行。 | 切换解析器, 更改初始猜测, 重新启动全局 |
| `unbounded` | HIGHS证明了目标不受约束. | 停下来 |
| `evaluation_error` | 每次评价都失败了. | 更改初始猜测, 切换解析器 |
| `solver_error` | 解决者提出,或报告错误. | 切换解析器 |
| `budget` | 解决方案没有评估或时间。 | 本地精炼,停 |
| `rejected_by_guard` | `distrust`发现一个谎言. | 切换解析器 |

从选项的值或它的 `OptionSpec` 默认值中, 以 100 乘以每个解析器的容忍度, 并在 spec 的上界停止 。 以规则为基础的规划人员每次最多放宽两次。

## 规划员

`RuleBasedPlanner`是决定性的,没有使用符号. 它的排名是登记单位的排名,最专业的排名是`lp`,`convex_qp`和`milp`. 在所有其他类上,它都把一个全局解析器放在首位,因为没有一个语法测试可以排除几个迷你. 全球解决方案之后,从点上进行局部改进。 至`max_restarts`(3)再进行一轮,每轮使用新种子,首轮重启以不超过`objective_rtol`的比分不领先最佳. 一个失败的解决方案需要其诊断的第一补救措施,即规划者仍然可以进行. 一个人口解析器得到的人口覆盖了它分配的大约150几代人,5到15每个变量的成员.

`LLMPlanner(provider)`通过`generate_structured`为同一 JSON对象询问 模型. 迅速掌握问题的类别和领域,解决者及其选项范围,其余的迭代,评价及信物,现任者以及早期每次迭代的摘要. 它没有墙钟值,所以一个`ReplayProvider`磁带重播. `PROPOSAL_SCHEMA`故意留下额外字段,这样要求放松约束的回复就达到了运行时间的验证,并被记录为`problem_edit`. 两次失败的回答是`malformed`提案. 任何其它供应商的失败都以`planner_failed`结束运行. `MockProvider`和`ReplayProvider`在没有网络的情况下运行.

`RecordedPlanner(proposals)`按顺序为有记录的提案服务. `replay`用.

## 运行目录并重播

`save_run(outcome, run_dir)`写道:

| 文件 | 内容 |
|---|---|
| `problem.json` | 问题在于 |
| `trace.json` | 追踪,计划`agent-trace-1.2.json`。 |
| `results/<label>.json` | 运行的每个结果,全部,通过标签(`iteration-0003`,`verify-1-scipy.slsqp`). |
| `sealed/` | 其最终证据的最佳结果`seal_run`,格式`orbitopt.optimization.run/2`. |
| `tool-calls.json` | 工具调用日志, 当运行使用 `ToolView` 。 |
| `manifest.json` | 格式 `orbitopt.agent.run/1` : 上面每个文件的SHA-256 , 微量文摘, 和密封的运行的表单文摘. |

The sealed run's manifest digest lives in `manifest.json`, outside `sealed/`,
as `seal_run` asks. `save_run` returns the digest of `manifest.json` itself.
Keep it out of band, since whoever can rewrite the directory can rewrite the
manifest too.

`check_run(run_dir, *, tools=None, registry=None, expected_manifest_digest=None, alternates="recorded")`是诚信检查. 它重新执行什么。

1. 每个文件名 清单 和封存的文件夹,去
   通过 `safe_relative`. 一个带有 `..` 的名称, 一个绝对路径, 一个反斜
   or a character outside `[A-Za-z0-9_.-]` raises `UnsafePathError` before
   除了`manifest.json`之外,任何文件都会被读取. 封存的运算单
   在读完任何封存文件之前,名字也会通过它。
2. 每个文件必须散列到文摘中 显示记录和
   名单必须封印`problem.json`和`trace.json`. 与
   `expected_manifest_digest`, 显示本身必须散去。
3. 追踪必须符合自己的文摘,在
   `problem.json`,并匹配 清单的痕量摘要。
4. 追踪中的每一结果摘要必须是全部结果的摘要
   `sealed/result.json`必须是字节
   将追踪名称作为最佳结果。
5. 警卫将`sealed/`与`verify_run`重新等级,钉在密封的上
   说明文摘。 与追踪状态相矛盾的判决是
   寻找。

守门员把`alternates="recorded"`的每个最终结果... 验证的副手都保存下来了 它会重新检查替补的点和比较,并运行不求解. 它不能显示解决者产生这个点. `alternates="rerun"`再次以`registry`的录制分配,种子和选项进行换行. 卫兵重估分数 跑步时通过`tools`

`CheckReport`持有`integrity`(每个发现一行),警卫的`report`,追踪的`recorded_status`,追踪,问题和每一个文件的字节. `intact`是真实的,没有发现。 `verified`也是真实的,当它也完好无损,守卫再次通过密封跑道. `orbitopt agent verify`运行`check_run`,候补再运行.

`check_run`的渔获量和它没有的:

- 一个变化的字节,除非重新写出清单以匹配. 扣紧
  与`expected_manifest_digest`一起显示 赶上这个。
- 伪造者重写痕量和每份摘要:痕量不再
  汇总保存的结果, 或封存的结果不再重新评价
  对它的索赔。
- 伪造者重写一个不是最好的结果, 它的痕迹输入和
  每份文摘都没有被捕捉,因为没有东西能重新运行解决. `replay`是.

`replay(run_dir, *, tools=None, registry=None, expected_manifest_digest=None)`运行`check_run`,候补再运行. 然后,它用`RecordedPlanner`重新执行所记录的关于`problem.json`问题的建议书,并有记录的种子,限制和政策. 它将新追踪与12记录的 显著数字,墙上时钟的字段相提并论。 `ReplayReport.check`是支票,`integrity`和`sealed_verdict`从中读取. `ReplayReport.equal`是真实的,当文件没有变化,没有什么不同的时候. 伪造者编辑一个结果并重算每个摘要仍然失败,因为解析者重算结果. 其能力将`deterministic_given_seed`设定为虚假的解析器,除非复制其结果,否则显示为差异.

## 追踪

`Trace.to_dict()`对`agent-trace-1.2.json`进行验证. 每个迭代记录都持有原始提案,决定或违规,结果摘要,屏幕(核查,有原因的四级状态,发现代码),诊断,评价及其成本. 每个验证记录都持有候选人的标签,候补跑步和警卫的报告. 文摘为IR的`canonical_digest`(12显著的位数),没有`wall_s`,`wall_time_s`和`latency_s`(`orbitopt.optimization.VOLATILE_KEYS`,`result_digest`也省略). 两人以同样的问题运行,计划者,种子和解子分享. 横跨12的数位圆形边界仍将其分割.

三个顶级的字段说是谁和什么运行了循环:

| 外地 | 内容 | 在摘要中 |
|---|---|---|
| `provider` | LLM计划员背后的聊天提供商(其`name`),`null`计划员无一人. | 是:另一个供应商是另一个运行。 |
| `model` | 提供商的模型,`null`当无或未知时. | 是的,另一个模型是另一个运行。 |
| `orbitopt_version` | 运行该循环的已安装的轨道opt版本. | 不:它描述的是机器(`ENVIRONMENT_KEYS`),而重播也没有比较. |

计划员宣布`provider`和`model`. `LLMPlanner`从它的聊天提供商那里读到,`RuleBasedPlanner`没有,`RecordedPlanner`把它们当作论据,`replay`从追踪中填充. `ReplayProvider`报告提供商和模型的磁带录制,所以从磁带中重播的运行会保存录制的运行摘要.

`outcome.cancel` is where a cancellation fired, or `null`: `iteration` is
the number of iterations completed, and `reason` is the check that fired,
`before_iteration` (before the next planner proposal) or
`before_verification`. It is part of the digest. `replay` passes it to
`solve` as a `CancelPoint`, so a cancelled run replays equal.

### 1.0和1.1版本

1.2版本增加了`outcome.budget`. 1.1追逐`agent-trace-1.1.json`,`upgrade_trace`给它自己的记录显示的事件(`derive_budget`). 当前运行时间复制的运行是相等的,除非它因为缺少评价而跳过一个舞台,旧的跟踪无法记录.


1.1版本增加了`outcome.cancel`,`provider`,`model`和`orbitopt_version`. 1.0追踪仍然载着:`load_trace`对照其`schemaVersion`名称(`TRACE_SCHEMAS`)检查每个跟踪,并按记录返回,其摘要按记录检查. `upgrade_trace`将1.2的形状与`cancel`,`provider`,`model`和`orbitopt_version`设定为`null`和由此衍生的预算事件,`check_run`和`replay`为此工作. 因此,1.0运行时没有取消信号,就像1.0下的那样。 未取消的1.0运行的重播等值,因为重播并不比作`schemaVersion`,但其重播的文摘是1.2文摘,与录制的文摘不同. 已取消的 1.0 运行不重复播放, 因为重播无法知道它在哪里停止 。 它仍然经过`check_run`。

## 测量运行

`tests/test_e14_05_runtime.py` runs six catalogue instances with each
planner. The scripted LLM is a `MockProvider` whose function reads the
runtime's prompt and answers: a global or specialised solve first, a local
refinement from the incumbent, restarts with new seeds for a heuristic, then
stop. Measured with seed 0 on one machine (macOS arm64, Python 3.12, scipy
from the `orbitopt-py312` environment). The rule-based rows were re-measured after the solver adapters
started to restore band points; the two that moved gained 1 and 3 verification evaluations, and the LLM
rows were not re-measured:

| 实例 | 规划员 | 状态 | 目标(参考) | 解决方案 | 评价,循环 | 评价,验证 | 限制 | 键 |
|---|---|---|---|---|---|---|---|---|
| cqp-worked-d 2 | 规则 | 已核实 | 0.5 (0.5) | 1 | 7 | 801 | 4000 | 0 |
| cqp-worked-d 2 | LLM | 已核实 | 0.5 (0.5) | 1 | 7 | 800 | 4000 | 555 |
| lp-wyndor-d2 | 规则 | 已核实 | 36 (36) | 1 | 2 | 396 | 4000 | 0 |
| lp-wyndor-d2 | LLM | 已核实 | 36 (36) | 1 | 2 | 393 | 4000 | 617 |
| 背包-p 01 -n 10 | 规则 | 已核实 | 309 (309) | 1 | 2 | 2051 | 20000 | 0 |
| 背包-p 01 -n 10 | LLM | 已核实 | 309 (309) | 1 | 2 | 2051 | 20000 | 769 |
| 拉斯特里金-d 5 | 规则 | 已核实 | 1.4e-14 (0) | 4 | 5678 | 2170 | 10000 | 0 |
| 拉斯特里金-d 5 | LLM | 已核实 | 1.4e-14 (0) | 2 | 3757 | 3131 | 10000 | 1038 |
| rosenbrock-d20 | 规则 | 已核实 | 4.1e-10 (0) | 4 | 26271 | 7926 | 40000 | 0 |
| rosenbrock-d20 | LLM | 已核实 | 1.5e-9 (0) | 2 | 17500 | 12311 | 40000 | 1578 |
| 缩略语 -1e-10 | 规则 | 已核实 | 5 (5) | 3 | 479 | 2051 | 4000 | 0 |
| 缩略语 -1e-10 | LLM | 已核实 | 5 (5) | 4 | 646 | 2051 | 4000 | 1583 |

限制这些数字的是什么:

- 在对流和线性实例中,循环需要一个解决,
  验证成本最高. 这个成本是替代的差别
  演变是趋同的,而不是预算:它远远低于其份额。
- 在Rastrigin和Rosenbrock上 环路花的是规划者给的
  差异演化,每回合剩下的一半,因为
  解析结束于评价限制。 规则规划员的第二个
  圆形没有改善,所以它的重启就在那里结束.
- 在两个有限的域上,进行点数验证。 `knapsack-p01-n10`有
  1024积分:计数候补花1024,
  多点1024,最终报告通过`optimality_certified`.
  我也一样。 合并实例(4规则乘以 N 至 512, 2048积分)
  有不同的进化和数。 计数符
  N=5与2048评价,故运行为`verified`. 1470
  剩下的评价不能包括第二次统计,所以
  `optimality_certified`仍为未评估. 在列举之前,它没有
  alternate and was `unverified_best`.
- `knapsack-p07-n15`(32,768点) 默认限值30,000为
  与HIGHS相比,通过差异演化来验证,而不列举.
  通过100,000评价,基于规则的运行得到核实和认证
  65,539. 在1456, 第一次解答是差异进化,
  被计数候补和警卫自己的计数所反驳,
  在1458的第二次尝试中被验证.
- Rastrigin的无转动优化是盒式中心,SLSQP开始于
  默认。 那边的替补支票几乎都一样 变种
  没有这样的捷径。
- Tokens是模拟的词数, 即时加回答, 所以他们显示如何
  快速增长与历史, 而不是什么真正的模型会帐单。 这个
  enumeration solver's entry in the prompt added 32 tokens on the knapsack
  和80在融合案中,超过两个和四个电话。

## 测试

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
