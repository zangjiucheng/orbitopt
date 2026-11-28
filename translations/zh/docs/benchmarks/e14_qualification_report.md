# E14-10资格报告:第一次竞选.

> 由[campaign v2](e14_qualification_report_v2.md)取代, 本报告及其结果保持记录。

这是[E14-10 benchmark specification](e14_benchmark_spec.md)的首次得分赛. 它用Spec's 10的种子运行了三臂,在29的所有评分例上不需要用模型,并检查了安全指标. 需要模型的四条臂和度量衡没有运行,因为没有配置任何提供者。

**Read this first.** No LLM was called. Arm `E-rule` is the E14 runtime with its deterministic rule-based planner and
its verification guard, run on the hand-written reference problem. It is not the pre-registered arm E, which compiles
the natural-language statement and plans with a model. `E-rule` against A or B measures the runtime's solver
selection, restarts and verification. **It says nothing about whether an LLM helps.**

## 1. 问答

Spec询问,每个问题分类的分析组,循环中的LLM是否会改变成功率,评价成本或墙钟相对于传统臂,验证层是否曾经通过过一个错误的结果.

这场运动能回答什么:

- **LLM值:未回答.**武器C,D和E以及配方精度没有运行(3节).
- **B against A (primary).** The rule table beats the fixed differential-evolution solver overall, on success
  (+0.255 success rate, 95 % CI [0.19, 0.33], Holm p = 0.0004 of 15), evaluations (ERT ratio 0.23, CI [0.17, 0.30])
  and wall clock (ERT-time ratio 0.35, CI [0.29, 0.40]). The whole effect sits in the algebraic group (LP, convex
  QP,MILP),存在精确的解析器. 光谱将它预言为 一种控制状态的检查,而不是一个发现。
- **E-rule against B (secondary).** Not shown different on success (-0.04, CI [-0.13, 0.05]). E-rule is inferior
  on evaluations (11.3x, CI [9.0, 14.2]) and wall clock (4.9x, CI [3.7, 6.5]), because its final verification costs
  评估远多于解决方案。 在连续的非凸轮组(Rosenbrock, Rastrigin, Acley)上是
  superior on evaluations (0.80x, CI [0.74, 0.86], Holm p = 0.035 of 9).
- **Safety.** No faulty artifact passed the guard: 0 of 1,008 in memory and 0 of 1,008 as sealed runs (one-sided
  95%上行0.30%). 错误时未验证分数运行 。 重播是EQUAL 全部870 得分规则臂
  cells. Budget enforcement is 290/290 for A and B but **286/290 for E-rule**: four cells used exactly their
  (c) 评价预算和报告`planner_stop`(第6.2节)。 硬约束在E-规则中
  (2,400 of 2,400).

## 2 设置

### 冰冻文摘

冻结是在任何得分的shard run(承诺`8c06de7`,文件[`benchmarks/results/e14/freeze.json`](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/results/e14/freeze.json). 每个得分的shard拒绝运行,除非所有三个摘要仍然匹配,并记录在它的文件中. `analyze`拒绝其文摘差异的软体.

| 摘要 | 价值 |
|---|---|
| spec (`e14_spec.spec_digest()`, 引自该 spec) | `sha256:e5104c340d419f388643bed2ad1de32662f8384f7b1c4adb7a05b90ad877db47` |
| 协议(跑者自己的选择,`e14_common.PROTOCOL`) | `sha256:ecefd6a0a83ffb8b4abd637c6cb218f88ea92ff4c499974f8f35bdd058119e46` |
| 代码( 每个运行符和 `src/orbitopt` 源文件) | `sha256:d4b73f6935d150d60bf763f95d4bdaf125b5e3c4ea0cd32ac97e0a01dff483b3` |
| 冻结文件内容 | `sha256:7c7fde0ca571983eee48cd20467eecdeb59a1a0a93925b3bf676724b86cba47f` |

笔记本电脑和服务器计算出相同的三个摘要

### 军火

| 手臂 | 问题 | 解析器 | 说明 |
|---|---|---|---|
| A级 | 参考可调用 | `scipy.optimize.differential_evolution`,细胞种子,`polish=True`,弹出大小15,几代配属90% 预算,从可变类型中集成. | 多目标:Tchebycheff超过50权重,每个权重由预算的DE与1/50相同. |
| B | 参考可调用 | Spec的规则表:HIGHS `linprog` / `milp` 系数从可调用物中探明,`trust-constr` 从盒式中心探明,SLSQP 多点启动直到预算,arm-A DE为`global`,列举`black_box`增加的申报成本,Tchebycheff+SLSQP为多点目标. | `global` is the same DE as A, so B = A there by construction |
| E-规则 | 参考文献 IR | `orbitopt.agent.runtime.solve`与`RuleBasedPlanner(max_restarts=3)`,默认政策和注册,IR的默认验证政策 | 用 `save_run` 保存的每个单元格, 用 `replay` 重播 |

A and B call scipy directly on the catalogue's Python callables through their own budgeted evaluator, so they share
no code with the runtime. The budget is the spec's: 2000 x max(d, 2) point evaluations and 600 s per cell.

### 种子、调制和搁置

The spec fixes the seeds: scoring seeds 0-9 and tuning seeds 1000-1004. The runner uses those.

The tuning split was used once, before the freeze, as a dry run (120 cells). It changed one parameter: B's
`trust-constr` rule stopped 1.6e-4 short of the QP optimum on `cqp-worked-d2` with scipy defaults, and reaches
8.7e-13 with 3-point differences, BFGS Hessians, tolerances 1e-12 and an initial barrier parameter of 1e-6. Nothing
else changed. No scoring instance was run before the freeze. All three entries are in `PROTOCOL["tuning_adjustments"]`.

### 跑到哪里去了? 跑到哪里去了? 跑到哪里去了?

| 项目 | 价值 |
|---|---|
| 服务器 | `j7zang`(`linux.student.cs.uwaterloo.ca`,负载平衡器;每个得分的shard运行在主机`ubuntu2404-002`). |
| 环境 | `~/e11env`: Python 3.12.13, numpy 2.5.5, scipy 1.18.1, pygmo 2.20; nothing installed |
| 工作空间 | `~/e14-10-campaign/`(代码为`code-8c06de7/`,结果为`out/`,日志为`logs/`,发射器为`e14run.sh`,`e14jobs.sh`),留在原位. |
| 任务 | 4进程一次交错,由20 s,`OMP/OPENBLAS/MKL_NUM_THREADS=1`,`nice 10`左右组成. |
| 墙壁时间 | 9 min 10 s for all 105 shards (03:13:01 to 03:22:11 UTC). The slowest shard, `cells.scoring.E-rule.knapsack`, took 180 s including replays. No shard came near the 30-minute cap. |
| 失败和重运行 | 无。 早先的一次调制干线运行在代码`63c30a2`下;它的硬盘运行在冻结代码下. |

一个过程依次贯穿着一个家庭的每个臂膀,所以一个实例的臂膀共享一个主机.

## 3. 什么没有运行,为什么

| 项目 | 为什么 | 如何运行它 |
|---|---|---|
| (模型建议点数) | 无提供者 | 设置下面的变量, 并重运行 `cells.*.C.*` 硬盘 |
| 臂 D (编译器 + 固定解析器) | 无提供者 | 同样,`cells.*.D.*` |
| 臂E,记录为`E-LLM`(编译器+LLM规划器) | 无提供者 | 同样,`cells.*.E-LLM.*` |
| `formulation_accuracy`(58病例×10种子). | 需要一个真正的模型 | `formulation.scoring` |
| 编译器的 `ir_schema_validity` | 需要一个真正的模型 | 武器D和E-LLM和配方 |
| 代号成本 | 0代币和0按建筑规则的A,B和E- | C、D、E-LLM |
| 初级比较 C-A, D-A, E-D, E-B. | 每个臂没有运行 | counted as p = 1 in the Holm family of 15 |
| 外部 CUTEst 和 COCO 实例 | 未销售(第3节) | |

每个未运行的单元格仍然在结果中,作为`not_run`的一排和`no provider`的原因:C,D和E-LLM各有290的得分排. 没有任何模拟或磁带回答模型在任何得分行。

仅从环境配置提供者 :

| 变量 | 含义 |
|---|---|
| `E14_LLM_KIND` | `openai_compat`(默认)或`local` |
| `E14_LLM_BASE_URL` | 终点,例如 `https://api.openai.com/v1` 或 `http://127.0.0.1:11434/v1` |
| `E14_LLM_MODEL` | 模型 id,记录在每一个硬体的出处 |
| `E14_LLM_API_KEY` | `openai_compat` 的密钥; 提供者在呼叫时读取, 从未存储过 |

Historical names: these are the variables v1 read. From campaign v3 on, the campaign, the CLI and Mission
Studio read one set, `LLM_KIND`, `LLM_MODEL`, `LLM_BASE_URL` and `LLM_API_KEY`
([providers](https://github.com/zangjiucheng/orbitopt/blob/main/docs/agent/providers.md#configuration-from-the-environment)); the `E14_LLM_*` names are gone.

The provider arms record every model call to a cassette, and their replay re-runs the cell from it. The pipeline is
tested in `tests/test_e14_10_campaign.py` with mock providers, labelled `pipeline-smoke`. Analysis refuses those rows
unless told it is a pipeline test, and then labels every number "not evidence".

## 4 每类结果

成功计数细胞(Intances x 10种子). 中位差是光谱4节的常态间隙. "提醒"是其间隙低于负容忍度的单元格(光谱6.1节);光谱说这样的单元格不是胜负,所以算不上成功.

| 类 | 手臂 | 单元格 | 成绩 | 可行的 | 提醒 | 中位数差距 | 预算用尽 | 评价中值 | 中值墙s |
|---|---|---|---|---|---|---|---|---|---|
| ưμ㼯A | A级 | 20 | 0 | 20 | 0 | 5.18e-06 | 0 | 9.22e+03 | 0.489 |
| ưμ㼯A | B | 20 | 20 | 20 | 0 | 0 | 0 | 8 | 0.00146 |
| ưμ㼯A | E-规则 | 20 | 20 | 20 | 0 | 0 | 0 | 3e+03 | 0.284 |
| 对流 _ qp | A级 | 40 | 5 | 40 | 0 | 5.12e-08 | 5 | 4.78e+03 | 0.264 |
| 对流 _ qp | B | 40 | 40 | 40 | 0 | 1.37e-11 | 0 | 160 | 0.0253 |
| 对流 _ qp | E-规则 | 40 | 21 | 40 | 19 | 1.05e-15 | 0 | 3.97e+03 | 0.281 |
| 米尔普 | A级 | 40 | 33 | 40 | 0 | 0 | 0 | 4.02e+03 | 0.81 |
| 米尔普 | B | 40 | 40 | 40 | 0 | 0 | 0 | 15.5 | 0.013 |
| 米尔普 | E-规则 | 40 | 40 | 40 | 0 | 0 | 0 | 9.8e+03 | 0.919 |
| 无 | A级 | 40 | 36 | 40 | 0 | 1.04e-10 | 0 | 1.38e+04 | 0.562 |
| 无 | B | 40 | 40 | 40 | 0 | 6.43e-09 | 40 | 1.5e+04 | 0.472 |
| 无 | E-规则 | 40 | 40 | 40 | 0 | 3.41e-11 | 3 | 1.25e+04 | 0.825 |
| 全球 | A级 | 60 | 33 | 60 | 0 | 2.31e-08 | 0 | 1.82e+04 | 0.794 |
| 全球 | B | 60 | 33 | 60 | 0 | 2.31e-08 | 0 | 1.82e+04 | 0.787 |
| 全球 | E-规则 | 60 | 50 | 60 | 0 | 6.53e-09 | 0 | 1.65e+04 | 1.05 |
| 黑 _盒 | A级 | 60 | 53 | 60 | 0 | 0 | 0 | 319 | 0.0424 |
| 黑 _盒 | B | 60 | 60 | 60 | 0 | 0 | 0 | 104 | 0.0197 |
| 黑 _盒 | E-规则 | 60 | 60 | 60 | 0 | 0 | 0 | 2.38e+03 | 0.109 |
| 多 _目标 | A级 | 30 | 29 | 30 | 0 | 0.0178 | 0 | 9.41e+03 | 0.599 |
| 多 _目标 | B | 30 | 30 | 30 | 0 | 0.0105 | 30 | 1e+04 | 0.762 |
| 多 _目标 | E-规则 | 30 | 21 | 30 | 0 | 0.0151 | 0 | 3.41e+03 | 0.0944 |

The spec's target for A, B, D and E is success = 1.00 on lp, convex_qp and milp, "a miss is a defect". B meets it.
A misses it, as the spec expected of differential evolution (its QP gaps of about 5e-8 sit just above the 1e-8
tolerance, and its LP gaps near 5e-6). E-rule meets it on lp and milp and misses it on convex_qp: that is the alarm
of section 6.1, a defect in how the runtime picks its best candidate.

### 比较

每对试验对臂按例和种子,先在每对试验对子内组装种子,并视例为单位:在成功时进行精确的集群标志-flip测试,在每对试验对子日志ERT比率上进行精确的威尔科森签名-排名测试,在每对试验对子中进行集群靴子CI(10,000复标,种子1410,层=家族),每个家族内部的霍尔姆,以及Spec判断规则. *高级*至少需要6信息实例,Holm p < 0.05和CI排除0的有利侧. 低于1的比例意味着治疗更便宜. 间隙端点为二级:中间间隙的平均对数10比(gaps downed at 1e-12),负表示治疗接近.

**Overall (29 instances).** Primary family: Holm over the spec's 15 hypotheses, the 12 not run counted as p = 1.
Secondary family (E-rule): Holm over 6.

| 比较 | 结束点 | 效果 | 95 % CI | 资料 | p | pholm (m) (韩语). | 判决 |
|---|---|---|---|---|---|---|---|
| B对A | 成功率差异 | +0.255 | [0.19, 0.328] | 16 | 3.05e-05 | 0.000397 (15) | 高级 |
| B对A | ERT比率 | 0.225 | [0.168, 0.3] | 23 | 2.62e-05 | 0.000387 (15) | 高级 |
| B对A | ERT-时间比 | 0.345 | [0.292, 0.401] | 27 | 2.58e-05 | 0.000387 (15) | 高级 |
| B对A | 对数 10差距比 | -1.21 | [-1.88, -0.629] | 14 | 0.0245 | 0.0245 (3) | 高级 |
| E-规则对As | 成功率差异 | +0.217 | [0.1, 0.338] | 20 | 0.0108 | 0.0431 (6) | 高级 |
| E-规则对As | ERT比率 | 1.59 | [1.32, 1.91] | 27 | 0.786 | 1 (6) | 未显示不同 |
| E-规则对As | ERT-时间比 | 1.07 | [0.899, 1.29] | 27 | 0.53 | 1 (6) | 未显示不同 |
| E-规则对As | 对数 10差距比 | -2.2 | [-3.24, -1.3] | 18 | 0.000145 | 0.000435 (3) | 高级 |
| E-规则与Bs | 成功率差异 | -0.0379 | [-0.128, 0.0517] | 8 | 0.617 | 1 (6) | 未显示不同 |
| E-规则与Bs | ERT比率 | 11.3 | [9, 14.2] | 28 | 0.00611 | 0.0305 (6) | 低级 |
| E-规则与Bs | ERT-时间比 | 4.92 | [3.68, 6.47] | 28 | 0.00225 | 0.0135 (6) | 低级 |
| E-规则与Bs | 对数 10差距比 | -0.642 | [-1.52, 0.127] | 16 | 0.0739 | 0.0739 (3) | 未显示不同 |

The success rates behind the first row are 0.907 (B) and 0.652 (A). As checks, an exact McNemar test on seed 0 only
(one pair per instance, where the spec's cluster test reduces to McNemar) gives 8 instances where only B succeeded
and 0 the other way, p = 0.0078. A McNemar test pooled over seeds gives 74 to 0 and p = 1e-22. That pooled p
ignores clustering and is not evidence: it treats 10 seeds of one instance as 10 independent problems.

ERT CI只涵盖两臂至少解一次(24代表B vs A);只有一只臂解入Wilcoxon测试作为无限比例(5代表B vs A,全部为B)并计入成功终点的情况.

**Per analysis group.** Holm runs over the 9 tests in each group.

| 团体(事件) | 比较 | success diff [CI], verdict | ERT ratio [CI], verdict | ERT- 时间比 [ CI ] 判决. |
|---|---|---|---|---|
| 代数 (10) | B对A | +0.62 [0.44, 0.81], superior | 0.005 [0.0019, 0.013], superior | 0.019 [0.0087, 0.037], superior |
| 代数 (10) | E-规则对As | +0.43 [0.14, 0.72], not shown different | 1.85 [1.05, 3.14], not shown different | 1.03 [0.61, 1.73], not shown different |
| 代数 (10) | E-规则与Bs | -0.19 [-0.38, 0], not testable (2 informative) | 225 [148, 328], inferior | 53 [23.5, 109], inferior |
| 连续 _ 非凸起( 10 ) | B对A | +0.04 [ 0 0.09 ] 不可考 | 0.99 [ 0.96 1.03 ] 不可考 | 0.92 [0.84, 1.04], not shown different |
| 连续 _ 非凸起( 10 ) | E-规则对As | +0.21 [0.07, 0.36], not shown different (Holm p 0.22) | 0.79 [0.72, 0.88], superior | 1.13 [1.05, 1.21], not shown different |
| 连续 _ 非凸起( 10 ) | E-规则与Bs | +0.17 [0.04, 0.33], not testable (4 informative) | 0.80 [0.74, 0.86], superior | 1.24 [1.09, 1.40], not shown different |
| 黑_盒 (6) | B对A | +0.12 [ 0.03 0.23 ] 不可考 | 0.22 [0.09, 0.45], not shown different | 0.40 [0.31, 0.49], not shown different |
| 黑_盒 (6) | E-规则与Bs | 0 [ 0 0 ] 不可考 | 30 [15.8, 58.7], not shown different | 5.6 [5.1, 6.2], not shown different |
| multi_objective (3) | 全部 | 仅说明(规格:3实例不能达到p < 0.05) | | |

The black-box group cannot reach p < 0.05 unless all 6 instances are discordant in one direction. Its cost ratios are
6 of 6 in one direction with exact p = 0.031, but Holm over 9 tests lifts that to 0.28, so the verdict stays "not
shown different".

**Sensitivity.** The frozen scorer counts the 19 alarm cells as successes. With that reading, E-rule vs B is 0.0 on
algebraic success (no informative instance) and +0.028 [-0.035, 0.09] overall (not shown different), and E-rule vs A
is +0.283 [0.19, 0.38] overall (superior, Holm p 0.008). No verdict on B vs A changes. The full table is in
[`tables.md`](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/results/e14/tables.md).

### 何以限制这些数字

- **E-rule's evaluation cost is its verification.** From the saved run directories
  ([`erule_evaluation_split.jsonl`](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/results/e14/erule_evaluation_split.jsonl),描述性,后加)
  运行后:在 LPs 和 knapsacks 上, 规划器循环花费 2 评价(一个 HIGHS 解析), 以及最终
  验证用418到24,526(候补解析器和计数器). 在集成实例中,循环
  spends about 300 and verification 2,051, which is the enumeration of the 2,048-point domain. A and B verify
  没有,所以ERT比较 收费E-规则 工作基线不做。 在非对流组上,循环是
  59到77占总数的%. ERT由成功种子分出,所以E-规则的下部ERT有一部分来源于它.
  成功率较高(从定义中推断,不单独衡量)。
- **B's `nlp`和多目标规则通过设计花费整个预算**("SLSQP多点开始直到预算"),所以.
  their 70 `budget_exhausted` cells are the rule, not a failure, and their ERT is the budget divided by the success
  费率。
- **B = A on `global`.** The rule table names the same DE, so 6 of the 10 continuous-nonconvex instances cannot be
  B对A不和
- **Wall clock is short and shared.** Every cell finished in under 3.5 s against the 600 s cap. All shards ran on one
  共享主机在大约2到7的负载下,所以墙-钟比携带该主机的噪音. 墙钟覆盖
  只解决,不保存或重放运行。 高斯- Legendre 规则缓存在每一个单元格的第一个单元格之前建立
  硬,所以没有手臂支付它。
- **Most group-level tests are not testable.** Ten instances per group is the minimum the spec's exact test can use,
  和一些比较中,6不和谐的事例较少.

## 5  验证地位(E-规则).

| 运行状态 | 单元格 | 地点 |
|---|---|---|
| 已核实 | 214 | |
| 未验证的 _ 最佳 | 76 | `rastrigin-d10-shifted`(10,全错),三个得分的MO例(30:NSGA-II没有交替),`cqp-projection-d2`和`cqp-equality-d3`(20,报警格),`ackley-d10-shifted`(10,全错:候补人没有到达其盆地),`integ-nearsing-1e-6`,`knapsack-p07-n15`和`knapsack-correlated-n20`(2各). |

`optimality_certified`通过20细胞,全部在`knapsack-p01-n10`和`knapsack-2d-n12`上,它们的域位(1,024和4,096点)适合给警卫自己计数留下的评价. 未评估在另一个270: 持续的非对流问题从未被认证, 并且较大的域不适合. 看守在30牢房(MO案例:没有合并的说法和没有证据)的判决失败. 每个经核实的牢房都成功:在无法使用或无法接受的情况下,没有一间牢房得到核实。

A和B只携带警卫的屏幕(没有独立证据),因此他们的判决是`not_assessed`或`failed`(解方停止预算时的`failed`),因此没有汇合. 他们的判决不能是故意的。

## 6. 安全标准

| 计量 | A级 | B | E-规则 | 目标 |
|---|---|---|---|---|
| 预算执行 | 290/290 | 290/290 | **286/290** | 1.00 |
| 独立柜台=报告的评价 | 290/290 | 290/290 | 290/290 | |
| 大多数评价使用/预算 | 1.0 | 1.0 | 1.0 | <= 1 |
| 实验重播 EQUAL | 290/290 | 290/290 | 290/290 | 1.00 |
| 保留了硬性限制 | n/a(无制剂) | 无 | 2,400/2,400 | 1.00 |
| 盘点已校验但错误 | 0 | 0 | 0 214 已核实 | 0 |
| 符号, USD | 0, 0 | 0, 0 | 0, 0 | 只报告 |

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

### 6.3 假验证 通行证:断层注射

Every scoring instance got every fault on up to four hosts: the seed-0 point of A, of B and of E-rule, and the stored
optimum where the catalogue has one (25 instances). That is 112 hosts x 9 faults = 1,008 faulty artifacts, each
graded in memory (`verify_result`) and as a sealed run (`verify_run`).

| 路径 | 职等 | 假通行证 | 95 % 上限 | 名称 | 诚实主机通过 | 错传这些 | 上面写着: |
|---|---|---|---|---|---|---|---|
| 校验 _结果 | 1,008 | 0 | 0.30 % | 993 | 792 | 0 | 792 |
| 验证 _ 运行 | 1,008 | 0 | 0.30 % | 993 | 792 | 0 | 792 |

A fault on a host whose honest artifact already fails proves little, so the informative count is the 792 artifacts
whose honest host passed (88 hosts). On those, every fault was rejected for its own reason (one-sided 95 % upper
bound on the false-pass rate 0.38 %). The honest host fails for 24 hosts: 15 whose point is not optimal (A on the QPs,
the LPs and Rastrigin, B and E-rule on `rastrigin-d10-shifted`, E-rule on the two alarm QPs, and so on) and the 9 MO
hosts, which have no reference objective to give as independent evidence. The 15 cells per path where the fault's own reason was not
named are all MO hosts: F09 (a forged verifier id) cannot be named when there is no evidence item, and F07 replaces
the first objective with `x[0]`, which on ZDT1 and ZDT2 is the objective already, so the injection changes nothing.
They were all rejected.

870得分规则-arm运行后,公尺分母为1,878每路径;假路数为0.

## 7. 失败

Every instance that an arm did not solve on all 10 seeds, from the generated table:

| 手臂 | 实例 | 成绩 | 中位数差距 | 差距最大 |
|---|---|---|---|---|
| A级 | 拉斯特里金-d 20 | 0/10 | 36.8 | 72.6 |
| A、B 类 | rastrigin-d 10 - 转移 | 0/10 | 10.9 | 15.9 |
| E-规则 | rastrigin-d 10 - 转移 | 0/10 | 6.96 | 10.9 |
| A、B 类 | 拉斯特里金-d 5 | 6/10 | 3.9e-14 | 0.995 |
| A级 | cqp-projection-d 2, cqp-simplex-d 5, cqp-simplex-d 10 | 0/10 each | 5e-8至6e-6 | |
| A级 | 平等-d 3 | 5 / 10 (预算中的5单元格). | 4e-7 | 0.396 |
| A级 | Lp-wyndor-d 2, knapsack-p 01 -lp-laxy | 0/10 each | 9e-6, 7e-7 | |
| A级 | 背包-cor-l-n 20 | 5/10 | 0.0018 | 0.0036 |
| A级 | 融合(4,6例). | 6 / 10 至 9 / 10 | 0 | 最多为1.62 |
| A级 | rosenbrock-d5, rosenbrock-d20, ackley-d20, ackley-d10-shifted, knapsack-2d-n12, knapsack-p07-n15, mo-zdt2-d5 | 7 / 10 至 9 / 10 | | |
| B | ackley-d20 (9/10), ackley-d10-shifted (8/10) | | | |
| E-规则 | cqp-预测-d 2(0/10),cqp-平等-d 3(1/10). | 6.1区警报室 | | |
| E-规则 | mo-zdt2-d5 | 2/10 | 0.020 | 0.333 |
| E-规则 | mo-zdt1-d5 | 9/10 | 0.017 | 0.043 |

`rastrigin-d10-shifted` and `rastrigin-d20` defeat every arm here; the shifted variant is the one the spec added
because no textbook optimum points at it. E-rule's NSGA-II front on ZDT2 (non-convex) misses the 5 % hypervolume
tolerance in 8 of 10 seeds, where B's 50-weight Tchebycheff passes all 10.

## 8. 对有效性的威胁

- **No LLM.** Nothing here measures what a model adds. The comparisons that answer the spec's question (C-A, D-A,
  E-D,E-B)没有数据.
- **E-rule is not arm E.** It runs on the hand-written problem, so it cannot fail at formulation, and its planner is
  固定代码。 其结果将时间和警卫的工作与一个完美的配方和一个谨慎的规划者联系在一起。
- **Verification is charged to E-rule only.** A and B return unverified points; E-rule pays thousands of evaluations
  以核实。 成本终点比较一个答案和一个经过核实的答案.
- **Runner choices the spec left open** (A's generation count, A on multi-objective, the ideal point of the
  Tchebycheff 权重,B按说明的成本订购黑盒网格,即计分器中的整体性术语)是:
  运行前在协议文摘中固定。 其他合理的选择可以移动A和B。
- **一个调制变化**(B的`trust-constr`选项)被设定在调音拆分上. 没有它B的对流 - QP规则
  可能会错过调音实例 也可能错过得分QP
- **宽容的边缘。** 几个 失败是容忍度的几倍(QPs at about 5e-8 at 1e-8),
  E-规则警示细胞 坐在约束容忍度。 接近容忍度的成功对解析器设置很敏感.
- **Small groups.** Ten instances per group is the floor of the exact test; black box and multi-objective cannot
  种子不是独立的单位,所以它们增加的动力很少。
- **备忘录不适用**没有LLM,所以`well_known`的分裂没有被分析.
- **一个共享主机**用于所有壁点数,在其他用户负载下.
- **Scorer and analysis.** The frozen scorer did not apply the spec's alarm rule; the analysis applies it, and the
  敏感性表显示计分器的读数. 分析模块在代码摘要之外,其输出为
  可转载:在本次更改前,笔记本电脑将服务器的`analysis.json`重制为相同的内容摘要.

## 9. 数据和完整性

| 文件 | 地点 | 摘要 |
|---|---|---|
| 105硬文件 (4.5 MB). | 服务器, `~/e14-10-campaign/out/shards/` | [`manifest.json`](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/results/e14/manifest.json) (sha 256和内容摘要) |
| E-规则运行目录(7.9 MB). | 服务器, `~/e14-10-campaign/out/runs/`, `~/e14-10-campaign/runs-8c06de7.tar` | 焦油`sha256:187fe9d02033cb53e5ac6ec0b44f7fa1bd947e4e216541457919fc6ca1a86b09`;每跑的表单摘要是行的`artifact_digest`. |
| CSV | [`benchmarks/results/e14/cells.csv`](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/results/e14/cells.csv) | 在仪表上 |
| 分析 | [`benchmarks/results/e14/analysis.json`](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/results/e14/analysis.json), [`tables.md`](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/results/e14/tables.md) | 内容摘要 `sha256:26c076f83d7ad9c90d335fd61020139fadf1b05062cbaee3ef0fbd1b4cd52834` |
| 冻结 | [`benchmarks/results/e14/freeze.json`](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/results/e14/freeze.json) | 2节 |

Every shard file seals its own content digest. The copy-back was `rsync` of `out/` without `runs/` to the laptop,
then `verify`, which re-hashed every file the manifest names: all matched. Nothing on the server was deleted.

## 10 复制

在服务器上(或任何带有NumPy,scipy,jsonschema和pygmo的机器),来自冻结承诺的检查:

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

`run`跳过在同一文摘下已经存在的每一个硬体,所以一个中断的工作通过运行同一个命令来恢复. 分析和表格来自当前分支,在复制的硬文件上:

```bash
python benchmarks/bench_e14_end_to_end.py analyze --out <copy>
python benchmarks/bench_e14_end_to_end.py report --out <copy>
python benchmarks/e14_analysis.py <server out dir>                # the post-hoc E-rule evaluation split
```

The pipeline test, with no server and no model:

```bash
PYTHONPATH=src python -m pytest -q tests/test_e14_10_campaign.py
```
