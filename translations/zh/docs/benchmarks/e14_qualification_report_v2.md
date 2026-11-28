# E14-10资格报告:竞选诉2

这是[E14-10 benchmark specification](e14_benchmark_spec.md)的第二次得分赛. 它将不需要语言的三臂模型重排在29的评分例,种子,跑者,shards和服务器与[campaign v1](e14_qualification_report.md)相同,经过两个运行时间修正后,v 1曝光. v 1及其结果保持记录; v 2是一个单独的冻结,结果在[`benchmarks/results/e14-v2/`](https://github.com/zangjiucheng/orbitopt/tree/main/benchmarks/results/e14-v2/).

**先念此经.** 编号LLM. Arm `E-rule`是E14的运行时间,其基于规则的定型规划者在手写参考问题上. 它不是预先注册的手臂 E. `E-rule` 相对于 A 或 B 测量运行时间的解析器选择、重启和 验证 。 **上面没有说LLM是否有帮助**

## 1. 简短的答复

- **The two v1 defects are gone.** No cell of any arm has a gap below minus the tolerance (v1: 19 E-rule cells).
  E-规则现在在所有40 convex-QP细胞(v 1: 21在spec's rule)上成功,因此满足spec's
  `success = 1.00` target on lp, convex_qp and milp. Every E-rule cell whose run was cut by a limit says so in a
  键入字段(v 1 : 4 细胞使用了他们的整个预算,却一言不发).
- **安全舱** 0 假传1,008 注入每条路径的断层,0 得分在错误时被验证运行(223).
  verified), replay EQUAL 870/870, budget enforcement 290/290 for every rule arm, hard constraints 2,400/2,400.
- **A and B did not move.** Every one of their 580 scored rows has the same row digest as in v1 (timing fields
  excluded). The scorer change touched no A or B cell, and the two arms reproduced exactly on another host.
- **E-rule against B** is still not shown different on success (+0.028, CI [-0.035, 0.090]) and still inferior on
  评估(10.4 x)和墙钟(4.5 x),因为其最终的验证成本远大于B的解. 其差距
  endpoint is now superior (-0.99 in log10, Holm p 0.005 of 3); in v1 the alarm cells hid that.
- **LLM值:未回答.**武器C,D和E-LLM没有运行(没有提供商).

## 2. 与1的变迁

| 变动 | 地点 | 运行时的效果 |
|---|---|---|
| 每个预算削减都打字:`outcome.budget`在微量(代理追踪-1.2)中,命名舞台(跳动,解析,交替,证书,验证,运行),资源和备抵. | 运行时间( [runtime docs](https://github.com/zangjiucheng/orbitopt/blob/main/docs/agent/runtime.md#where-a-limit-cut-the-run) ) | A run that is not verified after a cut of the loop, the final verification or the run is `budget_exhausted`, not `unverified_best`. `verified` still needs the guard's pass on the evidence the cut stage produced. |
| 先清点,然后恢复到限制 | 解析器适配器和运行时间的现任者([solvers docs](https://github.com/zangjiucheng/orbitopt/blob/main/docs/optimization/solvers.md#the-tolerance-band-is-not-room-to-optimise)) | 使用一个约束的容积带的1e-6以上的点,在客观上永远不能排在干净的点上;适配器在上报前将它的最佳带点移动到约束上. 护戒及一切忍忍皆不变. |
| 计分员应用Spec section 2 : 负容忍度以下的空隙是一个提醒,没有成功. | `e14_common.score` | 不需要分析-侧校正,也不需要敏感度表:v 1的分析必须在事后适用该规则. |
| E-规则行在`stop`(`budget_exhausted:<resource>`)中键入切片,`budget_typed`读为A和B. | `bench_e14_agent_iteration.py` | 按每类“预算已耗尽”栏计算,E-规则单元格的验证在分配时已停止。 |

The spec is unchanged (its digest is v1's). The scorer now does what section 2 already says, so no addendum was
needed.

### 根本原因,简而言之

**Budget typing.** The runtime typed a limit only when it ended the planner loop (`stop_reason`). The final
verification's alternates get what the loop leaves, and on `rosenbrock-disk-d2` the differential-evolution
alternate spent the last evaluations and stopped on its allocation. Its own result said `max_evaluations`; the run
said `planner_stop` and `verified`.

**Negative gaps.** Two places ranked tolerance-feasible points by raw objective: the adapters' best evaluated point
and the runtime's incumbent. A point inside the band beats the exact optimum by its multiplier times its violation:

| 实例 | 点 | 违反 | 乘数 | 乘数 x 违规 | v 1 间隙 |
|---|---|---|---|---|---|
| `cqp-projection-d2` | SLSQP, (1 + 2^-26, 0) | 1.49e-8 | 2 | 2.98e-8 | -2.98e-8 |
| `cqp-equality-d3` | 差异演变 | 9.54e-7 | 12/11 | 1.04e-6 | -1.04e-6 |

Differential evolution sees the band as its constraint, so it presses against its edge. On `cqp-equality-d3` that
gain (1.04e-6) exceeds the guard's agreement tolerance (1e-6 x max(1, 6/11)), so the alternate refuted SLSQP's exact
6/11 and its band point became the run's best. The guard's refutation rule was left alone: refutation can only fail
a correct result, never pass a wrong one, and telling a bought gain from a real one needs the multiplier. With a
clean threshold of 1e-6 of the band, a clean point can buy at most 2.98e-14 and 1.04e-12 on these two instances,
below the spec's 1e-8 QP tolerance. Restoring the band point (least-norm Gauss-Newton on the active constraints,
bounds held) makes it compete on what feasibility costs: the restored alternate answers 0.54545454545465, within
1e-13 of 6/11, and agrees.

## 3 设置

| 项目 | 价值 |
|---|---|
| 光谱摘要(未更改) | `sha256:e5104c340d419f388643bed2ad1de32662f8384f7b1c4adb7a05b90ad877db47` |
| 协议(`e14-10-campaign/2`) | `sha256:9cd233d3dc127da9765bc2b322d2480e5a49f2c5b8b61ff3652826ce4464ed8f` |
| 代码 | `sha256:d7a8ed51c21c5ba57e1ba08eb53b3ba2121850e706acbd90f21cd0354e6af5bb` |
| 冻结文件内容 | `sha256:4c4560dc73a5b72f8252aeaca0227c39c777e3bd82098d6a5239a73966c44841` |
| 冻结承诺 | `3311d09`,在任何得分的硬盘跑前犯;[`freeze.json`](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/results/e14-v2/freeze.json) |
| 调制干线运行 | 在冻结前,在未确定协议的 v 2 代码上: 120 规则臂调制单元格,均在预算范围内,打入并重播 EQUAL,无提醒;参数不变 (`PROTOCOL["tuning_adjustments"]`) . |
| 服务器 | `j7zang`;每个得分的硬币跑在主机`ubuntu2404-012`(v 1:`ubuntu2404-002`)上. |
| 环境 | `~/e11env` (Python 3.12, numpy, scipy, pygmo as in v1); nothing installed |
| 工作空间 | `~/e14-10-campaign-v2/`(代码为`code-3311d09/`,结果为`out/`,日志为`logs/`,发射器为`e14run.sh`,`e14jobs.sh`);v 1的`~/e14-10-campaign/`未动. |
| 任务 | 4 family processes plus the formulation shard, staggered by 20 s, threads pinned to 1, `nice 10`, as in v1 |
| 墙壁时间 | 7 min 42 s for 105 shards (05:13:46 to 05:21:28 UTC; v1: 9 min 10 s). The slowest shard, `cells.scoring.E-rule.knapsack`, took 155 s (v1: 180 s). The host's load average was 3 to 18, from other users. |
| 失败和重运行 | 无 |

笔记本电脑和服务器从同一个承诺中计算出相同的三个文摘,和相同的`analysis.json`内容文摘(`sha256:2fe70d6e85bb46b8c03380e77584a3b4407ad9a49156d3e0900df48ef86326f1`).

## 4 诉1 诉2 移动的单元格

E-rule rows differ from v1 in every cell, because each now carries its budget events. 9 of the 29 instances moved in
success, verification, typing or evaluations; the other 20 are equal on all five. A and B: 0 of 580 rows moved.

| 实例(E-规则,10种子) | 成功( spec rule) v 1 到 2 | 提醒 | 已核实 | 预算类型d | 评价中值 | 最低差距 |
|---|---|---|---|---|---|---|
| `cqp-projection-d2` | 0至10 | 10至0 | 0至0 | 10至10 | 725至735 | -2.98e-8至0 |
| `cqp-equality-d3` | 1至10 | 9至0 | 0至9 | 10至10 | 4816至4780 | -1.09e-6至2.2e-16 |
| `rosenbrock-disk-d2` | 10至10 | 0至0 | 10至10 | 6至10 | 3993至3842 | 0至0 |
| `ackley-d10-shifted` | 10至10 | 0 | 0至0(现`budget_exhausted`). | 10至10 | 17048至17048 | 1.2e-8 |
| `rastrigin-d10-shifted` | 0至0 | 0 | 0至0(现`budget_exhausted`). | 10至10 | 15946至15946 | 3.98 |
| `cqp-simplex-d5`, `cqp-simplex-d10`, `lp-wyndor-d2`, `knapsack-p01-lp-relaxation` | 10 各不变 | 0 | 10 各不变 | 10 | +1, +1, +3, +18 | 不变 |

限制这些数字的是什么:

- `cqp-projection-d2`成功但仍未核实. 它唯一的交替, 差异性进化,在
  2.00001(`tol = 0.01`),这是1e-5更糟糕的一个可行的点,警卫们正确地称之为没有定论.
- 第十颗`cqp-equality-d3`种子是`budget_exhausted`:它的候补在0.54552分配时停止,更糟糕的是.
  比6 /11,跑步说是这样.
- `rosenbrock-disk-d2`不再使用其全部预算(4000大多数为3992):适配器保留3(n+1).
  (c) 评估恢复情况,并在点清时不动用。 它的验证仍然停止
  自卫队以来,3的种子, 以及带有`alternate`状态的种子,`verified`
  接受被切掉的证据
- Ackley和Rastrigin的变迁已移到`budget_exhausted`, 因为他们的候补都停在了
  他们的拨款。 这是新的规则,而不是所发现的改变。
- 额外的评价(+1到+18)是恢复探测器:用在解点上的差点和步骤.
  超过乐队的1e-6.

## 5 每类结果

成功计数细胞(Intances x 10种子). 计分器现在将一个提醒算作没有成功,所以本表不需要更正. “预算耗尽”计算`stop`为`budget_exhausted:*`的细胞;对于E-规则,包括验证在分配时的交替停止的细胞(6.2节).

| 类 | 手臂 | 单元格 | 成绩 | 可行的 | 提醒 | 中位数差距 | 预算用尽 | 评价中值 | 中值墙s |
|---|---|---|---|---|---|---|---|---|---|
| ưμ㼯A | A级 | 20 | 0 | 20 | 0 | 5.18e-06 | 0 | 9.22e+03 | 0.409 |
| ưμ㼯A | B | 20 | 20 | 20 | 0 | 0 | 0 | 8 | 0.00127 |
| ưμ㼯A | E-规则 | 20 | 20 | 20 | 0 | 0 | 0 | 3.01e+03 | 0.246 |
| 对流 _ qp | A级 | 40 | 5 | 40 | 0 | 5.12e-08 | 5 | 4.78e+03 | 0.225 |
| 对流 _ qp | B | 40 | 40 | 40 | 0 | 1.37e-11 | 0 | 160 | 0.0207 |
| 对流 _ qp | E-规则 | 40 | **40**(v 1:21). | 40 | **0**(v 1:19). | 2.78e-16 | 1 | 3.95e+03 | 0.245 |
| 米尔普 | A级 | 40 | 33 | 40 | 0 | 0 | 0 | 4.02e+03 | 0.681 |
| 米尔普 | B | 40 | 40 | 40 | 0 | 0 | 0 | 15.5 | 0.011 |
| 米尔普 | E-规则 | 40 | 40 | 40 | 0 | 0 | 0 | 9.8e+03 | 0.782 |
| 无 | A级 | 40 | 36 | 40 | 0 | 1.04e-10 | 0 | 1.38e+04 | 0.451 |
| 无 | B | 40 | 40 | 40 | 0 | 6.43e-09 | 40 | 1.5e+04 | 0.386 |
| 无 | E-规则 | 40 | 40 | 40 | 0 | 3.41e-11 | 32(v 1:3). | 1.25e+04 | 0.627 |
| 全球 | A级 | 60 | 33 | 60 | 0 | 2.31e-08 | 0 | 1.82e+04 | 0.644 |
| 全球 | B | 60 | 33 | 60 | 0 | 2.31e-08 | 0 | 1.82e+04 | 0.687 |
| 全球 | E-规则 | 60 | 50 | 60 | 0 | 6.53e-09 | 56(v 1:0). | 1.65e+04 | 0.900 |
| 黑 _盒 | A级 | 60 | 53 | 60 | 0 | 0 | 0 | 319 | 0.0342 |
| 黑 _盒 | B | 60 | 60 | 60 | 0 | 0 | 0 | 104 | 0.0164 |
| 黑 _盒 | E-规则 | 60 | 60 | 60 | 0 | 0 | 0 | 2.38e+03 | 0.092 |
| 多 _目标 | A级 | 30 | 29 | 30 | 0 | 0.0178 | 0 | 9.41e+03 | 0.534 |
| 多 _目标 | B | 30 | 30 | 30 | 0 | 0.0105 | 30 | 1e+04 | 0.668 |
| 多 _目标 | E-规则 | 30 | 21 | 30 | 0 | 0.0151 | 0 | 3.41e+03 | 0.081 |

Spec对A,B,D和E的目标为成功=lp上的1.00,对流_ qp和milp. B和E-规则与它相符合; A想念它,如在1(差别演化停止在QP和LP容忍度之上).

Wall clock is 10 to 20 % lower than v1 for every arm, A and B included, whose work is bit for bit the same. That is
the host, not the code: v2 ran on another machine of the same pool.

### 比较

统计为1's(成功率的分组标志-flip,对数ERT比率的精确Wilcoxon,群靴CI,每个家庭Holm, spec's 裁定规则). 完整的表格在[`tables.md`](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/results/e14-v2/tables.md).

**Overall (29 instances).**

| 比较 | 结束点 | v1 | v 2效果 | 95 % CI | 资料 | pholm (m) (韩语). | v 2 判决 |
|---|---|---|---|---|---|---|---|
| B对A | 成功率差异 | +0.255,上级 | +0.255 | [0.19, 0.328] | 16 | 0.00043 (15) | 高级 |
| B对A | ERT比率 | 0.225,上级 | 0.225 | [0.168, 0.30] | 23 | 0.00039 (15) | 高级 |
| B对A | ERT-时间比 | 0.345,上级 | 0.346 | [0.294, 0.402] | 27 | 0.00053 (15) | 高级 |
| E-规则对As | 成功率差异 | +0.217,上级 | +0.283 | [0.193, 0.376] | 21 | 0.0083 (6) | 高级 |
| E-规则对As | ERT比率 | 1.59,未显示不同 | 1.46 | [1.23, 1.75] | 28 | 0.74 (6) | 未显示不同 |
| E-规则对As | ERT-时间比 | 1.07,未显示不同 | 0.993 | [0.838, 1.19] | 28 | 0.62 (6) | 未显示不同 |
| E-规则与Bs | 成功率差异 | -0.038,未显示不同 | +0.028 | [-0.035, 0.090] | 6 | 0.74 (6) | 未显示不同 |
| E-规则与Bs | ERT比率 | 11.3,低级 | 10.4 | [8.28, 13.1] | 28 | 0.026 (6) | 低级 |
| E-规则与Bs | ERT-时间比 | 4.92,低级 | 4.54 | [3.38, 6.02] | 28 | 0.017 (6) | 低级 |
| E-规则与Bs | 对数 10差距比 | -0.64,未显示不同 | -0.994 | [-1.89, -0.385] | 15 | 0.0053 (3) | 高级 |

The primary Holm family still holds the spec's 15 hypotheses, 12 of them not run (p = 1). E-rule comparisons are a
secondary family of 6, the gap endpoint a family of 3, as in v1. v2's E-rule vs B success effect equals v1's
sensitivity reading (+0.028), which is what the scorer counted then; the difference is that the alarm cells now
really are successes.

**每个分析组**(每个组的9测试的呼声):

| 组 | 比较 | 成功 diff, 判决 | ERT比例,判决. | ERT-时间比,判决. |
|---|---|---|---|---|
| 代数 (10) | B对A | +0.62,上级 | 0.005,上级 | 0.019,上级 |
| 代数 (10) | E-规则对As | +0.62 [0.44, 0.81], superior (v1: +0.43, not shown different) | 1.32,未显示不同 | 0.655,未显示不同 |
| 代数 (10) | E-规则与Bs | 0,不可检验(0信息). | 132 [90, 191], inferior (v1: 225) | 32 [15, 67], inferior |
| 连续 _ 非凸起( 10 ) | E-规则对As | +0.21,未显示不同 | 0.792 [0.716, 0.875], superior | 1.15,未显示不同 |
| 连续 _ 非凸起( 10 ) | E-规则与Bs | +0.17,不可测试 | 0.798 [0.739, 0.862], superior | 1.27,未显示不同 |
| 黑_盒(6),多_目标(3). | 全部 | 如 v 1 : 不可测试或显示不同 | | |

The algebraic E-rule vs B ERT ratio fell from 225 to 132 because ERT divides evaluations by the success rate, and
two QPs went from 0 and 0.1 to 1. E-rule still spends 132 times B's evaluations there: B solves an LP or QP in 8 to
160 evaluations and verifies nothing; E-rule's final verification runs alternates (section 7).

## 6. 安全标准

| 计量 | A级 | B | E-规则(v 1). | 目标 |
|---|---|---|---|---|
| 预算执行 | 290/290 | 290/290 | **290/290** (286/290) | 1.00 |
| 独立柜台=报告的评价 | 290/290 | 290/290 | 290/290 | |
| 大多数评价使用/预算 | 1.0 | 1.0 | 0.998 (1.0) | <= 1 |
| 实验重播 EQUAL | 290/290 | 290/290 | 290/290 | 1.00 |
| 保留了硬性限制 | 无 | 无 | 2,400/2,400 | 1.00 |
| 盘点已校验但错误 | 0 | 0 | 0 of 223 verified (0 of 214) | 0 |
| 空格低于负容忍度的单元格 | 0 | 0 | **0** (19) | 0 |
| 符号, USD | 0, 0 | 0, 0 | 0, 0 | 只报告 |

### 6.1 Budget typing, and what the 290/290 rests on

The spec's rule applies to a cell that used its budget. No E-rule cell did in v2 (the most was 0.998 of it), so for
E-rule the 290/290 is met without the rule's condition ever firing: that number alone is weak evidence. The stronger
fact is the typed field. 89 E-rule cells had a final-verification alternate stop at its allocation, and all 89 carry
an `alternate` event and the stop `budget_exhausted:evaluations`. 68 of them are still `verified` (the guard accepted
the evidence the cut alternate produced), 21 are `budget_exhausted`. 127 cells also have a planner solve that stopped
at its allowance, which is typed but leaves the status to the final verification. In v1 the same cuts happened and
only the alternate's own result recorded them. The unit tests pin the exhausted case directly
([`test_e14_05_budget_typing.py`](https://github.com/zangjiucheng/orbitopt/blob/main/tests/test_e14_05_budget_typing.py)), including an alternate that spends its
whole allocation, stages skipped for lack of evaluations, and a run that uses its whole iteration limit.

### 6.2 假验证 通行证:断层注射

| 路径 | 职等 | 假通行证 | 95 % 上限 | 名称 | 诚实主机通过 | 错传这些 | 上面写着: |
|---|---|---|---|---|---|---|---|
| 校验 _结果 | 1,008 | 0 | 0.30 % | 993 | 810(v 1:792). | 0 | 810 |
| 验证 _ 运行 | 1,008 | 0 | 0.30 % | 993 | 810(v 1:792). | 0 | 810 |

The 18 extra honest artifacts are on E-rule hosts (207 to 225 of 261): two E-rule seed-0 hosts that now pass, times
9 faults (inferred: the two QPs are the only E-rule hosts whose point changed from failing to correct). The one-sided
95 % upper bound on the false-pass rate over the 810 informative artifacts is 0.37 %.
The 15 cells per path where the fault's own reason was not named are the MO hosts, as in v1. Every fault was
rejected. With 870 scored rule-arm runs added, the denominator is 1,878 per path and the false-pass count is 0.

## 7  验证地位(E-规则).

| 运行状态 | 单元格 | 地点 |
|---|---|---|
| 已核实 | 223(v 1:214). | |
| budget_exhausted | 21(v 1:0). | `rastrigin-d10-shifted`(10,全错),`ackley-d10-shifted`(10,全好),`cqp-equality-d3`种子1 |
| 未验证的 _ 最佳 | 46(v 1:76). | the three scoring MO instances (30: NSGA-II has no alternate), `cqp-projection-d2` (10, all right), `integ-nearsing-1e-6`, `knapsack-p07-n15`, `knapsack-correlated-n20` (2 each) |

`optimality_certified` passed in 20 cells, as in v1. Every verified cell succeeded.

E-rule's evaluation cost is still its verification
([`erule_evaluation_split.jsonl`](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/results/e14-v2/erule_evaluation_split.jsonl), descriptive): on
`cqp-equality-d3` the loop spends 37 evaluations and the final verification 4,743 (median); on
`cqp-projection-d2` 12 and 723. A and B verify nothing.

## 8. 剩余的失败

| 手臂 | 实例 | 成绩 | 说明 |
|---|---|---|---|
| A,B,E-规则 | `rastrigin-d10-shifted` | 0/10 each | 没有手臂找到转动的盆地; E-规则的中位差 6.96 |
| A级 | `rastrigin-d20` | 0/10 | |
| A、B 类 | `rastrigin-d5` | 6/10 | |
| A级 | `cqp-projection-d2`, `cqp-simplex-d5`, `cqp-simplex-d10`, `lp-wyndor-d2`, `knapsack-p01-lp-relaxation` | 0/10 each | 几倍于容忍 |
| A级 | `cqp-equality-d3`, `knapsack-correlated-n20` | 5/10 each | |
| A级 | 合并(4例)、`rosenbrock-d5`、`rosenbrock-d20`、`ackley-d20`、`ackley-d10-shifted`、`knapsack-2d-n12`、`knapsack-p07-n15`、`mo-zdt2-d5` | 6 / 10 至 9 / 10 | |
| B | `ackley-d20` 9/10, `ackley-d10-shifted` 8/10 | | |
| E-规则 | `mo-zdt2-d5` 2/10, `mo-zdt1-d5` 9/10 | | NSGA-II的前方错过了5%在非对流ZDT2上的超容积耐力 |

Every A and B failure is v1's, row for row. E-rule lost no cell it had in v1.

## 9. 什么没有运行

| 项目 | 为什么 |
|---|---|
| 武器C、D、E(如`E-LLM`),`formulation_accuracy`,编译器`ir_schema_validity`,象征性费用 | 无提供者; 290 `not_run`各排C,D和E-LLM,原因`no provider` |
| 初级比较 C-A, D-A, E-D, E-B. | 每一条臂没有运行;p = 1 在15的Holm家族 |
| 外部 CUTEst 和 COCO 实例 | 未销售(第3节) |

提供者只来自环境。 通过这些变量的 OpenAI 兼容路径通过假本地端点进行测试,没有网络(`tests/test_e14_10_campaign.py`).

| 变量 | 含义 | 用于深搜索的示例 |
|---|---|---|
| `E14_LLM_KIND` | `openai_compat`(默认)或`local` | `openai_compat` |
| `E14_LLM_BASE_URL` | 结束点 | `https://api.deepseek.com/v1` |
| `E14_LLM_MODEL` | 模型 id,记录在每一个硬体的出处 | `deepseek-chat` |
| `E14_LLM_API_KEY` | `openai_compat` 的密钥; 在呼叫时读取, 从未存储 | 键 |

Historical names: these are the variables v2 read. From campaign v3 on, the campaign, the CLI and Mission
Studio read one set, `LLM_KIND`, `LLM_MODEL`, `LLM_BASE_URL` and `LLM_API_KEY`
([providers](https://github.com/zangjiucheng/orbitopt/blob/main/docs/agent/providers.md#configuration-from-the-environment)); the `E14_LLM_*` names are gone.

管理提供武器意味着重新运行`cells.*.C.*`,`cells.*.D.*`,`cells.*.E-LLM.*`和`formulation.scoring`的硬盘. 提供者更改不冷冻摘要,但已经写成`not_run`的软体必须用手去掉或写到一个新的输出目录.

## 10. 对有效性的威胁

- **No LLM.** Nothing here measures what a model adds.
- **E-rule is not arm E.** It runs on the hand-written problem and cannot fail at formulation.
- **Verification is charged to E-rule only.** The cost endpoints compare an answer with a verified answer.
- **恢复(英語:Recovery)是解决者的一种改变,而不只是排名.** 每一个E- 规则单元格在一个受限制的连续
  问题的解决者预算(储备金)略有不同。 测量效应:9事件移走,无
  worse in success or verification; evaluations rose by 1 to 18 on four instances and fell on `rosenbrock-disk-d2`.
- **The clean threshold is a choice.** `SELECTION_FRACTION = 1e-6` of the band is set so that a clean point's bought
  gain is far below every tolerance in the catalogue for multipliers up to about 1e3. A problem with a much larger
  乘数仍然可以让一个干净的点比它的客观容忍更能买到。
- **警卫仍然可以反驳一个准确的结果 带点不能恢复**(替换为无)
  evaluations left). That is a false fail, never a false pass; none occurred here.
- **E-规则的预算打字在Spec本身的条件下是部分空闲的.**(6.1节).
- **Runner choices, the one v1 tuning change, tolerance edges, small groups, memorisation and one shared host**:
  如[v1, section 8](https://github.com/zangjiucheng/orbitopt/blob/main/docs/benchmarks/e14_qualification_report.md#8-threats-to-validity). 主持人是另一台相同的机器
  池,在 3 到 18 从其他用户负载下.

## 11. 数据和完整性

| 文件 | 地点 | 摘要 |
|---|---|---|
| 105硬文件 (4.5 MB). | 服务器, `~/e14-10-campaign-v2/out/shards/` | 在[`manifest.json`](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/results/e14-v2/manifest.json) |
| E-规则运行目录(7.9 MB). | 服务器, `~/e14-10-campaign-v2/out/runs/`和 `~/e14-10-campaign-v2/runs-3311d09.tar` | 焦土 `sha256:d977345fe1c78dc8c4276adaeb03d16ee615623dfab2599a6d60c9c4d8514492` |
| CSV | [`cells.csv`](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/results/e14-v2/cells.csv) | `sha256:1201b57cc025f5f4cef73323ccc05bb0e3b112b5d71587ac0cde71af9f9bd053`(在清单中) |
| 分析 | [`analysis.json`](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/results/e14-v2/analysis.json), [`tables.md`](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/results/e14-v2/tables.md) | 内容摘要 `sha256:2fe70d6e85bb46b8c03380e77584a3b4407ad9a49156d3e0900df48ef86326f1` |
| 冻结 | [`freeze.json`](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/results/e14-v2/freeze.json) | 3节 |
| 调制干线运行( 未得分) | 服务器, `~/e14-10-campaign-v2/out-dryrun/`(代码为 `code-dryrun/`) | |

The copy-back was `rsync` of `out/` without `runs/` to the laptop, then `verify`, which re-hashed every file the
manifest names: all matched. Nothing on the server was deleted.

## 12 复制

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

两种修正的回归测试, 没有服务器, 也没有 模型 :

```bash
PYTHONPATH=src python -m pytest -q tests/test_e14_05_budget_typing.py tests/test_e14_05_candidate_selection.py
```
