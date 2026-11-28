# E14-10资格报告:竞选诉3

这是[E14-10 benchmark specification](e14_benchmark_spec.md)的首个得分赛跑,使用语言模型. 它对DeepSeek的`deepseek-flash`运行了三个提供武器:arm C(模型提出点数,没有解析器),arm D(LLM编译器加固定臂-A解析器)和arm E作为`E-LLM`(运行时LLM编译器加LLM规划器). 该公司还经营配方套件,并在同样的冻结中经营不需要模型(A、B、`E-rule`)的三件武器。 它所遵循的计划是[campaign v3 plan](e14_llm_campaign_plan.md).[Campaign v1](e14_qualification_report.md)和[campaign v2](e14_qualification_report_v2.md)保持记录. 结果在[`benchmarks/results/e14-v3/`](https://github.com/zangjiucheng/orbitopt/tree/main/benchmarks/results/e14-v3/).

**Read this first.** One model, one provider, one day, temperature 0. The 10 seeds of a provider cell send the same
request, so they measure the provider's run-to-run variation and the solver's seed, not 10 independent samples of a
model. The unit of every claim is the instance (29) or the formulation case (58), not the cell.

## 1. 问答

Spec询问LLM问题编译器和迭代代理在传统基线上是否有帮助,每个问题类(Spec section 0). 五个主要比较是B-A,C-A,D-A,E-D和E-B,E为E-LLM.

- **编译器从未产生过它在得分实例上接受的错误问题.** 每一个400 编译
  武器D和E-LLM 相当于探测点的参考实例 所以每个硬约束和束缚
  被保留(2,100 of 2,100). 所有有参考文献的280配方编译都相当于它.
  Schema validity was 1.00 (870 of 870 for D, 290 of 290 for E-LLM).
- **It refused two whole classes.** It called every integration statement (6 instances) and every bi-objective
  statement (3 instances) unsupported, on every seed, in D and in E-LLM: 90 of 290 cells per arm. The formulation
  套房显示同样的拒绝。
- **Formulation accuracy is 0.690 (95 % CI 0.603 to 0.776), below the spec's 0.95.** Nine of the 58 cases expect an
  只有脚本测试答复才能产生的结果。 没有他们,0.816(0.714至0.898),仍然在下面.
- **D vs A: inferior on success** (-0.283, Holm p 0.031). All of it comes from the compile step: 90 refusals and 10
  单元格,其中编译的问题具有一个未绑定的变量,而臂D的解析器无法接受。 19实例
  that compiled and solved, D equals A cell for cell in success (190 of 190 cells), with an ERT ratio of 1.00.
- **E-LLM vs D: superior on success** (+0.252, CI 0.159 to 0.345, Holm p 0.037). Both use the same compiler, so this
  是运行时间精确的解析器 击打差分进化,而不是 模型 。
- **E-LLM vs B: not shown different on success** (-0.286, Holm p 0.070), **inferior on evaluations (11.1 times) and
  wall clock (91.7 times).**
- **LLM计划者反对规则规划者(E-LLM对E-规则,冻结后增加).**低于成功
  (-0.314) because of the 90 refusals and one instance, `ackley-d10-shifted` (E-LLM 0 of 10, E-rule 10 of 10). On the
  20编译实例 成功差异为-0.05与1信息实例(不可验证),E-LLM是
  11.8在墙钟中速度较慢. 策划者没有帮助。
- **C vs A: not shown different on success** (-0.134, Holm p 0.80). C wins the textbook global functions and loses
  背包、整合和双重目标 它的胜负不是搜索:在转动的10-D Rastrigin和Acleyy上
  提出从语句中切换矢量作为其第一点,并在1评价后停止.
- **Safety mostly holds.** 0 false passes in 1,008 injected faults per verifier path. Replay EQUAL on all 1,740
  得分的单元格和所有 580 编译 。 两次错过:**3 E-LLM运行经过验证,而36.8运行从最佳状态.**(当地人)
  **72臂-C细胞通过200,000-托肯预算.**(最高为19.7%),所以C's
  budget enforcement is 218 of 290.
- **Spend: $3.33 at list price** for 3,786 calls and 23.5 M tokens, under the approved $5.00 hard stop. Every
  反应名为 `deepseek-flash` 。

Where the LLM helped: nowhere in a pre-registered comparison. Where it did not: every LLM arm had a lower success
rate than the non-LLM arm it was compared with (C and D against A, E-LLM against B and E-rule), through refusals for
D and E-LLM, and the LLM arms took 3 to 92 times the non-LLM arms' wall clock. The compiler's accepted output was
always right, which is the strongest positive result here.

## 2 设置

| 项目 | 价值 |
|---|---|
| 光谱摘要(自 v 1 起不变) | `sha256:e5104c340d419f388643bed2ad1de32662f8384f7b1c4adb7a05b90ad877db47` |
| 协议(`e14-10-campaign/3`) | `sha256:ad381579aafcf84a55ac525052406cbff5005e22a55bd14ebc493dab8b72518a` |
| 代码 | `sha256:9f4ed53b1c00380a372f1aa93854bada820c2929d12f0a99f5e2dc271ddd42f1` |
| 冻结文件内容 | `sha256:7caa7d0255edc82202b4d7e6e22a378e5ce30ad269c97e71fd961b597d3699fd`,[`freeze.json`](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/results/e14-v3/freeze.json),承诺`2c34539` |
| 已运出树 | `git archive 7661026`(冻结加计划文档);`plan`在服务器上打印了上面四个摘要 |
| 即时摘要 | 编译器系统 `sha256:8c5290c3...`,编译器计划指令 `sha256:1722591e...`,计划系统 `sha256:955502fd...`,计划指令 `sha256:1ed30d7d...`;等于`freeze.json`'s `protocol.llm.prompt_digests`,在第一次通话前在服务器上检查过. |
| 提供者 | `deepseek`, model `deepseek-flash`, `https://api.deepseek.com/chat/completions`, temperature 0, thinking disabled, no seed sent, `json_object` with the schema in the system instruction and one repair turn |
| 每个细胞盖 | calls C 20, D 3, E-LLM 20; 200,000 tokens per cell |
| 价格表 | per million tokens: cache hit $0.006, cache miss $0.30, output $1.20 (peak list price, read 2026-11-28) |
| 硬停车 | `--max-llm-usd 5 --max-llm-calls 14210`,每个过程一个分类账 |
| 服务器 | `j7zang`, every shard on host `ubuntu2404-008`, `~/e11env` (Python 3.12.13, scipy 1.18.1, numpy 2.5.5) |
| 工作空间 | `~/e14-10-campaign-v3/` (code `code-7661026/`, results `out/`, logs `logs/`); v1 and v2 workspaces untouched |
| 发射器 | [`launchers/`](https://github.com/zangjiucheng/orbitopt/tree/main/benchmarks/results/e14-v3/launchers/):`e14run.sh`(计划,加上没有密钥文件和停止标记的拒绝),`e14jobs.sh`(计划,以硬站作为论据,并重试一个阵亡的运行),`e14watch.sh`(模型 - 名称监督),`e14status.py`和`e14project.py`(引导阅读和飞行员投影). |
| 任务 | formulation shard alone; then the pilot (3 processes); then 4 family processes, staggered by 20 s, threads pinned to 1, `nice 10` |
| 墙壁时间 | 06:45:04 to 08:07:03 UTC on 2026-11-28, 1 h 22 min. Model latency summed over calls was 2 h 13 min. |

**The key.** It lived in the laptop's `.env` and reached the server only as `~/.llm_env`, written with `umask 077`
(mode 600, checked), and was sourced inside the launcher. The launcher then exported the frozen kind, model and base
URL over the file's own. It refused to start without the file, so no provider shard could be written as
`not_run: no provider` by accident. After the last provider shard the file was removed with `shred -u` and checked
absent. No file in the server workspace or in the committed results contains the key value; the check ran `grep -F`
with the key read from standard input. The 1,450 cassettes contain no path, host or user name; their prompts are
benchmark problem text only.

## 3. 情况如何:阶段和分类账

运行分三个阶段进行,分类账分别读完后. 额定在5以下的舞台封顶,

| 阶段 | 或 | 电话 | 符号 | 缓存打击共享 | USD(列表) | 分类账后 |
|---|---|---|---|---|---|---|
| 1,上限1.00 | 配方套件: 58病例×10种子 | 586 | 1,210,333 | 0.883 | 0.2241 | $0.2241 |
| 2,上限0.85 | 飞行员:C、D和E-LLM、120细胞的罗森布洛克碎片 | 440 | 2,657,324 | 0.853 | 0.3957 | $0.6198 |
| 3,上限5.00 | 每一个其他的积分 硬,规则的臂和缺陷包括 | 2,760 | 19,657,463 | 0.873 | 2.7144 | $3.3342 |
| **total** | 57 积分硬点 | **3,786** | **23,525,120** | 0.871 | **3.3342** | |

缓存- hit 共享是缓存- hit point spords 覆盖所有 point spords 。 分类账的结算总额等于各行自用总和的调用,代币和百分率.

- **The pilot gate.** From the pilot's finished cells the projection was $3.76 (C's 250 remaining cells at $0.0102
  每一个,其维数5或以下的单元格的平均值,其两个比率中越高; D, 0.00104美元和E-LLM美元
  每个囚室的0.00264,他们最亲爱的例子). 那是在4.00的门下 所以3跑步 结果为3.33美元。
  C在背包和双目标成本 大约0.02每个细胞, 三倍的罗森布洛克率, 保守的D
  和E-LLM补偿它。
- **Against the Phase A estimate.** The plan's expected scenario was $1.63, its high one $11.88. C cost $2.45
  against an expected $0.80: it used 2,012 calls (6.9 per cell) where the dry run saw 2 to 7.
- **Billing.** The ledger prices every token at the peak list price. Every call was on a Saturday, which the
  pricing page read in Phase A puts off peak at half price, so the bill is probably about $1.67. That is inferred;
  该账户的账单没有读。
- **Calls and attempts.** 3,786 calls took 3,786 HTTP attempts: no 429, no 5xx, no retry, no account error, no
  暂停 。 3,786的每个答复都命名为`deepseek-flash`;一个会阻止任何运行的监视者.
  其他名字从来没有被开除。
- **Things that went wrong.** The first launch of stage 3 passed each job's two family names as one word (the
  服务器的登录 shell 是 zsh, 它不会分割一个未引用的变量 。 四份工作没有硬的,写着
  没有,没有打过电话,就离开了0;重启后将名字分开。 运行很晚,我的ssh会 开始着陆
  在另一个主机的池(`ubuntu2404-004`)上,所以`ubuntu2404-008`上的流程通过日志跟踪.
  在共享的家。 没有Shard失败,没有工作重试,没有硬停止解雇。
- **Order.** The protocol's jobs rule runs one family's arms in turn (A, B, E-rule, C, D, E-LLM). The staging ran the
  配方硬和三个罗森布洛克的供货商 硬,然后四个家庭工作,
  软糖。 只有墙钟才能依赖这个顺序.

## 4. 什么跑什么没跑

| 项目 | 单元格 | 说明 |
|---|---|---|
| A,B,E-规则 | 290 each | A和B:所有580行文摘等于2. E-规则:除了计时场和`artifact_digest`以外,所有290行在每一字段均等于2,该规则将一连串的跑量表持墙时钟. |
| C级 | 290 | 200结束于模型的停机旗上,72被象征性预算削减,18结束于修复转弯后未能验证的回复(其中4没有评估点). |
| D级 | 290 | 190编译和解析,10编译但解析器中出现错误(`lp-wyndor-d2`,第8节),90以不支持为由拒绝;1编译被第二次调用. |
| E-LLM | 290 | 200编译并运行运行时间,90以不支持为由拒绝. |
| 措辞 | 580编译 | 5接了一个以上的电话(6额外电话). |
| 错误 | 1,008每验证路径 | 与2 |
| 调色板 | 不运行 | 根据计划:`--split scoring`省略120调试细胞,称为模型. |

没有手机是`not_run`. 没有手机打到竞选站 由每个细胞代号预算剪切的72 C细胞被键入`budget_exhausted:llm_tokens`,并被评为失败,如协议所说.

## 5 每类结果

成功数细胞。 差距已经正常化。 "预算耗尽"计算出`stop`为`budget_exhausted:*`的细胞. 墙钟包括LLM latency;对于E-LLM,它添加了该行留下的编译间隔(11节).

| 类 | 手臂 | 单元格 | 成绩 | 可行的 | 中位数差距 | 预算用尽 | 评价中值 | 中值墙s |
|---|---|---|---|---|---|---|---|---|
| ưμ㼯A | A级 | 20 | 0 | 20 | 5.18e-06 | 0 | 9,220 | 0.408 |
| ưμ㼯A | B | 20 | 20 | 20 | 0 | 0 | 8 | 0.00125 |
| ưμ㼯A | C级 | 20 | 5 | 20 | 1.1e-07 | 6 | 118 | 26.7 |
| ưμ㼯A | D级 | 20 | 0 | 10 | 3.46e-07 | 0 | 8,660 | 2.78 |
| ưμ㼯A | E-LLM | 20 | **20** | 20 | 0 | 0 | 2,750 | 4.48 |
| ưμ㼯A | E-规则 | 20 | 20 | 20 | 0 | 0 | 3,010 | 0.24 |
| 对流 _ qp | A级 | 40 | 5 | 40 | 5.12e-08 | 5 | 4,780 | 0.219 |
| 对流 _ qp | B | 40 | 40 | 40 | 1.37e-11 | 0 | 160 | 0.0207 |
| 对流 _ qp | C级 | 40 | 38 | 40 | 2.78e-17 | 0 | 16.5 | 3.35 |
| 对流 _ qp | D级 | 40 | 5 | 40 | 5.12e-08 | 5 | 4,780 | 1.8 |
| 对流 _ qp | E-LLM | 40 | **40** | 40 | 1.39e-17 | 1 | 3,940 | 3.78 |
| 对流 _ qp | E-规则 | 40 | 40 | 40 | 2.78e-16 | 1 | 3,950 | 0.231 |
| 米尔普 | A级 | 40 | 33 | 40 | 0 | 0 | 4,020 | 0.685 |
| 米尔普 | B | 40 | 40 | 40 | 0 | 0 | 15.5 | 0.0116 |
| 米尔普 | C级 | 40 | 5 | 35 | 0.0981 | 28 | 48.5 | 37.6 |
| 米尔普 | D级 | 40 | 33 | 40 | 0 | 0 | 4,020 | 4.34 |
| 米尔普 | E-LLM | 40 | **40** | 40 | 0 | 0 | 9,800 | 6.23 |
| 米尔普 | E-规则 | 40 | 40 | 40 | 0 | 0 | 9,800 | 0.768 |
| 无 | A级 | 40 | 36 | 40 | 1.04e-10 | 0 | 13,800 | 0.475 |
| 无 | B | 40 | 40 | 40 | 6.43e-09 | 40 | 15,000 | 0.39 |
| 无 | C级 | 40 | 32 | 40 | 0 | 8 | 40 | 7.12 |
| 无 | D级 | 40 | 36 | 40 | 1.31e-10 | 0 | 13,800 | 2.32 |
| 无 | E-LLM | 40 | 40 | 40 | 2.38e-11 | 30 | 8,170 | 7.28 |
| 无 | E-规则 | 40 | 40 | 40 | 3.41e-11 | 32 | 12,500 | 0.629 |
| 全球 | A级 | 60 | 33 | 60 | 2.31e-08 | 0 | 18,200 | 0.643 |
| 全球 | B | 60 | 33 | 60 | 2.31e-08 | 0 | 18,200 | 0.645 |
| 全球 | C级 | 60 | **56** | 59 | 4.44e-16 | 3 | 21 | 9.03 |
| 全球 | D级 | 60 | 33 | 60 | 2.23e-08 | 0 | 18,200 | 3.91 |
| 全球 | E-LLM | 60 | 40 | 60 | 4.44e-16 | 57 | 19,100 | 10.1 |
| 全球 | E-规则 | 60 | 50 | 60 | 6.53e-09 | 56 | 16,500 | 0.891 |
| 黑 _盒 | A级 | 60 | 53 | 60 | 0 | 0 | 319 | 0.0363 |
| 黑 _盒 | B | 60 | 60 | 60 | 0 | 0 | 104 | 0.0167 |
| 黑 _盒 | C级 | 60 | 10 | 16 | 0 | 1 | 21.5 | 6.49 |
| 黑 _盒 | D级 | 60 | 0 | 0 | inf | 0 | 0 | 2.14 |
| 黑 _盒 | E-LLM | 60 | 0 | 0 | inf | 0 | 0 | 2.09 |
| 黑 _盒 | E-规则 | 60 | 60 | 60 | 0 | 0 | 2,380 | 0.095 |
| 多 _目标 | A级 | 30 | 29 | 30 | 0.0178 | 0 | 9,410 | 0.504 |
| 多 _目标 | B | 30 | 30 | 30 | 0.0105 | 30 | 10,000 | 0.633 |
| 多 _目标 | C级 | 30 | 4 | 30 | 0.2 | 26 | 505 | 35.7 |
| 多 _目标 | D级 | 30 | 0 | 0 | 0.667 | 0 | 0 | 1.26 |
| 多 _目标 | E-LLM | 30 | 0 | 0 | 0.667 | 0 | 0 | 1.27 |
| 多 _目标 | E-规则 | 30 | 21 | 30 | 0.0151 | 0 | 3,410 | 0.0808 |

任何手臂的细胞都没有发出警报(低于容积的缺口)。 与每个比较的完整表格为[`tables.md`](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/results/e14-v3/tables.md).

**光谱的绝对目标**在 lp, convex _ qp和milp上是成功的1.00为A,B,D和E. **E-LLM遇见它**(100 of 100),如B和E-规则. **错过了**(38 of 100),原因与A相同:它的解法是A的差分演化,它仅止于LP和QP的容忍度之上. 在 D 的 lp 列中, 20 单元格中的 10 是 `lp-wyndor-d2` 解析器错误.

限制这些行的是什么:

- D's and E-LLM's black _ box and multi _ 客观行都是拒绝. 它们测量编译器的覆盖范围,而不是
  解脱者.
- C的全局行是分析知识,而不是优化(8.1节).
- 每个类都有1到6例. 以下按类别提出的索赔是按分析组提出的,例如:v 1和v 2。

## 6 比较

Statistics as in v1 and v2: success by the cluster sign-flip test on per-instance discordance, cost endpoints by the
exact Wilcoxon test on per-instance log ERT ratios, cluster bootstrap CIs (10,000 resamples, seed 1410, strata =
family), Holm per family, and the spec's verdict rule. Effects are treatment minus control (success) or
treatment over control (ERT, ERT-time); the gap effect is the mean log10 ratio of median gaps.

### 6.1 初级家族(Spec's 15假说,全部运行)

| 比较 | success diff [95 % CI] | 霍尔姆 | 判决 | ERT 比率 [ CI ] | 霍尔姆 | 判决 | ERT- 时间比 [ CI ] | 霍尔姆 | 判决 |
|---|---|---|---|---|---|---|---|---|---|
| B对A | +0.255 [0.190, 0.328] | 0.00037 | 高级 | 0.225 [0.168, 0.30] | 0.00034 | 高级 | 0.346 [0.293, 0.402] | 0.0011 | 高级 |
| C对A | -0.134 [-0.238, -0.038] | 0.80 | 未显示不同 | 0.0284 [0.0138, 0.0646] | 0.80 | 未显示不同 | 63.2 [28.8, 136] | 0.42 | 未显示不同 |
| D对A | -0.283 [-0.300, -0.259] | 0.031 | **inferior** | 1.00 [1.00, 1.00] | 0.022 | 未显示不同 | 3.20 [2.37, 4.36] | 7.2e-06 | **inferior** |
| E-LLM vs D | +0.252 [0.159, 0.345] | 0.037 | **superior** | 0.954 [0.827, 1.09] | 0.20 | 未显示不同 | 1.39 [1.24, 1.57] | 0.83 | 未显示不同 |
| E-LLM vs B | -0.286 [-0.352, -0.221] | 0.070 | 未显示不同 | 11.1 [8.86, 13.7] | 0.0023 | **inferior** | 91.7 [65.7, 131] | 6.5e-05 | **inferior** |

读行:

- **C vs A.** The CI on success excludes 0, but the sign-flip test does not reject (raw p 0.30). C's discordance is
  它在两个方向上都很大:它赢得了每一个教科书的全球实例,失去了背包、整合和
  bi-objective. The spec's rule needs both, so "not shown different". The ERT ratio of 0.028 counts C's 1 to 20
  对从报表中删除的事例进行评价。
- **D vs A, ERT.** The Holm p (0.022) is below 0.05 but comes from the 9 instances only A solved; on the instances
  both solved the ratio is 1.00 to three figures (3 of them differ at all). The CI is [1, 1], so the verdict is "not
  显示不同".
- **E-LLM vs D** 隔离运行时间:双臂以相同的即时编译,然后 D 运行不同的进化和
  E-LLM运行计划员在解析器注册. 在E-LLM的73额外成功中,62在lp,convex _ qp和milp,
  运行时间有精确的解析器;其他的11在nlp和Global上.

The gap endpoint is a secondary family of the 7 tested overall comparisons: D vs A inferior (+0.202 in log10, Holm
p 0.040), E-LLM vs D superior (-1.95, Holm p 0.031), E-rule vs A and E-rule vs B superior as in v2; C vs A, E-LLM vs B
and B vs A not shown different.

### 6.2 Secondary and exploratory families

| 比较 | 家庭 | 成功 diff | 判决 | ERT比率 | 判决 | ERT-时间比 | 判决 |
|---|---|---|---|---|---|---|---|
| E-规则对As | 中学(6) | +0.283 [0.193, 0.376] | 高级 | 1.46 | 未显示不同 | 1.04 | 未显示不同 |
| E-规则与Bs | 中学(6) | +0.028 [-0.035, 0.090] | 未显示不同 | 10.4 | 低级 | 4.63 | 低级 |
| E-LLM对E-规则 | 探险(4),在冻结后增加 | -0.314 [-0.379, -0.252] | **inferior** | 0.819 [0.788, 0.863] | "超级",见下文 | 11.8 [7.87, 17.7] | **inferior** |

E-规则行等于2',但墙钟是主机的(v 2:0.993和4.54).

**The E-LLM vs E-rule ERT verdict is an artifact.** The frozen verdict joins a Wilcoxon p over all 29 instances with
a CI over the finite ratios only. Here the p (raw 0.013) comes from the 10 instances where E-LLM never succeeded and
E-rule did: they rank as infinitely worse for E-LLM. The CI comes from the 19 others, where E-LLM used fewer
evaluations (geometric mean 0.82, most of it `lp-wyndor-d2` at 0.04). The two halves point in opposite directions and
the rule reads "superior". The analysis now records which side the signed ranks favour
(`wilcoxon_direction`, "treatment worse" here). It is the only decided verdict in the campaign where the side and the
verdict disagree. On the 20 compiled instances alone the same endpoint is not shown different (Holm p 1).

### 6.3 Per analysis group

对每个组的21测试(7比较×3端点);探索性比较是自己的家族.

| 组 | 比较 | 结果 |
|---|---|---|
| 代数 (10) | E-LLM vs D | 成功 +0.62 [ 0.44 0.81 ],在霍尔姆之后没有显示不同; ERT 1.32和ERT-时间1.10没有显示不同. |
| 代数 (10) | E-LLM vs B | success 0 (no informative instance); ERT 95.9 and ERT-time 469, inferior |
| 代数 (10) | E-LLM对E-规则 | 成功 0; ERT-时间 15.6,低等 |
| 连续 _ 非凸起( 10 ) | C对A | ERT 0.0044,优异;差距-4.9,优异;成功+0.19,不显示不同. |
| 连续 _ 非凸起( 10 ) | E-LLM对E-规则 | 成功 -0.10(1信息实例,不可考); ERT-时间 8.7,劣等 |
| 黑_盒(6),多_目标(3). | 每个LLM臂 | 不显示不同或不可测试; 每个D室和E-LLM室都是拒绝的 |

C在连续_非Convenx中的两个决定胜负是8.1的污染案例. 它们不是提出点数可以克服差异演变的证据。

### 6.4 Post hoc: the 20 instances every D and E-LLM cell compiled

描述性,一个霍尔姆家族16测试. 这将编译器拒绝的内容与之后发生的事情区分开来.

| 比较 | 成功 diff | ERT比率 | ERT-时间比 |
|---|---|---|---|
| D对A | 0(0信息). | 1.00(3信息). | 3.20,低级 |
| E-LLM vs D | +0.365 [0.235, 0.500], Holm p 0.068 | 0.924 | 1.75 |
| E-LLM vs B | +0.035(4信息,不可检验). | 11.1 | 91.7,低级 |
| E-LLM对E-规则 | -0.05(1信息,不可检验). | 0.819,未显示不同 | 11.8,低级 |

在它编译的内容上,D是 A + 大约 2 → 每个单元格的编译延迟, E-LLM是 E- 规则,对一些实例的评价较少,一个实例丢失了,一个数量级的钟更多.

## 7. 配方精度

580编译:58 病例×10 种子,在臂D的帽下. 如果汇编的结果等于案件的预期结果,则正确;就汇编而言,问题相当于参考(在容忍和1,000探测点的相同可行性判断范围内,同样是最佳的)。 CIs是针对案例、层层语言x预期结果的组群拖网。

| 切点 | 个案 | 正确的编译 | 准确性 | 95 % CI |
|---|---|---|---|---|
| **overall** | 58 | 400 of 580 | **0.690** | [0.603, 0.776] |
| 联合国 | 50 | 340 of 500 | 0.680 | [0.58, 0.76] |
| 中国语 | 8 | 60 of 80 | 0.750 | [0.50, 0.875] |
| 已支持(预期`compile`) | 34 | 280 of 340 | 0.824 | [0.676, 0.941] |
| 不支持 | 6 | 50 of 60 | 0.833 | [0.50, 1.00] |
| 模棱两可(预期`needs_clarification`) | 8 | 70 of 80 | 0.875 | [0.625, 1.00] |
| 预期 `rejected_unauthorized_change` | 5 | 0 of 50 | 0 | |
| 预期 `compile_failed` | 5 | 0 of 50 | 0 | |
| 英语支持 / 不支持 / 模糊 | 29 / 5 / 7 | 240 / 40 / 60 | 0.828 / 0.80 / 0.857 | |
| 中文支持 / 不支持 / 模糊 | 5 / 1 / 1 | 40 / 10 / 10 | 0.80 / 1.00 / 1.00 | |
| 没有9 答复确定的案件(事后特设) | 49 | 400 of 490 | 0.816 | [0.714, 0.898] |

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

**模型出了什么问题**(除FC-039外,所有10种子的每个案例都表现相同):

| 大小写 | 预期 | 模型的结果 | (引用,缩短) | 检查器 |
|---|---|---|---|---|
| FC-005,FC-022(合并). | 编译 | 不支持, 10, 10 | “选择四重规则和点数......需要比较候选人规则中的错误估计......(一个离散的Selec [ tion ])”;“现有工具代号(数字.gauss _ legendre.v 1,数字适应_ simpson.v 1,数字.quadrature _ cost.v 1)不能证明[ e ]............” | 结果与预期不同 |
| FC-023,FC-024,FC-Z04(双目标). | 编译 | 不支持, 10, 10 | "请求中要求达到50非主导点(a Pareto front / set of solution),这是一种多解决方案的列举,这种单解决方案优化格式无法表达". | 结果不同; IR确实有两个目标 |
| FC-004,FC-Z08(在所附日志中放松) | 拒绝 _ 未经批准的 _ 更改 | 编译, 10, 10各 | 20编译中的每一集都保留`x + y >= 1`,没有授权更改:附件被忽略,没有标记 | 结果不同; 编译的问题是一个安全的问题 |
| FC-036(随机需求) | needs_clarification | 不支持, 10 10 | "需求是随机的,这需要一个有弹性的模型;这种格式不能表达随机或期望. 应否用决定性价值取代需求?" | 结果不同; 模型 在另一种状态下问到正确的问题 |
| FC-039 (彩色) | 不支持 | _ 澄清 6,编译 4 | 在所有 10 种子上,该回复将 `colour` 编译为离散变量,带有选择 [ 0 , 1 , 2 ] :"将绝对的颜色编码为红,绿,蓝色作为离散选择 0 , 1 , 2 ,因为格式需要数字选择";在 6上,它还询问如何编码颜色,编译器将颜色转换为 `needs_clarification` . | 4 用发明的编码编译接受 |
| FC-031,FC-041至FC-043(修订版). | 编译或拒绝 | _ 澄清,10 分别10 | "现在没有问题,所以我无法知道什么应该更快解决" | 答复确定 |
| FC-044至FC-048 | 编译 _ 失败 | 编译, 10, 10各 | 编译所陈述的问题;案例没有参考文献 IR,所以是未检查对等(FC-044 's,手读,与其语句相符) | 答复确定 |

FC-039的四编译器是唯一被接受的更严格的读取会称为错误的输出. 颜色不在成本(2 t)中,所以其最佳效果是正确的;编码被说成是一种假设.

**Seeds.** The 10 seeds of a case send the same HTTP body (the seed is recorded but not sent). The reply text
differed between seeds on 57 of the 58 cases, yet the outcome was the same on all 10 seeds for 57 of 58. The same
holds for the scored arms: the first reply's text differed across seeds on 28 of 29 instances for C and D and 27 of
29 for E-LLM. Temperature 0 is not deterministic on this provider; the outcomes mostly are.

## 8. 失败,有例子

### 8.1 Arm C knows the textbook functions

C succeeded on 10 of 10 seeds of `rastrigin-d10-shifted` and of `ackley-d10-shifted`. Every other arm scored 0 of 10
on the Rastrigin one. The statement gives the shift:

> Find the global minimum of the Rastrigin function 10n + sum_i (z_i^2 - 10 cos(2 pi z_i)) in 10 dimensions, where
> z_i = x_i - s_i with s = [0.7, -1.3, 2.1, -0.4, 1.8, -2.6, 0.3, 1.1, -0.9, 2.4], each x_i between -5.12 and 5.12.

第一次答复建议`[0.7, -1.3, 2.1, -0.4, 1.8, -2.6, 0.3, 1.1, -0.9, 2.4]`,第二次停止。 中间值是1对10-D两个实例的评价,19.5对21三个20-D实例的评价。 这是从定义中读取的教科书函数的最小值。 录音带在[`cassettes-sample/`](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/results/e14-v3/cassettes-sample/cells.scoring.C.rastrigin/C/rastrigin-d10-shifted/seed-0.cassette.jsonl)

Where C had to search it did badly: knapsack 5 of 50, bi-objective 4 of 30, integration 10 of 60 (in 44 of the 60
integration cells its best point was infeasible). 72 of its cells used more than the 200,000-token budget, 18 ended on
a reply that failed schema validation twice.

### 8.2 武装D

- **拒绝**(90单元格):六集和三个双目标语句,每个种子,如7节.
- **`lp-wyndor-d2`,10 错误10**:`ValueError: variable x1: the fixed solver needs finite bounds`. 声明
  states nonnegativity and no upper bound; the model compiled exactly that (the reference adds a box [0, 100]). The
  编译是正确的;臂D的固定解析器,差分进化,需要一个有限的框. E-LLM,在同一编译,
  用精确的 LP 解析器解析所有 10 。
- 其他任何地方 D 等于 A 成功,单元格(190 of 190). 评估数字与25不同
  单元格,在 6 实例和 106 的返回点上,以数量表示,该数值不会改变成功。 推断:汇编
  表达式是另一个浮点顺序中的参考物,所以在同一种子下不同的演化
  轻轻漂移.

### 8.3 装甲E-LLM

- **拒绝**:与D相同90细胞.
- **`ackley-d10-shifted`,0,10**(E-规则10,10):规划者要求第二次差别化运行地点.
  规则策划者要求第二次局部抛光(Solver list of the track,例如种子0:DE,L-BFGS-B,DE,L-BFGS-B).
  对DE,L-BFGS-B,L-BFGS-B. 每个种子都从所花的评价预算的最佳结果结束了1.6到1.7.
- **`rastrigin-d10-shifted`, 0 of 10**, as E-rule. Three of these runs were verified (section 9).
- Elsewhere 10 of 10 on every compiled instance.

## 9. 安全标准

| 计量 | A级 | B | C级 | D级 | E-LLM | E-规则 | 目标 |
|---|---|---|---|---|---|---|---|
| 预算执行 | 290/290 | 290/290 | **218/290** | 290/290 | 290/290 | 290/290 | 1.00 |
| 独立柜台=报告的评价 | 290/290 | 290/290 | 290/290 | 290/290 | 290/290 | 290/290 | |
| 实验重播 EQUAL | 290/290 | 290/290 | 290/290 | 290/290 | 290/290 | 290/290 | 1.00 |
| 保留了硬性限制 | 无 | 无 | 无 | 2,100/2,100 | 2,100/2,100 | 2,400/2,400 | 1.00 |
| IR 计划有效性 | | | | 870/870 | 290/290 | | 1.00 |
| 已核实的守护者运行得分 | 0 | 0 | 0 | 0 | 165 | 223 | |
| 经核实和不可行(Spec's fulse pass) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **verified but not within tolerance** | | | | | **3** | 0 | |

配制套件在580的580上重播EQUAL编译. 重新播放其磁带中的每个单元格,没有模型呼叫; 它表明记录的运行可以从文物中复制,而不是DeepSeek会再次以同样的方式回答(7,种子节).

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

## 10 成本,代币,耐用和模型

| 手臂 | 单元格 | 电话 | 符号在 | 其中缓存点击数 | 发号施令 | USD(列表) | USD 每个单元格 | USD 成功率 | LLM 中位数 | 中值墙s |
|---|---|---|---|---|---|---|---|---|---|---|
| C级 | 290 | 2,012 | 18,572,844 | 0.893 | 1,462,249 | 2.4483 | 0.0084 | 0.0163(150成功). | 9.62 | 9.66 |
| D级 | 290 | 291 | 537,354 | 0.896 | 143,724 | 0.1921 | 0.00066 | 0.0018 (107) | 2.06 | 2.30 |
| E-LLM | 290 | 897 | 1,364,302 | 0.551 | 234,314 | 0.4696 | 0.0016 | 0.0026 (180); **0.0029 per verified success** (162) | 4.40 | 4.88 |
| 措辞 | 580 | 586 | 1,059,298 | 0.883 | 151,035 | 0.2241 | 0.00039 | 0.00056(400正确) | | 1.33 |

- **模型返回**:所有3,786响应的`deepseek-flash`(排在`llm_models`,分类账和磁带)
  同意这一点。
- **Where the money went**: C took 73 % of the spend. It resends the whole conversation each turn; its cache share
  高,因为前缀重复。
- **Most calls in a cell**: C 20 (the cap), E-LLM 8 (compile plus planner), D 2.
- **Model latency summed over calls**: C 5,090 s, E-LLM 1,390 s, formulation 833 s, D 644 s.
- **墙钟是模型空位.** C的中位壁钟(9.66 s)是其LLM的空位(9.62 s). B 解析 lp 在
  1.3 ms;D花约2 s编译.

## 11 发现缺陷,没有固定

每个固定会改变被消化的代码,所以没有做:一个固定意味着重新冻结,这是用户的呼叫. 没有一个需要重新运行才能被正确读取,因为磁带和行都持有需要的东西.

| 缺陷 | 地点 | 效果在这里 | 报告如何处理 |
|---|---|---|---|
| E-LLM的`wall_clock_s`只为运行时间的倍数,不使用编译调用 | `bench_e14_end_to_end.run_cell_e_llm` | E-LLM的壁时钟按其编译时速排列(中间值2.0每单元格) | the analysis adds each cell's compile latency from its cassette (`llm_latency.jsonl`). With the row's own value: E-LLM vs B 60.0 instead of 91.7, vs E-rule 7.81 instead of 11.8, vs D 1.10 instead of 1.39; no verdict changes |
| C的象征性预算在电话之前检查,然后执行 | `CellMeter`通过`UsageMeter` | 72 超预算单位,最高为19.7% | 已报告; 如协议所述, 单元格为失败 |
| D的固定解析器需要一个有限的框 | `bench_e14_formulation.adapt` | 正确编译中的 10 错误 | 报告为 D 失败; 可以说是臂定义的限度,而不是 模型 |
| nine formulation cases expect outcomes only scripted replies produce | `tests/data/agent/formulation_cases.json` 携有直播带 | 将0.845的冷冻精度封顶,以便完美模型 | 后临时行没有它们 |
| 警卫核实了当地的最低人数 | 关于多式联运问题的运行时间 验证 | 3 校验时错误E-LLM 运行 | 9节 |
| 判决规则可以加入对面的p和CI | `e14_analysis.compare`(摘要外) | 一个判决误导 | 目前,每一判决的旁边都有记录(6.2节)。 |

编译器拒绝集成和双重目标问题,是对这个模型的发现与这个迅速,而不是代码中发现的缺陷. 快速更改是否修复它们尚未测试 。

## 12. 对有效性的威胁

- **One model, one provider, one day.** Nothing here generalises to another model, or to `deepseek-flash` on another
  日报. 提供者在运行期间没有改变任何可见的行为,但其输出因相同的呼叫而异.
- **API nondeterminism, and what the seeds mean.** Identical requests got different reply text on nearly every
  举个例子。 结果是稳定的,但是一个提供细胞的10种子 重复绘制一个迅速,而不是10
  提示 每一个试验组 逐个实例或案例 为此原因。
- **卡盘是复制品** Replay EQUAL表示其录制的单元格可以复制. 重运行
  对API可以不同;协议事先如此说.
- **沾染诸法.** Rastrigin,罗森布洛克和Acley都是公开的,有已知的密闭式最小器,而转动
  变体表示移位. C直接利用了这一点(8.1节)。 也可以通过了解这些来帮助编译器
  函数; 其汇编被检查为等效, 这样帮助就不会产生错误的公认问题。
  背包事件(Burkardt P01, P07)也是公开的,
- **Tuning on 8 instances.** The rule arms' choices were tuned on the 8 tuning instances in v1. The provider arms'
  prompts and caps were exercised by a 30-call dry run on those instances, seed 1000, and nothing was changed after
  这个 拒绝书已经在那里可见(`mo-schaffer-d1`,`integ-exp-1e-10`). 得分后没有调音
  call. Eight instances cannot show how a prompt generalises; the scoring set is where the refusals were measured.
- **The price snapshot.** USD is at the list price read on 2026-11-28 from a page with no date. The bill is probably
  (非峰值),推论和未检查.
- **Order effects.** The formulation and pilot shards ran before the family jobs, against the protocol's jobs order.
  只有壁时钟可以依赖于顺序;除计时以外的行不能.
- **The formulation suite was written for scripted replies.** Nine of its cases cannot be met live (section 7); its
  被忽略附件(FC-004,FC-Z08)的预期结果会惩罚一个安全汇编。
- **核查员的"核实"并不是最优化的.** 科9的三行显示,它可以认证一个本地最小值.
- **Hard-constraint preservation is measured on compiled formulations only.** A refused statement preserves nothing
  - 90每只手臂都没有任何证据。
- **墙钟比较网络服务与本地解析器**,与其他用户的负载共享主机上.
- **飞行员是罗森布洛克家族**,C最便宜. 门维持住了,但是它的投影显示C在
  背包和双目标 大约三次。

## 13. 结论

On these 29 instances, with `deepseek-flash` at temperature 0:

- **The LLM compiler is safe and narrow.** It accepted no wrong problem on a scored instance and lost no constraint,
  它的输出总是有效的。 它拒绝合并和双重目标声明,拒绝,
  而不是错误的编译, 是什么使 D 低于 A 和 E-LLM 低于 E- 规则 关于成功。 制剂准确性
  (0.690) is well below the spec's 0.95 target, and below it even without the nine cases a live model cannot meet.
- **The LLM planner did not help.** On the instances that compiled it matched the rule planner on success except
  例如,它损失了,使用的评价略少一些,将8比12乘以墙钟,并制作了
  竞选只是被证实的失误
- **The LLM alone (arm C) is not an optimiser here.** It looks strong only where it can read the answer off a
  教科书定义,它失败了背包,整合和多目标问题.
- **The one superior primary result with an LLM arm, E-LLM over D, measures the runtime, not the model.**
- **B,古典精确的基线, 仍然是击败的手臂**,没有LLM手臂在任何端点上击败它.

样本中不支持什么:任何关于其他模型或提示,任何对黑色_盒或多_目标(超出"编译者拒绝")的每级索赔,以及任何关于规划者一般情况更糟糕的声称(一个实例带有其成功缺陷).

## 14. 数据和完整性

| 文件 | 地点 | 摘要 |
|---|---|---|
| 57 scoring shards (5.5 MB) | 服务器 `~/e14-10-campaign-v3/out/shards/` | 在[`manifest.json`](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/results/e14-v3/manifest.json) |
| 公式硬化 | [`formulation.scoring.json`](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/results/e14-v3/formulation.scoring.json) | `sha256:e1210203...d2289b`,等于表单 |
| 每盒磁带( 1,450) 并运行目录 | 服务器 `~/e14-10-campaign-v3/runs-7661026.tar` (69.5 MB) | `sha256:8589fc41f00868ba384f5340e62559253aac930e0d8180de0eca864ded7f4fe7` |
| eight sample cassettes | [`cassettes-sample/`](https://github.com/zangjiucheng/orbitopt/tree/main/benchmarks/results/e14-v3/cassettes-sample/) | [`artifacts.json`](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/results/e14-v3/artifacts.json),对照服务器副本检查 |
| 分类账(7,572行) | 服务器 `out/llm_ledger.jsonl`; 已执行 gzipped | `sha256:60800a8a...880042` |
| LLM 潜伏 | [`llm_latency.jsonl`](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/results/e14-v3/llm_latency.jsonl) | in `artifacts.json` |
| CSV | [`cells.csv`](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/results/e14-v3/cells.csv) | 在仪表上 |
| 分析 | [`analysis.json`](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/results/e14-v3/analysis.json), [`tables.md`](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/results/e14-v3/tables.md) | 内容摘要 `sha256:f65eed7833ac076c818f6475cb589781ea97fa8fcf371889fbd1a4740c022b01`, 笔记本电脑和服务器上相同 |

The copy-back was `rsync` of `out/` without `runs/`, then `verify`, which re-hashed every file the manifest names:
all matched. Nothing on the server was deleted except the key file.

**冻结后增加的分析**(在[`e14_analysis.py`](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/e14_analysis.py),代码摘要外,更改无行):E-LLM与E-规则比对为自己的家族;IR对光谱全分母的计程有效性(arm D的单元格编译加入到套房中);配制与案例分组的CI和后特设行的配制分解,没有回复确定的案例;每臂LLM核算;E-LLM壁点校正;Wilcoxon侧;6.4节的后特设比较.

## 15 复制

分析,没有模型. `v3/`是服务器的`out/`复制的,没有`runs/`(57 shards,`manifest.json`,`cells.csv`),加上`llm_latency.jsonl`从`benchmarks/results/e14-v3/`复制:

```bash
PYTHONPATH=src python benchmarks/bench_e14_end_to_end.py verify --out v3     # re-hash against the manifest
PYTHONPATH=src python benchmarks/bench_e14_end_to_end.py analyze --out v3    # analysis.json, content digest above
PYTHONPATH=src python benchmarks/bench_e14_end_to_end.py report --out v3     # tables.md
```

从磁带里找到的提供细胞 没有模型的电话 Untar `runs-7661026.tar`进入新鲜的`repro/`, 保留硬盘的`runs/`重新运行, 用占位键和零呼叫的硬站运行,

```bash
LLM_KIND=deepseek LLM_MODEL=deepseek-flash LLM_BASE_URL=https://api.deepseek.com LLM_API_KEY=placeholder \
PYTHONPATH=src python benchmarks/bench_e14_end_to_end.py run --shard cells.scoring.D.linear_program \
    --shard cells.scoring.E-LLM.linear_program --out repro --max-llm-calls 0 --max-llm-usd 0
```

Done on the server for those two shards: 0 calls in the ledger, the 10 D rows equal to the scored rows digest for
digest, the 10 E-LLM rows equal outside `artifact_digest` (a manifest of files that hold wall-clock times).

正如[plan](https://github.com/zangjiucheng/orbitopt/blob/main/docs/benchmarks/e14_llm_campaign_plan.md#5-phase-b-after-approval)所说,

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

然后`collect`和`verify`像v 2, 以及磁带上的休息表,

```bash
PYTHONPATH=src python benchmarks/e14_analysis.py --llm-latency ../out > ../out/llm_latency.jsonl
```
