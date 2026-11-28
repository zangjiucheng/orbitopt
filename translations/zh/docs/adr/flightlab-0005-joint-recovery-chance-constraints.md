# ADR 飞行实验室 -0005:联合机缘限制恢复规划(E3-02).

- **Status:** Accepted
- **Date:** 2026-11-14
- **Deciders:** Recovery Guidance & Robustness (Epic #217), E3-02 (#259),
  follow-up to R-01 (ADR flightlab-0003) and R-07 (epic #217 stack audit)
- **Code:** `src/orbitopt/flightlab/recovery/chance.py` (`JointChanceSpec`,
  `evaluate_joint_chance`, `joint_penalty`),
  `src/orbitopt/flightlab/recovery/boostback.py` (`plan_boostback_entry(joint_chance=...)`,
  `_evaluate`, `CorridorReport`), `src/orbitopt/flightlab/recovery/montecarlo.py`
  (样本记录`landing_probability`/`joint_violation`).
- **Evidence:** `docs/benchmarks/e3_02_joint_recovery.json`
  (`benchmarks/bench_e3_02_joint_recovery.py`),
  `tests/test_flightlab_recovery_chance.py`,
  `docs/benchmarks/r07_nominal_dispersed.json`(这个 ADR 修正的回归)

## 背景和问题

R-01(ADR飞行实验室-0003)将走廊交接的两个与着陆有关的后果——转移的落地燃烧必须关闭,它继承的推进剂边距——移入Boostback目标为*软加权数*(`landing_divert_penalty = (miss/budget)^2`,`reserve_shortfall^2`). R-07再将R-01堆放在申报的起动器着陆堆上,并测量了两样东西(Epic # 217审计):

* 堆叠的列是 *项目名称相同* 到 R-01 单—— 额外
  坚固层在软条件之上买不到任何东西;
* 着陆散射 **倒退**:最大触地半径 338(FL-305)
  • 833(R-01堆)。

这两个症状都有同样的根源:加权总和交易. 惩罚权重是一种价格,而不是一项要求——只要加权金额说交易所盈利,而*人口数*散开是可见的账单,Opimiser高兴地将惩罚用于购买差额(或反向). 任务(E3-02)要求三种回收量——燃料比值,转移可行性,着陆概率——进入**作为制约,或机会制约, 而不是惩罚权重调**计划.

## 所审议的备选方案

1. **重新调整 R-01 重量**( 较大的分流, 较大的储备)
   体重). 拒绝:将贸易结构R-07仅计量为
   原因; 没有原则性方法来选择权重; R-07 的零增益堆栈
   说家人累坏了
2. **FL-302内部的硬可行性限制**(拒绝不合格)
   推出,例如`NonlinearConstraint`下的`minimize`. 拒绝:从
   不可行的温暖开始 Nelder – Mead simplex 可以看到一个被拒绝的高原
   零梯度信息的顶点和横跨走廊
   条件本身在计划矢量中是不连续的;被接受的计划
   停止作为推出到推出的比较,即`CorridorReport`输
   它的意义。 (同理FL-302 已经用来对付硬地形/
   走廊大门。 )
3. **Joint chance constraints with a deterministic equivalent and an exact
   penalty** (adopted, below): each requirement becomes a probability
   从 *记录* 散射波段、
   概率为总和 *无交易*(最坏的规范化)
   违反,一个lQQ结构。
   惩罚在每一可交易术语之外加一个尺度——没有重量。

## 决定

`orbitopt.flightlab.recovery.chance`定义了联合要求设置的`JointChanceSpec`和`plan_boostback_entry(..., joint_chance=...)`选择模式(它*替换*,两个R-01软名词——规划者拒绝构建一种两者都会重复计算的模式,报告将R-01字段`None`保留在联合模式中):

* **Fuel margin, chance form.** The handoff margin must cover the p-quantile
  推进剂在移交时仍有风险的散失:
  `margin ≥ Φ⁻¹(p_fuel) · σ_loss(exposure)`, with σ_loss = exposure-weighted
  1σ of the recorded FL-304 dispersion bands (thrust 5 %, Isp 3 %, density
  10 %, C_d 8 %; Gaussian-composed) applied to the propellant *at risk*
  (already burned + landing need). Default `p_fuel = 0.9`, σ floor 40 m.
  在名义上的20 t暴露时,需要+1.03 t-R-01的手选.
  1000公斤设计点,现在得出而不是断言.
* **Divert feasibility, scatter-derived.** The corridor divert bar is not a
  自由键,但来源于接受指标本身:要求
  Rayleigh触地得分的q-quantile(与闭路距离相同)
  measurements, σ = σ₀ + gain·divert, σ₀ floor 40 m, gain 0.5 — the ADR
  -0003表,0.35千米转弯在~0.4米处着陆,1三椭圆
  1 m)封顶*预测*的分流位置.
  `(budget/√(−2·ln(1−q)) − σ₀)/gain` ≈ **852 m** for budget 1000 m at
  q = 0.9,交接的雷利报道 `P(radius ≤ budget)`是
  记录为`p_divert`.
* **Landing probability floor.** `p_land = p_fuel · p_divert ≥ 0.8`
  (默认),接受要求的联合声明.
* **Aggregation without trading.** The three normalized violations enter the
  目标为带l∞(最恶劣违反)的`JOINT_PENALTY_SCALE · max_i g_i`
  结构及精确的处罚等级为25——高于
  走廊差错/影响带(25)和任何可信的基期差额——
  因此,选取者无法以偏移或反向方式购买边际:它必须修正
  是最恶劣的违反者 `JointChanceSpec`将两个判决记录在
  `CorridorReport` (`landing_probability`, `joint_violation`), which the
  Monte Carlo的每个样本记录, 所以尾巴的推出 带有自己的
  E3-03的概率声明.

**Defaults are design assumptions pinned to recorded evidence** (documented
as `DESIGN ASSUMPTION` in the spec docstring): p_fuel 0.9 ≈ R-01's
validated 1 t design point; the scatter budget 1000 m is the E3 acceptance
scale (well inside the 2.5 km landing tolerance); the Rayleigh parameters
are fitted on ADR flightlab-0003's landing table. Nothing is fitted from the
evidence set below, so the comparison stays honest.

## 后果(n = 32,种子1,对联,20 deg/s TVC 钉在每一臂上).

`bench_e3_02_joint_recovery.py`在相同的上升段/散射(JSON的每条样板行;以构造对齐)上飞行三种客观模式:

| 手臂 | 成功(维尔森95%) | touchdown radius p50 / p90 / max | 1 + 椭圆 |
|---|---|---|---|
| fl 305 _基线 | 0.406 [0.255, 0.577] | 0.9 / 295 / 436 m | 1.2 × 0.7 m |
| r01_landing_aware | 0.500 [0.336, 0.664] | 1.0 / 182 / 638 m | 1.1 × 0.7 m |
| **e3_02_joint_chance** | **0.531 [0.364, 0.691]** | **0.9 / 118 / 628 m** | **1.1 × 0.6 m** |

* **Do-no-harm, exactly:** the 13 samples both FL-305 and the joint arm land
  有 **不变** 触地得分( paired Δ = 0.0 m) —— 联合模式
  只有在约束约束的情况下,才与基线不同。
* **Paired flips:** +4 landings (samples 1, 19, 24, 28), −0 (McNemar 4–0,
  p = 0.125 at n = 32 – E3-03 较大的基准结算意义;
  跑道上最差的走廊差值增加-1461千克-448千克.
* **R-07的账单:** 喜欢R-01 联合
  臂最大半径(628 m)超过基线(436 m),整体
  空隙为*选择效果*:最大辐射样本(idx 18)是其中之一.
  基线从未着陆(`INSUFFICIENT_PROPELLANT_RESERVE`;
  进入了联合臂的着陆点 在628米, 关闭容忍, 准确的说,
  R-01的手臂确实做了——而联合手臂的最大长度在R-01的638米以下.
  在降落时,联合臂的散射是三部曲中最好的
  (p90 118 m vs 295/182 m; ellipse 1.1×0.6 m). The constraint set converts
  推进剂储备损失进入边际着陆——更好的故障模式
  必须进行认证,交易E3-03现在必须进行统计确认.
* **Reason-code signature:** `INSUFFICIENT_PROPELLANT_RESERVE` 8 → 1 while
  `CORRIDOR_MISSED` 1 → 3: the quantile constraint rejects handoffs whose
  *预测*差值是薄的*在此之前* 燃烧消耗储备.
* 起动器限制在武器之间保持不变(声明20 deg/s TVC pin,
  E3-01一致性)——规划员切换是唯一的区别.

## 展望

E3-03在其统计样本(n≥64)上重新运行了这个基准,并消耗`paired_rows`进行每样测试;配对-flip计数和着陆-条件散数,而不是原始最大半径,是引用的数字. 如果E3-03确认,R-01的软条件成为倒置模式,`joint_chance`可以成为默认目标.

## Amendment (E7-01, task #277): penalized optimization, enforced verdict

The method adopted above is a **penalized optimization** — Nelder–Mead on a
worst-violation exact penalty — not a constrained optimization, and the E3-02
benchmark exposed the gap that left: 26 of 32 planned joints carried
`joint_violation > 0` (p50 0.19, p90 1.03, max 2.8), ten of them with
`p_land ≥ 0.8` yet still busting fuel/divert, while `ReturnPlan.feasible`
said nothing about the joint constraints and reported them satisfied.

E7-01明确解决命名问题——API和docs说*受罚优化*(这里的ADR文本,`chance.py`模块docstring,FL-302计划者docstring)——并关闭语义:

* `JointChanceSpec`搭载`violation_tolerance`(默认1e-6,a)
  数字层,不是贸易节点)和`JointChanceSpec.satisfies(worst)`
  定义满意:`worst_violation <= violation_tolerance`.
* FL-302的可行性裁定现在包括了联合制约:a
  被接受的交还,其最恶劣的正常侵权行为超过容忍
  回来`feasible=False` 理由代码**`JOINT_CHANCE_VIOLATED`**
  (failure class `planning_infeasible` in the R-06 classification), checked
  在硬走廊/推进器/入口限制之后,所以规范代码
  两次休息时保持物理状态。

因此,`joint_violation > tolerance`计划永远不能被标记为满足机会限制,在规划者的裁决,蒙特卡洛理性直方图,或者失败阶级崩溃.
