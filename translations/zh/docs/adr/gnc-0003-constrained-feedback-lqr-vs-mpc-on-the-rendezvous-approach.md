# ADR gnc -0003:受限制的反馈——LQR对MPC关于会合方式(E13-05).

- **Status:** Accepted
- **Date:** 2026-11-27
- **Deciders:** E13 Closed-Loop GNC & Relative Navigation (Epic #416), E13-05 (#422)
- **Code:** `src/orbitopt/gnc/lqr.py`, `src/orbitopt/gnc/mpc.py`, `src/orbitopt/gnc/constraints.py`,
  `src/orbitopt/rendezvous/constrained_campaign.py`, `benchmarks/bench_e13_05_constrained_control.py`
- **Evidence:** `tests/test_e13_05_constrained_control.py`,
  `docs/benchmarks/e13_05_constrained_control.{json,md}`
- **Builds on:** ADR gnc-0001 (contracts/clock), gnc-0002 (EKF seam), and the E13-04 closed-loop
  `docs/benchmarks/e13_04_closed_loop_guidance.md`记录的基线

## 背景情况

E13-04将CW中程重定向定律放入闭环中,并起作用(258/264在LEO-RDV-001阻截中). E13-05问下一个问题:*排程形状的反馈是否正确?* 实施了两个标准替换,并比较了相同的接缝——相同的真理物理,相同的EKF和感知堆,相同的计划出发燃烧,相同的扰动和执行图,同样的分级——仅在反馈法中有所不同:

- **LQR** (`CwLqrLaw`): the frozen CW LQR gain from the shared Riccati sequence, saturating at the
  脉冲盖,完全没有限制模型.
- **MPC**(`CwMpcLaw`):12 - 步(120 s) 折返-视距缩合QP 在同一CW冲动上.
  模型,由于任务的安全限制*内部* 优化器——每脉冲推力 帽,
  剩余 XQv 预算(烟雾化的 `sum|u|` 代用), 保留球体, 跨轨道走廊, 终端
  视锥,和测距指数的视锥速度上限;终端费用按同一价格计价
  Riccati序列,所以一个短的视野仍然重视尾巴.

在任何方法的比较是公平的之前,实验迫使三个结构决定;它们和数字一样是ADR的物质.

## 决定

1. **The terminal pair is scheduled, not law-dependent.** A single 3-DOF impulse cannot null both
   position and velocity: measured directly on the seam, the best capped burn 27 s from terminal
   留下~148 m的剩余值,因为速度重量(容忍度合同,0.2 m/s对50 m)
   正确的支配。 双臂(仅通过相同的`ManeuverPlan`对基准时间表)
   搭载: trim at t _ f− 40 s / 3 m/s 车顶/速度重量 0.25 抵达 at _ f− 20 s / 8 m/s 车顶/s
   速度重 100)在固定时间点点燃 `scheduled_burns`;反馈取代基线的
   MID-COURSE只重新瞄准目标. 法律在自己的预算门内 保留了未着火的火帽
   (`budget_available_m_s`),所以反馈勾选永远无法拉动终端对. 这使得
   比较一个干净的*中级法律*比较,把每个手臂都放在任务内部
   机动合同。
2. **Feedback tracks the sample's own planned arc, not the hold point.** Both laws regulate
   `ρ − ρ_ref(t)`与样本实现的计划转移(同一弧E13-04苍蝇),所以a
   中途法是根据它如何遵守所下达的任务计划以及制约来判断的
   计划将车辆放置在什么地方,对边距进行评估。
3. **MPC falls back to the LQR gain, never to coast.** A failed or non-finite SLSQP solve issues the
   冻结的再增加步骤,并增加基准报告的一个计数器。 指导性法则
   滴答使特派团付出了不止一个 冒着一个滴答边际的风险
4. **预测是线性 - CW 在QP**——研究说明的设计假设:a
   120的地平线 J2/drag相对效应为mm–cm,在500的下方的两个命令保持不动.
   约束很在意 如果未来的任务缩小了这一差距,预测模型(而不是QP).
   是该替换的东西。

## 测量(完整的264 - 样本搁置,相同的图纸;见JSON)

| 手臂 | 成绩 | 威尔逊 95% | 终端有药量(米) | Δv 医疗(m/s) | 实际存在的侵犯家庭行为 | 决策数表示 |
|---|---|---|---|---|---|---|
| 基线 | 258/264 | [0.951, 0.990] | 42.1 | 11.38 | 0 | 0.011 |
| LQR | 260/264 | [0.962, 0.994] | 45.8 | 11.50 | 0 | 0.36 |
| MPC | 255/264 | [0.936, 0.982] | 45.4 | 15.97 | 0 | 23.4 |

Paired exact-McNemar (Holm): neither arm's success endpoint separates from
the baseline (LQR 3↔1, p = 0.625; MPC 3↔6, p = 0.508; both violation
endpoints 0↔0). Failure *classes* move: baseline loses 6 samples to the
campaign safety checks, MPC to capped-arrival misses (6 `arrival_accuracy`,
worst 176 m) caused by its 6.05 m/s mean mid-course spend eroding the
terminal reserve; the four realized families held in every arm, with MPC
flying tightest to the walls it can see (speed +0.07 m/s, cone +9.8 m).

## 后果

- **Applicability.** LQR is the right default when the plan's constraint margins are wide and the
  机上电脑不是:每个决定是~0.36 ms,闭式,然后到达.
  260 / 264 与它更换的时间表相同. MPC在某处挣钱
  the mission's own families bind — it is the only arm whose optimizer *knows* the keep-out,
  走廊、圆锥和速度上限,而且零实际违反规定(14 036倒计时,15.7%的决定,
  表示23 ms/ decision – 仅以实时容量的QP解析器和10 s支付.
  cadence).
- **The speed family is where a schedule-shaped law and a constraint-aware law can disagree about
  feasibility**: the plan itself flies inside +2.5 m/s of its ceiling at the closest approach, and
  早期使用比值的反馈法 根本就没有了 MPC的视野看到那堵墙
  来了 LQR在滴滴发现它发生。
- **QQV不是自由的,也不是歧视者**:25预算内的两块军火地
  MPC的中途成本溢价来自反冲,以克服自身的制约
  成型,可见于文物中的每阶段的QQv表.
- **两个臂部都没有实质性地改变终端统计** — 错误是终端- pair- 有限
  (decision 1), so method choice is about constraint handling and compute, not arrival accuracy, on
  这个任务。
- 基准中的身份保护人员在与基准臂发生任何分歧时重新显示并中止
  冰冻的E13-04文物;三臂号码只是因为护卫跑步而可以比较的.

## 拒绝的替代品

- **法律范围内的终极燃烧**(每个臂在窗口端发射的共享重靶):被拒绝——
  它使武器在任务无法比拟的维度上有所不同,一个3 - DOF明显燃烧.
  无法关闭两个宽容渠道。
- **MPC在解析失败时掉回海岸:**拒绝——在圆锥内500米处滴入滴滴.
  最坏的情况是,制约是用来防止的。
- **控制控点,而不是计划弧:**拒绝了——它会给法律分级.
  against a trajectory the mission never flies and hide plan-constraint interactions.
- **Hard `sum|u|` budget constraint in the QP:**拒绝支持平滑代孕
  (eps = 1 mm/s ≪ 执行错误)——非smooth版本需要NLP解码器,否则无法操作.
  增减。
