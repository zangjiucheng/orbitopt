# E13-05 — 限制控制研究:LQR / MPC 与基线,在悬架上对齐(任务# 422 ).

Epic #416. This task asks what a *different mid-course feedback law* buys on
the closed-loop seam E13-04 built (`orbitopt.rendezvous.gnc_campaign`): an
unconstrained CW-LQR arm and a constraint-in-the-QP MPC arm
(`orbitopt.gnc.lqr`, `orbitopt.gnc.mpc`, flown by
`orbitopt.rendezvous.constrained_campaign`), compared against the E13-04
baseline on **identical** perturbation and execution draws. The machine
reading is [`e13_05_constrained_control.json`](https://github.com/zangjiucheng/orbitopt/blob/main/docs/benchmarks/e13_05_constrained_control.json);
reproduce with `PYTHONPATH=src python
benchmarks/bench_e13_05_constrained_control.py --jobs 16 -o …` (20 min wall at 24 jobs on a 64 - 核心主机,仅CPU). 基线臂通过过路定时包装器和跑步**终止**重流,除非它复制了被冻结的E13-04文物样本逐个样本(它确实:精确于所有264).

## 实验

The same LEO-RDV-001 holdout as E13-04: 22 clusters × 12 samples, seed base
20 000 000, identical perturbations, ground plan (tof 3416.67 s, planned
10.345 m/s), tolerances (50 m / 0.2 m/s) and the untouched 25 m/s budget
(`guards.budget_unchanged: true`). Each sample flies three arms that share
the departure burn, the EKF/sensing stack, the plant and the grading:

* **baseline** — the E13-04 schedule (mid-course retarget at t₁ + 0.5·tof,
  终端修剪,抵达;
* **坐标**——冻结的CW-LQR在补偿估算上的增益,规范.
  样品在1/s脉冲盖上饱和;
* **mpc** - 12 - 步(120 s) 折返-视距缩合 QP 相对于同一 CW
  冲动模型 与任务的限制 家属*输入* 问题
  (保持出行500米,跨轨100米,圆锥形,射程指数速度上限),
  推力上限,其余的XΔv预算为平滑`sum|u|`
  代用; 终端费用 = LQR 从共享的 Riccati 费用到
  顺序。 解答失败回落到LQR增益,计算.

Both new arms fire the mission's own terminal pair (`ManeuverPlan`s at t_f −
40 s and t_f − 20 s, caps 3 / 8 m/s) as *scheduled* burns and reserve those
caps inside their budget gate — so the arms differ only in the mid-course
phase (ADR gnc-0003, decision 1).

## 数字怎么说的?

| 手臂 | 成绩 | 威尔逊 95% | 终端 pos (med/max) m | Δv(中/最大)米/秒 | 实际存在的侵犯家庭行为 | decision ms (mean/p95) |
|---|---|---|---|---|---|---|
| 基线 | 258/264 | [0.951, 0.990] | 42.1 / 82.0 | 11.38 / 12.82 | 0 | 0.011 / 0.006 |
| 坐标 | 260/264 | [0.962, 0.994] | 45.8 / 50.2 | 11.50 / 12.16 | 0 | 0.36 / 0.43 |
| mpc | 255/264 | [0.936, 0.982] | 45.4 / 176.2 | 15.97 / 24.92 | 0 | 23.4 / 48.7 |

* **Paired tests**: none of the four McNemar comparisons is significant
  (lqr success 3↔1 discordant, p = 0.625; mpc 3↔6, p = 0.508; violation
  endpoints 0↔0, p = 1.0; Holm-adjusted p = 1.0 throughout). On this mission
  *法律形状*不移动成功率;终端对和
  预算政策占主导地位。
* **失败的班级移动,虽然。** 基线的6损失是:
  `safety_violation` (campaign trajectory checks); LQR's 4 and MPC's 9 are
  3 `safety_violation` + 1 / 6 `arrival_accuracy` MPC最差到达者.
  (176 m,1.0 m/s)共享一种机制:其中途食欲.
  (fedback 6.05 m/s表示LQR的1.07) 花在终端-pair上.
  预备队,而封顶的8/s到达燃烧无法再完成任务.
  约束意识 QQV-饥饿 终极风险 – 与 14 036 倒背勾
  at a 15.7 % rate across the campaign's ~89 000 MPC decisions.
* **The four realized constraint families (keep-out, cone, cross-track,
  speed) held in every sample of every arm.** The families' worst flown
  边距排在预期位置: MPC最紧近其能看到的墙壁(速度)
  +0.07 m/s,圆锥+9.8 m),LQR松鼠(+2.5 m/s,+20.7 m),这里是 the.
  一个方面,武器在实物而不是程度上确实不同。
* **计算** 跨三个数量级:闭式时间表
  0.011 ms/ decision, 冷却-增量LQR 0.36 ms, SLSQP-MPC 23 ms表示 / 49 ms
  p95 — fine offline or at a 10 s cadence, but MPC buys nothing *on this
  任务编号**,除了经认证的视界内约束处理.
* **Tracking**: RMSE ≈ 79 m in all arms — dominated by the long approach-leg
  由计划本身而不是由法律规定的抵消;
  在各地的预算范围内(最大24.92 < 25,无`fuel_over_budget`).

## 结论的限制

单任务,单场场场景家族,QP内部的线性-CW预测,SLSQP(不是实时的解答器),以及一个坚守者,其计划保持2/秒的速率上限. *拿着*边界边缘(或侵蚀边界的压力装置)的飞行任务是MPC设计案的所在地; 这里测量到的细 MPC 边距 显示它的束缚 活动集确实有作用 。 适用性/限制结论为ADR gnc -0003.
