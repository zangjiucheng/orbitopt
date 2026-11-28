# ADR会合-0002:全轨迹的安全约束,配置的添加剂v 1.0(E12-04).

- **Status:** Accepted
- **Date:** 2026-11-25
- **Deciders:** E12 Orbital Rendezvous & Mission Operations (Epic #389), E12-04 (#394)
- **Code:** `src/orbitopt/rendezvous/safety.py`, `src/orbitopt/rendezvous/config.py`,
  `src/orbitopt/schemas/rendezvous-mission-1.0.json`, `src/orbitopt/data/leo_rdv_001.yaml`
- **Evidence:** `tests/test_e12_04_safety_constraints.py`
- **Builds on:** ADR rendezvous-0001 (two-impulse J2-targeted Lambert), ADR relative-0001 (`propagate_pair`)

## 背景和问题

E12-04在参考任务的安全封套上贴上:500 mi的守望区,靠近走廊,距离依赖的守望区-速度上限,(已经固定的)燃料预算和时间窗——在**整体** 轨迹上检查,烧伤之间的海岸,而不仅仅是到达时;加上被动安全:燃烧流产后无控制的追逐者不得进入守望区. 两种测量定型设计:

1. *走廊不能是V-的横管。* 每 0 - rev 兰伯特弧
   in the feasible window climbs 10 km of radial separation by flying **over**
   目标:E12-03 参考转移达到LOS = 179.9 deg from
   -V-bar at ~1.5 km range (inside-1.2-km LOS stays <= 79.6 deg). A full-range
   Aft cone会拒绝 整个冲动的双脉冲家族—— 制约
   设定将是一个声明,任务不能飞行,而不是安全信封。
2. *安全屏蔽的可行装置不是凸轮。* 保持退出(最小范围 < 500 m)
   屏幕向外显示 ~ 1500 → 2400 → 终端圆锥屏幕~ 2450 → 2950 → s,所以
   the feasible set is two islands: [1400, ~1500] s at dv ~13.98 m/s and
   [~2950, 3600] s down to dv 10.34 m/s. A golden-section search given the
   屏幕丢弃其前两个不可行的样本上的好盆地(它
   报告13.81 m/s为参考配置的"最佳".

## 决定

1. **`safety` is an optional additive block inside schema version 1.0 — no
   version bump.** Every pre-E12-04 document validates and behaves exactly as
   之前(没有块的配置有 `safety: None`,并保留 E12-03)
   管道); 文档 *与* 区块有计划验证和边界
   由加载器进行交叉检查(节速严格增加,盖)
   射程不降低,保持球体必须排除控点。
   拆卸一个1.1计划 将加载器和强制迁移
   从特性中获取任何好处的配置.
2. **走廊 = 终端船尾锥 + 封闭。** 内 `cone_range_m`
   (1100 m, one Hold-Point-radius beyond arrival) the line of sight from the
   目标必须保留在 `cone_half_angle_deg`(90 deg ——"永远不要在前面")
   -V-bar; everywhere, |z| <= `cross_track_max_m` (100 m). The mid-range over-
   径向几何由 keep-out 球体和速度描述规范
   相反。 所有范围内的更紧的管子都是有动力的铁导线
   (ADR会合-0001范围说明)。
3. **接近速度是一个片面线性上限,范围为** - 下方不变
   第一个结,结之间的线性, 恒定在最后一个上方—— 有结
   [500, 1000, 2500, 9500, 13000] m and caps [9, 9, 12, 22, 26] m/s. The
   profile 承认双脉冲家族的测量运动(最糟糕的参考文献)
   sample: 6.46 m/s inside 1 km, 18.5 m/s crossing 9.5 km) rather than an
   强力转让的理想;
   筛选出中窗弧 和任何合成快速方法。
4. **Every constraint is evaluated per epoch on a dense re-propagated grid**
   (`check_sample_dt_s` = 2 s, 腿端和燃烧的时代总是被采样, 两者都是
   (每条烧伤的侧面)漂流的腿、转移的腿和后插入
   样本;`fuel_budget`和`time_window`是同一报告的事件检查.
   每张支票都带有签名的差值系列和最差的历元+LVLH
   因此,侵犯是可避免的,而不仅仅是可数的。 被动安全
   three no-thrust coasts on the same propagator — from t0 (no burn executed),
   从闸门(烧掉-1),从Hold Point(插入后损失)
   ——`passive_contingency_s` = 5400 s(一个轨道周期),
   仅比照保持距离。
5. **Safety-constrained optimization = coarse scan + refined basin.**
   `plan_rendezvous_safe`扫描固定的托夫窗口(12间隔)通过一个
   per-candidate hard screen (budget + the four trajectory checks on the dense
   转动腿,然后是金色的剖面 围绕最可行的扫描
   在括号内,然后独立地重新验证胜者
   (`verify_plan_safety`,包括漂流的腿和被动的海岸). 没什么
   运行时会放松; 当屏幕杀死每个候选人时, 错误
   携带每节收紧最差的边距(“计划安全,或者说”
   "为什么"。
6. **Reference margins are measured, not guessed.** The packaged numbers give
   分钟边距:保留+413.0 m(t = 3898.6 s),圆锥+19.99 deg,跨轨
   +79.9米,接近速度+2.54米/秒(1公里内),燃料+14.65米/秒,窗口
   +183.3 s,被动+9529 / +9529 / +500 m. 受限制的最佳
   3416.7 s, dv 10.345 m/s) matches the E12-03 unconstrained plan — the screen
   removes half the window but not the optimum; the report says so per epoch.

## 后果

- `plan_rendezvous`保持E12-03语义字节换字节;可选
  `feasible`钩在未使用时是惰性的. 安全规划运行时间
  参考配置为~18 s(扫描+精炼+验证+3被动海岸).
- `evaluate_trajectory_safety(cfg, times, rho, rho_dot)`公开:任何
  测量或构造的 LVLH 样本系列可以进行分级,这就是如何
  接受的负测试 (“ 已建的“ 关闭” 必须失败 ) 是
  证明没有绕过轨道。
- 网格解析是一个已宣布的忠贞键: `check_sample_dt_s` = 2 s 找到
  圆锥边距 18.8 deg vs 8 deg 在粗 5 s扫描网格上——间距
  烧伤和测量它们的网格一样尖
- The 2 s grid aliases the true LOS peak (79.6 deg at 1 s vs 70 deg at 2 s for
  参考文献); 包装的 90 deg 圆锥保持 19 的网格独立
  差值(按1 s:71.2 dig inside 1100 m),所以判决没有.
  取决于别名。
- 守门员也在守门员那里检查过,
  点坐落在1000;装货机禁止保持半径| 持有点|,
  因此,一个特派团永远不能禁止自己的到达点。
