# ADR 汇合-0001 :双联赛J2- 瞄准兰伯特与在边界强制执行的地球木(E12-03).

- **Status:** Accepted
- **Date:** 2026-11-24
- **Deciders:** E12 Orbital Rendezvous & Mission Operations (Epic #389), E12-03 (#393)
- **Code:** `src/orbitopt/rendezvous/`, `src/orbitopt/schemas/rendezvous-mission-1.0.json`,
  `src/orbitopt/data/leo_rdv_001.yaml`
- **Evidence:** `tests/test_e12_03_rendezvous_planner.py`, `tests/test_e12_03_lambert_earth_mu.py`
- **Builds on:** ADR relative-0001 (LVLH frames, `propagate_pair`), `orbitopt.lambert.cpu`,
  `physics_io`乐队检查

## 背景和问题

`lambert/cpu.py:solve_lambert_single` defaults to `mu=pk.MU_SUN`. That is right
for every interplanetary scene in the library and catastrophically wrong for a
500 km rendezvous — and *silent* wrong in the sense that pykep will happily
return a "trajectory": it is `physics_io`'s plausibility band that catches the
mixup, and only if the pair actually gets band-checked. E12-03 reuses this
solver for an Earth mission, so the default has to be structurally unreachable,
not merely avoided. Separately, the acceptance is *nonlinear*: the plan must
arrive within 50 m / 0.2 m/s of a 1 km Hold Point under J2 propagation, so any
model used to plan must be targeted against the model used to verify.

## 决定

1. **The mission is a config file, fixed before optimization.**
   `schemas/rendezvous-mission-1.0.json` + `data/leo_rdv_001.yaml` (LEO-RDV-001:
   500 km target, 490 km chaser, 20 km initial separation, 1 km V-bar-aft Hold
   点,25 m/s 预算,5400 s 漂移窗口,1400 – 3600 s 转移窗口,
   出发门 y = − 8000 Z,容度 50 Z / 0.2 Z m/s). 计划很严格
   (无未知的密钥); 加载器添加交叉检查, 计划无法
   express — in particular `dynamics.mu` must **equal the exact packaged Earth
   value** `3.9860044150e14`, so this config family cannot silently name another
   中央机构。
2. ** **计划 模型校验 模型:全 `propagate_pair`配有配置的μ和.
   J2 ** ** 一个双体的Lambert溶液是*仅初始猜测*.
   drove this: iterating the aim point (`aim += miss`) under J2 has gain ≈ 1 —
   转让和持有轨道之间的差值J2为
   ~ 6.5 km at 1200 stof 并且不收缩-所以它从不聚合; 牛顿上
   the *departure velocity* against the full propagation converges in ≤ 4
   迭代(一个批号为4-pair `propagate_pair`).
   到达目标为预言的首级状态 at t 2 重新表达为悬浮
   所以牛顿残留的*这是*的接受错误
3. **The optimizer searches the user's window, never outside it.** Golden-section
   将 [改为 _ min, 将 _ 最大 ] 最小化 无法跨越的计划
   `drift_max_s`内部的漂移门 而不是伸展窗户。
4. **μ防御是三个独立的网.**(a) 上文描述的平等检查;
   (b) 每个规划员呼叫网站明确通过`mu=cfg.mu`作为关键词;
   (c) 回归测试——对LEO对的默认微调必须提高
   `PhysicsIOError`, 和一个猴子帕奇间谍`planner.solve_lambert_single`
   表示每个记录的μ完全等于被包装的地球值。

## 后果

- 一对LEO 交给解答者的*默认* μ在太阳波段响亮的失败
  检查( `|r| outside [1e7, 1e13]` ); 两端关闭无声路径 。
- Reference plan (deterministic, ~10 s): gate at t1 = 560.6 s, tof = 3389.0 s,
  total Δv = 10.34 m/s (margin 14.66 m/s), arrival position error 5·10⁻⁷ m and
  relative speed 5·10⁻¹⁰ m/s under the verifying propagation — the acceptance
  由于计划员是
  目标在同一宣传器上 接收开始。
- 接受判决只存在于`verify_plan`(一次失败的检查不能)
  提高——例如,预算过于紧凑,使支票翻转,不会中止);
  `plan_rendezvous`只在配置下*没有计划*时加价.
- 范围是近圆形单曲V- bar家族的参考任务
  住在这里。 多脉冲转移、椭圆目标轨道和动力终端
  指导需要他们自己承认; `plane_normal` Pinning(赤道飞机)
  兰伯特)和`max_revs=0`的选择被固定在规划者而不是配置中.
