# E13-04 — closed-loop guidance framework: paired closed- vs open-loop holdout (task #421)

Epic # 416. 这个任务将E13- 0×GNC的部件汇入一个运动级闭环(`orbitopt.rendezvous.gnc_campaign`):真理~传感器~导航~导航~指令~指挥~激活错误~真理,在E13-01合同缝合上被冻结在闭环任务`leo_rdv_001`上. 指导法总是决定*时间戳*的估计,补偿激活历元——从不"现在". 机器读数是[`e13_04_closed_loop_guidance.json`](https://github.com/zangjiucheng/orbitopt/blob/main/docs/benchmarks/e13_04_closed_loop_guidance.json); `PYTHONPATH=src python
benchmarks/bench_e13_04_closed_loop_guidance.py --jobs 16 -o …` (25 minh wall at 16 jobs,仅CPU;开放式飞臂作为配对护卫双飞,如果与冻结的E12-10 文物有分歧则中止跑步).

## 实验

Both arms fly the **same** E12-10 holdout: 22 clusters × 12 samples, seed base
20000000, identical perturbation definitions (thruster misalignment/bias/
impulsive draws, sensor noise and dropouts, J2 + drag residuals), identical
ground plan (tof 3416.67 s, planned 10.345 m/s), identical tolerances (50 m /
0.2 m/s) and the unchanged 25 m/s fuel budget. The machine-readable guard block
records sha256 of the frozen E12-10 artifact and of both YAML configs; nothing
was re-tuned to raise the success rate.

The closed-loop arm replaces the two planned burns-only profile with feedback:
departure and arrival keep their paired Δv draws, and between them the
mid-course law (`CwMidCourseLaw` on a compensated estimate) flies the chaser
toward a CW-re-aimed intercept, re-planned at each 60 s controller tick:

* **Schedule.** One maneuver per tick, decided at the estimate's timestamp
  (一个胆囊,一个胆囊,一个胆囊,一个胆囊)
  delay. Three feedback slots: mid-course at half the transfer (t₁ + 0.5·tof ≈
  1709 s, `midcourse_fraction = 0.5`), terminal trim at t_f − 40 s, and the
  arrival burn at t_f − 22 s, so the last actuation lands ≈ 7 s before the
  到达历元.
* **Compensation.** `RelativeNavEKF.compensate_to` propagates the filtered
  法律评价前历元(及其共性)
  E13-02的独立实验显示166 m stale 偏差
  shrinking to −3.7 m through this path.
* **Re-aim.** `solve_cw_intercept` with a weighted least-squares velocity
  词组取代仅用于位置的 CW 解析,在 J2 下是
  占优势的到来源。

失败归属(在冻结的E12-10类之上):导航在跟踪RMSE > 20 m(由导引增益缩放)时发生故障,被指挥的QΔv本身突破预算时进行导引,被指挥的配置适合但实现XΔv时执行,安全_限制否则.

## 数字怎么说的?

* **打开式身份控件**:13/264(4.92%),故障类{87安全,
  164到来 与冰冻的文物完全相同 所以两只手臂
  以同样的图纸比较。
* **闭路:258/264=97.7%**(维尔森95.1 – 99.0%). 每一个
  264样本'EKF汇合;对位差为**+92.8 pp**,
  whole-cluster bootstrap CI [+89.4, +95.8] pp.
* **Paired McNemar (exact, Holm-corrected across the four endpoints):**
  成功 b = 245 /c = 0 ,到达 _ 精度 164 / 0 安全 _ 违反 82 / 1 – 第一次.
  霍尔姆之后的0;燃料_超过_预算0/0(p=1). 无闭路采样
  从被冻结的成功到到达或燃料故障。
* **Fuel: the budget was never binding.** Realized Δv median 11.38 m/s, p90
  12.05,最大12.82——舒适的在25 m/s;`over_budget_samples = 0`.
  反馈烧伤了自己,因为Re-aim会汇合.
* **Arrival:** position error median 42.1 m (p90 44.7), relative speed median
  0.064 m/s——所有258在容忍度内的成功;唯一 > 50 m行是
  six aborted ones below.
* **剩下的六个失败都是安全_的限制**, 所有六是
  *被动释放*规则在停靠点海岸的射击(保持距离)
  − 106至−478米),即在任何指导决定存在之前:这些图是:
  最初状态已经侵犯走廊的情况。 那个
  闭路回归对被冻结的成功(集群7,指数6)是一个
  ——在478米处的被动释放流产,
  终端修剪机被发射(运行的单次修剪拒绝)并离开82.0 m
  到达。 没有任何导航、导航或执行归属
  坚持

## 需要说明的限额

* 追踪RMSE(~79 m)以EKF的第一滴为主.
  仍然通过初始瞬间运行,而引导则从
  975 mi hold 冲抵;在本基准中记录而不是加闸。
* 这六次安全失败是无法减少的 这场运动的冻结
  被动释放几何——固定它们意味着触摸压力定义,
  which the task charter forbids. They bound closed-loop success at 97.7 % for
  任务/压力配对。
