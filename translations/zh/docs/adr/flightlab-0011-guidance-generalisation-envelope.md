# ADR 飞盘-0011 :上级制导信封(E6-03).

- **Status:** Accepted
- **Date:** 2026-11-16
- **Deciders:** E6 6-DOF Production Maturity (Epic #265), E6-03 (#275)
- **Code:** `src/orbitopt/flightlab/sixdof/guidance_matrix.py`,
  `benchmarks/bench_e6_03_guidance_matrix.py`
- **Evidence:** `docs/benchmarks/e6_03_guidance_matrix.json`,
  `tests/test_flightlab_e6_03_guidance_matrix.py`
- **Builds on:** ADR flightlab-0002 (closed-loop two-stage mission), ADR
  -0006(限速着陆制导-伺服器模型)此矩阵
  重复使用),E1-03 /F-07(共享的高级定律及其单项披露).
  假设点)

## 背景和问题

E1-03 cut the 6-DOF/3-DOF upper-stage altitude difference from 235 km to
1.65 km, and disclosed that the result was obtained on a **non-parity variant**
(1 t upper dry mass, 7 t propellant, 3 deg gimbal, one 220 km target, the F-06
pad on heading 090). One scenario point says the law is *shared*; it does not
say where the law works, and three questions were left open:

* **Is the result target-dependent?** 220 km was the only declared orbit.
* **Is it inclination-dependent?** No inclination was ever declared -- the
  28.3 deg F-06 继承的只是上升段意识到的.
* **Is it the law, the vehicle or the servo?** When a cell misses, "the guidance
  不准确”不是发现;干燥的6.5 t坦克,0.02 rad/s
  无法跟踪的伺服器 无法终止的定律看起来都一样
  一次失忆的容忍布尔。

E6-03的接受是一个矩阵,覆盖所有三轴,每个被命名的误差的原因,包括ORIGINAL F-06的车型完全没有TVC轴.

## 决定

以**基于规则的归属被宣布的艺术**为基体,

* **One law, one runner.** Every guided cell flies the *identical*
  `UpperStageGuidanceSpec`通过包装的`run_mission`;只有申报的.
  `ParkingOrbitTarget`(海拔,轨道倾角,容力)和细胞的
  车辆/服务费变动。 没有每个细胞增益调试:每个点的调试会测量
  调音,不是法律。
* **Guided cells hold the stack wet mass at the parity 78 t and the upper-stage
  wet mass at the parity 8 t**, so the tank *split* is the payload axis. The
  -1型机车 姿态程序和分离簿记是
  每个导盲室都有同样的飞行
* **Inclination is realised by launch latitude on a due-east heading.** The
  native plant steers with one TVC axis and no bank -- a probe at 45 deg with a
  dog-leg heading put the rigid body in a 179 x 7 567 km orbit, so dog-leg is
  也不可能是诚实的检验 等级
  轨道倾角是大地对地中心
  垫面的纬度转换(~ 0.19 deg,在申报的1 deg内)
  band),是每个单元格记录而不是隐藏.
* **助动轴被故意应用两次**:单元格的旅行限制
  设BOTH `RigidBodyEngine.gimbal_limit_rad`(植物)和
  `TvcActuator.limit_rad`(服务机). 一个轴,不是两个可以不同意的把手.
* **错失按规则归结于记录的可观察者**,依此顺序:
  TVC旅行-> *车辆能力*; 正好插入了一个后端 - >
  *行为者制约*(无点质量双胞胎是控制);
  每个后端与终端轨道**更紧**运行干燥,超过目标
  圆 - > *车辆能力*; 其他任何东西 - > *指导*. 歧视者
  后两个轨道之间是终点轨道的**能源**
  (`a_terminal - r_target`),不是推进剂布尔:一个油门跟踪
  关闭在干燥油罐的尽头 与命令的断层是分不开的,
  所以每张唱片上都有`burn_ended_by`
  `{no-law, still-burning, depletion, law-cutoff}`.

## 测量(17细胞,dt = 0.05 s,每个细胞都有原生后端).

6细胞插入,11 miss(7引导,3车辆能力,1驱动器约束);参见`docs/benchmarks/e6_03_guidance_matrix.json`.

* **Inclination axis -- no dependence found.** At 220 km the stage inserts at
  28.5, 45.0 and 51.6 deg with periapsis errors of +5.3/+0.5, +2.6/+2.4 and
  +0.8 / +0.9 km(3 - DOF / 6 - DOF )和跨后卫近拱点的差值.
  0.09 -4.84公里. E1-03结果不是平纬事故.
* **高度轴 - 以BOTH结尾为界,按标准,不是坦克.** at
  180 离停火还有 478 / 484 公斤, 近拱点  insideZ 在波段内
  while apoapsis sits +52.8/+50.8 km outside it: the criterion grades
  近拱点和远拱点随即出现. 从260向上移动
  近拱点从不关闭目标,舞台燃烧耗尽,
  over-shoots (+5 647 to +7 096 km apoapsis). The tank is exonerated by the
  300公里处的200公斤MORE推进剂使失事情况更糟
  (+6 862 -> +17 763 km apoapsis), which only a termination failure does.
* **Payload axis -- the heavy stage is a vehicle limit.** 1.5 t dry / 6.5 t
  propellant runs dry with its terminal orbit 23-88 km *tighter* than the target
  180和220 km(5 581)的圆形,相对于所披露的预算
  变体为6 933 m/s),而0.8 t / 7.2 t变体则在220 km处插入.
  与披露的相同。
* **Actuator axis -- one constraint failure, cleanly separated.** The derated
  (2 deg / 0.05 rad/s / 0.15 s) and wide (5 deg / 0.15 rad/s / 0.08 s) sets
  插入。 2 deg / **0.02 弧度/秒** / 0.25 的点质量双子仍插入
  (+5.3/+8.5 km) while the rigid body lands with periapsis 6 217 km under the
  目标轨道和轨道倾角 41.6 deg -- -- 伺服器,而不是法律,而不是
  有足够的推进剂可以指责。
* **最初的F-06阶段**(2 t / 6 t,零 gimbal travel)不能拥有:
  法律即时化;4 620 m/s预算的公开循环其近拱点土地
  3 082 km (3-DOF) / 2 718 km (6-DOF) low, and the two backends disagree by
  365 km -- the pre-E1-03 baseline, recorded so the matrix is not all good news.

## 后果

* E1-03索赔用信封重述:共享法插入此句
  stage at 220 km (and at every declared inclination) inside +/-25 km / 1 deg;
  它不插入180 km,也不终止于~240 km. 两者
  界限是**截断标准**,而不是车辆的 - 下一个指引
  任务应添加能量/ 远拱点 终止期限,而不是更大的罐。
* `docs/flightlab.md`(E6-03)报告矩阵并链接研究;F-07.
  block 的“共享定律并不意味着一般的精确度”现在被测量了 而不是
  而不是断言。
* 矩阵是一个基准,不是单位测试: 17细胞×2后端为~15 min
  of CPU. The test module locks the committed study (schema, coverage,
  宣告判决,归属一致性,E1-03点的乐队)和
  在`slow`下,
* `burn_ended_by`现在是记录词汇的一部分:任何未来的误差分析
  在指定一个原因之前,必须先将受命停产与干罐区分开来。

## 审议的其他办法

* **每个假想点的定律**(重力期增益,命令-alpha限制)
  直到每个单元格插入:在 300 km - `gravity_term_gain=1.0`和
  `command_alpha_limit_deg=15` both made the over-shoot worse (+18 266 and
  +36 423 km apoapsis). Rejected: the acceptance asks what the shipped law does.
* **发射方位角宣布 轨道倾角**带狗腿:如上被否决.
  (植物不能飞行;结果将是方向-程序动脉)。
* **成绩只有近拱点,**:被否决——正是
  blindness that makes the 180 km cell miss. Periapsis, apoapsis and inclination
  被评为与公开的容忍度相对应
* **将一个干燥的罐装在车辆上**:被拒绝,因为法律不能
  也终止油箱的空置。 终端轨道的能量将它们分开.

## Follow-through: E6-05 (task #292)

后果要求下一个指导任务"一个能量/远拱点终止期". 它现在存在,信封没有移动:

* The cutoff is a two-sided insertion box (periapsis *and* apoapsis inside the
  远拱点 - 边距代理, 带有调用 - 远地点
  计划保护一个被宣布的容忍(25 km),而这种容忍的存在只是为了保持
  当盒子坐落在目前的轨道下时,制导目标有限,以及
  远地点 - 最高控制器在盒子装好后,会发出微调命令。
  拖动是默认的(`apogee_ceiling_gain`);矩阵飞8e-4.
* The matrix was re-flown on the generalised law: the same six insert, the same
  eleven miss, zero unexpected. The law change is verdict-neutral on every cell.
* The seven guidance-class misses split into two mechanisms, both measured:
  **six**与终点轨道耗尽 *范围*与目标近拱点
  仍然不足 乐队地板 而坦克空空—— 盒子从不手臂,
  任何截断、盖子、饲料或持具,都不能武装它。
  重新贴上“车辆能力”标签
  与 ADR 的最后一个选项相悖 。
  **一个**(a 180 -i 28.5 -V1-A1)是这个确诊的ADR的超射:盒子.
  *将*臂,但远拱点已经过天花板;
  远地点 的反向修正在物理上无法实现
  ± 30 ° 速度-帧命令夹(速度30 °内的任何命令)
  87% 升级),所以近地点通道仍然印30 – 70 km
  超过天花板。
* Widening that clamp to 45° converts the a180 cell on both backends
  (-21/+17 km, 517 kg to spare) — and flips a220-i45 and a220-i51.6 from
  插入到 3 – 6 km *下级* +近拱点 楼层,带有 220
  参考单元格的近拱点比值降至~1 km. 一个共同的法律不能
  与另外两个单元格一起购买; 30 ° 保留,拒绝记录为
  信封的边界 过度射击类的转换需要定律
  命令超出机体可实现的 AoA 窗口的姿态(a)
  投球反转 / 制动- 制动系统), 而不是更大的限制器 。
* JSBSim频道(`sixdof/jsbsim_matrix.py`,
  记录`benchmarks/e6_05_jsbsim_matrix.json`:每条腿重新启动JSBSim at
  the native run's frozen separation state, and 16 of 17 verdicts agree. The
  例外是 220 - i 28.5 - V1-A3 - 此矩阵已经属性到的单元格
  服务实现——用480公斤推进剂切断JSBSim
  原生刚性体耗竭(6.85%推进剂历史)
  divergence, recorded and expected, not gated away). On the six inserted
  单元格 引擎历史 同意 0.05%, 截断时间为 < 1 s,
  0.00 o / o / o / o /
  点火; 预测缺失的细胞燃烧出追逐无法到达的目标
  并且根据他们的判决来判断 这两条渠道都漏掉了

## Follow-through: E10-08 (task #305) -- the guidance class converted

E10-08 asked for a broader envelope on the guidance-class misses without
buying it with a relaxed band. Reading the six "depletion" misses on the
trace, not just the terminal elements, overturns E6-05's last finding. On a260
the stage peaks at 227 km with `v_r` passing through zero, then burns the
rest of the tank horizontally. The osculating periapsis can never exceed the
current radius, so the box could not arm. The law never flew to the target
*radius*. Its `Lambda` targets velocity only, so the climb is whatever the
handoff leaves: 2 km/s of vertical speed and a 326 km ballistic apogee at
ignition. The energy rule was right (terminal orbits thousands of km wider
than the target), and the binding limit was the law, not the delta-v budget.

**Decision.** The shared law gains a terminal-radius radial channel
(`upper_stage` module docstring, E10-08), on for every guided matrix cell
and off by default (the F-07 point is untouched):

* **Radial plan.** The classic PEG radial channel: a vertical thrust
  fraction `s(t) = A + B t` that lands `r(T) = r_target` *and* `v_r(T) = 0`,
  使用封闭式的 推力 组件 `L, J, S, Q`. `T` 解析所以
  计划水平的推力 整体购买 正好是缺失水平
  开快 在12下方进行定律节速向导.
* **Settle cutoff.** Inside the declared band the burn continues while the
  下一个控制样本预计会降落在靠近圆圈的地方, 在同一处
  side. At ~20 g the band is crossed in a handful of 0.05 s samples.
* **q-alpha limiter.** The 30 deg pitch-bias bound becomes
  `clip(30 000 Pa deg / q, 30, 90)`. q Q 1 kPa和
  opens only near vacuum (q ~ 10 Pa at the ~90 km ignition). Inside the fixed
  30 deg 夹住计划不能指向陡峭的交接速度矢量下方,
  这正是它所需要的权威。
* **Shared command history.** The command is held prograde until ignition
  至於灭亡的灭亡者,已在那里繁殖了。 命令被击毙是6 deg/s,而不是
  E1-03 3 deg/s. At 3 deg/s the end-game swing from the plan's pitch-down
  进入速度是无法制造的 僵硬的身体横扫
  the band on a300. At 10 deg/s every cell inserts, but the JSBSim PID lags
  the commanded slew by 5-8 deg, over the E6-05 5 deg arrest allowance. At
  6 deg/s that growth is <= 0.4 deg.

**计量**(`docs/benchmarks/e6_03_guidance_matrix.json`,17细胞,两者均为原生后端):**12从6插入,无意外判决.**.

* **Converted.** Five of the six depletion-class guidance misses (a260,
  a300, a300-V2, a260-i45, a300-i51.6) and the overshoot cell a180 now
  在两个后端插入。 法律的终结 与
  65 -684公斤的推进剂;最差的近拱点误差为-22.1公里.
  (a220-i51.6, point mass). The six E6-03 insert cells still insert, and native
  点质量/刚性体推进剂同意在每插入的10千克
  细胞。
* **Still misses, and why.**
  * **a400: vehicle-capability.** No continuous full-thrust burn from this
    handoff arrives level at 400 km. The radial plan needs `|s| > 1` (more
    than all of the thrust vertical) over the whole burn, and the stage has
    no coast or relight. The unchanged attribution rule now names it a
    capability limit: dry tank, orbit 1.7-1.9 Mm *tighter* than the target.
    That label comes from the data, not from a relabelling.
  * **V3 heavy stage at 180 and 220 km: vehicle-capability.** Unchanged:
    dry tank, orbit 59-211 km tighter than the target.
  * **A3 rate-limited servo: actuator-constraint.** The point mass inserts
    on the same command and the rigid body does not.
  * **F-06 original stage: vehicle-capability.** No TVC axis, so no law.
  * 这些没有经过, 和容忍都没有移动。
* **Alternatives measured and rejected.**
  * `s(T) = 0`(平面`s`):从此不可行
    handoff above 220 km, because it needs `s(0) > 1`.
  * 适应性水平到达权重:不连续的扰动计划,以及
    a300 regressed to -133 km.
  * 领先地平线上的高度控制终端: 计划下潜了
    outruns the 3 deg/s slew.
  * 5的终端交接:僵硬的身体在300上横扫乐队.
  * 先前的命令语义(在僵硬状态下从分离中参考法律)
    body and JSBSim, only from ignition on the point mass): a ~70 deg
    command step at ignition for the frozen trackers. The result was a 35 kg
    native/JSBSim propellant split on a180, over the 0.5 % parity gate.

**JSBSim 均等**(`docs/benchmarks/e6_05_jsbsim_matrix.json`,从每个囚室的原封不动的隔离状态中重新引用新法). 所有130 E6-05检查通行证,没有闸门重新申报.

* **Verdicts.** 16 of 17 agree. The single difference is again
  a220-i28.5-V1-A3, the cell attributed to the servo realisation
  (`verdict_difference_expected`):JSBSim在1.15 deg/s servo上插入.
  原生的僵硬体会衰竭 所以17 /17 符合期望.
* **Co-inserted cells.** Propellant histories agree to <= 0.11 %. Cutoff
  Times agree to 1.0 s, 在本地的1的样本网格上. 电源窗口
  点火时在漂移时的姿态增长是2.3 deg。
* **Terminal elements.** The median |d_periapsis| / |d_apoapsis| between
  JSBSim和土生土长是12.0 / 3.1 km. JSBSim的0.01 控制台阶让出.
  settle cutoff stop within ~0.4 km of the target periapsis, where the native
  0.05的台阶止步至~22公里短. 两者都在宣布的乐队内.

**Consequences.**

* E1-03信封声明再次重述:共享法,与
  E10-08频道,在180 -300公里处和每申报一次时插入此阶段.
  轨道倾角 内+/-25 km/1 deg,位于三株植物上.
* The next envelope boundary is 400 km. It is a mission-profile limit (no
  海岸或重光),不是法律限制,所以扩展需要海岸到远地点.
  燃烧,是一种车辆/序列变化。
* 定点截断的准确性为~20 g的控制步骤. 一个矩阵
  wants tighter apsides than ~20 km on the native plants needs a finer
  高级控制间隔,不是不同的定律。
