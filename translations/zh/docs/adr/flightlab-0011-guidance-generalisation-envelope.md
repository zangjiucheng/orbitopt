# ADR飞行板 -0011:上级导盲信封(E6-03).

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
  不准确”不是发现; 运行干燥的6.5 t坦克, 0.02 rad/s
  无法跟踪的伺服器 无法终止的定律看起来都一样
  一次失去的容忍布尔。

E6-03的接受是一个矩阵,覆盖所有三轴,每个被命名的误差的原因,包括完全没有TVC轴的ORIGINAL F-06车.

## 决定

以**基于规则的归属被宣布的艺术**为基质,

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
  band),每个单元格记录而不是隐藏.
* **助动轴被两次故意应用**:单元格的旅行限制
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
  +0.8 / +0.9 km(3 - DOF / 6 - DOF ) 和跨后卫近拱点 的差数.
  0.09 -4.84公里. E1-03结果不是平纬事故.
* **高度轴 - 以BOTH结尾为界,按照标准,不是坦克.** at
  180 km the cutoff fires with 478/484 kg left and periapsis inside the band
  而远拱点坐落于其外侧的+52.8/+50.8:标准等级.
  近拱点和远拱点随即出现. 从260向上移动
  近拱点从不关闭目标,舞台燃烧耗尽,
  over-shoots (+5 647 to +7 096 km apoapsis). The tank is exonerated by the
  300公里处的200千克MORE推进剂使失事情况更糟.
  (+6 862 -> +17 763 km apoapsis), which only a termination failure does.
* **Payload axis -- the heavy stage is a vehicle limit.** 1.5 t dry / 6.5 t
  propellant runs dry with its terminal orbit 23-88 km *tighter* than the target
  180和220 km(5 581)的圆形,相对于所披露的预算
  变体为6 933 m/s),而0.8 t / 7.2 t变体则在220 km处插入.
  与披露的相同。
* **Actuator axis -- one constraint failure, cleanly separated.** The derated
  (2 deg / 0.05 rad/s / 0.15 s) and wide (5 deg / 0.15 rad/s / 0.08 s) sets
  插入。 2 deg / **0.02 弧度/秒** / 0.25 的点-质量双子仍插入
  (+5.3/+8.5 km) while the rigid body lands with periapsis 6 217 km under the
  目标轨道和轨道倾角 41.6 deg -- -- 伺服器,而不是法律,而不是
  有足够的推进剂可以指责。
* **最初的F-06阶段**(2 t / 6 t,零 gimbal travel)不能拥有:
  法律即时化;4 620 m/s预算的公开循环其近拱点土地.
  3 082 km (3-DOF) / 2 718 km (6-DOF) low, and the two backends disagree by
  365 km -- the pre-E1-03 baseline, recorded so the matrix is not all good news.

## 后果

* E1-03索赔用信封重述:共享法插入此句
  stage at 220 km (and at every declared inclination) inside +/-25 km / 1 deg;
  它不插入180 km,也不终止于~240 km. 两者
  界限是**截断标准**,不是车辆的 - 下一个方向
  任务应添加能量/ 远拱点 终止期限,而不是更大的槽。
* `docs/flightlab.md`(E6-03)报告矩阵并链接研究;F-07.
  block 的“共享定律并不意味着一般准确性”现在被测量了 而不是
  而不是断言。
* 矩阵是基准,不是单位测试: 17细胞×2后端为~15 min
  of CPU. The test module locks the committed study (schema, coverage,
  宣告判决,归属一致性,E1-03点的乐队)和
  在`slow`下,
* `burn_ended_by`现在是记录词汇的一部分:任何未来的错失分析
  在指定原因之前,必须区分受指令的截断与干罐。

## 审议的其他办法

* **每个假想点的定律**(重力期增益,命令-alpha限制)
  直到每个单元格插入:在 300 km - `gravity_term_gain=1.0`和
  `command_alpha_limit_deg=15` both made the over-shoot worse (+18 266 and
  +36 423 km apoapsis). Rejected: the acceptance asks what the shipped law does.
* **通过发射方位角宣布 轨道倾角**带狗腿:如上所述被拒绝.
  (植物不能飞行;结果将是方向-程序动脉)。
* **成绩只有近拱点,**:被否决——正是
  blindness that makes the 180 km cell miss. Periapsis, apoapsis and inclination
  被评为与宣布的容忍度相对应。
* **将一个干燥的罐装在车辆上**:被拒绝,因为法律不能
  也终止油箱的空置。 终端轨道的能量将它们分开.
