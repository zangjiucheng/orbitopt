# ADR飞行实验室 -0007:生产JSBSim忠心(含空气的上升段路径).

- **Status:** Accepted
- **Date:** 2026-11-14
- **Deciders:** E1 6-DOF Fidelity Integration (Epic #246), E1-04 (#253)
- **Code:** `src/orbitopt/flightlab/sixdof/jsbsim_aero.py`
  (生成者:`aerodynamics_block`,`_cg_cancel`,`promotion_fidelity`,).
  `regenerate`, `aircraft_aero_note`),
  `src/orbitopt/data/flightlab/jsbsim/aircraft/spike_stage_aero/`
  (生成动脉),
  `src/orbitopt/data/flightlab/aero/spike_stage.yaml`(真相来源),
  `src/orbitopt/flightlab/sixdof/production_compare.py`
  (`compare_production_paths`),
  `src/orbitopt/data/flightlab/sixdof/e104_aero_ascent.yaml`
- **Evidence:** `tests/test_flightlab_sixdof_jsbsim_aero.py`,
  `tests/test_flightlab_backend_identity_parity.py`(生产检查)

## 背景和问题

Since ADR flightlab-0001 the JSBSim backend is the only Flight Lab engine
with gimballed TVC actuators and propellant-driven mass/CG/inertia -- but
E1-01 deliberately left it flying **zero aero** (ADR 0001's disclosed
boundary): the packaged `spike_stage` aircraft carries an all-zero
`<aerodynamics>` section, so no JSBSim scenario could ever exercise an
aero-bearing ascent, and the aero-fidelity studies could only be done on the
native rigid-body plant with its illustrative coefficient file.

E1-04 要求生产 JSBSim 路径,飞*不变* 空气 原生植物飞`data/flightlab/aero/spike_stage.yaml`(对称:`ca` / `cn` / `cy`表,5% 静差值在`cm` / `cn_yaw` ,腐烂) – 要求将与原生刚体植物的区别量化和解释,或一条路径宣布生产原因.

## 决定

**The production JSBSim path is the generated aircraft `spike_stage_aero`,
and it is the production path for aero-bearing 6-DOF ascent.** The aero
document stays the source of truth; the aircraft is compiled from it and
regenerated, never hand-edited (`regenerate(check=True)` in the test suite).

原生硬体植物仍然是原生研究的参考植物(F-02 / F-04 / F-06 / F-07 ),并保持其发动机-真性优点(环境校正推力 和质流);JSBSim生产路径带来空气动力学,活大众记和原生植物没有的TVC 开动器模型. 它们之间的量化三角洲是下表,而不是手电波.

### 文档的编译方式

- `ca` / `cn` 被旋转到文档的稳定性轴上
  `(alpha, Mach)`网格并写为JSBSim 2 - D表
  (`CD0`,`CL0`;`cy`变为`CYbeta`,静差表变为.
  `CMa` / `CNb`,腐烂的词变为`CMq` / `CNr`,和β-drag
  递增变为`CDbeta`(绑定在文件本身的`cn`基值上,
  从未装配).
- 参考几何来自文档:`Sw = area_m2`,
  `bw = span_m`, `cbarw = length_m`, AERORP at the *dry* CG.
- `_check_expressible`拒绝其表格沿轴变化的文档
  生成的术语不能表达,而不能默默降低系数。
- **CG-coupling cancellation (`_cg_cancel`).** The document applies its
  forces at the *live* CG and folds the static margin into the couple terms;
  JSBSim在一个固定点上运用其力量. 未经改正,
  applied normal/side force carries an extra `arm x force` couple that
  取决于推进剂载荷 -- -- 测量:天真的汇编
  生产机身恢复硬度与原产模型 'Z differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differZ differ
  最大为~200×在满载坦克上(α 2.9 deg对0.000 deg通过最大Q).
  因此生成的表格正好减去了这对夫妇
  `(cg - aerorp) / cbar` 乘以它所属的力量表,作为一个
  function of `inertia/cg-x-in`. The net moment is the document's own
  CG- 在每个推进剂装载时参考对数;取消后
  `cm` / `cn_yaw` 的剩余晋级错误为机器精度.
  (`promotion_fidelity.cm < 1e-9`).
- *部队*的促销错误是诚实的内插残留:
  worst 0.0035 absolute, at the 15 deg grid edge the E1-04 ascent never
  到达(测试波段0.005).
- 平面门: `check_parity` 现在读 `jsbsim.production_aircraft`
  并声称生产飞机继承同样财产
  质量/推进剂/内燃机/发动机/喷雾器,*声明*空气动力学,并宣布
  文件的参考几何(`parity-jsbsim-production-*` ).

## 数量:相同的上升段,两种植物

`production_compare.compare_production_paths()` flies the packaged E1-04
ascent (sea-level pad, held-vertical hold, full throttle to burnout,
`dt = 0.02 s`) on both plants from the same initial state and publishes
`Check` verdicts. The native side runs `j2=False` so the comparison isolates
thrust and aero from gravity models (JSBSim carries central gravity).
Numbers below are the measured run of 2026-11-14 (Python 3.13,
jsbsim 1.3.3) -- the table stands as published, i.e. **before** the E6-01
engine alignment; ADR flightlab-0009 carries the same table after it
(pad thrust 0.999995, Max-Q 0.14 %).

| 终止原因 | 原生刚体 | JSBSim制作 | rel. diff | 带 | 解释 |
|---|---|---|---|---|---|
| 推力 在垫上 | 1.677 MN(在2 m^ 2出口后按环境借记) | 1.883 MN(键`0.001 ft^2`出口,真空式) | 1.08%与预测的`Isp_vac/I_sl`比1.135% | 2 % | 螺旋发动机XML保持E4符号喷嘴;本土借记后压. 所披露的引擎——模型差数,以预测比率确认. |
| 马克斯-Q | 71.5 kPa | 76.1 kPa | 6.1 % | 20 % | 12%pad-推力借方差额:JSBSim通过更稠密的空气攀升更快. |
| 燃烧(吨=104 s)高度 | 150.4公里 | 165.4公里 | 10.0 % | 15 % | 同:真空式推力飞行较高. |
| 燃烧速度 | 4493 m/s | 4878 m/s | 7.8 % | 10 % | 同样的。 |
| 推进器簿记 | 从环境校正的 推力 / Isp 的 mdot | 固定 `propflowmax` 600公斤/秒 | 164 最大负载,0.27% | 0.6 % | 原生`T(p)/Isp(p)`运行0 – 1在低海拔时的热度%. 结束干质量相同。 |
| 弹出率,最大 abs diff | — | — | 0.0189 弧度/秒 | 0.02 | 激活 TVC vs 土生土长 gimbal: 同一指令配置. |
| Alpha, pad..60 s (aero authority) | - 2.9 deg | - 0.000 deg | 2.76 deg | 3.0 deg | 原生的α在它自己的引擎/重力模型下,作为被控制的身体轴上的无节制原生的轨迹 sags生长; JSBSim的气压在速度上 不是空值差:CG-校正厂分享文件的恢复坡度(`cm`精确). |
| 阿尔法在MECO | - 27 deg | - 0.0 deg | 27 deg | **not checked** | 过去MECO是一个无导弹道体:用0,恢复瞬间消失在两个工厂上,两个帧/重力集成没有约束。 被披露为`alpha_coast_deg`,没有检查波段. |
| JSBSim Alpha,全程飞行 | — | ≤ 0.0 deg | — | 1.0 deg | The production airframe flies the commanded ascent on its own aerodynamics. |

所有检查通过(`e104-*` in `production_compare.py`;波段生活在该模块中,并由`tests/test_flightlab_sixdof_jsbsim_aero.py::test_backend_comparison_stays_inside_its_disclosed_bands`断言).

## 后果

- `jsbsim` backend: packaged *base* aircraft stay zero-aero (they remain the
  TVC/质量纸测试E4/FL-401;生产路径飞扬编译.
  图表。 `docs/flightlab.md`的忠诚表和后端的音符
  `backends.py`说.
- New scenario `e104_aero_ascent` is the reference aero-bearing 6-DOF ascent;
  F-03质量 - 模型检查通过正式的适配器运行.
  (`backend_self_check`,所有质量/CG/inertia检查通过;`physics_verified`为.
  ADR 飞行实验室 -0004
- 重生是决定性的:动画、其ADR和对等源
  move together or the test gate fails (`stale artefact` and the three
  生产均等检查)。
- 未在此解开, 与 E1-01 相同 : 引擎 XML 的代号喷嘴
  退出区域(JSBSim 推力在任何高度都类似真空)和固定区域
  `propflowmax`. 两者都可见为垫-推力/燃烧状态三角洲
  表格中; 关闭它们将改变 FL-401 / FL-403 的结果,并且是
  无法执行航空任务
  *Closed by ADR flightlab-0009 (E6-01, #273)*, 重排 FL-401
  /FL-403结果和上表:垫-推力行现为0.999995和.
  the Max-Q row 0.14 %, and what survives of the burnout-state delta is the
  轨迹/配制剩余,不再为引擎缺口.
- The 27 deg coast-phase alpha divergence means the production path is *not*
  a guidance-validation plant for the post-MECO coast of an *unguided*
  舞台; F-04- 风格跟踪器关闭了循环( E2) , 身体遵循
  它的命令和差异是一个控制器问题,而不是一个空气问题。

## 审议的其他办法

- *手写 `spike_stage_aero.xml`*:被拒绝了,
  两个引擎读取同一文档; 手写文件烂掉。
- *AERORP在固定折中站,而不是干CG*:被拒绝-
  它只将一个推进剂负载依赖错误换成另一个;
  `_cg_cancel`的校正使净对子在每一负载上都精确.
- *Tuning the JSBSim engine to match native burnout states* (bigger nozzle
   `thrustsl`)退出区域:因此任务被拒绝——它打破了E1-01.
   平等门的推力(`isp·g0·ṁ`)和FL-401/403的结果
   三角洲被揭开
   *由ADR飞行实验室取代 -0009*,它关闭了空隙而不调音
   knob: 物理 2 m^ 2 出口+ 窒息 `propflowmax` 来源于
   `T_vac/(Isp_vac*g0)`离开`isp·g0·ṁ`的衍生精确度(与
   `isp_sea_level_s`衍生而非假设)和FL-401/403基线.
   重录.
- *原生的硬体作为"生产路径"*:被拒绝了——它没有生命力.
  推进剂簿记器,无TVC助推器和说明系数;
  任务涉及JSBSim路径的忠诚,上表是
  选择它的价格清单。
