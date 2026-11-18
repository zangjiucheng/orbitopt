# ADR 飞行实验室 -0009 :发动机与质量流均价(一个推进定律,两个厂)

- **Status:** Accepted — supersedes the "not solved here" engine bullet of
  ADR飞行台-0007
- **Date:** 2026-11-15
- **Numbering note:** 0008 is taken by E6-04 (#276)'s sampled-data control
  记录;这是下一个免费号码。
- **Deciders:** E6 6-DOF Production Maturity (Epic #265), E6-01 (#273)
- **Code:** `src/orbitopt/data/flightlab/jsbsim/engine/spike_engine.xml`,
  `spike_nozzle.xml`, `src/orbitopt/data/flightlab/parity/spike_stage_parity.yaml`,
  `src/orbitopt/flightlab/parity.py` (`_check_nozzle_area`, `_check_pad_thrust`),
  `src/orbitopt/flightlab/sixdof/production_compare.py`,
  `src/orbitopt/verify/cross_model.py`,
  `src/orbitopt/flightlab/sixdof/tracking.py` (`run_tracking_spike(..., cutoff_t_s=)`)
- **Evidence:** `tests/test_flightlab_backend_identity_parity.py`,
  `tests/test_flightlab_sixdof_jsbsim_aero.py`,
  `tests/test_flightlab_sixdof_tracking_jsbsim.py`,
  `tests/test_verify_cross_model.py`,
  `tests/fixtures/flightlab/opentsiolkovsky/fl403_spike/`(重新录制)

## 背景和问题

ADR flightlab-0007 flew the same E1-04 ascent on both plants and published the
engine-model gap it had to leave open:

- JSBSim read **1.883 MN** on the pad where the native plant read **1.677 MN**.
  XML 搭载E4 令牌`0.001 ft^2` 出口,故`FNozzle`
  借项基本上没有后压,推力在每一个字都是真空的
  高度; 原生植物借记`A_exit * p_ambient` 乘以2 m^ 2 出口
  其对等源声明。
- JSBSim燃烧一个圆形、固定的`propflowmax`(600公斤/秒),而原生来源
  从`T(p)/Isp(p)`流出.

ADR-0007拒绝关闭:*"它打破了E1-01平等门的推力衍生(`isp·g0·ṁ`)和FL-401/403基于它的结果".*. E6-01就是关闭. 接受目标不是"通过调音使数字相等"——这是两个工厂都执行**一个**推进定律,因此比较中留下的任何剩余都是对*植物制剂*的主张,而不是对一个 k的主张.

## 决定

一种推进法,写作三种方式(Pality YAML,JSBSim XML,原生植物):

| 终止原因 | 数值 | 何处 |
|---|---|---|
| 真空 推力 `T_vac` | 1.8800 MN (unchanged) | `spike_engine.xml`, 原生阶段 |
| 真空喷雾器 | 320 s(不变) | 不变 |
| 质量流动(选择) | `T_vac/(Isp_vac*g0)` = **599.083275公斤/秒** = 1320.7525 lb/s | `propflowmax` |
| 喷嘴出口区 | **2.0 m^2** = 21.527821 ft^2 | `spike_nozzle.xml`,对等Doc |
| 海平面上升 | **285.506383 s** = `T_pad/(mdot*g0)` | 等值文件c |
| Pad thrust | `T_vac - A*p_sl` = **1.67735 MN** | 生成, 检查两侧 |

值得明确指出的三种后果:

1. **The E1-01 derivation survives.** The parity gate's thrust is still
   `isp * g0 * mdot`; 流量停止为四舍五入的600千克/秒,成为
   `isp_sea_level_s`不再是一个假设,而是一个值
   使两点的原生Isp法 复制在窒息流
   垫。 因为原生定律是线性 降压和衍生
   `T_sl/Isp_sl`等于`T_vac/Isp_vac`,本地`T(p)/Isp(p)`为*同样的*.
   每个高度的599.083275千克/秒——两个mdot历史是相同的
   函数,而不是两个紧密的函数。
2. **A physical exit area replaces a numerical token.** `0.001 ft^2` was never
   一个喷嘴,这是一个写"不借记"的方法。 现在这个区域是真实的
   2 m^2, which is what the parity document always claimed and what the native
   总是使用借项。
3. **MECO成为指挥部,而不是坦克能力事故.**带599.083公斤/秒
   60 t负载持续100.153 s,故旧巧合"`burn_end_s: 100` ==
   罐子干燥了,没了 `run_tracking_spike`获得`cutoff_t_s`,V-02
   6 - DOF适配器现在将情景的`burn_end_s`传递到它:每一个后端
   关闭*同样的禁赛*,60 t坦克能力是
   又是一个时间参数。

## 量产:同为E1-04 上升段,均为植物.

`compare_production_paths()`, measured 2026-11-15 (Python 3.13, jsbsim 1.3.3,
OT CLI 1.0.0). "Before" is the ADR-0007 table; "after" is this change.

| 终止原因 | 在此之前 | 之后 |
|---|---|---|
| 推力 在垫上 | 1.677 vs 1.883 MN (12.6% 相隔,已披露). | 1.6774对1.6774 MN,比例**0.999995** |
| 马克斯-Q | 71.5 vs 76.1 kPa, 6.1% | 71.230 vs 71.329 kPa, **0.14 %** |
| 终点(T +103.6 s) 高度 | 150.4 vs 165.4 km,10.0% | 149.7 vs 157.4 km, **5.9 %** |
| 结束点速度 | 4493对4878 m/s,7.8% | 4501 vs 4786 m/s,**5.9 %** |
| 推进器簿记 | 164千克,负载的0.27% | **12.0 kg**, 0.02 % |
| 弹出率,最大 abs diff | 0.0189 弧度/秒 | 0.01885 弧度/秒 |
| Alpha, pad..60 s | 2.76 deg | 2.739 deg |
| 阿尔法在MECO(录制,不装门) | 27 deg | 26.8 deg |

`production_compare.py`:pad-推力比`0.02 -> 0.005`,最大`0.20 -> 0.05`,推进剂`0.006 -> 0.002`. 端点高度/速度波段停留在0.15/0.10,原因如下. 所有八张`e104-*`支票

### 剩余寿命

- **Closed (implementation, was the engine law).** Pad thrust, Max-Q and the
  推进剂书. 4.7e-6垫-推力残片为JSBSim的lb/ft^2圆形.
  加上大气层对`pressure_pa(0)`的海平面压力;12.0公斤
  JSBSim的储油罐排水沟(F-03)被披露的一至二帧滞后.
  假设允许 2框架 *599.083公斤/秒* 0.02 s = 24 kg),而非流
  差别。
- **开放(执行,轨迹配方,原已存在).** 5.9 %
  三角洲终点是*没有* 推力级:`thrust_n`通过3e-5同意.
  燃烧和质量书到12千克, 但沿途的差别去
  -0.9 m/s at T+10 s, -2.7 m/s at T+60 s, +27.0 m/s at T+80 s, +285 m/s at
  燃烧。 它长得正好是α-窗口行居住的地方(原生体)
  sags ~ 2.7 调用自身速度向量, 而激活 TVC 并编译
  Aero将JSBSim的鼻子抱在alpha~0,一个近垂直的海岸放大.
  ~ 2.5 m/s 燃烧速度进入高度公里. 这是
  原生植物vs - FEM 差额ADR-0007 购买的空气路径,现在可见.
  上面没有引擎缺口 这就是为什么终点带
  *没有*收紧。
- **Open (formulation, disclosed as before).** Post-MECO alpha 26.8 deg: past
  燃烧的舞台是无导弹体 q - > 0,所以恢复
  瞬间在两种植物上消失 记录为`alpha_coast_deg`,从未入门.
- **New and expected.** FL-403's dt-halving spread grew from 30 m to 58 m.
  The pad thrust is 11 % lower than the old vacuum-like JSBSim figure, so TVC
  权威(`thrust * arm`)为11% 下,循环运行更接近其
  起动器滞后(峰值偏移 0.42 - > 0.47 deg). MECO高度偏差
  是 -110 / -52 / -22 m at dt = 0.02 / 0.01 / 0.005 s,一名理查森.
  **+8 m**的外推法: FL-403偏差是步码限制的循环滞后,
  而整数则坐落于10××的500米线下,无论是哪种方式.

## 重新基线的手工艺品(审计记录)

- **FL-403 3-DOF reference** `tests/fixtures/.../fl403_spike/` re-recorded with
  OT CLI 1.0.0 from the aligned vehicle file (1.8800 MN, choked flow noted,
  `nozzle_exit_area_m2: 2.0`,指挥`burn_end_s: 100`. 录制的 MECO
  (T +100 s):73.94 km,5233.9/s地球关系,Mach 18.00,最大95760 Pa;
  dry mass at MECO 10091.7 kg (91.7 kg of commanded-MECO residue).
- **FL-403 GO的界限不变.**(2 deg / 500 m / 20 m/s). 现在测量:
  0.51 deg whole-burn, 0.052 deg settled, -52 m MECO altitude, -2.03 m/s,
  +0.015 deg FPA, 0.47 deg peak gimbal, 0.0 % saturation. `acceptance_matrix.yaml`
  `control.measured.meco_position_m` 移动 51 - > 52 随本注;调查结果.
  医生负责扫地
- **V-02 交叉模型** (`verify.cross_model`):3 - DOF 上升段 RHS 现在借记
  `A_exit * p_ambient`通过`atmosphere.pressure_pa`,`CrossModelScenario`
  从对齐的引擎中生成默认值, 命令 6 - DOF 侧
  to cut at `burn_end_s`. The 3-DOF-vs-OT MECO pair tightened from ~1.3 km to
  半主轴的**410 m**(0.009%),地球相对速度的0.7米/秒,以及
  同样的干重,所以`test_default_pair_passes_and_matches_the_fl403_reference`
  重置为 = 4367061 m, e = 0.544891, v _ e = 5219 m/s.
- **ADR flightlab-0001 keeps its numbers.** That ADR records the E4 spike
  以信物的喷嘴为证,而加以重写,那将破坏记录。
  这个ADR是更改记录代替.

## 后果

- JSBSim的悬崖台、其含空气的生产机体、助推器和
  上面的台阶都继承了 窒息的流 和2 m^2退出
  共享引擎XMLs——一个变化,四个消费者. 生成的机体是
  由`python -m orbitopt.flightlab.sixdof.jsbsim_aero`和
  `--check`城门坚固,如前.
- 平等门增加了两张支票(`nozzle exit area`,`pad thrust`)
  因此,法律不能悄悄地再次漂移:区域、流动、
  衍生的海平面 Isp 和 paid 推力 现在必须一起移动或
  `tests/test_flightlab_backend_identity_parity.py`失败.
- `run_tracking_spike(..., cutoff_t_s=)`是唯一的API变化. FL-403做
  不使用它(它让植物运行到油罐枯竭,这就是FL-401)
  V-02系绳就是如此。
- 回收和任务侧引擎模型未受影响:它们自带
  `T(p)/Isp(p)`表,不是这些XML的消费者.

## 审议的其他办法

- *离开符号 `0.001 ft^2` 退出并不断披露空档*:被拒绝-
  ADR-0007结束,12% 垫-推力三角洲被迫进入
  表格是 E6-01 要删除的。
- *`thrustsl` / a JSBSim 推力 比例因子匹配本地*:被否决-a
  自由把手,只复制一个高度的协议。 实际退出
  区域从一个数字得到每个高度的协议。
- *给原生植物带来压力变化的质量流动*:被否决——没有必要.
  以 `isp_sea_level_s` 推导,两点线性原生法是
  相同的恒流,所以使`mdot(p)`压力依赖
  *添加*是区别而不是去除一个.
- *Set the OT fixture's `burn_end_s` to the 100.153 s depletion instant so the
  3 - DOF参考文献将负载也烧掉 **:尝试和拒绝. 参考文献
  CSV在1的网格上进行采样,因此不能回读非网格截断
  (横跨的线性插值花费21 m/s纯蒿),它制造了
  JSBSim、OT和V-02对 " MECO " 的含义有分歧。 一个命令
  100在每个后端的截断是说明自己的公约的版本.
