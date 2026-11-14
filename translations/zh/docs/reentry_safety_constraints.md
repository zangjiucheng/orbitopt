# Re-entry safety constraints — assessment and design (task #163)

**Status: proposal.** The lunar free-return targeter checks entry geometry; it
does not assess heating, loads, skip-out or survivability. The atmospheric
assessment described below is not implemented for this targeter.

## 1. 今天的代码保证

`target_free_return_entry_corridor()` and the export-side recheck propagate in
vacuum under Earth, Moon and Sun point-mass gravity. At the downward crossing
of a spherical entry interface (Earth radius 6,378 km plus 121.92 km), they
check the **inertial** flight-path angle against [−7°, −5°] and report the
entry speed. Passing this check establishes only the stated geometry.

### 为什么几何走廊还不够

热载和空气动力载荷取决于大气相对速度`v_rel = v_inertial − ω × r`,以及车辆和制导. 地球自转可以将相对角移到一个学位的十分之十;球面和WGS-84高度在极点附近大约可差21. 热负荷和跳出需要大气轨迹,而不只是它的初始角度.

## 2 包括的限制

提议的3-DOF大气通道将报告这些数量. 车辆必须提供限值;缺失限值表示**未评估**.

| 约束 | Metric and proposed model |
|---|---|
| 峰值热通量 | Sutton–Graves对流`q̇ = k √(ρ/R_n) V³`,`k = 1.7415 × 10⁻⁴`在SI单位;分别报告低真度辐射估计. |
| 热负荷 | 将热通量整合在通道上 |
| 峰值感应负载 | `max √(L² + D²) / (m g₀)`,不包括重力. |
| 跳过 | 发现一个向上进入的接口 并分类 退出能量。 |
| 动态压力 | 可选 `max ½ρV²`. |

Classify an exit as `skip_bounded` when its Earth-centered **inertial**
specific energy `ε = |v_I|²/2 − μ/|r|` is negative, or `skip_escape` when it
is non-negative. A bounded skip's subsequent guided entry remains unassessed.
If no upward crossing occurs before the terminal condition, classify the
modeled pass as `captured`.

Use `v_rel` for aerodynamics, never for the inertial energy test. Convert an
Earth-fixed exit state to an inertial frame with the full SPICE `sxform` 6×6
state transformation; a position-only rotation omits the transport velocity.
Record the frame in the report.

## 3. 走廊界限而不是固定带(建议)

在固定入口位置,历元(速度和航向),拟议的根搜索会改变航迹角,以找到最浅的捕捉角,以完全升降,最陡峭的角在加热和负载限度内,以完全升降. 由此产生的车辆特定界限是这些常数银行假设下的必要条件,而不是可行的有指导的入境证明。

## 4. 所需投入和模型忠诚度

### 输入

评估需要进入状态和历元;质量、面积、拖动系数、升降比、岸角和鼻子半径;大气和地球自转;用户提供的热量、负荷和压力限制。 示例胶囊参数必须标明说明性。

### 忠诚等级

| 级别 | 模型和目的 |
|---|---|
| T0:当前 | 真空进入角度仅用于几何筛选. |
| T1:分析 | 弹道/抬升近似和用于测智的指数大气。 |
| T2:拟议 | 3 - DOF,旋转地球,US76大气层,恒定空气和银行,与加热相关. |
| T3:未来 | 引导录入,空气动力数据库和大气散射. |

这些关联的准确性估计是规划假设,而不是这个存储库的验证结果. T2会停留在低空或速度阈值,或向上进入的交叉点.

## 5. 与 `target_free_return_entry_corridor()` 的接口

默认情况下保持现有真空目标器不变. 拟议的`verify/entry_assessment.py`将分别评估其进入状态,然后可选择计算车辆专用走廊界限。 其结果将携带相对速度和角度、热通量和负载、峰值g和压力、跳过分类、假设和每限传/故障/未知值。

The handoff state is Earth-centered ECLIPJ2000. Its absolute epoch is
`reference_epoch_ephemeris_seconds + initial_epoch + entry_epoch` in TDB
seconds past J2000. Start from the existing spherical-interface state and
report its geodetic altitude; do not silently re-target it. Any new report
fields and schema changes are follow-up work.

## 6 验证 未来执行计划

- 将弹道箱与 Allen- Eggers 进行比较, 并检查预期缩放
  弹道系数和鼻子半径。
- 测试角度和银行反应趋势,并验证`steepest < shallowest`.
- 受限和逃逸试验跳伞,包括固定地球到惰性
  `sxform`绕行 近速逃逸.
- 对照物理范围检查说明性的月球返回情况;不
  将与分析近似协议解释为飞行验证.

## 7. 范围:**没有**做什么

The proposal excludes guided or 6-DOF entry, atmospheric dispersions, detailed
aerodynamic and thermal-protection models, and descent below roughly 30 km.
It does not model parachutes, splashdown or crew physiology. Reports must say
"within modeled limits under the stated assumptions", not "survivable".

## 8. 建议的后续任务

实施选入评估和测试,加入报告/schema字段,然后加入车辆专用走廊界限。 对现有接口或地球形状公约的任何改动都需要单独审查。
