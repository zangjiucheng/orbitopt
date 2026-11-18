# 参考容忍度审计(E8-04)

与承诺的参考数据进行比较的每一个回归都必须是可复制的**有点公平**:相同的输入,相同的锁定的包堆和相同的地球方向. 当一个波段改变时,此页和参考文献中的机器可读审计`PROVENANCE.json`记录*为什么*,从旧值到新值. CI运行`scripts/fl601_reference.py tag-check`,所以在参考目录下没有`[fl601-reference-update]`承诺的diff失败,`tests/test_e804_reference_data.py`如果此页面的容限块从执行的`tolerances.json`漂移则失败.

## 统一的 EOP 定点机制

Astropy的离线IERS默认在主要版本之间发生了变化(Astropy 6苍蝇IERS-B离线;Astropy 8在下载时更喜欢IERS-A),而光是将框架桥移动米. 因此,任何引用回归都不得依赖所安装的天体发生的情况。 B门和E2-04通过`orbitopt.reference_eop`(`pin_no_download`和`freeze_reference`)冻结地球方向,根据参考文献的记载方式选择两种模式:

- **`pinned-bundled`**(B组,`tests/data/flightlab/reference/`):
  在*默认录制堆栈的离线表*号上飞行了轨道
  (Astropy 6.1.7 + `astropy-iers-data 0.2026.8.31.0.57.9`,一个IERS-B).
  table). The snapshot records the recording-stack version, the active
  表格中,UT1-UTC/极地运动的有效历元和精确
  括号行; 验证 重划所有已安装的
  捆绑的桌子,如果有什么不同的话拒绝(`ToolchainMismatch`).
  无下载(`IERS_AUTO_DOWNLOAD = False`).
- **`override`** (E2-04, `tests/data/mission/reference/e2-04-frozen-mission/`):
  参考文献是针对IERS-A行的 所以快照
  四行 *成为* 在回归时间,任何堆栈。

E5-05目前使用`scripts/e505_reference.py`中的范围快照语境:它自己的录制包IERS-A行超越主机表. 这不是B门捆绑政策,储存的快照并不意味着每一行都是最后的衡量。

`scripts/fl601_reference.py update` re-flies the Gate B reference and
records fresh digests. For a comparison-policy-only change, use
`update --tolerances-only --reason "..."`: it preserves flight output,
requires the reason, and appends the old→new tolerance/hash audit to
`PROVENANCE.json`. `verify` checks all 13 files against the recorded digests.

## RAAN/面向地球的基准

`tests/test_e804_raan_benchmark.py` grades the frame bridge against
references independent of the recorded tracks: the analytic sidereal
rotation angle for a UT1 shift (omega_E = 360.9856235279 deg/day), the
polar-motion vector norm as the tilt the matrix must show, the
pure-z-rotation signature of UT1 (dRAAN = sidereal angle, dInclination
still), and an astropy-free IAU-76 precession + ERA pipeline whose
external RAAN must band within 0.60 deg of the recorded track (the band
absorbs only the IAU-76-vs-IAU-2000 CIP/CIO model gap; structural frame
errors move RAAN by degrees). Bands were measured 2026-11-15 on the
recording stack with >100x margin against the analytic values.

## 跨平台一致性

`tests/test_e804_cross_platform.py` re-derives the committed GCRS tracks
from the raw recorded OpenTsiolkovsky outputs on the running platform and
requires the difference inside `cross_platform` below. CI runs this on the
identical locked stack on linux-x86_64 (`test` job), macOS ARM
(`test-macos`, osx-arm64) and macOS Intel (`test-macos`, osx-64);
observations per platform are ledgered in
`tests/data/e804/cross-platform.json` (`cross-platform-record` prints the
line for a platform). The committed ledger currently contains a Linux
observation; both macOS entries are `null` (not recorded). A configured CI
matrix is not a stored cross-platform result.

## 历史

| 日期 | 变动 | 理由 |
|---|---|---|
| 2026-10-28 `286ebe2` | B门位置/高度/下行距 1e-6 - > 1e-4米,速度 1e-8 - > 1e-6米/秒 | 跨机层: 参考 OT 二进制本身在建置主机的这个级别上不同; 频段只需要证明在录制平台自身可复制性之外没有回归性 |
| 2026-10-28 `e97e147` | Gate B `t_s_abs` 1e-9 -> 1e-6 s | 记录时间栏的同一跨机层 |
| 2026-11-15 `3051848` | E2-04 add `ascent_build_{position,velocity}_m_abs` 1e-6 and `landing_plan_diagnostic_abs` {fuel_planned_kg 1e-3, tof_flying_s 1e-2} | 跨平台保存的物理判断;每个运行时重建的诊断需要自己的波段 |
| 2026-11-15 (this task) | Gate B add `cross_platform` {position_m_abs 1e-4, velocity_m_s_abs 1e-6} | E8-04跨平台重构演练与飞行波段相同的钉框桥,因此波段等于它们;`PROVENANCE.json`完全审计条目 |
| 2026-11-17 (this task) | 增加了E8-01全链带(`FULL_CHAIN_TOLERANCES`,下表). | E8-01 成绩 记录的E5-05 链与原始的OpenTsiolkovsky 记录和闭式力学;波段来自记录器网格,元素的生成和环境 - 模型 预算,而不是后热调 |

## 当前强制容忍

下面的每个块必须等于引用的`tolerances.json`字节的解析形式(由`tests/test_e804_reference_data.py`强制).

```json
<!-- tolerances: gateb-leo-two-stage (tests/data/flightlab/reference/tolerances.json) -->
{
  "t_s_abs": 1e-06,
  "mass_kg_abs": 1e-06,
  "position_m_abs": 0.0001,
  "velocity_m_s_abs": 1e-06,
  "extra_abs": {
    "altitude_m": 0.0001,
    "latitude_deg": 1e-09,
    "longitude_deg": 1e-09,
    "downrange_m": 0.0001,
    "dynamic_pressure_pa": 0.01,
    "mach": 1e-07
  },
  "extra_abs_default": 1e-06,
  "check_value_abs": 1e-09,
  "check_value_rel": 1e-09,
  "event_data_abs": 1e-09,
  "event_data_rel": 1e-09,
  "cross_platform": {
    "position_m_abs": 0.0001,
    "velocity_m_s_abs": 1e-06
  }
}
```

```json
<!-- tolerances: e2-04-frozen-mission (tests/data/mission/reference/e2-04-frozen-mission/tolerances.json) -->
{
  "ascent_build_position_m_abs": 1e-06,
  "ascent_build_velocity_m_s_abs": 1e-06,
  "t_s_abs": 1e-06,
  "mass_kg_abs": 1e-06,
  "position_m_abs": 0.01,
  "velocity_m_s_abs": 0.0001,
  "extra_abs": {
    "altitude_m": 0.01,
    "latitude_deg": 1e-08,
    "longitude_deg": 1e-08,
    "downrange_m": 0.01,
    "dynamic_pressure_pa": 0.1,
    "mach": 1e-06,
    "accel_g": 1e-06,
    "q_bar_pa": 0.1,
    "density_kg_m3": 1e-09
  },
  "extra_abs_default": 0.001,
  "check_value_abs": 0.0001,
  "check_value_rel": 1e-09,
  "event_data_abs": 1e-06,
  "event_data_rel": 1e-09,
  "landing_plan_diagnostic_abs": {
    "fuel_planned_kg": 0.001,
    "tof_flying_s": 0.01
  },
  "report_value_abs": 0.0001,
  "report_value_rel": 1e-09
}
```

## Full-chain reference bands (E8-01, task #281)

`src/orbitopt/verify/full_chain.py` 所记录的E5-05链与其承诺的参考文献进行分级;下面的波段是真理的单一来源(`FULL_CHAIN_TOLERANCES`,每份报告中嵌入的逐字记录),是`tolerances.json`文件中的NOT ——它们与承诺的文件进行分级,而不是重播,所以E8-04的重播容忍机制不适用. 在看到剩余部分后,没有哪个波段被调谐;每个波段都来自录音机自己的磁盘化,一个模型级差异预算,或者比较的物理,记录时观察到的数值被列出.

| 检查 | 单位 | 带 | 观察 | 依据(摘要) |
|---|---|---|---|---|
| `ascent.event.meco` / `.fairing_jettison` | 编号 | 2.0 | 0.0 / 0.0 | 一个OpenTsiolkovsky输出-网格间隔;事件从原始推力/质量列重新提取 |
| `ascent.event.separation` / `.seco` | 编号 | 2.0 | 1.0 / 1.83 | 同样; SECO 坐在一个附加的事件样本上,所以最近的网格样本离它只有一步远 |
| `ascent.separation_window_position` | m | 5000 | 4.7 | gfold vs 原始 - OT 碎片覆盖共享无动力窗口:环境 - 模型 预算 |
| `ascent.separation_window_velocity` | 小时/秒 | 5.0 | 0.002 | 相同的窗口, 速度 |
| `insertion.semi_major_axis` | m | 250 | −67 | 300 km /~ 12 min 的振荡(外部) v J2- 表示(链)元素 |
| `insertion.eccentricity` | − | 5e-6 | 1.2e-6 | 同一元素-公约差异 |
| `insertion.inclination` / `.raan` | (单位:千美元) | 0.05 / 0.05 | −0.0002 / −7.7e-5 | 平均游离津贴,~100×差值 |
| `ascent.seco_state_position` / `_velocity` | m/ m/s 时间 | 1 / 0.01 | 5.8e-11 / 9.4e-13 | 密封记录往返于固定框桥; 浮动 64文本序列化 |
| `analytic.seco_orbit_energy` (Δa) | m | 1.0 | 0.005 | 在一个状态上相对于活体身份; 双圆比例 |
| `analytic.seco_orbit_eccentricity` | − | 1e-8 | −6.6e-10 | 不变 |
| `analytic.seco_orbit_period` | 编号 | 0.5 | 8e-6 | 不变 |
| `analytic.seco_orbit_inclination` | (单位:千美元) | 0.05 | 0.018 | 覆盖录制时看到的外部元素阴性散射 |
| `ascent.stage2_dv_ledger` | − | 0.15 | 0.122 | Tsiolkovsky 理想 -\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\} 楼阁真空的重力+破损损失预算 上升段 |
| `recovery.coast_arc_position` | m | 1000 | 575 | 精确的双体对J2 +重力梯度轨道传播 在~207海岸上空 |
| `recovery.coast_arc_velocity` | 小时/秒 | 10.0 | 3.77 | 相同的弧, 速度 |
| `recovery.event.landing_burn_ignition` | 编号 | 2.0 | 0.5 | 记录器网格:事发后首次供电 助推器 样本 |
| `recovery.event.touchdown` | 编号 | 1.0 | 0.0 | 密封记录必须完全一致; 1 s 的浮游圆形前厅 |
| `recovery.touchdown_gates` | − | 1.0 | 0.70 | design-input gates (target.yaml limits), composite utilisation ≤ 1 |
