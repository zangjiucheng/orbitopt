# ADR 飞盘 -0002 :闭路航站楼着陆制导+柔和的FL-302 交接

- **Status:** Accepted
- **Date:** 2026-11-12
- **Deciders:** Flight Lab E3 (Epic #195), FL-305 (#214), follow-up to FL-304 (#205)
- **Code:** `src/orbitopt/flightlab/recovery/landing.py` (`TerminalGuidance`,
  `terminal_guidance_command`, `fly_landing_burn_closed_loop`),
  `src/orbitopt/flightlab/recovery/pipeline.py` (`NOMINAL_PLAN_GUESS`,
  `plan_options["landing"]`)
- **Evidence:** `docs/benchmarks/fl305_recovery_montecarlo.json`
  (`benchmarks/bench_fl305_recovery_montecarlo.py`),
  `tests/test_flightlab_simulate_recovery.py`,
  `tests/test_flightlab_recovery_terminal_guidance.py`

## 背景和问题

FL-304 wired the full recovery chain (simulate → FL-302 boostback/entry →
FL-303 G-FOLD landing burn), but from the textbook RTLS separation (8 km
east, 40 km up, 250 m/s, 65 t) the nominal recovery came back red.

- FL-302 handed the landing burn a corridor state at h ≈ 3.0 km, 307 m/s
  down, **1.76 km west of the pad**, 41.1 t (11.1 t propellant).
- FL-303驾驶一个对流计划开舱. 模型没有看到拖动,
  它在引擎丢失时使用插入-海拔 推力信封
  ~ 10% 其推力到接近地面的环境压力. 此处重新测量:
  触地得分,在**- 107.7米/秒,331米** 从垫。
- FL-304's green E2E 仅仅因为取代了FL-303的认证而通过.
  名义着陆(200 m,10 m/s,54 t)为登陆台阶.
- 发动机的40%地板是~1.5×车辆的重量接近地面(没有)
  ),所以最后几秒钟没有出错的余地.

## 所审议的备选方案

1. **在循环中重解 G-FOLD**(将SOCP从宣传中重新规划).
   高度触发状态 飞行时间缩短 重新计划停机
   高处). FL-304测得~4.5到~30 m/s,倾斜过40 °. A.
   任务中的原型更进一步: 13 重解、封装和拖动
   偏差在当前高度下重新评价,油门下垂
   到0.12,以便车辆可以徘徊。 它仍然在** ** − 8.3 m/s,
   4.9 离 off,653 千克低于储备** **. min-fuel arcs 是砰-海岸爆炸:
   他们把节流阀固定在极限 所以每一次重新计划
   用于吸收拖曳和推力不匹配的权限。 在地面附近,他们
   海岸太长,然后在全推力弹跳(升至11,陷入).
   被拒绝。
2. **合成车号**(降低登陆储备或节流阀)
   楼层). 这会让后备队或盘旋检查变得绿色
   更换FL-301/FL-302/FL-303认证车. 不需要(见
   因此YAML号机车是**不变**:4,200公斤储备,油门
   0.4 – 1.0,一台发动机.
3. **带边距的闭路终端指导**:分析法,
   重新规划了来自被宣传状态的每一个指导步骤, 与
   throttle authority held in reserve. **Chosen.**
4. **Gentler FL-302 handoff**: hand over closer to the pad. **Chosen, in
   combination.** FL-302's corridor penalty is `(miss / radius)²` with an
   8千米半径,所以一个1.76千米错失了它~0.05,它没有理由修复
   这个 试验了更紧的走廊半径(1公里,0.6公里)。 在那个
   18 – 30 – 评估 Nelder-Mead 预算 选择者不能步
   0.5的量化增速持续时间,所以结果是
   `CORRIDOR_MISSED`. 修补有效是温暖的开始。

## 决定

**(a) Closed-loop terminal guidance in FL-303** (`TerminalGuidance`, the
default landing step of the pipeline). Every integration step
(`dt = 0.1 s`), from the propagated pad-ENU state:

- **Vertical: throttle-limited hoverslam.** The guidance computes
  `a_req = (v_z² − v_td²) / 2h`, 持续减速达到
  `v_td = 1.5 m/s`在垫层. 直到`a_req`到达
  `design_throttle = 0.9` of the *sea-level* thrust minus gravity, then
  接下来是`a_req + g`. 因此,10% 的拖曳权限,环境推力
  和质量错误。 晚烧,硬烧也是重力损失最低的烧伤.
- **Lateral: ZEM/ZEV divert.** The lateral command is
  `6·ZEM/t² − 2·ZEV/t`,使用垂直剖面的逐时减a
  2 s settle window. Inside that window a PD loop nulls the residual drift.
  If a burn started now would need more than 70 % of the tilt-limited
  横向权威,发动机灯光提前(*偏转感知点火*).
- **Attitude in the loop.** The lateral command is clipped to a 30° thrust
  锥形,位于40 ° 触地得分检查之下. 推力为上限为4.5 g(检查)
  限制为5 g).
- **截断曲调为无头T/W.** 一旦`a_req`跌落到什么之下
  40 % floor delivers, the profile can no longer be tracked. Below 3 m the
  一旦弹道到达速度,引擎就永远关闭
  `sqrt(v_z² + 2gh)` is ≤ 2.2 m/s. One ignition, one cutoff, no restarts.

The insertion SOCP is still solved. It remains the feasibility certificate:
an infeasible state is `LANDING_INFEASIBLE` exactly as before, with no
flight. **The referee is unchanged.** One propagation through FL-301 (drag,
J2, rotation, min-throttle clamp, the caller's dispersed physics) is graded
by FL-303's `landing_touchdown_checks`, which the guidance never sees.
`plan_options={"landing": {"mode": "single"}}` keeps FL-303's open-loop flow
available.

**(b) A gentler FL-302 handoff.** `NOMINAL_PLAN_GUESS` changes its boostback
duration from 19.1 s to 17.75 s. FL-302 keeps it as its accepted script: it
is feasible, uses less boostback propellant, and the 18-evaluation
refinement stays on it. The handoff becomes h = 3.0 km, 306 m/s down,
**0.35 km** off, **41.7 t** (11.7 t propellant).

## 结果

Nominal chain (`test_slow_e2e_nominal_recovery_is_green_through_the_real_chain`,
real simulate → FL-302 → FL-303, nothing substituted):

| 移交 | 着陆流量 | 触地得分 v _ z | 垫底。 | 横向 | 最大倾斜 | 准备金差值 | 判决 |
|---|---|---|---|---|---|---|---|
| FL-304(1.76 km,41.1 t) XQ. | 单一计划 | - 107.7米/秒 | 331 m | 72米/秒 | 37.5° | +1839公斤 | 红色 |
| FL-304(1.76 km,41.1 t) XQ. | 关闭循环 | - 1.6米/秒 | 224 m | 39米/秒 | 30.0° | - 514公斤 | 红色( 偏移预算) |
| FL-305(0.35 km,41.7 t) XQ. | 单一计划 | - 85.0米/秒 | 27 m | 16米/秒 | 29.2° | +2921公斤 | 红色 |
| **FL-305 (0.35 km, 41.7 t)** | **closed loop** | **−1.6 m/s** | **0.6 m** | **0.04 m/s** | **19.3°** | **+992 kg** | **green** |

Limits: |v_z| ≤ 3 m/s, lateral ≤ 1 m/s, pad distance ≤ 25 m, tilt ≤ 40°,
≤ 5 g (measured 3.5 g), and propellant ≥ the 4,200 kg reserve. Neither
change alone is enough. The guidance is what gets touchdown speed under
control. The handoff is what brings the divert inside the propellant
budget.

### 蒙特卡洛在 / 之后(FL-304 牵引,默认散射,种子 1,n = 32)

`docs/benchmarks/fl305_recovery_montecarlo.json`. 这两种跑步都是确定每个种子。 "forefore"是FL-304作为合并(旧的温暖开始,单计划);"after"是FL-305默认.

| 理由 | (FL-304) | 在(FL-305)之后 |
|---|---|---|
| `LANDED` | 0 | **14** |
| `LANDING_POSITION_OUT_OF_TOLERANCE` | 23 | 4 |
| `LANDING_PROPELLANT_RESERVE_LOW` | 0 | 5 |
| `INSUFFICIENT_PROPELLANT_RESERVE` (FL-302) | 8 | 8 |
| `CORRIDOR_MISSED` (FL-302) | 1 | 1 |
| **success rate** | **0.000** | **0.438** |

Landed subset (after): pad distance p50 0.86 m / p95 2.2 m, lateral speed p95
0.19 m/s, vertical speed −1.50 m/s.

每一个失败都有一个诚实的理由:

- **9 never reach the landing burn.** FL-302 rejects them (separation
  分散 ± 800 m / ± 4 m/s),前后相同.
- **4 position failures.** These handoffs end up kilometres off, beyond
  30 ° 圆锥能从3公里处分道扬镳
- **5 reserve failures.** The divert or density dispersion costs more than
  ~ 1 t 的推进剂,额定叶片在储备之上.

进一步提高成功率是一个FL-302/车辆测距问题,而不是终端指南。

## 后果

- 管道的默认着陆步骤是闭环. 着陆
  假设字符串是这样说的,运行的 `landing_start` 事件仍然
  携带插入计划。
- `NOMINAL_PLAN_GUESS`(也是蒙特卡洛的温暖开始)是17.75 s.
  脚本。
- FL-304的猴斑绿色试验现标注为**仅线条**. 他们
  将合并和统计管道与单计划流量和a
  替代判决。 物理绿色的E2E 取代了什么。
- FL-302的走廊处罚仍然几乎盲目失守
  走廊半径。 将走廊与着陆的火力相撞
  如果分散的成功率重要的话,预算是自然的下一步。
- 范围以外,没有改变:FL-302 opimiser的密度分散,以及
  the OpenTsiolkovsky stub scenario's `INSUFFICIENT_PROPELLANT_BOOSTBACK`
  判决。
