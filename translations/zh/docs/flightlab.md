# Flight Lab

Flight Lab模拟上升段和助推器从代码或CLI恢复. Mission Control的**Flight Lab**模式显示保存在其共享的3 D视图,时间线和检查器中运行.

![A Flight Lab run with telemetry, events and verification checks](https://raw.githubusercontent.com/zangjiucheng/orbitopt/main/docs/assets/flightlab-replay.png)

## 快速启动

安装 [compute environment](https://github.com/zangjiucheng/orbitopt/blob/main/docs/quickstart.md#installation),获取绑定的二进制,然后运行被包的点质量示例 :

```bash
python scripts/fetch_opentsiolkovsky.py
orbitopt flight validate src/orbitopt/data/flightlab/scenarios/leo_two_stage.yaml
orbitopt flight simulate src/orbitopt/data/flightlab/scenarios/leo_two_stage.yaml \
  --vehicle src/orbitopt/data/flightlab/vehicles/synthetic_two_stage.yaml -o runs/leo/
orbitopt
```

选择 **Flight Lab** 打开运行 。 要导出独立场景, 请使用 `orbitopt flight to-scene runs/leo/ -o scene.json` 。 [command reference](https://zangjiucheng.github.io/orbitopt/api/cli/flight/)列出所有选项.

## 查看运行于 Mission Control

- **运行列表 :**扫描 `./runs`, `ORBITOPT_FLIGHT_RUNS`(平台路径分隔符)
  还记得**添加文件夹...**的位置 **刷新**找到新的运行;
  `orbitopt launch -o DIR`生产列出的`DIR/ascent`运行.
- **共享视图 :** 选择加载 3 D 场景,时间线, 检查员和
  验证标签. 返回特派团恢复了以前的任务。
- **飞行标签 :**发射投入,目标/实现轨道,判决,可寻.
  事件、故障解释和回收细节。
- **Telemetry:** altitude, speed, dynamic pressure, load, mass and throttle;
  点击图表查找时间表 。 **比较运行...**比较两跑.

目标和车辆在YAML或通过Python设计师API进行编辑;桌面模式使运行而非配置或飞行.

## 车辆视图

打开**Flight Lab → Vehicles**并选择编目车. 这个只读的页面显示了它的舞台轮廓,质量,推力,理想的QΔv,恢复储备和申报的推力/stage调度. 配置文件在可用时使用已申报的几何;否则它们会被标记为**图表, 不用于缩放** 。 来源、日期和说明的不确定性。

[![Vehicle view showing the illustrative RTLS stack, declared thrust schedule and recorded E5-05a flight comparison](https://raw.githubusercontent.com/zangjiucheng/orbitopt/main/docs/assets/flightlab-vehicle.png)](https://raw.githubusercontent.com/zangjiucheng/orbitopt/main/docs/assets/flightlab-vehicle.png)

要将定义与飞行相比较,请选择 **运行**中保存的运行,然后返回 **车辆**. 该时间表增加了飞行事件标记;**宣告为飞行**列出了时间和质量差异. 车头说明运行是否使用了准确的车辆文件或其摘要不同. 缺失值为`—`;差异本身并不是验证判决.

截图使用说明性`synthetic_two_stage_rtls`车和录制的[E5-05a ascent](computed-mission.md). 理想XQV排除引力,拖动和引导损失. 编辑YAML型机车改变设计;返回**运行**型机车,用于3 D型机车的飞行视野和遥测.

## 特派团管道

`flight-lab` runs an ascent through the unified pipeline. `two-stage-leo`
also grades station order, staging, mass, orbit insertion and optional
propagation/recovery:

```bash
orbitopt run src/orbitopt/data/flightlab/mission_leo_two_stage.yaml -o run/scene.json
orbitopt run src/orbitopt/data/mission_two_stage_leo.yaml
```

分离后,上层轨道和助推器轨道分别设有时间线道和边界检查. 中央重力+J2海岸桥梁,架设缺口; 后插入的**显示海岸**不是验证. `ascent.mode: run_dir`重播录制输入,而不是优化一个上升段.

The packaged `src/orbitopt/data/mission_two_stage_leo_recovery.yaml` is the
**E2-04 frozen integration fixture**: two-sample, illustrative ascent arcs
with a prescribed 400 km insertion and 35 t of booster propellant at
separation. Its recovery is simulated, but ascent is not re-flown and parking
propagation is explicitly out of scope. It is not evidence of one physically
propagated launch-to-orbit-and-return trajectory. See
[regression procedure](https://github.com/zangjiucheng/orbitopt/blob/main/docs/flightlab-regression-ci.md#e2-04-the-frozen-dual-branch-reference-mission-task-257)
and the [acceptance results](acceptance-e1-e4.md). For the
recorded **computed-ascent RTLS mission**, use the
[E5-05 workflow](computed-mission.md): it has a fixed-program OT ascent,
one shared separation handoff and optional cross-environment parking
propagation.

<span id="fidelity-levels"></span>

## 后端身份和忠诚程度

验证输入和`scene.flightRun.backend`:

| 后端 | 范围 | 边界 |
|---|---|---|
| `opentsiolkovsky` | 限定态点-质量 上升段,Cd(Mach),GCRS输出. | 默认 上升段/优化血统;没有姿态动态或TVC. |
| `native-pointmass` | 原生US76 / J2 / 旋转-地球双胞胎有动静的态度. | 原生比较植物,区别于OpenTsiolkovsky. |
| `native-rigidbody` | 原生硬体,系数表空气和落后/速率限制的TVC跟踪. | 说明参考厂;无完整卷自动驾驶. |
| `jsbsim` | JSBSim僵硬的身体,有活体质量/CG/inertia,并被赋予TVC,ECEF输出. | 基座悬索式飞机为零气;`spike_stage_aero`是生成的含气参考. Opt-in,位于上升段优化外. |

`flight simulate --backend opentsiolkovsky|jsbsim`为前者搭载YAML,后者搭载**JSBSim 情景 YAML**. 关于含空气的参考文献(需要`orbitopt[sixdof]`):

```bash
orbitopt flight simulate src/orbitopt/data/flightlab/scenarios/leo_two_stage.yaml \
  --backend jsbsim --vehicle src/orbitopt/data/flightlab/sixdof/e104_aero_ascent.yaml \
  --no-refly -o runs/e104/
```

这种持有的垂直上升段演练机身;400公里轨道-目标检查预计会失败. `--no-refly`离开独立物理 验证未评估. `flight mission --backend native-pointmass|native-rigidbody|BOTH`运行本地比较. 遗产 `3DOF` / `6DOF`别名在原切入点内解析;它们不是犬类输出身份.

### 相符合的实际情况(E1-01)

The [parity document](https://github.com/zangjiucheng/orbitopt/blob/main/src/orbitopt/data/flightlab/parity/spike_stage_parity.yaml)
drives native stage construction and checks packaged JSBSim mass, inertia,
propellant, mounting and aero geometry. Since E6-01 the engines share the
propulsion law too: a choked `T_vac/(Isp_vac*g0)` = 599.083 kg/s flow through
the same physical 2 m^2 exit, so both sides pay the same ambient back-pressure
debit and agree on 1.6774 MN of pad thrust. The generated production airframe
adds the aero tables and cancels the fixed-AERORP/live-CG force couple. See
[ADR 0009](adr/flightlab-0009-engine-and-mass-flow-parity.md) for the aligned
residuals, [ADR 0007](adr/flightlab-0007-production-jsbsim-aero-path.md) for
the aero path, [ADR 0001](adr/flightlab-0001-jsbsim-6dof.md) for the base spike.

回收仍是一个单独的地球固定3 - DOF链带J2,旋转,大气层和发动机/活动模型. 选择 6 - DOF 上升段不会使恢复六维.

## 验证判决

| 运行字段 | 含义 |
|---|---|
| `backend_self_check` | 生产后端自己的支票。 |
| `physics_verified` | 使用 tudatpy 的独立插入/海岸重飞; JSBSim ECEF 输出首先连接到 GCRS. |
| `optimizer_converged` | 乐观的判决,当在场。 |
| `mission_constraints_passed` | 宣布飞行目标限制。 |

`null`表示未分摊. 这些领域是独立的;自我检查不是独立的重新飞行。 见[ADR 0004](adr/flightlab-0004-independent-physics-verification.md). 任务卷起后保留了这面旗帜作为独立的模型证据. 假旗故障轨道注射及全任务门;无效仍未评估. 见[acceptance results](acceptance-e1-e4.md).

<span id="native-aero-model-and-the-reference-plant-f-02"></span> <span id="tvc-actuator-and-attitude-tracking-f-04"></span> <span id="6-dof-vs-3-dof-gravity-turn-ascent-f-05"></span> <span id="two-stage-separation-and-upper-stage-ignition-f-06"></span> <span id="mission-outputs-selectable-backends-reports-and-mission-control-f-07"></span>

## 6 - DOF研究.

目前的记录包括E6-01引擎/质量流对齐和E6-04控制采样. 这些是申报的学习车辆和管制员,而不是任何合格的生产车辆:

| 学习 | 记录结果 | 范围/来源 |
|---|---|---|
| F-05 / E1-02 | 动力姿态错误 4.77 °峰值,α 14.42 °,最大-71.2 kPa. | 动力门和海岸门不同;后期的62.6 °α峰位于近真空海岸. [Record](https://github.com/zangjiucheng/orbitopt/blob/main/docs/benchmarks/fl605_gravity_turn_ascent.json). |
| F-07 / E1-03 | 原点质量:225.3 / 228.5 km peri/apo;硬体:220.5 / 226.5 km,在± 25 km范围内. | 共享关于已披露的 1 t 干 / 7 t 推进剂上级的反馈定律,与 3 ° gimbal;非原 F-06 等价. [Record](https://github.com/zangjiucheng/orbitopt/blob/main/docs/benchmarks/fl607_mission_outputs.json). |
| E6-02 | 引导生产 JSBSim 上升段与 TVC, MECO, 中转和延迟点火; 动力错误 0.44 ° 峰值, alpha 11.7 °. | Its 70 t stack and spike-engine children are illustrative; native and JSBSim controller differences are attributed, not assumed away. [ADR 0010](adr/flightlab-0010-guided-jsbsim-ascent.md), [record](https://github.com/zangjiucheng/orbitopt/blob/main/docs/benchmarks/e6_02_guided_ascent.json). |
| E6-03 | Six of 17 declared scenario points insert on both native backends. | Sweeps altitude, inclination, vehicle and actuator sets; misses distinguish vehicle, terminal-law and actuator limits. [ADR 0011](adr/flightlab-0011-guidance-generalisation-envelope.md), [matrix](https://github.com/zangjiucheng/orbitopt/blob/main/docs/benchmarks/e6_03_guidance_matrix.json). |
| E6-04 | 每一步抽取一次国家指导/控制样本,在RK4阶段进行。 | The study justifies `dt_s = dt_c_s = 0.02`; this is not a universal backend step. [Sampled-control ADR](adr/flightlab-0008-sampled-data-control-and-rk4-plant.md), [sweep](https://github.com/zangjiucheng/orbitopt/blob/main/docs/benchmarks/e604_step_convergence.json). |

运行本地双后端任务并打开其保存的产出 :

```bash
orbitopt flight mission --backend BOTH -o runs/f07-mission
orbitopt
```

For the guided JSBSim study (requires the six-DOF dependency):

```bash
PYTHONPATH=src python benchmarks/bench_e6_02_guided_ascent.py -o runs/e602.json
```

研究撰写了测量JSON,而不是Flight Lab运行目录. [API reference](https://github.com/zangjiucheng/orbitopt/blob/main/docs/api.md#mission-and-qualification-interfaces)涵盖其情景,跑步和比较功能.

## 硬门

轨道和触地得分检查适用申报的速度、位置、轨道、载荷和推进剂限制。 报告记录帧,时代,单位,出处和故障原因代码. 通行证仅到达模型检查和限制;[full-mission phase verdicts](https://github.com/zangjiucheng/orbitopt/blob/main/docs/verification.md#full-mission-verification-gate-e2-03-task-256)显示没有证据或分支失败的地方.

## 什么不是模拟

默认 上升段是点质量. 模型省略CFD-级空气,结构载荷,slosh,级碰撞,扰动和射程安全. 恢复可能使用一个稳定的分散风向器;其重新规划不是机载计算机模拟。

## 车辆数据、来源和不确定性

[vehicle catalog](https://github.com/zangjiucheng/orbitopt/tree/main/src/orbitopt/data/flightlab/vehicles/)包含一个说明性合成飞行器和Falcon -9-级公共估计,而不是官方SpaceX性能数据. Fields记录来源/日期/证明;不确定波段可以提供蒙特卡洛默认.

## 恢复的不确定性(R-04)

```bash
orbitopt flight montecarlo src/orbitopt/data/flightlab/scenarios/leo_two_stage.yaml \
  --vehicle src/orbitopt/data/flightlab/vehicles/synthetic_two_stage.yaml \
  -n 32 --seed 7 --capture-tracks 240 -o montecarlo.json
```

通过 `--uncertainty unc.yaml` 独立配置源 :

```yaml
uncertainty:
  thrust: {enabled: true, distribution: uniform_pct, half_width_pct: 5.0}
  wind: {enabled: true, distribution: normal, sigma_east_m_s: 4.0,
         sigma_north_m_s: 3.0, sigma_up_m_s: 0.5}
```

风波影响接受重飞,而非助推器优化推出. 采样是固定种子定型,独立于`--jobs`.

## 蒙特卡洛统计(R-06)

Reports include failure classes, Wilson success-rate intervals and per-source
sensitivity. Joint-dispersion associations are not isolated causal effects.
`--solo-sensitivity` adds one sample pass per active source; `--capture-tracks N`
retains at most N ENU samples per trajectory without changing random draws.

## 名义/分散比较(R-07)

[32-sample evidence](https://github.com/zangjiucheng/orbitopt/blob/main/docs/benchmarks/r07_nominal_dispersed.json)记录14每臂登陆. 基线假设无限制TVC带宽;堆积飞行20 °/s shot + 0.2 s 滞后补偿. 它们是不匹配的精算师。 堆栈的标称触地得分是0.03距离垫;位置故障从4上升到7. 见[ADR 0006](adr/flightlab-0006-rate-limited-landing-guidance.md)和[ADR 0003](adr/flightlab-0003-landing-aware-boostback-objective.md).

## 联合机会限制规划(E3-02)

`plan_boostback_entry(joint_chance=...)`用联合燃料取代R-01软客观术语,分流/散射和着陆概率限制,解为**受罚优化**(Nelder上最恶劣的违反精确处罚-  Mead 搜索——而不是一个受约束的解析器. 自E7-01以来,强制判决得到了执行,而不是刑罚的暗示:最恶劣的违反规定超过`JointChanceSpec.violation_tolerance`的计划以`JOINT_CHANCE_VIOLATED`的理由代码返回`feasible=False`。 所记录的32-样本联合臂着陆地17;假设见[ADR 0005](adr/flightlab-0005-joint-recovery-chance-constraints.md)和[paired results](https://github.com/zangjiucheng/orbitopt/blob/main/docs/benchmarks/e3_02_joint_recovery.json). 其规划概率为代机,而非保证飞行成功率.

## 强力蒙特卡洛资格(E3-03).

[four-seed study](https://github.com/zangjiucheng/orbitopt/blob/main/docs/benchmarks/e3_03_robust_montecarlo.json)使用168配对的样品,每只手臂配对的图纸,并宣布20 °/s / 0.2 的导体物理学:

| 武装 | 着陆/168 | 95% 威尔逊间隔 |
|---|---|---|
| FL-305基线 | 45 | 20.7–33.9% |
| E3-01赔偿 | 49 | 22.8–36.4% |
| R-01目标 | 55 | 26.1–40.2% |
| E3-02 联合机会 | 56 | 26.6–40.8% |

根据基线,McNemarp的精确值分别为0.125,0.0309和0.0192. 这些是**未调整配对**测试,不是家庭意义索赔. E3-02对R-01仍未解决(两个对一个不和谐的着陆);配对的只着陆的散射试验拒绝任何东西. [benchmark script](https://github.com/zangjiucheng/orbitopt/blob/main/benchmarks/bench_e3_03_robust_montecarlo.py)记录了所有样本行,故障类和敏感性. 在本次验收审查中,从这些列中重新计算了统计数据;672航班没有重新运行。

**The E3-02 row predates E7-01.** It measures the old penalty-only mode.
With feasibility enforced, E7-04 re-flies the same draws and lands **35/168**;
99 plans are rejected as `JOINT_CHANCE_VIOLATED`. The baseline comparison has
McNemar p = 0.110. The old p = 0.0192 does not describe today's enforced mode.

## Probability calibration of p_land (E7-02)

[calibration report](benchmarks/e7_02_probability_calibration.md)重用168记录成对样本. 0.5 – 0.8 bin预测0.656但登陆0.076;0 – 0.5 bin登陆0 / 42. 在0.8以上, 平均预测值属于观测到的威尔逊间隔, 其小的高吸附样品留下了广泛的不确定性。 这只支持记录的情景/分散集,而不是保证成功的概率. API:`orbitopt.flightlab.recovery.calibration`.

## 联合风险模型(E7-03)

The [joint-risk report](benchmarks/e7_03_joint_risk.md) rejects independent
fuel/divert failure channels (Fisher p = 0.010). A Gaussian copula fitted on
three other arms uses ρ = −0.273; on the held-out joint arm, Brier score
improves 0.222 → 0.214. The mid-band remains overconfident. The shipped
`JointChanceSpec.landing_dependence_rho` default is **0.0** (the product);
the fitted constant is opt-in. API: `orbitopt.flightlab.recovery.joint_risk`.

## 故障驱动优化(E7-04).

The [paired re-fly report](benchmarks/e7_04_failure_driven.md) compares the
same 168 dispersion draws with TVC lag compensation and an optional lateral
cutoff gate. Lateral-speed failures fall 24 → 18 and landings rise 45 → 49;
this is **not significant after Holm correction** (raw category p = 0.0703,
adjusted 0.3516). The gate alone reproduces baseline verdicts; compensation
carries the measured change. The enforced joint arm stays 35/168 with or
without the fix. Default cutoff and corridor-aim knobs remain off; see
`orbitopt.flightlab.recovery.failure_driven` for the paired analysis.
