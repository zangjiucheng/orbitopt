# Flight Lab

Flight Lab模拟火箭上升段和助推器回收. 在Mission Control中,切换到**Flight Lab**来设计,重播和比较运行.

![Flight Lab replaying a two-stage ascent with telemetry and checks](https://raw.githubusercontent.com/zangjiucheng/orbitopt/main/docs/assets/flightlab-replay.png)

## 快速启动

安装 [compute environment](https://github.com/zangjiucheng/orbitopt/blob/main/docs/quickstart.md#installation), 然后获取绑定的 OpenTsiolkovsky 二进制并运行包含的示例 :

```bash
python scripts/fetch_opentsiolkovsky.py
orbitopt flight validate src/orbitopt/data/flightlab/scenarios/leo_two_stage.yaml
orbitopt flight simulate src/orbitopt/data/flightlab/scenarios/leo_two_stage.yaml \
  --vehicle src/orbitopt/data/flightlab/vehicles/synthetic_two_stage.yaml -o run/
orbitopt flight to-scene run/ -o scene.json
orbitopt view scene.json
```

目标导出,优化,模拟和回收选项见[command reference](https://zangjiucheng.github.io/orbitopt/api/cli/flight/).

## 特派团管道

`flight-lab`任务型通过统一管道运行上升段. 它的交接包括帧,历元,位置,速度和质量;失败产生最佳警告. `two-stage-leo`参考情景也分级了舞台界限,注入和传播的停车轨道:

```bash
orbitopt run src/orbitopt/data/flightlab/mission_leo_two_stage.yaml -o run/scene.json
orbitopt run src/orbitopt/data/mission_two_stage_leo.yaml
```

Mission Control显示一个时间表上的阶段和交接检查. 后插入显示器的海岸不是验证. 就参考链而言,只有上升段系被分级;两体的海岸连接了中转缺口。 `ascent.mode: run_dir`重放一个已有的运行而不是优化它. 报告含义见[verification](verification.md)。

## 贞操水平

| 模型 | 范围 |
|---|---|
| 上升段 | OpenTsiolkovsky 3-DOF point mass with a prescribed attitude program and Cd(Mach). |
| 恢复 | 3 - DOF 地球固定点质量与J2,旋转,大气层,发动机和事件探测. Monte Carlo对每一个分散的样本进行了重新计划. |
| JSBSim 6 - DOF QX | Optional `flight simulate --backend 6DOF` backend. The packaged illustrative vehicles use zero aero; it is outside the optimization loop. See [ADR 0001](adr/flightlab-0001-jsbsim-6dof.md). |
| 原生6 - DOF研究 | 一个单独的RK4硬体参考厂,具有系数表空气动力学和TVC跟踪;它不是JSBSim的后端. |

<span id="native-aero-model-and-the-reference-plant-f-02"></span> <span id="tvc-actuator-and-attitude-tracking-f-04"></span> <span id="6-dof-vs-3-dof-gravity-turn-ascent-f-05"></span> <span id="two-stage-separation-and-upper-stage-ignition-f-06"></span> <span id="mission-outputs-selectable-backends-reports-and-mission-control-f-07"></span>

## 6 - DOF研究.

原生植物使用示意图的Mach/angle系数表,US76,地球旋转,J2,推力,以及一个滞后和速率有限的 gi. 它的基本跟踪器不控制卷或代表一个完整的自动驾驶. 在[gravity-turn study](https://github.com/zangjiucheng/orbitopt/blob/main/docs/benchmarks/fl605_gravity_turn_ascent.json)中,6 - DOF减3 - DOF的最大差异是海拔2.13公里,速度2.08米/秒,而0.47°为航迹角.

两阶段研究应用了平等而相反的分离冲动,然后独立地传播每个阶段. 将其原生点质量和刚性体后端与:

```bash
orbitopt flight mission --backend BOTH -o runs/f07-mission
```

产出包括FlightResult运行,交割,验证报告,场景和比较报告. 在[recorded comparison](https://github.com/zangjiucheng/orbitopt/blob/main/docs/benchmarks/fl607_mission_outputs.json)中,末端的顶级与235千米不同:6 - DOF的顶级是惯性指向的,而3 - DOF 模型遵循其规定的方案. 这些数字是假设结果,不是一般准确的界限。

## 硬门

运行记录框架,时代,单位和出处. 轨道需要申报的速度,近地点和轨道倾角检查;着陆需要所有申报的位置,速度和负载限制. 判断通过CLI可以复制,失败包括理性代码. 通行证只适用于规定的模型和限额。

## 什么不是模拟

默认的上升段没有6 - DOF的态度动态. 模型省略CFD-级空气动力学,结构载荷,推进剂散射,级碰撞,风场和射程安全. 回收样本可以包括一个稳定的风向矢量,但没有扰动;指导性重新规划不是机载计算机模拟.

## 车辆数据、来源和不确定性

[vehicle catalog](https://github.com/zangjiucheng/orbitopt/tree/main/src/orbitopt/data/flightlab/vehicles/)型机车包括**说明**型合成机车和猎鹰-9-级**公众估计数**型,而非SpaceX正式性能数据. 数字字段记录来源,日期和来源;不确定波段可以提供蒙特卡洛默认. 设计师展示了车辆的出处.

## 恢复的不确定性(R-04)

以目标和车辆进行回收散射:

```bash
orbitopt flight montecarlo src/orbitopt/data/flightlab/scenarios/leo_two_stage.yaml \
  --vehicle src/orbitopt/data/flightlab/vehicles/synthetic_two_stage.yaml \
  -n 32 --seed 7 --capture-tracks 240 -o montecarlo.json
```

`--uncertainty unc.yaml`分别配置风,密度,推力,Isp,拖和分离态分布. 例如:

```yaml
uncertainty:
  thrust: {enabled: true, distribution: uniform_pct, half_width_pct: 5.0}
  wind: {enabled: true, distribution: normal, sigma_east_m_s: 4.0,
         sigma_north_m_s: 3.0, sigma_up_m_s: 0.5}
```

报告记录了分发情况。 风速进入接受重飞,而非助推器优化推出. 采样是一个固定种子的定型,独立于`--jobs`.

## 蒙特卡洛统计(R-06)

`montecarlo.json`报告故障班,威尔逊成功率间隔和每源敏感. 联合分散协会并不是孤立的因果关系。 `--solo-sensitivity` 单向每个活动源重播,每个源需增加一个样本通过. `--capture-tracks N`记录到N站点 - ENU每个样本对信封地块的点数不改变图纸.

## 名义/分散比较(R-07)

In the recorded 32-sample comparison (seed 1, identical dispersions and
referee), the baseline landed 14 and the newer objective landed 17. The
latter's Wilson interval was [0.36, 0.69]; position failures rose from 4 to 7.
See [ADR 0003](adr/flightlab-0003-landing-aware-boostback-objective.md) and
[the recorded results](https://github.com/zangjiucheng/orbitopt/blob/main/docs/benchmarks/r07_nominal_dispersed.json) for the setup
and failure breakdown.

关于实施和可复制性的细节,见[JSBSim findings](flightlab-jsbsim-spike-findings.md),[regression CI](flightlab-regression-ci.md)和[landing guidance](adr/flightlab-0002-closed-loop-landing-guidance.md).
