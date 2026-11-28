---
title: Computed mission workflow
parent: Flight Lab
nav_order: 0
description: Replay the recorded LEO and RTLS mission, add propagation across environments, and inspect evidence.
---

# 计算任务工作流程

按其车辆、飞行目标和证据范围选择参考。 名义着陆,联合物理传球,独立-模型一致性和分散可靠性是单独的结果.

| 工作流程 | 什么运行 | 证据边界 |
|---|---|---|
| E2-04 | 合成上升段端点+模拟回收. | 快速整合固定;没有计算 上升段或停车 轨道传播. |
| E5-05 重播 | 记录计算 上升段+新模拟回收. | 一个峡谷隔离状态;停车轨道传播仍为`not_checked`. |
| E5-05 跨环境 | 录音上升段+tudatpy停车轨道传播+gbold恢复. | 联合运行必须证明所有五个物理检查。 |
| E5-07 重播 | 有记录的星舰级计算 上升段,没有恢复阶段. | 物理学资格证明;报告恢复差距(`RECOVERY_GAP.md`),`full_mission`没有申报. |
| E5-07 跨环境 | 记录上升段+tudatpy任务段与停车轨道传播. | 轨道传播在外环境证明;联合门NOT-DECLARED设计. |
| E5-08设计变体 | 在库鲁和星座基地有动力进军RTLS. | DC7和Starbase通过名义上的联合物理门. 最初的差距仍然是消极的倒退。 [Results](e5-08-new-vehicle-report.md). |
| E8-01 证明人查询 | 将封存的记录与原始后端输出和封存的形式进行比较. | 没有新的飞行;验证提取、连续性和说明的参考比较。 |
| E12会合 | 计划并重播一个目标 和追逐者在LEO。 | 名义上的、独立的再飞行和散射结果涵盖了所宣布的参考设想。 [Replay and evidence](https://github.com/zangjiucheng/orbitopt/blob/main/docs/computed-mission.md#e12-orbital-rendezvous). |

## E12轨道会合

[Rendezvous](rendezvous.md)部分载有快速启动,相对运动理论,任务配置和验证完整报告.

**LEO-RDV-001**任务工作室重播的包装在`examples/runs/leo-rdv-001`下预先计算. 不计算堆栈打开场景 :

```bash
orbitopt view examples/runs/leo-rdv-001-scene.json
```

or add the `examples/runs/leo-rdv-001` directory under **Runs** in Mission
Control. Follow the target and chaser, inspect maneuvers and safety checks in
**Flight → Rendezvous**, and select range, relative speed or the radial,
along-track and cross-track components in **Telemetry → Plots**. The
[quickstart](https://github.com/zangjiucheng/orbitopt/blob/main/docs/rendezvous-quickstart.md#replay-it-in-flight-lab) shows how to
rebuild the run from the sealed plan and attach the matching TudatPy report.

[verification report](rendezvous-verification-report.md)为每个结果提供耐受性,计划消化和散射间隔.

| 证据 | LEO-RDV-001结果 |
|---|---|
| 名义计划 | **10.345 m/s**共计Δv. 抵达 预算 和整个轨迹安全检查通行证 [Safety constraints](adr/rendezvous-0002-safety-constraints-whole-trajectory.md). |
| 独立飞行 | tudatpy通过封存计划的名义支票. 抵达地点在距离目标距离1000米的Hold Point;整个轨迹最接近的接近点是**913.03 Zmm**(到达距离内的转移海岸). [Re-flight report](https://github.com/zangjiucheng/orbitopt/blob/main/docs/benchmarks/e12_05_independent_refly.json). |
| 驱散运动 | **14 of 264**搁置案件通过申报的压力等级(5.3%). [Campaign design](adr/rendezvous-0005-dispersion-campaign-not-a-safety-factor.md). |
| 独立稳健性 | **13 of 264**在包括被动应急检查时通过搁置案件(4.9%). tudatpy同意所有**23**选取的每例判决. [Re-flight report](https://github.com/zangjiucheng/orbitopt/blob/main/docs/benchmarks/e12_10_independent_reflight.json). |

对于tudatpy环境中的独立复飞,使用`orbitopt crossenv run --plan src/orbitopt/data/crossenv_e12_05_plan.yaml`. [independent verification design](adr/rendezvous-0003-independent-refly-tudatpy.md)和[CW validity study](benchmarks/e12_02_cw_validity.md)的限制参见模型.

## 重播和观看 RTLS任务

Use the [landing environment](https://github.com/zangjiucheng/orbitopt/blob/main/envs/README.md), with gfold **0.3.2**, from
the repository root:

```bash
python scripts/e505_reference.py verify
python scripts/e505_reference.py run --out runs/e505
python scripts/e505_reference.py compare runs/e505
orbitopt flight to-scene runs/e505/recovery/run -o runs/e505/scene.json
orbitopt view runs/e505/scene.json
```

The last command needs the local viewer. Computation may run on another
machine: copy the output directory back, then open its scene locally. For
telemetry, add `runs/e505/recovery/run` in Mission Control's **Runs** workspace.
The scene is the combined flight run; the unified mission report is
`runs/e505/mission.verification.json`.

The reference was recorded in `64f64a6`, using code `74f30c8`. Its
`synthetic_two_stage_rtls` vehicle flew a fixed-program OpenTsiolkovsky
ascent from Kourou and reached **265.5 × 278.1 km at 51.0°**. This replaces
the old blocked 400 km/CMA-ES recording proposal. `update` re-flies that
computed ascent and the fixed recovery recipe; routine replay does not
re-optimize or re-fly ascent. See the
[vehicle decision](adr/flightlab-0008-rtls-capable-vehicle-redesign.md) and
[recording procedure](https://github.com/zangjiucheng/orbitopt/blob/main/docs/flightlab-regression-ci.md#e5-05-the-reproducible-full-physical-reference-mission-task-272).

承诺的单一环境报告通过了其适用的阶段检查,但其物理门为**`not_checked`**,因为停放轨道传播没有运行. 读一下那扇门 而不是把绿色报告当做完成实际任务的证据

## 在环境中添加停车轨道传播

Install the checkout in the locked TudatPy Python 3.10 environment and a
separate Python 3.12 landing environment. Supply their interpreter paths:

```bash
orbitopt crossenv run --plan src/orbitopt/data/crossenv_e505_plan.yaml \
  --env tudat=/path/to/orbitopt/bin/python \
  --env gfold=/path/to/landing/bin/python -o runs/crossenv-e505
python scripts/e505_reference.py crossenv-check runs/crossenv-e505
```

每个舞台用SHA-256文摘和解释/工具链出处的版本交换封存其文物. 消费者在使用前进行核查;回收阶段必须报告同样的分离手递身份。 参考计划还选择了两个口译员中记录的地球导向快照;各阶段报告记录其摘要,`crossenv-check`核实。 联合报告必须实际通过轨道传播,着陆,插入,质量和连续性检查;单是设计这个计划并不能证明通过了. E5-06之后,联合报告载有`joint_qualification`文件:`crossenv-check`从两个密封信封重新分级,要求通过所有五个物理检查,每个都命名环境和信封消化其证据. 计划宣布`full_mission: true`, 所以一个缺失的轨道传播或降落腿级`not_checked`,

## 物理资格证明(E5-07)

E5-05 above is the **Software Integration Reference**: a synthetic vehicle
with ADR-recorded relaxations, proving the chain -- never engineering-grade
verification. The **Physics Qualification Reference** flies the
public-sourced `starship_class_public_estimate` vehicle against
`leo_starship_class.yaml` (Max-Q 55 kPa public, 3.0 g ASSUMPTION never
relaxed; the flight peaks at 2.89 g), reaching **232.5 × 292.1 km at
51.98°**:

```bash
python scripts/e507_reference.py verify
python scripts/e507_reference.py run --out runs/e507
python scripts/e507_reference.py compare runs/e507
orbitopt crossenv run --plan src/orbitopt/data/crossenv_e507_plan.yaml \
  --env tudat=/path/to/tudatpy/bin/python -o runs/crossenv-e5-07
python scripts/e507_reference.py crossenv-check runs/crossenv-e5-07
```

The booster RTLS leg is **reported, not flown**: the vehicle's restart-2
aerodynamic-entry architecture has no representation in the FL-302 planner
(see `tests/data/mission/reference/e5-07-physics-qualification/RECOVERY_GAP.md`
for the verbatim planner rejections), so the mission declares no recovery,
the physical gate caps at `not_checked`, and the plan does not declare
`full_mission` -- the joint gate reports NOT-DECLARED instead of a relaxed
pass. Details in the
[recording procedure](https://github.com/zangjiucheng/orbitopt/blob/main/docs/flightlab-regression-ci.md#e5-07-the-physics-qualification-reference-starship-class-task-291).

## 检查已记录的链条而不重飞

```bash
python -m orbitopt.verify.full_chain --out runs/e505/full-chain.verification.json
```

请检查22参考比较. 其独立的-模型证据涵盖链条提取的原始OT 上升段数据以及助推器共享的无动力窗口. 触地得分限额为**设计投入**;不主张历史遥测或独立验证着陆. 见[verification](https://github.com/zangjiucheng/orbitopt/blob/main/docs/verification.md#full-chain-reference-validation-e8-01-task-281).

## E10恢复和指导研究

| 学习 | 记录结果 |
|---|---|
| E10-01 / E10-02 | 恢复计划员支持无动力进入,模型航班代表热中转. [hot-staging benchmark](https://github.com/zangjiucheng/orbitopt/blob/main/docs/benchmarks/e10_02_hot_staging.json)比较时间. |
| E10-03 | 168-案例搁置中,DC7登陆44案例,Starbase登陆11. DC8武器在零情况下降落。 [Closure study](benchmarks/e10_03_robust_rtls_closure.md). |
| E10-04 | DC7在纠正事件状态提取后的独立检查通过. Starbase的碎片撞击速度差异仍然位于它的波段之外. [Discrepancy audit](https://github.com/zangjiucheng/orbitopt/blob/main/docs/e5-08-new-vehicle-report.md#e10-04-closing-the-two-open-discrepancies-task-301). |
| E10-05 / E10-06 | 比较[recovery planning strategies](benchmarks/e10_05_recovery_planner_robust.md)和[graded admission](benchmarks/e10_06_chance_constraint_redesign.md). |
| E10-07 | 独立的模型从实际的DC8分离交接中重新飞回,复制走廊故障. [Re-flight report](e10-07-full-recovery-reflight.md). |
| E10-08 | 12 of 17 scenarios insert on both native backends, up from six. [Guidance matrix](https://github.com/zangjiucheng/orbitopt/blob/main/docs/flightlab.md#6-dof-studies). |
| E10-09 | CI级 冻结的DC7和Starbase计划 并检查原E5-08复制其预期失败. [CI evidence rules](https://github.com/zangjiucheng/orbitopt/blob/main/docs/verification.md#final-schemes-in-cross-environment-ci-e10-09-task-306). |

## E11 空气动力学 RTLS 关闭

| 学习 | 记录结果 |
|---|---|
| E11-01 | 网鳍降低了DC8峰值动态压力,将最佳到达速度从1,068降低到362 m/s. 双燃设计仍然错过了350/s速度和15 °倾斜极限. [Grid-fin authority](benchmarks/e11_01_grid_fin_authority.md). |
| E11-02 | The separation-state screen made zero false rejects across 672 recorded hold-out cases. [Feasibility envelope](benchmarks/e11_02_recovery_feasibility_envelope.md). |
| E11-03 | 联合设计降落在0,168对联搁置案,对阵44,DC7和11星基地. [Recovery co-design](benchmarks/e11_03_robust_recovery_co_design.md). |
| E11-04 | 24 of 32 newly admitted cases landed in counterfactual re-flights. [Admission validation](benchmarks/e11_04_counterfactual_admission.md). |
| E11-05 | Starbase’s independent debris impact-speed discrepancy remained +12.70% against a 12% tolerance. [Aero qualification](benchmarks/e11_05_starbase_aero_qualification.md). |
| E11-06 | 实验的星座基地重播记录了两次助推器点火,一次用于助推,一次用于连续进入和降落. 它节省的着陆和轨道再飞行检查通过. [Inspect the run](https://github.com/zangjiucheng/orbitopt/blob/main/docs/flightlab.md#viewing-runs-in-mission-control). [Gate record: missions, re-flights, FL-304 rates with CIs](https://github.com/zangjiucheng/orbitopt/blob/main/docs/computed-mission.md#e11-06-gate-record-task-390). |
| E11R-01 | Training-ranked candidates landed in **35–36 of 36** hold-out cases per search seed. First-feasible candidates landed in **6–33 of 36**. The gain over the sealed 33/36 champion was statistically inconclusive. All five seeds reuse the same hold-out cases. [Campaign report](https://github.com/zangjiucheng/orbitopt/blob/main/tests/data/mission/reference/e11r-01-robust-candidate-selection/campaign.md). |
| E11R-02 | 双胞胎的18.2%的条目峰-q差距是密度图示形式,在主模型的单0 – 25 km表段内;12%波段不变. [Peak-q error budget](benchmarks/e11r_02_peak_q_error_budget.md). |

在 **运行** 中添加存档的 `tests/data/mission/reference/e11-06-starbase-continuous-burn/mission-hs/recovery/run` 文件夹,以检查热稳定燃烧序列和鳍遥测.

在热中转时,任务链可以利用观测到的加速度和质量流动,同时母体仍在发射,从而弥合取样轨道之间的缺口。 交接报告记录了桥梁方法和容忍度. 电源极限在一次星基分离上进行了校准. 断线后的分离使用现有的海岸桥. E11-06 fin drag使用假设校准,保存的重播显示名义模拟.

### E11-06 gate record (task #390)

两条存档的任务链都通过71列出的支票. 单独的SECO轨道复飞也通过. FL-304记录了33/36在每架上升段上以固定的脚本登陆,36/36在密封的上升段上以每个样本重新规划. 独立的双胞胎的峰值动态压力由**18.2%**与**12%**乐队不同. E11R-02将这一差距归结为主模型的单0 – 25 km表段内部的密度透视形式,而不移动波段:[peak-q error budget](benchmarks/e11r_02_peak_q_error_budget.md). [gate archive](https://github.com/zangjiucheng/orbitopt/blob/main/tests/data/mission/reference/e11-06-starbase-continuous-burn-reports/README.md)包括间隔,鳍拖曳假设和搜索灵敏度.

## 先前的研究

| 学习 | 阅读或复制 |
|---|---|
| E6 :引擎对等,引导JSBSim中转,引导信封和控制步集 | [6-DOF studies](https://github.com/zangjiucheng/orbitopt/blob/main/docs/flightlab.md#6-dof-studies)和ADRs 0008 – 0011. |
| E7:强制联合约束,概率校准,结合风险和配对故障分析 | [Recovery studies](https://github.com/zangjiucheng/orbitopt/blob/main/docs/flightlab.md#joint-chance-constrained-planning-e3-02). |
| E8:所需阶段资格、差异归属和参考可移植性 | [Verification](verification.md)和[tolerance audit](reference-tolerance-audit.md). |

研究脚本导出度量JSON;它们不一定产生Flight Lab运行目录. 记录的结果描述了其申报的车辆、控制器和散射,而不是一般飞行资格。
