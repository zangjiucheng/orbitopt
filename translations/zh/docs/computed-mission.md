---
title: Computed mission workflow
parent: Flight Lab
nav_order: 0
description: Replay the recorded LEO and RTLS mission, add propagation across environments, and inspect evidence.
---

# 计算任务工作流程

E5 – E8 加一个计算-上升段 → RTLS的参考文献,结合分离交割,跨环境执行,指导JSBSim的研究和更严格的证据门. 选择回答您问题的运行 :

| 工作流程 | 什么运行 | 证据边界 |
|---|---|---|
| E2-04 | 合成上升段端点+模拟回收. | 快速整合固定;没有计算 上升段或停车 轨道传播. |
| E5-05 重播 | 记录计算 上升段+新模拟回收. | 一个峡谷隔离状态;停车轨道传播仍为`not_checked`. |
| E5-05 跨环境 | 录音上升段+tudatpy停车轨道传播+gbold恢复. | 联合运行必须证明所有五个物理检查。 |
| E8-01 证明人查询 | 将密封记录与原始后端输出和封闭窗体进行比较。 | 没有新的飞行;验证提取、连续性和说明的参考比较。 |

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
telemetry, add `runs/e505/recovery/run` in Mission Control's **Flight Lab**.
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

承诺的单一环境报告通过适用的阶段检查,但其实际大门为**`not_checked`**,因为停放轨道传播没有运行。 读一下那扇门 而不是把绿色报告当做完成实际任务的证据

## 在环境中添加停车轨道传播

Install the checkout in the locked TudatPy Python 3.10 environment and a
separate Python 3.12 landing environment. Supply their interpreter paths:

```bash
orbitopt crossenv run --plan src/orbitopt/data/crossenv_e505_plan.yaml \
  --env tudat=/path/to/orbitopt/bin/python \
  --env gfold=/path/to/landing/bin/python -o runs/crossenv-e505
python scripts/e505_reference.py crossenv-check runs/crossenv-e505
```

每个舞台用SHA-256文摘和解释/工具链出处的版本交换封存其文物. 消费者在使用前进行核查;回收阶段必须报告同样的分离手递身份。 参考计划还选择了两个口译员中记录的地球导向快照;各阶段报告记录其摘要,`crossenv-check`核实。 联合报告必须实际通过轨道传播,着陆,插入,质量和连续性检查;单是设计这个计划并不能证明通过了.

## 检查已记录的链条而不重飞

```bash
python -m orbitopt.verify.full_chain --out runs/e505/full-chain.verification.json
```

请检查22参考比较. 其独立的-模型证据涵盖链条提取的原始OT 上升段数据以及助推器共享的无动力窗口. 触地得分限额为**设计投入**;不主张历史遥测或独立验证着陆. 见[verification](https://github.com/zangjiucheng/orbitopt/blob/main/docs/verification.md#full-chain-reference-validation-e8-01-task-281).

## 其他合并研究

| 学习 | 阅读或复制 |
|---|---|
| E6 :引擎对等,引导JSBSim中转,引导信封和控制步集 | [6-DOF studies](https://github.com/zangjiucheng/orbitopt/blob/main/docs/flightlab.md#6-dof-studies)和ADRs 0008 – 0011. |
| E7:强制联合约束,概率校准,结合风险和配对故障分析 | [Recovery studies](https://github.com/zangjiucheng/orbitopt/blob/main/docs/flightlab.md#joint-chance-constrained-planning-e3-02). |
| E8:所需阶段资格、差异归属和参考可移植性 | [Verification](verification.md)和[tolerance audit](reference-tolerance-audit.md). |

研究脚本导出度量JSON;它们不一定产生Flight Lab运行目录. 记录的结果描述了其申报的车辆、控制器和散射,而不是一般飞行资格。
