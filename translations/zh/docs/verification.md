# 验证和基准

验证报告记录检查,容忍度,数字设置和物理模型. 通行证只适用于这些假设;通行证并不确立现实世界的飞行安全。 Mission Control显示**VERIFIED**,**VERIFY FAILED**或**未检查**.

## 报告的内容

报告采用`verification-report-1.0`计划,并包括残余物,强制模型和省略效果,集成器设置和舟子SHA-256文摘. 由Flight Lab,`orbitopt launch`和参考语境连锁书写。

## 三种证据

| 类别 | 它检查什么 |
|---|---|
| `numerical_consistency` | 模型范围内的数值趋同或保存. |
| `model_agreement` | 与另一模型协议,预测或约束. |
| `historical_validation` | 与公布的现实世界数据达成一致。 |

不检查的类别是**未检查**,没有通过.

## 参照公众参考的基准

`orbitopt benchmarks` checks a Kepler two-body orbit, J2 secular drift and the
Starship Flight 15 booster catch against analytic or published references. It
returns exit status 1 if any check fails; `--json` outputs a verification
report. See [engineering notes](https://github.com/zangjiucheng/orbitopt/blob/main/docs/engineering-notes.md#benchmark-scenarios-with-public-references--analytic-solutions-task-245)
for the references.

```bash
orbitopt benchmarks
orbitopt benchmarks --json
```

## Cross-model comparison

比较单级上升段和横跨可用的3-DOF,JSBSim,tudatpy以及分析后端. 结果被重印到一个共同的时间网格上;阈值可以被相位和度量所超越.

```bash
python -m orbitopt.verify.cross_model --backends three_dof,six_dof,tudatpy
python -m orbitopt.verify.cross_model --threshold ascent.position_m=1000 -o cross_model.json
```

## 集成器趋同

汇合性研究对LEO和GTO海岸的耐受度和适当时的阶梯尺寸进行了扫射,重新进入和供电上升段. 它估计了数字误差并评估了每种情景的生产环境。

```bash
python -m orbitopt.verify.integrator_convergence --plot
```

## 模型 覆盖范围和宣布的公约

任务结果记录了每个阶段的模型和局限性. `orbitopt.physics_io`在物理边界检查申报的框架,时间和单位约定.

## 进一步阅读

[Flight Lab](flightlab.md) · [Regression CI](flightlab-regression-ci.md) · [Examples and validation](examples.md) · [Engineering notes](engineering-notes.md)
