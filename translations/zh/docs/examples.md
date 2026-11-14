---
title: Examples and validation
parent: Missions and examples
nav_order: 1
---

# 实例和鉴定

内置任务显示模型在不同忠诚级别. 与历史飞行的比较与解决已公布目标不同;经过的验证报告仅涵盖其申报的检查和模型.

| 特派团 | 什么样的模型 | 口译 |
|---|---|---|
| 火星 2020 | 发射窗口搜索、注入、巡航和火星轨道插入 | 历史上的发射时间比较;注射报告记录了9 /9检查记录的重播. 见[`14_mars2020_replay.py`](https://github.com/zangjiucheng/orbitopt/blob/main/examples/14_mars2020_replay.py)及其[config](https://github.com/zangjiucheng/orbitopt/blob/main/examples/mars2020_replay.yaml). |
| 阿尔忒弥斯II-喜欢 | 月球飞过无动力地球回归 | A 6,545 km perilune is an input target, not an independent reconstruction. The chosen departure is 2026-08-01; the real flight launched on 2026-04-01. The entry-angle check is vacuum geometry, not an assessment of heating or survival. See [re-entry constraints](reentry_safety_constraints.md). |
| GOES | GTO-至-GEO 以阿特拉斯V/GOES-16的参考简介提升. | 模拟运动;内置场景开始**未检查**,直到高真验证运行. 见[mission plan](goes_gto_geo_mission_plan.md). |
| 星舰飞行15 | 利用假定轨道和车辆特性进行回程预测 | 飞行前的预测,不是重建. 上升段是一个说明. 验证报告在10%升降下失败;参见[`starship_flight.py`](https://github.com/zangjiucheng/orbitopt/blob/main/src/orbitopt/viz/starship_flight.py)中的假设. |
| 两级 LEO(参考方案) | 400 × 400 km,51.6 ° 停车轨道:上升段,中转,插入和轨道传播. | 分为一连串,有交接门和预期的停放轨道元素;合成飞行器是说明性的,而不是真正的火箭. 运行 `orbitopt run src/orbitopt/data/mission_two_stage_leo.yaml`;参见 [Flight Lab](https://github.com/zangjiucheng/orbitopt/blob/main/docs/flightlab.md#mission-pipeline). |
| 太阳系 | 行星电光视觉 | 没有轨迹优化或验证. |

## 历史性的发射窗口比较

| 比较 | 轨道选择 | 历史参考 |
|---|---|---|
| Mars 2020, KSC, 2020-07-30 | 11:51:28 UTC optimum | 11:50 UTC lift-off |
| 火星2020,窗口中最小的C3 | July 18, 13.2 km²/s² | Launch period opened July 17 |
| 哨兵-2 A/2 B,库鲁SSO窗口 | 在 6 - 9分钟内 | 实际起飞时间 |
| ISS, KSC | 44.9 ° 发射方位角; 大约每天稍早的 23 分钟 | 大致公布的数值 |

就火星发射时间比较而言,飞行的起飞日期和飞行时间都是投入;日间时间比较之后. 发射场和飞行器数字是规划一级数据:见[sources and limitations](launch-data-sources.md).

`orbitopt benchmarks` turns two more references into regression gates: the
analytic two-body and J2 drift solutions, and the Starship Flight 15 booster
catch against its published timeline. See [verification](verification.md).

## 运行示例

```bash
python examples/14_mars2020_replay.py
orbitopt run examples/mars2020_replay.yaml
orbitopt view artemis2
```

[examples directory](https://github.com/zangjiucheng/orbitopt/tree/main/examples/)还包括兰伯特,多重力辅助,猪肉跳,自由回归和Flight Lab的例子. 计算的结果取决于已安装的数字堆栈和选定的配置.
