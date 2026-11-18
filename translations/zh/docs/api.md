---
title: API reference
nav_order: 6
permalink: /api/
description: Python modules, command-line options and data schemas.
---

# API 参考文献

| 接口 | 参考 |
|---|---|
| Python | [Modules, classes, functions and methods](https://zangjiucheng.github.io/orbitopt/api/python/) |
| 命令行 | [Commands and options](https://zangjiucheng.github.io/orbitopt/api/cli/) |
| 数据 | [Mission, scene, report and Flight Lab schemas](https://zangjiucheng.github.io/orbitopt/api/schemas/) |
| 跨环境数据 | [Plan](https://zangjiucheng.github.io/orbitopt/api/schemas/crossenv-plan-1.0/)和[sealed exchange](https://zangjiucheng.github.io/orbitopt/api/schemas/crossenv-exchange-1.0/) |

参考文献来自源头,在建站时间进行包装。 公共名称通过命名公约列入;列入并不意味着内部执行是一种稳定的兼容性承诺。 私名和继承的第三方方法省略. 如果缺少源代码,则引用会说出并链接到执行中.

[quickstart](quickstart.md)引入了共同的工作流程. [engineering notes](engineering-notes.md)解释数字假设.

## 任务和资格接口

| 任务 | Python 参考文献 |
|---|---|
| 装入计划、运行阶段和验证交换 | [Cross-environment API](https://zangjiucheng.github.io/orbitopt/api/python/orbitopt.crossenv/) |
| 构建共享分离并进行分级 | [Coupled branches](https://zangjiucheng.github.io/orbitopt/api/python/orbitopt.mission.coupled_branches/) |
| 开展指导性JSBSim/本土比较研究 | [Guided ascent](https://zangjiucheng.github.io/orbitopt/api/python/orbitopt.flightlab.sixdof.guided_ascent/) |
| 评价校准、联合风险和配对故障计数 | [Calibration](https://zangjiucheng.github.io/orbitopt/api/python/orbitopt.flightlab.recovery.calibration/), [joint risk](https://zangjiucheng.github.io/orbitopt/api/python/orbitopt.flightlab.recovery.joint_risk/), [failure analysis](https://zangjiucheng.github.io/orbitopt/api/python/orbitopt.flightlab.recovery.failure_driven/) |
| 建立记录链报告或属性 模型 差异 | [Full-chain reference](https://zangjiucheng.github.io/orbitopt/api/python/orbitopt.verify.full_chain/), [discrepancy budget](https://zangjiucheng.github.io/orbitopt/api/python/orbitopt.verify.discrepancy_budget/) |

对于可执行的示例以及重播和新鲜的轨道传播的区别,从[computed mission workflow](computed-mission.md)开始.
