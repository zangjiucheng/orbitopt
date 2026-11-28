---
title: E1–E4 acceptance review
parent: Engineering
nav_order: 5
description: Scoped verification, resolved findings and evidence boundaries.
---

# E1 – E4 接受审查

**Date:** 2026-11-15. **Initial reviewed revision:**
`d41ae5fb241fa072c8e2109067175339c1d12932`.
**Result: both reproduced blockers resolved in the accompanying fix.**
This is scoped local acceptance, not proof of deployed documentation or remote CI.

审查内容包括:犬科后端身份、JSBSim aero、本土上层指导、双分支整合、恢复动因/chance/统计工作以及证据/接受矩阵。 最初的运行有599通过,一个跳过和两个E2-04失败,加上下面的独立物理反例.

## 1. 上升段物理证据被放弃.

**Resolved P1.** Previously, changing only the frozen ascent's
`physics_verified` flag to false still produced a passing chain, report and
full-mission gate. Neither boolean value appeared as independent evidence.
The chain now emits `ascent.physics_verified` for both boolean values:

| 投入 | 检查/ 独立证据 | 轨道注射/全程 |
|---|---|---|
| `true` | 通过 | 当其他城门经过时, |
| `false` | 失败 (`physics_verification_failed`) | 失败 |
| `null` | 明确未评估; 没有独立的检查 | 受其他适用检查制约 |

回归涵盖所有三种有恢复和无恢复的值,并验证了序列报告。 努尔不会成为积极的独立验证证据. 仅凭设计限制的传票仍不能证明曾进行过独立的再飞行。

代码:[`two_stage_leo.py`](https://github.com/zangjiucheng/orbitopt/blob/main/src/orbitopt/mission/two_stage_leo.py),`run_two_stage_leo`;[regression](https://github.com/zangjiucheng/orbitopt/blob/main/tests/test_mission_phases.py).

## 2. 冷冻参考可移植性不被接受

**Resolved P2.** The initial macOS run exposed frame roundoff of about
`1.3924e-7 m` and `m/s`, exact equality on numeric event lists, and advisory
landing-plan drift of `4.817e-4 kg` fuel / `3.029e-3 s` duration.
An isolated landing-CI dependency environment also exposed different offline
IERS defaults between Astropy 6 and 8, producing metre-scale frame differences.

参考文献现在包括原记录数据包中的四排地球导向快照。 只有引用建立/重运行它;之后调用者的表格会被恢复. 递归比较将数值预算应用于数组,同时拒绝改变的形状,密钥,布尔值和非无限值. 重建的框架组件有`1e-6`绝对预算. 只有咨询性起落架燃料/时场使用`0.001 kg` /`0.01 s`预算,与溶解器的趋同和整流器的解析为理由.

**Recorded flight outputs are unchanged.** The physical-state limits remain
`0.01 m` and `1e-4 m/s`. Tests explicitly reject larger flown-state drift and
out-of-budget planning drift. The metadata-only update verifies existing hashes
first and preserves original recording provenance. See the
[reference procedure](https://github.com/zangjiucheng/orbitopt/blob/main/docs/flightlab-regression-ci.md#e2-04-the-frozen-dual-branch-reference-mission-task-257).

代码:[`e204_reference.py`](https://github.com/zangjiucheng/orbitopt/blob/main/scripts/e204_reference.py);[regressions](https://github.com/zangjiucheng/orbitopt/blob/main/tests/test_e204_reference.py).

## 修复 验证

| 环境/范围 | 结果 |
|---|---|
| macOS Python 3.12, NumPy 2.5.4 / SciPy 1.18.1 / Astropy 8.0.2; reference, phases, mission chain and reports | 142通过,1跳过,1解选. |
| Isolated Python 3.12 landing-CI requirements: NumPy 1.26.4 / SciPy 1.13.1 / Astropy 6.1.7; reference and phases | 92通过 |

其他检查:后端/平面、指导、恢复和任务/报告核心(非缓慢):**283通过,1跳过**。 实际JSBSim生产上升段/比较,合并上升段/恢复和数字/分析接受:**4通过**. 额外场景,合同,CLI /API 和外屏 Qt/时间线检查(不慢):**280通过**. 这些内容与上表重叠,不是添加剂总数。

这两个环境都运行了真正的冷冻回收模拟,并将其与原始记录的输出与gfold 0.3.2进行对比. 孤立的环境是对macOS的依赖兼容性检查,而不是执行的Linux CI任务. 取消选用的测试是昂贵的全引用优化器; 跳过的负测试预计pygmo无法使用, 但pygmo已安装 。 一些一般的相位固定装置警告默认的IERS数据覆盖;参考重播时会使用其冷冻快照.

对于E3-03,记录的168每臂对排复制了所有四个成功计数,威尔逊间隔和六个准确的麦克尼玛p值. 这些是未经调整的双向测试;672回收航班没有重新运行.

## 证据界限

E2-04使用合成,二样的上升段交接和规定轨道加上真实的回收模拟. 停车轨道传播已超出范围. 通过这个固定装置并不能建立持续传播的上升段,轨道和返回. (E5-01,任务# 268:任务代码路径上的空隙被关闭——全任务链的上升段现在由OpenTsiolkovsky计算出事件确定的中转和全时历史,`tests/test_mission_e5_real_ascent.py`; 合成的E2-04定格仍然仅作为快速测试的例外。 ) 土生土长的F-07比较采用被披露的经修改的上层阶段.

这次审查没有建立全贵的优化器/M-07启动,全存储套件,交互式桌面渲染,CUDA,Windows或Linux CI接受. 历史基准文件和屏幕外部件不能取代这些检查。
