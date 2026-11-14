# 火星——地球到轨道插入:任务计划

## 范围

The Mars mission models departure, heliocentric cruise and Mars orbit
insertion. It does not model launch ascent, atmospheric descent or landing.
The default trail starts in a 185 km parking orbit; an optional launch
configuration adds a site, vehicle, launch window and verified injection.
See [`mars2020_replay.yaml`](https://github.com/zangjiucheng/orbitopt/blob/main/examples/mars2020_replay.yaml).

## 为什么一个连续的太阳心形场景

地球离开和火星方法在以行星为中心的自然框架中传播,然后用行星的SPICE位置翻译,每个历元. 出口小径采用公里内一个太阳心框架。 行星细节在全任务规模上是很小的;使用Mission Control的焦点和跟踪控制来检查.

## 使用的实际参考数字

The documented 2026-window search found a departure near 2026-10-30 and
arrival near 2027-09-06, with a 310.67-day flight and C3 around 9.2 km²/s².
The 185 km parking orbit and approximately 500 × 20,000 km Mars capture orbit
are modeling choices, not a reconstruction of a particular flight. The Earth
sphere-of-influence handoff uses 924,000 km.

## 阶段性边界理想化(明确表述,不隐藏)

1. **出发 :** 兰伯特溶液假设在地球起飞
   太阳中心位置 所传播的地球逃逸状态不同;
   documented 2026 case, an uncorrected perturbed cruise misses the Lambert
   arrival by over 1.27 million km.
2. **Cruise correction:** `target_mars_approach` uses differential correction
   和B-平面的改进,以近拱点为目标. 瞄准固定点
   单历元不能说明重力聚焦. 目标可以
   失败;报告记录已实现的结果。
3. **轨道插入 :** 当地改良后推广
   大约近拱点. 稍有冲动的烧伤目标,然后
   短轨道检查。
4. **场景连续性 :** 显示的巡航使用相同的数值
   轨迹作为近拱点定位器. 独立地重新宣传两者
   棋子大小不同,以前产生了明显的接合。

## 可选真实启动( `launch` 配置部分)

`launch`区补给场,车辆,有效载荷和停车轨道选择. 搜索应用发射限制,计算每日窗口,并规划站点到注入序列. 一个钉住的离开跳过搜索。 对注射进行数字检查,然后得到自己的报告。 `launch`,或设置`enabled: false`,保留理想的停放轨道启动.

## 模块

| 构成部分 | 目的 |
|---|---|
| `launch_injection.py` | 发射窗口、停放轨道和经核实的注入 |
| `viz/mars_transfer.py` | 搜索、特派团管道和现场出口 |
| `verify/mars_insertion.py` | 巡航修正,近拱点和插入 |
| `schemas/mission-mars-transfer-1.0.json` | 特派团配置计划a |

## 这没有声称的

模型不重建发射上升段或火星登陆. 对每一个要求的几何学都无法保证趋同. 有不安全型号的火星近拱点候选人被拒绝;如果没有候选人通过验证门,则出口被确认为**尽最大的努力**,而不是经过核实的成功. 见[engineering notes](https://github.com/zangjiucheng/orbitopt/blob/main/docs/engineering-notes.md#candidate-selection-and-the-verification-gate-task-171).
