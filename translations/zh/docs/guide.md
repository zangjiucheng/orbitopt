---
title: Mission Control
nav_order: 3
permalink: /guide.html
description: Use the 3D view, timeline and verification reports in Mission Control.
---

# 使用 Mission Control

Mission Control将3的视图,时间表,图表和验证的报告链接起来. [Install orbitopt](https://github.com/zangjiucheng/orbitopt/blob/main/docs/quickstart.md#installation),然后运行`orbitopt`,并在侧边栏选择一个任务.

<figure>
<img src="assets/mission-control-starship-landing.png" alt="Mission Control showing a Starship landing scene and its timeline" />
<figcaption>Sentinel-2 cloudless 2024 by EOX IT Services GmbH, containing modified Copernicus Sentinel data 2024 (CC BY-NC-SA 4.0).</figcaption>
</figure>

## 安装和打开

由包装数据打开的内置场景. 标题栏提议,如果一个场景是用旧代码制作的,则重新计算;计算场景会被缓存. 以 *文件 QQ 打开场景文件...* 打开 JSON 场景,或使用 `orbitopt view <mission>` 进行单景查看.

## 窗户

- **边栏 :**选择任务或切换到[Flight Lab](flightlab.md).
- **3 D视图 :**选择一个机构;使用**CENTER**改变参考机构.
- **图:** 点击或拖动距离图以改变时间; 发射窗口
  图表显示有经过筛选的候选人。
- **探长:**检查身体,动作和选定的候选人. 内容
  验证标签显示支票及其耐受性.
- **时间线 :**在任务中演奏,洗涤和放大.

小组分享选定的机构和任务时间。 用头条折叠它们; `I` 切换右面板.

## Moving around in 3D

| 控件 | 行动 |
|---|---|
| 左撇子 | 旋转 |
| 滚动或右擦拭 | 缩放 |
| Shift- Drag 或 中间拖曳 | 笔 |
| 单击天体 | 选择它 |
| 命令/点击第二个实体 | 测量距离 |
| 标签/ Shift+标签 | 下一个重点/前一个机构 |
| `T` / `M` / `R` | 轨迹/ 焦点/ 重置相机 |

*视图*菜单也提供固定相机角度. 选择动作跳转到并设定其燃烧范围 。

## 时间和时间表

### 正在播放

`Space`前奏或暂停;`Shift+Space`后奏. 将 `Shift+Left/Right` 用于上一个或下一个事件, 并 `Home` 重新启动 。

### 赛道

| 控件 | 行动 |
|---|---|
| 在轨道上滚动/ 右擦拭 | 缩放光标/ 泛 |
| 拖动概览括号 | 打开可见间隔 |
| 单击或拖动标尺 | 移动时间; 拖曳暂停播放 |
| 拖动播放头 | 不跳不跳的摇摆 |
| 移动拖车 | 细十倍 |
| 单击/ 双击事件或阶段 | 跳动/ 放大到它 |
| `=` / `-` / `0` | 缩小/缩小/适合任务 |
| 左边/ 右边 | 步骤 1% 可见间隔; `Alt`步骤 0.1% |

### 到一个时候

点击 MET读出或按 `G`. 输入 `T+22:32:34`, `6m35s`或 `1d 02:00:00`; 使用 `+30s`进行相对跳跃, 或使用日历 历元 输入 UTC 任务日期 。 一个赤裸的号码使用时间轴的单位. 无效的输入以红色概括 。

## 时间曲速

梯形选择一个速率,从0.1×到10,000,000×;1×是实时的. 使用`,`和`.`在关卡之间踩踏,`[`和`]`微调,`/`返回到1×. 时钟可以自动缓冲近事件。

## 看飞机

### 飞行数据标签

Press `L` to toggle labels. **H** is height above the ellipsoid, or **R** is
distance from the scene center. **V** is inertial speed; **surf** is speed
relative to the rotating ground below 120 km. Labels also show cumulative Δv
and the current burn.

### 检查员

选择一个机构,策略或候选人来查看其细节. 点击策略跳转到它的时间 。 验证标签列出了模型的假设,剩余和耐受性.

### 图表

距离图跟随播放头;点击或拖动任务时间. 发射窗口图显示受支助飞行任务的候选者。

## 地球图像

*查看高分辨率地球图像*使流星的哨兵-2瓷砖靠近地球. 它们在macOS上的`~/Library/Caches/orbitopt/tiles`或Linux上的`~/.cache/orbitopt/tiles`下被缓存. 设定 `ORBITOPT_TILE_CACHE` 以更改缓存位置或 `ORBITOPT_EARTH_TILES=off` 以禁用下载 。

{: . 注释}
> Sentinel-2 cloudless 2024 by EOX IT Services GmbH contains modified
> 哥白尼哨兵数据2024并持有执照
> [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
> 非商业用途。 显示瓷砖时显示归属 。

## 候选人和验证

对于优化后的飞行任务,选择轨迹检查其筛选弧和验证. 徽章为**VERIFIED**,**VERIFY FAILED**或**未检查**. GOES打开不受限制;使用其验证标签运行tudatpy复飞. 验证以模型和宽容为条件.

验证标签中的每张支票标签为数值一致性,模型一致性或历史数据验证;没有支票的一类证据读作**未检查**,从未通过. 见[verification](verification.md).

"星舰飞行15"报告包括独立复飞,恢复散射,载荷和时序检查. 假设的10%升降率失败. 假设和预测与历史比较的区别,见[mission examples](examples.md).

## 发射任务

A Flight Lab run can also open as a mission in the main view, one continuous
timeline from the pad to orbit: export it with `orbitopt flight to-scene
run/ -o scene.json`, or run a `flight-lab` or `two-stage-leo` config with
`orbitopt run`, then open the scene with *File › Open scene file…*.

- **阶段行**牌 整个任务: 在垫,助推,中转,
  末级燃烧,然后停放轨道和轨道传播(双体显示)
  引擎截断后的海岸,标签为显示数据,而不是验证.
- 每个**阶段交接**都有判决. 分离国家与群众
  各个阶段之间必须是连续的;在引擎截断时,插入是
  对照目标轨道检查 。 判决出现在事件笔记中
  在时间线和验证标签中的**MISSION HANDOFFS**部分.

大门见[Flight Lab](https://github.com/zangjiucheng/orbitopt/blob/main/docs/flightlab.md#mission-pipeline).

## 你的任务

```bash
orbitopt validate my_mission.yaml
orbitopt run my_mission.yaml
orbitopt view my_mission.yaml
```

见[mission configs](https://github.com/zangjiucheng/orbitopt/blob/main/docs/quickstart.md#mission-configs). 编辑代码后,使用标题栏**重新装入**按钮或Command/Ctrl+R重启计算工人,并重算当前的任务.

## 键盘快捷键

按 `F1` 在应用程序中完整引用. 除了上述控制外,`I`还切换督察和指挥/Ctrl+QX退出.

## 解决问题

- **任务需要几分钟时间:** 一个自定义或重算的场景是
  计算;之后的负载可以使用缓存.
- **无卫星图像 :**检查视图设置,网络连接和
  `ORBITOPT_EARTH_TILES`;瓷砖只出现在地球附近.
- **黑色 3 截图中的 D 区域 :**使用`scripts/capture_screenshots.py`,
  它捕捉和复合 OpenGL 视图。
- 关于其他问题,[open an issue](https://github.com/zangjiucheng/orbitopt/issues)
  任务和繁衍的步骤
