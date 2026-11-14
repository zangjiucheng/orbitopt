# 发射场和运载火箭数据:来源、信心和限制

两个小型参考数据库:

| 文件 | 目录 | 装填器 |
|---|---|---|
| `src/orbitopt/data/launch_sites.yaml` | 平台位置、高度、地球旋转增益、牧场安全方位角走廊 | `orbitopt.launch_sites` |
| `src/orbitopt/data/launch_vehicles.yaml` | LEO / GTO / TLI能力,C3- vs-负载曲线,上级重启/海岸 | `orbitopt.launch_vehicles` |

**This is planning-level data, not authoritative data.** It is good enough
to rank trajectories and reject clearly infeasible ones. It must not be
quoted as a vehicle's or range's actual capability. For a real mission,
use the provider's current user's guide and a mission-specific performance
quote.

## 每个记录都包含着什么

每个数据项记录其来源. JSON计划执行,加载者拒绝不执行的条目:

| 字段 | 网站(`sources[*]`) | (轨道入口/`c3_curve`/`upper_stage`) |
|---|---|---|
| `source` / `ref` | 所需经费 | 所需经费 |
| `provenance` | required: `official` · `public_estimate` · `derived` | required: `official` · `public_estimate` |
| `accuracy` | 说明或估计准确性 | 不变 |
| `as_of` | `YYYY[-MM[-DD]]`:文件的出版日期或网页访问日期 | 不变 |

发射场必须至少为每个战区组举一个来源(`location`,`altitude`,`azimuth_corridors`,`rotation_speed_gain`). 否则`load_launch_sites`会提高`ValueError`.

### 证明词汇

- **公务**:取自发射供应商发表的一份文件
  或机构,如用户手册、参考指南或车辆网页。
  数字是官方的,但还是四舍五入,经常引用为:
  未说明的参考轨道,可以在问题之间变化。
- **_估计数**:其他一切. 包括二级参考
  (Isakowitz等人),公开汇编(维基百科坐标,GCAT)
  发射日志),手动数字化地块(NASA LSP性能曲线)和
  reconstructions from mission history. **These are approximations. Do not
  present them as facts.**
- **来源**:通过其他字段的轨道选择计算,例如:
  从纬度和高度计算出旋转增益。 它的信心是
  这些投入分别列出。

发射场还保留一个简易场`confidence: published | approximate`,用于其方位角走廊. `published`表示走廊限制出现在已公布的文档中. `approximate`表示它们是从实际飞行的倾角重建的.

## 对报告的信任

使用此数据的产出报告其来源:

- `LaunchConstraints.describe()`(印于
  `examples/12_launch_constrained_transfer.py`)有一个`data_sources`区块.
  它拥有总体的`confidence`,每个项目有一个记录在C3后面.
  限制(车辆的`c3_curve`)和DLA(场地的`location`)
  and `azimuth_corridors`).
- `orbitopt launch-window <site> ...`(表输出)以a结尾
  站点位置及其使用的通道的"数据源"块.
  `--azimuth`的走廊取代了现场的走廊,所以它们
  未列出。 `--json`输出不变.
- 方案:`site.data_sources()`、`vehicle.data_sources()`和
  `orbitopt.data_provenance.format_data_sources(...)` /
  `data_sources_report(...)`.

总体信心是`public_estimate`,如果**任何**贡献项目是一个估计. 只有每件物品都是正式的或衍生出来的,才算`official`.

## 已知的限制

运载火箭

- **C3 curves are all `public_estimate`.** Falcon 9, Falcon Heavy and
  Atlas V 551从NASA发射服务方案地块中手动数字化
  (about ±10 %). SLS, Vulcan and Long March curves come from presentations
  或任务历史(约 ± 15 – 20%). 没有使用官方的C3表.
- 官方LEO / GTO数字为头号数字. 他们经常被四舍五入
  引自未说明的轨道(轨道倾角,高度,GTOΔv)
  to GEO), and Falcon numbers assume full expendability. They are not
  提供方之间没有正常化的可比性。
- 一些TLI值是读取或接近C3曲线的估计数。
  C3 ≈ −2 km²/s². The Falcon 9 TLI value (4400 kg) is lower than its own
  曲线(5200千克). 两者都是估计数。
- `max_restarts`/`max_coast_s`都是`public_estimate`.
  它们是从飞行特派团推断的,而不是从公布的资格证书推断的
  限制。 实际限制取决于任务包、推进剂燃烧和
  热抑制.
- 有效载荷适应器质量、美容限制、性能储备和
  可重复性(Drone-ship或RTLS)不是模型化的.

发射地点

- 帕德坐标主要来自公开汇编(约100 mi;
  星座基座约为500 m,因为没有源头能分离其两个轨道垫.
  这对旋转增益和发射窗口时间来说是足够的,但
  这不是调查等级。 每个网站都有一个参考页。
- 只有库鲁的走廊来自发射提供文件. 那个
  东边和西边的走廊属于二级
  (2004. 中国遗址和拜科努尔的走廊被重建
  从飞行倾角(约 ± 5 Z °). Starbase的40 – 115 °来自a.
  single secondary report on the FAA's 2026 Starbase PEIS, not the FAA
  文档本身。 设定了真正的范围安全限制
  每个任务,取决于车辆和投放区,不公开。
- 倾角仅限上升段直接倾角. 狗腿动作,什么
  真实的车辆是飞行的,不是模型走廊的一部分。

日期

- `as_of`给出文件的发布日期或年份为网络
  页面已访问。 提供者的网页不经通知而更改。 重新检查
  在依赖价值之前

## 更新数据

复制YAML文件并编辑,保留`source`,`provenance`,`accuracy`和`as_of`每个项目. 然后装上`load_launch_sites(path)`/`load_launch_vehicles(path)`. 只有在您直接从提供方或代理方的文档中获取时, 才标出 `official`, 并引用该文件的问题和日期 。
