---
title: E5-08 design variants
parent: Flight Lab
nav_order: 0.5
description: Recorded Kourou and Starbase RTLS designs, nominal outcomes and remaining qualification gaps.
---

# E5-08设计变体

E5-08在E5-07公共估计堆栈中增加了电源入口RTLS. E5- 08 b 改变中转(DC7 );E5- 08 c 将设计移到星座基地并重新调整其上升段 . 这些是模型设计,而不是历史飞行验证.

## 记录结果

| 记录 | 原文:E5-08 | E5-08b: DC7 | E5- 08 c:恒星基地. |
|---|---|---|---|
| 编目车 | `starship_class_rtls_derived` | `starship_class_rtls_dc7` | `starship_class_rtls_starbase` |
| 发射场/目标轨道倾角 | 库鲁/51.6 ° | 库鲁/51.6 ° | 星座 / 30.5 ° |
| 交付轨道,公里 | 232.5 × 292.1 | 232.0 × 234.8 | 232.2 × 264.5 |
| 上升段峰负载 | 2.888 g | 2.976 g | 2.928 g |
| 名义恢复 | 着陆失败 | 着陆;113.1吨 | 着陆;107.1吨 |
| 跨环境联合大门,记录摘要 | 失败: 降落 | 已经通过 | 已经通过 |
| 独立检查-模型 | 27 / 27通过 | 27 / 27通过 | 26 / 27通过 |
| 暂停恢复, 固定脚本 | 未记录 | 11/36 | 4/36 |
| 暂停恢复,每例改进 | 0/30 | 3/36 | 0/36 |

DC7更早切断了1的舞台,使得助推器更重更慢,同时花费更多的船推进剂. 记录的舰只保留了在库鲁的101.2 t,在星基地的102.5 t,而在E5-07则保留了118.8 t. 星基也改变发射几何和目标平面;其结果并不是孤立的站点比较.

The variants retain the 55 kPa ascent Max-Q cap, 3.0 g ascent limit,
230 km periapsis floor, 250 ± 60 km insertion band, 3/1 m/s vertical/lateral
touchdown limits and 100 t reserve. Starbase changes the inclination target
to 30.5°. Read each scenario rather than the inherited 51.6° prose in some
recording metadata.

## 还有什么还开着?

- **原设计:** 电源进入脚本到达走廊,但
  着陆被拒绝或失败。 其预计着陆后储备为:
  96.1 t, about 3.9 t below the 100 t requirement. A reserve check on an
  中止的推出不能证明成功着陆 。
- **DC7 独立动态:**由E10-04关闭,波段不变(见下文).
- **星座独立动态 :**碎片撞击速度差异是
  12.80%对12%波段. E10-04证明它是一个真正的拖拉 -模型
  差异,不是执行文物(见下文)。 乐队没有
  更改并记录为失败。 E11-05 合格助推器
  动力关闭拖动 模型( 建立、 记录飞行倒置、 网格长度状态):
  所记录的航班倒转 4 -7% *下级* 已宣布的极地,没有支持
  correction closes the gap, and the replay stays at +12.70% - the difference
  站台,局部为OT/双空圈线
  ([E11-05 report](benchmarks/e11_05_starbase_aero_qualification.md)).
- **可靠性 :** 两种名义上的关闭设计都不太成功
  在申报的FL-304散射下. 固定和每样精制
  实验使用不同的规划模式;不汇集其费率。

The DC7 and Starbase experiments each use seeds 8, 9 and 10, with 12 flights
per seed. Original-design results use seeds 5, 6 and 7, with 10 per seed.
See [E7-05](benchmarks/e7_05_out_of_sample_reliability.md) for cluster-aware
statistics. The DC7 fixed-versus-replanning gap is diagnosed and a robust
single-script arm is compared on the same hold-out in
[E10-05](benchmarks/e10_05_recovery_planner_robust.md). The original
independent landing check starts from a declared design corridor and is
recorded as a `design_input`, not full-path
verification: it does not re-fly the actual recovery, so it cannot be read
as independent physical verification of the complete RTLS chain. The E10-07
full recovery re-flight ([E10-07 report](e10-07-full-recovery-reflight.md))
supersedes it for that purpose: it re-flies boostback, unpowered entry and
the landing phase from the actual DC8 hot-staging separation handoff under
an independently written twin model.

## E10-04: closing the two open discrepancies (task #301)

宽容也没有改变。 两种公开检查关闭方式不同,两种结果的区别是练习点.

**DC7 SECO semi-major axis — implementation artifact, fixed (27.7 km -> 5.85 km
against the same 15 km band).** The independent-dynamics twin compares itself
against the raw OpenTsiolkovsky state history interpolated at each declared
event epoch. That interpolation was a straight chord across the bracketing 2 s
segment. On this design MECO (123.0 s) and SECO (514.663 s) fall mid-segment,
across the engine-cutoff discontinuity: the chord averaged the powered and
coasting slopes, ~12.4 m/s too low at MECO and ~13 m/s too low at SECO - about
1.75 km of semi-major axis per m/s, which was ~22 km of the 27.7 km. It also
over-read the recorded coast mass at MECO by ~8.4 t. The fix reconstructs the
state inside such a segment from the LEFT raw row and its own recorded
accelerations (position quadratic, velocity linear, mass from the left row's
thrust/isp flow), which is exact up to the mid-segment switch; event times that
land on a grid row or a smooth segment behave exactly as before. After the fix
all 28 DC7 checks pass, with the measured budget lines (E8-02 discipline)
accounted in the report: MECO mass discrepancy −6,436 kg with the debris-mass
line −6,436 kg (one engine-Isp line, ~1.2% of the stage-1 burn), SECO velocity
+5.2 m/s, SECO SMA +5.85 km. Regression tests:
`tests/test_e805_independent_dynamics.py` (kink reconstruction, ignition mass
hold, recorded-DC7 SMA invariance).

**Starbase debris impact speed — genuine model difference, kept over-tolerance
(12.80% against the same 12% band).** Replaying the twin's own power-off debris
integrator from the RECORDED separation state and RECORDED debris mass
(identical initial conditions, same US76 atmosphere, no shared code path)
reproduces the gap: +12.7% at the declared CA polar versus the report's 12.8%,
so timing, frame convention, interpolation and engine lines are ruled out. The
twin reproduces the recorded descent to within 0.5 % at every shared altitude down to
22 km (dynamic pressure ~98 kPa) and from there to impact the recorded body sheds
393 m/s against the twin's 190 (~2x the deceleration); scaling the
declared blunt-body power-off polar by x1.1 closes the gap (+10.8%). The two
independent models disagree on the power-off drag coefficient of a tumbling
upper stage in dense transonic flow - an illustrative design-table input, not a
measurement - so this is recorded as a genuine model difference per the E8-05
evidence rule, the check stays red, and the band stays at 12%. Evidence and
sweep: `debris-model-difference.json` in the Starbase reports directory.

**E10-02 hot staging (DC8) re-evaluation.** The same fix was applied to the
DC8 pseudo-reference comparison (`docs/benchmarks/e10_02_hot_staging.json`,
regenerated). It moves every affected band the right way - SECO velocity
discrepancy 45.2 -> 37.0 m/s, SECO SMA 50.7 -> 34.5 km, SECO mass −2.9 ->
−2.3 t, separation position back inside its band - because DC8's MECO (125.5 s)
and SECO (513.663 s) also sit mid-segment. What still fails is the declared
hot-timeline representation split itself (OT ends all stack power at
separation; the twin burns the booster's post-separation floor burn), which is
larger on this timeline because the stack flies powered through the events, and
is disclosed in the benchmark, not absorbed.

## 重放和视图

从寄存器根使用 [landing environment](https://github.com/zangjiucheng/orbitopt/blob/main/envs/README.md),包括 gfold 0.3.2 。 选择 `dc7`或 `starbase`; 取消原空白记录的变量 。 与E5-05脚本不同,`run`采用位置输出目录:

```bash
export ORBITOPT_E508_VARIANT=starbase
python scripts/e508_new_vehicle.py run runs/e508-starbase
python scripts/e508_new_vehicle.py compare runs/e508-starbase
orbitopt flight to-scene runs/e508-starbase/recovery/run -o runs/e508-starbase/scene.json
orbitopt view runs/e508-starbase/scene.json
```

对于遥测,在**运行**中加入`runs/e508-starbase/recovery/run`. 选择 **车辆** 中的匹配目录工具, 用于其声明的对流覆盖 。 重播使用录制的上升段;没有重新优化设计.

## 增加停车轨道传播

在两个口译员中安装检出, 然后运行匹配计划 :

```bash
orbitopt crossenv run --plan src/orbitopt/data/crossenv_e508_starbase_plan.yaml \
  --env tudat=/path/to/tudatpy/bin/python \
  --env gfold=/path/to/landing/bin/python -o runs/crossenv-e508-starbase
ORBITOPT_E508_VARIANT=starbase python scripts/e508_new_vehicle.py \
  crossenv-check runs/crossenv-e508-starbase --no-run
```

For DC7, use `crossenv_e508_dc7_plan.yaml` and `ORBITOPT_E508_VARIANT=dc7`.
A single-environment replay omits parking propagation and its physical gate
remains `not_checked`, even when its ordinary mission report passes. The
joint checker must evidence all five physical checks; a nominal joint pass
does not clear the independent-model failures or the dispersion study.

## 证据文件

Each `tests/data/mission/reference/e5-08-new-vehicle[-dc7|-starbase]/` tree
contains the sealed inputs, `PROVENANCE.json`, `comparison.json` and mission
reports. The sibling `*-reports/` directory contains independent dynamics
and Monte Carlo JSON. The
[DC7 notes](https://github.com/zangjiucheng/orbitopt/blob/main/tests/data/mission/reference/e5-08-new-vehicle-dc7-reports/README.md)
and [Starbase notes](https://github.com/zangjiucheng/orbitopt/blob/main/tests/data/mission/reference/e5-08-new-vehicle-starbase-reports/README.md)
record the experiments and joint-gate outcomes. These summaries describe
committed results; the commands above establish results for a fresh run.
