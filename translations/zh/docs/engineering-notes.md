# 工程说明

执行决定、数字陷阱和轨道opt飞行任务管道的证据。 以下的测量描述命名的例子和硬件;它们并不是对其他输入的准确性保证.

## pygmo:`bfe`在包装前必须附加.

Attach a batch fitness evaluator to the raw UDA before wrapping it in
`pg.algorithm`. Not every pygmo algorithm supports `set_bfe`;
`optimize.runner` handles the supported cases. A missed attachment can fall
back to serial evaluation.

## SPICE:`J2000`和`ECLIPJ2000`不能互换.

框架因地球大约为23.4 °oooobliquity而异. 混合它们的状态可能造成数十万公里的错误. 每次交接时检查框架和来源;轨道电流路径使用`ECLIPJ2000`.

## RK4 台阶尺寸,对海岸来说是罚款的,对飞行来说是罚款的

近距离飞行,海岸大小的步子可以错过遭遇. 有文件记载的一个案例是81,733最接近的距离,600 s RK4步,4,379公里的步数为15秒或以下。 对正在研究的几何学进行反复的趋同检查;一个固定步骤不是普遍的准确性保证.

## Verification layer: adaptive integrator + event location (task #136)

The verification propagator defaults to tudat's adaptive RKDP 8(7)
(`integrator="dop853"`); `rk4` is available for comparison. Events use
root-finding on dense output, with fine re-propagation where supported. In the
Keplerian example in `examples/11_event_location_comparison.py`, a 600 s
sample interval missed a 1,837.400 km perilune by 48.42 km; event location
recovered it to within 1 mm. The result is example-specific. An encounter
whose range rate does not change sign between samples can fall back to a
sample-wise minimum.

## Verification report: every verification emits machine-readable JSON (task #137)

报告包括残余物、集成器设置和趋同、养护检查、模型和省略效果。 通常在扰动的轨迹中不保存能量和角动量;适用的比较使用只使用中心体的重运行. 报告采用`verification-report-1.0.json`计划. `orbitopt run`在现场旁边写报告;`ORBITOPT_VERIFICATION_REPORT_DIR`也允许库调用报告输出.

每张支票都在三个类别(任务# 243)下存档,所以一张通行证永远不会模糊实际显示的内容. *数值一致性* – 集成和定点是等式所说的(步骤趋同,养护,定点趋同). *模型一致性*-此模型的结果与参考文献模型,瞄准点或走廊(残基默认值)相匹配. *历史数据验证* – 模拟匹配一个已公布的,飞行的现实(Starship宣布的时间线;一个飞行任务一旦存在的遥测). 制片人必须将每张支票进行分类(`status_from_checks`拒绝其他方式);状态带有每张支票地图和每张分类的滚动,验证标签在其分类标题下列出支票. Bundles将各类别逐个报告。 添加的1.0图案是添加剂:没有类别的文件仍然像以前一样验证和渲染.

## GPU并非总能更快 -- -- 测量,没有假设

GPU的性能取决于批量大小和内核发射的俯冲. Precomputing Moon/Sun ephemerides从筛选环中移除了重复的解析;可选的导火索Numba CPU后端加速了最接近的处理筛选. 有关主机,计时,准确度和未计量的CUDA估计值参见[backend benchmark](benchmarks/fused_kernel_evaluation.md). 在更改后端默认值之前测量目标工作量 。

## 牛顿在真实(简单)投入上存在分歧,不只是理论上

一个粗糙的月球飞虫种子曾经送来一个已蒙蔽的牛顿溶解成分化. 目标检查了精细的海岸猜想和盖子以及牛顿台阶的后退. `tests/test_cislunar.py`练习之前失败的种子;趋同仍然依赖输入.

## Qt/VTK:截图API实际捕捉3 D视图.

`QWidget.grab()`可能返回黑色嵌入的OpenGL视图. 使用PyVista的`plotter.screenshot()`作为取景口,并将其与窗口捕捉相混合. `scripts/capture_screenshots.py`处理这个,包括Retina逻辑坐标放置. VTK测试需要在Qt的屏外插件无法创建OpenGL上下文的平台上显示.

## VTK 点标记: 恒定的明显大小, 和此边台阶的错误

VTK的`render_points_as_spheres`使标记在恒定的屏幕-像素大小,避免镜头缩放时的世界空间缩放错误.

## PyVista 滑动器: `interaction_event` 默认只释放

`add_slider_widget`默认发布后召回. 拖曳时使用 `interaction_event="always"` ; 见 `tests/test_pv_viewer.py` 。

## Cislunar / Artemis II自由返回管道,全程

The built-in mission uses an Earth-only Lambert seed followed by joint lunar
B-plane and free-return targeting in Earth/Moon/Sun dynamics. An independent
re-propagation checks the entry-interface angle against [−7°, −5°]. This is
a **vacuum geometric check**, not an assessment of atmospheric heating, loads
or survival; see [re-entry constraints](reentry_safety_constraints.md).

The 6,545 km perilune is a target input. The 2026-08-01 departure and 185 km
parking orbit are modeling choices, so the result is not a reconstruction of
the real Artemis II flight. The older `examples/05_artemis2_free_return.py`
targets flyby distance only and does not guarantee a return.

## 窗口搜索: 最小一个猪头不是一个盆地

`optimize.multistart`识别分离的本地迷你马,每个都优化,同时也搜索整个窗口. 它记录单盆地基线并合并重复的Ottoima. MGA窗口筛选同样保持了多样的第一腿盆地,并使用全MGA运行来重新排序. `tests/test_multistart.py`包含一个双Valley回归案例.

## Launch constraints in the fitness: measured effect on the optimum (task #153)

`LaunchConstraints`采用C3型车,并采用场地脱落限值进行滑板和批量健身. 它拒绝列表C3曲线外的有效载荷,而不是推算. 无效的考生会遭到分级的违纪行为;处罚方式需要足够大的重量来阻止违纪行为.

In the documented 2028 Earth–Mars example (Falcon 9, 4,000 kg), the
unconstrained optimum exceeded the C3 limit: 9.09 versus 9.00 km²/s². The
feasible solution departed about 2.4 days later and cost 26.8 m/s more. See
`examples/12_launch_constrained_transfer.py` and
[launch-data sources](launch-data-sources.md); these inputs are planning-level
estimates.

## Epochs: one leap-second-aware UTC -> TDB conversion (task #159)

`orbitopt.timescales` treats MJD2000 as a UTC label and converts to TDB seconds
past J2000 noon using leap seconds and the TDB−TT correction. It is checked
against SPICE and ERFA in `tests/test_timescales.py`. Earlier paths disagreed
by roughly 69 s for modern epochs, so frame and epoch conversions must use
the shared implementation.

## 统一任务管道(任务# 170)

`pipeline.run_pipeline`运行配置的生命周期 :

```text
config → constraints → search → optimize → select → refine → verify → export
```

它的`PipelineResult`记录每个阶段的情况,候选人,物理模型,数字检查和时间. 未使用的阶段为`skipped`;失败后的阶段为`not_run`. CLI也写了一条管道JSON报告. `geo-raising`和`mars-transfer`实施阶段记录.

## Candidate Selection and the Verification Gate (task #171)

`candidate_selection.py`删除不可行且几乎重复的候选人,在适用候选人限制时将剩余候选人排位,并赞成多样性. 验证大门记录了每位候选人是否被接受,被否决,被贬低或者未被评价. 一个失败或不受约束的候选人不是公认的成功。

对于火星来说,如果没有选定的候选者通过,管道可以输出退化或倒置的轨迹作为**尽最大的努力**,同时发出警告和`success: false`. 大门、场景和CLI带警告 单靠完成的计算并不意味着验证。

## Benchmark scenarios with public references / analytic solutions (task #245)

`orbitopt.verify.benchmarks`(CLI:`orbitopt benchmarks`)定义三种假想,每种假想都有*冻结的预期值*(一个已公布的编号或封闭式溶液)和*宣布宽容*;测量从未看到参考:

1. `kepler_two_body` - DOP853 轨道传播 定义的LEO- 类似状态结束
   ~ 1.28反对封闭式双体解决方案的革命(容忍).
   0.5 m / 5 mm/s),加上对数字运行进行精确的能量驱动检查.
2. `j2_secular_drift` -- one day of osculating J2 propagation for a
   sun-synchronous 700 km orbit (node drift vs the public +360 deg/tropical
   year reference, 0.01 deg/day) and a critical-inclination MEO orbit
   (冻结近地点:近地点漂移~0,0.02 deg/日).
3. `starship_booster_catch` -- the Flight 15 booster return solved by
   `orbitopt.viz.starship_ascent`(反弹+弹道坠落+着陆)
   与SpaceX公布的捕获量(~T +6 :~35,±60 s)和塔台相对应.
   捕获位置( ± 10 m) ; 公告绝不是输入 。

`tests/test_verify_benchmarks.py`运行在快速CI套件中的所有三个——回归门,同时也从它们的公式中重新识别被冻结的常数,因此地球常数的漂移或参考数字不能默默地移动门柱. `benchmark_report()`以验证 - report -1.0文件(没有新的计划);`orbitopt benchmarks`从命令行运行(故障时退出1).


## Closed-loop GNC data contracts and the simulation clock (task #418)

`orbitopt.gnc` defines the five records a closed-loop run exchanges — truth, observation,
estimate, command, control event — under one versioned schema bundle (`gnc-contracts-1.0.json`),
and a `ClockSpec` that makes every sample/update/tick instant a pure function of its tick index,
so a dropped frame deletes a record and never retimes a later one. Plant truth flows only into the
append-only `GncRunLog`: guidance sees a `GuidanceInput` exposing one `Estimate`, the guidance-side
module contains no truth/plant identifier (AST-enforced), and the run log's causal gates plus the
E12-08-policy run digest make a sealed log replayable, tamper-evident evidence. Rationale:
ADR gnc-0001; usage: `docs/gnc-contract.md`.

## Relative-navigation EKF (task #419)

`orbitopt.gnc.ekf` adds the navigation state the closed loop estimates: a 6-vector LVLH EKF whose
process models delegate to E12-02 (`CwProcess` → `propagate_cw`, `NonlinearRelativeProcess` →
`propagate_pair`, reference chief advanced with the Jacobian batch) and whose measurement models
extract from E13-01 `Observation` records -- Cartesian readings rotated through the caller's chief
state, native range/angle/rate observables read from the record metadata, never re-derived from the
Cartesian field (that would double-count the noise). Every update returns the innovation/NIS report
the consistency gates consume, and `make_estimate` publishes onto the contract with the LVLH
covariance mapped inertially by `frame_covariance`. `orbitopt.gnc.nav_metrics` is the acceptance
toolkit: chi-square bands, NEES/NIS, k-sigma coverage, per-cluster stats, and the R-weighted
stacked-observability sweep. Evidence: `docs/benchmarks/e13_02_relative_nav_ekf.{json,md}` --
gates in `tests/test_e13_02_relative_nav_ekf.py`; rationale: ADR gnc-0002.
