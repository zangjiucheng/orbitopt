# ADR会合-0003:tudatpy双飞飞艇作为独立的验证(E12-05)重新飞行.

- **Status:** Accepted
- **Date:** 2026-11-25
- **Deciders:** E12 Orbital Rendezvous & Mission Operations (Epic #389), E12-05 (#395)
- **Code:** `src/orbitopt/verify/rendezvous_refly.py`, `src/orbitopt/verify/tudat_propagate.py`,
  `src/orbitopt/crossenv/`(对接_级)
- **Evidence:** `docs/benchmarks/e12-05-refly-run/` (sealed exchange + stage report),
  `tests/test_e12_05_independent_verification.py`, `benchmarks/bench_e12_05_independent_refly.py`
- **Builds on:** ADR relative-0001 (LVLH frames, `propagate_pair`), ADR rendezvous-0001
  (LEO-RDV-001 config, `verify_plan`), E5-03 crossenv stage protocol, task #261 degraded rule

## 背景和问题

E12-05 verifies the LEO-RDV-001 plan with an *independent* propagator: both
spacecraft re-flown in TudatPy with the burns applied as impulses, reusing the
E5-03 crossenv stage so the verification runs in a second environment
(tudatpy 1.0, separate interpreter) and arrives through a sealed, revalidated
exchange. Three constraints shaped the design.

首先,接受要求错误来源为*可分离*:数字轨道传播错误,大致模型错误(CW vs 非线性),优化不趋同和交叉模型分歧是四个不同的负债,一个混合的`passed`会将其中任何一个隐藏在另一个后面.

第二,tudatpy翻译的SPICE内核池定义了完全没有GCRS / ICRS帧(`SPICE(UNKNOWNFRAME)`),一个被否决的帧使得tudat *将 轨道传播 终止于 t = 0 不提高*,递回一个一模一样的历史——无声的截断,是验证带最糟糕的故障模式.

Third, when the re-flight is forced out of GCRS, the frames are *not* related
by the ~25 mas precession one naively assumes: GCRS and ECLIPJ2000 differ by
the 23.44° obliquity of the ecliptic. A zonal J2 field is axisymmetric only
about the spin axis, so feeding GCRS components in as ecliptic components
misplaces every J2 torque — measured at 133 m of false cross-model
disagreement on the hold point (and a matching min-approach offset), orders of
magnitude above the integration error being certified.

## 决定

1. **Fly in ECLIPJ2000, rotated correctly.** The config's GCRS states and the
   惯性烧伤由 IAU76 obliquity 矩阵旋转,大约为 x(已验证)
   against SPICE `compute_rotation_matrix_between_frames("J2000","ECLIPJ2000",0)`
   to 1e-7) before propagation in ECLIPJ2000 at anchor epoch 0.0. Every graded
   数量是头-LVLH或一种规范,在单一常数下变化不定。
   两个航天器都采用了旋转方式——因此不需要作任何比较。
   `run_refly_in_current_environment`的中止守护者拒绝任何历史
   将Tudat的无声t=0 流产变成
   大声的`ReflyBackendError`.

2. **Force the config's exact physics onto the tudat environment.**
   `central_gravity={"gravitational_parameter": mu, "reference_radius": Re,
   "j2": J2}` overrides the SPICE-default gravity so the only differences
   left are gravity *formulation* and *integration*. Residual: **0.16 mm**
   hold-point position error against the main nonlinear J2 model.

3. **Four separate error-source lines, never one blended verdict.**
   `numerical_propagation`(分析双体重播+步/容忍扫描),
   `approximate_model_cw`(CW与非线性参考),
   `optimizer_nonconvergence`(落地残基+Δv位),
   `independent_model_agreement`(在四个接收点对主飞行)
   数量。 证据类型是`independent_model`,`analytical`,
   `numerical_consistency`;`historical_telemetry`在结构上缺席.
   (0检查). 缺少后端 `not_checked`
   `REFLY_BACKEND_MISSING`, exit code 2 — never a fake pass (task #261).

4. **Compare sweeps at the latest common sample epoch.** A fixed-step RK4
   将样品储存在自己的网格上,并将要求的尾端*跨界*;
   比较最后的样本将测量时间抵消(60 s 网格外观)
   2.3e5 m错误位于7.5 km/s),而非集成错误. 每次扫荡都是
   与3.75最常使用的历元≤腿的参照
   结束; 适应性的 DOP853 容忍轴完全终止在腿端
    (residual 0.0 m, order n/a), the step axis shows order **4.10** and a 21.6
    m position / 0.023 m/s velocity bound at 60 s steps, 2.3× below the
    declared 50 m. Each metric is sliced per quantity -- position over both
    crafts' position columns of the concatenated state, velocity over the
    velocity columns; the original baseline normed the mixed pair state
    (`d[:6]` / `d[6:]`), reporting the deputy position difference as a
    velocity, corrected and regenerated under E12-09 (task #411).

5. **The circular-orbit Kepler replay is exact.** Both initial orbits are
   恰如其分的圆形(e ≈ 0),其中古典开普勒元素管道
   乘以 e; `verify/benchmarks.py` 现在使用纬度参数表示
   平均值异常和关于(唯一)角动轴的旋转。
   残留与点质量重飞:**1.1e-7 Zmm**.

## 后果

- 跨- 模型波段 1.0 m / 0.01 m/s 与 > 6000 × 比值(测量)
  1.6e-4 m / 5.9e-8 m/s); CW波段 600 m / 0.5 m/s hold at seased 252.7 m /s
  0.20 m/s,记录线化规划师模型花费多少.
- The E12-09 slice correction (task #411) does not move any E12-05 hold-point
  结论:标题协议比较接受数量
  直接,永远不要通过扫荡切片, 重复在1.6e-4 m /
  通过。 只有数字行的扫荡列变化——步轴60 s
  位置绑定移动 15.2 → 21.6 m(仍为 2.3 × 在其 50 m波段内)和
  步轴速度列变成一个真实的 0.023 m/s 而不是
  副位置差异,它曾经显示。
- The obliquity lesson generalizes: *any* frame change in a J2 mission is a
  物理改变,不是记账。 警卫评论住在
  旋转站点, 所以下一个框架编辑先满足它 。
- tudatpy是专用口译中的dev/验证依赖,而不是一个
  软件包要求:没有软件包,所有东西都会降解为`not_checked`。
