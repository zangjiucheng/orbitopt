# ADR会合-0006:独立逐样强力再飞(E12-10).

- **Status:** Accepted
- **Date:** 2026-11-26
- **Deciders:** E12C Rendezvous Qualification Closure (Epic #408), E12-10 (#412)
- **Code:** `src/orbitopt/rendezvous/robustness.py` (`passive_contingency_in_success`,
  `select_reflight_samples`, `build_reflight_spec`), `src/orbitopt/verify/rendezvous_refly.py`
  (`run_sample_reflight_in_current_environment`, `sample_reflight_comparison`,
  `verify_sample_reflight`),`src/orbitopt/verify/tudat_propagate.py`(涂装上.
  `ReusableCoastPropagator`), `src/orbitopt/crossenv/stages.py` (`rendezvous_refly_samples`)
- **Evidence:** `docs/benchmarks/e12_10_independent_reflight.json`,
  `benchmarks/bench_e12_10_independent_reflight.py`, `tests/test_e12_10_independent_reflight.py`
- **Builds on:** ADR rendezvous-0002 (TudatPy re-flight, ECLIPJ2000 rotation, crossenv
  ,ADR会合-0003(E12-07运动,根据这项任务修正),
  E12-08(封条计划)

## 背景和问题

E12-07在散射下测得LEO-RDV-001:264的搁置样本通过14. 每个数字都来自轨道导航仪本身。 tudatpy线是`NOT_CHECKED`,它的一个tudatpy进入点飞行了名义E12-05计划,没有扰动的样本. 成功测试还排除了名义核查员检查的被动应急释放. 在漏掉一个门后,一个样品可能会通过海岸,漂流到保持球体中。

## 决定

1. **Passive contingencies are part of success under E12-10 grading.** The
   campaign flag `passive_contingency_in_success` flies the three E12-04 releases
   (不烧在T 0, 门中止在T 1, 海岸从控点) 在每个样本上
   拥有扰动物理学. 进入内在保留空间的释放
   `passive_contingency_s`是一个名为`passive_<kind>`的失败的检查,分类
   `safety_violation`. 包装的竞选活动 保持旗子的关闭,所以E12-07
   文物仍在繁殖。 E12-10基准开启.
2. **The re-flight selection is a declared rule, not a hand pick.** Every
   hold-out success (cap 16, above the 13 the reference campaign yields), up to 8
   被动- 紧急情况失败, 首先是被动- 唯一失败, 边界
   每个故障类别样本(最接近其阈值)和
   `planning_infeasible`行记录为跳过. 总上限32, 断链
   `(cluster, index)`. 规则,计数 和是否每一个成功
   回流进入了文物。
3. **Each sample is re-flown with its own perturbation.** The spec carries the
   样品的重新规划门历元, 其已执行的烧伤(及其执行错误),
   它的真理J2及其拖动因子. 光谱被消化。 工人拒绝
   请检查哪些任务, 配置 sha 或文摘不匹配 。 tudatpy判决
   来源于`grade_trajectory_checks`,与主用模型的同级生,
   适用于独立飞行的轨迹.
4. **乐队是绝对的,并且来自测量.** 双双模型 飞来飞去
   同样的每样物理 在同一5的网格, 所以一个剩余测量
   仅限执行。 在23重流样本中,最大的残留物是
   4.5e-6在到达位置,5.8e-9在到达速度,9.2e-7在到达位置.
   网格最小范围,和 2.3e-6 m 覆盖所有 69 被动保持边距。 乐队
   0.01米,1e-5米/秒,0.01米和0.01米. 这至少是1000的两倍
   低于合同比额的1000次
   年级(50米,0.2米/秒,一个500米的保持半径).
5. **Wording.** The E12-07 conclusion is a **Robustness Characterization**.
   The E12-10 artifact may say **Independently Verified Robustness
   Characterization** only when the re-flight is `checked`, every hold-out
   成功被重传,没有判决翻转 没有乐队
   违反情况。 基准的标签来自这四个事实。 内容
   `outcome`球场表示发现了什么. 该标签证明*证据*,而不是
   设计: "Robust Security Qualification"的第一个措辞,读作:
   result for a design that succeeds on 4.9 % of the hold-out samples, so it was
   掉下来了 没有宣布成功率要求,所以文物记录
   费率,并不声称设计是坚固的。

## 第一次运行暴露了两条绳子缺陷

The first full runs reported 6 verdict flips (5 of 8 successes failing the
hold-point coast in TudatPy), 10 m arrival residuals and about 150 m of
min-range disagreement. A drag-free, J2-free re-flight of one sample still
disagreed by 156 m. Two point-mass integrations of the same pulses should agree
to millimetres, so these residuals were harness bugs, not physics:

- **固定步RK4 双腿过头了历元** Tudat的RK4停靠在 the
  第一个网格点通过请求的端。 560.64 s gate on a 5 s gring 结束
  在565 s,并燃烧1, 转移腿和扣点海岸都开始了
  从那个州。 海岸取样员现在把样品 扔到最后
  在它上采取一个部分步骤,它与主模型的包容性相匹配
  `_grid`.
- **图达特测量了平均射线球的拖曳高度。** 图达特的默认值
  Earth shape is a 6371.0 km sphere. The main model measures altitude from
  `EARTH_RE`(6378.137公里). 7.1千米乘以13%的密度差值,位于
  H = 50公里,到达时约11米. 拖车模型 现在搭载`body_radius_m`
  并将中心身体的形状固定在它上。

两者都固定了,每个Keplerian,J2和J2+drag变体都同意微米,完整的选择报告0翻.

## Results (reference campaign, hold-out arm, E12-10 grading)

| | E12-07基线 | E12-10 重新叙级 |
|---|---|---|
| 成绩 | 14 / 264 (5.3 %) | 13 / 264 (4.9 %) |
| 威尔逊 95% | 3.2 – 8.7 % | 2.9 – 8.2 % |
| 集束靴 95% | 2.7 – 8.0 % | 2.7 – 7.6 % |
| `safety_violation` | 69 | 87 |
| `arrival_accuracy` | 181 | 164 |

包括被动检查将18样本移入`safety_violation`:1前成功,17前到失败. 安全优先于进入分类系统。 每一个被动的失败都是悬点海岸。 74安全违规事件,18仅此而已. 独立再飞行包括23样本(13成功,8被动失败,2边界样本),0判决翻转和0频带违规.

## 后果

- 稳健的结果现在有独立的交叉核对,但设计
  它描述的不是坚固的。 95% 分散样本失败,
  持有点海岸一旦分级,即为主导安全模式。
- 选择是失败的样本,并非全部. 失败
  没有重播 仍然有一个主 - 模型 判决, 和文物名称
  重新流出的样本。
- 微米尺度的波段将会在改变
  集成器或场会. 这就是重点:这种改变必须是:
  重新测量,没有吸收。
