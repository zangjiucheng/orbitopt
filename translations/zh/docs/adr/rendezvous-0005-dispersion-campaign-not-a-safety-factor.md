# ADR会合-0005:分散运动,而非安全因素(E12-07).

- **状态 :**接受;分类由E12-10修正(见下文*修正*).
- **Date:** 2026-11-25
- **Deciders:** E12 Orbital Rendezvous & Mission Operations (Epic #389), E12-07 (#397)
- **Code:** `src/orbitopt/rendezvous/robustness.py`, `src/orbitopt/data/leo_rdv_001_robustness.yaml`,
  `src/orbitopt/verify/tolerances.py`
- **Evidence:** `docs/benchmarks/e12_07_rendezvous_robustness.json`,
  `tests/test_e12_07_rendezvous_robustness.py`, `benchmarks/bench_e12_07_rendezvous_robustness.py`;
  独立飞行:`docs/benchmarks/e12_10_independent_reflight.json`,
  `benchmarks/bench_e12_10_independent_reflight.py`,ADR会合-0006.
- **Builds on:** ADR rendezvous-0001 (LEO-RDV-001, `plan_rendezvous`), ADR rendezvous-0002
  (whole-trajectory safety constraints), ADR rendezvous-0003
  (tudatpy再次飞行 其后卫这一运动)
  ADR相对-0001(`propagate_pair`,`_accel`),E3-03/E7-05.
  (样本大小规划,搁置学科),E10-03 /R-06(集群间隔,源敏感).

## 背景和问题

E12-03 planned LEO-RDV-001, E12-04 proved the plan satisfies its constraints *on
nominal*, E12-05 confirmed the nominal trajectory in an independent propagator.
None of that prices the mission against the things that are not nominal: a
navigation solution that is a few metres off, a burn a degree off its commanded
direction, a gravity field that is not exactly the declared J2, an atmosphere
the plan does not model at all. The task asks for a hold-out success rate with
credible intervals, a failure taxonomy, a sensitivity reading, and an
independent re-check of the results that matter. Four decisions were forced by
that wording.

这场运动决不能自行解决。 人们在选择扰动水平时所看到的单一样本池,报告的比率部分是对设计的一种衡量,部分是对运行者耐心的衡量。

间隔期决不能是关于独立的。 在一个重力场和一个空间-天气条件下抽取的样本不是该环境的独立图:一个差的场对每一个都不好,264相关样本上的二元间隔报告它没有精确度。

The truth model must differ from the planning model *only* by the declared
disturbances. A harness with a private re-implementation of the shared physics
measures the difference between two implementations, and reports it as robustness.

而独立的重新检查有一个在这个容器中不存在的后端. 诚实的状态 一个无法运行的交叉检查 是无记录的或者说 所以,从来没有一个安静的通过。

## 决定

**Train and hold-out arms on disjoint seed streams.** `parse_campaign` refuses a
declaration whose two seed bases are less than `max(n)+1` apart, so a sample
identified by `[seed_base, cluster, index]` cannot appear in both arms. The
hold-out arm is the headline; the train arm is printed beside it and explains
nothing. Both arms fly the *same* ground plan (`plan_rendezvous_safe`'s
constrained optimum) at *its* time-of-flight: what is priced is a locked design
under dispersion, not a per-sample re-optimisation. The re-plan a sample does
get is the guidance the spacecraft would actually run — re-open the gate on its
own estimated state, re-target burn 1 at the fixed time-of-flight, re-screen
against the fixed constraints — and a sample that no longer admits a screened
plan is a `planning_infeasible` outcome, a class of its own, not a flight with
fabricated margins.

**Cluster, then sample.** Each cluster draws the environment once (a J2 factor
and a log-normal density factor, median 1) from `default_rng([seed_base,
cluster])`;集群内的每个样本然后从 `default_rng([seed_base, cluster, index])` 中抽取自己的导航和执行错误,在禁用或零width 组件不消耗随机性时按固定顺序抽取. 因此集群是可交换单位,率上的间隔被两次引用:集合计数上的威尔逊间隔(如果集群是独立的万物图),和一个整集群百分位制的靴子,它们定价的是集群实际分享的. 未选择样本计数:它是以最小的`n`,其威尔逊间隔在申报的`target_ci_width`范围内,以最坏情况规划速率,然后四舍五入为整组.

**Truth is the planning physics plus the declared disturbances.** The two-body
and J2 terms are `orbitopt.relative.propagation._accel`, imported and not
restated; drag is the repo's `ExponentialAtmosphere` in a co-rotating frame; the
integrator is the same fixed-step RK4 discipline as `propagate_pair`. Drag is
present whenever the source is enabled — the campaign's premise is that the plan
ignores the atmosphere, so the truth carries the atmosphere and a log-normal
factor prices what is unknown about it. The navigation error moves the *estimate*
and not the truth: the vehicles start at the declared config states, and the
planner sees the chaser's state plus the declared sigma. The control test asserts
what this buys: with every family off, the flown arrival matches the planned
arrival to 1.6e-4 m, so every metre the campaign measures belongs to a
disturbance.

**Grade with the mission's own checker, at the campaign grid.** Every success is
an E12-04 pass (`evaluate_trajectory_safety`) on the trajectory actually flown,
plus the config's own budget and arrival tolerances, and a sample that violates
two gates is attributed once, by declared precedence
(`planning_infeasible > safety_violation > fuel_over_budget > arrival_accuracy`).
The truth grid is `sample_dt_s` = 5 s, coarser than the config's
`check_sample_dt_s` by necessity — a Monte Carlo cannot afford the plan-check
grid on 264 samples twice over — and `grid_convergence` re-flies the marginal
samples at 1 s to measure whether the interval can clip a violation, reporting
`grid_dependent` if a verdict ever moves.

**Three independent lines, and only one of them is an independent model.** The
spot check re-flies the samples whose verdict a small model difference could
move (tightest of each constraint, worst-arrival successes, one per failure
class), never a random cut. The *independent integration* line re-flies those
samples adaptively (DOP853 at `orbitopt.verify.tolerances.VERIFY_RTOL/ATOL`)
against the production RK4 — same force law, so it bounds integration error and
is measurable anywhere; that is why the tolerance pair now lives in a
module that does not import `tudatpy`. The *independent formulation* line
replays the plan on the linear CW model against E12-05's declared validity
bands — CW is not a candidate truth model for a mission with J2 and drag, so
this line bounds what an approximate model could explain, and is not
confirmation. The *independent model* line is the E12-05 TudatPy re-flight; with
no backend it is recorded `not_checked` with `REFLY_BACKEND_MISSING`, never
passed.

**Report the association and the marginal, and let them disagree.** Under joint
dispersion the sensitivity reading ranks each source by the rank correlation of
its own sigma-equivalents against failure, and labels a source it cannot assess
`uninformative` or `inactive_or_bias_only` rather than reporting a zero
correlation that would read as "measured, found nothing". `solo_campaigns` runs
the same layout and the same seeds with one family at a time, which is the only
reading that separates a source that hurts on its own from one that merely
co-occurs with failure.

## 后果

The headline number is a rate with two intervals and a per-cluster spread, not a
percentage. The campaign's declared levels are stress levels — the edge of what
mission analysis can bound per source (documented line by line in the campaign
YAML), not an RSS — so the failure taxonomy is populated and the sensitivity
ranking has a spread to rank. The measured answer for LEO-RDV-001, at the
packaged campaign (22 clusters × 12 samples per arm), is a hold-out success rate
of **0.053** (14/264): Wilson [0.032, 0.087], whole-cluster bootstrap
[0.027, 0.080], train arm 0.053. Failures: 181 arrival-accuracy, 69 safety
(keep-out and terminal cone), zero planning-infeasible, zero over-budget. The
two-impulse, open-loop design is fragile — a velocity error of a few mm/s is
carried through the drift leg into tens of metres the burns cannot see, and a
mis-pointed 7 m/s burn is a position error that grows with the drift leg. The
solo lineup is where the sources separate: burn execution alone leaves 0.064,
navigation error alone 0.508, drag alone 0.591, J2 alone 1.000 — pointing is the
one source that is mission-critical by itself. The joint rank-correlation
reading ranks `drag, execution, j2, initial_state`, and the two readings
disagreeing is expected: with a 95 % base failure rate the correlation against
failure saturates, which is exactly why the solo lineups run. The reading for
the design is recorded with the number — a mid-course correction, or a pointing
budget priced against the 50 m gate rather than against the launcher's
convenience.

Costs: a sample re-plan and truth flight costs about a second and a half, so a
full campaign with the solo lineup is ~1,800 trajectories and runs on the
compute host with `--jobs`; the re-plan assumes the onboard guidance re-solves
the transfer, which is a guidance assumption the mission's GNC would have to
confirm; and the 1 s convergence check is only as convincing as the samples it
picks, which is why the pick is the marginal ones and the reported number is the
largest observed change rather than a mean.

## Amendment (E12-10, task #412): Robustness Characterization

这场运动留下了两件事情, E12-10关闭——其中一件是措辞规则,所以属于这里.

1. **Success here does not include the passive contingencies.** The 69 safety
   *在命令的轨迹*遭到侵犯。 E12-04
   contract also carries passive contingency releases (no-burn at t0, gate abort
   在1,持点海岸,一个样本可以通过这场运动的成功测试
   失守后的无控制的海岸 漂流在守夜者中
   走廊。 。 。 E12-10 加`passive_contingency_in_success`(默认 `false`):此
   文物数量记录不变; E12-10基准重运行相同
   并重新报告这些标准下的费率和分类。
2. **The independent-model line was `NOT_CHECKED`, and the grid/physics
   comparison covered only 6/5 samples.** So, until task #412's per-sample TudatPy
   上面的结论是**强性字符化**
   ——对设计进行测量的散射研究——不得报告为
   强健或合格。 更强烈的措辞,** 独立验证的强健性
   特性**(它证明证据,而不是设计),只有在
   E12-10的文物记录了`checked`飞过
   样本其宣布的选定名称(每次成功、被动驱动的失败,
   每个失败类别有一个边界),没有判定翻转和每个样本
   在申报的横跨模型带内残留. 见
   `docs/benchmarks/e12_10_independent_reflight.json`(分类区块)和
   ADR会合-0006.

   Outcome: under the E12-10 criteria the hold-out rate is 13/264 (4.9 %, cluster
   CI 2.7–7.6 %) against this campaign's 14/264. The TudatPy re-flight of 23
   selected samples agrees with every main-model verdict. This campaign's own
   numbers remain a Robustness Characterization.
