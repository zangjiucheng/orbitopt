# ADR飞行实验室 -0008:抽样数据控制环和RK4厂(E6-04数字控制审核).

- **Status:** Accepted
- **Date:** 2026-11-15
- **Deciders:** E6 6-DOF Production Maturity (Epic #265), E6-04 (#276)
- **Code:** `src/orbitopt/flightlab/sixdof/rigidbody.py` (`propagate`,
  `_deriv`), `src/orbitopt/flightlab/sixdof/tvc.py` (`AttitudeTvcLaw`,
  `run_closed_loop_ascent`), `src/orbitopt/flightlab/sixdof/gravity_turn.py`
  (`propagate_point_mass`, `_point_mass_deriv`),
  `src/orbitopt/flightlab/guidance/upper_stage.py`
  (`UpperStageGuidance.dcm_from_state`)
- **Evidence:** `tests/test_flightlab_e6_04_step_convergence.py`,
  `benchmarks/bench_e6_04_step_convergence.py`,
  `docs/benchmarks/e604_step_convergence.json`

## 背景和问题

`AttitudeTvcLaw`是一个*状态*控制器——一个跟踪器滴答,两个集成状态和两个限速的激活器模型——在`dt_c_s`采样,并持有零序到厂上,与`dt_s`的固定步RK4结合. 一项预防性审计(没有观察到缺陷)提出了两个问题,即两个网格使非三进制:控制器是否在RK4中段时间推进,控制步骤和集成步骤的错误是否可以完全分离,以及它们的名称所隐含的命令。

审计前,`propagate`在**每个** RK4阶段评价了控制法(舞台衍生品重算`control(t_stage, y_stage)`),点质量宣传器对`UpperStageGuidance`做了同样评价.

## 调查结果

**F1 — the tick fired at stage times and its command leaked one stage back.**
With the shipped default `dt_c_s == dt_s == 0.02` the tick fired at the *k4*
stage: the tracker/actuator state advanced with the stage-predicted state
`y + h·k3` rather than the accepted boundary state, and the post-tick command
entered the *current* step's k4 derivative — the next interval's command was
applied a partial step early. Measured consequence (closed loop, single
90→60 deg/12 s ramp, `dt_c_s` fixed 0.04): a **first-order `dt_s` plateau**
— final-altitude error −1.78e-3 / −8.29e-4 / −3.55e-4 / −1.18e-4 m at
`dt_s` = 0.04 / 0.02 / 0.01 / 0.005, halving ratios 2.15 / 2.34 / 3.00 ≈ 2.

**F2 — unaligned grids ticked at mid-stage times.** With `dt_c_s = 0.025`
against `dt_s = 0.02` the tick fired at `t + h/2`, i.e. at a stage estimate,
so the actuator substep and the tracker `dt` were driven by a
non-accepted state.

**F3 — a control tick below the plant step is silently decimated.** A fixed-
step plant cannot resolve a tick faster than its own step; pre-audit
`dt_c_s < dt_s` just ticked once per step with the merged period and nobody
said so.

**F4 — the guidance slew-limiter state was corrupted every stage pair.**
`UpperStageGuidance.dcm_from_state` slewed its command only when the clock
had *strictly advanced*; a re-read at a time it had already answered — which
RK4 guarantees, since k2 and k3 share `t + h/2` — fell through the guard,
returned the **raw un-slewed** pointing *and* re-seeded the slew memory from
it. The command-slew limit (`command_slew_rate_deg_s`) was therefore not
enforced across stage-pair re-reads on both the 6-DOF
(`GuidedUpperStageControl`) and the point-mass (`_point_mass_deriv`) paths.

**F5 — interior program kinks are order-zero plateaus.** A waypoint program
whose slope changes mid-flight is an O(1) discontinuity of the feedforward
command at the kink tick: boundary accumulated-time jitter of one ulp samples
the slope on either side of it. Measured: a 0.137 m final-altitude cliff
between `dt_s ≤ 0.005` and `dt_s ≥ 0.01` runs with *identical* tick
schedules — a plateau no amount of `dt_s` refinement removes. This is not a
code defect (the program itself is discontinuous in `C¹`); it is recorded so
a two-segment ascent program is never used for step-size validation, and the
benchmark scenario is a single ramp for exactly this reason. Actuator
travel/slew limit engagement is the same story in nonlinear form — limit
engagement is a discrete event — so the audit scenario also keeps margin on
both limits.

**F6 — loop contraction masks integration error in the loop output.** With
the audit fix in place, the closed-loop terminal state is invariant to
`dt_s` at the 2.3e-8 m float floor for every `dt_s ≤ 0.04` (the tracker
absorbs the truncation transient like the disturbance it is). The plant's
truncation order therefore has to be measured where it is visible — the
open-loop plant — and it is **4th order** there (vacuum plant, constant
command, measured ratios ≈ 16 per halving). The `dt_s` choice must still
resolve the plant, not just the loop: guidance states, aero and the J2
torsional pendulum do not get a feedback path.

## 决定

**D1 — one law evaluation per integration step.** `propagate` samples the
`ControlLaw` once per step, at the step-boundary state, and holds the
returned `ControlCommand` constant through all four RK4 stages. That is
exactly the law's declared ZOH semantics: a sampled-data controller knows
one instant per tick and no mid-stage estimate may advance the tick, the
tracker integrators or the actuator state. `run_closed_loop_ascent` and
every existing test semantics survive: ticks, history rows and sample-grid
re-evaluations all happen at boundaries.

**D2 — the point-mass path holds its guidance the same way.**
`propagate_point_mass` samples `command_from_state` once per step at the
boundary and passes the held `(Tec2b, throttle)` tuple to all stages. An
open-loop program (a pure function of stage time/state) stays
stage-resolved — holding it would be wrong.

**D3 — 在已经回答的时间重读的指南返回控件命令**,永远不要原始指针,永远不要重新播种致命的内存(F4).

**D4 — `run_closed_loop_ascent`在`dt_c_s < dt_s`时发出警告.**(F3):有效控制期变为`dt_s`.

**D5 — the default steps stay (dt_s, dt_c_s) = (0.02, 0.02).** 证据(`docs/benchmarks/e604_step_convergence.json`,12 s 单兰普上升段) :

- 植物断裂(真空,`dt_c_s`固定,参考`dt_s = 0.0025`):
  4th order — 9.2e-6 / 5.7e-7 / 2.5e-8 m at `dt_s` = 0.04 / 0.02 / 0.01
  (订购4.01,4.51;下一对位于参考浮楼);
- closed-loop output vs `dt_s` (dt_c fixed 0.04): flat,
  ≤ 2.24e-8,每 `dt_s ≤ 0.04`( F6);
- 控制盘式(`dt_s = 0.005`固定,参考`dt_c_s = 0.005`):
  ZOH 错误, +18.9 / +6.8 / +2.3 毫米
  `dt_c_s` = 0.04 / 0.02 / 0.01 – 这个频道,而不是 `dt_s` , 主宰着这个频道.
  默认总错误;
- 默认 vs 精制 `(0.0025, 0.0025)` : +7.9 mm 高度, +6.1e-4 deg 倾斜,
  +6.0e-6 rad 最大跟踪器在~1 km的攀登(0.0008%)上发生误差,已完全清点.
  以`dt_c_s`任期为准;
- 不对齐的勾选符会向网格缓冲:每个勾选周期在
  [`dt_c_s`, `dt_c_s` + `dt_s`) with boundary error ≤ 2.2e-13 s, and the
  勾选时间表( 因此, 解决方案) 只取决于崩溃的时间表
  (`dt_c_s` = 0.05: 240 ticks at `dt_s` = 0.05 and 0.025 with a 1.9 mm
  最终高度残留——空表穿越时间,F5,对200滴答
  断裂的`dt_s` = 0.02网格).

Both defaults sit at the knee of the cost curve: `dt_c_s = 0.02` is the
coarsest common F-04 tick, and at `dt_s = 0.02` the plant truncation is two
orders below the ZOH term. Halving either buys ≤ 1e-5 m closed-loop for ≥ 4×
the steps — production propagation is the F-05 attribution study, whose
pre-declared phase constraints have 1e3–1e6 m margins.

## 后果

- 法律每一步看一个评估, 所以一个状态法的概念
  时间网是集成网;内部时间表的法律有变化
  由它们先前在ZOH取样的半步相(最多)
  正确的阶段。 与旧的中阶段相对应的开放法则
  由 `dt_s` / 2 进行相位换乘;在回转中没有.
- 每个步骤进行一次控制评价,而不是四加:闭路
  上升段是~3×每步法律呼号便宜.
- 与 `dt_c_s < dt_s` 的现有测试配置( 0.05 / 0.02 成对)
  `tests/test_flightlab_sixdof_tvc.py`)现在发出F3警告;运行.
  警告是重点。
- 步骤大小趋同被锁定
  `tests/test_flightlab_e6_04_step_convergence.py` 反对罪犯
  基准 JSON; 重运行
  `PYTHONPATH=src python benchmarks/bench_e6_04_step_convergence.py -o
  Docs/benchmarks/e 604 _步_趋同. json'再生曲线.
