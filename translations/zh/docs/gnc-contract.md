# GNC模拟合同(E13).

闭路制导,导航和控制在轨道opt交换中运行 精确的五种记录,在一个钟上,每个运行封存到一个摘要. 这页写了每个记录所说的话和对绳索的保证;`src/orbitopt/schemas/gnc-contracts-1.0.json`是规范的策略. 设计原理见ADR gnc -0001.

## 五个记录

| 类型( `kind` 字段) | 说 | 卡莉丝 |
| --- | --- | --- |
| `orbitopt.gnc.truth-1.0` | 工厂其实是什么 | 嵌入式 `relative-state-1.0` 文档,动态勾选 |
| `orbitopt.gnc.observation-1.0` | 检测到的传感器 | 状态估计 + 对角位置/速度共变,传感器 id,样本勾选,到达时间,来源勾选 |
| `orbitopt.gnc.estimate-1.0` | 过滤器相信什么 | 状态估计 + 全 6 x 6 共变,源观测 ids |
| `orbitopt.gnc.command-1.0` | 要求提供何种指导 | 命令的 delta-v, 激活器 ID, 火灾时间, 请求的激活时间, 源估计 ID |
| `orbitopt.gnc.control_event-1.0` | 实际执行的 | 实现 delta-v( `executed` 假时为零),源指令 id |

未执行的`ControlEvent`必须携带一个零实现的delta-v:误射被记录为不执行,绝不是静火燃烧. 真相只流入运行日志——见下文隔离保障.

## 一个钟

`ClockSpec` (in `orbitopt.gnc`) declares the run epoch (a `TT`/`TDB` `Epoch`), the dynamics step,
one `SensorSpec` per sensor (period, phase, latency, `dropped_ticks`), the estimator cadence and
the controller cadence. Every instant is a pure function of the tick index: `t0 + phase + n*period`.
A dropped frame deletes that measurement and never shifts the next one; declared latencies are
enforced by the run log on append.

## 运行和重播

    from orbitopt.gnc import ClockSpec, SensorSpec, EstimatorSpec, ControllerSpec, \
        SensorNoise, ProportionalLaw, simulate, write_run_log, read_run_log

    log = simulate(run_id="rdv-01", clock=clk, frame="GCRS", target_id="tgt", chaser_id="chaser",
                   chief_r_m=..., chief_v_m_s=..., deputy_r_m=..., deputy_v_m_s=...,
                   laws={"chaser": ProportionalLaw(1e-2, 0.2)},
                   noise={"navcam": SensorNoise(1.0, 0.02, 7)}, t_end_s=80.0)
    write_run_log(log, "rdv-01.json")     # seals run_digest
    log = read_run_log("rdv-01.json")     # re-validates every record and the digest

`read_run_log`拒绝一个其记录经过编辑(digest mission)的日志,其测量数据声称其样本或到达之前,其估计值在到达之前,其激活时间错过了要求的时间,或者其任务名册历元不同意时钟——验收墙在`tests/test_e13_01_*`试验为负第一.

## 隔离保障

`run_guidance_step(law, feed, ...)` is the only way guidance sees the loop; the law receives a
`GuidanceInput` exposing exactly one `Estimate` (`__slots__ = ("_latest",)`). Feeding a
`TruthSample` (or any non-estimate) raises `TypeError`, and `orbitopt.gnc.feed` contains no truth or
plant identifier — an AST test in `tests/test_e13_01_truth_isolation.py` fails the suite if one is
ever added, so the guidance side of the package cannot even name plant state.

## 传感器 模型(E13-03).

`orbitopt.gnc.sensors`声明*观察来自何处,如何说谎*. `RelativeSensorModel`由任务配置文件(`orbitopt/schemas/gnc-sensor-config-1.0.json`;`reference_task_config()`船船为文件质量配对)构建,并声明每个通道:白噪sigma(严格为正——拒绝零变位请求),恒定偏差,以及每个错误参数(引用或`DESIGN ASSUMPTION: …`)上强制`source`出处字符串. 已执行配置文件:`polar_range_angle`(范围+LOS方位/线程,可选多普勒范围率)和`cartesian_relnav`(偏差相对位置,可选速度);新频道或配置文件是`CHANNELS`/`SENSOR_KINDS`的新行,而非新类.

四个错误机制以及谁拥有它们:

| 错误 | 模型 | 墙 |
| --- | --- | --- |
| 白噪声 | 高斯人,在`(seed, tick)`上播种,就像每个轨道平面图一样 | sigma 必须 > 0; 切换配置的瞬间 |
| 偏见 | 每次运行时常数, 在噪音前添加 | `bias_covariance_mode`: `none` (filter models it; NIS inflates by `Σ bias²/σ²` — demonstrated) or `bounded` (declared variance + = bias²) |
| 延迟 | 时钟不变,从不随机 | `t_available == t_sample + latency`执行E13-01;`measure()`拒绝异议勾选 |
| 数据损失 | / tick Bernoulli + 几何暴动( 平均值 ≥ 1 ) + 明确情景勾 | generated as `dropout_schedule(n_ticks)` from `(seed, n_ticks)`, merged into `sensor_spec()` → the clock stays the single timing authority |

A dropped frame is the *absence* of an Observation (`observe()` returns `None`; the
`SensorFix` layer carries the explicit `valid` flag and reason for plant-side bookkeeping).
Observation output maps the diagonal measurement covariance through the channel Jacobians
(polar position: `(Jᵀ W J)⁻¹`; range-rate-only velocity: rank-1 `σ² r̂ r̂ᵀ`, honestly along-LOS);
a sensor with no velocity channel ships no velocity covariance rather than an invented one.
The polar reconstruction refuses the ±Z boresight singularity (`1e-6·|r|` of azimuth).
`measurement_nis(fix, truth)` is the shared measurement-consistency statistic (E13-02 consumes it).

Consistency evidence: `tests/test_e13_03_*` (per-error moments, NIS gates) and
`benchmarks/bench_e13_03_sensor_consistency.py` (sealed at
`docs/benchmarks/e13_03_sensor_consistency.json`: chi-square moments z-scored < 4, polar
covariance coverage 0.9985, bias inflation predicted). Filter-level NEES was deferred to the
merge of E13-02 (#419) — no Estimate stream existed on that branch — and the reason was asserted
in `tests/test_e13_03_nis_consistency.py` so the deferral failed loudly when #419 landed. #419
has landed: the deferral is discharged there by #419's sealed 6-DoF NEES gate
(`docs/benchmarks/e13_02_relative_nav_ekf.json`).

## 独立封闭式验证(E13-07)

`orbitopt.verify.gnc_closed_loop` 等级 封存的运行日志带有四个SEPARATE 错误分类,从未混入一个旗帜;每条线路都带有自己的证据类型和自己的`cannot_prove`限制:

| 类 | 提问 | 证据类型 |
| --- | --- | --- |
| `numerical_error` | 重新整合所记录的命令是否与所记录的真理(步骤精细扫描+容恕门)汇合? | `numerical_consistency` |
| `model_error` | 一个INDEPENDENT宣传员(tudatpy,多角论所记录的事件)是否转载了真相? 缺席后端:`not_checked`,从未估算. | `independent_model` |
| `navigation_inconsistency` | 观察和估计在统计上是否与它们的DECLARED(NIS /NEES)的共性一致? | `analytical` |
| `optimizer_control_nonconvergence` | 每项命令是否都在其宣布的限度内执行,生产者的趋同断言(如果有的话)是否与记录一致? | `analytical` |

    from orbitopt.gnc import read_run_log
    from orbitopt.verify.gnc_closed_loop import verify_gnc_closed_loop, verify_evidence_binding

    report = verify_gnc_closed_loop(read_run_log("rdv-01.json"), out_path="rdv-01-gnc.json")
    verify_evidence_binding(report, "rdv-01.json")   # digests re-match, or ValueError

重播在激活瞬间分拆段,并携带PRE-冲动真理样本在每次(根据参考sim事件优先级的证据);一个短于两个样本的真理流被记录为`not_checked`,没有得分;验证文摘将这个判断绑在这个准确的运行上——翻转任何检查,或者将同一报告指向另一个运行,上升. 验收:`tests/test_e13_07_independent_closed_loop_verification.py`.

## 任务工作室在Flight Lab(E13-08)重播.

`orbitopt.gnc.studio`重播封存的运行日志为Flight Lab运行目录(`manifest.json`+轨迹+`gnc-studio-1.0`侧车在`gnc.json`),因此闭路任务可以在时间线上观看,而不是只从报告读到:

```console
orbitopt gnc studio --leo-rdv-001 -o runs/gnc-studio        # fly, verify, replay the reference sample
orbitopt gnc studio --run-log rdv-01.json --report rdv-01-gnc.json -o runs/gnc-studio
orbitopt view runs/gnc-studio                               # or `orbitopt flight to-scene`
```

重播显示真相对斯估计轨迹(`chaser_est`轨迹将记录的相对状态锁定在真相首席上——显示生成;可审计估计保留记录),共变椭圆形作为半透明场景体,测量帧下降,燃烧事件和监督状态变化作为关键事件,报告曲线作为遥测通道. 估计音轨的`gnc_est_pos_err_m`,`gnc_est_vel_err_m_s`,`gnc_nees_per_dof`,`gnc_cov_pos_trace_m2`和`gnc_cov_1sigma_rho_m`列是验证报告逐字记录中的`navigation_consistency`列,在得分估计时间表上——没有重插,所以曲线和报告同意这个位.

运动日志记录了真核植物使用的引力常数和拖动模型. 校验者使用燃烧前的真伪进行观察,并记录在估计之前或之后是否发射过同一种Tick冲动. 更新旧的竞选记录 没有记录的模型进行这些检查

两条证据线在侧车厢和GNC标签中分别存在:`local_replay`携带运行日志摘要,用文字表示重播是NOT独立;`independent_verification`携带E13-07报告摘要和四个错误等级的判断,每条都得分(一个`not_checked`分类显示其原因,例如`REFLIGHT_BACKEND_MISSING`). 一份对日志不作重算的报告(`verify_evidence_binding`)被拒绝. 没有被消耗的报告意味着标签显示一个诚实的缺口,从不显示绿色. 验收:`tests/test_e13_08_gnc_mission_studio.py`.
