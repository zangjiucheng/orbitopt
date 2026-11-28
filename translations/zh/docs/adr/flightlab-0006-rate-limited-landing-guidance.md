# ADR飞行实验室 -0006:限速着陆引导(声明TVC链+落后意识定律).

- **Status:** Accepted
- **Date:** 2026-11-14
- **Deciders:** Recovery Robustness (Epic #248), E3-01 (#258), follow-up to
  R-05(ADR飞盘) -0003血统,`benchmarks/bench_r05_actuator_margins.py`.
- **Code:** `src/orbitopt/flightlab/recovery/landing.py`
  (`TerminalGuidance.tvc_lag_compensation`,
  `TerminalGuidance.tvc_lag_compensation_cap_deg`,
  `_lag_compensated_command`, `terminal_guidance_command`,
  `_ClosedLoopLanding`, `TERMINAL_GUIDANCE_E301`,
  `DECLARED_LANDING_ACTUATOR`), `src/orbitopt/flightlab/recovery/montecarlo.py`
  (`MC_PLAN_OPTIONS["landing"]`)
- **Evidence:** `docs/benchmarks/r05_actuator_margins.json`
  (`benchmarks/bench_r05_actuator_margins.py`),
  `docs/benchmarks/r07_nominal_dispersed.json`,
  `docs/benchmarks/r01_boostback_objective.json`,
  `tests/test_flightlab_recovery_terminal_guidance.py`

## 背景和问题

R-05 asked "how much does a 20 deg/s TVC slew limit cost the recovery
stack?" and answered honestly but with a blunt instrument: the limiter was
clipped onto the law's command, and the servo lag the repo's own JSBSim
spike-stage model declares (`gimbal_lag_tau_s = 0.2`, declared alongside
`gimbal_rate_limit_rad_s = 0.35` in
`src/orbitopt/data/flightlab/jsbsim/aircraft/spike_stage/spike_stage.xml`,
`<lag>`, then `<rate_limit>`, then the travel clamp) was left out of the
accounting entirely. E3-01 closes both gaps: the declared chain becomes the
default the recovery benchmarks fly, and the guidance stops treating the
actuator as pure loss.

连锁店比其个别要素更重要。 (n=32,种子1,默认分散,同样温暖的开始和裁判-`docs/benchmarks/r05_actuator_margins.json`):

| 栏 | 着陆制导 | 成绩 | 着陆 | 横向速度失败 | 已实现最大杀 |
| --- | --- | --- | --- | --- | --- |
| baseline_fl305 | FL-305定律,无限助动器 | 0.531 | 17 | 0 | 215.6 deg/s |
| r 05 _ 助变器 | + 20 deg/ shot 弹夹 | 0.500 | 16 | 3 | 20.0 deg/s |
| e301_lag_naive | + 0.2(系列,无赔偿) | 0.375 | 12 | 8 | 20.0 deg/s |
| e 301 _申报 | + 延迟补偿(ADR) | 0.438 | 14 | 5 | 20.0 deg/s |

基线的"边际"是无限滑雪假设的动脉——R-05测得54.9 deg/s(最大167.6)的平均达阵相位需求——天真系列链以特定,可移动的方式残忍. 每个附加的失败是`LANDING_LATERAL_SPEED_EXCEEDED`,基准JSON的`failure_records`中的每个样本轨迹定位:失败都发生在最后几百米,ZEM分支(`6 (−r − v t) / t²`)的需求在其中逆向硬化; a gimbal,需要0.2 s到达,20 deg/s到达那里,对斜锥进行限制循环,用1.1 - -5.3 m/s的横向速度左转. 转机没有"有点晚"到来——在垫子的响应时间里,法律要求的是被宣布的开动者无法实际做的事情.

## 所审议的备选方案

1. **飞链天真,盲目引导**(`e301_lag_naive`).
   column): honest, but 0.375, and it hands the ADR a failure mode the
   指导可以预见到。 设为基准栏,而非默认值.
2. **反应保留点火边界** : 缩短分流时间到
   所以法律会更早点燃/保留。
   首先测量,拒绝--它前装的是相同的无法追踪的ZEM
   而不是躲开它 直接降落 点火
   boundary stays the FL-305 law's (ADR flightlab-0002).
3. **已键入调用计时器状态**(缩写为ZEM / PD)
   跟踪误差产生的增益:按标称的大小写网格计量,
   在不固定终端阶段的情况下降解跟踪; 终端阶段
   是一个时间问题,而不是增益问题。 被拒绝。
4. **Make the law see the actuator at its two failure points**(通过):
   命令界面,ZEM分支停止时
   可追踪。 `TERMINAL_GUIDANCE_E301`(比例为20 deg/s,落后于0.2 s,
   `tvc_lag_compensation=True`)是默认值
   `MC_PLAN_OPTIONS["landing"]`.

## 决定

`TerminalGuidance.tvc_lag_compensation`(需要一个已宣布的`tvc_lag_tau_s`;关闭-位精确的FL-305 / R-05 - 否则)做两件事,两者都来源于所宣布的τ 单,没有新的自由键超出上限:

- **演员界面**(`_lag_compensated_command`):命令线索.
  以它自己的跟踪错误, `u = d + (d − s)` , 上限为
  `tvc_lag_compensation_cap_deg`(默认 10 deg),并且仍然有限杀.
  The error is taken against the *modelled lag state* the wrapper already
  携带,而不是需求衍生工具 -- -- 一个可变整合步骤不能
  炸开它。 真正意识到的金宝 跟踪需求而不是跟踪它
  一次反应,第一顺序的滞后 停留在杀前
  限制器正好是`spike_stage.xml`放的地方.
- **终端移交**:`t_h = t_go − settle_s − τ`赔偿.
  分流的最后一个反应时间 由被关闭的终端法飞行
  (`lateral_damping_per_s`,`lateral_stiffness_per_s2` - 已经是一个
  overdamped PD, ω_n τ ≈ 0.09, so the lag barely touches it), because a
  ZEM在响应时间内的突起 正是所宣布的需求
  链条无法跟踪 。 没有补偿,FL-305交接点
  站稳了

## 后果

- 现在所有回收基准 默认使用已宣布的启动器
  (`MC_PLAN_OPTIONS["landing"] = DECLARED_LANDING_ACTUATOR`;R-01;
  R-07证据再生(R-07名义回收)
  已声明两列的链条; 堆栈列匹配已声明的
  column above). The FL-305-era evidence files stay bit-exact -- those
  基准将`landing={}`标注在基线栏上。
- 宣布的连锁 仍然需要着陆 与无限的Slew
  基线(0.438 vs 0.531):四栏表的诚实解读
  FL-305堆的17着陆需要~216 deg/s. 这个
  compensation recovers 2 of the 5 landings the naive lag loses and halves
  不少于横向速度故障计数(8 - > 5);
  5是深角散射图(记录在基准JSON)中,
  没有20 deg/s 引爆器从走廊漏水中轻轻降落 -- -- 这是一种
  助推-客观问题(R-01),不是着陆法问题.
- 在宣布的链条下有边际的名义土地:封闭式坡道
  textbook-handoff case puts down with realised slew peaking at 18.9 deg/s
  补偿收紧了名义上的触地得分
  横向速度(0.030 - > 0.014 m/s).
- `ActuatorMargin`记录`tvc_lag_compensation`,所以运行的假设
  `bench_r05_actuator_margins.py` 保存
  the four-column before/naive-clip/naive-lag/declared comparison and the
  每一个故障的样本位置作为长期证据。
