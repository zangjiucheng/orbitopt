# Flight Lab E4 — JSBSim 6-DOF spike findings (FL-401, #206)

E4 go/no-go ADR(Epic # 196)的输入. 代码:`src/orbitopt/flightlab/sixdof/`,模型:`src/orbitopt/data/flightlab/jsbsim/`,检查:`tests/test_flightlab_sixdof_jsbsim.py`(在顶端申报的阈值).

## 已经建成的东西

- `spike_stage`:从零开始写一个液体阶段(J246改为结构)
  only). 10 t dry + 60 t LOX/fuel (MR 2.5), Isp 320 s, 600 kg/s → 1.883 MN
  vacuum thrust, 100 s burn. Rocket engine + nozzle thruster with `<orient>`,
  two FCS actuator channels (lag τ = 0.2 s, 0.35 rad/s slew limit, ±5° travel)
  驾驶`propulsion/engine[0]/{pitch,yaw}-angle-rad`. 零空气动力学,
  无地面反应,代号喷嘴出口区(任何高度的真空推力).
- `SixDofSim` driver (jsbsim 1.3.3 Python API): geodetic/attitude ICs, ignite,
  SI逐步采样,
  `BodyTrack`(框架`ECEF`)加列:之四(来自`Tec2b`),
  体率、血压/实际(pitch、yaw)、推力、节流器、惯性、
  CG,海拔高度,TVC瞬间.

## 有效(计量,dt = 0.01 s)

| 检查 | 结果 | 宣布的容忍 |
|---|---|---|
| Vertical hop Δv vs. Tsiolkovsky − ∫(g − ω²r) dt, 30 s, throttle 1.0 / 0.5 | 2e-4 rel | 1e-3 |
| 推力 vs. Isp·g 0·ṁ | 准确 (1e-9) | 1e-4 |
| 质量对m 0 − ṁt | 在一个流动框架内 | 1帧+1千克 |
| 质量, Ixx/Iyy/Izz 严格减少; CG 移动船尾( LOX 向前) | 对 | — |
| 发动机耗尽时饿死,质量 干 | 对 | 1公斤 |
| TVC moment vs. T·sin δ·(_ stra -_ cg). | 准确 | 1 % |
| q(t) vs. QQ M/Iyy dt (1 s step) (中文(简体) ). | 0.6 % | 2 % |
| 演员对一阶延迟(图斯汀,半步延迟) | 0.03% cmd | 1 % |
| 费率限制/旅行剪辑得到遵守和有效 | 对 | 1e-6 |
| dt 半衰期( 0.01 → 0.005 s, 20 → gimballed burn): *%pos/flown、%v/speed、态度 | 1.1e-3,6e-4,0.027 °;体率1.4e-5 rel | 2e-3, 2e-3, 0.05°; 1e-4 |

运行时间:~5 k步/s单行本(全测试文件~5 s).

## 什么没有,有 恰克和工作

1. **`set_dt()` must come before `load_model()`.** FCS components (actuator
   延迟/ 速率过滤器) 在构建时缓存执行者 dt; 调用
   `set_dt`之后悄悄离开FCS,在默认的1 /120 s同时离开.
   动态使用新步骤(旗下反应出独立**)
   frame*). Workaround: driver calls `set_dt` first.
2. **Locations need `<x>/<y>/<z>` child elements.** `<location unit="M"> 1 2 3
   </location> ' 无差错地将其描述为(0,0,0)——CG,所有坦克和推进器.
   TVC产生零瞬间 没有警告。
3. **气轴名称**为`DRAG/SIDE/LIFT`(+`ROLL/PITCH/YAW`);`MOMENT`等.
   拒绝。 所有三根力轴都是零时需要的
   `<ground_reactions>`元素必须存在(空是细的).
4. **必须声明驱动文件属性**(`<property>` 英寸
   `<flight_control>`)或`run_ic()`失败.
5. **Gimbal 标志 :**正数 `pitch-angle-rad` / `yaw-angle-rad`在船尾
   推进器给出了鼻下/鼻左瞬间; FCS应用增益 - 1所以
   positive command == nose-up (q > 0) / nose-right (r > 0).
6. **One-frame latencies:** the engine lights one frame after the ignition
   命令(推力 0在帧1)和质量报告一个帧 stale. 两者都是
   O(dt),使集成在 dt 中只汇合 **第一顺序**
   (误差减半,以dt计)——可接受,对于任何
   精确预算。
7. **Body rates are relative to the rotating ECEF frame** (`velocities/p,q,r`);
   惯性率是`pi/qi/ri`. 期望~ 1e-8 rad/s 横轴率来自
   地球自转.
8. **无四角财产**;使用`fdm.get_propagate().get_Tec2b()`(返回 a)
   `numpy.matrix`; 驱动程序压制了待发折旧警告).
   Euler角度为垂直飞行器的gimbal-clocked.
9. **Packaging:** `jsbsim==1.3.3` is pip-only for our targets; kept out of the
   孔达锁(锁测试禁止解锁`pip:`区段),并作为运出
   hash 绑定 `envs/sixdof-requirements.txt` + `orbitopt[sixdof]` 额外.
   内置的`<engine>` / `<thruster>`定义没有使用——引擎和
   nozzle live in `engine/*.xml` referenced by `file=`.

---

# Flight Lab E4 — — — — — — — — — — —(FL-402,#,207) — XQ XQ XQ XQ XQ XQ XQ XQ XQ XQ XQ XQ XQ XQ XQ

## 已经建成的东西

`orbitopt/flightlab/sixdof/staging.py` + two child aircraft models
(`spike_booster`, `spike_upper` — the aft 25 m / forward 20 m segments of
`spike_stage`). `run_staging_scenario()` flies the stack 20 s (10 s in tests),
freezes the state (`capture_frozen_state`: ECEF position/velocity, Tec2b
quaternion, body rates, empty+tank mass), splits dry mass and remaining
propellant (`StageSplit`), spawns each child as a **fresh top-level
`FGFDMExec`** initialized purely through `ic/...` properties (+ an optional
momentum-conserving separation impulse along body +x), then steps both
instances interleaved in one process for >= 60 s. `to_flight_run()` emits a
valid three-body FlightResult v1 run (`stack` + two children parented to it,
children re-based onto the stack clock).

## 判断:工作——两个独立实例并存是可行的

2e5 N.s 冲动, dt = 0.01 s, 61 s 海岸(均测量):

* **质量关闭 :**儿童总质量=冷冻堆积质量至8.8e-4千克
  (1.4e-11相对;剩余部分为JSBSim的内部 lb四舍五入).
* **Momentum residual:** 0.16 kg·m/s against |p| = 1.17e7 (1.4e-8 relative;
  声明容忍1e-6相对+1 N. 冲动本身是
  完全按簿记方式应用(+J上方 / -J 助推器 ).
* **Handoff continuity:** position to 2.1e-7 m, attitude bit-identical
  (0 rad),速度为 < 1e-6 m/s.
* **决定主义 :** 完整样本日志比特-相同通过重跑.
* **步骤顺序 :** `booster→upper` vs `upper→booster` 样本日志位式相同
  ——实例除了过程之外,没有什么共同之处.
* **无渗漏 :** 写在一个例子上的财产在另一个例子上增加;
  节流阀/支气管的指令使另一个样本不受影响。

## 抓获和绕行(jsbsim 1.3.3 Python捆绑)

1. **No state-injection API.** `FGPropagate` exposes only `dump_state` and
   DCM 获取器;一切必须经过`ic/`属性+`run_ic()`.
2. **`ic/h-sl-ft`是地理参数**,而冻结状态是ECEF,以及
   `ic/h-agl-ell-ft`不承认. 工作周期:固定点迭代
   (纬度,h-sl)对已实现的ECEF位置(Bowring给出
   起始点;汇合到~1e-7 m在~3 `run_ic()`呼叫).
3. **ECEF 速度注射**通过`ic/vn|ve|vd-fps`(NED).
   冻结点);旋转框ECEF匹配准确( < 1e-6 m/s),因为
   `velocities/ecef-*-fps`本身是旋转框架.
4. **来自四角星的 Euler IC**(Euler是 gimbal-clocked 垂直):旋转
   `Tec2b` into NED (`Tbn = Tec2b · R_n2e`) and read the 3-2-1 angles;
   绕行至~1e-10 rad. *在此之前* `run_ic()`
   生存和被使用——这就是质量分裂的注射点。
5. **`inertia/mass-slugs` runs one frame (mdot·dt) high while the engine
   burns** (already seen in FL-401): the frozen stack mass must be taken as
   `inertia/empty-weight-lbs + Σ tank contents`,否则质量/运动
   簿记显示一个虚假的 mdot-dt( 6 kg) 不匹配 。
6. **`--` inside XML comments is fatal**:JSBSim的解析器提升
   "not well-formed (invalid token)" on `load_model` (verified) — the child
   模型评论避免了双连字符(ASCII与em-dashes的亲信是好的).

---

# Flight Lab E4 – 姿态追踪 + go/no-go(FL-403,# 208 ).

## 已经建成的东西

`orbitopt/flightlab/sixdof/tracking.py`:一个SO(3)在角加速空间的大地测量姿态-err PID与投球速率饲料前进,通过*仪器直播*厂增益`T·arm/I`(推力,Iy/Izz,CG从模型每步读回——这个增益时间表取消了燃烧质量/CG/Inertia扫射);加上3-DOF参考管道(OT `trajectory.csv`重播,ECEF-攀升高度比较,取消了地理抵消). 参考:`fl403_spike`定型,OT型机车的一跑,从尖顶级反射(同质量/Isp/ṁ/燃烧,零气);与`scripts/fl403_record_reference.py`可再生. 绘图 + 所有度量衡通过 `scripts/fl403_run_spike.py` 重生(自 F-01 ,任务 # 226 ,一个报告生成器在正版 6 - DOF后端配置`orbitopt/data/flightlab/sixdof/fl403_program_tracking.yaml` 上,而非实验切入点).

## 数字(dt = 0.01;ADR的完整表格)

Whole-burn attitude error **0.51° max** (the t = 8 s pitch-over kick),
settled (t ≥ 60 s) **0.061° max / 0.036° RMS**; gimbal saturation **0.0 %**
of burn, peak deflection 0.42° of ±5° travel (commands also within travel);
MECO deviation vs 3-DOF: **−51 m alt / −2.0 m/s / +0.016° FPA**; dt-halving
agreement 30 m / 0.85 m/s / 0.003°; no non-finite states to MECO+5 s at
Mach 18.9.

## 在接受中找到的发现: 之四=DCM方向

The first FL-403 revision reported a 24° early error and blamed the
program (near-vertical elevation slew "unachievable by TVC"). The real
cause was ours: `tracking.dcm_from_quat` returned the transpose of the
`Tec2b` that `jsbsim_driver.quat_from_dcm` encodes, and the launch IC roll
(φ = 90°) had been tuned until that wrong matrix read zero error — in
JSBSim's real attitude it was 180° off in roll, so pitch-over was steered
about the wrong body axes. Fixed: exact `dcm_from_quat`/`quat_from_dcm`
round trip, IC `(θ, ψ, φ) = (90°, 90°, −90°)` verified against `Tec2b`
itself (body x up / y south / z east; a nose-down gimbal tips the nose
east). Lesson: check attitude conventions against JSBSim's own matrices,
never only against our own helpers.

## 判决

**GO** – 关于申报的标准,证据表和E5草案范围,参见`docs/adr/flightlab-0001-jsbsim-6dof.md`.
