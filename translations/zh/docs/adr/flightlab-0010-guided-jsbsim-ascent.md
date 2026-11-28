# ADR 飞行实验室 -0010 : 引导JSBSim 上升段 中转(E6-02 ).

- **Status:** Accepted
- **Date:** 2026-11-15
- **Deciders:** E6 6-DOF Production Maturity (Epic #265), E6-02 (#274)
- **Code:** `src/orbitopt/flightlab/sixdof/guided_ascent.py`,
  `benchmarks/bench_e6_02_guided_ascent.py`
- **Evidence:** `docs/benchmarks/e6_02_guided_ascent.json`,
  `tests/test_flightlab_sixdof_guided_ascent_jsbsim.py`
- **Builds on:** ADR flightlab-0007 (production aero path), flightlab-0008
  (抽样数据控制),飞行实验室-0009(引擎对等);FL-402中转
  adapter, FL-403 PID-on-production-path, F-06 two-stage scenario

## 背景和问题

之前的每张唱片都停止了完全的制作循环:

- ADR 0007 flew the production airframe under the *native* law, open loop,
  和已公布的残留物; 它明确离开闭路生产
  路径为 E6-02 。
- FL-402 证明 JSBSim 装配适配器(运动保护财产)
  注射),但只使用~150 m/s hop,其绝对交接位置
  容忍(1e-6 m/1e-6 m/s)是有意义的.
- F-06's two-stage mission (ADR 0002/0004) closes the loop with staging —
  只在原生植物上
- FL-403关闭了生产路径的环路(PID,TVC遥测),但是.
  never reached MECO, let alone separation.

E6-02的接受是交汇点:一个**闭路引力转动TVC并启动JSBSim生产模型**,按宣布的制约逐个阶段分级,由信道(动力学/导/集成/模型)归属的原生比较.

## 决定

`e602_scenario` / `run_e602_study()`在`guided_ascent.py`:

- **车辆:**生产70 t堆栈(E6-01对齐引擎,包装)
  `spike_stage_aero`表),在 t = 95 s 中分为 4 t / 6 t 干和.
  0.4 / 0.6 推进剂-FL-402的分块,在E1-04车上.
  (不是F-06's 80 t变体).
- **指导:** F-04 / F-06 重力转弯路点程序,从
  110 s to **140 s** before hold. Under the production FCS lag the 110 s
  斜坡将 AoA 驱动至 15.21 °, 在 ± 15 ° 表信封外(列表)
  在140,任务最高峰在11.7°。 这个
  披露了生产路径的指导设计选择。
- **Law:** FL-403's production PID (`PidGains` defaults, no packaged
  由两家工厂自行制造的模型;
  在0.05 s.
- **纸 :** 28.5 °N / 80.6 °W(F-05 / F-06 世系垫). 赤道地区
  FL-403的便利框架)和*本地语言*的双柱
  轨迹方位角环流~42 °,与开普板的4.3 °相对.
  假设选择术 法律的方位角测量, 不是JSBSim -vs -
  土生土长的残留,所以研究的运行方向是系系的调节位置.
- **显示 :**在分离时指挥MECO(E6-01语义),10 kN−s
  沿身体+x、FL-402 财产注入的儿童的冲动,延迟
  103 s在`spike_upper`上点火(不携带空气——它继承)
  -1型螺旋桨发动机 晚点烧~3; (已披露)。
- **阶段限制表**(`E602_PHASE_CONSTRAINTS`) 申报时
  rationales before measurement: powered phases hold 6° peak / 3.5° RMS,
  ^ 15 ° AoA在制表机身上(25 °在无空气儿童上),其中:
  绑定结构门是`q·alpha ≤ 5e5`, 海岸阶段是
  未激活(F-06的零海岸命令),因此只能通过
  最大- Q- 得出数量; 报告海岸的姿态漂移,没有门
  (E1-02学科).
- **设置账簿检查**保持FL-402的绝对位置容忍度
  但以 ** ** 圆圈带缩放的方式重新表达速度 / 动力 / 姿态
  与交接状态 ** **(5e-6相对;1e-5 rad Euler在附近重新预测)
  投出90:FL-402的绝对值预设其跳跃状态;在~4 km/s
  separation state the property round-trip (m/s ↔ ft/s, NED ↔ ECEF,
  eular-deg)与该州同级. 本地人还分开
  关闭这些在 1e-12 —— 区别在于注射路径本身
  会计,归结于**模型**频道.
- **点火时间表容忍度**是1.5×X×每个植物的发芽 *观察*
  周期(JSBSim钩的整合步骤,土著的样本网格)
  study): 0.03 s vs 0.3 s.

## 测量(dt = 0.02 s,样本0.2 s;所有38检查合格证).

| 度(装机) | 联合统计局 | 本地语言 |
|---|---|---|
| 最大姿态错误 / RMS | 0.44° / 0.22° | 4.26° / 2.57° |
| AOA 最大 / `q·alpha` 最大 | 11.7° / 3.9e5 | 11.6° / 4.0e5 |
| 最大数量 - Q | 71 353 Pa | 71 180 Pa(0.24% 相隔) |
| TVC 差/饱和 | 0.96 / 0 s | 0.99 / 0 s |
| 装配质量/动力 | 准确 / 92 N. 往返旅行 | 准确/准确 |

上动力相:6.13 ° / 6.15 °在两株植物上的峰值——两法在*不变*无动因子动力出现时同意0.02 °.

## 剩余归属(接收的四个渠道).

- **动态** — 0.076 ° 最大 cmd — 实际偏移滞后(生产 FCS 滞后)
  parameter 5.0, i.e. tau ≈ 0.2 s, rate 0.35 rad/s, vs the native
  法学- 内部 0.1 s 驱动器; 55% 命令- 峰值差异相同
  程序: 不同的起动器和每帧惯性
  记账
- **指导** – att-error *组合*(0.44 °s 4.26 °)为指南.
  channel itself: PID-on-live-instrumentation vs 0.05 s tracker ticks. Both
  坐在他们自己的6 °的带内, 土方匹配FL-605
  世系测量(4.77 °). 跨法协议带(5 ° 峰值) /
  3° RMS) are declared from those lineage numbers, not from a shared truth.
- **融合** – pad 推力 3e-6 rel,最大-0.24% rel,分离状态
  ~ 0.17% rel; dt-halving 转向分离高度由 5.8e-4(JSBSim)和
  7e-7(原生):这两种植物在运出的阶梯上都呈网状汇合.
- **模型**——上面的房产注入回转波段;海岸
  姿态飘移(92 ° 两种植物,0.29 °)是两个没有作用的体.
  在同一中心重力下,不是法律上的区别。

## 后果

- 生产道路在自己封闭的状态下飞翔、阶段和重新燃起
  CI; `tests/test_flightlab_sixdof_guided_ascent_jsbsim.py` 警卫
  承诺的学习 加上一个活的短期飞行。
- FL-402的绝对交接力范围为低速交接力;
  高速中转必须使用缩放波段(`guided_ascent.py`输出)
  常数。
- 海岸态度没有因设计而改变;今后的任何恢复模式
  研究需要自己的海岸法和自己的制约表。
- F-04追踪器的赤道平方位角漂移有记录; a
  未来想要赤道垫子的任务应该固定测量表,而不是移动
  任务

## 审议的其他办法

- **JSBSim植物的原生法**-拒绝:接受要求
  TVC *and* the production path; FL-403 already shows the production law
  将MECO高度控制在0.4°上.
- **F-06's own 80 t vehicle** — 被否决:E6分级生产成熟度.
  vehicle (E1-04 70 t), and F-06's own record stays native-only.
- **110 s 程序 + 宽 AoA 带** — 被否决: ± 15 °的极限是
  表格信封(物理),不是调音节,而是程序移动。
- **Keeping the equatorial pad and widening native bands to 45°** —
  拒绝:将追踪器的仪表动脉分级为
  植物差异 毒害接受要求的比较
