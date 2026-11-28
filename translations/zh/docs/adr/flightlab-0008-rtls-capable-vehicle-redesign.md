# ADR飞行板 -0008:RTLS-能力车辆重新设计和计算分离状态.

- **Status:** Accepted
- **Date:** 2026-11-16
- **Deciders:** E5 Physically Continuous Mission (Epic #264), E5-05a (#285);
  amended by E5-05b (#286), E5-05d (#288) and E5-05c (#287, F10 resolution)
- **Code:** `src/orbitopt/data/flightlab/vehicles/synthetic_two_stage_rtls.yaml`
  (新目录车辆),`src/orbitopt/data/flightlab/scenarios/leo_two_stage_rtls.yaml`
  (flight target), `src/orbitopt/data/mission_two_stage_leo_rtls.yaml`
  (任务文件,上升段模式`run_dir`),,
  `tests/data/mission/reference/e5-05a-rtls-mission/`(重播上升段+PROVENANCE.json) ;
  E5- 05 c:`src/orbitopt/flightlab/recovery/boostback.py`(闭路返回).
  选项),`.../recovery/landing.py`(登陆预报包括高峰负载),
  `scripts/e505c_recovery_search.py`, `scripts/e505c_recovery_montecarlo.py`
- **Evidence:** `tests/test_mission_e5_05a_rtls.py`,
  `tests/test_flightlab_recovery_closed_loop_return.py`,
  `tests/test_flightlab_recovery_landing_forecast.py`; 竞选日志
  `context-forge/implementation/result.json`; 上下文数据库
  `e5-05-rtls-infeasibility-evidence`(# 272,前身).

## 背景和问题

E5-05(# 272)试图将双级LEO+RTLS的参考情景端对端记录在绑定的编目车`synthetic_two_stage`上,并由于结构原因失败,而非调子(ContextDB `e5-05-rtls-infeasibility-evidence`):

- `OpenTsiolkovsky`燃烧阶段1耗尽,所以分离状态是
  在每次运行时的干重(30 000千克)——无论是在探测器中还是在探测器中,都是如此.
  已封存的 Gate-B 参考运行 。
- *实际数*分离状态的恢复链计划(ADR)
  -0004的独立规则);没有规划室
  mechanism, and none should be invented — an RTLS booster is recovered by
  推进剂的发动机实际上留在了MECO的罐内。
- RTLS MECO 需要数十吨级-1
  储备;申报的4 200千克储备为~10×太小,上层
  级全松(零有效载荷时为5 488千克)仅购买~400 m/s.
  -1比值.
- FL-305核实了*教科书* 65 t分离状态的着陆情况;
  E2-04骑着同样的说明上升段. E5的接受是:
  链在**计算** OpenTsiolkovsky分离状态上运行.

## 决定

**A new catalog vehicle, `synthetic_two_stage_rtls`, sized around an
explicit stage-1 propellant reserve, flies a computed OT ascent whose
separation state carries the reserve the recovery chain plans against.**
The sealed `synthetic_two_stage` (and every sealed reference built on it,
E2-04 included) is untouched; E2-04 remains the fast integration test.

### 车辆参数及其来源

下面的每个数字都是工程级数字,有一个公共线锚,所以选择任务结束;尺寸计算在本节末尾的竞选表.

| 参数 | 数值 | 线条/衍生 |
|---|---|---|
| 1阶段湿/干 | 620 t / 30 t | Falcon 9 - 级 助推器 大众乐队(549 t / 22 t发布;+13%湿度用于预备役任务) |
| Stage 1 thrust | 11 MN (single unit) | ~1.45x Merlin-class cluster vacuum thrust (7.6 MN); keeps pad TWR 1.81 with the reserve tankage |
| 1 Isp阶段(空缺) | 310 s | kerolox gas-generator class (Merlin 1D 311 s — the sealed catalogue value) |
| 烧/ 分离 | 0-116 s / sep 123 s | ~ 2 / 3级-1推进剂故意留在MECO的坦克中; RTLS 参考剖面切除低速减速,以保持回升三角洲V和入口加热 |
| **Computed separation reserve** | **171 621 kg** (SEP 201 621 kg = 30 t dry + reserve; measured 32.8 km / 1256 m/s / 56.4 km downrange on the OT run) | structural consequence of the burn table, read off the OT run, not declared: 620 t - 116 s x 11 MN/(310 s x g0), matches the run within sample spacing |
| 2阶段湿/干 | 190 t / **12 t** | 上级大小,以适应*下级* 预备队(SEP 201.6 t级-1 + 190 t堆:舞台-2 装载量在预备队边界上大小); 干重是捆绑的杠杆,见sizing段——93.5%推进剂分量是一个复合主结构假设(C/LiAl油箱,栅格鳍/腿仅位于1级),在可重复使用的上级结构分量具有攻击性但有记录的末端. |
| 2级引擎 | 5.4 MN -> 3.6 MN two-point throttle schedule, Isp_vac **350 s (catalogue, unchanged)** | 最高的初始推力是让拉力飞翔(发现F3); 密封的350 s被保留着——Isp扫描证明Isp是NOT插入杠杆(调查F2). |
| 2 燃烧 | 点火 7 秒后,断线战役调整(边界 140 -172 秒当地) | 如此大小的 ~ 155 -170 → s 的 173 t 负载在阁楼内; CMA-ES战役中准确的截断+程序,烤入车辆yaml和PROVENANCE.json. |
| 态度程序 | vertical ~15 s, pull-over to 76-58 deg through max-Q, 24-46 deg pull-up hold, terminal flatten to <10 deg | hand-lofted family then CMA-ES refined (8-point elevation spline, azimuth 36-44 deg); see campaign table |

Sizing: one burn to a 400 x 400 km orbit requires **burnout at >= ~380 km
altitude with near-zero flight-path angle at ~7.67 km/s** — perigees above
the burnout height are impossible on a suborbital cutoff (finding F4), so
the burnout height is the hard wall. From the computed separation state
(1125 m/s at 31 km) the ideal delta-v is ~8.0 km/s and a lofted profile
costs ~1.1 km/s gravity + ~0.3 km/s drag: the mass budget
`Isp*g0*ln(m0/dry) = 3433*ln(185/12) = 9.39 km/s` covers 8.0 + 1.4 km/s of
losses; the inherited 17 t dry stage (8.19 km/s) leaves 0.2 km/s and was
proven infeasible by two full CMA-ES campaigns (table). Dry mass, not Isp,
is the redesign lever.

### 为什么烧掉一个 后端能表达什么

OpenTsiolkovsky's config model gives each stage one window
(`burn_start_s`/`burn_end_s`) and the thrust `Curve` is **piecewise linear,
clamped** — a near-zero throttle valley cannot coast, so a perigee-kick +
apogee-kick upper stage is inexpressible in this backend (verified empirically:
a valley table interpolates and burns ~2.5x the coast's nominal flow; a two-burn
restart key does not exist in the config schema). The design therefore commits
to a single continuous insertion burn with a lofted profile.

### Campaign (design provenance, all flights `run_ascent` OpenTsiolkovsky 1.1.0)

参数化家庭(CMA-ES,pygmo,OT- in-loop,`/tmp` 竞选日志存档于`result.json`):

| 运动 | 列弗尔 | 结果 |
|---|---|---|
| c5-c10 | -1在目录350 s,线性投球斜拉桥的大小扫描 | rp卡在-3.2.. -3.6公里,而ra dead-on 400:一个线性斜拉桥的断面坐落在圆弧的平面部分——近拱点被钉住烧尽高度,一个低的推力斜拉桥从不升过~135公里. |
| c11/c12 | 燃烧 _ 结尾 118 s, 预备目标 | INFEASIBLE(+316 / -381 km rp 残):家族饱和,罚分重量被证明很强(全带=1目标跨度). |
| 楼阁 1 -4 | hand-lofted arcs at 3.6 MN | 崩溃/倾覆: 在 3.6 MN, 牵引力超过可用的 推力 - 最大- Q 下的重量边距 |
| 楼阁 5 -17 | hand-lofted arcs at 5.4/3.6 MN schedule, 350 s | rp reaches -4.0 km (loft5, first periapsis closure of the campaign) but ra then saturates >= 1.2 Mm: rp and ra cannot both close at 17 t dry — the burnout-height wall (F4) |
| 顿 1 | 9/6 MN table (TWR 4.9) | worse: 3.77 km/s net — high TWR climbs faster *through* max-Q, drag loss +~800 m/s: thrust beyond the loft's requirement is a liability |
| 18 /hi 1 / 山谷探测器 | el-42 hold; 9/6 MN; throttle valley | 能量过量(9.14公里/秒,192公里);拖-死;海岸无法表达——见F3/F5. |
| 干-17运动×2(11 - dim CMA-ES:姿态+截断+2 - 点推力,武力约束). | 除了干燥质量 | PLATEAU at rp = -2. -4 km(fitness 136 -233),界角溶液:F4确认墙为结构质. |
| 干 -12 | 同为11 - dim家族,舞台-2干17 - > 12 t | PLATEAU rp = -0.2. -2 km:干燥的-12 单从33 -42 公里(F4  again再次)的距离,无法抬高过去~200的燃烧高度. |
| s 1 - loft + 铅角(loft 4 / 5 ). | lofted stage-1 program (MECO at el 44 deg, separation 42-48 km), stage-2 LEAD-angle program (commanded elevation 55-80 deg: the point-mass flight path lags attitude by the alpha needed to turn at low speed, F6) | sep 48 km / 1.25 km/s: periapsis improves to -0.3 km but the insertion energy at cutoff stays grossly over — still short |
| 长期燃烧的分离制度(Loft 6 -8) | stage-1 `burn_end_s` 124-150 s as a design variable, terminal stage-1 pitch-down, stage-2 dry 12 t, m0 70-130 t clamped to the computed SEP mass, two-point throttle re-read as OT's linear ramp (F7), recovery-reserve penalty | moves the separation state into the RTLS-plausible regime (78-115 km, 2.0-2.9 km/s, ~106-157 t SEP) — exactly the ~180 t-class separation mass the task brief predicts once the burn is shortened for reserve; `fly_recovery` reserve check PASSES on the computed state (13.6 t at touchdown, 136.7 t SEP) |
| objective fix + loft9-14 | 校正健身缩放(F8),从楼阁8开始温暖;13 - 然后是8 - 然后是10 -/11 - 随着墙面的出现而凹陷的家庭; 投球下结点移动到耗尽前,懒惰的终端高度,终端投球节点 | PLATEAU rp -0.7..-1.3 km / ra >= 1.4 Mm across every family and dimension — F9: the band is unreachable in this backend, and the campaigns confirm it, they do not fail to try |
| lazy1 (BAKED REFERENCE) | conserving turn authority: max elevation 42 deg, gentle terminal flatten, T2 2.8 MN lazy burn | best closure of the whole campaign: sep 89.9 km / 2.277 km/s / 117.6 km / SEP 136.7 t, SECO rp -0.74 km / ra 5696 km / inc 48.0 deg — the baked vehicle program |
| 走廊网格(b 108 / 116 / 124 × 5 计划食谱+猜想扫荡). | `fly_recovery` on the computed states, R-01 knobs, entry-g 12, horizon 1200, entry duration 35-60 s | - CORRIDOR_MISSED-F10; 距离交接时间最近:1.5 -2.4 公里内8 公里内交接速度~600 m/s对350 m/s天花板; 每个计算状态的后备门PASSES |
| E5-05c probe (E5-05d state, 135.4 t SEP) | 定时 7 - 参数脚本; 双燃条目 (`entry_burns=2`); 节奏条目 | the corridor GEOMETRY already closes (247 m/s, 4.8 km) — the blockers are propellant (11.8 t at h0 vs 18.9 t needed) and entry load (12.7 g vs 8 g); a second braking burn spends exactly the propellant that is short, so `entry_burns=2` does not help (kept as an option) |
| E5- 05 c 步数规模审计 | same +30 t script re-flown at dt 2 / 0.5 / 0.25 / 0.1 s | 交接805 / 386 / 331 / 276 m/s:FL-301定步判决在规划员默认时没有步态结合(每步抽取一次触发/截断);搜索运行在0.25 s,判决在0.1 s. |
| E5- 05 c 预备尺寸(代号:+dm在E5- 05 d 分离状态) | 闭路着陆返回(IIP增速,减速进入,闭路着陆预报),DE+n=64 蒙特卡洛 | +30 t: 0/64 (touchdown p05 5.8 t; spread 0.99-correlated with the Isp draw, an inherited 5 % band); +48 t: 35-37/64 corridor misses until the IIP burn's 60 s cap was replaced by a propellant floor; then +48 t 64/64, +50 t 63/64 (one engine-on handoff, cut at 2.9 km) |
| E5-05c stage-1 resize | thrust + mass flow x k at the 134 s cutoff, k 1.14-1.22 (live OT) | +48 t as dead load sinks stage-1 dv ~340 m/s (rp -546 km); k 1.165-1.185 closes the band, k = 1.175 centres it: rp 272.6 / ra 285.1 km, SEP 183.4 t at 2129 m/s / 93.4 km (E5-05d: 2135 m/s / 93.2 km) |
| E5-05c computed-state Monte Carlo (n=64, seed 1, Isp 3 %) | 固定的食谱, 固定的脚本, 没有 dm | landing law at 4.0 g: 59/64 (5 hot-engine samples at 5.1 g peak load); at 3.6 g: **64/64** (Wilson 95 % lower bound 0.943), touchdown propellant min 12.0 / p05 12.6 / median 14.2 t vs the 10 t gate |

调查结果F1-F10(驾驶决定):

- **F1 — grading discipline.** Reading the orbit at the track's *last*
  近地点 - sskimming拖曳数小时后,样本等级为SECO;全部
  阁楼系列不得不在`seco`赛事中重新叙级. 现在目录
  记录SECO在事件样本中表示.
- **F2 — Isp is not the lever (measured).** At an unchanged thrust-time
  计划一个更高级的Isp发动机 消耗*减*推进剂并切断
  *slower*: identical loft16 shape, 146 s cutoff — Isp 350 s burned 159.6 t
  (8.4 t 预备队)在7.69 km/s; Isp 380 s 燃烧 147.1 t (20.9 t 预备队)
  6.84 km/s. OT烧伤是推力时间有限,而不是Isp/mass-ratio有限.
  目录350留下; 能量拨号是截断时间和终端
  推力,近拱点拨号为截号航迹角(=燃烧高度,F4).
- **F3 — thrust-for-loft.** The loft is only flyable where T/W during
  提拔保持1.35; 5.4 - > 3.6 MN 时刻表最便宜.
  引擎满足它,同时保持最大Q在舞台1.
- **F4 – 燃烧-高墙.** 轨道几何:近地点 高度可以
  从未超过亚轨道燃烧的截断高度(在截断时为~0),
  380公里需要燃烧 ~380公里;每17 t-干重配置封顶
  120 -190公里. 这是干质量改变买。
- **F5 — coast is inexpressible.** A throttle valley between 5.4 and zero
  将片面线性并仍然燃烧的推进剂插入(测量~2.5 x)
  名义流量; 在配置图中不存在重启密钥 。 单次燃烧是
  a backend constraint, not a preference. *E5-05b (#286): the orbitopt adapter
  现在表示海岸为真正的零-推力段(a `[tc,0]..[tc+coast,0]`).
  高原,零流)由`restart`车块合成,于是海岸和
  重新点火在适配器层是可表达的——尽管F9显示它们没有
  by themselves close the band.*
- **F6 — attitude is commanded, flight path lags.** OT integrates the
  精确的指挥姿态;点质量飞行路径在
  angle-of-attack deficit `sin(alpha) ~ m v gamma_dot / T`. At 2 km/s and
  150 t a 0.3 deg/s 路径转弯需要α~40 deg,所以程序是
  wants the *path* at 30 deg must *command* 60-80 deg mid-burn. Every
  早期运动中的浅线方案
  this reason; the sealed 400 km program gets away with el 26-30 only
  因为它的近轨道分离使得路径转速微小.
- **F7 — thrust tables are ramps, and cutoff is depletion.** A
  `thrust_vac_n`表斜坡线性地介于 t=0的节点之间(一个"持有")
  然后是油门上的油门,还有引擎
  如果推进剂运行,则在`burn_end_s`定时器前在耗尽时死亡
  首先:截断时间由-2 mi 0 控制(声明到
  SEP质量计算,不是按定时器计算.
- **F8 — campaign objective scaling.** The loft4-8 campaigns shared a
  健身能力将一公里值的rp/ra 乘以1000,
  在轨道倾角术语下加固轨道残留~10 ^ 4 x:
  运动使轨道倾角汇合,盲目徘徊轨道.
  修正的缩放(loft 9)是显示rp/ra梯度的原因;
  所有早期高原结论都只因为*手动*
  探测器(而不是优化器)产生了这些数字。
- **F9 – 在这个后端插入波段是无法到达的.** 共计
  OT燃烧的转机是`∫ T/(m v) dt ~ 35 deg`;关闭
  保留状态的乐队需要 ~ 45 dgg 摇摆
  (F5:没有海岸,没有)
  重新激活)不能将推进剂花掉第二次燃烧。 六次竞选
   和两个CMA-ES 维度高原在rp=-0.7.-1.3 km;
   近地点楼是结构,不是调音误. 关闭乐队需要一个
   后端特性(OT配置图中的启动/海岸能力),不是
   这辆车的重新设计。
  - **F9 resolution (E5-05b, task #286) — the capability landed; the wall was
    the vehicle, not the backend.** E5-05b delivered exactly the restart/coast
    capability F5/F9 called for: an optional vehicle `restart` block the OT
    adapter folds into a coast-gap thrust curve + restart attitude leg, with the
    coast-start burnout suppressed so a multi-burn stage maps to a single final
    `seco` (see the adapter's `_apply_restart` / `_mapped_events`). Re-flying the
    E5-05a reference with a coast gap (main-burn cut at 224 s, 120 s ballistic
    coast, 1 MN re-ignition) and sweeping coast/ignition/timing does NOT close
    the band: the best computed insertion is rp ~= -0.4 km / ra ~= 5.1 Mm
    (single-burn floor rp ~= -0.7 km). Coast + re-ignition cannot manufacture
    turning authority the stage never had, nor a shallow enough transfer to
    circularise from; the perigee floor persists. **Conclusion: F9 was
    misattributed.** The band is not a backend-capability gap — the restart/coast
    feature is now present and still misses. Closing the 250 km band from a
    reserve-carrying separation state needs a vehicle change (more upper-stage
    delta-v / higher thrust-to-weight for turn, or a higher band such as the
    400 km option the same capability now enables), tracked as the real follow-up
     for #272. The single-burn ascent stays the shipped reference because it is
     the best computed orbit; the reference re-bake only records this finding.
   - **F9 closure (E5-05d, task #288) — the vehicle redesign landed and the
     band closed.** The shipped catalog vehicle is redesigned per the F9
     follow-up: stage 2 becomes a 128 t wet / 8.5 t dry restartable upper stage
      at Isp 380 s with a 14 MN boost leg shortened to 17 s, a declared main-burn
      cutoff at 224.5 s, a 155 s ballistic coast and a 9.65 MN / +1.7 deg
      re-ignition circularisation burn (single final `seco` at 396.2 s, burnout
      on depletion). The computed ascent now reaches
      **rp 281.8 km / ra 296.1 km at SECO** — the 250 km insertion band closes on
     the same computed run that carries the stage-1 reserve, and
     `test_seco_insertion_band` is promoted from strict-xfail to a hard
     assertion. The redesign couples back into stage 1: the +43 t stage-2 stack
     at liftoff sank MECO by ~330 m/s and drained the *landing* propellant
     reserve to 3 t (below the 10 t gate). The stage-1 engine therefore moves to
     Isp 330 s at unchanged mass flow (11 -> 11.702 MN, 3615 kg/s, cutoff still
     134 s — the E5-05a declared-cutoff reserve mechanism is untouched: 484.4 t
     burned, 135.4 t computed SEP mass = 105.4 t reserve), restoring MECO energy
     (2.4 km/s) and the touchdown reserve (13.2 t, matching E5-05a). Campaign:
     19-dim differential-evolution + hand-lattice searches on the restart-schedule
      knobs (boost-leg length, coast start/duration, re-ignition elevation and
      burn throttle), corner closed at rp 235.6 / ra 257.1 before the stage-1
      re-engineering shifted the coast geometry; the schedule was then re-tuned
      around a re-ignition-elevation ladder to the shipped rp 281.8 / ra 296.1.
- **F10 — the corridor referee does not accept computed MECOs.** With the
  回答保留问题(大门通过每个计算状态)
  landing gate's *only* blocker is the FL-301 corridor: engines-on at 3 km
  under 350 m/s within 8 km of vertical. The FL-302 seven-parameter script
  reaches 1.5-2.4 km miss (inside the radius) from every state probed
  (78-118 km downrange, 1.6-2.9 km/s), but never under 350 m/s — the single
  输入燃烧无法制动 1.6 km/s+ 级以制动内部的地下速度
  q/g框,低能(b 108 / 116)和高能州贸易失利
  而不是进入可行性。 门一直只有绿色
  the FL-302 textbook state (250 m/s, 8 km downrange): a guidance-model
  限制走廊的设计假设编码,而不是推进剂
  不足——正是任务简报所预言的缺口(“经过核实的”
  the textbook state, not a computed OT one").
  - **F10 resolution (E5-05c, task #287) — closed-loop return guidance and a
    sized stage-1 reserve; the landing closes on the computed state.** The
    diagnosis moved twice. On the E5-05d state (135.4 t SEP) the timed script
    already met the corridor *geometry* (247 m/s, 4.8 km); what failed was
    propellant (11.8 t left at h0 against 18.9 t landing need + reserve) and
    the entry load (a full-throttle three-engine burn exceeds 8 g by itself
    below ~69 t). And every earlier corridor verdict was read off a
    propagator that is not step-converged at the planner's 2 s / 0.5 s steps
    (one script: 805 m/s at 2 s, 276 m/s at 0.1 s). A multi-burn entry was
    tried and does not help — a second braking burn spends the propellant
    that is short — so the resolution is closed-loop guidance plus reserve:
    - *planner options* (all default-off, bit-for-bit): `boostback_cutoff=
      "iip"` (the boostback ends on its vacuum-IIP aim or at a dry + reserve
      propellant floor, and steers on the IIP — a timed burn moved the impact
      ~10 km per 0.5 s); `entry_cutoff="speed"` + `entry_throttle` (the entry
      burn ends at a cutoff speed, throttled; the handoff is only valid once
      it has cut off — engine-on crossings are rejected — with 1.5 km of
      planned clearance above h0); `landing_need_model="closed_loop"` (the
      handoff is graded by a forecast of the FL-305 landing it will fly,
      touchdown state and *peak proper load* included, instead of the
      350 m/s cap); `landing_reserve="target"` (one graded reserve, the
      scenario's 10 t, for planner and landing check) with a 2.5 t planning
      margin; `search` (offline global search hook, `"keep"` pins a script);
      `verdict_physics` (density/wind on the corridor leg in Monte Carlo);
    - *reserve sizing*: an n=64 Monte Carlo of the fixed script on the
      separation state + dm put the reserve at **+48 t** (Isp band 3 %,
      stated for Isp itself — it used to inherit the stage's 5 %, which
      alone drove the touchdown spread);
    - *vehicle*: stage 1 scaled x1.175 in thrust and mass flow at the
      unchanged 134 s cutoff (11.70 -> 13.75 MN, 620 -> 752.77 t loaded,
      569.2 t burned) carries the +48 t to separation (183.4 t SEP, 153.4 t
      reserve) at the E5-05d liftoff T/W and MECO state, so F9 stays closed
      (rp 272.6 / ra 285.1 km); the fixture is re-baked;
    - *landing law*: peak load = law cap + ~1.1 g of drag at ignition; the
      4.0 g cap flew 5.1 g (> the 5 g landing-max-g limit) on hot-engine
      samples, 3.6 g holds it at <= 4.7 g;
    - *result*: the pinned recipe lands from the live computed separation
      state (`test_recovery_landing_gate_on_the_computed_state`, hard
      assertion), and 64/64 dispersed flights of the same fixed script land
      (touchdown propellant min 12.0 / p05 12.6 / median 14.2 t vs 10 t;
      cutoff >= 5.3 km; corridor miss p95 532 m; entry load <= 10.2 g;
      landing load <= 4.7 g). DESIGN ASSUMPTIONS carried by the recipe:
      entry 12 g / 250 kPa (uncrewed metallic booster), landing law 3.6 g,
      FL-301 steps 0.25 s (search) / 0.1 s (verdict). The +50 t population's
      single failure (an engine-on handoff, cutoff 2.9 km) marks where the
      cutoff clearance runs out under dispersion.

### 计算状态上的回收链

The recovery pipeline (`fly_recovery`) runs from the OT separation state at
183.4 t (E5-05c). Pinned recipe (`plan_options` in the mission document),
each item with a reason:

- `initial_guess`(8值:延迟,IIP瞄准km,助推节流,投球,
  yaw 修剪、 输入触发、 输入截断速度、 输入节流)
  `search: keep`:剧本的飞行方式与蒙特卡罗经核实的完全相同.
  (由`scripts/e505c_recovery_search.py`用这些选项找到);
- `boostback_cutoff: iip`, `entry_cutoff: speed`, `entry_throttle: true`,
  `landing_need_model: closed_loop`:闭路返回(F10)
  这些是吸收推力/Isp/密度/分离的
  散开;
- `corridor_aim_margin_m: 7000`:计划1公里的射程在8公里裁判内;
- `landing_reserve: target` + `landing_reserve_margin_kg: 2500` :分级
  10 t reserve, plus the Monte Carlo nominal-to-p01 touchdown spread;
- `entry_q_max_pa: 250_000`,`entry_g_max: 12`:未碎金属 助推器
  条目限制( DESIGN ASSUMPTIONS; 仅全喉管组超过)
  8 g at RTLS 进入群;
- `dt_opt: 0.25`,`dt_final: 0.1`:继位的FL-301判决;
- `landing.max_thrust_accel_g: 3.6`与申报的TVC助动器组.

The flown chain lands with 14.2 t (Monte Carlo median) of the 153.4 t
separation reserve, above the 10 t gate on every dispersed sample.

### 任务文件:`run_dir`重播,不重模拟.

`mission_two_stage_leo_rtls.yaml` flies `ascent.mode: run_dir` against the
committed run-directory fixture
`tests/data/mission/reference/e5-05a-rtls-mission/ascent` (generated by
`run_ascent` on exactly the shipped vehicle and target, PROVENANCE.json
beside it), and the recovery leg `drive_from_separation: true` plans from
that same computed state. The E5-01 tripwire forbids *preset* SECO/injection
states in the scenario document; a replay of a genuine OT flight is the
sanctioned pattern (gateb precedent). Re-simulating live in both E5-03
environments stays on #272, which this task unblocks.

**Mission chain (E5-05c finding, left to #272).** With F9 and F10 closed
on the computed state, the packaged mission *chain* still fails gates that
are not F10 — present on the pre-E5-05c fixture too, masked until now by
the chain test's `xfail(F10)` (now `strict=True` on the true reasons):
(1) the E5-02 shared handoff hands the booster the t = 140 s GCRS sample
while the pipeline plans from the t = 141 s separation state, and the
booster handoff continuity checks fail — the fixed script misses 7.4 km
from it; (2) the document's expected parking orbit (250 x 250 km at
51.6 deg, +/-20 km / 0.2 deg) is not the band the scenario and vehicle fly
(rp 272.6 / ra 285.1 km at 51.0 deg), and the chain reads the parking
elements before the stage-2 re-ignition; (3) the OT ascent's own
max-accel verification. These belong to the reference-scenario recording
(#272).

## 后果

- `synthetic_two_stage_rtls`是所有RTLS任务的目录载体.
  必须连锁插入; `synthetic_two_stage` 保持密封
  FL-3xx/Gate-B results.
- 回收链的输入现在是计算分离状态
  E5任务;FL-305的教科书状态验证仍为
  廉价的回归。
- Unblocks #272: both walls are closed on the computed state — F9 by the
  E5- 05 d 上期重新设计,F10 由E5- 05 c 闭路返回和
  -1调整大小(插入带状,降落绿色,64/64蒙特卡洛). 什麽?
  #272 inherits is the mission-chain list above (E5-02 booster handoff
  状态,预期轨道/停放-元素读出,OT最大角,而不是a
  物理墙.
- FL-301走廊判决在规划者默认情况下对阶梯大小敏感
  (2 s / 0.5 s). E5- 05 c 配方标注 0.25 s / 0.1 s; 默认路径
  参考文献(FL-305,E2-04)保留其记录步骤. 一个事件位置
  在宣传器中触发/关闭会消除敏感性(后续)。
- 披露边界:储备是结构性的(引擎周期簿记)
  在宣布的截止时,不是模拟中止;输入加热带是
  未编织的级设计假设,而非机组级; 原始 OT配置
  没有原生的海岸/启动键,所以轨道适配器合成海岸
  作为`restart`车挡(E5- 05 b)的零-推力级弯路段.
  - 足够机制使用,但真正的指导法多点燃烧的条目仍然存在
  想要本地的多烧后端(JSBSim,两者都是多烧).

## 审议的其他办法

- *Re-size the sealed `synthetic_two_stage`* :被拒绝 – 密封摘要
  (Gate-B,FL-301.;305,E2-04;与该车并列的重新设计车).
  既定模式。
- *Hold the 400 x 400 km reference orbit*: rejected after five OT-in-the-loop
  CMA-ES运动——燃烧-高墙(F4)是结构化的.
  留下RTLS储备的单燃插件
  400 km-band solution found so far sits at rp = -0.2..-4 km. The reference
   任务重设在**250 x 250 km 51.6 deg**(仍为LEO;任务).
   acceptance is "insertion still reaches LEO"), and the sealed 400 km E2-04
   regression is untouched. E5-05b added the restart/coast backend capability
   that can now *express* a circularisation burn (and a 400 km band), but F9's
   分辨率显示此飞行器仍不能使用 *到达* 任一波段; 到达
   400公里现在等待车辆重新设计,而不是后端功能.
- *宣布在旧车上设名义4.2 t储备*:被拒绝——这是
  # 272的核心负结果(~10×太小),与大小重排
  以上数字。
- *复原段的教科书分离状态*:被拒绝——违反
  任务接受标准(完全计算)和ADR飞行实验室-0004 's
  独立规则。
- *三级车辆(2级作为踢踏级)* :被拒绝 – 增加中程
  如果任务计划以及E5-03环境都需模型,
  而RTLS在舞台上的物理1没有变化.
