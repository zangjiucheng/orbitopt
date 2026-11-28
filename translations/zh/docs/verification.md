# 验证和基准

报告记录检查、容忍度、强制模型和数字设置。 通行证适用于这些假设;通行证不能确立飞行安全。 Mission Control显示**VERIFIED**,**VERIFY FAILED**或**未检查**. 关于所审查的E1 – E4整合及其修补,见[2026-11-15 acceptance report](acceptance-e1-e4.md).

## 报告的内容

The `verification-report-1.0` document contains residuals, omitted effects,
integrator settings, provenance and a canonical SHA-256 digest. Reports are
written beside mission scenes and by Flight Lab, `orbitopt launch` and the
reference-scenario chain. Inspect both the overall result and its evidence.

## 三种证据

检查类别说明其目的:

| 类别 | 目的 |
|---|---|
| `numerical_consistency` | 在模型范围内的趋同或保护. |
| `model_agreement` | 同意另一种预测,模型或约束. |
| `historical_validation` | 与公布的现实世界数据达成一致。 |

不检查的类别是**未检查**.

## Five evidence types (task #263)

证据类型表示检查所使用的参考:

| 类型 | 参考 | 通行证的范围 |
|---|---|---|
| `analytical` | 同/简化的模型的封闭形式. | 同意这一解决办法。 |
| `independent_model` | 单独配制的宣传机或模型. | 在检查的阶段交叉模型一致性。 |
| `historical_telemetry` | 已公布飞行观察,而非模型输入. | 同意这些意见。 |
| `design_input` | 目标轨道,目标点或申报信封. | 解决办法满足了设计方面的限制。 |
| `numerical_consistency` | 在独立/紧数下是相同的模型. | 数字错误或保存检查合格。 |

`status.check_evidence`地图检查类型;`status.evidence`卷起来. `status.independent_validation`只包括`independent_model`和`historical_telemetry`. 它的`not_checked`结果意味着没有包含独立证据,即使报告的总结果是绿色的. 独立模型一致性仍未确立现实世界的有效性.

### 设计投入比较及其必须贴标签的原因

Hitting the orbit or corridor the optimizer targeted is `design_input`, not
historical validation. Producers must supply that evidence override rather
than inheriting the `model_agreement` default of `independent_model`.
Missing references or skipped checks must remain visibly unassessed.

### 三个参考基准

| 基准 | 证据 |
|---|---|
| 开普勒双体 | `analytical`;能量漂移为`numerical_consistency`. |
| J2 世俗漂移 | `analytical`. |
| Starship Flight 15 booster catch | `historical_telemetry`对照其公布的时间表. |

渔获量有助于进行两项独立验证检查;其他五项基准检查是分析性的或数字性的。 这是范围大的证据,而不是每个任务阶段都得到验证的说法。

## Full mission verification gate (E2-03, task #256)

`inputs.full_mission`分三个阶段:

| 阶段 | 检查/适用性 |
|---|---|
| `orbit_injection` | 上升段判决,站点命令,中转,质量和预期轨道;始终适用. |
| `parking_propagation` | M-05 轨道传播;仅在`expected.propagation_result: not_checked`被宣布时(在`full_mission`被宣布时被废除)才超出范围. |
| `recovery_touchdown` | 回收/处理/触摸限制;在`recovery:`申报时适用(在`full_mission`申报时适用)。 |

Among applicable phases, **failed** takes precedence over **not_checked**,
which takes precedence over **passed**. An applicable phase with no checks
is `not_checked`. On a run that does not declare `scenario.full_mission`, a
declared out-of-scope phase does not block the verdict. The chain's
diagnostic `status` and the pipeline's `full_mission_verdict` are separate
fields; inspect the phase document when a run is only partially successful.

### Verification coverage gate (E8-03, task #283)

一个情景可能宣布`scenario.full_mission: true`:运行的*这是*是完整的物理任务,所以上面的所有三个阶段都变成了**所需物理阶段**,不能通过声明而脱离范围:

- `expected.propagation_result: not_checked`是**拒绝**——阶段.
  继续适用,没有证据记录,并读作`not_checked`;
- 失踪的`recovery:`区块同样离开`recovery_touchdown`
  `not_checked`;
- 要求的阶段上限
  `not_checked`,所以范围外的路线无法再产生
  从未向停放轨道宣传的特派团的总体通行证
  或者从来没有带助推器回家。

新生成的大门文件涵盖所需阶段——`inputs.full_mission.coverage`列出每阶段状态(`checked`,`failed`计数如检查),`not_checked`,`declared_out_of_scope`,`not_declared`)和`summary`,如`2/3`,管道核查舞台记录(`full_mission_coverage`),并在注释中呼应——以及`inputs.full_mission.qualification`区块. 报告的总体描述可能要求**独立核查** 只有当`qualification.independently_verified`是`true`时:跑队被宣布为`full_mission`,跑门判决是`passed`,全覆盖`3/3`,跑队的独立证据滚滚(`independent_validation`)通过. 部分运行(例如消耗性特派团)无论适用阶段如何绿色,都没有任何条件——其涵盖范围摘要准确地说明了运行达到的程度。

The chain preserves `ascent.physics_verified` as independent-model evidence:
`true` passes the check; `false` fails orbit injection and the full-mission
gate. `null` remains explicitly unassessed and contributes no independent
validation. A green design-constraint rollup with a null flag does not prove
independent verification. See the [acceptance report](acceptance-e1-e4.md).

### Full physical mission gate (E5-04, task #271)

这种范围外的豁免恰恰是*计算*任务绝不能躲在后面(E2-04宣布停放轨道传播不在范围内,仍然通过). `inputs.full_mission.physical`重级与**five required physical checks**相同的链式检查——无适用性逃生舱:

| 需要检查 | 证据(链条的现有检查) |
|---|---|
| `orbit_insertion` | 车站链,上升段判决,预期轨道协议 |
| `parking_propagation` | M-05 propagation (a `not_checked` *declaration* is not evidence) |
| `landing` | 恢复达阵/预备队/手掌式——`not_checked`的未流第一级盖 |
| `mass_conservation` | 测量的每个质量门(`mass.*`,边界`*.mass_conservation`) |
| `epoch_frame_continuity` | 边界连续性门(`handoff.*`,`parking.*`:历元,框架身份,组合共享源). |

Each check carries its own tri-state verdict and its own reason codes (a
missing check reports `<check>_not_evidenced` with a reason read from the
scenario/mission notes). The overall verdict is the same conjunction as the
phase gate — failed beats not_checked beats passed — so **any required check
that is `failed` or merely `not_checked` keeps the physical gate from
passing**; only all-five-passed yields `passed`. The pipeline records it as
`physical_mission_verdict`; the E2-03 phase verdict is reported alongside and
unchanged, so a frozen mission like E2-04 still reads `passed` per phase while
the physical gate honestly says `not_checked`.

### The reproducible reference mission (E5-05, task #272)

The reference is now committed under
`tests/data/mission/reference/e5-05-reference-mission/` (recording commit
`64f64a6`). It uses the RTLS vehicle redesign, a fixed-program computed OT
ascent and recovery driven from the canonical separation handoff. The
recorded orbit is **265.5 × 278.1 km at 51.0°**; the earlier blocked 400 km
recording proposal is superseded.

The sealed single-environment report passes its applicable phase checks but
has **physical verdict `not_checked`**, with parking propagation not
assessed. It predates the E8-03 coverage/qualification fields. A current
cross-environment run must supply the missing TudatPy propagation evidence
and satisfy the physical gate; a stored green report alone does not do so.
See [running and viewing the reference](computed-mission.md) and the
[regression procedure](https://github.com/zangjiucheng/orbitopt/blob/main/docs/flightlab-regression-ci.md#e5-05-the-reproducible-full-physical-reference-mission-task-272).
E2-04 remains the synthetic integration fixture.

## Full-chain reference validation (E8-01, task #281)

E5-05 commits the full-physical reference mission; E8-01 grades that
*recorded chain* — MECO, separation, fairing jettison, SECO/insertion, the
RTLS boostback/entry coast and the landing — against the independent
references committed with it, in one `full-chain-reference` report
(`python -m orbitopt.verify.full_chain`, gated by
`tests/test_e801_full_chain.py`). Every band and its derivation are listed
in the [reference tolerance audit](https://github.com/zangjiucheng/orbitopt/blob/main/docs/reference-tolerance-audit.md#full-chain-reference-bands-e8-01-task-281);
the PROVENANCE digest check re-runs on every build, so the report can never
be graded off edited data. No simulation runs: all compared numbers are the
committed reference files.

E4-03证据的分歧是诚实的,

* `independent_model` — 共享无动力弹道窗口:原始
  OpenTsiolkovsky级-1碎片轨对格子回收腿之间
  分离和助推点火——两个独立配制的模型
  the same physical arc. Since E8-05 (task #294) this is the only
  independent-model check in this report: the ascent event times and the
  插入元素比对 OpenTsiolkovsky的 *自有原始输出*
  将链条与计算 上升段 的程序进行比较,所以它们都是
  标注为 `numerical_consistency`(活动从原始状态提取)
  和`analytical`(Kepler身份超过记录状态),
  are not counted as independent-model evidence.
* `analytical`——相对于所记录的SECO状态的viva/Kepler身份,
  齐奥尔科夫斯基理想的-2 燃烧的舞台上 和精确的双体
  轨道传播 助推海岸到入口接口.
* `numerical_consistency` — SECO- 国家序列化/桥梁往返旅行,
  着陆和触地得分事件 针对密封的助推器州
  历史
* `design_input` – 触地得分的门,对准飞行目标.
  自身极限。 合成车辆的着陆没有独立性,或
  遥测参考,无检查要求
  (`historical_telemetry: not_checked`).

Because the ascent was computed *by* OpenTsiolkovsky, the event checks
validate the chain's extraction, framing and reporting of the external
recording rather than two independent ascent simulators; the recovery leg
is where two independently formulated models (the 3-DOF recovery model vs the pinned
external binary) genuinely cross. The report's
`inputs.honesty_notes` repeat this so a reader of the JSON alone cannot
miss it.

## Independent dynamics validation (E8-05, task #294)

E5-07物理资格参考是一个星舰级任务,上升段由OpenTsiolkovsky计算,所以——按照接受规则,OpenTsiolkovsky与OpenTsiolkovsky对比不独立——模型证据——E8-05重算它与模型是独立的*该程序*,对申报的波段内的差异进行分级(`python -m orbitopt.verify.independent_dynamics`,由`tests/test_e805_independent_dynamics.py`守门,报告类`independent-dynamics`):

* **Ascent and stage-1 debris: a native point-mass twin.** The repo's own
  3 - DOF厂(`propagate_point_mass`),由作者驱动 *打开*
  姿态程序和节流调度(无事件驱动反馈),来自
  MECO,海岸至分离,船舶至SECO; 废级复入
  通过一个单独写的RK4 集成器与动力的CA 极.
  MECO/separation/SECO event states, mass, max-Q, the SECO orbit and the
  碎片撞击点与密封的原始状态历史和
  有记录的撞击事件。 将E8-02-的风格分成一个
  测量引擎- Isp 线(在录制的 Isp 329.26 s), a
  测量空气/大气线(密度1.03),一个测量数值线
  (dt/2),是一条已宣布为零帧/时线(同名历元和ECI).
  常规, 垫子匹配 < 1 m), 一个已宣布的零控制线( 相同)
  authored program), and a *reported, gated residual* that absorbs the
  互动和跟踪-执行术语——从未零化。
* **Powered landing: two independent re-flights of one plan.** No landing is
  在E5-07(`RECOVERY_GAP.md`)中飞行,所以燃烧在申报时被验证.
  垫面的设计走廊(插入状态和3-引擎着陆)
  配置是`design_input`; 车辆文件本身记录
  13 - 引擎集不能在重量以下节流: gfold SOCP 弧重流
  由FL-301宣传员和新撰写的独立编组员
  (本国大地测量学,IGF-80重力,US76表,物理-推力指令重播).
  触地得分位置、垂直/边速、末端燃烧状态和剩余
  在重新飞行之间比较推进剂;FL-301设计门是
  分级为`design_input`.

乐队及其衍生作品位于[reference tolerance audit](https://github.com/zangjiucheng/orbitopt/blob/main/docs/reference-tolerance-audit.md#independent-dynamics-bands-e8-05-task-294). 本报告没有使用`historical_telemetry`证据进行检查:参考车为公众估计,而非飞行硬件.

## Full recovery independent re-flight (E10-07, task #304)

E8-05's landing evidence starts from a *declared* design corridor, so it
never re-flew an actual recovery trajectory; it is not independent physical
verification of the complete RTLS chain (the E5-08 report now records the
corridor-start check as `design_input`). E10-07 closes that gap end to end
(`orbitopt.verify.full_recovery_reflight`, benchmark
`benchmarks/bench_e10_07_full_recovery_reflight.py`, gated by
`tests/test_e10_07_full_recovery_reflight.py`, report kind
`full-recovery-reflight`, committed results in
`benchmarks/e10_07_full_recovery_reflight.json`):

* 启动状态是*实际数* DC8 热静离散式交接
  从OpenTsiolkovsky OT厂中新鲜出来——分离时 t = 125.0 s,质量.
  900 172 kg, 1 569.8 m/s — pinned by a SHA-256 handoff digest that the
  快速测试从记录状态重算. 没有宣布的走廊
  用作起始条件。
* 一个独立书写的双胞胎将整个链条重飞——回升到
  截断,E10-01无动力气动进入走廊读取,
  和撞击的终端下降——拥有自己的RK4集成器,拥有
  WGS-84大地测量学,拥有J2+ECEF旋转动力学,教科书US76
  (地质潜力)表和推力重播*不变*
  编写任务剧本. 实际跨走廊的着陆阶段
  交接(1 067 m/s)是用`fly_landing_burn`探测的,是诚实的.
  报告不可行(`LANDING_INFEASIBLE`),没有门作为通行证。
* Nine acceptance quantities (cutoff time/position/speed, entry peak-q and
  将峰值-g、撞击时间/位置/速度、剩余推进剂)与
  所测量的波段内的主 FL-301 模型,每个差异都是
  拆分为错误预算行: 已声明的零帧线和控制线,
  a 测量数值线(dt/2),测量重力线(双线,无
  J2)和测量的大气线——主模型的双重运行.
  密度简介。 大气的*窗体*线占了整个
  输入-高峰-q和撞击-速度差异(剩余0.0),所以这些检查
  由两个密度剖面而成,而不是由宣传者而成。
* The verdict stays honest: the main model itself reports
  `CORRIDOR_MISSED`+`ENTRY_Q_LIMIT_EXCEEDED`在此交接(E10-01).
  NOT-CLOSED状态,报告也这么说;独立复飞.
  协议在这里证明*链式物理*,而不是封闭设计.
  All checks carry `independent_model` evidence under the `CHECK_MODEL`
  主体——比走廊启动检查更明显,更严格的类别.

见[E10-07 full recovery re-flight report](e10-07-full-recovery-reflight.md).

The [E5-08 variants](e5-08-new-vehicle-report.md) apply these checks to new
designs. E10-04 closed both open discrepancies without touching a band: the DC7
SECO semi-major-axis fail (27.7 km against 15 km) was an interpolation artifact
in the comparison itself and now measures 5.85 km; the Starbase debris impact
speed (12.80% against 12%) was evidenced by identical-state replay to be a
genuine power-off drag-model difference between the independent models and is
recorded as still failing. See
[E10-04](https://github.com/zangjiucheng/orbitopt/blob/main/docs/e5-08-new-vehicle-report.md#e10-04-closing-the-two-open-discrepancies-task-301).
Their nominal joint physical passes do not clear the remaining
independent-model failure.

## Final schemes in cross-environment CI (E10-09, task #306)

E10-09 promotes the two sealed E5-08 variants -- **DC7** and **Starbase** --
to a dedicated cross-environment CI regression: the `cross-env` job now flies
each scheme's own frozen plan (`crossenv_e508_dc7_plan.yaml`,
`crossenv_e508_starbase_plan.yaml`) across the tudatpy/gfold boundary and
re-grades the joined run with `scripts/e1009_starbase_dc7.py crossenv-check`:
all five physical checks passed with every evidence source pinned to an
envelope digest that re-hashes, `physical_mission_verdict` passed, the
qualification block matching a re-grade (its `independently_verified` verdict
and reason recorded as-is -- False with the one admissible blocker, the
mission chain carrying no validation-grade independent check), the recovery
leg driven from the sealed separation handoff under the sealed
Earth-orientation snapshot, and the flown touchdown inside the frozen
tolerances. The original E5-08 record stays in the same job as the
**expected-fail negative regression**: its run must keep grading a clean joint
fail and the check keeps accepting exactly the sealed gap shape. The reference
outputs are frozen by `tests/data/mission/reference/e10-09-ci/tolerance-audit.json`
(digests of both tolerance files and crossenv plans plus the full audit trail,
including E10-04's finding that no band was ever relaxed); the G5 pair
`verify` + `tag-check` refuses any tolerance or plan change that carries no
new audit entry in the same diff. The epic's aspirational two-ignition
hot-staging gate remains classified out of vehicle capability (E10-01,
E10-03, E10-07), so these three-ignition records are graded exactly as the
final schemes they are.

## 参照公众参考的基准

```bash
orbitopt benchmarks
orbitopt benchmarks --json
```

如果有任何基准失败, 命令退出 1 。 参考资料和假设载于[engineering notes](https://github.com/zangjiucheng/orbitopt/blob/main/docs/engineering-notes.md#benchmark-scenarios-with-public-references--analytic-solutions-task-245)。

## Cross-model comparison

对比一个常见的上升段/coast 横跨可用的3-DOF,JSBSim,tudatpy与分析后端:

```bash
python -m orbitopt.verify.cross_model --backends three_dof,six_dof,tudatpy
python -m orbitopt.verify.cross_model --threshold ascent.position_m=1000 -o cross_model.json
```

`three_dof_vs_two_body_analytic`对为`analytical`;其他后端对为`independent_model`. 这些吊带名称与Flight Lab的[canonical run backend identities](https://github.com/zangjiucheng/orbitopt/blob/main/docs/flightlab.md#backend-identities-and-fidelity-levels)不同.

## 集成器趋同

LEO / GTO 海岸,复入和动力-上升段 研究扫射耐受度,并在支持时,进行步长. 他们的支票是`numerical_consistency`:

```bash
python -m orbitopt.verify.integrator_convergence --plot
```

## 跨后端接受矩阵

`src/orbitopt/data/acceptance_matrix.yaml`声明每对后端对,相和度量限制,并附有其理由和故障原因类. 结果使用标准报告/门机制。 一个无法使用的后端是`not_checked`有原因,而不是默默通过. 附加条件或明确加挂的行被记录;`--threshold key=none`禁用该行的门.

```bash
python -m orbitopt.verify.acceptance_matrix
python -m orbitopt.verify.acceptance_matrix --matrix my.yaml \
  --threshold three_dof_vs_two_body_analytic.coast.position_m=0.005
```

一个可用对子的通行证不包括其余的矩阵。 当前接受运行的是真实的数字/分析对,而不是每个后端对.

## Model discrepancy budget (E8-02, task #282)

一个有门的总数说*是否为*,而不是*什麽*. `orbitopt.verify.discrepancy_budget`再利用接受矩阵(相同的吊带运行,相同的解容),并将每个测量对数的差分为五个源线——帧/时间,数值集成,发动机,空气,控制器——记录组件和总数之间的剩余. 剩余部分是未分配的植物配制剩余部分(刚体对点质量,FCS/actuator动态),说明而非零加成. 线路是按以下方式获得的:

- **frame/time** — one shared bridge at one epoch: 0 applied, with the
  IERS 0.9 脱氧核糖核酸敏感度记录为预算;
- **数字整合** – 耐受-放松探测器(3 - DOF参考文献).
  rtol/atol × 1000; 固定步骤边是已公布的E6-04绑定;
- **引擎**-实测E6-01对等余值为推力-比例.
  探测器( 拒绝失败的对齐, 没有编入预算); 质量通道
  带有已披露的`mdot·dt`单面记账约束;
- **飞机**-0通过声明(`parity-jsbsim-aero-zero`),与.
  已记录的*潜力*生产机体,未汇总;
- **控制器** – 循环悬浮探测器份额(2 × PID增益).

不飞的对称以矩阵本身的因子代码产生`not_measured`线条——从不假零. `--no-probes`记录结构,剩余部分为全部总数. 只有矩阵行的投票:报告状态反映了矩阵判决.

```bash
python -m orbitopt.verify.discrepancy_budget            # all installed backends
python -m orbitopt.verify.discrepancy_budget --no-probes --json
```

## 模型 覆盖范围和宣布的公约

任务结果记录了每个阶段的模型和局限性. `orbitopt.physics_io`检查框架,历元,时间尺度和物理边界的单位常规. 边界的连续性确立了一致的交割;它没有证明记录的上升段被动态传播。

## 进一步阅读

[Flight Lab](flightlab.md) · [Regression CI](flightlab-regression-ci.md) · [Examples and validation](examples.md) · [Acceptance findings](acceptance-e1-e4.md)
