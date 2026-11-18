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
| `analytical` | 同/简化的模型的封闭形式. | 同意这个解决办法。 |
| `independent_model` | 单独配制的宣传机或模型. | 在检查的阶段交叉模型一致性。 |
| `historical_telemetry` | 已公布飞行观察,而非模型输入. | 同意这些意见。 |
| `design_input` | 目标轨道,瞄准点或宣布的信封. | 解决办法满足了设计方面的限制。 |
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

渔获量有助于进行两项独立验证检查;其他五项基准检查是分析的或数字的。 这是范围大的证据,而不是每个任务阶段都得到验证的说法。

## Full mission verification gate (E2-03, task #256)

`inputs.full_mission`分三个阶段:

| 阶段 | 检查/适用性 |
|---|---|
| `orbit_injection` | 上升段判决,站点命令,中转,质量和预期轨道;始终适用. |
| `parking_propagation` | M-05 轨道传播;仅在`expected.propagation_result: not_checked`被宣布时(在`full_mission`被宣布时重复)才退出范围. |
| `recovery_touchdown` | 回收/处理/触碰限制;在`recovery:`申报时适用(在`full_mission`申报时适用)。 |

Among applicable phases, **failed** takes precedence over **not_checked**,
which takes precedence over **passed**. An applicable phase with no checks
is `not_checked`. On a run that does not declare `scenario.full_mission`, a
declared out-of-scope phase does not block the verdict. The chain's
diagnostic `status` and the pipeline's `full_mission_verdict` are separate
fields; inspect the phase document when a run is only partially successful.

### Verification coverage gate (E8-03, task #283)

一个情景可能宣布`scenario.full_mission: true`:运行*这是*是完整的物理任务,所以以上三个阶段都变成**所需物理阶段**,不能通过声明而脱离范围:

- `expected.propagation_result: not_checked`是**拒绝**——阶段.
  继续适用,没有证据记录,并读作`not_checked`;
- 失踪的`recovery:`区块同样离开`recovery_touchdown`
  `not_checked`;
- 要求的阶段上限
  `not_checked`,所以范围外的路线无法再产生
  从未传播到其停车轨道的特派团的总体通行证
  或者从来没有带助推器回家。

新生成的大门文件涵盖所需阶段——`inputs.full_mission.coverage`列出每阶段状态(`checked`,`failed`计数如检查,`not_checked`,`declared_out_of_scope`,`not_declared`)和`summary`,如`2/3`,管道校验舞台记录(`full_mission_coverage`),并在注释中呼应——以及`inputs.full_mission.qualification`块. 报告的总体描述可能要求**独立核查** 只有当`qualification.independently_verified`是`true` :跑队宣布为`full_mission`,跑门判决是`passed`,全覆盖`3/3`,跑队的独立证据滚滚(`independent_validation`)通过. 部分运行(例如消耗性特派团)无论适用阶段如何绿色,都没有任何条件——其覆盖情况摘要准确地说明了运行达到多大程度。

The chain preserves `ascent.physics_verified` as independent-model evidence:
`true` passes the check; `false` fails orbit injection and the full-mission
gate. `null` remains explicitly unassessed and contributes no independent
validation. A green design-constraint rollup with a null flag does not prove
independent verification. See the [acceptance report](acceptance-e1-e4.md).

### Full physical mission gate (E5-04, task #271)

这种范围外的豁免恰恰是*计算*任务绝不能躲在后面(E2-04宣布停放轨道传播不在范围内,至今仍未通过). `inputs.full_mission.physical`重级与**five required physical checks**相同的链式检查——无适用性逃生舱:

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

* `independent_model` — the raw OpenTsiolkovsky v1.0.0 recording
  (外部开发的二进制,绑定文摘):链条的MECO,
  分离、 丢弃和 SECO 事件时间
  状态历史( 推力 - off / 中转质量转换, 不是事件)
  记录已复制的适配器、已实现的停放轨道与
  框式缩放原SECO状态,与原Z againstGbold回收腿相对
  OT级 -1碎片轨迹 位于共享无动力窗口之间
  分离和助推点火。
* `analytical`——相对于所记录的SECO状态的viva/Kepler身份,
  齐奥科夫斯基理想的-2 燃烧 和精确的双体
  轨道传播 助推海岸到入口接口.
* `numerical_consistency` - SECO- 国家序列化/桥式往返旅行,
  着陆和触地得分事件 针对密封的助推器州
  历史
* `design_input` – 触地得分的门,对准飞行目标.
  自身极限。 合成车辆的着陆没有独立性或
  遥测参考,无检查要求
  (`historical_telemetry: not_checked`).

Because the ascent was computed *by* OpenTsiolkovsky, the event checks
validate the chain's extraction, framing and reporting of the external
recording rather than two independent ascent simulators; the recovery leg
is where two independently formulated models (the 3-DOF recovery model vs the pinned
external binary) genuinely cross. The report's
`inputs.honesty_notes` repeat this so a reader of the JSON alone cannot
miss it.

## 参照公众参考的基准

```bash
orbitopt benchmarks
orbitopt benchmarks --json
```

如果有任何基准失败, 命令退出 1 。 参考资料和假设载于[engineering notes](https://github.com/zangjiucheng/orbitopt/blob/main/docs/engineering-notes.md#benchmark-scenarios-with-public-references--analytic-solutions-task-245)。

## Cross-model comparison

对比一个常见的上升段/沿海3-DOF,JSBSim,tudatpy与分析后端:

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

`src/orbitopt/data/acceptance_matrix.yaml`声明每个后端对对,相和度量的限值,及其理由和故障原因类. 结果使用标准报告/门机制。 一个无法使用的后端是`not_checked`有原因,而不是默默通过. 附加条件或明确加挂的行被记录;`--threshold key=none`禁用该行的门.

```bash
python -m orbitopt.verify.acceptance_matrix
python -m orbitopt.verify.acceptance_matrix --matrix my.yaml \
  --threshold three_dof_vs_two_body_analytic.coast.position_m=0.005
```

一个可用对子的通行证不包括其余的矩阵。 当前接受运行的是真实的数字/分析对,而不是每个后端对.

## Model discrepancy budget (E8-02, task #282)

一个有门的总数表示*是否为*,而不是*什麽*. `orbitopt.verify.discrepancy_budget`重用接受矩阵(相同的吊带运行,相同的解容),并将每个测量对数的差分为五个源线——帧/时间,数值集成,引擎,空气,控制器——记录组件和总量之间的剩余. 剩余部分是未分配的植物配制剩余部分(硬体对点质量,FCS/actuator动态),说明而非零加成. 线路是按以下方式获得的:

- **frame/time** — one shared bridge at one epoch: 0 applied, with the
  IERS 0.9 脱氧核糖核酸敏感度记录为预算;
- **数字整合** – 耐受-放松探测器(3 - DOF参考文献).
  rtol/atol × 1000; 固定步骤边是已公布的E6-04绑定;
- **引擎**-实测E6-01对等余值为推力-比例.
  探测器( 拒绝失败的对齐, 没有编入预算); 质量通道
  带有已披露的`mdot·dt`单面记账约束;
- **aero** — 0 by declaration (`parity-jsbsim-aero-zero`), with the
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
