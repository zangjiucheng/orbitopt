# ADR飞行实验室 -0004:从独立物理验证分离后端自查.

- **Status:** Accepted
- **Date:** 2026-11-14
- **Task:** E4-01 Independent 6-DOF Verification (#261), parent epic E4 Verification Assurance (#249)
- **Contract:** Flight Lab v1 `verification.json` (`schema_version` stays `1.0`; `backend_self_check` is a new **optional** property)

## 背景和问题

The Flight Lab run contract carries three independent Gate-B verdicts
(`optimizer_converged`, `physics_verified`, `mission_constraints_passed`). Two
backends fill `physics_verified` with **different meanings**:

- **3-DOF (OpenTsiolkovsky):** `physics_verified` is the outcome of the
  独立 tudatpy 插入 **重飞** – 一个 *不同* 宣传器
  重新整合海岸/插入状态和Gate-G5元素驱动检查
  评断(`orbitopt.flightlab.simulate.orbit_refly`). 这是真的
  cross-model verification.
- **6-DOF (JSBSim, formal backend, task #226):** `physics_verified` was simply
  JSBSim的*拥有*在运行中的检查组合(无限状态,质量)
  簿记、交接的连续性、跟踪范围。 这就是整形师
  **为自己担保**,不是独立的检查——同字段名称的意思
  有两样东西,绿色的6 -DOF悄悄地声称有保证
  从来没有

两项要求都没有幸存的验收审查:验证旗必须反映除生产者之外的其他东西的证实,或者不得设置.

## 决定

将两个想法分成两个字段,并给6 - DOF后端同样的*独立*物理检查3 - DOF的血统已经:

1. **New field `backend_self_check` (optional bool/null).** The conjunction of
   生产后端的支票。 由每个适配器编写
   (`opentsiolkovsky._verification`,`sixdof.run_ascent`)和由 spike尖.
   司机们 因为它是可选的,用 `.get(...)` 读取,运行目录
   写入此字段之前, 读作 `None` , 并且仍然验证 。

2. **`physics_verified`现在在两个后端上意味着一件事:**的结果
   独立复飞 / 交叉模型检查——从不由制片人自言自语.
   `None` =无独立复飞(未评估),从未通过.

3. **6-DOF independent re-fly.** `simulate(backend="6DOF")` calls
   `simulate.orbit_refly_6dof`,即:
   - 读取运行的海岸/插入状态(ECEF,如6-DOF跟踪它),
   - 通过帧桥旋转固定的地球 → GCRS
     (`orbitopt.flightlab.frames.to_gcrs`, carrying the `ω × r` velocity term),
   - 将GCRS状态输入**不变** `_refly_gcrs_state`核心(以及
     same tudatpy point-mass propagation and Gate-G5 drift checks) that
     `orbit_refly` uses for the 3-DOF state.
   海岸状态不是封闭轨道的垂直跳/亚轨道情景读作**未评估**(`physics_verified=None`),绝不是物理故障. 如果帧桥或tudatpy不可用,则支票是未评估(`REFLY_BACKEND_ERROR` /`REFLY_BACKEND_MISSING`)——绝不是假的`True`.

4. `checks.evaluate`和回收合并**保护**两个字段逐字;
   `simulate`线`backend_self_check`通过复飞判决.

## 后果

- A green 6-DOF `physics_verified` now carries the same weight as a green 3-DOF
  一,独立宣传员同意. 在无法评估的情况下,实地
  读`None`,跑的假设说原因.
- 后端的自我一致性仍然被记录和诚实——它刚刚转移到了
  `backend_self_check`,所以自摸永远不会误认为是确证.
- 字段是添加/ 可选的, 因此现有的运行目录和 CLI
  读取器继续工作; `orbitopt flight simulate` / `optimize` 两种印刷
  分开判决
- 接受限制: 6 - DOF 重新飞行是点质量的重新整合.
  海岸州(比对3 - DOF支系自身重飞),不是6 - DOF重跑;
  验证轨道状态 上升段 而不是姿态动力。
  态度/演员忠诚度是FL-401 /402 /403故事(ADR飞行实验室-0001),
  在那里作为自己的支票追踪。

## 证据指数

- 字段+语义:`src/orbitopt/flightlab/contract.py`(`Verification`),
  `src/orbitopt/schemas/flight-result-1.0.json`.
- 3 - DOF 重飞核心,6 - DOE~GCRS 桥入口:
  `src/orbitopt/flightlab/simulate.py` (`_refly_gcrs_state`, `orbit_refly`,
  `orbit_refly_6dof`, `_simulate_6dof`).
- 适配器记录自查:`adapters/opentsiolkovsky.py`,
  `adapters/sixdof.py`.
- 测试:`tests/test_flightlab_contract.py`(回合/遗产违约/
  独立),`tests/test_flightlab_simulate.py`(桥令+失败)
  路径), `tests/test_flightlab_sixdof_backend.py`(适应器自检查,而非
  物理学。
