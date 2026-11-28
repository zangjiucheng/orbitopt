# ADR会合-0004:任务演播室重播Flight Lab跑车(E12-06).

- **Status:** Accepted
- **Date:** 2026-11-25
- **Deciders:** E12 Orbital Rendezvous & Mission Operations (Epic #389), E12-06 (#396)
- **Code:** `src/orbitopt/rendezvous/studio.py`, `src/orbitopt/flightlab/ui/rendezvous_panel.py`,
  `src/orbitopt/flightlab/{compare,scene_export}.py`, `src/orbitopt/flightlab/ui/{telemetry,flight_panel}.py`,
  `src/orbitopt/cli.py` (`orbitopt rendezvous studio`)
- **Evidence:** `tests/test_e12_06_mission_studio.py`
- **Builds on:** ADR rendezvous-0001 (plan), ADR rendezvous-0003 (independent verification), ADR relative-0001
  (`propagate_pair`, LVLH frames), the frozen FlightResult/scene/mission/replay v1 contracts

## 背景和问题

LEO-RDV-001 was planned and verified (E12-03) but had never been *seen*: the task
requires the mission to replay in Flight Lab with both trajectories, relative
range/speed curves, the maneuver timeline and the safety verdicts — reusing the
existing panels, not a parallel UI. Three contract facts shape the design:

1. `flight-result-1.0`被冷冻,其事件类的enum是上升段 -只使用
   (`ignition`…`MECO`…), so rendezvous maneuvers cannot enter the run manifest
   没有打破 每一个消费者的阴谋。
2. 合同运行有一个地球场景,没有车辆:`vehicles=()`是合法的,
   但是,没有地方可以执行计划, 球场或球场
   验证报告(`RendezvousVerification`)不是
   `Verification` block, so its five gate checks do not appear in the Checks tab
   除非制片人重复一遍
3. `propagate_pair`产生的密集显示轨迹不会在
   已核实的到达状态到位点: 子步骤到网格终点不同
   从经核实的双点腿乘~4e-5米,并重新确定从
   `arrival_epoch_s - drift_end_s` 而不是计划`tof_s` 需要另一个
   ulp. 上次对报告有异议的重播就是
   让验证的故事禁止。

## 决定

1. **The studio run is a `FlightResult` v1 run directory plus one new sidecar.**
   `build_studio_run`写合同运行(`orbitopt rendezvous studio -o DIR`),
   `MultiSpacecraftMission`文档(`mission.json`)和统一重播
   (`replay.json`)——所有已发运的计程验证——并添加`rendezvous.json`.
   (`rendezvous-studio-1.0`):计划,验证,事件,保持结果,坚持.
   点和静态标记位置。 人工/持有事件和区数据实况
   **仅限**在侧车厢;跑步的`events`保持`()`. `RunBundle.rendezvous`
   随身携带到UI;
   不变。
2. **The last timeline sample is the verified arrival state, bit-exact.** The
   追逐者最后的样本是从`verify_plan`的精确双点腿重生的
   从相同的 t 0 中运行( `tof_s` , 不重算差异), 不
   from the dense grid; the arrival row also carries the *post*-Δv2 velocity. A
   guard in `build_studio_run` (tolerance 1e-9 m) compares the re-derived
   与报告相对,拒绝对分歧进行跑步——
   第一个版本随密集端点运出, 守护失败 。
   4.13e-5 m,这是发现差异而不是发运的方式.
   顺序: `rel_speed` / `rel_*_lvlh` 曲线在准确数字上结束
   (测试称`==`,而不是`approx`)。
3. **Display extensions without renderer surgery.** The keep-out centre and the
   作为现场标记机构(发射场)进入现场
   mechanism, `hideTrail` static trails); the two sidecar events join
   `timeline["events"]`喜欢任何合同活动. 五个相对通道再利用
   外柱遥测管道 并排放**仅当存在时**,所以
   发射运行的频道集未受影响。 重新对接的标签
   侧车; 关键事件表显示侧车操纵与合同
   事件。
4. **Deterministic provenance markers.** `git_sha`/`created_utc` and the omitted
   `backend`野外携带固定的有文件记载的标记(全零sha,任务历元).
   因此运行会保持字节复制;`target_sha256`是舟-JSON.
   任务配置摘要

## 后果

- 生成 LEO-RDV-001与 `orbitopt rendezvous studio -o runs/leo-rdv-001-studio`,
  then select `leo-rdv-001-studio` in Mission Control's **Runs** workspace.
- 副车厢显示元数据: `scene_export` 跳过错误的标记
  并且没有UI路径将其读入物理.
- 未来带有事件的合同版本可以吸收侧车的`events`
  逐字逐句;在编审中别无他法需要改变.
