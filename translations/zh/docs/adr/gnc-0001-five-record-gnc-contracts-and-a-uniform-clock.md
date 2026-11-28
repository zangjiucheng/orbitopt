# ADR gnc-0001: five record contracts and a uniform clock for closed-loop GNC (E13-01)

- **Status:** Accepted
- **Date:** 2026-11-26
- **Deciders:** E13 Closed-Loop GNC & Relative Navigation (Epic #416), E13-01 (#418)
- **Code:** `src/orbitopt/gnc/` (`contracts.py`, `clock.py`, `feed.py`, `recording.py`, `sim.py`),
  `src/orbitopt/schemas/gnc-contracts-1.0.json`
- **Evidence:** `tests/test_e13_01_gnc_contracts.py`, `tests/test_e13_01_clock_contract.py`,
  `tests/test_e13_01_truth_isolation.py`, `tests/test_e13_01_replay.py`
- **Builds on:** ADR relative-0001 (`RelativeState`, pair propagation), ADR rendezvous-0004
  (封装重播),E12-08 舟子计划文摘政策(sha 256,8 - 显著数字化).

## 背景情况

E12 qualified rendezvous plans open-loop: a plan was verified, then propagated. Closed-loop
guidance and navigation (#416) introduces a plant whose true state must never be an input to the
law that steers it, and a sensor/estimator/control loop whose timing decides what each module was
allowed to know. Two failure modes have to be closed architecturally, not by care in a notebook:

1. **Truth leakage.** If plant truth reaches the guidance law — or the record claimed as an
   观察所携带的是真理而不是一个衡量标准——每个一致性和稳健性数字
   循环生产是虚构的。
2. **Clock drift.** If a dropped frame shifts the next sample, or latency is retro-applied, the
   运行的时间戳不再说明估计者消耗了什么以及何时消耗;重播停止成为证据.
   E12-10还测量了作为主要故障源的激活器执行错误,那么,
   *其实*型推进器确实需要自己的记录,与所指挥的分离.

## 决定

1. **Five versioned record kinds** — `TruthSample`, `Observation`, `Estimate`, `Command`,
   `ControlEvent`-在`gnc-contracts-1.0.json`中捆绑的图案,每个图案都有出处字段.
   (源传感器,源观测/指令,源指令). `Observation`和`Estimate`
   将 E12-01 `relative-state-1.0` 文档逐字嵌入;无平行相对状态类型
   介绍。
2. **One `ClockSpec` owns every instant.** Timescale is uniform `TT` or `TDB` (UTC inside a control
   循环被拒绝). 每个样本、更新和勾选时间都是其勾选指数的一个纯函数
   (`phase + n·period`); 滴放的传感器滴滴在时钟文档中宣布,所以
   放下删除记录, 以后永远不要重回记录。 宣布并强制执行感应延缓;
   控制将命令射击时间与激活时间分开。
3. **Guidance sees a one-record view, not the plant.** `GuidanceInput` is a frozen capability
   对象为 `__slots__ = ("_latest",)` 携带一个 `Estimate` ; 法律是自由函数
   `(GuidanceInput, time)`. `feed.py`禁止提及真相:任何AST测试均失败.
   模块中的 `truth` / `plant`标识符. 植物将真理直接写到运行日志.
4. **A run log closes and seals the run.** `GncRunLog` validates every causal gate on append
   (抽样后的计量、抵达后的估计、请求时间的启动、一个事件)
   每个命令,任务名册历元等于时钟`t0`),然后将`run_digest`封在上方.
   使用 E12-08 文摘策略的目录文档; `read_run_log` 重新验证和重新检查
   文摘,所以密封的运行是可重播的证据,而不是笔录.
5. **可选 `plan_digest`** (64 -hex) 将一跑绑在E12-08的犬形计划上而不强制.
   封闭式学习要有计划

## 后果

- Closed-loop results become replayable and tamper-evident under the same digest discipline E12
  已建立; # 416 后续继承时钟,不能重新定义时间.
- 隔离是机器检查(推侧`TypeError`,`__slots__`,AST扫描),所以是一个未来.
  方便"只是把真理传递给法律" 打破了测试,而不是沉默的实验.
- 成本:再将一个计划捆绑到版本; a 运行不能混合墙钟 UTC(故意); 联合国
  `sim.py`的参考sim是J2 / CW-级对子传播器——独立物理重飞停留.
  tudatpy工具的工作(ADR与-0003/0004),不属于本合同.

## 拒绝的替代品

- **消息总线/ pub- sub 模拟框架 :** 添加运行时间依赖,仍需要一个
  sealing story for replay; the five record kinds plus a push-only feed give the same isolation
  没有新的机器。
- **传感器时标 :**为E12故意避免的转换记账;制服
  具有宣布的阶段、时间和时间的时钟已经包括同步取样。
- **将真理传递到旗下快速测试的指引:**是一个带有TODO的泄密;
  # 418的接受标准要求结构隔离.
