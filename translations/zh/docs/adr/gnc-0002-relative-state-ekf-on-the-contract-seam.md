# ADR gnc -0002:合同接缝上的相对态EKF(E13-02).

- **Status:** Accepted
- **Date:** 2026-11-27
- **Deciders:** E13 Closed-Loop GNC & Relative Navigation (Epic #416), E13-02 (#419)
- **Code:** `src/orbitopt/gnc/ekf.py`, `src/orbitopt/gnc/nav_metrics.py`,
  `benchmarks/bench_e13_02_relative_nav_ekf.py`
- **Evidence:** `tests/test_e13_02_relative_nav_ekf.py`,
  `docs/benchmarks/e13_02_relative_nav_ekf.{json,md}`
- **Builds on:** ADR relative-0001 (CW + `propagate_pair` + LVLH frames), ADR gnc-0001
  (`Observation` /`Estimate`合同,统一时钟,出处规则).

## 背景情况

闭路导引(E13-04)需要通航状态,而不只是传播的:接受这项任务是统计性的(RMSE,NEES/NIS在申报的chi-quare波段内,覆盖范围,可观察性,模型不匹配下的行为,多种子组). 在任何数字都意味着什么之前,必须解决三个问题:

1. **In which frame does the filter live?** The planning model the whole epic reuses (E12-02 CW)
   定义在 LVLH; 过程-噪声直觉(白色加速度)是帧固定的; 合同
   携带惯性矢量。
2. **模型的过程从何而来?** Duplication CW 或内部的对子传播器
   过滤器将把第二个真相来源放在 E12-02 旁边 — 这个 repo 的 ADR 的准确漂移
   继续杀人
3. **What is a sensor measurement?** The `Observation` contract carries a Cartesian relative state.
   测距/测距传感器是*没有*测量的;从笛卡尔字段和
   然后用过滤器“修正”会重复计算记录中已经包含的噪音。

## 决定

1. **Filter state = 6-vector LVLH `(ρ, ρ̇)`**;控制`u`是LVLHΔv冲动通过.
   `B = [0; I]`. 通过`make_estimate`的出版将状态和共性转换为合同的.
   发布时与酋长状态的惯性框架; 共变由
   linear chief-fixed change-of-frame Jacobian (`frame_covariance`, finite-difference-verified
   对`lvlh_to_eci`.
2. **The process model is a pluggable pair `(f, F)` that delegates.** `CwProcess` calls E12-02's
   `propagate_cw`;`NonlinearRelativeProcess`与参考主管合称`propagate_pair`.
   (酋长的动议是雅各比亚批次名义上的一行)
   轨道传播),`F`由该批次的中央差异而产生. 测试都对E12-02
   直接调用: 过滤器重用 E12-02 的动态, 它不重用 。
   Q是块形-恒定白-加速元件(`white_noise_acceleration_q` ).
3. **Measurement models are components with a `dof` property** (`RelativePosition`,
   `RelativeVelocity`,`RangeRate`,`RangeAngle`,`Stacked`. 笛卡尔提取转换
   合同的惯性读取通过呼叫者供应的酋长国读取到LVLH(模型拒绝)
   没有它。 原生传感器从`Observation.metadata`读取在声明的密钥下
   `native_measurement` / `native_measurement_covariance` – 摘录 **拒绝** 重排.
   角度或范围从笛卡尔字段(消息说为什么:重复计算噪音).
4. **标准数字,一种报告类型.** Joseph-form更新,对称共变; 每
   更新返回 `UpdateReport`( 创新、 创新共变、 NIS 、 已消耗的 ids) 因为
   接受度量衡消耗过滤器本身的输出,而不是重新执行。
5. **Acceptance is a gated Monte-Carlo JSON.** 70 cells × 5 seeds
   (`docs/benchmarks/e13_02_relative_nav_ekf.json`):匹配配对,不匹配(CW/TwoBody/J2/drag).
   truth vs CW/nonlinear filter), five sensor families, two geometries; declared χ²(6) 90 % NEES
   a. 带和NIS带; 3 = 覆盖; R-加权20-历元堆叠的可观察性扫描.
   仅在笛卡尔细胞上测试门的一致性——角细胞NEES损失为**报告.
   查找 **( 在 30 m 上错误的线性偏差), 并给它编码错误 。

## 后果

- Matched Cartesian cells are consistent to the declared gates (NEES pass ≥ 0.79, coverage 1.000);
  错配研究显示,NEES在未建模的差拖曳下下降至0.12 – 0.14 ——
  过滤 *检测自己* 错误, 关闭的循环会关闭属性 。
- Native angle sensors carry a linearisation bias the plain EKF cannot fix (NEES 0.22–0.51 on
  匹配的真相 : E13-04通过一个重写/ UK更新或接受; JSON是基线.
- 拖动不匹配仅在 模型 中可以修复 : 拖动状态或Q 膨胀大小
  力 – 两者都是 `f` / `Q` 插件在此接口后面,没有过滤器更改.
- `NonlinearRelativeProcess` costs 13 batched pair propagations per update (f + 12 FD columns):
  首先,一个分析-变量的`F`可以在以后无需API更改的情况下取代批量.

## 拒绝的替代品

- **Inertial-frame EKF:** the CW planning model (and its Q) are LVLH objects; an inertial filter
  或失去便宜的模型,或将CW线性化为惯性框架——二阶簿记.
  for no observed gain at 300 m.
- **错误状态( 错误中的 KF) 配方 :** 这里绝对相对错误很小
  (tens of metres at 300 m separation); a full-state EKF is simpler and the mismatch study already
  显示线性化实际被咬的地方(角更新,而不是状态大小)。
- **从笛卡尔领域重新开发本地观测器:**会使每一个本土感应细胞
  look as good as Cartesian while lying about the noise model; refused at the extraction boundary.
