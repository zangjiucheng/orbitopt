# E10-07 — Full recovery independent re-flight (task #304)

E8-05's independent landing evidence starts from a **declared design
corridor** ([E5-08 report](e5-08-new-vehicle-report.md)), so it never
re-flew an actual recovery trajectory. That check is a `design_input`: it
cannot be read as independent physical verification of the complete RTLS
chain. E10-07 re-flies the **whole chain from the actual separation
handoff** — boostback, the E10-01 unpowered aerodynamic entry, and the
landing phase — under an independently written twin model, and grades the
discrepancies against the main FL-301 model inside measured bands with an
error-budget split.

## 开始条件:实际交接,不是走廊

The start state is the DC8 hot-staging separation state (E10-02 timeline)
flown fresh through the OpenTsiolkovsky OT plant (run `ot-ec11389b-da42dd33`,
sealed in the committed report): separation at **t = 125.0 s**, mass
**900 172.42 kg**, speed **1 569.85 m/s**, ECEF state pinned by SHA-256
handoff digest
`60b3130118befaba86d8806d8f539adb6de4e1099ef129104934d52f768ab99c`.
The fast gate recomputes the digest from the recorded state and cross-checks
the OT event times against `benchmarks/e10_02_hot_staging.json`. No
declared-corridor state enters the re-flight.

## 双胞胎

`orbitopt.verify.full_recovery_reflight`通过独立写成的点-质量双胞胎重新发布*相同作者 `_Mission` 脚本*(密封的E10-01保持矢量`[21.2, -2.0, 1.0, 0, 0]`,IIP截断,无动力进入,闭路登陆需要模型):

* 拥有RK4整合器(名义0.1 s,数字预算项目减半);
* 拥有WGS-84大地测量(固定点纬度迭接);
* 自身重力:带有Coriolis和离心条件的ECEF的点质量+J2;
* 教科书 US76 地质潜力海拔表层,同质
  支架续用(独立于主模型 ' s ' s配`entry` 's)
  指数剖面图);
* 物理 - 推力 从车辆文件重放(计数 ^ 级别 ^ 真空减数 -
  环境推力,环境校正Isp,Cd从车极.

控制线被宣布为零(相同的脚本);框架线被宣布为零(相同的ECEF框架和历元).

## 结果(报告类`full-recovery-reflight`,所有检查类PASS)

| 终止原因 | 主场(FL-301) | 双 | Δ | 带 |
| --- | --- | --- | --- | --- |
| 截止时间(s) | 182.25 | 182.50 | 0.25 | 0.6 s |
| 截止位置 | — | — | 194.2 Zmm | 600 Zmm |
| 截断速度( m/s) | — | — | 3.50 | 6 |
| 输入峰值q(千帕) | 459.8 | 437.7 | 4.81 % | 12 % |
| 输入峰值正则- g | 6.31 | 6.04 | 0.28 g | 1.0 g |
| 影响时间(s) | 400.225 | 400.985 | 0.76 | 1.0 s |
| 影响状况 | — | — | 543.9 Zmm | 1500 Zmm |
| 撞击速度(米/秒) | 892.2 | 814.3 | 8.73 % | 12 % |
| 截断时的推进剂(公斤) | 300 652 | 301 105 | 453 | 1200 |

撞击速度检查是相对的(%),与E8-05 上升段碎片影响先例相匹配;终端潜水速度是密度敏感.

### 错误预算拆分

* **frame** — declared zero (same frame, epoch, rotation rate).
* **控制器** – 宣告为零(同名作者`_Mission`脚本).
* **numerical** — twin at dt/2: ≈0 on q, 0.08 m/s on impact speed, 116 m on
  撞击位置。
* **gravity** — twin without J2: 0.31 s impact time, 111 m position.
* **atmosphere (form)** — twin re-run on the main model's `orbitopt.entry`
  density profile (a labelled perturbation experiment): **5.1 %** of the
  4.8 % peak-q delta and **8.7 %** of the 8.7 % impact-speed delta — i.e.
  这两个密度剖面图说明*整个*在这些差异中的差异。
  检查;门面残基为**0.0**.

## 真正的判决

The main model itself is **not closed** at this handoff: `plan_boostback_entry`
returns `CORRIDOR_MISSED` + `ENTRY_Q_LIMIT_EXCEEDED` (corridor reached with a
2 246 m miss at 1 067.5 m/s), consistent with the E10-01 VEHICLE_CAPABILITY
finding. The landing burn is **unplannable** from the real crossed handoff
(`fly_landing_burn` probe: `LANDING_INFEASIBLE`; closed-loop touchdown would
be 740 m/s vertical), so the terminal comparison is the ballistic impact —
and the report says so. The independent re-flight certifies the **chain
physics**, not a closed design; the corridor-start landing check remains
`design_input` and is superseded for full-chain purposes by this report.
All nine checks carry `independent_model` evidence.

## 重现

```bash
python benchmarks/bench_e10_07_full_recovery_reflight.py \
    --out docs/benchmarks/e10_07_full_recovery_reflight.json   # ~6 min
PYTHONPATH=src python -m pytest tests/test_e10_07_full_recovery_reflight.py -q
```
