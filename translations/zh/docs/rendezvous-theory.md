# 交际理论

四层,每层依其上层分级:线性规划模型,非线性对联传播器,安全约束集,以及不同物理堆叠的独立重飞.

## LVLH的相对动议

所有相对状态均以目标LVLH(地方垂直,地方水平)框架表示,框架以酋长为中心:x沿射线,y沿-跟踪速度(V- bar),z完成右手设定. 离异是相对状态的规范。 框架公约及其后果见[ADR relative-0001](adr/relative-0001-lvlh-relative-frames-cw-and-pair-propagation.md)。

## CW规划模型及其测量的信封

Clohessy – Wiltshire 线性化相对运动用于 **仅规划** – 绝不用于接受. 伴生基准([study write-up](benchmarks/e12_02_cw_validity.md),[raw data](https://github.com/zangjiucheng/orbitopt/blob/main/docs/benchmarks/e12_02_cw_validity.json))扫射10 m...100 km对齐对齐对齐的传播器. 错误 :

* grows **quadratically with separation** — absolute error ≈ 10⁻⁴·ρ² per
  轨道:圆形车长(0.01米,10米,1.05米,100米,105米)
  1 km) — so the *relative* error is ~1 % per orbit only near ρ ≈ 100 m
  (b) 采用公尺; 没有“1%在任何尺度上”;
* grows ~t²;
* is dwarfed by any chief eccentricity ≥ 0.01 (129 % of a 100 m
  在一个轨道上分离)。

The gate `tests/test_e12_02_cw_validity.py` pins the planning envelope:
circular chief, ρ ≤ 1 km, ≤ 1 orbit, error < 25 % of separation. Any
verdict runs on `propagate_pair`, not on CW.

## 接受轨道传播:`propagate_pair`

The acceptance model propagates chief and deputy together — two-body plus
J2, fixed-step RK4, chief-centred LVLH output — so J2 differential effects
appear without a linearisation assumption. The plan and its verification
share one model, one frame (GCRS) and the exact packaged Earth μ
(3.9860044150×10¹⁴ m³/s²), enforced at the config boundary
([ADR rendezvous-0001](adr/rendezvous-0001-two-impulse-j2-targeted-lambert.md)).

## 任务几何

A coplanar chaser sits 10 km below the target so the orbit-rate drift closes
the initial 20 km separation until the chaser crosses the departure gate
(8 km aft, on V-bar). There the planner solves a two-pulse Lambert transfer,
J2-targeted with a Newton correction on burn 1, that arrives **hovering** at
the hold point — 1 km V-bar-aft — with zero relative velocity. Δv budget and
both time windows are fixed in the config before any optimisation runs
([mission configuration](rendezvous-mission-config.md)).

## 整体- 轨迹安全

Safety constraints gate the **entire** trajectory, not just the terminal
approach
([ADR rendezvous-0002](adr/rendezvous-0002-safety-constraints-whole-trajectory.md)):

* a 500 mi 围绕每个样本的目标保持球体;
* 一条横跨轨道的通道(±100),禁止前进
  a. 在终端锥内游览;
* a. 距离依赖的相对速度天花板;
* 被动应急检查:从t 0出发的无推力海岸、大门和
  守住点必须保持清醒

检查在密集的网格上运行(2 s),燃烧的时代总是取样,判决报告每个检查边距.

## 独立验证和稳健性

A TudatPy two-spacecraft re-flight
([ADR rendezvous-0003](adr/rendezvous-0003-independent-refly-tudatpy.md))
re-flies the pair in a different physics stack, graded only when maneuver
digests match the sealed plan. A dispersion campaign
([ADR rendezvous-0005](adr/rendezvous-0005-dispersion-campaign-not-a-safety-factor.md))
prices the locked design under stress. The Mission Studio replay
([ADR rendezvous-0004](adr/rendezvous-0004-mission-studio-flightlab-replay.md))
re-propagates the sealed plan locally for inspection — it is a replay, not
an independent check. Recorded numbers, with nominal, independent and robust
results kept separate: [verification report](rendezvous-verification-report.md).
