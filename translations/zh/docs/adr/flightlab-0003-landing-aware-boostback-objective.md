# ADR飞行板 -0003:着陆-意识后助目标(二倾预算+燃料比值).

- **Status:** Accepted
- **Date:** 2026-11-12
- **Deciders:** Recovery Guidance & Robustness (Epic #217), R-01 (#233),
  follow-up to FL-305 (#214, ADR flightlab-0002)
- **Code:** `src/orbitopt/flightlab/recovery/boostback.py`
  (`_evaluate`, `plan_boostback_entry`, `DEFAULT_LANDING_DIVERT_BUDGET_M`,
  `DEFAULT_LANDING_RESERVE_MARGIN_KG`, `CorridorReport`),
  `src/orbitopt/flightlab/recovery/montecarlo.py`(样本记录)
- **Evidence:** `docs/benchmarks/r01_boostback_objective.json`
  (`benchmarks/bench_r01_boostback_objective.py`),
  `docs/benchmarks/fl305_recovery_montecarlo.json`,
  `tests/test_flightlab_recovery_boostback.py`

## 背景和问题

ADR 飞行实验室-0002 关闭了着陆燃烧的环路,并重新固定FL-302 温暖的开始,但是它自己的展望将剩下的耦合命名为:*" FL-302的走廊罚仍然几乎盲目,无法在走廊半径下失守. 如果分散的成功率很重要的话,*

The FL-305-baseline Monte Carlo (n = 32, seed 1, default dispersions) says
where that blindness costs: 14/32 land, and 13 of the 18 failures are
propellant-margin failures (`INSUFFICIENT_PROPELLANT_RESERVE` 8,
`LANDING_PROPELLANT_RESERVE_LOW` 5), not corridor failures. Meanwhile the
objective graded the corridor miss against the 8 km radius -- at 1.76 km
that penalty is 0.048, invisible -- and asked for nothing beyond
"landing need + reserve" until a sample violated it outright, while the
closed-loop landing burn (ADR 0002) had measured what it can actually close:
0.35 km diverts land inside every limit, 1.76 km does not.

走廊交接决定着陆的可行性——转移的着陆燃烧必须关闭,它继承的推进剂. R-01两个都移动到Boostback目标中,而不是在下游发现它们作为理性代码.

## 所审议的备选方案

1. **FL-302的硬可行性限制**(拒绝候选人:
   转出或差值失败)。 Nelder-Mead 不受约束; 硬拒绝
   惩罚悬崖让每个简单的顶点 都一暖和的开始
   邻里不可行,接受的计划不再是推出
   比较。 被拒绝。
2. **提高搜索预算**(30 - > 120评价每个样本). 费用
   天平~4×,目标对其购买的不动. 被拒绝
   作为主要杠杆;它随后由这一决定构成。
3. **Soft objective terms at the crossed corridor, defaults as design
   assumptions, disabled state reproducing FL-305 exactly** (adopted).
   没有新的理由代码,没有修改 冻结的可行性检查 -
   术语指导, 检查仍然管理。

## 决定

在`_evaluate`的跨分支处罚中有两个无尺寸的添加剂平方词,两者都由各自的预算规范化(与现行违规条款相同):

- 着陆可行性:`(miss_m / landing_divert_budget_m) ** 2`-
  同样的量度误差,现在也与着陆燃烧可能关闭的东西相对应;
- 燃料储备:`short = need + reserve + landing_reserve_margin_kg - prop`
  (clamped at 0), where FL-305 used margin 0.

Defaults: 1000 m divert budget and 1000 kg margin -- DESIGN ASSUMPTIONs
anchored on the measurements above (the accepted handoff is 0.35 km; the
landing need's momentum balance sees no dispersion allowance, so one tonne
of headroom on a ~15 t recovery burn under 5 % thrust / 3 % Isp / 10 %
density bands). `landing_divert_budget_m=None` and
`landing_reserve_margin_kg=0.0` reproduce the FL-305 objective bit-for-bit
(both terms become exact-0 additions); that switch is what lets the same
conditions be compared before/after. `CorridorReport` now carries the bar
(`landing_divert_budget_m`, `min_landing_reserve_margin_kg`) so a consumer
can check the accepted plan against the bar it was steered to, and the
Monte Carlo records gained `corridor_miss_m` / `corridor_fuel_margin_kg`.

## 结果

名义上(FL-305和R-01温暖开始,包括搜索):接受的交割没有变化 -- -- 351关闭,+1662千克以上需要+储备. FL-305重置的脚本已经坐落在R-01栏内;绿色的E2E没有移动. R-01术语作用于名义假设破裂的地方,即分散.

蒙特卡洛,条件相同(种子1,n = 32,默认分散,FL-305温暖开始,闭环;基线列复制`fl305_recovery_montecarlo.json`准确):

| 理由 | FL-305目标 | R-01目标 |
|---|---|---|
| `LANDED` | 14 | **17** |
| `INSUFFICIENT_PROPELLANT_RESERVE` (FL-302) | 8 | **2** |
| `LANDING_PROPELLANT_RESERVE_LOW` | 5 | 5 |
| `LANDING_POSITION_OUT_OF_TOLERANCE` | 4 | 7 |
| `CORRIDOR_MISSED` (FL-302) | 1 | 1 |
| **success rate** | 0.438 | **0.531** |

走廊移交所有32样本(百分比):

| 交割量度 | FL-305 | R-01 |
|---|---|---|
| 移位分钟 / p 50 | 203 / 925 m | 285 / 911 m |
| 移转 p 90 / 最大 | 1724 / 2692 m | 1879 / 3717 m |
| 燃料边距分钟 | **−1461 kg** | **−448 kg** |
| 燃料差值p 50 | 1079公斤 | **1349 kg** |

The reserve class -- FL-305's dominant failure mode -- collapses from 13 to
7 and the worst handoff shortfall from −1461 kg to −448 kg: the margin term
does exactly its job. The cost is honest and recorded: the 30-evaluation
budget trades divert against propellant where both cannot be had, so the
divert tail grows and 3 samples move into the position-tolerance class
(nominal landed-subset pad-distance p95 nonetheless tightens, 2.18 -> 1.72 m).
Net +3 landings, and every sample is where the reason codes say it is.

## 后果

- FL-302的默认目标为R-01;FL-305仍为文献记录.
  禁用状态,且在基准标注前/基准标注后FL-305(它)
  数字保持历史记录)。
- FL-305-时代的只有Monte Carlo的测试针钉上相同的开关(它)
  测试统计管道,而不是目标。
- 其余的`LANDING_POSITION_OUT_OF_TOLERANCE`样本交给外方
  what the guidance cone closes at 3 km; and 2 samples still hand off
  简称为储备。 重量较大、预算较大或车辆
  缩小是这些人的杠杆,而不是恢复结合。
