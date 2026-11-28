# E11-01 — grid-fin aerodynamic authority (task #308)

Epic # 307,基地承诺`2e20641`. DC8型机车的复进一直纯粹是弹道:每台RTLS型机车台是"带网鳍的弹缸级台",宣传机没有升降机,攻击角度或银行信道(`recovery/propagator.py:48`,`ballistic_return`). 这个任务增加了两个状态(赋予/部署)的网格式气动模型,一个共有攻击角/银行制导定律连接到FL-301/FL-302作为`entry_aero={"mode": …}`,以及一个独立的双子执行,然后问接受问题:无动力进入DC8的走廊与`entry_burns = 0`是否关闭? 它没有,而这个文件 量化了原因。 机器读数为[`e11_01_grid_fin_authority.json`](https://github.com/zangjiucheng/orbitopt/blob/main/docs/benchmarks/e11_01_grid_fin_authority.json);与`PYTHONPATH=src python benchmarks/bench_e11_01_grid_fin_authority.py --out …`(3 min,CPU只)复制.

## 模型

X-配置的四条鳍,搭载`starship_class_rtls_dc8`车(`grid_fins`区块,出处`public_estimate`). 堆积:表未触动(在`entry_aero`缺席时,与前E11行为完全相同). 按步骤部署:身体正常力`cn_body·sin(α)`,鳍正常力`cl_α,f·α·(1 − 1/trim effectiveness)`在停机坪剪接,抬起`L = q̄·S_ref·(cn + c_δ)·cos α`,拖动增量`ΔCD = CD0_add(M) + cn·sin α + k·δ²`,平均|偏移|,调和/滚动差要求. 引爆器将5 kPa 激流在70 km上方部署/安装,即时(在授权范围内)在30 °/s,20 °/s;每条鳍饱和度在±40 °。 指导定律(schedule或closed-loop head-error roil)在主模型和双双——指导是作者脚本的一部分,E10-07规则.

DESIGN ASSUMPTIONS (echoed in the JSON; none of these are public for this
class): fin area 4 × 4 m² (±25%); fin normal-force slope table (±30%,
perforated-plate/Ackeret scaling); trim effectiveness 1.8 / static margin
0.08 L (±50%); δ_max 40° and 20 °/s (±50%); bank rate 30 °/s (±50%).

## A部分——可控跨距/下距信封

Coast from the committed E10-07 separation handoff (`handoff.state`,
`t = 125 s`), steering at a constant α (schedule), bank fixed, no engine,
to the 3 km corridor plane. Down-range from pad at crossing ≈ 207 km for
every case; shifts are relative to the finless coast (which crosses at
1398 m/s and 902 kPa — unsurvivable, the E10-01 result):

| 大小写 | 五. 3公里 | 峰值q | 峰值平方圆 | 横跨( N) | Δ下 (英) |
|---|---|---|---|---|---|
| α30 β0 | 719 m/s | 290 kPa | 18.5 m/s² | +2 111 m | +1 554 m |
| α45 β0 | 610 m/s | 249 kPa | 18.4 m/s² | +2 426 m | +1 787 m |
| α60 β0 | 547 m/s | 222 kPa | 14.2 m/s² | +1 838 m | +1 357 m |
| α45 β+45° | 638 m/s | 254 kPa | 18.7 m/s² | +151 Zmm | +2 700 m |
| α45 β−45° | 638 m/s | 254 kPa | 18.7 m/s² | +2 613 m | −660 m |
| α60 β+45° | 564 m/s | 225 kPa | 14.4 m/s² | 0 Zmm | +2 184 m |
| α60 β−45° | 564 m/s | 225 kPa | 14.4 m/s² | +2 074 m | −646 m |

Readings: peak lateral authority is **≈ 1.9 g** (18.8 m/s²) at max fin lift;
banking ±45° moves the ±2.4 km north-south envelope across ≈ 2.6 km of
cross-range at fixed speed, and every case now crosses under the 250 kPa
entry limit (peak q 222–295 vs 902 kPa finless). The response is frame-
asymmetric — the bank sign interacts with the heading error to the pad, and
at α30 the vehicle is still at Mach > 2.5 where the trim balance changes the
effective lift direction; the ±2–3 km envelope numbers are therefore
case-dependent, not a symmetric circle. Full matrix in the JSON (`envelope`).

## B部分——两刑终结审判

Full plan (`plan_boostback_entry`, committed E10-07 keep recipe: script
[21.2, −2.0, 1, 0, 0], `entry_burns=0`, closed-loop landing need, unrelaxed
limits: q ≤ 250 kPa, g ≤ 12, 3 000 m / ≤ 350 m/s / ≤ 15° corridor at 8 km
radius, 2 500 kg landing reserve):

| 时间表 | 判决 | 五. 3公里 | 倾斜 | 错过 | 峰值q |
|---|---|---|---|---|---|
| 弹道( 参考) | CORRIDOR_MISSED | 1067.5 | 13.9° | 2 246 m | 460 kPa |
| α 40 康斯特 | CORRIDOR_MISSED | 434.8 | 30.1° | 292 Zmm | 140 kPa |
| α 50 康斯特 | CORRIDOR_MISSED | 391.8 | 29.3° | 251 Zmm | 128 kPa |
| α 60 康斯特 | CORRIDOR_MISSED | 361.5 | 26.3° | 169 Zmm | 118 kPa |
| α40→α20 @18 km | CORRIDOR_MISSED | 527.4 | 25.1° | 312 Zmm | 161 kPa |
| α40→α60 @12 km | CORRIDOR_MISSED | 375.8 | 27.5° | 157→143 m | 134 kPa |
| α50→α60 @10 km | CORRIDOR_MISSED | 369.6 | 27.5° | 157 Zmm | 128 kPa |
| α 40 → 潜水 @ 20公里 | CORRIDOR_MISSED | 1000.5 | 15.2° | 1 476 m | 372 kPa |
| α40→α60@12k→dive@6k | CORRIDOR_MISSED | 576.5 | 22.6° | 97 Zmm | 134 kPa |

**Verdict: NOT CLOSED.** Fins fix every entry-environment violation — peak q
falls from 460 kPa to 118–161 kPa, miss from 2 246 m to 97–312 m, landing
reserve is met in every fin case — and cut arrival speed from 1 067 m/s to
**361.5 m/s best** (α60), 11.5 m/s short of the 350 ceiling. The residual
is the **flight-path tilt at the corridor plane**: the best fin cases hold
25–30° against a 15° requirement. A multi-start Nelder–Mead search over the
full schedule-gate space (α and gate altitudes, 40 evals) does not find the
corner either, and neither do aim/delay sweeps — the tilt floor is robust.

Why physically: the corridor is crossed at v ≈ 360–435 m/s, where q ≤ 250 kPa
leaves the fin group ≈ 12 % of vehicle normal-force share (trim-dominated)
and L/D ≤ 0.4; rotating a still-decelerating 350–435 m/s trajectory from
≈ −40° to ≤ −15° inside the last ≈ 2 km needs sustained normal load ≥ 2 g of
which the fins supply well under 1 g after trim — gravity rotates the path at
only ≈ 0.5°/s, and diving (α → 0, rows 8–9) buys tilt only at the cost of
speed (577–1 000 m/s). With ±50 %/±30 % DESIGN-ASSUMPTION bands the fin
share moves but not the sign of the shortfall: the 15°/350 m/s corner needs
≈ 2.4 × the normal-force authority of this fin set, i.e. a vehicle change
(bigger fins) or one entry burn — **the DC8 class stays three-burn under the
E5-06 checks as flown; grid fins alone cannot close it**. This is the
acceptance alternative: quantified non-closure, guidance and vehicle both
accounted. (The JSON `best_tilt_deg` = 13.9° is the ballistic row, which
takes a steep-but-unsurvivable entry; among fin cases the best is 22.6°.)

## C部分——独立双胞胎

The twin (`orbitopt.verify.full_recovery_reflight`, `_TwinFins`) re-flies the
same authored script and steering law from the same handoff with its own
actuator integration, own force decomposition and own US76 layer. Worst
deltas over the two schedules: peak q 13.7 %, peak g 0.96 g, speed at 3 km
24.1 % — all inside the declared bands (20 %, 2 g, 40 %,
`fin.entry_peak_*`). The budget split says the discrepancy is the
atmosphere-form line, not the fin model: running the twin with the main
model's density profile shrinks every residual to ≤ 0.7 %, ≤ 0.03 g,
≤ 0.4 % (JSON `budget`). The E10-07 atmosphere-form line (5 % on peak q for
ballistics) is re-confirmed and now measured *with* lift on; it grows toward
3 km because the fin entry spends its terminal phase inside the denser
layers where the two profiles disagree.

## 史诗的剧情

E11-02(指导)继承了一个具有真实的银行通道和1.9 g 横向封套的工厂,诚实的制约套:在DC8-级位权限内,走廊倾斜限是无法到达的,因此史诗对DC8的关闭索赔要么必须保持入口燃烧(E10-07双燃基线),将车辆能力发现记录为E11结果,要么用新的DESIGN ASSUMPTIONS的车辆换钱换大鳍.
