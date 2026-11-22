# E11-01 — grid-fin aerodynamic authority (task #308)

Epic #307, base commit `2e20641`. The DC8 re-entry was, until now, purely
ballistic: every RTLS vehicle table is a "blunt-cylinder-class table with grid
fins stowed" and the propagator had no lift, angle-of-attack or bank channel
(`recovery/propagator.py:48`, `ballistic_return`). This task adds a two-state
(stowed/deployed) grid-fin aerodynamic model with an actuator, a shared
angle-of-attack/bank steering law wired into FL-301/FL-302 as
`entry_aero={"mode": …}`, and an independent twin implementation, then asks
the acceptance question: does unpowered entry close the corridor on the DC8
with `entry_burns = 0`? It does not, and this file quantifies why. The machine
reading is [`e11_01_grid_fin_authority.json`](e11_01_grid_fin_authority.json);
reproduce with `PYTHONPATH=src python benchmarks/bench_e11_01_grid_fin_authority.py --out …`
(≈3 min, CPU only).

## The model

Four fins, X-configured, on the `starship_class_rtls_dc8` vehicle
(`grid_fins` block, provenance `public_estimate`). Stowed: the table is
untouched (bit-for-bit identical to the pre-E11 behaviour when `entry_aero`
is absent). Deployed, per step: body normal force
`cn_body·sin(α)`, fin normal force `cl_α,f·α·(1 − 1/trim effectiveness)`
clipped at stall, lift `L = q̄·S_ref·(cn + c_δ)·cos α`, drag increment
`ΔCD = CD0_add(M) + cn·sin α + k·δ²`, with δ the mean |deflection| the
trim/roll differential demands. The actuator slews deploy/stow with 5 kPa
hysteresis above 70 km, α instantaneously (within authority), bank at 30 °/s,
δ at 20 °/s; every fin saturates within ±40°. The steering law (schedule or
closed-loop heading-error steer) is identical in main model and twin —
guidance is part of the authored script, the E10-07 rule.

DESIGN ASSUMPTIONS (echoed in the JSON; none of these are public for this
class): fin area 4 × 4 m² (±25%); fin normal-force slope table (±30%,
perforated-plate/Ackeret scaling); trim effectiveness 1.8 / static margin
0.08 L (±50%); δ_max 40° and 20 °/s (±50%); bank rate 30 °/s (±50%).

## Part A — controlled cross-range/down-range envelope

Coast from the committed E10-07 separation handoff (`handoff.state`,
`t = 125 s`), steering at a constant α (schedule), bank fixed, no engine,
to the 3 km corridor plane. Down-range from pad at crossing ≈ 207 km for
every case; shifts are relative to the finless coast (which crosses at
1398 m/s and 902 kPa — unsurvivable, the E10-01 result):

| case | v at 3 km | peak q | peak lateral accel | Δcross (N) | Δdown (E) |
|---|---|---|---|---|---|
| α30 β0 | 719 m/s | 290 kPa | 18.5 m/s² | +2 111 m | +1 554 m |
| α45 β0 | 610 m/s | 249 kPa | 18.4 m/s² | +2 426 m | +1 787 m |
| α60 β0 | 547 m/s | 222 kPa | 14.2 m/s² | +1 838 m | +1 357 m |
| α45 β+45° | 638 m/s | 254 kPa | 18.7 m/s² | +151 m | +2 700 m |
| α45 β−45° | 638 m/s | 254 kPa | 18.7 m/s² | +2 613 m | −660 m |
| α60 β+45° | 564 m/s | 225 kPa | 14.4 m/s² | 0 m | +2 184 m |
| α60 β−45° | 564 m/s | 225 kPa | 14.4 m/s² | +2 074 m | −646 m |

Readings: peak lateral authority is **≈ 1.9 g** (18.8 m/s²) at max fin lift;
banking ±45° moves the ±2.4 km north-south envelope across ≈ 2.6 km of
cross-range at fixed speed, and every case now crosses under the 250 kPa
entry limit (peak q 222–295 vs 902 kPa finless). The response is frame-
asymmetric — the bank sign interacts with the heading error to the pad, and
at α30 the vehicle is still at Mach > 2.5 where the trim balance changes the
effective lift direction; the ±2–3 km envelope numbers are therefore
case-dependent, not a symmetric circle. Full matrix in the JSON (`envelope`).

## Part B — two-burn closure trial

Full plan (`plan_boostback_entry`, committed E10-07 keep recipe: script
[21.2, −2.0, 1, 0, 0], `entry_burns=0`, closed-loop landing need, unrelaxed
limits: q ≤ 250 kPa, g ≤ 12, 3 000 m / ≤ 350 m/s / ≤ 15° corridor at 8 km
radius, 2 500 kg landing reserve):

| schedule | verdict | v at 3 km | tilt | miss | peak q |
|---|---|---|---|---|---|
| ballistic (reference) | CORRIDOR_MISSED | 1067.5 | 13.9° | 2 246 m | 460 kPa |
| α40 const | CORRIDOR_MISSED | 434.8 | 30.1° | 292 m | 140 kPa |
| α50 const | CORRIDOR_MISSED | 391.8 | 29.3° | 251 m | 128 kPa |
| α60 const | CORRIDOR_MISSED | 361.5 | 26.3° | 169 m | 118 kPa |
| α40→α20 @18 km | CORRIDOR_MISSED | 527.4 | 25.1° | 312 m | 161 kPa |
| α40→α60 @12 km | CORRIDOR_MISSED | 375.8 | 27.5° | 157→143 m | 134 kPa |
| α50→α60 @10 km | CORRIDOR_MISSED | 369.6 | 27.5° | 157 m | 128 kPa |
| α40→dive @20 km | CORRIDOR_MISSED | 1000.5 | 15.2° | 1 476 m | 372 kPa |
| α40→α60@12k→dive@6k | CORRIDOR_MISSED | 576.5 | 22.6° | 97 m | 134 kPa |

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

## Part C — independent twin

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

## Consequence for the epic

E11-02 (guidance) inherits a plant with a real bank channel and a 1.9 g
lateral envelope, and the honest constraint set: within DC8-class fin
authority the corridor tilt limit is unreachable unpowered, so the epic's
closure claim for DC8 must either keep the entry burn (the E10-07 two-burn
baseline), record the vehicle-capability finding as the E11 result, or
re-fund larger fins as a vehicle change with new DESIGN ASSUMPTIONS.
