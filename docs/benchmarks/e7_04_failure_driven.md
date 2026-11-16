# E7-04 failure-driven optimization: paired re-fly report

Knobs: lateral cutoff gate `0.8` m/s (referee limit 1.0 m/s), E3-01 command-side TVC lag compensation ON in the fix arms, corridor aim margin `0.0` m (corridor radius 8000.0 m, 0 = off). Seeds [1, 2, 3, 4], 42/seed, 168 pooled -- E3-03's exact dispersions and declared actuator (20 deg/s + 0.2 s lag), the identical [seed, idx] draw streams.

_Pairing notes: pairing validity: a gate-only re-fly of the FL-305 objective reproduced the recorded fl305_baseline histogram exactly (45/168, every reason count), so FL-305-mode recorded rows are bit-valid baselines on current HEAD; the JOINT-mode recorded rows are NOT (E7-01 feasibility semantics landed after them -- a plain re-fly rejects with JOINT_CHANCE_VIOLATED, a reason the recorded report never contains), hence the fresh e7_04_joint_base; the gate alone re-flew baseline verdicts unchanged (no-harm/no-gain tail safety), so the guidance fix rides on E3-01's lag compensation; the aim margin trades INSUFFICIENT_PROPELLANT_RESERVE down for LANDING_LATERAL_SPEED_EXCEEDED up and ships at 0 here_

| arm | landed | reason histogram |
|---|---|---|
| `e7_04_guidance_fix` | 49/168 | {'CORRIDOR_MISSED': 5, 'INSUFFICIENT_PROPELLANT_RESERVE': 38, 'LANDED': 49, 'LANDING_LATERAL_SPEED_EXCEEDED': 18, 'LANDING_POSITION_OUT_OF_TOLERANCE': 41, 'LANDING_PROPELLANT_RESERVE_LOW': 17} |
| `e7_04_joint_base` | 35/168 | {'CORRIDOR_MISSED': 22, 'INSUFFICIENT_PROPELLANT_RESERVE': 12, 'JOINT_CHANCE_VIOLATED': 99, 'LANDED': 35} |
| `e7_04_joint_fix` | 35/168 | {'CORRIDOR_MISSED': 22, 'INSUFFICIENT_PROPELLANT_RESERVE': 12, 'JOINT_CHANCE_VIOLATED': 99, 'LANDED': 35} |

## Paired comparisons (Holm-corrected per-category McNemar)

### `e7_04_guidance_fix` vs recorded `fl305_baseline`

landed pair: exact McNemar p = 0.125 (not significant), discordants 4/0 (a-only/b-only), raw rate 0.292 vs baseline 0.268.

| category | baseline | candidate | Δ | b/c | raw p | Holm p | verdict |
|---|---|---|---|---|---|---|---|
| `CORRIDOR_MISSED` | 5 | 5 | +0 | 0/0 | 1 | 1 | no significant change |
| `INSUFFICIENT_PROPELLANT_RESERVE` | 38 | 38 | +0 | 0/0 | 1 | 1 | no significant change |
| `LANDING_LATERAL_SPEED_EXCEEDED` | 24 | 18 | -6 | 7/1 | 0.07031 | 0.3516 | no significant change |
| `LANDING_POSITION_OUT_OF_TOLERANCE` | 42 | 41 | -1 | 1/0 | 1 | 1 | no significant change |
| `LANDING_PROPELLANT_RESERVE_LOW` | 14 | 17 | +3 | 0/3 | 0.25 | 1 | no significant change |

`initial_state` sensitivity: |rho| 0.403 -> 0.384 (-0.020), rank position 1 -> 1 (ranking ['initial_state', 'thrust', 'drag', 'density', 'isp']), low->high tail failure-gap shift -0.007.

Verdict: **target category counts fell (literal clause; not Holm-certified at this family size), no non-target regression under correction**; counts fell: LANDING_LATERAL_SPEED_EXCEEDED (raw non-target increases: LANDING_PROPELLANT_RESERVE_LOW).

### `e7_04_joint_fix` vs fresh re-fly `e7_04_joint_base`

landed pair: exact McNemar p = 1 (not significant), discordants 0/0 (a-only/b-only), raw rate 0.208 vs baseline 0.208.

| category | baseline | candidate | Δ | b/c | raw p | Holm p | verdict |
|---|---|---|---|---|---|---|---|
| `CORRIDOR_MISSED` | 22 | 22 | +0 | 0/0 | 1 | 1 | no significant change |
| `INSUFFICIENT_PROPELLANT_RESERVE` | 12 | 12 | +0 | 0/0 | 1 | 1 | no significant change |
| `JOINT_CHANCE_VIOLATED` | 99 | 99 | +0 | 0/0 | 1 | 1 | no significant change |

`initial_state` sensitivity: |rho| 0.391 -> 0.391 (+0.000), rank position 1 -> 1 (ranking ['initial_state', 'thrust', 'drag', 'density', 'isp']), low->high tail failure-gap shift +0.000.

Verdict: **no target category count decreased**.

## Evidence from the recorded report (zero re-flight)

### `e3_01_declared_chain` vs recorded `fl305_baseline` -- lag compensation alone, FL-305 objective (recorded, zero re-flight)

landed pair: exact McNemar p = 0.125 (not significant), discordants 4/0 (a-only/b-only), raw rate 0.292 vs baseline 0.268.

| category | baseline | candidate | Δ | b/c | raw p | Holm p | verdict |
|---|---|---|---|---|---|---|---|
| `CORRIDOR_MISSED` | 5 | 5 | +0 | 0/0 | 1 | 1 | no significant change |
| `INSUFFICIENT_PROPELLANT_RESERVE` | 38 | 38 | +0 | 0/0 | 1 | 1 | no significant change |
| `LANDING_LATERAL_SPEED_EXCEEDED` | 24 | 18 | -6 | 7/1 | 0.07031 | 0.3516 | no significant change |
| `LANDING_POSITION_OUT_OF_TOLERANCE` | 42 | 41 | -1 | 1/0 | 1 | 1 | no significant change |
| `LANDING_PROPELLANT_RESERVE_LOW` | 14 | 17 | +3 | 0/3 | 0.25 | 1 | no significant change |

`initial_state` sensitivity: |rho| 0.403 -> 0.384 (-0.020), rank position 1 -> 1 (ranking ['initial_state', 'thrust', 'drag', 'density', 'isp']), low->high tail failure-gap shift -0.007.

Verdict: **target category counts fell (literal clause; not Holm-certified at this family size), no non-target regression under correction**; counts fell: LANDING_LATERAL_SPEED_EXCEEDED (raw non-target increases: LANDING_PROPELLANT_RESERVE_LOW).

### `e3_02_joint_chance` vs recorded `fl305_baseline` -- the E3-02 regression this task must undo (recorded context)

landed pair: exact McNemar p = 0.01921 (Holm-significant), discordants 15/4 (a-only/b-only), raw rate 0.333 vs baseline 0.268.

| category | baseline | candidate | Δ | b/c | raw p | Holm p | verdict |
|---|---|---|---|---|---|---|---|
| `CORRIDOR_MISSED` | 5 | 22 | +17 | 1/18 | 7.629e-05 | 0.0003052 | regressed |
| `INSUFFICIENT_PROPELLANT_RESERVE` | 38 | 12 | -26 | 30/4 | 6.165e-06 | 3.082e-05 | improved |
| `LANDING_LATERAL_SPEED_EXCEEDED` | 24 | 32 | +8 | 5/13 | 0.09625 | 0.2888 | no significant change |
| `LANDING_POSITION_OUT_OF_TOLERANCE` | 42 | 34 | -8 | 20/12 | 0.2153 | 0.4307 | no significant change |
| `LANDING_PROPELLANT_RESERVE_LOW` | 14 | 12 | -2 | 3/1 | 0.625 | 0.625 | no significant change |

`initial_state` sensitivity: |rho| 0.403 -> 0.474 (+0.070), rank position 1 -> 1 (ranking ['initial_state', 'drag', 'thrust', 'isp', 'density']), low->high tail failure-gap shift +0.115.

Verdict: **no target category count decreased**.

