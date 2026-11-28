# E7-04故障驱动优化:配对复飞报告

Knobs: lateral cutoff gate `0.8` m/s (referee limit 1.0 m/s), E3-01 command-side TVC lag compensation ON in the fix arms, corridor aim margin `0.0` m (corridor radius 8000.0 m, 0 = off). Seeds [1, 2, 3, 4], 42/seed, 168 pooled -- E3-03's exact dispersions and declared actuator (20 deg/s + 0.2 s lag), the identical [seed, idx] draw streams.

_Pairing notes: pairing validity: a gate-only re-fly of the FL-305 objective reproduced the recorded fl305_baseline histogram exactly (45/168, every reason count), so FL-305-mode recorded rows are bit-valid baselines on current HEAD; the JOINT-mode recorded rows are NOT (E7-01 feasibility semantics landed after them -- a plain re-fly rejects with JOINT_CHANCE_VIOLATED, a reason the recorded report never contains), hence the fresh e7_04_joint_base; the gate alone re-flew baseline verdicts unchanged (no-harm/no-gain tail safety), so the guidance fix rides on E3-01's lag compensation; the aim margin trades INSUFFICIENT_PROPELLANT_RESERVE down for LANDING_LATERAL_SPEED_EXCEEDED up and ships at 0 here_

| 手臂 | 着陆 | 直方图 |
|---|---|---|
| `e7_04_guidance_fix` | 49/168 | {'CORRIDOR_MISSED': 5, 'INSUFFICIENT_PROPELLANT_RESERVE': 38, 'LANDED': 49, 'LANDING_LATERAL_SPEED_EXCEEDED': 18, 'LANDING_POSITION_OUT_OF_TOLERANCE': 41, 'LANDING_PROPELLANT_RESERVE_LOW': 17} |
| `e7_04_joint_base` | 35/168 | {'CORRIDOR_MISSED': 22, 'INSUFFICIENT_PROPELLANT_RESERVE': 12, 'JOINT_CHANCE_VIOLATED': 99, 'LANDED': 35} |
| `e7_04_joint_fix` | 35/168 | {'CORRIDOR_MISSED': 22, 'INSUFFICIENT_PROPELLANT_RESERVE': 12, 'JOINT_CHANCE_VIOLATED': 99, 'LANDED': 35} |

## Paired 比较(每个类别Holm修正的McNemar)

### `e7_04_guidance_fix`对录制的`fl305_baseline`

着陆对:准确的McNemar p = 0.125(不显著),不和谐者4 / 0(只/b-),原始率0.292 与基线0.268.

| 类别 | 基线 | 候选人 | Δ | (单位:千美元) | 原始页 | 霍尔姆p | 判决 |
|---|---|---|---|---|---|---|---|
| `CORRIDOR_MISSED` | 5 | 5 | +0 | 0/0 | 1 | 1 | 无重大变化 |
| `INSUFFICIENT_PROPELLANT_RESERVE` | 38 | 38 | +0 | 0/0 | 1 | 1 | 无重大变化 |
| `LANDING_LATERAL_SPEED_EXCEEDED` | 24 | 18 | -6 | 7/1 | 0.07031 | 0.3516 | 无重大变化 |
| `LANDING_POSITION_OUT_OF_TOLERANCE` | 42 | 41 | -1 | 1/0 | 1 | 1 | 无重大变化 |
| `LANDING_PROPELLANT_RESERVE_LOW` | 14 | 17 | +3 | 0/3 | 0.25 | 1 | 无重大变化 |

`initial_state` sensitivity: |rho| 0.403 -> 0.384 (-0.020), rank position 1 -> 1 (ranking ['initial_state', 'thrust', 'drag', 'density', 'isp']), low->high tail failure-gap shift -0.007.

判决:**目标类别计数下降(文学条款;不在此家族规模下Holm认证),在更正下没有非目标回归**;计数:LANDING_LATERAL_SPEED_EXCEEDED(粗非目标增加:LANDING_PROPELLANT_RESERVE_LOW).

### `e7_04_joint_fix`对新飞`e7_04_joint_base`

着陆对:准确的McNemar p = 1(不显著),不和谐者0 / 0(只/b-),原始率0.208 与基线0.208.

| 类别 | 基线 | 候选人 | Δ | (单位:千美元) | 原始页 | 霍尔姆p | 判决 |
|---|---|---|---|---|---|---|---|
| `CORRIDOR_MISSED` | 22 | 22 | +0 | 0/0 | 1 | 1 | 无重大变化 |
| `INSUFFICIENT_PROPELLANT_RESERVE` | 12 | 12 | +0 | 0/0 | 1 | 1 | 无重大变化 |
| `JOINT_CHANCE_VIOLATED` | 99 | 99 | +0 | 0/0 | 1 | 1 | 无重大变化 |

`initial_state` sensitivity: |rho| 0.391 -> 0.391 (+0.000), rank position 1 -> 1 (ranking ['initial_state', 'thrust', 'drag', 'density', 'isp']), low->high tail failure-gap shift +0.000.

判决:**未减少目标类别数**.

## 记录报告的证据(零重新飞行)

### `e3_01_declared_chain` vs 记录`fl305_baseline` - 单是延迟补偿,FL-305目标(记录,零重飞).

着陆对:准确的McNemar p = 0.125(不显著),不和谐者4 / 0(只/b-),原始率0.292 与基线0.268.

| 类别 | 基线 | 候选人 | Δ | (单位:千美元) | 原始页 | 霍尔姆p | 判决 |
|---|---|---|---|---|---|---|---|
| `CORRIDOR_MISSED` | 5 | 5 | +0 | 0/0 | 1 | 1 | 无重大变化 |
| `INSUFFICIENT_PROPELLANT_RESERVE` | 38 | 38 | +0 | 0/0 | 1 | 1 | 无重大变化 |
| `LANDING_LATERAL_SPEED_EXCEEDED` | 24 | 18 | -6 | 7/1 | 0.07031 | 0.3516 | 无重大变化 |
| `LANDING_POSITION_OUT_OF_TOLERANCE` | 42 | 41 | -1 | 1/0 | 1 | 1 | 无重大变化 |
| `LANDING_PROPELLANT_RESERVE_LOW` | 14 | 17 | +3 | 0/3 | 0.25 | 1 | 无重大变化 |

`initial_state` sensitivity: |rho| 0.403 -> 0.384 (-0.020), rank position 1 -> 1 (ranking ['initial_state', 'thrust', 'drag', 'density', 'isp']), low->high tail failure-gap shift -0.007.

判决:**目标类别计数下降(文学条款;不在此家族规模下Holm认证),在更正下没有非目标回归**;计数:LANDING_LATERAL_SPEED_EXCEEDED(粗非目标增加:LANDING_PROPELLANT_RESERVE_LOW).

### `e3_02_joint_chance` vs 记录的`fl305_baseline` - E3-02 返回此任务必须撤销(已记录上下文)

着陆对:精确的McNemar p = 0.01921(Holm-sublic),不和者15 / 4(只/b-只),生率0.333 与基线0.268.

| 类别 | 基线 | 候选人 | Δ | (单位:千美元) | 原始页 | 霍尔姆p | 判决 |
|---|---|---|---|---|---|---|---|
| `CORRIDOR_MISSED` | 5 | 22 | +17 | 1/18 | 7.629e-05 | 0.0003052 | 倒退 |
| `INSUFFICIENT_PROPELLANT_RESERVE` | 38 | 12 | -26 | 30/4 | 6.165e-06 | 3.082e-05 | 改进 |
| `LANDING_LATERAL_SPEED_EXCEEDED` | 24 | 32 | +8 | 5/13 | 0.09625 | 0.2888 | 无重大变化 |
| `LANDING_POSITION_OUT_OF_TOLERANCE` | 42 | 34 | -8 | 20/12 | 0.2153 | 0.4307 | 无重大变化 |
| `LANDING_PROPELLANT_RESERVE_LOW` | 14 | 12 | -2 | 3/1 | 0.625 | 0.625 | 无重大变化 |

`initial_state` sensitivity: |rho| 0.403 -> 0.474 (+0.070), rank position 1 -> 1 (ranking ['initial_state', 'drag', 'thrust', 'isp', 'density']), low->high tail failure-gap shift +0.115.

判决:**未减少目标类别数**.
