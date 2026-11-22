# E10-05 recovery planner robust optimization (DC7)

Provenance in the JSON (`provenance`, `seed_manifest`). Why did per-sample replanning score 3/36 where the fixed script scored 11/36, and can one scenario-robustly-planned script beat both? Paired on the sealed DC7 hold-out.

## 1. Verdict

| arm | landed | cluster CI95 | Wilson CI95 |
|---|---|---|---|
| `fixed` | 11/36 | [0.25, 0.3333] | [0.18, 0.4686] |
| `replan40` | 3/36 | [0.0, 0.1667] | [0.0287, 0.2183] |
| `robust` | 11/36 | [0.25, 0.3333] | [0.18, 0.4686] |

Paired, Holm-corrected (family of 3):

| pair | discordant | diff | p (Holm) |
|---|---|---|---|
| fixed vs replan40 | 9-1 | 0.2222 | 0.064453125 |
| fixed vs robust | 0-0 | 0.0 | 1.0 |
| replan40 vs robust | 1-9 | -0.2222 | 0.064453125 |

## 2. Why replan lost (diagnosis)

2x2 (fixed x replan40): both 2, fixed-only 9, replan-only 1, neither 24.

Reason mix per arm:

- `fixed`: `CORRIDOR_MISSED` 25
- `replan40`: `CORRIDOR_MISSED` 33
- `robust`: `CORRIDOR_MISSED` 25

Budget ladder on the diagnostic subset (fixed-landed ∪ replan-landed ∪ first 6 both-failed controls, n = 18):

| max_fevals | landed | fixed-landed kept | both-failed rescued |
|---|---|---|---|
| 10 | 2 | 1 | 1 |
| 120 | 2 | 1 | 1 |

## 3. Robust arm

(1+8)-ES trust-region stages, trained on seeds [401, 402] x 5 (+ nominal), 66 candidates total:

| stage | sigma | candidates | best landed (train) | score |
|---|---|---|---|---|
| 0 | 4% | 33 | 4/11 | 0.6664 |
| 1 | 10% | 33 | 4/11 | 0.6664 |

Selected script = stage 0 (sigma 4.0%); training landed 4/11, hold-out landed 11/36. A tie keeps the incumbent (smaller region): no local perturbation beat the recorded script on training.

Script movement from the recorded vector (robust arm, per parameter): boostback delay [s] +0.00, boostback aim past pad [km] +0.00, boostback throttle +0.00, boostback pitch [deg] +0.00, boostback yaw trim [deg] +0.00, entry trigger alt [m] +0.00, entry cutoff speed [m/s] +0.00, entry throttle +0.00

## 4. Seed manifest and split

See `seed_manifest` in the JSON: hold-out [8, 9, 10] x 12 (the sealed DC7 grid), training [401, 402] x 5, recorded search seed 287, ES seed 41171. Training and hold-out seeds are disjoint by construction (asserted in the bench).

