# E7-05 out-of-sample reliability

Provenance in the JSON (`provenance`). All statistics deterministic (bootstrap seeds recorded).

## 1. Why the joint arm rejects (E7-04's 99/168, diagnosed)

99 rejections classified (inconsistent rows: 0). Binding constraint: `divert` 51, `fuel` 48.

| binding | severity | n |
|---|---|---|
| `divert` | deep | 20 |
| `divert` | mid | 19 |
| `divert` | near | 12 |
| `fuel` | deep | 31 |
| `fuel` | mid | 11 |
| `fuel` | near | 6 |

Severity split near/mid/deep (<= 0.15 / middle / > 0.5 worst normalized violation): 18 / 30 / 51.

**Landing-probability channel bound on 0 of 99 rejections (alone: 0)** -- the p_land term is not what makes the joint arm infeasible.

Paired against the FL-305 baseline on the identical grid: 20 of 99 rejected worlds DID land under FL-305 (constraint cost 0.202).

Verdict: the 59% rejections are physical, not probabilistic: 51/99 bind on the corridor-divert budget and 48 on the fuel quantile; NOT ONE rejection is bound by the landing-probability channel (the p_land term never decides alone, so a mis-calibrated p_land is not what makes the arm infeasible). The dispersion-band meeting the spec (deep violations) is the majority (51/99), planner-search-rescuable near-threshold rejections the minority (18); and 20/99 rejections are worlds the FL-305 grading DID land -- the measured constraint cost of the joint spec on the training grid.

## 2. The E7-03 headline under cluster-aware statistics

Pooled reference (E7-03): phi = -0.1266, Fisher p = 0.0104 at n = 409 (rows share seeds). Cluster-aware: per-seed 2x2 strata (fit arms summed inside each seed):

- seed 1: n 101, phi -0.027, Fisher p 1
- seed 2: n 102, phi -0.149, Fisher p 0.203
- seed 3: n 95, phi -0.117, Fisher p 0.406
- seed 4: n 111, phi -0.209, Fisher p 0.031

- Mantel-Haenszel (continuity corrected): X2 = 5.68, p = 0.0172
- Fisher-combined exact per-seed: X2 = 11.94, p = 0.1538 (k = 4)
- whole-cluster bootstrap CI for pooled phi: [-0.185, -0.058]

Per-arm MH p in the Holm family:

- `e3_01_declared_chain`: MH p 0.07396 -> Holm 0.2219
- `fl305_baseline`: MH p 0.1605 -> Holm 0.321
- `r01_landing_aware`: MH p 0.6479 -> Holm 0.6479

Reading: the negative association survives stratification (MH p below 0.05, cluster CI excludes zero) but is NOT uniform -- one seed sits at phi = 0 and Fisher-combination (which punishes heterogeneity) does not reach significance. The honest quote is the cluster CI, not the pooled p.

## 3. Hold-out calibration (fresh flights, seeds disjoint from training)

### `a_seeds` -- independent seeds, standard FL-304 bands, training vehicle/scenario/ascent (unseen draw of the training population)

vehicle `synthetic_two_stage.yaml`, seeds [5, 6, 7, 8] x 42 = 168 samples, bands x 1, recipe: FL-302 stock MC plan, joint-pinned.

landed 39/168 (0.232, Wilson [0.175, 0.302], whole-seed CI [0.19047619047619047, 0.27380952380952384]); reasons {'CORRIDOR_MISSED': 23, 'INSUFFICIENT_PROPELLANT_RESERVE': 13, 'JOINT_CHANCE_VIOLATED': 93, 'LANDED': 39}.

Brier 0.2885 vs climatology 0.1783 -> skill -0.6186373790229549 (whole-seed CI [-1.0313991309874524, -0.3268140216806088]), ECE 0.419, MCE 0.668.

| bin | n | mean p | observed | Wilson | seed-cluster CI | verdict |
|---|---|---|---|---|---|---|
| [0, 0.5) | 53 | 0.311 | 0.000 | [0.000, 0.068] | [0.000, 0.000] | overconfident |
| [0.5, 0.8) | 48 | 0.665 | 0.000 | [0.000, 0.074] | [0.000, 0.000] | overconfident |
| [0.8, 0.9) | 27 | 0.853 | 0.185 | [0.082, 0.367] | [0.036, 0.364] | overconfident |
| [0.9, 0.95) | 20 | 0.930 | 0.800 | [0.584, 0.919] | [0.500, 1.000] | overconfident |
| [0.95, 1) | 20 | 0.970 | 0.900 | [0.699, 0.972] | [0.833, 1.000] | consistent |

rejections: 93, binding {'divert': 50, 'fuel': 43}, severity {'near': 17, 'mid': 29, 'deep': 47}, prob-bound 0.

### `b_bands` -- independent seeds, dispersion bands x 1.5 (chance spec rebuilt from the bands flown), training vehicle

vehicle `synthetic_two_stage.yaml`, seeds [9, 10] x 42 = 84 samples, bands x 1.5, recipe: FL-302 stock MC plan, joint-pinned.

landed 2/84 (0.024, Wilson [0.007, 0.083], whole-seed CI [0.023809523809523808, 0.023809523809523808]); reasons {'CORRIDOR_MISSED': 17, 'INSUFFICIENT_PROPELLANT_RESERVE': 8, 'JOINT_CHANCE_VIOLATED': 57, 'LANDED': 2}.

Brier 0.2413 vs climatology 0.0232 -> skill -9.383258345424405 (whole-seed CI [-10.26419027971017, -8.50232641113864]), ECE 0.425, MCE 0.960.

| bin | n | mean p | observed | Wilson | seed-cluster CI | verdict |
|---|---|---|---|---|---|---|
| [0, 0.5) | 51 | 0.284 | 0.000 | [0.000, 0.070] | [0.000, 0.000] | overconfident |
| [0.5, 0.8) | 25 | 0.645 | 0.000 | [0.000, 0.133] | [0.000, 0.000] | overconfident |
| [0.8, 0.9) | 4 | 0.848 | 0.000 | [0.000, 0.490] | [0.000, 0.000] | insufficient_data |
| [0.9, 0.95) | 3 | 0.920 | 0.667 | [0.208, 0.939] | [0.500, 1.000] | insufficient_data |
| [0.95, 1) | 1 | 0.960 | 0.000 | [0.000, 0.793] | [0.000, 0.000] | insufficient_data |

rejections: 57, binding {'divert': 32, 'fuel': 25}, severity {'near': 1, 'mid': 12, 'deep': 44}, prob-bound 0.

### `c_rtls` -- independent seeds, widened bands, the independent ~180 t-class RTLS redesign vehicle (E5-05a recorded ascent, E5-05c frozen closed-loop recipe + joint chance)

vehicle `synthetic_two_stage_rtls.yaml`, seeds [11, 12] x 24 = 48 samples, bands x 1.5, recipe: E5-05c closed-loop recipe, joint-pinned.

landed 0/48 (0.000, Wilson [0.000, 0.074], whole-seed CI [0.0, 0.0]); reasons {'INSUFFICIENT_PROPELLANT_RESERVE': 10, 'JOINT_CHANCE_VIOLATED': 38}.

Brier 0.2772 vs climatology 0.0000 -> skill None (whole-seed CI None), ECE 0.513, MCE 0.582.

| bin | n | mean p | observed | Wilson | seed-cluster CI | verdict |
|---|---|---|---|---|---|---|
| [0, 0.5) | 17 | 0.388 | 0.000 | [0.000, 0.184] | [0.000, 0.000] | overconfident |
| [0.5, 0.8) | 31 | 0.582 | 0.000 | [0.000, 0.110] | [0.000, 0.000] | overconfident |
| [0.8, 0.9) | 0 | - | - | - | - | insufficient_data |
| [0.9, 0.95) | 0 | - | - | - | - | insufficient_data |
| [0.95, 1) | 0 | - | - | - | - | insufficient_data |

rejections: 38, binding {'fuel': 38}, severity {'near': 0, 'mid': 0, 'deep': 38}, prob-bound 0.

## 4. Training reference (same reading, training seeds)

`e7_04_joint_base` (168 rows, seeds 1-4): Brier skill -0.8435346751657389, ECE 0.438, MCE 0.656; the E7-02 recorded reading on the stale e3_02 rows (skill 0.00006) is kept in the JSON for the historical comparison.

| bin | n | mean p | observed | seed-cluster CI | verdict |
|---|---|---|---|---|---|
| [0, 0.5) | 42 | 0.268 | 0.000 | [0.000, 0.000] | overconfident |
| [0.5, 0.8) | 66 | 0.656 | 0.000 | [0.000, 0.000] | overconfident |
| [0.8, 0.9) | 32 | 0.854 | 0.281 | [0.174, 0.452] | overconfident |
| [0.9, 0.95) | 12 | 0.931 | 0.917 | [0.667, 1.000] | consistent |
| [0.95, 1) | 16 | 0.970 | 0.938 | [0.875, 1.000] | consistent |

## 5. Training vs hold-out, side by side

| cell | n | success | Brier skill | skill CI | ECE | MCE |
|---|---|---|---|---|---|---|
| training (e7_04 joint arm) | 168 | 0.208 | -0.8435346751657389 | (-1.6022339149653808, -0.5416505166442795) | 0.438 | 0.656 |
| `a_seeds` | 168 | 0.232 | -0.6186373790229549 | [-1.0313991309874524, -0.3268140216806088] | 0.419 | 0.668 |
| `b_bands` | 84 | 0.024 | -9.383258345424405 | [-10.26419027971017, -8.50232641113864] | 0.425 | 0.960 |
| `c_rtls` | 48 | 0.000 | None | None | 0.513 | 0.582 |

## 6. Claims

Success-rate improvement claimed: **False** -- E7-04 measured no success-rate improvement to confirm (guidance fix 45->49 McNemar p=0.125, joint fix 35->35), and the diagnosis puts the infeasibility in the physical channels, not the probability model -- this task calibrates out of sample first, as ordered. No improvement is claimed; the acceptance's paired-significance clause does not bind.
