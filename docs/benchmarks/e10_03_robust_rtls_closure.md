# E10-03 robust RTLS design closure (DC8 hot staging, E10-05 planning, E10-06 graded admission)

Provenance and the full seed manifest are in the JSON. Hold-out: seeds [501, 502, 503, 504, 505, 506, 507, 508] x 21 = 168 per arm, sized from the 0.15 Wilson-width target (required 167). Training seeds are disjoint from every sealed/E10-05 stream.

## 1. Verdict

| arm | landed | rate | cluster CI95 | Wilson CI95 |
|---|---|---|---|---|
| `dc7_pinned` | 44/168 | 0.262 | [0.1845, 0.3452] | [0.20127485645342014, 0.3331797475227076] |
| `starbase_pinned` | 11/168 | 0.066 | [0.0476, 0.0893] | [0.0369508952173935, 0.11342875703903946] |
| `dc8_robust` | 0/168 | 0.000 | [0.0, 0.0] | [0.0, 0.022354668349868045] |
| `dc8_binary` | 0/168 | 0.000 | [0.0, 0.0] | [0.0, 0.022354668349868045] |

Paired, Holm-corrected (family of 3):

| pair | discordant | diff | cluster CI95 | p raw | p (Holm) | significant improvement |
|---|---|---|---|---|---|---|
| dc8_robust vs dc7_pinned | 0-44 | -0.2619 | [-0.3452, -0.1845] | 1.137e-13 | 3.411e-13 | no |
| dc8_robust vs starbase_pinned | 0-11 | -0.0655 | [-0.0893, -0.0476] | 0.0009766 | 0.001953 | no |
| dc8_robust vs dc8_binary | 0-0 | 0.0 | [0.0, 0.0] | 1 | 1 | no |

## 2. Failure categories (planning-infeasible / execution / accuracy / fuel)

| arm | planning_infeasible | execution_failure | accuracy | fuel |
|---|---|---|---|---|
| `dc7_pinned` | 123 | 1 | 0 | 0 |
| `starbase_pinned` | 154 | 3 | 0 | 0 |
| `dc8_robust` | 168 | 0 | 0 | 0 |
| `dc8_binary` | 168 | 0 | 0 | 0 |

## 3. Robust arm (E10-05 method on DC8 with graded admission)

Selected stage 1 (sigma 10%), training landed 0/11, score 0.9709; DC7-script movement (box norm): 0.5622

| stage | sigma | candidates | best landed (train) | score |
|---|---|---|---|---|
| 0 | 4% | 33 | 0/11 | 1.0305 |
| 1 | 10% | 33 | 0/11 | 0.9709 |

## 4. Nominal gate and constraint audit

- nominal DC8/graded landed: False; recovery checks False
- baseline nominals: dc7=landed, starbase=landed
- constraints_unrelaxed: True
- parking propagation: not_evidenced_single_environment (crossenv command in the JSON)

## 5. Sealed E5-08 baselines (reference; not re-pooled)

| record | sealed hold-out | pinned |
|---|---|---|
| dc7 | seeds [8, 9, 10] | 11/36 |
| starbase | seeds [8, 9, 10] | 4/36 |

