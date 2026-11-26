# E11R-01: robust candidate selection for the E11-06 continuous-burn search

Task #414, epic E11R. Provenance in the JSON (`provenance`, `seed_manifest`). The E12-00 record showed the sealed search's first-feasible stop leaves the pinned hold-out rate seed dependent (6/36 to 33/36); this campaign harvests the feasible pool, ranks it on fresh training seeds by landing count / corridor margin, and flies the argmax pinned on the sealed 36-sample hold-out, paired against the sealed champion and the same run's stop point.

**Verdict.** Hold-out rate across search seeds -- selected arms 0.97..1.00 (range 0.03, std 0.0124), first-feasible stop points of the same runs 0.17..0.92 (range 0.75, std 0.2788); archived E12-00 first-feasible table 0.17..0.92. Pooled selected 179/180 (cluster CI [0.9833, 1.0]). Sealed champion re-flown: 33/36.

## Search seeds

| seed | pool kept/total | stop-point train score | selected train score | stop k/36 | selected k/36 | selected cluster CI |
|---|---|---|---|---|---|---|
| 1 | 10/100 | 7/12 @ 112482.6 kg | 12/12 @ 118137.2 kg | 24/36 | 35/36 | [0.9167, 1.0] |
| 2 | 10/81 | 2/12 @ 99968.2 kg | 12/12 @ 105684.6 kg | 6/36 | 36/36 | [1.0, 1.0] |
| 3 | 10/82 | 6/12 @ 87981.7 kg | 12/12 @ 87112.2 kg | 16/36 | 36/36 | [1.0, 1.0] |
| 4 | 10/94 | 7/12 @ 89089.8 kg | 12/12 @ 102436.0 kg | 17/36 | 36/36 | [1.0, 1.0] |
| 5 | 10/254 | 9/12 @ 86234.2 kg | 12/12 @ 87932.7 kg | 33/36 | 36/36 | [1.0, 1.0] |

## Paired against the sealed 33/36 champion (exact McNemar, Holm-corrected)

| selected arm | both / a-only / b-only / neither | p raw | p Holm | sig @ 0.05 |
|---|---|---|---|---|
| cand-s1-3 | 33/2/0/1 | 0.5 | 1 | False |
| cand-s2-5 | 33/3/0/0 | 0.25 | 1 | False |
| cand-s3-9 | 33/3/0/0 | 0.25 | 1 | False |
| cand-s4-9 | 33/3/0/0 | 0.25 | 1 | False |
| cand-s5-7 | 33/3/0/0 | 0.25 | 1 | False |

## Selected vs the same run's first-feasible stop (exact McNemar, uncorrected)

| pair | both / a-only / b-only / neither | diff | p raw |
|---|---|---|---|
| cand-s1-3 vs stop-s1 | 24/11/0/1 | +11 | 0.0009766 |
| cand-s2-5 vs stop-s2 | 6/30/0/0 | +30 | 1.863e-09 |
| cand-s3-9 vs stop-s3 | 16/20/0/0 | +20 | 1.907e-06 |
| cand-s4-9 vs stop-s4 | 17/19/0/0 | +19 | 3.815e-06 |
| cand-s5-7 vs stop-s5 | 33/3/0/0 | +3 | 0.25 |

## Constraints (not relaxed)

- graded limits unchanged and identical to the sealed record (True): {'max_touchdown_speed_m_s': 3.0, 'max_touchdown_lateral_speed_m_s': 1.0, 'min_propellant_reserve_kg': 100000.0, 'entry_q_max_pa': 250000.0, 'entry_g_max': 12.0}.
- grid-fin drag stays a **DESIGN ASSUMPTION**: v3_pp grid-fin absolute drag: no public Super Heavy data -- the per-planform calibration (bench_e11_06 FIN_SETS/PP_CD) is a DESIGN ASSUMPTION, inherited unchanged here; every check graded in this campaign inherits it.
- training and hold-out both fly the authored script pinned (search=keep): the only object under test is the SELECTION rule; search box and landing law unchanged: sealed E11-06 continuous burn (lit_at_start, 4.0 g cap, TVC 20 deg/s, engine stages 13->5->3 under E11_ENGINE_STAGES=1).

## Reproduce

```
E11_ENGINE_STAGES=1 python benchmarks/bench_e11r_01_robust_candidate_selection.py --workers 40 --cache campaign-cache.json -o e11r01-campaign.json --md e11r01-campaign.md
```
