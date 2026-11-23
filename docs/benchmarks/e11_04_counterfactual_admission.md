# E11-04 — counterfactual admission validation (measured re-flight)

Generated 2026-11-22T20:19:47Z (task #311). Re-flies **every newly admitted world** of the E10-06 regrade under the shipped graded admission rule — the measurement the E10-06 report could only infer from the paired FL-305 baseline. Inputs: `e7_04_failure_driven.json` (joint arm + plan options), `e7_05_out_of_sample_reliability.json` (hold-out cells), `e3_03_robust_montecarlo.json` (FL-305 pairing), `e10_06_chance_constraint_redesign.json` (inference under test).

Rule (shipped default): reject a crossed handoff when worst > 0.5 or second > 0.15 or P̂(land) < 0.208; else admit and fly the burn. Protocol: each flipped world re-flown *without* the policy (old-rule baseline on today's physics) and then *with* it; identical ascent, dispersion stream [seed, idx] and options; replay-vs-recorded checked to rel/abs 1e-09 where the recorded campaign is bit-valid on current HEAD.

## Newly admitted worlds, measured

| cell | flips | landed (policy) | measured rate (95% Wilson) | mean P̂ | replay reproduces recorded | old rule lands today |
|---|---|---|---|---|---|---|
| train | 15 | 8/15 | 0.533 (0.301–0.752) | 0.382 | 0/15 | 5 |
| a_seeds | 16 | 15/16 | 0.938 (0.717–0.989) | 0.347 | 16/16 | 0 |
| b_bands | 1 | 1/1 | 1.000 (0.207–1.000) | 0.308 | 1/1 | 0 |
| c_rtls | 0 | 0/0 | — | — | 0/0 | 0 |

**Pooled measured landing rate of the newly admitted worlds: 24/32 = 0.750** (Wilson 0.579–0.867, cluster 0.520–0.912), mean P̂ 0.362 — the floor the rule enforces is 0.208.

## Measured vs the FL-305 inference (training grid)

measured 8/15 = 0.533 vs inferred 0.667; exact binomial vs inference p = 0.282; McNemar vs FL-305 b/c = 3/5 p = 0.727.

Measured **below** inference — why: FL-305 is a DIFFERENT planner's trajectory over the same world -- an inference proxy, not a counterfactual: a world FL-305 happened to land can still miss its burn under the guidance this flight implies (lost-landing classes above; part of the gap is the E7-04-era physics the grid was recorded under, see era_validity), and vice versa. The measurement replaces the inference wherever they disagree; per-row agreement is reported, not assumed.

| seed | idx | P̂ | FL-305 | measured | class |
|---|---|---|---|---|---|
| 1 | 31 | 0.280 | landed | JOINT_CHANCE_VIOLATED | planning_infeasible |
| 2 | 1 | 0.342 | landed | JOINT_CHANCE_VIOLATED | planning_infeasible |
| 2 | 2 | 0.246 | landed | JOINT_CHANCE_VIOLATED | planning_infeasible |
| 3 | 25 | 0.494 | landed | JOINT_CHANCE_VIOLATED | planning_infeasible |
| 4 | 4 | 0.453 | landed | JOINT_CHANCE_VIOLATED | planning_infeasible |
| 1 | 8 | 0.581 | LANDING_LATERAL_SPEED_EXCEEDED | landed | — |
| 1 | 24 | 0.245 | INSUFFICIENT_PROPELLANT_RESERVE | landed | — |
| 1 | 28 | 0.313 | INSUFFICIENT_PROPELLANT_RESERVE | landed | — |

## Failure categories of admitted worlds

- **train** — planning_infeasible 7
  codes: JOINT_CHANCE_VIOLATED (planning_infeasible)×7
- **a_seeds** — landing_accuracy 1
  codes: LANDING_LATERAL_SPEED_EXCEEDED (landing_accuracy)×1

## Rule effect (paired on today's physics)

Over the 32 re-flown flips the policy flight lands where the fresh no-policy replay failed on 19 worlds (b) and lost none of 0 (c = old-rule landings the policy could undo — impossible by construction, verified: 0 satisfied rows newly rejected); exact McNemar p = 3.81e-06. Of 27 flips whose fresh replay STILL rejects on today's physics, the admitted worlds landing is the rule's live gain; the recorded campaigns project 0.162 → 0.203 pooled landing rate (468 recorded worlds).

## Era validity of the recorded campaigns

E7-05 hold-out cells are bit-valid on current HEAD (replays reproduce recorded rows to rel/abs 1e-9) — the pairing proof. The E7-04 joint arm predates the #287 closed-loop-guidance series; its rows are re-flown on today's physics with the fresh replay as old-rule baseline (the same precedent E7-04 itself set for the stale E3-02-era joint rows):

- train: 0/15 flip replays reproduce, controls 0/4
- a_seeds: 16/16 flip replays reproduce, controls 4/4 — bit-valid cell
- b_bands: 1/1 flip replays reproduce, controls 4/4 — bit-valid cell
- c_rtls: 0/0 flip replays reproduce, controls 0/0

## Verdict

- **A1 pairing verified**: pass (bit-valid cells reproduce: True; no SAMPLE_ERROR: True; 19 era-drift rows reported).
- **A2 measured, not inferred**: pass (32/32 flips flown).
- **A3 failure categories**: pass.
- **calibration → landings?** SUPPORTED: paired b/c = 19/0 p = 3.81e-06; measured admitted rate 0.75 vs floor 0.208; campaigns project 0.162 → 0.203.
- measured came in **below** the FL-305 inference; the discrepancy is analysed above (FL-305 ≠ counterfactual + era drift) and against the floor; rule re-fit not required.
- **overall**: PASS.

*Machinery: 76 flights, 37 s wall, workers=16.*

