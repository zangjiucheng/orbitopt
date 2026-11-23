# E11-03 robust recovery co-design (task #310)

_generated 2026-11-23T02:14:29Z, git ``_

E10-03 closed negative: the DC8 separation state is outside the feasible envelope and no script-in-the-box pulls it in (0/168). E11-03 co-designs the separation STATE itself (re-flown ascent timeline/attitude/burn re-trim) with the recovery script, using the E11-02 envelope screen (with E10-03 disturbance draws) as the co-design objective, and flies the result against the sealed baselines on a fresh, paired hold-out.

## Hold-out campaign (paired, fresh seeds 501-508 x 21)

| arm | landed / n | rate | cluster CI95 |
|---|---|---|---|
| `codesign` | 0 / 168 | 0.0% | [0.000, 0.000] |
| `dc7_pinned` | 44 / 168 | 26.2% | [0.191, 0.351] |
| `starbase_pinned` | 11 / 168 | 6.6% | [0.048, 0.089] |

## Paired comparisons (family of 2, Holm-corrected)

| comparison | diff | paired CI95 | McNemar exact p | p (Holm) | significant |
|---|---|---|---|---|---|
| codesign vs dc7_pinned | -0.262 | [-0.351, -0.191] | 1.137e-13 | 2.274e-13 | True |
| codesign vs starbase_pinned | -0.066 | [-0.089, -0.048] | 0.0009766 | 0.0009766 | True |

## Failure categories

| arm | planning_infeasible | execution_failure | accuracy | fuel |
|---|---|---|---|---|
| `codesign` | 163 | 0 | 0 | 5 |
| `dc7_pinned` | 123 | 0 | 0 | 1 |
| `starbase_pinned` | 154 | 0 | 0 | 3 |

Reason mix (top codes per arm):

- `codesign`: CORRIDOR_MISSED x157, JOINT_CHANCE_VIOLATED x6, INSUFFICIENT_PROPELLANT_RESERVE x5
- `dc7_pinned`: CORRIDOR_MISSED x123, INSUFFICIENT_PROPELLANT_RESERVE x1
- `starbase_pinned`: CORRIDOR_MISSED x154, INSUFFICIENT_PROPELLANT_RESERVE x3

## Training agreement (screen verdict vs flown outcome)

concordance 0.0 (None/None screen-ON samples that were planned-and-flown; a false-ON is a screen lie counted against it).

## Upper-stage insertion margin (reported, acceptance item)

sealed DC8 ship margin 103548 kg -> codesign 103548 kg (+0 kg, ship burn +0 s). the margin is stage-2 (ship) propellant above dry mass at SECO, from the re-flown OT tracks; a positive burn extension buys insertion re-trim by SPENDING ship propellant, so the honest direction of the delta is downward as dburn grows -- reported, never hidden

## E5-06 joint physical gate (codesign nominal)

`landing`: failed | `mass_conservation`: passed | `epoch_frame_continuity`: passed | `orbit_insertion`: passed | `parking_propagation`: not_evidenced_single_environment


Constraints unrelaxed: True (ascent checks all pass on the chosen ascent; graded reserve identical across arms). the codesign planner grades its reserve floor at 0 kg soft + the target's hard 100000 kg (a superset of admitted plans that the SAME target check grades; the sealed-recipe arms keep their own recipe floors -- E10-03/E10-06 semantics, no relaxation anywhere)

