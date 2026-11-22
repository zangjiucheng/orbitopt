# E10-06 — chance constraint redesign: graded admission

Task #303 (Epic #297). The inputs and the reading kit are named in `provenance`; everything below is an offline re-cut of the recorded E7-04 / E3-03 / E7-05 campaigns — no new flights, reproducible to the last digit. The machine reading is [`e10_06_chance_constraint_redesign.json`](e10_06_chance_constraint_redesign.json).

## The rule

The E7-01 binary gate rejected every handoff whose worst normalized joint violation exceeded the spec tolerance. The redesign (`orbitopt.flightlab.recovery.redesign`) grades the same violations instead, in this order:

1. **deep** — worst > 0.5: rejected exactly as before (E7-05 showed the deep layer is spec-vs-band disagreement, not a boundary call);
2. **second channel** — a second violation channel > 0.15: rejected (two independent bindings are not a boundary artifact);
3. **probability floor** — calibrated P̂(land) < 0.208 (the recorded joint-arm climatology 35/168, a DESIGN ASSUMPTION): rejected (the world cannot beat a random draw of its own campaign);
4. otherwise **admitted** — the landing burn flies.

Rows the old gate accepted are untouched: the policy can only ever un-reject, never re-reject (`previously_accepted_now_rejected = 0` below). Wired into FL-302 as `plan_boostback_entry(..., joint_admission=…)`; the search objective is unchanged.

## The calibrated probability

| model | LOSO CV skill | band ×1.5 skill | band ×1.5 ECE | monotone by fit |
|---|---|---|---|---|
| **physical `(fuel_z, divert_violation)` (shipped)** | 0.491 | -0.094 | 0.033 | yes |
| Platt-augmented `(+ p_raw_logit)` | 0.679 | -0.248 | 0.035 | no (negative fuel weight) |

Coefficients `(intercept, +2.733 fuel_z, −22.256 divert)` ship as `redesign.DEFAULT_CALIBRATION`, refit from the recorded file by this benchmark and asserted bit-equal. The Platt variant leads the in-band CV but is the one the task's band criterion forbids — its `logit p_raw` feature carries the training band's calibration with it.

## Acceptance 1 — hold-out calibration

On the E7-05 `a_seeds` hold-out (168 rows, seeds 5–8, scored against the recorded campaign outcomes — the same censored population that scored the raw surrogate −0.619): Brier skill **0.565** (cluster CI [0.4408, 0.6642]), against **-0.619** raw. Severe overconfidence (high-probability bin, edge ≥ 0.8, n ≥ 10, Wilson upper bound below the mean prediction): calibrated **0**, raw 2. The full bin table with Wilson and whole-seed cluster CIs is in the JSON (`holdout.a_seeds.calibrated.bands`).

**Pass: True** — Brier skill > 0, and no high-probability band is significantly below its own prediction. (The bins with n < 10 — p̂ ≥ 0.9 holds 8 rows, 3 landed — are reported and explicitly not counted; the E7-05 censoring by the old gate makes every rejected row a recorded failure, which biases high-probability bins down for any would-be-admit model, so the reading errs against the new rule.)

## Acceptance 2 — rejection rules

Regrading the training grid row-for-row: **99 → 84** rejections, false rejections (rejected yet FL-305 landed) **20 → 10**. Of the 15 newly admitted rows, **10 land** and 5 fail under the paired FL-305 baseline — a 0.67 landing rate on admitted violations, against the 0.208 climatology the floor equals. All 51 deep violations stay rejected (holding 4 false rejects back — the deep layer is deliberately not un-rejected even where it disagrees with FL-305); the second-channel gate keeps 0 and the floor 6 false rejects, the latter being the deliberate conservative margin. On the hold-out cells the rule removes 93→77 on a_seeds, 57→56 on b_bands, 38→38 on c_rtls.

**Pass: True** (false rejections down; previously-accepted set untouched; deep layer kept in full). The success-rate effect is NOT claimed — `no_success_claim` bounds it above and E10-03 owes the confirmation campaign.

## Acceptance 3 — band ×1.5

Cell `b_bands` (seeds 9–10, widened bands, spec rebuilt to match): raw Brier skill **-9.383** with ECE 0.425 / MCE 0.960; calibrated **-0.094** with ECE 0.033 / MCE 0.340. The calibrated model claims no high-probability band on this cell (its top 4 rows sit below 0.5) — the honest reading of a population whose climatology is 0.024. The skill stays marginally negative (a near-deterministic-failure population beats any probabilistic forecast on Brier alone); the task's criterion — calibration, not ranking — improves by an order of magnitude and the raw model's overconfident high band is gone. The Platt-augmented variant, for contrast, scores -0.248 here — the sensitivity the model-selection rule exists to catch.

**Pass: True**.

## Candidate ranking

The E10-06 rank orders candidates admissible-first, then by the hybrid probability. AUC against the paired FL-305 outcome on the training grid: raw p_raw 0.829 vs calibrated 0.814; hold-out `a_seeds` (recorded outcomes): raw 0.980, calibrated 0.949, hybrid 0.983. Top-25 landing rate on the hold-out: raw order 88.0% vs E10-06 order 88.0%. Ranking is preserved where the old rule already flew the burn (the raw surrogate is untouched there, E7-02-validated); what changes is that 15 rejected-but-landable training rows move into the feasible set at all — ranking inside a set the old rule never flew is the smaller effect, and both AUCs say the ordering stays sound.

## Verdict

Acceptance 1 / 2 / 3: **True / True / True** — overall **True**.
