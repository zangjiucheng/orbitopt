# E10-06——机会限制重新设计:分级录取.

Task #303 (Epic #297). The inputs and the reading kit are named in `provenance`; everything below is an offline re-cut of the recorded E7-04 / E3-03 / E7-05 campaigns — no new flights, reproducible to the last digit. The machine reading is [`e10_06_chance_constraint_redesign.json`](https://github.com/zangjiucheng/orbitopt/blob/main/docs/benchmarks/e10_06_chance_constraint_redesign.json).

## 规则

E7-01二进制门拒绝每一次交割,其最恶劣的常态共同侵权超过光谱容忍度. 重新设计(`orbitopt.flightlab.recovery.redesign`)则按此顺序对同样的违规行为进行评级:

1. **深处**——最糟的0.5:完全如前被否决(E7-05显示深层是光谱-vs-波段分歧,不是边界调用);
2. **第二频道**——第二个违规频道 > 0.15:被拒绝(两个独立的束缚不是边界文物);
3. **probability floor** — calibrated P̂(land) < 0.208 (the recorded joint-arm climatology 35/168, a DESIGN ASSUMPTION): rejected (the world cannot beat a random draw of its own campaign);
4. otherwise **admitted** — the landing burn flies.

所接受的旧门的行是未触及的:政策只能是永远不拒绝,永远不拒绝(下文`previously_accepted_now_rejected = 0`). 连接到FL-302为`plan_boostback_entry(..., joint_admission=…)`;搜索目标不变.

## 校准概率

| 模型 | LOSO CV skill | 带×1.5技能 | 带 × 1.5 ECE | 单调音量 |
|---|---|---|---|---|
| **physical `(fuel_z, divert_violation)` (shipped)** | 0.491 | -0.094 | 0.033 | 对 |
| 平板化 `(+ p_raw_logit)` | 0.679 | -0.248 | 0.035 | 无(负燃料重量) |

`(intercept, +2.733 fuel_z, −22.256 divert)`号飞船作为`redesign.DEFAULT_CALIBRATION`号,按照这个基准从记录文件中重排,并断言位数相等. 普拉特变体领先于波段的CV,但这是任务波段标准所禁止的——它的`logit p_raw`特性带有训练波段的校准.

## 接受1-搁置校准

On the E7-05 `a_seeds` hold-out (168 rows, seeds 5–8, scored against the recorded campaign outcomes — the same censored population that scored the raw surrogate −0.619): Brier skill **0.565** (cluster CI [0.4408, 0.6642]), against **-0.619** raw. Severe overconfidence (high-probability bin, edge ≥ 0.8, n ≥ 10, Wilson upper bound below the mean prediction): calibrated **0**, raw 2. The full bin table with Wilson and whole-seed cluster CIs is in the JSON (`holdout.a_seeds.calibrated.bands`).

**Pass: True** — Brier skill > 0, and no high-probability band is significantly below its own prediction. (The bins with n < 10 — p̂ ≥ 0.9 holds 8 rows, 3 landed — are reported and explicitly not counted; the E7-05 censoring by the old gate makes every rejected row a recorded failure, which biases high-probability bins down for any would-be-admit model, so the reading errs against the new rule.)

## 接受2-拒绝规则

Regrading the training grid row-for-row: **99 → 84** rejections, false rejections (rejected yet FL-305 landed) **20 → 10**. Of the 15 newly admitted rows, **10 land** and 5 fail under the paired FL-305 baseline — a 0.67 landing rate on admitted violations, against the 0.208 climatology the floor equals. All 51 deep violations stay rejected (holding 4 false rejects back — the deep layer is deliberately not un-rejected even where it disagrees with FL-305); the second-channel gate keeps 0 and the floor 6 false rejects, the latter being the deliberate conservative margin. On the hold-out cells the rule removes 93→77 on a_seeds, 57→56 on b_bands, 38→38 on c_rtls.

**传:真**(假拒绝;以前接受的套装未受影响;深层保存完整). NOT声称成功率效应——`no_success_claim`将其绑在上方,E10-03欠确认竞选.

## 接受 3 — 带 × 1.5

Cell `b_bands` (seeds 9–10, widened bands, spec rebuilt to match): raw Brier skill **-9.383** with ECE 0.425 / MCE 0.960; calibrated **-0.094** with ECE 0.033 / MCE 0.340. The calibrated model claims no high-probability band on this cell (its top 4 rows sit below 0.5) — the honest reading of a population whose climatology is 0.024. The skill stays marginally negative (a near-deterministic-failure population beats any probabilistic forecast on Brier alone); the task's criterion — calibration, not ranking — improves by an order of magnitude and the raw model's overconfident high band is gone. The Platt-augmented variant, for contrast, scores -0.248 here — the sensitivity the model-selection rule exists to catch.

**Pass: True**.

## 候选人排名

The E10-06 rank orders candidates admissible-first, then by the hybrid probability. AUC against the paired FL-305 outcome on the training grid: raw p_raw 0.829 vs calibrated 0.814; hold-out `a_seeds` (recorded outcomes): raw 0.980, calibrated 0.949, hybrid 0.983. Top-25 landing rate on the hold-out: raw order 88.0% vs E10-06 order 88.0%. Ranking is preserved where the old rule already flew the burn (the raw surrogate is untouched there, E7-02-validated); what changes is that 15 rejected-but-landable training rows move into the feasible set at all — ranking inside a set the old rule never flew is the smaller effect, and both AUCs say the ordering stays sound.

## 判决

接受1/2/3:**真/ 真/ 真**-整体**没错**.
