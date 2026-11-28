# E13-06 — safety supervisor: missed and false alarms against truth on the holdout (task #423)

Epic #416. This task adds the guidance-side safety supervisor
(`orbitopt.gnc.supervisor`): E12-04's four trajectory constraints plus a
passive-coast monitor, evaluated on the *estimate with its covariance* —
never on truth — feeding a `nominal → hold → abort` state machine that wraps
any `GuidanceLaw`. The machine reading is
[`e13_06_safety_supervisor.json`](https://github.com/zangjiucheng/orbitopt/blob/main/docs/benchmarks/e13_06_safety_supervisor.json); reproduce
with `PYTHONPATH=src python benchmarks/bench_e13_06_safety_supervisor.py
--jobs 16 -o …` (17 minh wall at 16 jobs,CPU只,528航班).

## 监督什么

Each monitor computes an E12-04 margin (keep-out range, cross-track cap,
terminal cone, speed-cap versus closing speed) and inflates it to a 3-sigma
lower bound `margin − k·σ` from the first-order delta method on the LVLH-
rotated estimate covariance. The fifth monitor propagates the estimate
zero-thrust for the 5400 s contingency (covariance grown by `Φ P Φᵀ + Q`) and
reports the minimum keep-out margin of that coast. The passive monitor is a
*gate, not a trigger* — mirroring the E12-04 design, which claims ballistic
safety only at declared release epochs: a hold **is** that coast, so HOLD
entry requires the coast lower bound to clear the 200 m warn margin (a trip
that fails the gate logs `hold_suppressed_passive_unsafe` and the law stays
in command), and a hold whose prediction turns red returns to nominal
(`passive_unsafe`) rather than drifting into the sphere. Instantaneous lower
bounds below zero abort directly; a hold that never clears within 180 s
aborts — on a coast that was green at entry and at every tick since.

## 实验

The same E12-10 holdout as E13-04 (22 clusters × 12 samples, seed base
20 000 000, ground plan and tolerances unchanged) is flown closed-loop twice
per sample on identical draws. **Clean arm:** the supervisor alone — the
specificity question. **Nav-fault arm:** measurements shrink 7.5 km along the
line of sight from t = 60 s (the truth plant never sees it) — the sensitivity
question, on runs where the corrupted guidance genuinely drives truth into
violation. Truth scoring is exclusively `evaluate_trajectory_safety` (E12-04)
on the sealed outcome samples; the supervisor's own events are evidence only
for alerts. The supervisor tuning (3σ, warn bands, 2-tick debounce, 180 s
hold ceiling, q = 3e-8 m²/s³) is a declared DESIGN ASSUMPTION recorded in the
artifact; nothing was tuned against the truth series.

## 结果

| 手臂 | 无 | 失踪/违反真相 | 假警报样本 | 已中止 | 过后保存通过 |
| --- | --- | --- | --- | --- | --- |
| 干净 | 264 | **0 / 0** | **1** (0.4 %, Wilson [0.07 %, 2.1 %]) | 1 | 1 / 1(最小差值+354 m) |
| 导航错误 | 264 | **0 / 247** (95 % CI upper 1.5 %) | 239 | 264 | 155 / 264 |

*Clean arm.* 263 flights fly bit-identical to the unsupervised E13-04 run —
the only touched sample (cluster 12, sample 2) trips a cross-track warning
after a suppressed hold attempt, holds, times out, and aborts at t = 2470 s
with the post-abort coast certified 354 m outside the sphere; the mission
grades `arrival_accuracy`, not a safety violation. The four truth checks are
clean on all 264 supervised flights, so the miss rate has no denominator
here — the sensitivity evidence is the fault arm and the negative test below.

*Nav-fault arm.* Every 264 flights abort (hard `approach_speed` trip once the
estimate's own closure breaks the interpolated cap): 38 directly, 226 after a
hold that the passive gate ended (`passive_unsafe` — guidance resumes because
braking beats an unsafe coast) before the hard trip. All 247 truth violation
intervals are alerted (zero missed); 352 alert intervals fall on samples whose
truth never violated that interval — false alarms are counted separately and
kept whole, not hidden behind the detections.

*Post-abort certification.* The burn-free coast after abort is re-checked
independently with the E12-04 checkers over the full contingency. It passes
whenever truth was not already compromised at the abort tick (155/264 fault
flights, minimum +6.3 m, median +427 m). The 109 failures all carry large
negative minima (median −239 m): the 7.5 km fault had already driven the true
trajectory inside the sphere *before* any estimate-consistent alert was
possible — no zero-thrust trajectory can undo a violation that already
happened, and the artifact lists every such sample rather than excluding it.

*Negative test* (unit layer, `tests/test_e13_06_safety_supervisor.py`): a
chaser inserted 1.2 km below the chief rising at 6 m/s on a collision course
is hard-aborted on the terminal-cone lower bound ~170 s before the truth
crossing; the constructed perturbation must cross the sphere, and it must be
alerted first (zero missed).

## 明确阐述的限制

- 西格玛乐观:下限是三角法,不是粒子保证;
  上面的破坏失败 是诚实的尾巴, 集中在图纸上
  被注射的错(不是主管)将真理移入其中。
- 弹道中断的海岸不能遵守终点线或交叉轨道
  * 走廊在附近漂流——运动分级计算
  安全等级的失败,即使保持球体保持清洁;
  以上保留证书与该等级分开报告。
- 200 m 警告/控入比值和15 m 跨轨带是
  调音选择;跨轨带坐落在任务本身的~30 m
  故意设计差值,因为警告说,
  几何是假警报源 干净的手臂存在测量。
