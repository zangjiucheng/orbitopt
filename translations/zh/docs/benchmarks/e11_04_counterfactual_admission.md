# E11-04 – 反事实接纳验证(计量复飞).

Generated 2026-11-22T20:19:47Z (task #311). Re-flies **every newly admitted world** of the E10-06 regrade under the shipped graded admission rule — the measurement the E10-06 report could only infer from the paired FL-305 baseline. Inputs: `e7_04_failure_driven.json` (joint arm + plan options), `e7_05_out_of_sample_reliability.json` (hold-out cells), `e3_03_robust_montecarlo.json` (FL-305 pairing), `e10_06_chance_constraint_redesign.json` (inference under test).

规则(已发运的默认):在最差 > 0.5或第二 > 0.15或P̂(土地) < 0.208时拒绝交割;否则承认并飞烧。 协议:每个翻转的世界将*无*政策(今天物理学的旧规则基线)重新流出,然后是*与*;相同的上升段,散流[种子,idx ]和选项;重放-vs录制到rel/abs 1e-09,其中记录的运动在目前的HEAD上比特有效.

## 新进入的世界,测量

| 单元格 | 翻转 | 着陆(政策) | 测量率(95% Wilson) | 意思是说 | 重播已记录的复制 | 旧统治的今天 |
|---|---|---|---|---|---|---|
| 火车 | 15 | 8/15 | 0.533 (0.301–0.752) | 0.382 | 0/15 | 5 |
| a _种子 | 16 | 15/16 | 0.938 (0.717–0.989) | 0.347 | 16/16 | 0 |
| b _带 | 1 | 1/1 | 1.000 (0.207–1.000) | 0.308 | 1/1 | 0 |
| c _ rts | 0 | 0/0 | — | — | 0/0 | 0 |

**Pooled measured landing rate of the newly admitted worlds: 24/32 = 0.750** (Wilson 0.579–0.867, cluster 0.520–0.912), mean P̂ 0.362 — the floor the rule enforces is 0.208.

## 测量与 FL-305 推论(训练网)

8 / 15 = 0.533 与推断的0.667 ; 精确的二进制对推论p = 0.282; 麦克内马尔对FL-305 b/c=3/5 p=0.727.

经测量的**下级**推论——为什么:FL-305是一个DIFFERENT计划者在同一个世界中的轨迹——一个推论代称,而不是一个反事实:一个FL-305碰巧在这种飞行所暗示的指引下着陆,仍然可以错过它的燃烧(上面的落地类;其中一部分差距是E7-04-时代的物理记录,见时代_有效性),反之亦然. 测量可以取代他们不同意的推论;每组协议都报告,而不是假设。

| 种子 | 数字 | P | FL-305 | 测量 | 类 |
|---|---|---|---|---|---|
| 1 | 31 | 0.280 | 着陆 | JOINT_CHANCE_VIOLATED | 规划_不可行 |
| 2 | 1 | 0.342 | 着陆 | JOINT_CHANCE_VIOLATED | 规划_不可行 |
| 2 | 2 | 0.246 | 着陆 | JOINT_CHANCE_VIOLATED | 规划_不可行 |
| 3 | 25 | 0.494 | 着陆 | JOINT_CHANCE_VIOLATED | 规划_不可行 |
| 4 | 4 | 0.453 | 着陆 | JOINT_CHANCE_VIOLATED | 规划_不可行 |
| 1 | 8 | 0.581 | LANDING_LATERAL_SPEED_EXCEEDED | 着陆 | — |
| 1 | 24 | 0.245 | INSUFFICIENT_PROPELLANT_RESERVE | 着陆 | — |
| 1 | 28 | 0.313 | INSUFFICIENT_PROPELLANT_RESERVE | 着陆 | — |

## 被接纳世界的失败类别

- **火车**——规划_不可行7.
  代码:JOINT_CHANCE_VIOLATED(规划_不可行)×7.
- **a _种子** — 着陆 _精度 1
  代码:LANDING_LATERAL_SPEED_EXCEEDED(登陆_精度)×1.

## 规则效果(对今天的物理学有所贡献)

在32的复流上翻转了政策飞行地,在19的世界(b)上,新鲜的无政策重播失败,并且没有失去0(c = 旧规则着陆可以取消——通过建造不可能),经过核实:0满足了新拒绝的行;准确的McNemar p = 3.81e-06. 在27的翻转中,新的重播STILL在今天的物理学上拒绝,被接纳的世界登陆是规则的活收益;记录的运动项目0.162~0.203集合着陆率(468记录世界).

## 已记录运动的时代有效性

E7-05 hold-out cells are bit-valid on current HEAD (replays reproduce recorded rows to rel/abs 1e-9) — the pairing proof. The E7-04 joint arm predates the #287 closed-loop-guidance series; its rows are re-flown on today's physics with the fresh replay as old-rule baseline (the same precedent E7-04 itself set for the stale E3-02-era joint rows):

- 列车:0/15翻转重播,控制0/4
- a _种子:16 /16 翻转重播,控制4 /4——位值单元格
- b _波段:1 / 1翻转重播,控制4 / 4 ——位值单元格
- c _ rts:0 / 0 翻转重播,控制0 / 0

## 判决

- **A1 配对已验证** : pass(bit-valid cells replace: True; no SAMPLE_ERROR : True; 19 时代行报告).
- **A2 测量,未推断**:通过(32/32翻转).
- **A3 失败类别**:通过.
- **calibration → landings?** SUPPORTED: paired b/c = 19/0 p = 3.81e-06; measured admitted rate 0.75 vs floor 0.208; campaigns project 0.162 → 0.203.
- **下级**, FL-305推论; 差异分析见上文(FL-305 + 时代漂移),与底稿相对; 不需要重新调整规则。
- **overall**: PASS.

*Machinery: 76 flights, 37 s wall, workers=16.*
