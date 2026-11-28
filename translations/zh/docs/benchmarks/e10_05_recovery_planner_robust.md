# E10-05恢复规划器强力优化(DC7).

JSON(`provenance`,`seed_manifest`)的证明. 为什么每个样板的重新规划得分3 /36 固定的剧本得分11 /36 ,而一个场景粗放的计划剧本能战胜两者? 被封印的DC7被扣留了

## 1 判决

| 手臂 | 着陆 | CI95组 | 威尔逊·CI95 |
|---|---|---|---|
| `fixed` | 11/36 | [0.25, 0.3333] | [0.18, 0.4686] |
| `replan40` | 3/36 | [0.0, 0.1667] | [0.0287, 0.2183] |
| `robust` | 11/36 | [0.25, 0.3333] | [0.18, 0.4686] |

Paired, Holm更正(3家庭):

| 组合 | 不和谐 | 维夫 | p (霍尔姆语) |
|---|---|---|---|
| 固定对 40 | 9-1 | 0.2222 | 0.064453125 |
| 固定对坚固 | 0-0 | 0.0 | 1.0 |
| 重新规划 40 vs 稳健 | 1-9 | -0.2222 | 0.064453125 |

## 2. 为什么重新规划失败(诊断)

2x2 (fixed x replan40): both 2, fixed-only 9, replan-only 1, neither 24.

每只手臂的混合理由 :

- `fixed`: `CORRIDOR_MISSED` 25
- `replan40`: `CORRIDOR_MISSED` 33
- `robust`: `CORRIDOR_MISSED` 25

诊断子集上的预算阶梯(固定着陆 QQ 重新规划 QQ 第一次6 两个失败的控制,n = 18 ):

| 最大 _ 级 | 着陆 | 固定着陆保存 | 两人均被救出 |
|---|---|---|---|
| 10 | 2 | 1 | 1 |
| 120 | 2 | 1 | 1 |

## 3 强臂

(1+8)-ES trust-region stages, trained on seeds [401, 402] x 5 (+ nominal), 66 candidates total:

| 阶段 | 西格玛 | 候选人 | 最佳着陆(火车) | 分数 |
|---|---|---|---|---|
| 0 | 4% | 33 | 4/11 | 0.6664 |
| 1 | 10% | 33 | 4/11 | 0.6664 |

选择的脚本 = 阶段 0(西格玛 4.0%); 训练登陆 4 / 11 , 坚守登陆 11 / 36 。 平局保持现任者( 较小的区域 ) : 在训练时没有地方扰动击败记录的脚本 。

Script movement from the recorded vector (robust arm, per parameter): boostback delay [s] +0.00, boostback aim past pad [km] +0.00, boostback throttle +0.00, boostback pitch [deg] +0.00, boostback yaw trim [deg] +0.00, entry trigger alt [m] +0.00, entry cutoff speed [m/s] +0.00, entry throttle +0.00

## 4. 种子显示和分裂

See `seed_manifest` in the JSON: hold-out [8, 9, 10] x 12 (the sealed DC7 grid), training [401, 402] x 5, recorded search seed 287, ES seed 41171. Training and hold-out seeds are disjoint by construction (asserted in the bench).
