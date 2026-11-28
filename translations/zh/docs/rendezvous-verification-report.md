# Rendezvous verification report

LEO-RDV-001携带五行证据. 他们都是**单独标签**——名义上的传球既不是独立传球,也不是强力的成功——而每一个都命名它等级的文摘.

The sealed plan: canonical planner `plan_rendezvous_safe` on
`leo_rdv_001.yaml` (normalized config SHA-256 `7915deab…bb0549`),
**plan_digest `11cb8cd24bd8c96391576703000cf14b42cde2de5978d4b0d8631ea4a34e9bfb`**
(maneuvers sha256 `d10771e1…8b9e5d`), full text in the file
`examples/runs/leo-rdv-001-sealed-plan.json`.
It spends **10.345 m/s** of a 25 m/s budget (Δv margin 14.65 m/s), time of
flight 3416.7 s, arrival 3977.3 s after the shared epoch.

Four digests, four questions (E13-00). `plan_digest` (8 significant digits,
rounding intentional): is this the same plan in the numerical sense?
`artifact_sha256` (raw file bytes): was the stored artifact modified?
`verification_digest`: do the recorded verdict and safety record belong to
this plan and model version? `environment_digest`: on which compute
environment and dependency set was it produced? `verify_seal()` recomputes
all of these and rejects a stored verdict that does not follow from the
checks.

## 1. 名义验证(同名模型-非独立).

验收宣传员重新运行密封计划,并对配置的检查进行分级:

| 检查 | 结果 |
|---|---|
| 到达位置错误 | 4.7×10⁻⁷ m (tolerance 50 m) |
| 到达相对速度 | 5.2×10⁻¹⁰ m/s (tolerance 0.2 m/s) |
| 共计v | 10.3454米/秒(预算25米/秒) |
| 门/飞行时间窗口 | 通过 |
| Overall `verify_plan` | **passed** |

Whole-trajectory safety on the dense grid also passes with margins: keep-out
min range 1000.0 m against a 500 m sphere (margin 413.0 m to surface),
cross-track margin 79.9 m, terminal-cone margin 20.0°, approach-speed margin
2.54 m/s, and 3 passive-contingency releases checked clear. Raw records:
`examples/runs/leo-rdv-001/rendezvous.json` and `verification.json`.

## 2. Mission Studio replay——本地,不独立

`orbitopt rendezvous studio` re-propagates the sealed plan with the same
`propagate_pair` model inside the container
([ADR rendezvous-0004](adr/rendezvous-0004-mission-studio-flightlab-replay.md)).
It exists to inspect and share a flight, not to verify physics. Its sidecar
records the consumed `plan_digest`; the precomputed run at
`examples/runs/leo-rdv-001` matches the sealed plan digest exactly.

## 3. 独立复飞(tudatpy-不同物理堆).

tudatpy双航天器非线性在单独环境下重新飞行是独立的线路([ADR rendezvous-0003](adr/rendezvous-0003-independent-refly-tudatpy.md),[report](https://github.com/zangjiucheng/orbitopt/blob/main/docs/benchmarks/e12_05_independent_refly.json)). 记录显示**通过** 已申报支票 要保持的两条距离:到达门是**1000 Zmm**与目标距离的Hold Point,而WHOLE 轨迹上方最接近的距离是**913.03 Zmm**——在最后插入之前到达距离内的转移海岸,所以安全检查对活追者到中心距离进行分级,从不进行Hold Point数字.

文摘纪律(未关闭,E13-00):复飞的等级是相对于整个密封计划,而不只是策略. 物理判断重编报告本身的文摘,然后要求报告嵌入式密封计划身份(config,初始状态,动作和计划文摘)与被分级的计划相匹配,非空工`environment_digest`——来自不同配置,初始状态,计划或未记录环境的证据被拒绝. 记录的证据载有Sha 256 `d10771e1…8b9e5d`的动作——密封计划(10.3454 m/s)——所以用`--refly-report`报告`physics_verified: true`所建的演播室运行,源于该配对. 其摘要不符或完全没有报告的证据给`physics_verified: null`(**未评估**,不是失败). 重新飞行涵盖*名义*密封计划;§5重新飞扰它的样本. 与`orbitopt crossenv run --plan src/orbitopt/data/crossenv_e12_05_plan.yaml`重新运行. 以[full report](https://github.com/zangjiucheng/orbitopt/blob/main/docs/benchmarks/e12-05-refly-run/rendezvous-refly-report.json)为`--refly-report`。 基准摘要列出了比较和容忍度。

## 4. 强力散射(Monte Carlo——压力价格,而非名义结果)

E12-07运动([ADR rendezvous-0005](adr/rendezvous-0005-dispersion-campaign-not-a-safety-factor.md),[data](https://github.com/zangjiucheng/orbitopt/blob/main/docs/benchmarks/e12_07_rendezvous_robustness.json))在申报应力下评价参考转移,进行集群采样和列车/停运拆分. 存档的E12-07 JSON记录了转会时间和Δv. E12-10明确记录密封计划文摘.

| 终止原因 | 数值 |
|---|---|
| 坚持成功 | **14 / 264 = 5.3 %** |
| 威尔逊 95 % CI | 3.2 – 8.7 % |
| 集束靴 CI | 2.7 – 8.0 % |
| 失败类别 | 181抵达精度,69安全,0规划/燃料. |

这个数字是**强力**,用负压武器表示;在1中,名义通行证是**名义**,用负压零表示。 报告一个是另一个,文件是错的。

## 5. 独立坚固性重飞(E12-10-相同密封计划,扰动)

The E12-10 benchmark
([ADR rendezvous-0006](adr/rendezvous-0006-independent-robustness-reflight.md),
[data](https://github.com/zangjiucheng/orbitopt/blob/main/docs/benchmarks/e12_10_independent_reflight.json)) re-grades the E12-07 hold-out with the passive contingency
releases (no-burn at t0, gate abort at t1, hold-point coast) counted in success, and re-flies a declared selection
of 23 samples in TudatPy, each with its own perturbation. Its ground plan is the sealed plan above (same
`plan_digest`).

| 终止原因 | 数值 |
|---|---|
| 坚持成功、被动意外情况 | **13 / 264 = 4.9 %** |
| 集束靴 95% CI | 2.7 – 7.6 % |
| 失败类别 | 164到达精度,87安全(持点海岸为主). |
| tudatpy 每样协议 | 23 of 23 verdicts agree, 0 flips, every residual inside its band |

The classification is **Independently Verified Robustness Characterization**: it says the *evidence* is
independent, not that the design is robust. The design succeeds on about one hold-out sample in twenty.

## 文摘稳定性

在两台机器上解开的计划在其烧伤的最后几位(大约在这里的3e-12相对)上有所不同,因此文摘散列漂移到8显著位数(`DIGEST_SIGNIFICANT_DIGITS`). 在此之前,一个配置给 不同的`plan_digest` 每一个尝试的机器。 四舍五入仍然可以将跨越一个边界的两个值(大约每数字为3e-4)分割开来,所以机器之间的平等在实践中是稳定的,不能保证的;不匹配应读作"调整和比较策略",而不是作为不同计划的证据. `tests/test_e12_08_canonical_plan.py`对照本代码计算摘要,检查承诺的密封计划,运行和基准.

## 数字所在的地方

| 证据 | 艺术 |
|---|---|
| 名义+安全 | `examples/runs/leo-rdv-001/{rendezvous,verification}.json` |
| CW 信封 | `docs/benchmarks/e12_02_cw_validity.{md,json}` |
| 独立飞行 | `docs/benchmarks/e12-05-refly-run/` + `e12_05_independent_refly.json` |
| 有力运动 | `docs/benchmarks/e12_07_rendezvous_robustness.json` |
| 独立稳健性 | `docs/benchmarks/e12_10_independent_reflight.json` |
