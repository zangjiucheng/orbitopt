# 验证 警卫

`orbitopt.optimization.verification`的成绩是`OptimizationResult`,与它应该回答的问题相对应. 它从来不相信结果对自己的看法。 结果的四个布尔(`solver_converged`,`constraints_satisfied`,`independently_verified`,`optimality_certified`)是索赔. 警卫是靠自己获得每个身份的,不能证实的说法本身就是一个结论。 IR问题在[Optimization IR](ir.md)描述.

## 职等结果

```python
from orbitopt.optimization.verification import ReferenceSolution, result_digest, verify_result

evidence = ReferenceSolution(id="doc", value=0.5, source="worked example",
                             problem_digest=problem.digest(), result_digest=result_digest(result))
report = verify_result(problem, result, evidence=[evidence])
report.verdict                                      # Outcome.PASSED
report.statuses["independently_verified"]           # StatusOutcome(passed, reason, evidence_ids=("doc",))
report.codes                                        # finding codes, empty here
```

`verify_result(problem, result, *, policy=None, evidence=(), artifact_digest=None, solved_problem=None, tools=None, seed=0)`不升. 警卫内部的错误变成了`guard_error`的发现和失败的判决.

| 参数 | 含义 |
|---|---|
| `policy` | 默认为 `problem.verification` 。 它列出了所需的状况和容忍度。 |
| `evidence` | `independently_verified`的证据项目,以及下限和盘点证书. 警卫检查每个物品本身。 |
| `artifact_digest` | 文摘在存储时记录了结果. 它必须等于`result_digest(result)`. |
| `solved_problem` | 解决者实际运行的问题, 当它与分级的问题不同。 缓存命中并不是一个论据:这是`result.provenance.cache_key`,结果摘要涵盖的. |
| `tools` | `ToolCall`目标和限制的解决者,通过工具id. |
| `seed` | 检查下限证书的样本种子 。 |

`result` may also be the dict form. The guard checks it against
the result schema its `schemaVersion` names (`optimization-result-1.1.json`,
or 1.0 for an older document) first, so a missing field gives
`malformed_result`.

## 每种状态如何产生

每个身份是`passed`,`failed`或`not_assessed`,有其理由和依据的证据.

| 状态 | `passed` 当 | `failed` 当 | `not_assessed` 当 |
|---|---|---|---|
| `solver_converged` | `termination`为`converged`. | 任何其他终止。 | 从来没有。 |
| `constraints_satisfied` | 警卫通过`result.x`重新评价`OptimizationProblem.evaluate`,所有约束性,整体性规则,离散选择和硬约束都存在于`constraint_tol`之内. `OptimizationProblem.constraints_satisfied`必须同意. | 缺少一个值,而不是一个数字或者没有限制,或者一个域或约束被违反,或者问题在点上没有定义. | `ToolCall`没有解析器,也没有提出工具. |
| `independently_verified` | 至少一个允许的证据项目证实了这一结果,并同意重新评价的目标。 没有经过验证的项目有异议。 | 经审定的项目有异议. | 无证据项目验证. |
| `optimality_certified` | 警卫对有限域的列举发现没有什么更好的,或者KKT证书通过,没有下限证书被反驳. | 这一点是不可行的, 点数发现一个更可行的点, KKT 检查失败, 或拒绝一个下限证书 。 | 未要求,问题在可认证类之外,或者无法评价点. |

当保险单需要时,当结果声称需要时,或者当提供低限或清点证书时,警卫会尝试`optimality_certified`.

## 判决

判决是`passed`, 只有当`policy.required`的每一个地位通过, 当要求的状态失败或任何发现都是致命的时,是`failed`. 否则就是`not_assessed`. 一项不需要任何地位的政策永远不会通过。

这些检查得出了致命的结论,无论情况如何:

- `problem_mismatch`:`result.problem_digest`不是问题摘要.
  结果声称答案。 问题是`solved_problem`何时
  和分级的问题。
- `unauthorized_relaxation`和`formulation_mismatch`:问题解决了.
  `hard_constraint_fingerprint()`与分级问题不同. 拓宽
  域名、 已丢弃或解锁的用户约束, 或 放松的右侧
  这是一种放松。 对用户目标的任何其他更改,锁定约束,
  变量或参数是一个配方不匹配。 保留
  指纹,例如附加的代理约束,作为
  `revised_problem`. 然后根据最初的问题分级。
- `stale_cache`:存储在另一个问题摘要下的一个条目的缓存命中.
- `digest_mismatch`:`artifact_digest`或密封文件不匹配.
- `objective_mismatch`,`residual_mismatch`和`non_finite`:索赔价值
  与警卫的重新评价不同, 或纳恩或无限。
- `constraint_violation` 和 `domain_violation` : 点断绑或
  约束,或整体性规则或独立选择。
- `claim_contradicted`:结果声称警卫没有通过状态.
- `forged_verification`:一个证据项来自结果自己的解析者.
- `malformed_result`, `malformed_artifact`, `missing_file`, `invalid_problem`
  和`guard_error`.

`missing_evidence`,`evidence_mismatch`,`evidence_disagrees`或`certificate_unsupported`等注释解释状态. 他们不会自己辜负判决。 当需要他们解释的状态时,他们仍然屏蔽通行证。 `FindingCode.fatal`将两种说法分开.

## 证据

Every evidence item names `problem_digest` (`problem.digest()`) and
`result_digest` (`result_digest(result)`). An item with a missing digest, a
digest of another problem or result, or a kind that
`policy.independent_methods` does not allow never counts. Evidence inside
`result.evidence` is the solver vouching for itself. The guard ignores it and
records `self_attestation`.

| 类型 | 温柔 | 什么警卫检查 |
|---|---|---|
| `AnalyticalOptimum` | `analytical` | 一个不空的`derivation`. `x`时,警卫检查`x`是可行的,并达到`value`. 然后将`value`与`objective_rtol * max(1, abs(value))`内部重新评价的目标进行比较. |
| `ReferenceSolution` | `reference_solution` | 一个不空的`source`. 容忍度是`min(tolerance, objective_rtol * max(1, abs(value)))`,因此引用可以收紧政策,但绝不放松政策. |
| `AlternateSolverCheck` | `alternate_solver` | 警卫给`solver(problem)`打电话 返回的结果必须回答分级的问题,来自申报的`solver_id`,它必须与分级的结果不同. 其要点必须是可行的。 比结果更好的目标可以反驳结果. 更糟的就是没有结果 守卫将目标比作给定的目标;溶解器适配器只在无法恢复时报告一个波段点(见下文). |
| `LowerBoundCertificate` | `optimality_certificate` | 见下一节. |
| `EnumerationCertificate` | `enumeration_certificate` | 请求,不是要求。 警卫列举了域本身。 见下一节. |

`policy.independent_methods`列出`independently_verified`的方法. 它不适用于仅涉及`optimality_certified`的两种证书类型(`CERTIFICATE_KINDS`). 证书的文摘必须仍然将这个问题和这个结果定着,它的`producer`不能成为结果的解决者.

`AlternateSolver`是一个协议:任何需要`OptimizationProblem`并返回`OptimizationResult`的可调用程序. 护卫船没有自己的解决者。

### 容忍乐队的候补成员

`constraint_tol` is the verification tolerance and stays the only one the
guard uses. A feasible point that uses the band can beat the exact optimum
by its multiplier times its violation, so an alternate that presses against
the band can refute an exact result. In the first E14-10 campaign the
differential-evolution alternate on `cqp-equality-d3` found
f = 0.5454535 at violation 9.54e-7, which is f* - (12/11)(9.54e-7), and
refuted SLSQP's exact 6/11.

The guard's rule did not change. Refutation is the safe direction: it can
fail a correct result, never pass a wrong one, and telling a bought gain
from a real one needs the multiplier, which the guard would have to
estimate. The solver adapters fix the comparison at its source instead: they
rank clean points first and restore their best band point onto the
constraints before they report it
([solvers](https://github.com/zangjiucheng/orbitopt/blob/main/docs/optimization/solvers.md#the-tolerance-band-is-not-room-to-optimise)). The
restored alternate on that instance answers 0.54545454545465, within
1e-13 of 6/11, and agrees. An alternate that cannot restore its point (no
evaluations left, a failed evaluation) still refutes as before.

## 警卫能证明什么

`optimality_certified`只通过一个由警卫自己计算出来的证书:一个有限域的点数,或者一个对流问题的KKT条件.

### 编号

An `EnumerationCertificate(id, problem_digest, result_digest, objective=None, max_points=MAX_ENUMERATION_POINTS)`要求警卫评价域内每个点(`enumeration_certificate`). 它适用于一个单一目标问题,其中每个变量都使用有限的许多数值:二进制、离散或有两种界限的整数(`Variable.domain_values` )。 域的产物最多有`min(max_points, MAX_ENUMERATION_POINTS)`点,`MAX_ENUMERATION_POINTS`是1,048,576(2 ^ 20),无论项目要求什么. 守门员用自己的点检走每个点,包括工具,并将每个可行点的目标与结果进行比较.

- `passed`:没有可行的点比结果多
  目标意义上的`objective_rtol * max(1, abs(f))`. 这是个
  任何目标的证据,包含一个黑匣子,因为什么都没有了
  出来。
- `failed`:一个可行的点击败它. 发现点的名字。
- `not_assessed` : 一个变量是连续的或未绑定的,域是
  大于上限,问题有几个目标,或者某个点
  无法评价(没有解析器,或提出工具),因为
  这可能是更好的一个。 警卫接着拿到KKT证书,
  也不适用于个别问题。

在表达的域外的一点是不可行的,就像其他的卫兵一样. 查点每点花费一个评价,这就是为什么它只按要求运行,从未在平板屏幕上运行的原因.

An enumeration solver's own convergence is never a certificate. Its result
counts toward `independently_verified` as an `AlternateSolverCheck`, like
any other alternate solver. Only the guard's own enumeration passes
`optimality_certified`.

### KKT条件

KKT证书(`kkt_certificate`)适用于这些属性的问题:

- 它只有一个目标 每个变量都是连续的
- 目标是在结构上采用2或以下的多名制。
  变量。 参数和字数算作常数. 黑森帝国
  目标在最小值和负值时为正半定值
  最大时半定时. 它的最小值至少必须是
  `-1e-8`是其最大的条目.
- 每一种限制都是结构性的。
- 没有通过`ToolCall`来评估。

该类涵盖线性程序和凸轮四极程序. 对他们来说,KKT条件足以实现全球优化. 护卫用`max(1, abs(x_i))`步骤的中位差计算梯度,这对四进制到四进制是准确的. 如果其残留值为零的`10 * constraint_tol * tolerance_scale(rhs)`,则不平等是活跃的。 乘数来自活动集上一个最小平方解. 虽然不平等的乘数是负数,但这一制约因素使组合和解决方案重复。 当静态残基最多为`optimality_gap * max(1, max abs(grad f))`,互补最多为`optimality_gap * max(1, abs(f))`. 初步可行性来自`constraints_satisfied`.

Every other problem is `not_assessed` by KKT: non-convex objectives
(Rosenbrock, Rastrigin, an indefinite quadratic), mixed-integer and discrete
problems, multi-objective problems and tool-evaluated problems. A discrete
problem whose domain is small enough can still be certified by enumeration.
**This guard never certifies global optimality of a continuous non-convex
problem**, even when the point is the known optimum.

`LowerBoundCertificate`表示约束及其衍生. 警卫检查结果是否达到`optimality_gap`的界限。 它还试图反驳可行集的种子样本上的束缚:一半的样本覆盖了可变框,一半是结果点的小扰动. 被反驳的约束或未达到的约束都不符合这一地位。 采样不能证明束缚,所以生存的束缚不能通过状态. 在对流问题上, KKT 证书决定了 。 在任何其他问题上,状况仍为`not_assessed`。 取样也可能错过一个小盆地,就像当地最低的Rastrigin一样。 因此,该证书是一种反驳检查,而不是证据。

## 密封运行

`seal_run(problem, result, evidence, run_dir)`写了四个文件并返回了表单摘要:

| 文件 | 内容 |
|---|---|
| `problem.json` | `problem.to_dict()` 解决器运行的问题 。 |
| `result.json` | `result.to_dict()`,其`provenance`记录有缓存命中. |
| `evidence.json` | 证义项. 一个`AlternateSolverCheck`是储存的,没有它的可调用. `EnumerationCertificate`是按原样存储的,所以`verify_run`再次列出域. |
| `manifest.json` | 格式 `orbitopt.optimization.run/2`,每个文件精确字节的 `sha256:`摘要, `problem.digest()`和 `result_digest(result)`. 1格式保存了`evidence.json`格式的缓存记录;一个1格式的运行以错误的形式被拒绝. |

每个文件都是cononical JSON :排序键,紧凑的分隔符,UTF-8,没有NaN或无限. `verify_run(run_dir, *, problem=None, policy=None, alternate_solvers=None, tools=None, expected_manifest_digest=None, seed=0)`从档案中重新找到所有的东西. 缺少的文件,更改的字节或文摘不匹配会给一个与查找失败的报告,而卫兵从修改后的运行中没有任何成绩. 将`problem`通过分级 以克服被问到的问题。 没有它,运行会被分级到自己的`problem.json`,所以谁写目录会选择问题. 能重写目录的人也可以重写清单. 保留`seal_run`返回的文摘,并以`expected_manifest_digest`通过.

The manifest records the SHA-256 of exact bytes, not the 12-digit quantised
digest that the IR uses. A tamper seal must catch a change in the 13th digit
of `x`. The quantised `result_digest` identifies a result for evidence
pinning, and it is not a tamper seal. It leaves out the wall-clock keys of
`orbitopt.optimization.VOLATILE_KEYS` (`wall_s`, `wall_time_s`, `latency_s`)
at any depth, as the agent trace digest does, so two runs that found the
same answer the same way share a result digest. The byte digests still seal
`wall_time_s`.

## 错误矩阵

`tests/test_e14_06_verification_guard.py` applies all nine injectors of
`benchmarks/e14_spec/faults.py` to an honest artifact of each of the 25
catalogue instances that the IR can express. The other instances use black
boxes, have several objectives or store no optimiser. Each faulty artifact
is graded twice, once with `verify_result` and once from a sealed run with
`verify_run`. Of the 450 graded cells, none passes, and each one names the injector's reason
among its findings. Rerun the matrix with this command:

```bash
PYTHONPATH=src python -m pytest -q tests/test_e14_06_verification_guard.py -k "fault_is_never or honest_benchmark"
```

盘点证书有自己的矩阵运行 。 三个目录夹最多为4096点(`knapsack-p02-n5`,`knapsack-p01-n10`,`knapsack-2d-n12`)得到每个注射器,一个`EnumerationCertificate`被钉在错误的结果上,所有四个状态都需要. 每一个都是在记忆和密封运行中分级的。 54细胞中,没有通过,每个名字都是注射者的原因. 每一份都通过证书,通过四个状态。 伪造的证书从未通过`optimality_certified` :一个在不优化的可行点上失败,一个被钉在另一个结果或问题上,一个来自结果自己的解析器,一个命名一个未知的目标,一个其封顶不覆盖域,一个在大于全球封顶的域上是未评估 .

```bash
PYTHONPATH=src python -m pytest -q tests/test_e14_06_verification_guard.py -k "enumeration or finite_hosts"
```
