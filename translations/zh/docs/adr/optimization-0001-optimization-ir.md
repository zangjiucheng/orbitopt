# ADR优化-0001:LLM-辅助优化(E14-01)一个键入IR.

- **状态 :**接受. 下面的接受阈值在此提出并冻结.
- **Date:** 2026-11-26
- **Deciders:** E14 LLM-assisted optimization, E14-01
- **Code:** `src/orbitopt/optimization/`, `src/orbitopt/schemas/optimization-problem-1.1.json`,
  `src/orbitopt/schemas/optimization-result-1.1.json`(用于解析1.0文件的1.0计划停留).
- **Amended:** IR 1.1, see the addendum at the end
- **Evidence:** `tests/test_e14_01_problem_ir.py`, `docs/optimization/ir.md`
- **Consumers:** E14-02 natural-language intake, E14-03 solvers, E14-04 tool registry,
  E14-05 agent runtime, E14-06 verification guard

## 背景和问题

五个E14工人必须就问题达成一致. 一个LLM提出,一个解器消耗它,工具评估它的部分,一个特工修改它,一个警卫检查它的答案. 如果每对工人都同意自己的形状,那么这些界限中的每一个都变成了一个翻译层,而削弱用户约束的代理编辑可以隐藏在翻译中.

The contract also has to be safe to evaluate when an LLM wrote it, so
expressions cannot be Python. It has to identify a problem the same way on
every machine. The E12 rendezvous plan digests hashed exact float reprs, and
last-bit noise gave a different digest on each machine. It has to let a guard
tell whether the user's hard constraints survived an agent's revision. And it
has to keep apart four claims that are easy to blur: the solver converged, the
constraints hold, someone else checked the answer, and the answer is provably
optimal.

## 决定

**Frozen dataclasses with a JSON form.** `OptimizationProblem` holds `id`,
`title`, `description`, `variables`, `parameters`, `objectives`,
`constraints`, `budget`, `verification` and `metadata`. `to_dict` and
`from_dict` round-trip exactly. The JSON Schema checks structure only: types,
required fields and enumerations. `validate_problem` owns every value rule and
returns typed `ProblemError(code, path, message)` values. `parse_problem` is
the boundary for untrusted dicts. It raises one `InvalidProblemError` that
carries all errors from the first failing stage. Construction never
validates, so an LLM's malformed proposal can still be built and asked for
its full error list.

**The IR stores what the user said.** The canonical mathematics is
`min f(x; theta)` subject to `g_i <= 0` and `h_j = 0`. A constraint still keeps
`expression <sense> rhs` with sense `<=`, `>=` or `==`, and an objective
keeps `minimize` or `maximize`. `Constraint.residual` maps a constraint onto
the canonical form: `lhs - rhs` for `<=` and `==`, `rhs - lhs` for `>=`. All
constraints are hard. There are no soft penalties in 1.0.

**Variables.** A variable has a `kind`: `continuous`, `integer`, `binary` or
`discrete`. Continuous and integer variables take `lower`/`upper`, and `None`
leaves a side open. Integer bounds must be integers. A binary variable sets
neither. A discrete variable takes a non-empty tuple of distinct numeric
`choices`. Every variable has a `unit` and an optional `initial` that must lie
in its domain. A `Parameter` is a named, finite constant with a unit.

**Expressions are a whitelisted subset, not Python.** A source string must
parse to numeric literals, names, `+ - * / **`, unary `-`/`+` and calls to 17
fixed math functions. Calls are positional only, and each function has a
fixed arity. Attributes, subscripts, strings, booleans, comparisons, lambdas,
comprehensions, keywords and names starting with `_` are rejected with a
specific `ErrorCode`. The limits are 2000 characters, 500 nodes, nesting
depth 200 and literal exponents of at most 100. Integer literals become floats
before compilation, so `9**9**9` overflows at once and cannot spin up a huge
integer. Only the validated tree is compiled, and it runs with no builtins.
The raw text never reaches `eval`. The same code path evaluates plain floats
and numpy arrays, and numpy is imported only when an array is passed in. The
alternative is a `ToolCall(tool_id, args)`. It references a registered tool,
and each argument is itself a safe expression.

**单位:内置SI表,尺寸检查.** Astropy 存在于康达计算环境中,但不是碱性依赖. 一个有不同行为和没有不同行为的验证员会在另一个地方重复E12的错误. `units.py` 解析单位字符串,在27符号上加括号,并跟踪六个SI基维,并有理性的注释. 角是无维度的,如SI. 测试套件对照安装了天体的天体对每个符号进行交叉检查. 校验拒绝不同维度的总和、维度论证的超功能、以及其表达维度与其申报的`unit`不同的目标或约束。 评审先将名字转换为SI-一致性值,所以`x [km] + y [m]`是正确的,然后将每个结果转换回其申报的单位.

**Provenance and locking.** Every objective and constraint has an `origin`:
`user`, `compiler` or `agent`. `locked` defaults to `origin == "user"`. Only
user constraints may be locked. A user may unlock one of theirs, and an agent
may then revise it. `hard_constraint_fingerprint()` digests the user
objectives and the locked constraints. It also digests every variable domain
(kind, bounds, choices, unit) and every parameter (value, unit). The task
asked only for the constraints and the objective. That is not enough: an agent
that raises `dv_max` from 4000 to 9000 m/s weakens `dv <= dv_max` without
touching its text, and widening a bound is a weakened hard constraint too. The
fingerprint is order-insensitive and ignores initial guesses, budget,
verification policy, descriptions, metadata and non-user items. An agent can
therefore add a constraint that tightens the problem and keep the fingerprint.

**Digest.** The digest is the SHA-256 of canonical JSON: sorted keys, compact
separators, `allow_nan=False`. Every number is rounded to 12 significant
digits first, and `-0.0` becomes `0.0`. `problem.digest()` covers every
semantic field and leaves out `title`, the descriptions and `metadata`.
Variable, objective and constraint order is part of the digest because it
fixes the solver's vector layout. A pinned digest in the test suite holds the
canonical form to this ADR. The worked example's digest is the same on
macOS arm64 and Linux x86_64. The residual risk: noise that straddles a
12th-digit rounding boundary still splits two digests. Compare the numbers
before calling such a split a real change.

**Results keep four independent statuses.** `OptimizationResult` carries
`problem_digest` and `solver_id`, a `Termination` (`converged`,
`max_iterations`, `max_evaluations`, `timeout`, `infeasible`, `error`,
`cancelled`), and the booleans `solver_converged`, `constraints_satisfied`,
`independently_verified` and `optimality_certified`. All four default to false
and nothing derives one from another. It also records the best `x`, objective
values, canonical residuals, evaluations, iterations, wall time, a message and
an `evidence` list. Each evidence entry has a kind (`solver`, `analytical`,
`reference_solution`, `alternate_solver` or `optimality_certificate`), a
summary and data.

## 接受阈值(拟议,此处冻结)

`VerificationPolicy`默认为E14接受阈值,除非有问题说明自己的:

| 阈值 | 默认 | 含义 |
|---|---|---|
| `required` | `solver_converged`, `constraints_satisfied`, `independently_verified` | 接受必须真实的状况。 `optimality_certified`是选入的,因为大多数解析者无法证明. |
| `constraint_tol` | `1e-6` | A constraint or bound holds when its violation is at most `1e-6 * max(1, abs(rhs or bound))`. An integer must be within `1e-6` of an integer. `OptimizationProblem.constraints_satisfied` is the single implementation. IR 1.1 replaces the scale with `tolerance_scale`; see the addendum. |
| `objective_rtol` | `1e-6` | `abs(f - f_ref) <= 1e-6 * max(1, abs(f_ref))`时, |
| `optimality_gap` | `1e-4` | 最佳性证书必须缩小相对差距。 |
| `independent_methods` | `analytical`, `reference_solution`, `alternate_solver` | 允许设置 `independently_verified`的方法. |
| 文摘量化 | 12 重要数字 | 更改为 IR 版本 。 |

## 后果

- IR没有从轨道opt进口其他任何东西,也不需要更多
  jsonschema (英语). 测试表明这一点。
- 一条规则只在两层都需要时才会出现两次:
  `validate_problem`,因为Python 从未制造问题
  通过计划。 其他规则都只住在`validate_problem`.
- 无法在此检查工具的输出单位 。 IR相信申报
  `unit`,`ToolCall`目标或约束,E14-04登记册有
  检查它与工具的宣布输出。
- `rhs`是限制单位的一个数字. A. 对一国的限制
  参数与左侧的参数一起写,如`dv - dv_max <= 0`.
- 分明的选择是数字. 分类选择需要字符串值
  变量,无法使用算术表达式。 他们已经超出了范围
  1.0.

## Addendum: IR 1.1 (2026-11-26)

Four E14 workers (the compiler, the solvers, the tool registry and the
guard) hit the same limits of 1.0 on their own. IR 1.1 changes the IR once
and migrates every caller in the same change, with no compatibility layer.

**指数化的金额。** `rosenbrock-d20`无法写入:未滚动,其目标超过500节点. 现在,一个表达式可以为已宣布的Scalar `x<k>`写`x[k]`,为总和大于整数指数写`sum(<term> for i in range(a, b))`. 指数和范围界限是整数算术与字数并附指数。 剖析法将每个总和扩展为平衡的加词树,用不变的白单检查那棵树,所以没有消费者看到新的语法. 源树保持其500 - 节点和深度-200限制,扩展树本身的极限为20000节点. 扩展在建时算数,所以工程以极限而不是范围为界限. 一种矢量可变型是另一种选择。 它会改变解析器矢量布局,`classify`,守护和等效检查,索引的scalars不需要这些. `sum`和`range`成为保留国名.

**工具电话。** `ToolCall`增益`constants`和`output`. `constants`是JSON的数值,没有改变地传递到工具上,是文摘的一部分. 字符串integrand,规则名称或层厚度的数组现在可以到达一个工具,集成案例不再在工具ID中编码该integrand. `output` 将字段命名为从返回对象的工具中读取,该工具取代注册的 `value_field` 。 `evaluate`内部或周围的每个工具的失败都是`ToolCallError`,是`EvaluationError`的子类,所以一个解答器跳过一个点,而这个点的工具拒绝它而不是停止.

**Tolerance scale.** `max(1, abs(rhs))` let a small right-hand side drown in
the tolerance: `error <= 1e-10` accepted errors up to `1e-6`. 1.1 scales
`constraint_tol` by `tolerance_scale(rhs)`, which is `abs(rhs)`, or 1 when
`rhs` is 0 and carries no magnitude. Bounds and choices follow the same
rule. An explicit per-constraint scale was the alternative. It would put the
fix in every author's hands, and the default would stay wrong. The
hard-constraint fingerprint keeps its shape, because the rule belongs to the
version and not to a field. A 1.0 constraint with `0 < abs(rhs) < 1` is
graded more strictly under 1.1.

**结果。** `Termination`增益`unbounded`和`stalled`,解析者报告为`error`. `OptimizationResult`增益`front`(Pareto指NSGA-II用于提供证据数据)和`provenance`(`Provenance.cache_key`). 缓存记录是结果的一部分,所以`result_digest`将其钉住,运行目录移动到格式`orbitopt.optimization.run/2`. `Evidence`增益`producer`和`problem_digest`. 结果里面的一件物品 无法标出结果的文摘 所以证据被钉在结果旁边 仍然在警卫的证据类型中

**Public helpers.** `Expression.tree`, `SCALAR_FUNCTIONS`,
`Variable.domain_violations`, `Constraint.violation`, `Constraint.allowance`,
`tolerance_scale` and `to_plain` replace the private names that `classify`,
the KKT certificate, the guard and the equivalence check imported or copied.

**版本与文摘.**文件携带`schemaVersion` `"1.1"`. `PROBLEM_SCHEMAS`和`RESULT_SCHEMAS`地图,每个版本仍然标注其图案,`parse_problem`和警卫选择文件名称的图案. A 1.0 文档解析, 而其缺失的 1.1 字段会接受它们的默认值 。 文摘散列了1.1表,`schemaVersion`包括其中,因此与同一份文件的1.0文摘不同. 12显著数字的量化不变,因此文摘仍然在机器之间达成一致. 固定测试文摘随版本一同移动.

**Outside the IR, done in the same change.** `StructuredOutputError` carries
the usage of the replies it cost. The OpenAI-compatible provider sends a
dotted tool id such as `math.erf.v1` as the function name `math-erf-v1`,
which matches `^[a-zA-Z0-9_-]{1,64}$`, and maps the name back on return.

**Not done.** The guard asked for a solved-problem digest in the result's
provenance. `problem_digest` already names the problem the solver ran, so a
second field would only be able to disagree with it. A result's own evidence
does not carry `result_digest`, for the reason given above.
