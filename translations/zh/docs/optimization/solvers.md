# 解决方案

`orbitopt.optimization.solvers`把每个解答者都放在一个呼叫后面. 你将同样的[IR problem](ir.md)交给任何兼容的解析器,并总是拿回`OptimizationResult`. 每个解答者都声明它能承受什么,`orbitopt.optimization.classify`从IR中获取问题的结构,因此代理运行时间和基于规则的选择者可以不运行任何东西而选择解答者.

```python
from orbitopt.optimization import Budget, parse_problem
from orbitopt.optimization.solvers import default_registry

registry = default_registry()
problem = parse_problem(data)
for solver in registry.compatible(problem):            # installed and capable, most specialised first
    result = solver.solve(problem, budget=Budget(max_evaluations=4000), seed=1)
    print(solver.capability.solver_id, result.termination.value, result.objective_values)

registry.solve_with(problem, "scipy.highs")            # the same call, by id
```

`solve(problem, *, budget=None, seed=0, x0=None, options=None, tools=None)`
takes `problem.budget` when `budget` is `None`. `x0` maps variable names to a
start, and `tools` maps `ToolCall` ids to callables, as
`OptimizationProblem.evaluate` takes them.

## 结构:`classify`的决定

`classify(problem)`返回一个`ProblemStructure`:每个目标一个`Form`,最一般的制约形式,以及问题是否具有整数,二进制或离散变量,平等约束和每个变量的有限界限. `domain_size`是变量产品域的点数,当每个变量只取有限的许多值(二进制值,离散值,或带有两个边框的整数),`None`则别. 此决定精确于表达式语法树. 每个表达式被扩展为具有合理系数的多诺米. 变量中任何非多诺态的子树都保持不透明原子. 没有任何函数被取样 。

| 表单 | 含义 | 实例 |
|---|---|---|
| `LINEAR` | 扩展后最多为 1 | `x/2`, `(x+1)**2 - x**2`, `sin(x) - sin(x)` |
| `QUADRATIC` | 2学位 | `x*y`,`x**k` 参数为`k = 2` |
| `SMOOTH` | 高度多诺米, 或仅由域内平滑函数构建的原子 | `x**3`, `1/x`, `x**0.5`, `exp(x)` |
| `NONSMOOTH` | 原子对变量应用 `abs`, `min`或 `max` | `abs(x)`, `max(x, 1)` |
| `BLACK_BOX` | a `ToolCall` | |

标题`ProblemClass`遵循这一优先性:`multi_objective`,`black_box`,然后`milp`或`minlp`为整数问题,然后`nonsmooth`,`lp`,`convex_qp`(其黑森州通过其意义上的精确合理的正-半限测试,具有线性限制),`box_smooth`(除界限外没有限制)和`nlp`.

可判定性,并报告:线性,四极性,四极目标对流性,可变类型,界限,约束感和客观计数. 不可判定,而且从未声称:非线性问题是凸轮还是多式联运,以及工具的计算。 没有"全球"类,因此基准的Rastrigin和Acley例被归类为`box_smooth`. 两个限制使答案保持保守,但绝不会出错. 不使用直译身份,所以`sin(x)**2 + cos(x)**2`读作`SMOOTH`. 超越`MAX_TERMS`(2000)术语的扩展成为原子. `SMOOTH`不承诺一个参数到达域边界的衍生物:`sqrt(x**2)`读作平滑,虽然它等于`abs(x)`.

## 内建解析器

| `solver_id` | 带 | 整数 | 制约因素 | 所需边界 | 趋同意味着 |
|---|---|---|---|---|---|
| `scipy.highs` | 线性,一个目标 | 是( 扩展为二进制) | 是的, 与平等 | 无 | `certificate` |
| `scipy.slsqp` | 光滑, 一个目标 | 无 | 是的, 与平等 | 无 | `first_order` |
| `scipy.lbfgsb` | 光滑, 一个目标 | 无 | 只限 | 无 | `first_order` |
| `scipy.differential_evolution` | 任何东西,一个目标 | 对 | 是的, 与平等 | 对 | `heuristic` |
| `cma.cma_es` | 任何东西,一个目标 | 无 | 只限 | 对 | `heuristic` |
| `pygmo.nsga2` | 任何目的,几个目标 | 对 | 只限 | 对 | `none` |
| `orbitopt.enumeration` | 任何东西,一个目标,每个可变的有限域,最多2 ^ 20点 | 对 | 是的, 与平等 | 是(有限域) | `exhaustive` |

`SolverCapability`携带这些列加上`needs_gradients`,`is_global`,`deterministic_given_seed`,`iteration_cap`(后端有其自身的迭代限制),`options`,`max_domain_size`(由行走整个域的解码器设置:它只有在`domain_size`充其量时才会有问题)和`unavailable_reason`. 无法找到后端模块的解析器 。 其`unavailable_reason`命名为模块,`compatible`将其排除在外,并称其返回`Termination.ERROR`而不是升起. `cma`没有卖出,所以`cma.cma_es`在`pip install cma`之前是无法买到的. `rejection(structure)`用文字说明为什么一个解决者不能接受问题.

**Iteration limits.** Every adapter enforces `Budget.max_evaluations` and
`Budget.max_wall_s`. `Budget.max_iterations` is the agent runtime's loop cap
and no solver reads it. A backend's own iteration limit is a separate
argument, `solve(..., max_backend_iterations=N)`: `maxiter` for SLSQP,
L-BFGS-B, differential evolution and CMA-ES, and generations for NSGA-II.
Passing it to a solver without `iteration_cap` (HiGHS) raises `ValueError`.

`options`每个选项`solve`接受`OptionSpec`:其类型(`bool`,`int`,`float`或`str`),其封闭范围`[low, high]`或`choices`,`tolerance`,这标志着解决者(决不是问题容忍)与后端运行的`default`的趋同容. `solve` 提出 `ValueError` 选项时, 解析器不声明或超出其规格的值 。 代理运行时间验证一个计划员针对相同规格的选项,并且只放宽标有`tolerance`的选项.

| 解决方案 | 选项(黑体表示容忍) |
|---|---|
| `scipy.highs` | **`mip_rel_gap`** [0, 1e-4], default 1e-9; `presolve` |
| `scipy.slsqp` | **`ftol`** [1e-16, 1e-2], default 1e-10 |
| `scipy.lbfgsb` | **`ftol`** [1e-16, 1e-2], default 2.2e-9; **`gtol`** [1e-16, 1e-2], default 1e-5 |
| `scipy.differential_evolution` | `popsize` 4-100; **`tol`** [0, 0.1], default 0.01; `atol` [0, 0.1]; `mutation` [0.1, 1.9]; `recombination` [0, 1]; `polish`; `strategy`; `init` |
| `cma.cma_es` | `sigma0` [1e-12, 1e12]; `popsize` 4-100; **`tolfun`**, **`tolx`** [1e-16, 1e-2], default 1e-11 |
| `pygmo.nsga2` | `population` 8-1000 (a multiple of 4); `cr` [0, 0.99]; `eta_c` [1, 99]; `m` [0, 1]; `eta_m` [1, 99] |
| `orbitopt.enumeration` | `max_points` 1到2 ^ 20,一个整数:一个解的下盖 |

`compatible`由`max_form`,约束,平等与整数支持,`is_global`将有能力的解子从最小排列到最一般,不管解子是否行走整个域(它最后出现),然后`solver_id`. 命令不取决于登记顺序。 无法检测到多式联运问题,因此对其中之一来说,排名第一的解决者可能是本地方法. 选择一个全球性的解决方案是呼叫者的工作。

## 每个适配器都遵守合同

**One evaluation is one point.** Every point a backend asks about goes
through the adapter's `Evaluator`. It refuses the call that would exceed
`max_evaluations` or `max_wall_s` and raises before evaluating, so a budget
is never exceeded by even one evaluation. Finite-difference points count.
The objective and the constraints at the same point count once. HiGHS reads
coefficients, so it spends exactly one evaluation: the IR's evaluation of its
answer.

**Values come from the IR.** A maximised objective is negated for the
backend and reported un-negated. `objective_values`, `constraint_residuals`
and `x` are the IR's own evaluation of the reported point. For one objective
that point is the best evaluated point: clean feasible points by objective,
then the other feasible points by objective, else least violation (see
[the tolerance band](https://github.com/zangjiucheng/orbitopt/blob/main/docs/optimization/solvers.md#the-tolerance-band-is-not-room-to-optimise)). A point whose evaluation failed is never
chosen, so when every evaluation fails the result has an empty `x` and
`constraints_satisfied` is false, even on a problem with no constraints.
Otherwise `constraints_satisfied` comes only from
`OptimizationProblem.constraints_satisfied`. `independently_verified` and
`optimality_certified` stay false; the verification guard sets them.

**Termination, in order:** `error` (a backend exception, a backend that
reports failure, or every evaluation failing), then `unbounded` (HiGHS
proved the objective unbounded), then the adapter's budget
(`max_evaluations`, `timeout`), then `infeasible` (no feasible point found),
then `converged` (the backend's own flag), then the backend's own iteration
limit or `stalled` (an SLSQP or L-BFGS-B line search that failed, or a
CMA-ES stop that is neither convergence nor a limit), and otherwise `error`
with the backend's message. `solver_converged` is true exactly when the
termination is `converged`. A backend that claims success at an infeasible
point is reported `infeasible`. From HiGHS, `infeasible` is a proof. From
the other solvers it means they stopped without finding a feasible point.

**`solver_converged`值多少**取决于`capability.convergence`. `certificate`表示HIGHS在其差距内(默认为`mip_rel_gap` 1e-9)被证明是最佳的. `exhaustive`表示对限定域的每个点都进行了评价,因此所报告的点是约束容忍度范围内对任何目标的全球最佳. `first_order`表示KKT点,这只在对流问题上是最佳的. `heuristic`表示人口停滞,所以这个点可能是错误的:随着种子1,差分演化在`knapsack-p07-n15`的1456汇合,其最佳演化是1458. `none`表示NSGA-II没有趋同测试,从未报告趋同.

## 忍耐乐团是无法优化的

`constraint_tol`决定哪些点是可行的. 它没有使两个可行的要点与客观相比较。 使用乐团的一点通过它所购买的违反来击败精确的受限最佳:第一顺序,限制的拉格朗格乘数乘以违反. 差异演化将乐队视为它的束缚(`g <= tol`,`|h| <= tol`),因此它向乐队的边缘施压. SLSQP可以停止限制以外的有限偏差步骤. 第一次E14-10运动衡量了这两点:

| 实例 | 点 | 违反 | 乘数 | 乘数 x 侵犯 | 目标 - 最佳 |
|---|---|---|---|---|---|
| `cqp-projection-d2`: min (x-2)^2+(y-1)^2, x+y <= 1, f* = 2 at (1, 0) | SLSQP, (1 + 2^-26, 0) | 1.49e-8 | 2 | 2.98e-8 | -2.98e-8 |
| `cqp-equality-d3`: x+y+z = 1, f* = 6/11 at (6, 3, 2)/11 | 差异演变 | 9.54e-7 | 12/11 | 1.04e-6 | -1.04e-6 |

两个"击败"最佳 完全通过乐队让他们买什么。 第二个缺口大于警卫的约定容忍度(`objective_rtol = 1e-6`乘以最大(1,6/11)),因此候补的乐队点反驳了SLSQP的准确点. 有两条规则可以阻止,每个适配器:

1. **Clean first.** A feasible point is clean when no constraint uses more
   比其乐队的`SELECTION_FRACTION = 1e-6`(`band_fraction`). 最好的一个
   point ranks clean points by objective, then band points by objective. A
   干净点最多可以买百万分之一的乐队边缘可以:
   2.98e-14在`cqp-projection-d2`,1.04e-12在`cqp-equality-d3`.
   below the 1e-8 the spec asks of a QP.
2. **Restore, then compete.** When the best band point beats every clean
   一,适配器在报告前将它移动到它的制约上:
   `RESTORE_STEPS = 3`高斯-牛顿阶梯,每个解决`J dx = -g`
   坐标上方的`dx` `g` 持有一切平等
   剩余和每一次不平等 超过干净的门槛值,`J`
   comes from forward differences, one evaluation per coordinate. A
   协调步骤会推过一个束缚 被剪断并固定在那里。
   恢复的点通过相同的`Evaluator`,所以被报告.
   当它是干净的,击败 干净最好的,它的目标
   包括可行性成本。 在线性约束上,一步是准确的
   到圆形。

一个解决方案保留了`restoration_reserve`评价,`n`连续变量的`3 x (n + 1)`,从修复的后端返回,并且只针对一个制约和连续变量的问题,一个目标,以及至少是储备的四倍的预算. 一个较小的预算留下了后端留下的恢复. 在`message`, 整数坐标从不移动,一个束缚不被当作一个波段:这里的每一个后端都将其点保留在界限内.

在这两个例子中,10各有种子,E-规则与竞选相同(laptop,`benchmarks/bench_e14_agent_iteration.py`):

| 实例 | 之前: 已核实的提醒 | 后: 已核实的提醒 | 之后: 最多违反 | 之后: 间隙 |
|---|---|---|---|---|
| `cqp-projection-d2` | 10, 0 | 0, 0 | 0 | 0 |
| `cqp-equality-d3` | 9, 0 | 0, 9 | 0 | 2.2e-16 |

`cqp-projection-d2`保持未验证,因为其唯一的交替,有差异的演化,在2.00001汇合,这证明什么也没有. 第十支`cqp-equality-d3`种子为`budget_exhausted`:在0.54552分配时其候补停止. 4评价每一次衡量的恢复成本:SLSQP解析`cqp-projection-d2`(6评价前,10后)和`cqp-equality-d3`(一步:三个差分和该步骤的点,从12的储备中)的差分演化候补.

## 编号

`orbitopt.enumeration`通过`Evaluator`来评价一个有限域的每个点,所以每个点都算预算,一个`ToolCall`目标或约束被评价为任何其他解子,最佳点由IR自己的可行性选择. Discrete 坐标走选择的索引,像每个适配器的搜索矢量一样.

- 域名必须是有限的,最多为`MAX_ENUMERATION_POINTS`(1,048,576,
  2 ^ 20)积分. 否则`rejection`说为什么 `compatible`离开
  解析器: `a variable is continuous or has an open bound`,或 `its
  domain has N points, more than the 1048576 it enumerates`.
- 大于`budget.max_evaluations`或大于`max_points`的域是:
  第一次评审前被拒绝:终止`max_evaluations`,零
  评估,没有意义。 部分列举将证明无济于事。
- 在评估了每一点之后,报告`converged`。 没有可能
  标点终止为 `infeasible`, 即为证明。
- 一个在任何时间提出的工具使查点无法证实:
  终止是`stalled`,而不是`converged`. 表达方式以外的一点
  域名只是无法操作。

On `knapsack-p07-n15` (15 binaries, 32,768 points) it costs 32,768
evaluations and returns 1458, the optimum. Differential evolution with seed
1 converges at 1456 under the catalogue budget of 30,000.

**Seeds.** The same problem, seed and budget give the identical result for
every built-in solver. The deterministic ones ignore the seed.

**Malformed requests raise.** An invalid problem raises
`InvalidProblemError`, and an unknown option or `x0` name raises
`ValueError`. A well-formed request that cannot be carried out (backend
missing, structure outside the capability) returns `Termination.ERROR` with
the reason and zero evaluations.

**Multi-objective results.** NSGA-II reports the front member with the
lexicographically smallest objective vector as `x`. The whole non-dominated,
feasible front is `OptimizationResult.front`, one `ParetoPoint(x,
objective_values)` 每个成员。

**Evaluation failures.** A point where the problem is undefined, or where a
tool raised (`ToolCallError`), counts as an evaluation and is infeasible, so
the backend moves on. The solver evidence names its `producer` (the solver
id) and the `problem_digest` it solved.

## 接受书

`tests/test_e14_03_solver_registry.py`将37 E14-10目录例翻译为IR. 它在目录预算(2000评价每个维度)下运行每个兼容的可用解析器,并将答案与目录本身的调用和容忍度进行评分. 该译文保留了目录的索引总和: 它内含了`n`和scalar数据,并宣布每个数据矢量`s`为参数`s0, s1, ...`. 37所有例都是有效的IR,`rosenbrock-d20`作为一个索引总和.
