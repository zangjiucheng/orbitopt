# 优化 IR

`orbitopt.optimization` describes an optimization problem as typed, frozen
data. The same description serves the natural-language front end, the solvers,
the scientific tools, the agent runtime and the verification guard. Each
problem has a JSON form (schema `optimization-problem-1.1.json`) and a digest
that comes out the same on every machine. The decisions behind the shape are
in [ADR optimization-0001](../adr/optimization-0001-optimization-ir.md), and
its IR 1.1 addendum records what changed from 1.0.

## 问题多大啊

| 编 | 含义 |
|---|---|
| `variables` | `continuous`和`integer`取`lower`/`upper`(`null`离开一侧),`binary`为{ 0,1},`discrete`取`choices`之一. 每个都有一个`unit`和一个可选的`initial`猜测. |
| `parameters` | 固定的命名常数与单位。 |
| `objectives` | 一个或多个。 每个词都有表达式,一个`sense`(`minimize`或`maximize`)和一个`unit`. |
| `constraints` | 硬约束`expression <sense> rhs`与`sense`之一 `<=`,`>=`,`==`. `unit`覆盖两侧. |
| `budget` | 可选限制:`max_evaluations`和`max_wall_s`,每个解码器执行,`max_iterations`(代理循环迭代)和`max_tokens`(LLM符),只有代理运行时间读. 没有预算字段设置后端自身的迭代限制. |
| `verification` | 哪些结果要求接受,以及后面的门槛。 |

## 表达式

一个表达式要么是一个安全算术字符串,覆盖已声明的名称,例如`"x**2 + y**2"`,要么是一个工具引用. 字符串可能使用数字、名称、 `+ - * / **`、 unary 减和函数 `abs sqrt exp log log10 sin cos
tan asin acos atan atan2 sinh cosh tanh min max`。在运行前,其他任何信息都会被拒绝:属性、字符串、比较、羊肉、关键词参数、以 `_`开头的名字,以及100的字面说明。

两张表格写了一个很长的和,没有写出来:

- `x[k]`读了所申报的scalar `x<k>`,所以`x[3]`是变数`x3`.
- `sum(<term> for i in range(a, b))` 在每个整数`i` 上加上`<term>`.
  `a`到`b - 1`. `range(n)`开始于0. 苏姆斯巢 和作为
  值是该数字。

指数或范围绑定值是整数算术(`+ - *`)大于整数立体,并附有总和指数。 20 - 维度 Rosenbrock 函数是一个表达式:

```text
sum(100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(19))
```

剖析法将每个总和和索引扩展为简单的语法,用相同的白名单检查结果,因此评价,单位和`classify`从看不到一个总和. 两个限制都保留了工作界限,两个限制都计算Python的语法树的每个节点,操作符和名称上下文包括:

| 限制 | 数值 | 这很重要 |
|---|---|---|
| `MAX_NODES` | 500 | 源树如文,扩张前. 其巢穴深度最多为200. |
| `MAX_EXPANDED_NODES` | 20000 | 每个和指数之后的树被扩展. 扩展值在极限时会计算,所以`range(10**9)`成本不超过极限. |

`sum`和`range`是保留的:任何变量或参数都不得取任一名称. `Expression.tree`返回扩展树,用于读取表达式结构的代码.

`{"tool_id", "args", "constants", "output"}`是一个工具参考:

| 外地 | 含义 |
|---|---|
| `tool_id` | 一个注册工具,例如`numerics.quadrature_cost.v1`. |
| `args` | 工具输入到表达式源,以 SI 单位评价. |
| `constants` | 工具输入到 JSON 常数中, 不改变: 字符串、 布尔、 有限数字或数组 。 它与`args`没有密钥,是文摘的一部分. |
| `output` | 输出字段从返回对象的工具读取。 `null` 用于返回数字的工具。 |

一个工具内部或周围的每个故障都会以`ToolCallError`,`EvaluationError`的子类到达呼叫者,以工具本身的例外作为其原因.

单位按尺寸检查. 长度和时间的总和被拒绝,当它的`unit`是一个时间时,其表达是长度的制约也是一样. 数值在评价前转换为SI,因此`x [km] + y [m]`评价正确.

## 宽容

A constraint holds when its violation is at most
`constraint_tol * tolerance_scale(rhs)`. The scale is `abs(rhs)`, or 1 when
`rhs` is 0. Bounds and discrete choices use the same rule, and an integer
must lie within `constraint_tol` of an integer. With the default
`constraint_tol = 1e-6`, `error <= 1e-10` accepts errors up to
`1e-10 * (1 + 1e-6)`, and `g <= 0` accepts `g` up to `1e-6`.
`Constraint.allowance(tol)` and `Variable.domain_violations(value, tol)` state
the rule once. `domain_violations` returns why a value is outside a domain:
`below_lower`, `above_upper`, `not_integer` or `not_a_choice`.

## 来源和指纹

每个目标和约束都记录着它的`origin`:`user`,`compiler`或`agent`. 用户限制默认为`locked`. `hard_constraint_fingerprint()`在用户目标,锁定约束,可变域或参数变化时都会改变,因此守护者可以捕捉削弱用户要求的代理.

## 工作实例

Minimise x² + y² subject to x + y ≥ 1 and x, y ≥ 0. The optimum is 0.5, at
x = y = 0.5.

```json
{
  "schemaVersion": "1.1",
  "id": "doc-quadratic-halfplane",
  "title": "Closest point to the origin on x + y >= 1",
  "variables": [
    {"name": "x", "lower": 0},
    {"name": "y", "lower": 0}
  ],
  "objectives": [
    {"name": "distance_sq", "expression": "x**2 + y**2"}
  ],
  "constraints": [
    {"name": "halfplane", "expression": "x + y", "sense": ">=", "rhs": 1}
  ]
}
```

```python
import json
from orbitopt.optimization import parse_problem

problem = parse_problem(json.load(open("problem.json")))   # raises InvalidProblemError with every error
point = {"x": 0.5, "y": 0.5}
ev = problem.evaluate(point)
ev.objectives                                     # {'distance_sq': 0.5}
ev.residuals                                      # {'halfplane': 0.0}  (g = rhs - lhs <= 0 holds)
problem.constraints_satisfied(point, ev.residuals)  # True
problem.digest()                                  # identical on every machine
```

`validate_problem(problem)` 不提升返回错误列表 。 每个错误都有一个`ErrorCode`,一个JSON-指针`path`和一个消息.

## 结果

一个解答者报告答案为`OptimizationResult`. 这个结果包含问题的文摘和四个单独的状态布尔:`solver_converged`,`constraints_satisfied`,`independently_verified`和`optimality_certified`. 四者从假起. 它还认为:

| 外地 | 含义 |
|---|---|
| `termination` | `converged`,`max_iterations`,`max_evaluations`,`timeout`,`infeasible`,`error`,`cancelled`,`unbounded`(后端报告目标无约束改善)或`stalled`(后端在收缩测试通过前停止进展). |
| `front` | 一个多目标解决方案的非主导性可行点,每个`ParetoPoint(x, objective_values)`. 否则就空了 |
| `provenance` | `Provenance(cache_key)` :问题消化一个缓存存储结果在下面,或者`None`作为新解. 它是结果的一部分, 所以结果消化 贴它。 |
| `wall_time_s` | 华尔街的破案时间 其挥发性很强,因此结果消化器会将其抛出(`VOLATILE_KEYS`). |
| `evidence` | 每个项目都有`kind`,`summary`,`data`,其`producer`(一个解码器id,工具id或方法)和它为之制作的`problem_digest`. |

## 版本和摘要

`to_dict`写着`schemaVersion` `"1.1"`. `parse_problem`对照其`schemaVersion`名称(`PROBLEM_SCHEMAS`)检查一个文件,所以1.0文件仍然被描述. 它所缺少的每个 1.1 字段都需要它的默认值:没有工具常数,没有输出,没有空前方和缓存密钥. 通过`RESULT_SCHEMAS`, 1.0文档,命名一个变量`sum`或`range`不再验证.

两种1.1规则改变一个不变的1.0文件的含义:

- 对`0 < abs(rhs) < 1`的限制和界限进行更严格的分级,
  because the tolerance scale is `abs(rhs)` and no longer `max(1, abs(rhs))`.
- 文摘散列1.1形式,包括`schemaVersion`. 一个问题
  1.1文摘因此与文摘不同,同一份文件的文摘如下:
  1.0,测试套房将1.1值固定. 文摘仍然环绕每个数字
  to 12 significant digits, so they agree across machines.
