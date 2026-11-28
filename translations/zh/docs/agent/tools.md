# 科学工具登记册

`orbitopt.agent.tools`拥有一个代理人,或IR `ToolCall`目标或约束可以称为的科学功能. 每个工具都声明其计划,单位,假设,有效性域,成本以及验证它的测试. 呼叫者只能通过允许的列表视图到达工具 。 每一个呼叫都位于一个散列链日志中,`replay`重新执行该日志在新登记册上. 导入包装 `jsonschema`和优化 IR. 它装入NumPy,只有当一个需要它们的工具运行时才切入.

## 调用工具

```python
from orbitopt.agent.tools import CallBudget, Caller, builtin_registry, replay

registry = builtin_registry()
view = registry.view(["math.erf.v1", "numerics.quadrature_cost.v1"])
agent = Caller("planner", CallBudget(max_calls=100, max_evaluations=10_000))

view.call("math.erf.v1", {"x": 0.5}, caller=agent)        # {'value': 0.5204998778130465}
view.call("numerics.quadrature_cost.v1",
	{"integrand": "exp(-t**2)", "a": 0, "b": 1, "rule": "gauss_legendre", "n": 7},
	caller=agent)                                          # value, abs_error, reference, ...

text = view.log.to_json()                                  # one JSON document
assert replay(view.log, builtin_registry()).ok             # same outcomes, bit for bit
```

给一个特工一个`ToolView`, 从来没有`ToolRegistry`. 视图没有跳过日志的调用路径 。 `view.list()`返回代理机可能使用的规格,`ToolSpec.to_dict()`给出他们的JSON表以迅速.

`ToolView.call(tool_id, args, *, caller)` runs these steps in order. The
call is logged whatever the outcome, and a failure raises `ToolError`.

1. 拒绝许可名单外的ID,拒绝使用`max_calls`的呼叫者。
2. Check that the arguments are JSON with no NaN or infinity, then validate them against `input_schema`.
3. 将 `integer` 字段中的完整浮点数转换为 `int` , 因为 IR 表达式总是评价为浮点数 。
4. 在工具运行前拒绝在已声明的 `domain` 外的值 。
5. Refuse the call when its declared evaluations would take the caller past `max_evaluations`.
6. 运行工具。
7. 校验结果对照 `output_schema`. NaN, 无限值或非 JSON值是错误, 永远不返回 。
8. 如果通话经过`max_wall_s_per_call`,则放弃结果.

## 错误代码

打电话者在`ToolError.code`, 一个`ToolErrorCode`。

| 代码 | 升为 | 日志中的结果 |
|---|---|---|
| `invalid_spec` | `register` :光谱打破了`spec_errors`检查的合同. | 不是电话 |
| `duplicate_tool` | `register` : id已经注册 | 不是电话 |
| `signature_mismatch` | `register`:可调用的参数与输入schema的属性不同,或其默认值与schema的可选字段不同 | 不是电话 |
| `unknown_tool` | `call`:没有工具有此ID | `refused` |
| `not_allowed` | `call`:工具存在,但视图不允许 | `refused` |
| `invalid_args` | `call` : 参数失败输入方案或不是有限 JSON | `error` |
| `out_of_domain` | `call`:一个输入在`domain`之外,或者该工具升起的`DomainError`. | `error` |
| `tool_failed` | `call`:工具提出任何其他例外 | `error` |
| `invalid_output` | `call` :结果失败输出方案或不是有限 JSON | `error` |
| `budget_exceeded` | `call`:电话会经过`max_calls`或`max_evaluations`. | `refused` |
| `timeout` | `call`:通话时间长于`max_wall_s_per_call`. | `refused` |
| `unit_mismatch` | `bind_problem`:一个维度不同 | 不是电话 |

`error`结果只取决于工具及其参数,因此重放重播后重播,并期望同样的代码. `refused`的结果取决于会话的许可列表和预算,所以重播会跳过它.

## 什么Spec申报

| 外地 | 含义 |
|---|---|
| `tool_id` | 以版本结尾的点缀小写名称, 如 `math.erf.v1` 。 已更改的工具会获得新的标识 。 每个有效的ID也是有效的IR `ToolCall.tool_id`. |
| `input_schema`, `output_schema` | JSON Schema 对象. 输入方案必须设置 `additionalProperties: false` 。 |
| `input_units`, `output_units` | 每个数字字段一个SI-的连贯单位字符串,无维度的`""`. 数组的单位适用于每个元素。 1以外的一个单位,如`km`,则被拒绝. |
| `assumptions` | 模型假设为文字. |
| `domain` | 每个数字输入一个闭合的`Interval`. 测试核实了这些范围内的工具。 一个工具将`DomainError`提出一个间隔不能表达的前提条件,如不稳定的显性时间步骤. |
| `cost` | `CostModel(cost_class, evaluations)`. `cost_class`为`trivial`,`cheap`,`moderate`或`expensive`. `evaluations`是一个IR在所需scalar输入(如`"3*n"`)上的表示,是收取到`CallBudget.max_evaluations`的金额. |
| `deterministic` | `False`标注了一种调味工具. 结构化工具必须使用所需的整数 `seed` 。 |
| `replay_tolerance` | 重播的相对耐受性允许,比例尺为1. `0`表示犬科结果必须相等. |
| `tested_by` | 对照独立引用验证工具的测试标识 。 |

## 呼叫记录

Each `CallRecord` stores `seq`, `tool_id`, `caller`, the canonical `args`,
the `result` or the `error` (`{"code", "message"}`), the `evaluations`
charged, `wall_s`, `prev_digest` and `digest`. The digest is
`canonical_digest` of every other field, which includes the previous
record's digest. `CallLog.verify()` therefore reports any record that was
edited, dropped or moved. `canonical_digest` rounds numbers to 12
significant digits, the rule the IR digest uses. Last-bit noise from another
machine does not change a digest. A change in the 12th digit does.

日志记录了论据和结果。 JSON 漂浮文本绕行一个双位位,所以在录制输入上重播.

`replay(log, registry)`返回一个 `ReplayReport`. 它的`integrity`字段列出了`verify()`发现的问题. 它每张唱片也持有一张`ReplayEntry`,每个条目都有一个这样的状态.

| 状态 | 含义 |
|---|---|
| `exact` | 结果是比特表示记录的结果,或者相同的错误代码重复出现。 |
| `quantised` | The results agree at 12 significant digits, which is what another machine's libm or LAPACK gives. |
| `within_tolerance` | 结果在工具的`replay_tolerance`中达成一致. |
| `not_executed` | 电话被拒绝,所以重新播放跳过. |
| `mismatch` | 还有什么 `detail`命名了第一个不同字段. |

`report.ok` is true when `integrity` is empty and no entry is a `mismatch`.
A forger who rewrites a result and then recomputes every digest passes
`verify()`. The forger still fails replay, because replay re-executes the
call.

## 优化问题中的工具

`bind_problem(problem, view, caller)`返回`tools=`所拍摄的`OptimizationProblem.evaluate`映射. 在运行前,它检查每个`ToolCall`的目标和约束.

- 工具必须在视图中。
- 呼号的`args`和`constants`必须绑定所有需要的输入,没有未知输入.
- `args`的表达式必须输入一个scalar数字输入,其维度必须等于输入所申报的单位.
- 一个常数未变地到达工具,在工具自己的SI单元中,输入计数在调用运行时进行检查. 字符串图案、规则名称或数组将在此进行。
- `output`必须指定一个需要的scalar数字输出,其维度必须等于`unit`,问题为这个目标或约束声明.

检查失败后, `ToolError` 和 `unit_mismatch` 或 `invalid_args` 升起 。 每个解析器返回工具的全部输出对象,IR读取`output`命名的字段. `ToolError`在`evaluate`内部发出呼号,如`out_of_domain`拒绝,到达呼号者为IR的`ToolCallError`,是`EvaluationError`的子类,`ToolError`是它的事业. 因此,解决者将这一点视为不可行并继续下去。 本检查涵盖[ADR optimization-0001](../adr/optimization-0001-optimization-ir.md)的开放点. IR信任在`ToolCall`上申报的单位,所以一个经过`validate_problem`的问题仍然可以宣布`W`是一个返回kelvin的工具. 评估者在调用前将参数转换为SI,所以一个分钟的变量以秒数到达工具. 每次通话都要通过`view.call`,所以解答者的评价被记录下来并编入预算.

```python
problem = OptimizationProblem(
	id="bessel-j0-minimum", variables=(Variable("x", lower=2.0, upper=5.0),),
	objectives=(Objective("j0", ToolCall("math.bessel_j0.v1", {"x": "x"}, output="value")),))
tools = bind_problem(problem, view, agent)
minimize_scalar(lambda x: problem.evaluate({"x": x}, tools=tools).objectives["j0"],
	bounds=(2.0, 5.0), method="bounded")                    # x = 3.8317..., the first zero of J1
```

## 内建工具

`builtin_registry()`在本表中注册工具. 命令`pytest tests/test_e14_04_tool_registry.py`对照表格名称的引用检查每个工具. 该参考文献没有与该工具共享代码。

| 工具 | 计算 | 测试中的参考文献 |
|---|---|---|
| `math.erf.v1`, `math.erfc.v1`, `math.gamma.v1`, `math.lgamma.v1`, `math.log1p.v1`, `math.expm1.v1` | 一个真实参数的功能, 通过 Python的 `math` 模块 | 20 - 数字 mpmath 值, 最多为 1e-13 的相对错误 |
| `math.bessel_j0.v1`, `math.bessel_j1.v1` | J0(x)和J1(x),通过`scipy.special`. | 20 - 数字 mpmath 值, 最多为 1e-13 的相对错误 |
| `numerics.gauss_legendre.v1` | N- 点高斯- Legendre 规则, 有错误估计值 \|_ n - _ 2 n\| | 2 n - 1 的 Polynomial 精确度,以及 0.1 内部对 e^t 真实错误的估算% |
| `numerics.adaptive_simpson.v1` | 在评价上限内使辛普森适应绝对容忍 | 1e-6和1e-10 |
| `numerics.quadrature_cost.v1` | One rule of the E14 integration family, with exactly n evaluations, and its absolute error against a QUADPACK reference | All seven E14 integration instances: the scan returns the verified minimum N and rule, and each reference matches its closed form to 1e-13 |
| `heat.layered_wall.v1` | 平稳的热通量通过层层 与对流在两面, 面部和界面温度 | 手计算 和每张胶卷和每层相同的通量 |
| `heat.transient_slab.v1` | 对于面部在固定温度下保持的板块,Theta-scheme 有限差异 | Fourier系列解决方案,其测量到的趋同顺序为空间2,以及时间1(隐形)或2(Crank-Nicolson). |

每个Spec列出其模型的假设和有效性域,因此`view.get(tool_id).to_dict()`描述工具. 内置工具为标准教科书模型.

## 添加工具

1. 写入一个纯函数,该函数使用关键词参数并返回一个 JSON 对象。
2. 用新版本的代号写一个`ToolSpec`. 将 `tested_by` 设定为将函数与独立引用相比较的测试 。
3. 将 `Tool` 添加到 `TOOLS` tuple 一个模块在 `orbitopt.agent.tools.builtin`. 对于新模块,在`builtin/__init__.py`中将其`TOOLS`加入`BUILTIN_TOOLS`.

`test_every_builtin_is_sound_and_names_tests_that_exist`当一个光谱中断合同,或者`tested_by`命名一个不存在的测试时失败.
