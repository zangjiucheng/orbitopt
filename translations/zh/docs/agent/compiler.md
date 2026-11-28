# 自然语言问题汇编器

`orbitopt.agent.compiler`将问题说明变为IR问题(`orbitopt.optimization.OptimizationProblem`). 模型从不写IR,也不写代码. 它通过结构输出来填充一个小的JSON对象,编译器从该对象中构建IR,验证它,并对照用户自己的单词检查它. 当编译器不能产生一个忠实的问题时,它会说为什么而不是猜测.

```python
from orbitopt.agent.compiler import Compiled, compile_problem
from orbitopt.agent.providers import LocalProvider

outcome = compile_problem("Minimize x^2 + y^2 subject to x + y >= 1 and x, y >= 0.",
                          LocalProvider("qwen2.5:7b"))
if isinstance(outcome, Compiled):
    problem = outcome.problem          # valid IR, user items locked
    outcome.assumptions                # everything the model added
    outcome.usage, outcome.attempts    # tokens over all rounds, rounds used
```

## 结果

`compile_problem` 准确返回五个值之一. 拒绝是一种价值,而不是例外。

| 结果 | `outcome` | 卡莉丝 | 何时 |
|---|---|---|---|
| `Compiled` | `compile` | `problem`, `assumptions` | 草稿是有效的IR,通过下面的每一张支票. |
| `NeedsClarification` | `needs_clarification` | `questions`,`partial`(原稿而非IR). | A value is missing, the wording is ambiguous, objectives conflict, or the text mentions something IR 1.0 may not express. |
| `Unsupported` | `unsupported` | `reason` (`category`, `explanation`) | 请求需要一个未知的工具或 模型 , 随机性, 动态, 软约束 , 一个 分钟- max 超过 无限 多例, 或 IR 以外的可变类型 。 |
| `RejectedUnauthorizedChange` | `rejected_unauthorized_change` | `violations` | 草稿更改了用户固定的东西,用户自己的单词不授权. |
| `CompileFailed` | `compile_failed` | `errors`, `raw` | 修缮预算内无有效草案. |

每一个结果都带有`usage`和`attempts`. `Compiled`拒绝持有`validate_problem`拒绝的问题,因此任何结果都不能携带无效的IR. `StructuredOutputError`以外的提供商失败宣传:`ProviderUnavailableError`,`BudgetExceededError`和`ReplayMismatchError`是基础设施故障,没有汇编结果.

## 两个计划,分开

模型填充`COMPILER_OUTPUT_SCHEMA`:

- `status`为`compiled`,`needs_clarification`或`unsupported`.
- `problem`拥有统一的变量、参数、目标和
  限制。 每件物品都有`source`,当所有者声明时,就是`user`.
  和`assumed`当模型加时. 一个表达式是 IR表达式
  字符串,它可能使用索引的金额,例如
  `sum(x[i]**2 for i in range(20))`,或工具呼叫
  `{"tool_id", "args", "constants", "output"}`. 融合案件通过
  以一个`quadrature.abs_error`的常数`integrand`表示
  工具。
- `questions`,`assumptions`和`unsupported_reason`让模型拒绝.
  或解释。
- `authorized_changes`列出`{"item": "constraint:<name>", "quote": ...}`.
  每个项目的修改。
- `tradeoff_quote`持持所有者言求取舍.

`draft_to_ir_dict`转换索引草稿,所以IR错误路径,如`/constraints/1/unit`命名为同一个草稿项目. 假定的目标或约束变为`origin="compiler"`,而不锁定。 每个用户项目都是`origin="user"`并锁定. 变量和参数在IR中没有来源,因此编译器列表假设在`metadata["assumptions"]`中. 每个假定项目都出现在`Compiled.assumptions`. 当模型给出一个项目的理由时,编译器会添加条目本身.

## 编译器检查什么

编译运行时间最多为`1 + MAX_REPAIR_ROUNDS`回合(`MAX_REPAIR_ROUNDS = 2`). 每回合都会打一个`generate_structured`电话. 提供商自发地重复了一次没有计划有效的JSON的回复. 然后由编译器按此顺序判断答复.

1. 模型报告`unsupported`,或草案外有变种.
   IR, 表示`soft`的制约, 或调用者没有的工具代号
   在`known_tool_ids`通过. 结果是`Unsupported`.
2. 模型 提问,或留下参数值或约束
   `rhs` 无效. 结果是`NeedsClarification`.
3. `parse_problem`拒绝IR,或一个变量打破一个`unit_hints`条目.
   编译器将打出的误差作为修补的转折送回,每转为
   `path: [code] message`.
4. 主人公的文字包含一个来自`UNSUPPORTED_MARKERS`("random,")的短语.
   "预期值","惩罚"等英文和中文版本),否
   假设,或两个目标冲突,或草案
   有几个目标没有提出取舍要求。 结果是
   `NeedsClarification`. 两个目标在价值观一致时发生冲突:
   每个样本点及其感官不同,或者当它们准确
   阴性同感.
5. 授权引文不在所有人的案文中,或修改
   保护项,没有保护项。 结果是`RejectedUnauthorizedChange`.
6. 问题中缺少所有者文本中的数值绑定 。 这个
   编译器将它作为修复转弯,编码为`uncovered_statement`.
7. 否则结果就是`Compiled`.

After the last round the result is `CompileFailed` with that round's errors.
Its `usage` sums every provider call, including one that ended in
`StructuredOutputError`.
The model's text only reaches `json.loads`. Its strings become IR field
values, and the IR's whitelisted evaluator is the only code that runs an
expression. A test walks the compiler's syntax trees and fails on any call to
`eval`, `exec`, `compile` or `__import__`.

## 授权

`split_prompt`将附着的材料从所有者的文本中移出. 材料是`attachments` 参数加上提示符中任何括号的块,这些块以“附着”,“附着”,“工具输出”,“日志”或附件、日志和工具输出的汉字开头。 模型看到材料,标签不可信. 从它引出的一句话从来不授权改变。

当你通过 `base_problem`, 编译是一个修订版。 `revision_diff`将结果与基项逐项比较. 它涵盖了`hard_constraint_fingerprint`保护的物品. 这些是锁定的限制,用户目标(删除,更改或添加),以及每个可变域和参数. 每个差异都需要`authorized_changes`条目。 条目的引用必须字面上出现在所有者的文字中,忽略大小写和间隔,并且必须至少四个字符长. 当更改设置新数字时,引号必须包含该数字. 因此,"让它更快地解决"并不授权放松`x + y >= 1`到`x + y >= 0.9`,"放松对x + y → 0.8的制约"确实授权将移动到0.8.

## 覆盖范围检查及其限度

覆盖检查阻止编译器默不作声地放弃一个声明的制约. `extract_statements`按图案在所有者的文本中找到数字界限:

- 比较者后有数字,如`>=`,`<=`,`=`,"至少","至少","至少".
  "不过"和"上", 和汉字至少,最多,
  不超过和不少于;
- 参照国面前的数字,如`0 <= x`;
- 时间间隔,如"A和B之间",`[A, B]`,以及中文形式;
- "非负性",读作`>= 0`;
- the number words zero to ten, and one Chinese numeral, after a comparator.

数字后的一个单位在`parse_unit`接受时读取,或者当它是少数中国单位词之一时读取. `uncovered_statements`在问题中需要每个号码. 当一个变量绑定或约束`rhs`有相同方向时,或者相反方向与相反的标志时,它就算. 一个参数值,一个离散的选择,或者一个表达式字面计算在两个符号中的任何一个. 答者不计. 模型的一个假设中的数字也算数,授权引文中的文字也算数。 当双方有相同尺寸的单位时,检查会将它们比作一个单位,所以一个说明的4 km/s不匹配4 m/s的束缚.

支票是一个数字存在 检查,它错过的东西。

- 它不知道一个数字属于哪个数量。 偶数相等
  在问题的任何地点都满足它。
- 它只发现有参照语句的界限. "规划1有4小时"
  和"包持有20千克"没有提取,所以一个草案 丢掉这些
  各种限制已经通过。
- 它把数字写成数字 零到十的单词
  中国数字报. "二十"不读.
- 没有单位的标注数字与任何单位的编译数字相匹配。
- 它检查业主的文本与草案相对,而不是相反. 用户
  模型 发明的条目仅在参考比较时被捕获
  或修正diff看到它。

下面的等值检查,以及E14-10对引用问题的评分,都是后卫.

## 与参考问题等效

`orbitopt.agent.equivalence.semantically_equivalent(compiled, reference)`在同一个种子点探测两个问题. 这些是256样本,最高为64箱角,通过`reference_x`时的参考最佳. 结果是`EquivalenceReport`。 它的`reasons`列出每一个分歧,它的`notes`记录的观察没有算作区别.

- 变量必须按名称匹配 。 重新命名的变量需要明确
  `aliases`条目. 支票从来就猜不出来 类型,单位
  维度和有限界限必须商定,不同规模的单位是
  转换。
- 目标必须具有相同的计数和感官. 他们的价值观必须同意
  SI在两个域的每个探测点.
- `OptimizationProblem.constraints_satisfied`的可行性判决必须
  每个探测点都同意 每种参考约束或有限约束
  剪切探针盒, 编译的制约或束缚必须保持准确的
  同样的点,反之。
- 工具调用必须匹配工具 ID,参数(与数字),感知,
  `rhs`和单位. 未执行的工具由同一工具取代
  双方的确定性不透明函数.

探测器来自两箱的结合,所以出现了一个拓宽的有限束缚. 一个问题的一方是开放的,另一个问题的一方是有限的,只比较到有限的一方,报告将其记录在`notes`。 这遵循了在实例框内判断FC-001的基准规则. 预算与验证政策没有比较. 检查样本。 它报告的两个问题相当 当它们只在一组探测器上出现分歧时 它报告说,平等因不同而分为两种不平等。

## 测试

`tests/test_e14_02_compiler.py` runs the 57 cases in
`tests/data/agent/formulation_cases.json`, eight of them in Chinese. The
cases use the E14-10 formulation case format with one extra outcome,
`compile_failed`. Each case scripts a `MockProvider` with realistic replies.
The test compares each outcome with `expected_outcome`, and each compiled
problem with its reference IR through `semantically_equivalent`. Every
catalogue reference IR is also checked against the benchmark's own Python
callables.

这些测试用脚本回复来练习编译器. 他们只字不提模型的真实语言 如何准确地表达问题。 E14-10用真正的模型测量.
