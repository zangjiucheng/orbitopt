# 优化

`orbitopt.optimization`将一次优化问题描述为有稳定文摘的输入数据,在没有LLM的情况下解决和分级. [LLM-assisted workflow](../agent/index.md)以此为基础:编译器写下这个IR,代理运行时间用这些解析器和这个守护器解决.

| 页次 | |
|---|---|
| [Optimization IR](ir.md) | 问题和结果文件,表达式,单位,容忍度和摘要. |
| [Solvers](solvers.md) | 注册,`classify`决定,每个解决者的能力和选项,以及每个适配器所保存的合同. |
| [Verification guard](verification.md) | 四个状态, 设置它们的证据, 密封运行和故障矩阵。 |
| [ADR optimization-0001](../adr/optimization-0001-optimization-ir.md) | Why the IR has this shape, and what IR 1.1 changed. |

## 安装

The IR, the classifier and the guard need only the base install. The
solvers need numpy, scipy and, for CMA-ES, `cma`:

```bash
pip install -e '.[solvers]'
```

NSGA-II需要pygmo,它来自康达. 锁定的conda环境(`envs/cpu.conda-lock.yml`)有NumPy,scipy和pygmo,但没有`cma`.

## 从Python

```python
import json
from orbitopt.optimization import default_registry, parse_problem, verify_result

problem = parse_problem(json.load(open("problem.json")))
solver = default_registry().compatible(problem)[0]
result = solver.solve(problem, seed=1)
report = verify_result(problem, result)   # no evidence yet: independently_verified is not_assessed
report.verdict, {name: s.outcome.value for name, s in report.statuses.items()}
```

包根再导出分类器,解析器注册器和护卫懒惰,所以`import orbitopt.optimization`只装入IR.

## 从命令行

```bash
orbitopt optimization validate problem.json    # errors, or digest, class and solvers
orbitopt optimization verify runs/exp-001      # re-grade a saved run; exit 0 only when verified
```

`validate`用它的JSON指针打印每个错误,并退出1,或者打印摘要,结构类和可以接受问题的安装的解析器. `verify`取代理运行目录(`orbitopt agent solve -o`)或赤封运行(`seal_run`). 对于一个代理运行它检查每个文件的字节和追踪,然后与最终的验证的替代解析器再次分级封存的最佳结果. 它印出判决 和四个状态 他们的理由。 都带上`--json`
