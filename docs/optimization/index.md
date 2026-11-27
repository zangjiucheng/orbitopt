# Optimization

`orbitopt.optimization` describes an optimization problem once, as typed
data with a stable digest, and solves and grades it without an LLM. The
[LLM-assisted workflow](../agent/index.md) builds on it: the compiler writes
this IR and the agent runtime solves it with these solvers and this guard.

| Page | |
|---|---|
| [Optimization IR](ir.md) | The problem and result documents, expressions, units, tolerances and digests. |
| [Solvers](solvers.md) | The registry, what `classify` decides, each solver's capability and options, and the contract every adapter keeps. |
| [Verification guard](verification.md) | The four statuses, the evidence that sets them, sealed runs and the fault matrix. |
| [ADR optimization-0001](../adr/optimization-0001-optimization-ir.md) | Why the IR has this shape, and what IR 1.1 changed. |

## Install

The IR, the classifier and the guard need only the base install. The
solvers need numpy, scipy and, for CMA-ES, `cma`:

```bash
pip install -e '.[solvers]'
```

NSGA-II needs pygmo, which comes from conda. The locked conda environment
(`envs/cpu.conda-lock.yml`) has numpy, scipy and pygmo but not `cma`.

## From Python

```python
import json
from orbitopt.optimization import default_registry, parse_problem, verify_result

problem = parse_problem(json.load(open("problem.json")))
solver = default_registry().compatible(problem)[0]
result = solver.solve(problem, seed=1)
report = verify_result(problem, result)   # no evidence yet: independently_verified is not_assessed
report.verdict, {name: s.outcome.value for name, s in report.statuses.items()}
```

The package root re-exports the classifier, the solver registry and the
guard lazily, so `import orbitopt.optimization` loads only the IR.

## From the command line

```bash
orbitopt optimization validate problem.json    # errors, or digest, class and solvers
orbitopt optimization verify runs/exp-001      # re-grade a saved run; exit 0 only when verified
```

`validate` prints every error with its JSON pointer and exits 1, or prints
the digest, the structure class and the installed solvers that can take the
problem. `verify` takes an agent run directory (`orbitopt agent solve -o`) or
a bare sealed run (`seal_run`). For an agent run it checks every file's
bytes and the trace, then grades the sealed best result again with the
final verification's alternate solvers. It prints the verdict and the four
statuses with their reasons. Both take `--json`.
