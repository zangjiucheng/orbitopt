# LLM-assisted optimization

`orbitopt.agent` turns a problem statement into an
[optimization IR](../optimization/ir.md) problem, solves it in a bounded loop
whose every candidate the [verification guard](../optimization/verification.md)
grades, and saves a run that replays without any LLM call. A language model
may write the problem and choose solvers. It never grades an answer: only the
guard does, from evidence.

| Page | |
|---|---|
| [Providers](providers.md) | One interface to OpenAI-compatible, local, mock and replay models; cassettes. |
| [Compiler](compiler.md) | Statement in, a validated IR problem or a stated refusal out. |
| [Tools](tools.md) | The scientific tool registry an IR `ToolCall` reaches, logged and replayable. |
| [Runtime](runtime.md) | The bounded loop: limits, policy, planners, diagnostics, run directories and replay. |
| [Optimize workspace](studio.md) | Mission Control's page: words in, the IR as editable math, a watched run, four statuses, replay. |
| [Benchmark specification](../benchmarks/e14_benchmark_spec.md) | The pre-registered E14-10 design that will measure whether the LLM helps. |

## Install

```bash
pip install -e '.[agent]'
```

The provider layer uses only the standard library and `jsonschema`, so no
LLM SDK is installed. `[agent]` is `[solvers]` (numpy, scipy, `cma`), which
the runtime and the built-in tools need. The base install still imports and
runs everything else, and `orbitopt --help` loads none of this.

## The five commands

The session below uses the mock provider with a recorded compiler reply, as
the tests do. Replace `--provider mock --mock-replies replies.json` with a
real provider (next section) to use a model.

```text
$ cat prompt.txt
A plant makes two products with profit 3 and 5 per batch. Plant 1 has 4 hours
(product 1 uses 1 hour), plant 2 has 12 hours (product 2 uses 2 hours), plant 3
has 18 hours (product 1 uses 3, product 2 uses 2). Batches are continuous and
nonnegative. Maximize profit.

$ orbitopt agent compile --prompt-file prompt.txt -o problem.json \
      --provider mock --mock-replies replies.json --cassette compile.jsonl
COMPILE after 1 round(s), 624 tokens
  problem      problem.json  (338cb4c5222376bcc428220cb56e7725d7b0249f7f3658e8e5f02440d3b4a92a)
  report       problem.compile.json

$ orbitopt optimization validate problem.json
VALID nl-74a320afd4c5 (IR 1.1)
  digest       338cb4c5222376bcc428220cb56e7725d7b0249f7f3658e8e5f02440d3b4a92a
  structure    lp (linear): 2 variable(s), 1 objective(s), 3 constraint(s)
  solvers      scipy.highs, scipy.slsqp

$ orbitopt agent solve problem.json --max-iterations 20 --max-evaluations 4000 \
      -o runs/experiment-001
VERIFIED (planner_stop) after 2 iteration(s), 14 evaluations, 0 tokens
  best         scipy.highs: {'profit': 36.0} at {'product_1': 2.0, 'product_2': 6.0}
  solver_converged         passed
  constraints_satisfied    passed
  independently_verified   passed
  optimality_certified     not_assessed
  run          runs/experiment-001
  manifest     sha256:7534331142fbac893c49fa69452193b59bc9c4b6ffe8044157876efa04b01dd0  (keep it; ...)

$ orbitopt optimization verify runs/experiment-001
VERIFIED runs/experiment-001 (agent run, recorded status verified)
  verdict      passed  (required: solver_converged, constraints_satisfied, independently_verified)
  solver_converged         passed        the solver reports termination 'converged'
  constraints_satisfied    passed        every domain and hard constraint holds on re-evaluation
  independently_verified   passed        independent evidence agrees with the re-evaluated result
  optimality_certified     not_assessed  not requested by the policy, claimed or evidenced

$ orbitopt agent replay runs/experiment-001
EQUAL runs/experiment-001 (sealed verdict: passed)
  trace        recorded 7bdc6c28..., replayed 7bdc6c28...
```

| Command | Does | Exit 0 when |
|---|---|---|
| `agent compile --prompt-file P -o problem.json` | Runs the [compiler](compiler.md). Writes the problem only when the compile is accepted, and always writes the report (`problem.compile.json`, or `--report`) with the assumptions, missing parameters, ambiguities, other questions, refusal reason, unauthorised changes or errors, and the tokens spent. `--allow-draft` writes the model's raw draft to `-o` on a refusal, marked `"draft": true`; it is not IR and `validate` rejects it. | the compile is accepted |
| `optimization validate problem.json` | Prints every error, or the digest, structure class and installed compatible solvers. | the problem is valid |
| `agent solve problem.json --max-iterations N --max-evaluations N -o DIR` | Runs the [runtime](runtime.md) loop and saves the run directory. `--max-seconds`, `--max-tokens` and `--seed` are optional; `--planner rule` (default) asks no model, `--planner llm` asks `--provider`. Refuses a directory that already holds a run. | the answer is verified |
| `optimization verify DIR` | Checks every byte and the trace, then re-grades the sealed best answer with the guard and the final verification's alternate solvers. Also takes a bare sealed run. | it is verified again |
| `agent replay DIR` | Re-executes the recorded proposals with no planner or model and compares the traces. | nothing diverged |

Exit 1 means the command ran and the answer is no; exit 2 means the request
itself is wrong (a missing file, a used run directory, a provider that cannot
be reached or has no key). Every command takes `--json`. `verify` and
`replay` take `--expect-digest` with the manifest digest `solve` printed:
keep that digest out of band, since whoever can rewrite a run directory can
rewrite its manifest too.

## Providers on the command line

`compile` and `solve --planner llm` take the same provider options.

| `--provider` | Needs | Notes |
|---|---|---|
| `openai-compatible` | `--base-url`, `--model` | Reads the API key from the environment variable `--api-key-env` names (default `OPENAI_API_KEY`), at call time. |
| `local` | `--model` | Ollama by default; `--base-url` for llama.cpp or vLLM. No key. |
| `mock` | `--mock-replies FILE` | A JSON list of replies, each a string or a JSON object sent as its JSON text. |
| `replay` | `--cassette FILE` | Serves a recorded cassette; any request it did not record is an error. |

No option takes a key, so a key never lands in shell history, a cassette, a
report or the output. `--cassette FILE` with any other provider records every
model call there, and `--provider replay --cassette FILE` serves the same
calls back later:

```bash
export OPENAI_API_KEY=...        # set in the environment, never on the command line
orbitopt agent compile --prompt-file prompt.txt -o problem.json \
    --provider openai-compatible --base-url https://api.openai.com/v1 --model gpt-4o-mini \
    --cassette compile.jsonl
orbitopt agent compile --prompt-file prompt.txt -o problem.json \
    --provider replay --cassette compile.jsonl          # same outcome, no model call
```

`agent replay` needs no cassette at all: the trace holds every proposal, so
an LLM-planned run re-executes without the model.

## From Python

The package root re-exports each part's entry points lazily, so
`import orbitopt.agent` loads nothing:

```python
from orbitopt.agent import LocalProvider, RuleBasedPlanner, compile_problem, replay, save_run, solve

outcome = compile_problem(statement, LocalProvider("qwen2.5:7b"))
run = solve(outcome.problem, planner=RuleBasedPlanner(), seed=0)
digest = save_run(run, "runs/experiment-001")
replay("runs/experiment-001", expected_manifest_digest=digest).equal
```

Names that two parts define (`ToolSpec`, `ToolCall`, `Violation`, the tool
log's `replay`) are not re-exported; import them from their part.
`orbitopt.agent.commands` holds what each command computes, as functions
with paths in and summaries out.
