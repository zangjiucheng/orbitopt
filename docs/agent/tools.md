# Scientific tool registry

`orbitopt.agent.tools` holds the scientific functions that an agent, or an
IR `ToolCall` objective or constraint, may call. Each tool declares its
schemas, units, assumptions, validity domain, cost and the tests that verify
it. A caller reaches a tool only through an allowlisted view. Every call
lands in a hash-chained log, and `replay` re-executes that log on a fresh
registry. Importing the package loads `jsonschema` and the optimization IR.
It loads numpy and scipy only when a tool that needs them runs.

## Calling a tool

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

Give an agent a `ToolView`, never the `ToolRegistry`. A view has no call
path that skips the log. `view.list()` returns the specs the agent may use,
and `ToolSpec.to_dict()` gives their JSON form for a prompt.

`ToolView.call(tool_id, args, *, caller)` runs these steps in order. The
call is logged whatever the outcome, and a failure raises `ToolError`.

1. Refuse an id outside the allowlist, and refuse a caller that has used all of `max_calls`.
2. Check that the arguments are JSON with no NaN or infinity, then validate them against `input_schema`.
3. Convert integral floats in `integer` fields to `int`, because IR expressions always evaluate to floats.
4. Refuse a value outside the declared `domain` before the tool runs.
5. Refuse the call when its declared evaluations would take the caller past `max_evaluations`.
6. Run the tool.
7. Validate the result against `output_schema`. NaN, an infinity or a non-JSON value is an error and is never returned.
8. Discard the result if the call ran past `max_wall_s_per_call`.

## Error codes

Callers branch on `ToolError.code`, a `ToolErrorCode`.

| Code | Raised by | Outcome in the log |
|---|---|---|
| `invalid_spec` | `register`: the spec breaks the contract that `spec_errors` checks | not a call |
| `duplicate_tool` | `register`: the id is already registered | not a call |
| `signature_mismatch` | `register`: the callable's parameters differ from the input schema's properties, or its defaults differ from the schema's optional fields | not a call |
| `unknown_tool` | `call`: no tool has this id | `refused` |
| `not_allowed` | `call`: the tool exists but the view does not allow it | `refused` |
| `invalid_args` | `call`: the arguments fail the input schema or are not finite JSON | `error` |
| `out_of_domain` | `call`: an input is outside `domain`, or the tool raised `DomainError` | `error` |
| `tool_failed` | `call`: the tool raised any other exception | `error` |
| `invalid_output` | `call`: the result fails the output schema or is not finite JSON | `error` |
| `budget_exceeded` | `call`: the call would pass `max_calls` or `max_evaluations` | `refused` |
| `timeout` | `call`: the call ran longer than `max_wall_s_per_call` | `refused` |
| `unit_mismatch` | `bind_problem`: a dimension disagrees | not a call |

An `error` outcome depends only on the tool and its arguments, so replay
re-executes it and expects the same code. A `refused` outcome depends on the
session's allowlist and budget, so replay skips it.

## What a spec declares

| Field | Meaning |
|---|---|
| `tool_id` | Dotted lower-case names that end in a version, such as `math.erf.v1`. A changed tool gets a new id. Every valid id is also a valid IR `ToolCall.tool_id`. |
| `input_schema`, `output_schema` | JSON Schema objects. The input schema must set `additionalProperties: false`. |
| `input_units`, `output_units` | One SI-coherent unit string per numeric field, `""` for dimensionless. The unit of an array applies to each element. A unit with a scale other than 1, such as `km`, is rejected. |
| `assumptions` | The model assumptions, as text. |
| `domain` | One closed `Interval` per numeric input. The tests verify the tool on these ranges. A tool raises `DomainError` for a precondition that an interval cannot express, such as an unstable explicit time step. |
| `cost` | `CostModel(cost_class, evaluations)`. `cost_class` is `trivial`, `cheap`, `moderate` or `expensive`. `evaluations` is an IR expression over the required scalar inputs, such as `"3*n"`, and is the amount charged to `CallBudget.max_evaluations`. |
| `deterministic` | `False` marks a stochastic tool. A stochastic tool must take a required integer `seed`. |
| `replay_tolerance` | The relative tolerance that replay allows, with a floor of 1 on the scale. `0` means the canonical results must be equal. |
| `tested_by` | The test ids that verify the tool against an independent reference. |

## The call log

Each `CallRecord` stores `seq`, `tool_id`, `caller`, the canonical `args`,
the `result` or the `error` (`{"code", "message"}`), the `evaluations`
charged, `wall_s`, `prev_digest` and `digest`. The digest is
`canonical_digest` of every other field, which includes the previous
record's digest. `CallLog.verify()` therefore reports any record that was
edited, dropped or moved. `canonical_digest` rounds numbers to 12
significant digits, the rule the IR digest uses. Last-bit noise from another
machine does not change a digest. A change in the 12th digit does.

The log stores arguments and results exactly. JSON float text round-trips a
double bit for bit, so a replay runs on the recorded inputs.

`replay(log, registry)` returns a `ReplayReport`. Its `integrity` field lists
the problems found by `verify()`. It also holds one `ReplayEntry` per record,
and each entry has one of these statuses.

| Status | Meaning |
|---|---|
| `exact` | The result is bit for bit the recorded result, or the same error code recurs. |
| `quantised` | The results agree at 12 significant digits, which is what another machine's libm or LAPACK gives. |
| `within_tolerance` | The results agree within the tool's `replay_tolerance`. |
| `not_executed` | The call was refused, so replay skips it. |
| `mismatch` | Anything else. `detail` names the first field that differs. |

`report.ok` is true when `integrity` is empty and no entry is a `mismatch`.
A forger who rewrites a result and then recomputes every digest passes
`verify()`. The forger still fails replay, because replay re-executes the
call.

## Tools in an optimization problem

`bind_problem(problem, view, caller)` returns the `tools=` mapping that
`OptimizationProblem.evaluate` takes. Before anything runs, it checks each
`ToolCall` objective and constraint.

- The tool must be in the view.
- The call's `args` and `constants` together must bind every required input and no unknown input.
- An expression in `args` must feed a scalar numeric input, and its dimension must equal the input's declared unit.
- A constant reaches the tool unchanged, in the tool's own SI units, and the input schema checks it when the call runs. A string integrand, a rule name or an array goes here.
- `output` must name a required scalar numeric output, and its dimension must equal the `unit` that the problem declares for that objective or constraint.

A failed check raises `ToolError` with `unit_mismatch` or `invalid_args`.
Each resolver returns the tool's whole output object, and the IR reads the
field that `output` names. A `ToolError` that a call raises inside
`evaluate`, such as an `out_of_domain` refusal, reaches the caller as the
IR's `ToolCallError`, a subclass of `EvaluationError`, with the `ToolError`
as its cause. A solver therefore treats the point as infeasible and goes on.
This check covers the open point in
[ADR optimization-0001](../adr/optimization-0001-optimization-ir.md). The IR
trusts the unit declared on a `ToolCall`, so a problem that passes
`validate_problem` can still declare `W` for a tool that returns kelvin. The
evaluator converts arguments to SI before the call, so a variable in minutes
reaches the tool in seconds. Each call goes through `view.call`, so the
solver's evaluations are logged and budgeted.

```python
problem = OptimizationProblem(
	id="bessel-j0-minimum", variables=(Variable("x", lower=2.0, upper=5.0),),
	objectives=(Objective("j0", ToolCall("math.bessel_j0.v1", {"x": "x"}, output="value")),))
tools = bind_problem(problem, view, agent)
minimize_scalar(lambda x: problem.evaluate({"x": x}, tools=tools).objectives["j0"],
	bounds=(2.0, 5.0), method="bounded")                    # x = 3.8317..., the first zero of J1
```

## Built-in tools

`builtin_registry()` registers the tools in this table. The command
`pytest tests/test_e14_04_tool_registry.py` checks each tool against the
reference that the table names. That reference shares no code with the tool.

| Tool | Computes | Reference in the tests |
|---|---|---|
| `math.erf.v1`, `math.erfc.v1`, `math.gamma.v1`, `math.lgamma.v1`, `math.log1p.v1`, `math.expm1.v1` | The function of one real argument, through Python's `math` module | 20-digit mpmath values, relative error at most 1e-13 |
| `math.bessel_j0.v1`, `math.bessel_j1.v1` | J0(x) and J1(x), through `scipy.special` | 20-digit mpmath values, relative error at most 1e-13 |
| `numerics.gauss_legendre.v1` | An n-point Gauss-Legendre rule, with the error estimate \|Q_n - Q_2n\| | Polynomial exactness for degree 2n - 1, and the estimate within 0.1% of the true error for e^t |
| `numerics.adaptive_simpson.v1` | Adaptive Simpson to an absolute tolerance, within a cap on evaluations | Closed-form integrals at tolerances 1e-6 and 1e-10 |
| `numerics.quadrature_cost.v1` | One rule of the E14 integration family, with exactly n evaluations, and its absolute error against a QUADPACK reference | All seven E14 integration instances: the scan returns the verified minimum N and rule, and each reference matches its closed form to 1e-13 |
| `heat.layered_wall.v1` | The steady heat flux through layers in series with convection on both faces, and the face and interface temperatures | A hand calculation, and the same flux through every film and layer |
| `heat.transient_slab.v1` | Theta-scheme finite differences for a slab whose faces are held at a fixed temperature | The Fourier series solution, with measured convergence orders of 2 in space and 1 (implicit) or 2 (Crank-Nicolson) in time |

Each spec lists its model assumptions and validity domain, so
`view.get(tool_id).to_dict()` describes the tool. The built-in tools are
standard textbook models.

## Adding a tool

1. Write a pure function that takes keyword arguments and returns a JSON object.
2. Write a `ToolSpec` with a new versioned id. Set `tested_by` to the tests that compare the function with an independent reference.
3. Add the `Tool` to the `TOOLS` tuple of a module in `orbitopt.agent.tools.builtin`. For a new module, add its `TOOLS` to `BUILTIN_TOOLS` in `builtin/__init__.py`.

`test_every_builtin_is_sound_and_names_tests_that_exist` fails when a spec
breaks the contract, or when `tested_by` names a test that does not exist.
