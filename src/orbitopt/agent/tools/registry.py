"""The tool registry, the allowlisted views agents call through, replay, and the IR binding.

Calls happen only through a ``ToolView``: an allowlist over a registry plus
the ``CallLog`` every call is written to. There is no unlogged call path. A
call validates its arguments against the input schema and the declared
domain, charges the caller's budget, runs the tool, validates the output
against the output schema (NaN, infinities and non-JSON values are errors)
and records the outcome. ``replay`` re-executes a log on any registry, and
``bind_problem`` turns a registry view into the ``tools=`` mapping that
``OptimizationProblem.evaluate`` takes, checking units first.
"""

from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from jsonschema import Draft202012Validator

from orbitopt.optimization import Expression, OptimizationProblem, ToolCall, canonical_digest, parse_unit
from orbitopt.optimization.units import format_dims

from .calllog import CallLog, ReplayEntry, ReplayReport, to_json_value
from .spec import DomainError, Tool, ToolError, ToolErrorCode, ToolSpec, numeric_fields, spec_errors

Code = ToolErrorCode


@dataclass(frozen=True)
class CallBudget:
    """Per-caller limits, each optional.

    ``max_calls`` counts every call except those refused for the allowlist or
    the budget, and is checked before the arguments are. ``max_evaluations``
    caps the sum of the tools' declared ``CostModel.evaluations``; a call
    that would go past it is refused before it runs. ``max_wall_s_per_call``
    discards the result of a call that ran longer and logs it as a
    ``timeout`` (the tool is not pre-empted; its declared domain bounds its
    work).
    """

    max_calls: int | None = None
    max_evaluations: float | None = None
    max_wall_s_per_call: float | None = None


@dataclass(frozen=True)
class Caller:
    """Who is calling, for the log, and the budget charged against that view's log."""

    id: str
    budget: CallBudget = CallBudget()


@dataclass(frozen=True)
class _Entry:
    tool: Tool
    inputs: Draft202012Validator
    outputs: Draft202012Validator


def _signature_errors(tool: Tool) -> list[str]:
    try:
        params = inspect.signature(tool.fn).parameters
    except (TypeError, ValueError) as exc:
        return [f"callable has no inspectable signature: {exc}"]
    errors = [f"parameter {name!r} must be passable by keyword and not variadic"
              for name, p in params.items() if p.kind not in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    properties = set(tool.spec.input_schema.get("properties", {}))
    if set(params) != properties:
        errors.append(f"callable parameters {sorted(params)} differ from input_schema properties {sorted(properties)}")
    required = set(tool.spec.input_schema.get("required", ()))
    no_default = {name for name, p in params.items() if p.default is p.empty}
    if no_default != required:
        errors.append(f"parameters without defaults {sorted(no_default)} differ from required inputs {sorted(required)}")
    return errors


class ToolRegistry:
    """Registered tools by id. Hand agents a ``view``, never the registry."""

    def __init__(self, tools: Iterable[Tool] = ()):
        self._entries: dict[str, _Entry] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Add a tool; raise ``ToolError`` for a duplicate id, an unsound spec or a signature/schema mismatch."""
        tool_id = getattr(tool.spec, "tool_id", "?")
        if tool_id in self._entries:
            raise ToolError(Code.DUPLICATE_TOOL, tool_id, "a tool with this id is already registered")
        errors = spec_errors(tool.spec)
        if errors:
            raise ToolError(Code.INVALID_SPEC, tool_id, "; ".join(errors))
        errors = _signature_errors(tool)
        if errors:
            raise ToolError(Code.SIGNATURE_MISMATCH, tool_id, "; ".join(errors))
        self._entries[tool_id] = _Entry(tool, Draft202012Validator(tool.spec.input_schema),
                                        Draft202012Validator(tool.spec.output_schema))

    def __contains__(self, tool_id: object) -> bool:
        return tool_id in self._entries

    def get(self, tool_id: str) -> Tool:
        if tool_id not in self._entries:
            raise ToolError(Code.UNKNOWN_TOOL, tool_id, "no tool is registered with this id")
        return self._entries[tool_id].tool

    def list(self) -> tuple[ToolSpec, ...]:
        """Every registered spec, sorted by id."""
        return tuple(self._entries[k].tool.spec for k in sorted(self._entries))

    def view(self, allowed: Iterable[str], log: CallLog | None = None) -> "ToolView":
        """A view limited to ``allowed`` ids, writing to ``log`` (a new one by default).

        Every allowed id must be registered, so a typo fails here rather
        than at the agent's first call.
        """
        allowed = frozenset(allowed)
        unknown = sorted(allowed - self._entries.keys())
        if unknown:
            raise ToolError(Code.UNKNOWN_TOOL, ", ".join(unknown), "cannot allow unregistered tools")
        return ToolView(self, allowed, CallLog() if log is None else log)


def _stored(args: Any) -> Any:
    try:
        return to_json_value(args)
    except (TypeError, ValueError) as exc:
        return f"<unserialisable arguments: {exc}>"


def _coerce_integers(args: dict, schema: Mapping[str, Any]) -> dict:
    """Integral floats become ints in ``integer`` fields: IR expressions always evaluate to floats."""
    out = dict(args)
    for name, prop in schema.get("properties", {}).items():
        if name not in out:
            continue
        if prop.get("type") == "integer":
            out[name] = int(out[name])
        elif prop.get("type") == "array" and prop.get("items", {}).get("type") == "integer":
            out[name] = [int(v) for v in out[name]]
    return out


def _parse_args(entry: _Entry, args: Any) -> tuple[Any, ToolError | None]:
    """Canonical arguments, or what to store plus the error. Checks JSON form, schema, then domain."""
    spec = entry.tool.spec
    try:
        args = to_json_value(args)
    except (TypeError, ValueError) as exc:
        return _stored(args), ToolError(Code.INVALID_ARGS, spec.tool_id, str(exc))
    if not isinstance(args, dict):
        return args, ToolError(Code.INVALID_ARGS, spec.tool_id, "arguments must be a JSON object")
    problems = sorted(entry.inputs.iter_errors(args), key=lambda e: list(map(str, e.absolute_path)))
    if problems:
        return args, ToolError(Code.INVALID_ARGS, spec.tool_id, "; ".join(
            f"/{'/'.join(map(str, e.absolute_path))}: {e.message}" for e in problems))
    args = _coerce_integers(args, spec.input_schema)
    for name, interval in spec.domain.items():
        if name not in args:
            continue
        values = args[name] if isinstance(args[name], list) else [args[name]]
        outside = [v for v in values if not interval.contains(v)]
        if outside:
            return args, ToolError(Code.OUT_OF_DOMAIN, spec.tool_id,
                                   f"{name}={outside[0]!r} is outside the validity domain "
                                   f"[{interval.lower}, {interval.upper}]")
    return args, None


def _run(entry: _Entry, args: dict) -> tuple[dict | None, ToolError | None]:
    """Call the tool on canonical arguments and validate what it returns."""
    tool_id = entry.tool.spec.tool_id
    try:
        raw = entry.tool.fn(**args)
    except DomainError as exc:
        return None, ToolError(Code.OUT_OF_DOMAIN, tool_id, str(exc))
    except Exception as exc:  # a tool is third-party code: its failure is a logged result, not a crash
        return None, ToolError(Code.TOOL_FAILED, tool_id, f"{type(exc).__name__}: {exc}")
    try:
        result = to_json_value(raw)
    except (TypeError, ValueError) as exc:
        return None, ToolError(Code.INVALID_OUTPUT, tool_id, str(exc))
    if not isinstance(result, dict):
        return None, ToolError(Code.INVALID_OUTPUT, tool_id, "a tool must return a JSON object")
    problems = list(entry.outputs.iter_errors(result))
    if problems:
        return None, ToolError(Code.INVALID_OUTPUT, tool_id, "; ".join(e.message for e in problems))
    return result, None


def _evaluations(spec: ToolSpec, args: dict) -> float:
    return float(Expression(spec.cost.evaluations).evaluate(
        {k: float(v) for k, v in args.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}))


class ToolView:
    """An allowlisted window on a registry; every call through it is logged in ``log``."""

    def __init__(self, registry: ToolRegistry, allowed: frozenset[str], log: CallLog):
        self._registry = registry
        self._allowed = allowed
        self.log = log

    @property
    def allowed(self) -> frozenset[str]:
        return self._allowed

    def get(self, tool_id: str) -> ToolSpec:
        """The spec of an allowed tool; ``ToolError`` otherwise. Not a call, so not logged."""
        refusal = self._refusal(tool_id)
        if refusal is not None:
            raise refusal
        return self._registry.get(tool_id).spec

    def _refusal(self, tool_id: str) -> ToolError | None:
        if tool_id in self._allowed:
            return None
        code = Code.NOT_ALLOWED if tool_id in self._registry else Code.UNKNOWN_TOOL
        return ToolError(code, tool_id, "this tool is not available to this view")

    def list(self) -> tuple[ToolSpec, ...]:
        """Specs of the allowed tools, sorted by id."""
        return tuple(spec for spec in self._registry.list() if spec.tool_id in self._allowed)

    def call(self, tool_id: str, args: Mapping[str, Any], *, caller: Caller) -> dict:
        """Run an allowed tool on ``args`` and return its validated result; log the call whatever happens.

        Raises ``ToolError``: ``unknown_tool``/``not_allowed`` (refused),
        ``invalid_args``, ``out_of_domain`` (before the tool runs, or raised
        by it as ``DomainError``), ``budget_exceeded`` (refused before it
        runs), ``tool_failed``, ``invalid_output`` or ``timeout``.
        """
        error = self._refusal(tool_id) or self._over_budget(tool_id, caller, 0.0)
        if error is not None:
            return self._finish(tool_id, caller, _stored(args), None, error, 0.0, 0.0)
        entry = self._registry._entries[tool_id]
        parsed, error = _parse_args(entry, args)
        if error is not None:
            return self._finish(tool_id, caller, parsed, None, error, 0.0, 0.0)
        evaluations = _evaluations(entry.tool.spec, parsed)
        error = self._over_budget(tool_id, caller, evaluations)
        if error is not None:
            return self._finish(tool_id, caller, parsed, None, error, 0.0, 0.0)
        start = time.perf_counter()
        result, error = _run(entry, parsed)
        wall = time.perf_counter() - start
        cap = caller.budget.max_wall_s_per_call
        if error is None and cap is not None and wall > cap:
            result, error = None, ToolError(Code.TIMEOUT, tool_id, f"ran {wall:.3g} s, over the {cap:g} s cap")
        return self._finish(tool_id, caller, parsed, result, error, evaluations, wall)

    def _over_budget(self, tool_id: str, caller: Caller, evaluations: float) -> ToolError | None:
        calls, used = self.log.usage(caller.id)
        budget = caller.budget
        if budget.max_calls is not None and calls + 1 > budget.max_calls:
            return ToolError(Code.BUDGET_EXCEEDED, tool_id, f"caller {caller.id!r} has used all {budget.max_calls} calls")
        if budget.max_evaluations is not None and used + evaluations > budget.max_evaluations:
            return ToolError(Code.BUDGET_EXCEEDED, tool_id,
                             f"{evaluations:g} evaluations would take caller {caller.id!r} past "
                             f"{budget.max_evaluations:g} (used {used:g})")
        return None

    def _finish(self, tool_id, caller, args, result, error, evaluations, wall) -> dict:
        self.log.append(tool_id=tool_id, caller=caller.id, args=args, result=result,
                        error=None if error is None else {"code": error.code.value, "message": str(error)},
                        evaluations=evaluations, wall_s=wall)
        if error is not None:
            raise error
        return result


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _difference(recorded: Any, replayed: Any, tolerance: float, path: str = "") -> str | None:
    """Where two results differ by more than ``tolerance`` (relative, floor 1), or ``None``."""
    numbers = (int, float)
    if isinstance(recorded, numbers) and isinstance(replayed, numbers) \
            and not isinstance(recorded, bool) and not isinstance(replayed, bool):
        if abs(recorded - replayed) <= tolerance * max(1.0, abs(recorded)):
            return None
        return f"{path or '/'}: recorded {recorded!r}, replayed {replayed!r}"
    if isinstance(recorded, dict) and isinstance(replayed, dict):
        if recorded.keys() != replayed.keys():
            return f"{path or '/'}: keys {sorted(recorded)} against {sorted(replayed)}"
        for key in sorted(recorded):
            found = _difference(recorded[key], replayed[key], tolerance, f"{path}/{key}")
            if found:
                return found
        return None
    if isinstance(recorded, list) and isinstance(replayed, list):
        if len(recorded) != len(replayed):
            return f"{path or '/'}: {len(recorded)} items against {len(replayed)}"
        for i, (a, b) in enumerate(zip(recorded, replayed)):
            found = _difference(a, b, tolerance, f"{path}/{i}")
            if found:
                return found
        return None
    return None if recorded == replayed else f"{path or '/'}: recorded {recorded!r}, replayed {replayed!r}"


def replay(log: CallLog, registry: ToolRegistry) -> ReplayReport:
    """Check the log's integrity and re-execute every executed record on ``registry``.

    Each record is replayed on its own recorded arguments, with no
    allowlist, budget or logging: replay asks whether the tool still gives
    the recorded outcome, not whether the session was allowed to ask.
    A record replays ``exact`` when the result is bit for bit equal (or the
    same error code recurs), ``quantised`` when equal at 12 significant
    digits, ``within_tolerance`` when within the tool's declared
    ``replay_tolerance``, ``not_executed`` for a refused call, and
    ``mismatch`` otherwise.
    """
    entries = []
    for record in log.records:
        if record.outcome == "refused":
            entries.append(ReplayEntry(record.seq, record.tool_id, "not_executed"))
            continue
        if record.tool_id not in registry:
            entries.append(ReplayEntry(record.seq, record.tool_id, "mismatch", "tool is not registered"))
            continue
        entry = registry._entries[record.tool_id]
        parsed, error = _parse_args(entry, record.args)
        result = None
        if error is None:
            if _json_text(parsed) != _json_text(record.args):
                entries.append(ReplayEntry(record.seq, record.tool_id, "mismatch",
                                           "recorded arguments are not in canonical form"))
                continue
            result, error = _run(entry, parsed)
        entries.append(_judge(record, result, error, entry.tool.spec.replay_tolerance))
    return ReplayReport(tuple(log.verify()), tuple(entries))


def _judge(record, result, error, tolerance) -> ReplayEntry:
    def entry(status, detail=""):
        return ReplayEntry(record.seq, record.tool_id, status, detail)

    if record.error is not None or error is not None:
        replayed = "ok" if error is None else error.code.value
        recorded = "ok" if record.error is None else record.code.value
        return entry("exact") if replayed == recorded else entry("mismatch", f"recorded {recorded}, replay gave {replayed}")
    if _json_text(result) == _json_text(record.result):
        return entry("exact")
    if canonical_digest(result) == canonical_digest(record.result):
        return entry("quantised")
    found = _difference(record.result, result, tolerance)
    if tolerance > 0 and found is None:
        return entry("within_tolerance")
    return entry("mismatch", found or "results differ")


def bind_problem(problem: OptimizationProblem, view: ToolView, caller: Caller) -> dict[str, Callable[..., float]]:
    """The ``tools=`` mapping for ``problem.evaluate``, after checking every ``ToolCall`` against its spec.

    For each tool objective and constraint the tool must be in ``view``; the
    call must bind exactly the tool's inputs, all scalar and numeric (with
    their defaults allowed to be left out); each argument expression's
    dimension must equal the input's declared unit; and the dimension of
    the tool's ``value_field`` output must equal the ``unit`` the problem
    declares. Any failure raises ``ToolError`` (``unit_mismatch`` for a
    dimension) before anything runs. Each resolved call goes through
    ``view.call`` and so is logged and budgeted. Tools take scalars, so
    evaluate one point at a time, not a numpy array of points. Assumes
    ``problem`` passed ``validate_problem``.
    """
    dims = {item.name: parse_unit(item.unit).dims for item in (*problem.variables, *problem.parameters)}
    bound: dict[str, Callable[..., float]] = {}
    for group, items in (("objectives", problem.objectives), ("constraints", problem.constraints)):
        for i, item in enumerate(items):
            call = item.expression
            if not isinstance(call, ToolCall):
                continue
            where = f"/{group}/{i}"
            spec = view.get(call.tool_id)
            _check_binding(spec, call, item.unit, dims, where)
            bound[call.tool_id] = _resolver(view, spec, caller)
    return bound


def _check_binding(spec: ToolSpec, call: ToolCall, unit: str, dims: Mapping, where: str) -> None:
    schema = spec.input_schema
    missing = set(schema.get("required", ())) - call.args.keys()
    unknown = call.args.keys() - set(schema.get("properties", {}))
    if missing or unknown:
        raise ToolError(Code.INVALID_ARGS, spec.tool_id,
                        f"{where}: missing inputs {sorted(missing)}, unknown inputs {sorted(unknown)}")
    scalars = {name for name, is_array in numeric_fields(schema).items() if not is_array}
    for arg, source in sorted(call.args.items()):
        if arg not in scalars:
            raise ToolError(Code.INVALID_ARGS, spec.tool_id,
                            f"{where}: input {arg!r} is not a scalar number, so an IR expression cannot supply it")
        found, expected = Expression(source).dimension(dims), parse_unit(spec.input_units[arg]).dims
        if found != expected:
            raise ToolError(Code.UNIT_MISMATCH, spec.tool_id,
                            f"{where}: argument {arg}={source!r} has dimension {format_dims(found)}, "
                            f"the tool expects {format_dims(expected)}")
    produced = parse_unit(spec.output_units[spec.value_field]).dims
    declared = parse_unit(unit).dims
    if produced != declared:
        raise ToolError(Code.UNIT_MISMATCH, spec.tool_id,
                        f"{where}: the tool's {spec.value_field!r} has dimension {format_dims(produced)} but the "
                        f"problem declares unit {unit!r} ({format_dims(declared)})")


def _resolver(view: ToolView, spec: ToolSpec, caller: Caller) -> Callable[..., float]:
    def resolve(**kwargs: Any) -> float:
        return view.call(spec.tool_id, kwargs, caller=caller)[spec.value_field]

    return resolve
