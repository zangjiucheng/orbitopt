"""The Optimize workspace's work: providers from the environment, offline examples, and the compile, run and
replay jobs.

Each job is a plain function of its inputs plus two callables, ``report``
(progress out) and ``cancelled`` (cooperative cancel in), so the workspace
runs it on a background thread and the tests run it inline. Jobs use only
the public E14 API: ``compile_problem``, ``solve``, ``save_run``, ``replay``
and ``load_trace``.

An **experiment directory** is what a run leaves on disk::

    <root>/<problem id>-<UTC stamp>/
        studio.json             orbitopt.studio.experiment/1: the prompt rounds, the edit list, the provider
                                kind and model (never a credential), planner, seed, digests
        compile.cassette.jsonl  every compiler call, recorded (absent for a hand-written problem)
        run/                    the runtime's own run directory (save_run): problem, trace, results, sealed/

Replaying an experiment re-runs every compile round from the cassette (no
model is called), re-applies the recorded edits and checks that the result
is the problem the run solved, then replays the run itself with the runtime.

Credentials never pass through here: a real provider reads its key from the
environment variable that ``ORBITOPT_LLM_API_KEY_ENV`` names, at call time.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from orbitopt.viz.optimize.editing import Edit, apply_edits, validate_doc

EXPERIMENT_FORMAT = "orbitopt.studio.experiment/1"
EXPERIMENT_FILE = "studio.json"
COMPILE_CASSETTE = "compile.cassette.jsonl"
RUN_SUBDIR = "run"
ENV_ROOT = "ORBITOPT_OPTIMIZATION_RUNS"
ENV_PROVIDER, ENV_MODEL, ENV_BASE_URL, ENV_KEY_ENV = (
    "ORBITOPT_LLM_PROVIDER", "ORBITOPT_LLM_MODEL", "ORBITOPT_LLM_BASE_URL", "ORBITOPT_LLM_API_KEY_ENV")
REQUIRED_MODULES = ("numpy", "scipy")
INSTALL_HINT = ("The Optimize workspace runs the E14 solvers, which need the agent extras:\n\n"
                "    pip install 'orbitopt[agent]'\n\n"
                "(numpy and scipy; cma adds CMA-ES). Restart Mission Control after installing.")

Report = Callable[[Any], None]
Cancelled = Callable[[], bool]


def missing_extras() -> tuple[str, ...]:
    """The modules of ``orbitopt[agent]`` this environment lacks (empty when the workspace can run)."""
    return tuple(m for m in REQUIRED_MODULES if importlib.util.find_spec(m) is None)


def default_root() -> Path:
    """Where experiments are saved: ``$ORBITOPT_OPTIMIZATION_RUNS``, else ``./runs/optimization``."""
    override = os.environ.get(ENV_ROOT, "").strip()
    return Path(override).expanduser() if override else Path.cwd() / "runs" / "optimization"


# --------------------------------------------------------------------------------------------------- providers
@dataclass(frozen=True)
class ProviderChoice:
    """A provider the user can pick. Only its kind, model and endpoint are known here, never its key."""

    key: str                  # "mock" or "env"
    kind: str                 # mock | openai-compatible | local
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None

    @property
    def real(self) -> bool:
        return self.kind != "mock"

    @property
    def label(self) -> str:
        if not self.real:
            return "Offline mock (no network)"
        where = f" @ {self.base_url}" if self.base_url else ""
        return f"{self.kind}: {self.model}{where}"

    def describe(self) -> dict:
        """What studio.json records: no key, no key variable value."""
        return {"kind": self.kind, "model": self.model, "base_url": self.base_url}


MOCK = ProviderChoice("mock", "mock")


def provider_choices(environ: Mapping[str, str] | None = None) -> tuple[ProviderChoice, ...]:
    """The offline mock, plus the provider the environment configures (``ORBITOPT_LLM_*``), if any."""
    env = os.environ if environ is None else environ
    kind = env.get(ENV_PROVIDER, "").strip() or "openai-compatible"
    model = env.get(ENV_MODEL, "").strip()
    base_url = env.get(ENV_BASE_URL, "").strip() or None
    if not model or kind not in ("openai-compatible", "local") or (kind == "openai-compatible" and not base_url):
        return (MOCK,)
    key_env = (env.get(ENV_KEY_ENV, "").strip() or "OPENAI_API_KEY") if kind == "openai-compatible" else None
    return (MOCK, ProviderChoice("env", kind, model, base_url, key_env))


def provider_banner(choice: ProviderChoice, choices: Sequence[ProviderChoice],
                    environ: Mapping[str, str] | None = None) -> str:
    """The sentence above the workspace that says which model answers, and where its key would come from."""
    env = os.environ if environ is None else environ
    if not choice.real:
        configured = any(c.real for c in choices)
        tail = ("A configured provider is available in the list." if configured else
                f"No LLM provider is configured. Set {ENV_MODEL} and {ENV_BASE_URL} (and {ENV_PROVIDER}=local "
                f"for a local server); the API key is read from the variable {ENV_KEY_ENV} names "
                "(default OPENAI_API_KEY), never typed here or stored.")
        return "OFFLINE · The mock provider answers the bundled examples or a loaded replies file only. " + tail
    if choice.api_key_env:
        state = "set" if env.get(choice.api_key_env) else "NOT SET"
        return f"Provider {choice.label}. API key from ${choice.api_key_env} ({state}); never shown or stored."
    return f"Provider {choice.label}. No API key."


def build_chat_provider(choice: ProviderChoice, *, script: Any = None):
    """The ``ChatProvider`` for ``choice``; the mock serves ``script`` (a list of replies or a function)."""
    from orbitopt.agent.providers import make_provider

    if choice.real:
        spec: dict[str, Any] = {"kind": "openai_compat" if choice.kind == "openai-compatible" else "local",
                                "model": choice.model}
        if choice.base_url:
            spec["base_url"] = choice.base_url
        if choice.api_key_env:
            spec["api_key_env"] = choice.api_key_env
    else:
        if script is None:
            raise ValueError("the offline mock needs scripted replies: pick an example or load a replies file")
        spec = {"kind": "mock", "script": list(script) if isinstance(script, (list, tuple)) else script}
    return make_provider(spec)


def mock_planner_reply(request) -> str:
    """The offline planner: solve once with the most specialised solver the runtime offers, then stop."""
    state = json.loads(request.messages[-1].content)
    solved = any(h.get("kind") == "solve" for h in state.get("history", []))
    if solved or not state.get("solvers"):
        return json.dumps({"action": "stop", "rationale": "offline mock planner: one solve, then stop"})
    return json.dumps({"action": "solve", "solver_id": state["solvers"][0]["solver_id"],
                       "rationale": "offline mock planner: the most specialised solver offered"})


# ---------------------------------------------------------------------------------------------------- examples
def _compiler_reply(status: str = "compiled", problem: Mapping | None = None, questions: Sequence = (),
                    unsupported: Mapping | None = None) -> str:
    return json.dumps({"status": status, "problem": problem, "questions": list(questions), "assumptions": [],
                       "unsupported_reason": unsupported, "authorized_changes": [], "tradeoff_quote": None},
                      sort_keys=True)


def _var(name: str, lower: float | None, upper: float | None) -> dict:
    return {"name": name, "kind": "continuous", "lower": lower, "upper": upper, "unit": "", "source": "user"}


_HALFPLANE = {"title": "Closest point on a half-plane",
              "variables": [_var("x", 0.0, 10.0), _var("y", 0.0, 10.0)], "parameters": [],
              "objectives": [{"name": "f", "expression": "x**2 + y**2", "sense": "minimize", "unit": "",
                              "source": "user"}],
              "constraints": [{"name": "sum", "expression": "x + y", "sense": ">=", "rhs": 1.0, "unit": "",
                               "source": "user"}]}
_OPEN = {**_HALFPLANE, "variables": [_var("x", 0.0, None), _var("y", 0.0, None)]}
_COST = {"title": "Cheapest mix meeting demand",
         "variables": [_var("x", 0.0, 10.0), _var("y", 0.0, 10.0)],
         "parameters": [{"name": "c", "value": None, "unit": "", "source": "user"}],
         "objectives": [{"name": "cost", "expression": "c*x + y", "sense": "minimize", "unit": "", "source": "user"}],
         "constraints": [{"name": "demand", "expression": "x + y", "sense": ">=", "rhs": 2.0, "unit": "",
                          "source": "user"}]}
_COST_ANSWERED = {**_COST, "parameters": [{"name": "c", "value": 3.0, "unit": "", "source": "user"}]}


@dataclass(frozen=True)
class Example:
    """A prompt the offline mock can compile: its scripted compiler replies and what the answer is known to be."""

    key: str
    title: str
    prompt: str
    replies: tuple[str, ...]
    reference: str = ""
    answers: tuple[str, ...] = field(default=())


EXAMPLES = (
    Example("halfplane", "Convex QP, bounded (verifies)",
            "Minimize x squared plus y squared, where x and y are between 0 and 10 and add up to at least one.",
            (_compiler_reply(problem=_HALFPLANE),),
            "cqp-worked-d2: optimum 0.5 at x = y = 0.5"),
    Example("fc001", "FC-001 as stated, no upper bounds (unverified best)",
            "Minimize x squared plus y squared, where x and y are nonnegative and add up to at least one.",
            (_compiler_reply(problem=_OPEN),),
            "cqp-worked-d2: optimum 0.5 at x = y = 0.5; with no upper bound no alternate solver can check it"),
    Example("clarify", "Missing parameter (asks first)",
            "Minimize c*x + y where x and y are between 0 and 10 and x + y >= 2.",
            (_compiler_reply("needs_clarification", _COST, [{"kind": "missing_parameter",
                                                             "text": "What is the unit cost c of x?",
                                                             "about": "parameters/c"}]),
             _compiler_reply(problem=_COST_ANSWERED)),
            "with c = 3 the optimum is 2 at x = 0, y = 2", ("c = 3",)),
)


def example(key: str) -> Example:
    return next(e for e in EXAMPLES if e.key == key)


def with_answers(prompt: str, questions: Sequence[Mapping[str, Any]], answers: Sequence[str]) -> str:
    """The prompt for the next compile round: the original text plus the user's answers, quoted as theirs."""
    lines = [f"- {q.get('text', '')} {a.strip()}" for q, a in zip(questions, answers) if a.strip()]
    return prompt.rstrip() + ("\n\nAnswers:\n" + "\n".join(lines) if lines else "")


# ------------------------------------------------------------------------------------------------------ compile
@dataclass(frozen=True)
class CompileResult:
    """One compile round, flattened for the UI and for studio.json. ``problem`` is IR (accepted), ``draft`` is
    the compiler's unaccepted draft as an IR document (it may not validate)."""

    outcome: str
    problem: Mapping[str, Any] | None
    draft: Mapping[str, Any] | None
    questions: tuple[Mapping[str, Any], ...] = ()
    assumptions: tuple[Mapping[str, Any], ...] = ()
    unsupported: Mapping[str, Any] | None = None
    violations: tuple[Mapping[str, Any], ...] = ()
    errors: tuple[str, ...] = ()
    tokens: int = 0
    attempts: int = 0
    record: Mapping[str, Any] = field(default_factory=dict)
    wall_s: float = 0.0


def known_tool_ids() -> list[str]:
    from orbitopt.agent.tools import builtin_registry

    return sorted(t.tool_id for t in builtin_registry().list())


def _draft_doc(partial: Any, problem_id: str) -> dict | None:
    from orbitopt.agent.compiler.schema import draft_to_ir_dict

    if not isinstance(partial, Mapping):
        return None
    try:
        return draft_to_ir_dict(partial, problem_id=problem_id, assumptions=())
    except (KeyError, TypeError, AttributeError):
        return None


def _result_digest(doc: Mapping[str, Any] | None) -> str | None:
    from orbitopt.optimization import canonical_digest

    return None if doc is None else canonical_digest(dict(doc))


def compile_round(prompt: str, provider, *, base: Mapping[str, Any] | None, tool_ids: Sequence[str]) -> CompileResult:
    """Run one compile round on ``provider``. A provider failure is an outcome too (``provider_error``)."""
    from dataclasses import asdict

    from orbitopt.agent.compiler import (CompileFailed, Compiled, NeedsClarification, RejectedUnauthorizedChange,
                                         Unsupported, compile_problem)
    from orbitopt.agent.providers import ProviderError, ReplayMismatchError
    from orbitopt.optimization import parse_problem

    started = time.perf_counter()
    base_problem = None if base is None else parse_problem(base)
    record = {"prompt": prompt, "base": None if base is None else dict(base), "known_tool_ids": list(tool_ids)}
    try:
        outcome = compile_problem(prompt, provider, base_problem=base_problem, known_tool_ids=list(tool_ids))
    except (ProviderError, ReplayMismatchError) as exc:
        record.update(outcome="provider_error", result_digest=None)
        return CompileResult("provider_error", None, None, errors=(f"{type(exc).__name__}: {exc}",), record=record,
                             wall_s=time.perf_counter() - started)
    problem_id = base_problem.id if base_problem is not None else None
    problem = draft = None
    if isinstance(outcome, Compiled):
        problem = outcome.problem.to_dict()
    elif isinstance(outcome, NeedsClarification):
        draft = _draft_doc(outcome.partial, problem_id or _prompt_id(prompt))
    elif isinstance(outcome, CompileFailed):
        try:
            raw = json.loads(outcome.raw) if outcome.raw else None
        except ValueError:
            raw = None
        draft = _draft_doc((raw or {}).get("problem") if isinstance(raw, Mapping) else None,
                           problem_id or _prompt_id(prompt))
    record.update(outcome=outcome.outcome, result_digest=_result_digest(problem if problem is not None else draft))
    return CompileResult(
        outcome.outcome, problem, draft,
        questions=tuple(asdict(q) for q in getattr(outcome, "questions", ())),
        assumptions=tuple(asdict(a) for a in getattr(outcome, "assumptions", ())),
        unsupported=asdict(outcome.reason) if isinstance(outcome, Unsupported) else None,
        violations=tuple(asdict(v) for v in outcome.violations) if isinstance(outcome, RejectedUnauthorizedChange)
        else (),
        errors=tuple(str(e) for e in outcome.errors) if isinstance(outcome, CompileFailed) else (),
        tokens=outcome.usage.total, attempts=outcome.attempts, record=record, wall_s=time.perf_counter() - started)


def _prompt_id(prompt: str) -> str:
    import hashlib

    return "nl-" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def compile_work(prompt: str, provider, *, base: Mapping[str, Any] | None,
                 cassette: Path) -> Callable[[Report, Cancelled], CompileResult]:
    """The compile job on ``provider``. Every call is recorded to ``cassette`` so the experiment can replay it.

    ``compile_problem`` has no cancel hook, so a compile runs to its end (at
    most ``1 + MAX_REPAIR_ROUNDS`` provider calls).
    """
    def work(_report: Report, _cancelled: Cancelled) -> CompileResult:
        from orbitopt.agent.providers import RecordingProvider

        recorded = RecordingProvider(provider, cassette)
        return compile_round(prompt, recorded, base=base, tool_ids=known_tool_ids())

    return work


# ---------------------------------------------------------------------------------------------------------- run
@dataclass(frozen=True)
class RunProgress:
    """What the run has done so far: the iteration records (trace form) and the step in flight."""

    iterations: tuple[Mapping[str, Any], ...]
    running: str | None
    limits: Mapping[str, Any]


@dataclass(frozen=True)
class RunResult:
    experiment_dir: Path
    run_dir: Path
    manifest_digest: str
    trace: Mapping[str, Any]
    problem: Mapping[str, Any]
    wall_s: float


class ObservedPlanner:
    """Wraps a planner and reports the run's records before each proposal; never changes a proposal."""

    def __init__(self, inner, report: Report, limits: Mapping[str, Any]):
        self.inner, self.report, self.limits = inner, report, limits
        self.name = inner.name

    def propose(self, state):
        history = tuple(r.to_dict() for r in state.history)
        self.report(RunProgress(history, "planner", self.limits))
        proposal = self.inner.propose(state)
        raw = proposal.raw if isinstance(proposal.raw, Mapping) else {}
        self.report(RunProgress(history, str(raw.get("solver_id") or raw.get("action") or "proposal"), self.limits))
        return proposal


@dataclass(frozen=True)
class Experiment:
    """What studio.json holds (besides the format and timestamps)."""

    rounds: tuple[Mapping[str, Any], ...]
    base: Mapping[str, Any]
    edits: tuple[Edit, ...]
    provider: Mapping[str, Any]
    planner: str
    seed: int
    prompt: str = ""

    def to_dict(self, *, confirmed_digest: str, manifest_digest: str | None) -> dict:
        return {"format": EXPERIMENT_FORMAT, "prompt": self.prompt, "rounds": [dict(r) for r in self.rounds],
                "base": dict(self.base), "edits": [e.to_dict() for e in self.edits], "provider": dict(self.provider),
                "planner": self.planner, "seed": self.seed,
                "confirmed_digest": confirmed_digest, "run": RUN_SUBDIR, "manifest_digest": manifest_digest,
                "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


def new_experiment_dir(root: Path, problem_id: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    for n in range(1, 10_000):
        path = Path(root) / (f"{problem_id}-{stamp}" + (f"-{n}" if n > 1 else ""))
        if not path.exists():
            return path
    raise RuntimeError(f"cannot find a free experiment directory under {root}")


def run_limits(problem) -> dict:
    """The run's limits as the runtime resolves them (no budget override)."""
    from dataclasses import asdict

    from orbitopt.agent.runtime import Limits

    return asdict(Limits.resolve(problem.budget, None, len(problem.variables)))


def run_work(problem_doc: Mapping[str, Any], experiment: Experiment, *, choice: ProviderChoice, planner_script: Any,
             root: Path, compile_cassette: Path | None) -> Callable[[Report, Cancelled], RunResult]:
    """The run job: solve with cooperative cancel, save the run with the runtime's persistence, write studio.json.

    A cancelled run is saved like any other; its trace says ``cancelled``.
    """
    def work(report: Report, cancelled: Cancelled) -> RunResult:
        from orbitopt.agent.commands import tools_for
        from orbitopt.agent.runtime import LLMPlanner, RuleBasedPlanner, save_run, solve
        from orbitopt.optimization import parse_problem

        started = time.perf_counter()
        problem = parse_problem(problem_doc)
        limits = run_limits(problem)
        if experiment.planner == "llm":
            inner = LLMPlanner(build_chat_provider(choice, script=planner_script or mock_planner_reply))
        else:
            inner = RuleBasedPlanner()
        report(RunProgress((), "planner", limits))
        outcome = solve(problem, planner=ObservedPlanner(inner, report, limits), seed=experiment.seed,
                        tools=tools_for(problem), cancel=cancelled)
        exp_dir = new_experiment_dir(root, problem.id)
        run_dir = exp_dir / RUN_SUBDIR
        digest = save_run(outcome, run_dir)
        if compile_cassette is not None and compile_cassette.is_file():
            shutil.copyfile(compile_cassette, exp_dir / COMPILE_CASSETTE)
        (exp_dir / EXPERIMENT_FILE).write_text(json.dumps(
            experiment.to_dict(confirmed_digest=problem.digest(), manifest_digest=digest), indent=2, sort_keys=True,
            ensure_ascii=False) + "\n", encoding="utf-8")
        return RunResult(exp_dir, run_dir, digest, outcome.trace.to_dict(), problem.to_dict(),
                         time.perf_counter() - started)

    return work


# ------------------------------------------------------------------------------------------------------- replay
EQUAL, DIVERGED, NOT_RECORDED = "equal", "diverged", "not_recorded"


@dataclass(frozen=True)
class ReplayPart:
    name: str
    status: str
    details: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReplayResult:
    """The replay of an experiment (or of a bare run directory): one part per stage, and the recorded trace."""

    path: Path
    parts: tuple[ReplayPart, ...]
    trace: Mapping[str, Any] | None
    problem: Mapping[str, Any] | None
    meta: Mapping[str, Any]
    digest_checked: bool
    wall_s: float = 0.0

    @property
    def equal(self) -> bool:
        return all(p.status != DIVERGED for p in self.parts) and any(p.status == EQUAL for p in self.parts)

    @property
    def status(self) -> str:
        return "EQUAL" if self.equal else "DIVERGED"


def _replay_compile(exp_dir: Path, meta: Mapping[str, Any], run_problem: Mapping[str, Any] | None) -> list[ReplayPart]:
    from orbitopt.agent.providers import ReplayProvider

    rounds = meta.get("rounds") or []
    parts: list[ReplayPart] = []
    if not rounds:
        parts.append(ReplayPart("compile", NOT_RECORDED, ("the problem was written by hand: no compile to replay",)))
    else:
        cassette = exp_dir / COMPILE_CASSETTE
        details, diverged = [], False
        try:
            provider = ReplayProvider(cassette)
        except (OSError, ValueError) as exc:
            provider, diverged = None, True
            details.append(f"cannot open the compile cassette: {exc}")
        for n, rnd in enumerate(rounds if provider is not None else (), start=1):
            got = compile_round(rnd["prompt"], provider, base=rnd.get("base"), tool_ids=rnd.get("known_tool_ids", ()))
            same = got.outcome == rnd.get("outcome") and got.record.get("result_digest") == rnd.get("result_digest")
            diverged |= not same
            why = "; ".join(got.errors) if got.errors else "the result digest differs"
            details.append(f"round {n}: {got.outcome}" + ("" if same else f" (recorded {rnd.get('outcome')}: {why})"))
        parts.append(ReplayPart("compile", DIVERGED if diverged else EQUAL, tuple(details)))
    try:
        edits = [Edit.from_dict(e) for e in meta.get("edits") or []]
        rebuilt, _ = validate_doc(apply_edits(meta["base"], edits))
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        return parts + [ReplayPart("edits", DIVERGED, (f"cannot re-apply the recorded edits: {exc}",))]
    want = meta.get("confirmed_digest")
    got = None if rebuilt is None else rebuilt.digest()
    from orbitopt.optimization import parse_problem

    ran = None
    try:
        ran = None if run_problem is None else parse_problem(run_problem).digest()
    except Exception:  # noqa: BLE001 -- an unreadable problem.json is the run part's integrity finding
        ran = None
    same = got is not None and got == want == ran
    parts.append(ReplayPart("edits", EQUAL if same else DIVERGED, (
        f"{len(edits)} recorded edit(s) re-applied to the compiled problem: digest {got}",
        f"confirmed digest {want}", f"run/problem.json digest {ran}")))
    return parts


class _Counted:
    """Counts the proposals a recorded planner has served."""

    def __init__(self, inner):
        self.inner, self.served, self.name = inner, 0, inner.name

    def propose(self, state):
        self.served += 1
        return self.inner.propose(state)


def _replay_to_cancellation(run_dir: Path, trace: Mapping[str, Any], tools) -> str:
    """The digest of the trace re-executed from its proposals with the cancel firing after the last one.

    Bridge for ``agent-trace-1.0``: that trace does not record where a run
    was cancelled, so ``replay`` runs a cancelled run's recording to its end
    and reports a difference. The runtime checks cancellation before each
    iteration, so the recorded position is exactly "after the last recorded
    proposal"; this re-execution passes that position as ``cancel``.
    """
    from orbitopt.agent.runtime import Policy, Proposal, RecordedPlanner, solve
    from orbitopt.optimization import Budget, parse_problem

    problem = parse_problem(json.loads((run_dir / "problem.json").read_text(encoding="utf-8")))
    proposals = [Proposal.from_dict(r["proposal"]) for r in trace["iterations"]]
    planner = _Counted(RecordedPlanner(proposals, name=trace["planner"]))
    outcome = solve(problem, planner=planner, budget=Budget(**trace["limits"]),
                    policy=Policy.from_dict(trace["policy"]), tools=tools, seed=trace["seed"],
                    cancel=lambda: planner.served >= len(proposals))
    return outcome.trace.digest()


def _replay_runtime(run_dir: Path, expected: str | None) -> tuple[ReplayPart, Mapping | None, Mapping | None]:
    from orbitopt.agent.commands import tools_for
    from orbitopt.agent.runtime import load_trace, replay
    from orbitopt.optimization import parse_problem

    try:
        problem_doc = json.loads((run_dir / "problem.json").read_text(encoding="utf-8"))
        tools = tools_for(parse_problem(problem_doc))
    except Exception:  # noqa: BLE001 -- replay reports the unreadable file itself
        problem_doc, tools = None, None
    report = replay(run_dir, tools=tools, expected_manifest_digest=expected)
    try:
        trace = load_trace(run_dir)
    except (OSError, ValueError):
        trace = None
    details = [f"recorded trace {report.recorded_digest}", f"replayed trace {report.replayed_digest}",
               f"sealed run re-graded: {report.sealed_verdict}"]
    if report.equal:
        return ReplayPart("run", EQUAL, tuple(details)), trace, problem_doc
    if (trace is not None and trace["outcome"]["status"] == "cancelled" and not report.integrity
            and not any(d.startswith("sealed/") for d in report.differences)):
        if _replay_to_cancellation(run_dir, trace, tools) == trace["digest"]:
            return ReplayPart("run", EQUAL, (
                f"cancelled run re-executed up to the recorded cancellation, after iteration "
                f"{len(trace['iterations'])}: trace {trace['digest']}", details[2])), trace, problem_doc
    return ReplayPart("run", DIVERGED, tuple(report.integrity) + tuple(report.differences)), trace, problem_doc


def replay_work(path: Path, *, expected_manifest_digest: str | None) -> Callable[[Report, Cancelled], ReplayResult]:
    """The replay job: no planner and no model is called. ``path`` is an experiment or a bare run directory."""
    def work(_report: Report, _cancelled: Cancelled) -> ReplayResult:
        started = time.perf_counter()
        root = Path(path)
        meta: Mapping[str, Any] = {}
        if (root / EXPERIMENT_FILE).is_file():
            try:
                meta = json.loads((root / EXPERIMENT_FILE).read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                return ReplayResult(root, (ReplayPart("experiment", DIVERGED, (f"studio.json: {exc}",)),), None,
                                    None, {}, expected_manifest_digest is not None)
            run_dir = root / str(meta.get("run", RUN_SUBDIR))
        else:
            run_dir = root
        run_part, trace, problem_doc = _replay_runtime(run_dir, expected_manifest_digest)
        parts = (_replay_compile(root, meta, problem_doc) if meta else []) + [run_part]
        return ReplayResult(root, tuple(parts), trace, problem_doc, meta, expected_manifest_digest is not None,
                            time.perf_counter() - started)

    return work
