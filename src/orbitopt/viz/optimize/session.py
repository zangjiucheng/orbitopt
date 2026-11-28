"""The Optimize workspace's presenter: one state machine, the IR editor, and the jobs it hands out.

The session never runs work itself. An action that needs work (compile,
run, replay) moves the phase and returns a :class:`Job`; whoever owns the
thread runs ``job.work(report, cancelled)`` and hands the outcome back
through :meth:`StudioSession.job_finished` (or ``job_failed``), and progress
through :meth:`StudioSession.job_progress`. The workspace does that on a
QThread; the tests do it inline. Listeners are told after every change.

Phases and the actions that move between them are one table,
:data:`TRANSITIONS`; an action the table does not list for the current
phase raises :class:`IllegalTransition`, so a stray click can never start a
run on an unconfirmed problem or edit a problem while it runs.
"""

from __future__ import annotations

import tempfile
import threading
from dataclasses import dataclass, field
from enum import Enum
from itertools import count
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from orbitopt.viz.optimize import backend
from orbitopt.viz.optimize.editing import Edit, IrDraft, blank_problem, coerce
from orbitopt.viz.optimize.views import RunView, run_view, run_view_from_trace, view_at_step


class Phase(str, Enum):
    IDLE = "idle"
    COMPILING = "compiling"
    CLARIFYING = "needs_clarification"
    DRAFT = "draft"
    EDITING = "editing"
    CONFIRMED = "confirmed"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"
    REPLAYING = "replaying"
    REPLAY = "replay"


class Action(str, Enum):
    COMPILE = "compile"
    COMPILED = "compiled"          # job: the compiler accepted the problem
    CLARIFY = "clarify"            # job: the compiler asks questions
    REFUSE = "refuse"              # job: unsupported, unauthorised change, compile failure, provider error
    EDIT = "edit"
    CONFIRM = "confirm"
    RUN = "run"
    PROGRESS = "progress"
    CANCEL = "cancel"
    FINISH = "finish"              # job: the run ended (any status but cancelled / refused)
    CANCELLED = "cancelled"        # job: the run stopped on the cancel request
    FAIL = "fail"                  # job: the run raised, was refused, or its planner failed
    REPLAY = "replay"
    REPLAYED = "replayed"          # job
    RESET = "reset"


_P, _A = Phase, Action
_AFTER_RUN = {_A.EDIT: _P.EDITING, _A.RUN: _P.RUNNING, _A.REPLAY: _P.REPLAYING, _A.RESET: _P.IDLE}
TRANSITIONS: dict[Phase, dict[Action, Phase]] = {
    _P.IDLE: {_A.COMPILE: _P.COMPILING, _A.EDIT: _P.EDITING, _A.REPLAY: _P.REPLAYING, _A.RESET: _P.IDLE},
    _P.COMPILING: {_A.COMPILED: _P.EDITING, _A.CLARIFY: _P.CLARIFYING, _A.REFUSE: _P.DRAFT},
    _P.CLARIFYING: {_A.COMPILE: _P.COMPILING, _A.EDIT: _P.EDITING, _A.REPLAY: _P.REPLAYING, _A.RESET: _P.IDLE},
    _P.DRAFT: {_A.COMPILE: _P.COMPILING, _A.EDIT: _P.EDITING, _A.REPLAY: _P.REPLAYING, _A.RESET: _P.IDLE},
    _P.EDITING: {_A.EDIT: _P.EDITING, _A.COMPILE: _P.COMPILING, _A.CONFIRM: _P.CONFIRMED, _A.REPLAY: _P.REPLAYING,
                 _A.RESET: _P.IDLE},
    _P.CONFIRMED: {_A.EDIT: _P.EDITING, _A.RUN: _P.RUNNING, _A.REPLAY: _P.REPLAYING, _A.RESET: _P.IDLE},
    _P.RUNNING: {_A.PROGRESS: _P.RUNNING, _A.CANCEL: _P.RUNNING, _A.FINISH: _P.FINISHED,
                 _A.CANCELLED: _P.CANCELLED, _A.FAIL: _P.FAILED},
    _P.FINISHED: dict(_AFTER_RUN),
    _P.CANCELLED: dict(_AFTER_RUN),
    _P.FAILED: dict(_AFTER_RUN),
    _P.REPLAYING: {_A.REPLAYED: _P.REPLAY},
    _P.REPLAY: {_A.EDIT: _P.EDITING, _A.REPLAY: _P.REPLAYING, _A.RESET: _P.IDLE},
}
JOB_PHASES = frozenset({_P.COMPILING, _P.RUNNING, _P.REPLAYING})
FAILED_RUNS = frozenset({"refused"})


class IllegalTransition(RuntimeError):
    def __init__(self, phase: Phase, action: Action, why: str = ""):
        self.phase, self.action = phase, action
        super().__init__(f"cannot {action.value} while {phase.value}" + (f": {why}" if why else ""))


def next_phase(phase: Phase, action: Action) -> Phase:
    try:
        return TRANSITIONS[phase][action]
    except KeyError:
        raise IllegalTransition(phase, action) from None


@dataclass
class Job:
    """Work for a background thread. ``cancel`` is the cooperative cancel flag the runtime polls."""

    kind: str
    work: Callable[[Callable[[Any], None], Callable[[], bool]], Any]
    id: int
    cancel: threading.Event = field(default_factory=threading.Event)

    def __call__(self, report: Callable[[Any], None] = lambda _u: None):
        return self.work(report, self.cancel.is_set)


@dataclass(frozen=True)
class CompileView:
    """The last compile round as the left column shows it."""

    outcome: str
    questions: tuple[Mapping[str, Any], ...] = ()
    assumptions: tuple[Mapping[str, Any], ...] = ()
    unsupported: Mapping[str, Any] | None = None
    violations: tuple[Mapping[str, Any], ...] = ()
    errors: tuple[str, ...] = ()
    tokens: int = 0
    attempts: int = 0
    wall_s: float = 0.0


class StudioSession:
    """The workspace's whole state. Mutate it only through its methods."""

    def __init__(self, *, root: Path | None = None, environ: Mapping[str, str] | None = None,
                 scratch: Path | None = None):
        self.root = Path(root) if root is not None else backend.default_root()
        self.environ = environ
        self.choices = backend.provider_choices(environ)
        self.provider = self.choices[0]
        self.planner = "rule"
        self.seed = 0
        self.phase = Phase.IDLE
        self.prompt = ""
        self.example: backend.Example | None = None
        self.mock_replies: list[str] | None = None
        self.planner_replies: Any = None
        self.compile_view: CompileView | None = None
        self.rounds: list[Mapping[str, Any]] = []
        self.compile_tokens = 0
        self.draft: IrDraft | None = None
        self.confirmed: Any = None
        self.limits: Mapping[str, Any] | None = None
        self.live: tuple[Mapping[str, Any], ...] = ()
        self.running_label: str | None = None
        self.result: backend.RunResult | None = None
        self.error = ""
        self.cancel_requested = False
        self.replay_result: backend.ReplayResult | None = None
        self.step = 0
        self.job: Job | None = None
        self._ids = count(1)
        self._mock = None
        self._scratch = Path(scratch) if scratch is not None else Path(tempfile.mkdtemp(prefix="orbitopt-studio-"))
        self.cassette = self._scratch / "compile-1.cassette.jsonl"
        self._listeners: list[Callable[[], None]] = []

    # ------------------------------------------------------------------------------------------------ plumbing
    def subscribe(self, listener: Callable[[], None]) -> None:
        self._listeners.append(listener)

    def _changed(self) -> None:
        for listener in list(self._listeners):
            listener()

    def can(self, action: Action) -> bool:
        """Whether ``action`` is legal now, including the guards beyond the phase table."""
        if action not in TRANSITIONS[self.phase]:
            return False
        if action is Action.CONFIRM:
            return self.draft is not None and self.draft.valid
        if action is Action.RUN:
            return self.confirmed is not None
        if action is Action.COMPILE:
            return bool(self.prompt.strip())
        if action is Action.CANCEL:
            return self.job is not None and not self.cancel_requested
        return True

    def _go(self, action: Action) -> None:
        self.phase = next_phase(self.phase, action)

    def _job(self, kind: str, work) -> Job:
        self.job = Job(kind, work, next(self._ids))
        return self.job

    def _current(self, job: Job) -> bool:
        return self.job is not None and job.id == self.job.id

    # -------------------------------------------------------------------------------------------- the inputs
    @property
    def real_provider(self) -> bool:
        return self.provider.real

    @property
    def banner(self) -> str:
        return backend.provider_banner(self.provider, self.choices, self.environ)

    def _idle_input(self) -> None:
        if self.phase in JOB_PHASES:
            raise IllegalTransition(self.phase, Action.EDIT, "a job is running")

    def set_prompt(self, text: str) -> None:
        self._idle_input()
        self.prompt = text
        self._changed()

    def select_example(self, key: str | None) -> None:
        self._idle_input()
        self.example = None if key is None else backend.example(key)
        self.mock_replies, self._mock = None, None
        if self.example is not None:
            self.prompt = self.example.prompt
        self._changed()

    def load_mock_replies(self, replies: Sequence[Any]) -> None:
        """Script the offline mock (the CLI's ``--mock-replies`` list: strings or JSON objects)."""
        import json

        self._idle_input()
        self.mock_replies = [r if isinstance(r, str) else json.dumps(r, sort_keys=True) for r in replies]
        self._mock = None
        self._changed()

    def set_planner_replies(self, replies: Any) -> None:
        """Script the offline LLM planner: a list of proposal objects or strings, or a request function."""
        import json

        self.planner_replies = replies if callable(replies) else \
            [r if isinstance(r, str) else json.dumps(r, sort_keys=True) for r in replies]

    def set_provider(self, key: str) -> None:
        self._idle_input()
        self.provider = next(c for c in self.choices if c.key == key)
        self._changed()

    def set_planner(self, kind: str) -> None:
        if kind not in ("rule", "llm"):
            raise ValueError(f"planner must be 'rule' or 'llm', not {kind!r}")
        self._idle_input()
        self.planner = kind
        self._changed()

    def set_connection(self, provider: backend.ProviderChoice, planner: str) -> None:
        """Apply the workspace settings together for subsequent jobs."""
        self._idle_input()
        if planner not in ("rule", "llm"):
            raise ValueError("Choose a rule-based or LLM planner.")
        self.provider = provider
        self.planner = planner
        self._changed()

    def _compile_provider(self):
        if self.provider.real:
            return backend.build_chat_provider(self.provider)
        script = self.mock_replies if self.mock_replies is not None else \
            (list(self.example.replies) if self.example is not None and self.prompt.startswith(self.example.prompt)
             else None)
        if script is None:
            raise IllegalTransition(self.phase, Action.COMPILE,
                                    "the offline mock answers the bundled examples or a loaded replies file only; "
                                    "pick an example, load replies, configure a provider, or write the problem by hand")
        if self._mock is None:     # one mock per script, so a clarification round gets the next reply
            self._mock = backend.build_chat_provider(self.provider, script=script)
        return self._mock

    # ---------------------------------------------------------------------------------------------- compiling
    def compile(self) -> Job:
        """Compile the prompt; from the editor this is a revision that may not change what the user locked."""
        if not self.can(Action.COMPILE):
            raise IllegalTransition(self.phase, Action.COMPILE, "" if self.prompt.strip() else "the prompt is empty")
        provider = self._compile_provider()
        base = self.draft.problem.to_dict() if self.phase is Phase.EDITING and self.draft and self.draft.valid else None
        if self.phase is Phase.IDLE:
            self.rounds, self.compile_tokens = [], 0
        self._go(Action.COMPILE)
        self.error = ""
        job = self._job("compile", backend.compile_work(self.prompt, provider, base=base, cassette=self.cassette))
        self._changed()
        return job

    def answer(self, answers: Sequence[str]) -> Job:
        """Answer the compiler's questions and compile again with the answers in the prompt."""
        if self.phase is not Phase.CLARIFYING:
            raise IllegalTransition(self.phase, Action.COMPILE, "there are no questions to answer")
        self.prompt = backend.with_answers(self.prompt, self.compile_view.questions, answers)
        return self.compile()

    def _compiled(self, result: backend.CompileResult) -> None:
        self.rounds.append(dict(result.record))
        self.compile_tokens += result.tokens
        self.compile_view = CompileView(result.outcome, result.questions, result.assumptions, result.unsupported,
                                        result.violations, result.errors, result.tokens, result.attempts,
                                        result.wall_s)
        if result.outcome == "compile":
            self.draft = IrDraft.start(result.problem)
            self._go(Action.COMPILED)
        elif result.outcome == "needs_clarification":
            if result.draft is not None:
                self.draft = IrDraft.start(result.draft)
            self._go(Action.CLARIFY)
        else:
            if result.draft is not None and self.draft is None:
                self.draft = IrDraft.start(result.draft)
            self._go(Action.REFUSE)

    # ------------------------------------------------------------------------------------------------ editing
    def start_blank(self) -> None:
        """Write the problem by hand: no compile, no LLM."""
        self._go_edit()
        self.draft = IrDraft.start(blank_problem())
        self.rounds, self.compile_view = [], None
        self._changed()

    def _go_edit(self) -> None:
        self._go(Action.EDIT)
        self.confirmed = None

    def edit_draft(self) -> None:
        """Open the current problem (or the compiler's draft) in the editor; unfreezes a confirmed problem."""
        self._go_edit()
        if self.draft is None:
            self.draft = IrDraft.start(blank_problem())
        self._changed()

    def apply(self, edit: Edit) -> None:
        if self.phase is not Phase.EDITING:
            raise IllegalTransition(self.phase, Action.EDIT, "the problem is not open for editing")
        self.draft = self.draft.apply(edit)
        self._changed()

    def set_cell(self, section: str, index: int, field_name: str, text: str) -> None:
        self.apply(Edit("set", section, index, field_name, coerce(field_name, text)))

    def add_item(self, section: str) -> None:
        self.apply(Edit("add", section))

    def remove_item(self, section: str, index: int) -> None:
        self.apply(Edit("remove", section, index))

    def confirm(self) -> None:
        """Freeze the problem: its digest is what the run will solve; only now can it run."""
        if not self.can(Action.CONFIRM):
            raise IllegalTransition(self.phase, Action.CONFIRM, "the problem has validation errors"
                                    if self.phase is Phase.EDITING else "")
        self._go(Action.CONFIRM)
        self.confirmed = self.draft.problem
        self._changed()

    # ---------------------------------------------------------------------------------------------- running
    def run(self) -> Job:
        if not self.can(Action.RUN):
            raise IllegalTransition(self.phase, Action.RUN, "confirm the problem first")
        problem = self.confirmed
        experiment = backend.Experiment(tuple(self.rounds), self.draft.base, self.draft.edits,
                                        self.provider.describe(), self.planner, self.seed, self.prompt)
        self._go(Action.RUN)
        self.live, self.running_label, self.result, self.error = (), "planner", None, ""
        self.cancel_requested, self.replay_result = False, None
        self.limits = backend.run_limits(problem)
        planner_script = self.planner_replies
        provider = self.provider if self.planner == "llm" else backend.MOCK
        job = self._job("run", backend.run_work(problem.to_dict(), experiment, choice=provider,
                                                planner_script=planner_script, root=self.root,
                                                compile_cassette=self.cassette if self.rounds else None))
        self._changed()
        return job

    def cancel(self) -> None:
        """Ask the runtime to stop at its next iteration boundary; the run is still saved."""
        if not self.can(Action.CANCEL):
            raise IllegalTransition(self.phase, Action.CANCEL)
        self._go(Action.CANCEL)
        self.cancel_requested = True
        self.job.cancel.set()
        self._changed()

    def _ran(self, result: backend.RunResult) -> None:
        self.result = result
        status = result.trace["outcome"]["status"]
        stop = result.trace["outcome"]["stop_reason"]
        if status == "cancelled":
            self._go(Action.CANCELLED)
        elif status in FAILED_RUNS or stop == "planner_failed":
            self.error = "; ".join(result.trace["outcome"].get("refusal") or []) or f"the run stopped: {stop}"
            self._go(Action.FAIL)
        else:
            self._go(Action.FINISH)

    # ----------------------------------------------------------------------------------------------- replay
    def replay(self, path: Path | None = None, *, expected_manifest_digest: str | None = None) -> Job:
        """Replay an experiment (default: the run just finished) with no model call."""
        if Action.REPLAY not in TRANSITIONS[self.phase]:
            raise IllegalTransition(self.phase, Action.REPLAY)
        if path is None:
            if self.result is None:
                raise IllegalTransition(self.phase, Action.REPLAY, "no run to replay; open a saved run")
            path, expected_manifest_digest = self.result.experiment_dir, self.result.manifest_digest
        self._go(Action.REPLAY)
        self.replay_result, self.error = None, ""
        job = self._job("replay", backend.replay_work(Path(path), expected_manifest_digest=expected_manifest_digest))
        self._changed()
        return job

    def _replayed(self, result: backend.ReplayResult) -> None:
        self.replay_result = result
        self.step = len(result.trace["iterations"]) if result.trace else 0
        if result.problem is not None:   # editing a replayed problem starts a new, hand-edited lineage
            self.draft = IrDraft.start(result.problem)
            self.confirmed, self.rounds, self.compile_tokens, self.compile_view = None, [], 0, None
        self._go(Action.REPLAYED)

    def set_step(self, step: int) -> None:
        if self.phase is not Phase.REPLAY or self.replay_result is None or self.replay_result.trace is None:
            raise IllegalTransition(self.phase, Action.PROGRESS, "no replay to step through")
        self.step = max(0, min(int(step), len(self.replay_result.trace["iterations"])))
        self._changed()

    def reset(self) -> None:
        self._go(Action.RESET)
        self.draft = self.confirmed = self.compile_view = self.result = self.replay_result = None
        self.rounds, self.compile_tokens, self.error, self.live = [], 0, "", ()
        self.cassette = self._scratch / f"compile-{next(self._ids)}.cassette.jsonl"
        self._changed()

    # ------------------------------------------------------------------------------------------- job results
    def job_progress(self, job: Job, update: Any) -> None:
        if not self._current(job) or self.phase is not Phase.RUNNING:
            return
        self.live, self.running_label, self.limits = tuple(update.iterations), update.running, update.limits
        self._changed()

    def job_finished(self, job: Job, result: Any) -> None:
        if not self._current(job):
            return
        self.job = None
        {"compile": self._compiled, "run": self._ran, "replay": self._replayed}[job.kind](result)
        self._changed()

    def job_failed(self, job: Job, message: str) -> None:
        """The job raised: a compile goes to draft, a run fails, a replay shows the error as divergence."""
        if not self._current(job):
            return
        self.job, self.error = None, message
        if job.kind == "compile":
            self.compile_view = CompileView("provider_error", errors=(message,))
            self._go(Action.REFUSE)
        elif job.kind == "run":
            self._go(Action.FAIL)
        else:
            self.replay_result = backend.ReplayResult(Path("."), (backend.ReplayPart("replay", backend.DIVERGED,
                                                                                    (message,)),), None, None, {},
                                                      False)
            self._go(Action.REPLAYED)
        self._changed()

    # -------------------------------------------------------------------------------------------- read models
    def _objective(self, doc: Mapping[str, Any] | None) -> tuple[str, str]:
        objectives = (doc or {}).get("objectives") or [{}]
        return objectives[0].get("name", ""), objectives[0].get("sense", "minimize")

    def run_view(self) -> RunView | None:
        """The run panels' data: the replay at its step, the finished run, or the live run."""
        if self.phase is Phase.REPLAY and self.replay_result and self.replay_result.trace:
            name, sense = self._objective(self.replay_result.problem)
            return view_at_step(self.replay_result.trace, self.step, objective_name=name, sense=sense)
        if self.result is not None and self.phase in (Phase.FINISHED, Phase.CANCELLED, Phase.FAILED):
            name, sense = self._objective(self.result.problem)
            return run_view_from_trace(self.result.trace, objective_name=name, sense=sense,
                                       extra_tokens=self.compile_tokens)
        if self.phase is Phase.RUNNING and self.limits is not None:
            name, sense = self._objective(self.confirmed.to_dict() if self.confirmed is not None else None)
            return run_view(self.live, limits=self.limits, objective_name=name, sense=sense,
                            extra_tokens=self.compile_tokens, running=self.running_label)
        return None

    @property
    def digest(self) -> str | None:
        if self.confirmed is not None:
            return self.confirmed.digest()
        return None if self.draft is None else self.draft.digest

    @property
    def editable(self) -> bool:
        return self.phase is Phase.EDITING
