"""E14-09 Optimize workspace, without a window: the presenter's state machine, the IR editor, and the compile,
run, cancel and replay jobs on the offline mock provider (no network, no model)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("scipy")

from orbitopt.viz.optimize import backend  # noqa: E402
from orbitopt.viz.optimize.editing import Edit, IrDraft, apply_edits, blank_problem, coerce  # noqa: E402
from orbitopt.viz.optimize.session import (TRANSITIONS, Action, IllegalTransition, Phase,  # noqa: E402
                                           StudioSession, next_phase)
from orbitopt.viz.optimize.views import (STATUS_NAMES, expression_html, problem_html, report_document,  # noqa: E402
                                         report_html, run_view_from_trace, view_at_step)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def session(tmp_path):
    return StudioSession(root=tmp_path / "runs", environ={}, scratch=tmp_path / "scratch")


def finish(session, job, *, on_progress=None):
    """Run ``job`` inline, the way the workspace's thread does, and hand the result back."""
    def report(update):
        session.job_progress(job, update)
        if on_progress is not None:
            on_progress(update)

    session.job_finished(job, job(report))


def compiled(session, key="halfplane"):
    session.select_example(key)
    finish(session, session.compile())
    return session


def ran(session, key="halfplane"):
    compiled(session, key)
    session.confirm()
    finish(session, session.run())
    return session


# ------------------------------------------------------------------------------------------ the state machine
def test_every_phase_has_a_row_and_jobs_take_no_user_input():
    assert set(TRANSITIONS) == set(Phase)
    for phase in (Phase.COMPILING, Phase.REPLAYING):
        assert not {Action.EDIT, Action.RUN, Action.COMPILE, Action.CONFIRM} & set(TRANSITIONS[phase])
    assert set(TRANSITIONS[Phase.RUNNING]) == {Action.PROGRESS, Action.CANCEL, Action.FINISH, Action.CANCELLED,
                                               Action.FAIL}
    assert next_phase(Phase.CONFIRMED, Action.RUN) is Phase.RUNNING
    with pytest.raises(IllegalTransition):
        next_phase(Phase.EDITING, Action.RUN)          # an unconfirmed problem never runs


@pytest.mark.parametrize("action", ["run", "confirm", "cancel", "set_step"])
def test_illegal_actions_raise_and_change_nothing(session, action):
    args = {"set_step": (1,)}.get(action, ())
    with pytest.raises(IllegalTransition):
        getattr(session, action)(*args)
    assert session.phase is Phase.IDLE and session.job is None


def test_no_edit_while_running_and_no_run_twice(session):
    compiled(session)
    session.confirm()
    with pytest.raises(IllegalTransition):
        session.set_cell("variables", 0, "upper", "5")      # confirmed: frozen
    job = session.run()
    with pytest.raises(IllegalTransition):
        session.run()
    with pytest.raises(IllegalTransition):
        session.set_prompt("something else")
    with pytest.raises(IllegalTransition):
        session.edit_draft()
    finish(session, job)
    assert session.phase is Phase.FINISHED


def test_a_stale_job_result_is_ignored(session):
    compiled(session)
    session.confirm()
    job = session.run()
    finish(session, job)
    session.edit_draft()
    session.job_finished(job, None)                          # a late duplicate: not the current job
    assert session.phase is Phase.EDITING


# ------------------------------------------------------------------------------------------------- compiling
def test_accepted_compile_opens_the_editor_with_user_items_locked(session):
    compiled(session)
    assert session.phase is Phase.EDITING and session.compile_view.outcome == "compile"
    assert session.draft.valid and session.draft.digest == session.digest
    (constraint,) = session.draft.doc["constraints"]
    assert constraint["origin"] == "user" and constraint["locked"] is True
    assert "<i>x</i><sup>2</sup>" in problem_html(session.draft.doc)


def test_the_offline_mock_refuses_text_it_has_no_reply_for(session):
    session.set_prompt("Maximise the profit of my bakery.")
    with pytest.raises(IllegalTransition, match="bundled examples"):
        session.compile()
    assert session.phase is Phase.IDLE


def test_clarification_shows_questions_and_answers_compile_again(session):
    compiled(session, "clarify")
    assert session.phase is Phase.CLARIFYING
    (question,) = session.compile_view.questions
    assert question["kind"] == "missing_parameter" and "unit cost" in question["text"]
    assert not session.draft.valid                                  # the draft holds c = null
    assert any(i.path == "/parameters/0/value" for i in session.draft.issues)
    finish(session, session.answer(["c = 3"]))
    assert session.phase is Phase.EDITING and session.draft.valid
    assert session.draft.doc["parameters"][0]["value"] == 3.0
    assert session.prompt.endswith("What is the unit cost c of x? c = 3")
    assert [r["outcome"] for r in session.rounds] == ["needs_clarification", "compile"]


def test_a_clarification_draft_can_be_finished_by_hand(session):
    compiled(session, "clarify")
    session.edit_draft()
    session.set_cell("parameters", 0, "value", "3")
    assert session.draft.valid
    session.confirm()
    assert session.phase is Phase.CONFIRMED


def test_unsupported_goes_to_draft_with_the_reason(session):
    session.set_prompt("Minimise the expected cost of a random demand.")
    session.load_mock_replies([{"status": "unsupported", "problem": None, "questions": [], "assumptions": [],
                                "unsupported_reason": {"category": "stochastic",
                                                       "explanation": "the demand is random"},
                                "authorized_changes": [], "tradeoff_quote": None}])
    finish(session, session.compile())
    assert session.phase is Phase.DRAFT
    assert session.compile_view.unsupported == {"category": "stochastic", "explanation": "the demand is random"}
    session.start_blank()                                           # nothing to edit: write it by hand
    assert session.phase is Phase.EDITING and session.draft.valid


# ----------------------------------------------------------------------------------------------- the editor
def test_edit_validation_errors_then_confirm_freezes_the_digest(session):
    compiled(session)
    before = session.digest
    session.set_cell("variables", 0, "upper", "ten")
    assert not session.draft.valid and session.digest is None
    assert [i.path for i in session.draft.issues] == ["/variables/0/upper"]
    assert session.draft.issues_for("variables", 0) and not session.draft.issues_for("variables", 1)
    assert not session.can(Action.CONFIRM)
    with pytest.raises(IllegalTransition, match="validation errors"):
        session.confirm()
    session.set_cell("variables", 0, "upper", "8")
    session.confirm()
    assert session.phase is Phase.CONFIRMED and session.digest == session.confirmed.digest() != before
    assert session.confirmed.variables[0].upper == 8.0


def test_a_user_edit_makes_the_item_user_owned_and_lists_protected_changes(session):
    session.load_mock_replies([{"status": "compiled", "problem": {
        "title": "", "variables": [backend._var("x", 0.0, 10.0), backend._var("y", 0.0, 10.0)], "parameters": [],
        "objectives": [{"name": "f", "expression": "x**2 + y**2", "sense": "minimize", "unit": "",
                        "source": "user"}],
        "constraints": [{"name": "sum", "expression": "x + y", "sense": ">=", "rhs": 1.0, "unit": "",
                         "source": "assumed"}]},
        "questions": [], "assumptions": [{"about": "constraints/sum", "text": "x + y >= 1 assumed"}],
        "unsupported_reason": None, "authorized_changes": [], "tradeoff_quote": None}])
    session.set_prompt("Minimize x squared plus y squared, where x and y are between 0 and 10 and add up to at "
                       "least one.")
    finish(session, session.compile())
    constraint = session.draft.doc["constraints"][0]
    assert constraint["origin"] == "compiler" and constraint["locked"] is False
    session.set_cell("constraints", 0, "rhs", "2")
    constraint = session.draft.doc["constraints"][0]
    assert constraint == {**constraint, "origin": "user", "locked": True, "rhs": 2.0}
    session.set_cell("variables", 1, "upper", "5")
    kinds = {(c.kind, c.item) for c in session.draft.protected_changes()}
    assert ("changed_domain", "variable:y") in kinds


def test_a_revision_compile_cannot_overwrite_what_the_user_locked(session):
    compiled(session)
    session.set_cell("constraints", 0, "rhs", "2")
    edited = session.draft.doc
    assert {c.kind for c in session.draft.protected_changes()} == {"changed_locked_constraint"}
    silently_back = json.loads(backend.example("halfplane").replies[0])     # rhs 1 again, no authorisation
    session.load_mock_replies([silently_back])
    finish(session, session.compile())
    assert session.phase is Phase.DRAFT
    assert session.compile_view.outcome == "rejected_unauthorized_change"
    assert [v["kind"] for v in session.compile_view.violations] == ["changed_locked_constraint"]
    assert session.draft.doc == edited and session.draft.doc["constraints"][0]["rhs"] == 2.0


def test_edits_are_data_that_rebuild_the_same_problem():
    base = blank_problem()
    edits = [Edit("add", "variables"), Edit("set", "variables", 1, "upper", coerce("upper", "4")),
             Edit("set", "objectives", 0, "expression", coerce("expression", "x**2 + x1")),
             Edit("add", "constraints"), Edit("remove", "constraints", 0)]
    draft = IrDraft.start(base)
    for edit in edits:
        draft = draft.apply(Edit.from_dict(json.loads(json.dumps(edit.to_dict()))))
    assert draft.doc == apply_edits(base, edits) and draft.valid
    assert coerce("lower", "") is None and coerce("lower", "1e3") == 1000.0 and coerce("lower", "abc") == "abc"


# ------------------------------------------------------------------------------------------------- running
def test_run_to_verified_on_the_catalogue_qp(session):
    ran(session)
    assert session.phase is Phase.FINISHED
    view = session.run_view()
    assert view.verified and view.status == "verified"
    statuses = {s.name: s.outcome for s in view.statuses}
    assert list(statuses) == list(STATUS_NAMES)                     # four separate statuses, in order
    assert statuses == {"solver_converged": "passed", "constraints_satisfied": "passed",
                        "independently_verified": "passed", "optimality_certified": "not_assessed"}
    assert view.best["objective_values"]["f"] == pytest.approx(0.5, abs=1e-6)
    assert view.fingerprint == session.confirmed.hard_constraint_fingerprint()
    assert {p.kind for p in view.points} == {"feasible", "alternate"}
    tokens = next(m for m in view.meters if m.label == "Tokens")
    assert tokens.used == session.compile_tokens > 0 and tokens.limit == 200_000
    exp = session.result.experiment_dir
    meta = json.loads((exp / "studio.json").read_text())
    assert meta["format"] == backend.EXPERIMENT_FORMAT and meta["confirmed_digest"] == session.confirmed.digest()
    assert meta["provider"] == {"kind": "mock", "model": None, "base_url": None}
    assert (exp / "run" / "manifest.json").is_file() and (exp / backend.COMPILE_CASSETTE).is_file()


def test_unverified_best_stays_unverified(session):
    ran(session, "fc001")
    view = session.run_view()
    assert session.phase is Phase.FINISHED and view.status == "unverified_best" and not view.verified
    statuses = {s.name: s.outcome for s in view.statuses}
    assert statuses["constraints_satisfied"] == "passed" and statuses["independently_verified"] == "not_assessed"


def test_live_progress_reports_each_iteration(session):
    compiled(session)
    session.confirm()
    seen = []
    finish(session, session.run(), on_progress=lambda u: seen.append((len(u.iterations), u.running)))
    assert seen[0] == (0, "planner") and (1, "planner") in seen and seen[-1][0] == 1


@pytest.mark.parametrize("planner", ["rule", "endless-llm"])
def test_cancel_mid_run_saves_a_replayable_cancelled_run(session, planner):
    compiled(session)
    if planner != "rule":
        session.set_planner("llm")
        session.set_planner_replies(lambda request: json.dumps({"action": "solve", "solver_id": "scipy.slsqp"}))
    session.confirm()
    job = session.run()

    def cancel_after_first(update):
        if update.iterations and not session.cancel_requested:
            session.cancel()

    finish(session, job, on_progress=cancel_after_first)
    assert session.phase is Phase.CANCELLED
    trace = session.result.trace
    assert trace["outcome"]["status"] == "cancelled" and len(trace["iterations"]) == 2
    assert session.run_view().best is not None                      # the best so far, unverified
    finish(session, session.replay())
    assert session.phase is Phase.REPLAY and session.replay_result.equal, session.replay_result.parts


def test_policy_violations_of_a_malicious_planner_are_shown_and_never_applied(session):
    compiled(session)
    session.set_planner("llm")
    session.set_planner_replies([
        {"action": "solve", "solver_id": "scipy.slsqp", "constraints": [], "rationale": "drop the constraint"},
        {"action": "solve", "solver_id": "scipy.slsqp", "options": {"constraint_tol": 0.5}},
        {"action": "solve", "solver_id": "scipy.slsqp"}, {"action": "stop"}])
    session.confirm()
    finish(session, session.run())
    view = session.run_view()
    assert [(v.iteration, v.code) for v in view.violations] == [(1, "problem_edit"), (2, "problem_edit")]
    assert [r.kind for r in view.rows] == ["rejected", "rejected", "solve", "stop"]
    assert view.fingerprint == session.confirmed.hard_constraint_fingerprint()
    run_problem = json.loads((session.result.run_dir / "problem.json").read_text())
    assert run_problem["constraints"][0]["rhs"] == 1.0


# -------------------------------------------------------------------------------------------------- replay
def test_replay_is_equal_end_to_end_without_any_model_call(session, monkeypatch):
    ran(session)
    from orbitopt.agent.providers import MockProvider, OpenAICompatProvider

    def no_model(*_a, **_k):
        raise AssertionError("a replay must not call a model")

    monkeypatch.setattr(MockProvider, "complete", no_model)
    monkeypatch.setattr(OpenAICompatProvider, "complete", no_model)
    finish(session, session.replay())
    result = session.replay_result
    assert result.status == "EQUAL" and result.digest_checked
    assert [(p.name, p.status) for p in result.parts] == [("compile", "equal"), ("edits", "equal"), ("run", "equal")]
    session.set_step(0)
    assert session.run_view().rows == ()
    session.set_step(1)
    assert len(session.run_view().rows) == 1 and session.run_view().best_label == "iteration-0001"


def _tamper_result(exp: Path) -> None:
    path = exp / "run" / "results" / "iteration-0001.json"
    data = json.loads(path.read_text())
    data["objective_values"]["f"] = 0.25
    path.write_text(json.dumps(data))


def _tamper_edits(exp: Path) -> None:
    meta = json.loads((exp / "studio.json").read_text())
    meta["edits"].append({"op": "set", "section": "variables", "index": 0, "field": "upper", "value": 9.0})
    (exp / "studio.json").write_text(json.dumps(meta))


def _tamper_cassette(exp: Path) -> None:
    path = exp / backend.COMPILE_CASSETTE
    path.write_text(path.read_text().replace('\\"rhs\\": 1.0', '\\"rhs\\": 0.5'))


@pytest.mark.parametrize("tamper,part", [(_tamper_result, "run"), (_tamper_edits, "edits"),
                                         (_tamper_cassette, "compile")])
def test_a_tampered_experiment_replays_diverged(session, tamper, part):
    ran(session)
    exp = session.result.experiment_dir
    tamper(exp)
    finish(session, session.replay())
    result = session.replay_result
    assert result.status == "DIVERGED"
    assert {p.name for p in result.parts if p.status == "diverged"} >= {part}


def test_a_bare_run_directory_replays_too(session, tmp_path):
    ran(session)
    finish(session, session.replay(session.result.run_dir))
    assert [p.name for p in session.replay_result.parts] == ["run"] and session.replay_result.equal


# ----------------------------------------------------------------------------------------- views and export
def test_math_rendering_degrades_to_text():
    assert expression_html("x**2 + y**2") == "<i>x</i><sup>2</sup> + <i>y</i><sup>2</sup>"
    assert "Σ" in expression_html("sum(x[i]**2 for i in range(3))")
    assert expression_html("x +* 2") == "<tt>x +* 2</tt>"                       # not parseable: plain text
    assert expression_html({"tool_id": "math.erf.v1", "args": {"x": "y"}, "output": None}).startswith("<tt>math")


def test_report_export_keeps_the_four_statuses_apart(session, tmp_path):
    ran(session, "fc001")
    view = session.run_view()
    doc = report_document(view, problem=session.result.problem, meta={"problem_digest": session.digest})
    assert doc["run_status"] == "unverified_best" and doc["verified"] is False
    assert set(doc["statuses"]) == set(STATUS_NAMES)
    page = report_html(doc)
    for name in ("Solver converged", "Constraints satisfied", "Independently verified", "Optimality certified"):
        assert name in page
    json.dumps(doc, allow_nan=False)


def test_views_mark_infeasible_points_and_step_through():
    trace = {"iterations": [
        {"index": 1, "kind": "solve", "evaluations": 5, "tokens": 0, "decision": {"solver_id": "a"},
         "result": {"objective_values": {"f": 1.0}},
         "screen": {"verdict": "failed", "statuses": {"constraints_satisfied": {"outcome": "failed"}}}},
        {"index": 2, "kind": "solve", "evaluations": 5, "tokens": 0, "decision": {"solver_id": "b"},
         "result": {"objective_values": {"f": 3.0}},
         "screen": {"verdict": "not_assessed", "statuses": {"constraints_satisfied": {"outcome": "passed"}}}}],
        "verifications": [], "limits": {"max_iterations": 20, "max_evaluations": 100, "max_tokens": 10},
        "outcome": {"status": "unverified_best", "stop_reason": "planner_stop", "best": "iteration-0002"},
        "usage": {"iterations": 2, "evaluations": 10, "tokens": 0}}
    view = run_view_from_trace(trace, objective_name="f", sense="minimize")
    assert [(p.kind, p.evaluations) for p in view.points] == [("infeasible", 5), ("feasible", 10)]
    assert view.best_path == ((10, 3.0),)                     # the infeasible lower point is never "best"
    assert view_at_step(trace, 1, objective_name="f", sense="minimize").best_label is None


# ----------------------------------------------------------------------------------- providers and startup
def test_providers_come_from_the_environment_and_never_carry_a_key(session, tmp_path):
    assert backend.provider_choices({}) == (backend.MOCK,)
    assert "No LLM provider is configured" in backend.provider_banner(backend.MOCK, (backend.MOCK,), {})
    env = {"ORBITOPT_LLM_MODEL": "gpt-test", "ORBITOPT_LLM_BASE_URL": "https://llm.example/v1",
           "OPENAI_API_KEY": "sk-not-for-the-ui"}
    real = backend.provider_choices(env)[1]
    assert real.real and real.api_key_env == "OPENAI_API_KEY"
    banner = backend.provider_banner(real, backend.provider_choices(env), env)
    assert "OPENAI_API_KEY (set)" in banner and "sk-not-for-the-ui" not in banner
    studio = StudioSession(root=tmp_path / "env-runs", environ=env, scratch=tmp_path / "s2")
    studio.set_provider("env")
    studio.start_blank()
    studio.confirm()
    finish(studio, studio.run())
    text = (studio.result.experiment_dir / "studio.json").read_text()
    assert "gpt-test" in text and "sk-not-for-the-ui" not in text and "OPENAI_API_KEY" not in text


def test_the_workspace_imports_no_agent_or_llm_sdk_until_a_job_runs():
    code = ("import os, sys; os.environ['QT_QPA_PLATFORM'] = 'offscreen'\n"
            "from PySide6.QtWidgets import QApplication; app = QApplication([])\n"
            "from orbitopt.viz.optimize.workspace import OptimizationWorkspace\n"
            "ws = OptimizationWorkspace()\n"
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('openai', 'anthropic', 'litellm')"
            " or m.startswith('orbitopt.agent')))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, timeout=120,
                         env={**__import__("os").environ, "PYTHONPATH": str(REPO / "src")})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "[]"
