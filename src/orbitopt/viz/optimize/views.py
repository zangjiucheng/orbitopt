"""Read models of the Optimize workspace: the problem as math, a run as rows, a curve and four statuses.

Everything here is a pure function of plain data (the IR's JSON form, the
runtime's trace dict), so the live run, a finished run and a replayed run
render through the same code, and the tests need no window.

Honesty rules the views keep:

* the four statuses of a report are shown one by one, each with its own
  outcome and reason; nothing folds them into one check mark;
* a run is ``verified`` only when the runtime says so; ``unverified_best``
  stays visibly unverified, and so does every curve point the guard did not
  find feasible;
* a status the run never graded reads ``not run``, never ``passed``.
"""

from __future__ import annotations

import ast
import html
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

STATUS_NAMES = ("solver_converged", "constraints_satisfied", "independently_verified", "optimality_certified")
STATUS_LABELS = {
    "solver_converged": "Solver converged",
    "constraints_satisfied": "Constraints satisfied",
    "independently_verified": "Independently verified",
    "optimality_certified": "Optimality certified",
}
RUN_STATUS_TEXT = {
    "verified": "Verified: the guard passed the final answer",
    "unverified_best": "Unverified best: a feasible point the guard did not pass",
    "no_feasible": "No feasible point found",
    "budget_exhausted": "Budget exhausted before a verified answer",
    "cancelled": "Cancelled: the best point so far is not verified",
    "policy_violation": "Stopped on a policy violation",
    "refused": "Refused: the runtime could not take this problem",
}
#: StatusBadge state for each run status and status outcome (only ``verified`` and ``passed`` are green)
RUN_STATUS_BADGE = {"verified": "verified", "unverified_best": "warning", "budget_exhausted": "warning",
                    "cancelled": "warning", "no_feasible": "failed", "policy_violation": "failed",
                    "refused": "failed"}
OUTCOME_BADGE = {"passed": "verified", "failed": "failed", "not_assessed": "idle", "not run": "idle"}


# ----------------------------------------------------------------------------------------------------- math
_PRECEDENCE = {ast.Add: 1, ast.Sub: 1, ast.Mult: 2, ast.Div: 2, ast.Pow: 4}
_OPS = {ast.Add: " + ", ast.Sub: " − ", ast.Mult: "·", ast.Div: " / "}
_FUNCS = {"sqrt": "√"}
_SENSES = {"<=": "≤", ">=": "≥", "==": "="}


def _num(value: float) -> str:
    if isinstance(value, bool):
        return str(value)
    if float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6g}"


def _render(node: ast.AST) -> tuple[str, int]:
    """(html, precedence) of one node; raises ``ValueError`` for anything outside the IR's grammar."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return _num(node.value), 9
    if isinstance(node, ast.Name):
        return f"<i>{html.escape(node.id)}</i>", 9
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        index, _ = _render(node.slice)
        return f"<i>{html.escape(node.value.id)}</i><sub>{index}</sub>", 9
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner, prec = _render(node.operand)
        sign = "−" if isinstance(node.op, ast.USub) else "+"
        return sign + (f"({inner})" if prec < 3 else inner), 3
    if isinstance(node, ast.BinOp) and type(node.op) in _PRECEDENCE:
        prec = _PRECEDENCE[type(node.op)]
        left, lp = _render(node.left)
        right, rp = _render(node.right)
        if isinstance(node.op, ast.Pow):
            left = f"({left})" if lp <= prec else left
            return f"{left}<sup>{right}</sup>", prec
        left = f"({left})" if lp < prec else left
        right = f"({right})" if rp < prec or (rp == prec and isinstance(node.op, (ast.Sub, ast.Div))) else right
        return f"{left}{_OPS[type(node.op)]}{right}", prec
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        name = node.func.id
        if name == "sum" and len(node.args) == 1 and isinstance(node.args[0], ast.GeneratorExp):
            return _render_sum(node.args[0]), 2
        args = ", ".join(_render(a)[0] for a in node.args)
        return f"{_FUNCS.get(name, html.escape(name))}({args})", 9
    raise ValueError(f"cannot render {type(node).__name__}")


def _render_sum(gen: ast.GeneratorExp) -> str:
    (comp,) = gen.generators
    call = comp.iter
    if not (isinstance(comp.target, ast.Name) and isinstance(call, ast.Call) and getattr(call.func, "id", "") == "range"
            and not comp.ifs):
        raise ValueError("unsupported sum")
    bounds = [_render(a)[0] for a in call.args]
    low, high = ("0", bounds[0]) if len(bounds) == 1 else (bounds[0], bounds[1])
    term, _ = _render(gen.elt)
    return f"Σ<sub>{comp.target.id}={low}</sub><sup>{high}−1</sup> ({term})"


def expression_html(expression: Any) -> str:
    """An IR expression as HTML math (superscripts, ·, Σ); the escaped source text when it cannot be parsed.

    Only ``ast.parse`` reads the text; nothing is evaluated. A tool call is
    shown as ``tool_id(args) → output``.
    """
    if isinstance(expression, Mapping):
        args = ", ".join(f"{html.escape(str(k))}={expression_html(v)}"
                         for k, v in (expression.get("args") or {}).items())
        output = expression.get("output")
        return (f"<tt>{html.escape(str(expression.get('tool_id', '?')))}</tt>({args})"
                + (f" → {html.escape(str(output))}" if output else ""))
    source = str(expression)
    try:
        return _render(ast.parse(source, mode="eval").body)[0]
    except (SyntaxError, ValueError, RecursionError, TypeError, AttributeError):
        return f"<tt>{html.escape(source)}</tt>"


def _unit(unit: Any) -> str:
    return f" <span style='color:#7a7a7a'>[{html.escape(str(unit))}]</span>" if unit else ""


def _origin_tag(item: Mapping[str, Any], *, constraint: bool) -> str:
    origin = item.get("origin", "user")
    locked = item.get("locked", origin == "user") if constraint else origin == "user"
    text = origin + (" · locked" if constraint and locked else "")
    color = "#b3b3b3" if origin == "user" else "#f2c14e"
    return f" <span style='color:{color}; font-size:small'>{html.escape(text)}</span>"


def _bound_html(v: Mapping[str, Any]) -> str:
    name = f"<i>{html.escape(str(v.get('name', '?')))}</i>"
    kind = v.get("kind", "continuous")
    if kind == "binary":
        return f"{name} ∈ {{0, 1}}"
    if kind == "discrete":
        return f"{name} ∈ {{{', '.join(_safe_num(c) for c in v.get('choices') or [])}}}"
    lo, hi = v.get("lower"), v.get("upper")
    core = name
    if lo is not None and hi is not None:
        core = f"{_safe_num(lo)} ≤ {name} ≤ {_safe_num(hi)}"
    elif lo is not None:
        core = f"{name} ≥ {_safe_num(lo)}"
    elif hi is not None:
        core = f"{name} ≤ {_safe_num(hi)}"
    else:
        core = f"{name} free"
    return core + (" , integer" if kind == "integer" else "") + _unit(v.get("unit"))


def _safe_num(value: Any) -> str:
    if value is None:
        return "<b style='color:#ff4d3d'>?</b>"
    return _num(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else html.escape(str(value))


def problem_html(doc: Mapping[str, Any], *, problem_class: str | None = None,
                 reference: str | None = None) -> str:
    """The problem (IR JSON form, valid or not) as rich text: objectives, constraints, bounds, parameters, units."""
    rows: list[str] = []
    for o in doc.get("objectives") or []:
        sense = "maximize" if o.get("sense") == "maximize" else "minimize"
        rows.append(f"<b>{sense}</b> &nbsp;<i>{html.escape(str(o.get('name', '')))}</i> = "
                    f"{expression_html(o.get('expression', ''))}{_unit(o.get('unit'))}"
                    f"{_origin_tag(o, constraint=False)}")
    constraints = doc.get("constraints") or []
    if constraints:
        rows.append("<b>subject to</b>")
        for c in constraints:
            rows.append(f"&nbsp;&nbsp;<span style='color:#7a7a7a'>{html.escape(str(c.get('name', '')))}:</span> "
                        f"{expression_html(c.get('expression', ''))} {_SENSES.get(c.get('sense'), '?')} "
                        f"{_safe_num(c.get('rhs'))}{_unit(c.get('unit'))}{_origin_tag(c, constraint=True)}")
    variables = doc.get("variables") or []
    if variables:
        rows.append("<b>bounds</b> &nbsp;" + ", &nbsp;".join(_bound_html(v) for v in variables))
    params = doc.get("parameters") or []
    if params:
        rows.append("<b>where</b> &nbsp;" + ", &nbsp;".join(
            f"<i>{html.escape(str(p.get('name', '?')))}</i> = {_safe_num(p.get('value'))}{_unit(p.get('unit'))}"
            for p in params))
    if problem_class:
        rows.append(f"<span style='color:#7a7a7a'>class</span> {html.escape(problem_class)}")
    if reference:
        rows.append(f"<span style='color:#7a7a7a'>reference</span> {html.escape(reference)}")
    return "<br>".join(rows) if rows else "<span style='color:#7a7a7a'>No problem yet</span>"


def problem_text(doc: Mapping[str, Any]) -> str:
    """The plain-text fallback of :func:`problem_html` (and what the report export writes)."""
    lines = []
    for o in doc.get("objectives") or []:
        lines.append(f"{o.get('sense', 'minimize')} {o.get('name', '')} = {_plain(o.get('expression'))}")
    for c in doc.get("constraints") or []:
        lines.append(f"  {c.get('name', '')}: {_plain(c.get('expression'))} {c.get('sense')} {c.get('rhs')}")
    for v in doc.get("variables") or []:
        lines.append(f"  {v.get('name')}: {v.get('kind', 'continuous')} [{v.get('lower')}, {v.get('upper')}]")
    return "\n".join(lines)


def _plain(expression: Any) -> str:
    return json.dumps(expression, sort_keys=True) if isinstance(expression, Mapping) else str(expression)


# ------------------------------------------------------------------------------------------------- runs
@dataclass(frozen=True)
class IterationRow:
    """One row of the iteration table: a planner proposal and what came of it."""

    index: int
    kind: str                 # solve | stop | rejected | running
    solver: str
    evaluations: int
    cumulative: int
    objective: float | None
    feasibility: str          # feasible | infeasible | not_assessed | ""
    verdict: str
    diagnosis: str
    tokens: int
    note: str


@dataclass(frozen=True)
class CurvePoint:
    """One candidate on the convergence chart; ``kind`` is how the guard's screen graded its point."""

    evaluations: int
    objective: float
    kind: str                 # feasible | infeasible | not_assessed | alternate
    label: str


@dataclass(frozen=True)
class StatusLine:
    name: str
    outcome: str              # passed | failed | not_assessed | not run
    reason: str


@dataclass(frozen=True)
class PolicyViolationRow:
    iteration: int
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class Meter:
    label: str
    used: float
    limit: float
    unit: str = ""


@dataclass(frozen=True)
class RunView:
    """Everything the run panels draw, from a trace (finished or replayed) or a partial history (live)."""

    rows: tuple[IterationRow, ...]
    points: tuple[CurvePoint, ...]
    best_path: tuple[tuple[int, float], ...]
    best_label: str | None
    status: str | None
    stop_reason: str | None
    statuses: tuple[StatusLine, ...]
    report_source: str
    verdict: str | None
    codes: tuple[str, ...]
    violations: tuple[PolicyViolationRow, ...]
    meters: tuple[Meter, ...]
    best: Mapping[str, Any] | None
    objective_name: str
    sense: str
    fingerprint: str | None
    live: bool

    @property
    def verified(self) -> bool:
        return self.status == "verified"


def _screen_feasibility(screen: Mapping[str, Any] | None) -> str:
    if not screen:
        return ""
    outcome = screen.get("statuses", {}).get("constraints_satisfied", {}).get("outcome", "not_assessed")
    return {"passed": "feasible", "failed": "infeasible"}.get(outcome, "not_assessed")


def _first_objective(result: Mapping[str, Any] | None, name: str) -> float | None:
    if not result:
        return None
    values = result.get("objective_values") or {}
    value = values.get(name, next(iter(values.values()), None)) if values else None
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _better(a: float, b: float, sense: str) -> bool:
    return a > b if sense == "maximize" else a < b


def run_view(iterations: Sequence[Mapping[str, Any]], *, limits: Mapping[str, Any],
             objective_name: str, sense: str = "minimize", verifications: Sequence[Mapping[str, Any]] = (),
             outcome: Mapping[str, Any] | None = None, usage: Mapping[str, Any] | None = None,
             fingerprint: str | None = None, extra_tokens: int = 0, running: str | None = None) -> RunView:
    """Build a :class:`RunView` from trace parts.

    ``iterations`` and ``verifications`` are the trace's records (or the
    records so far, live). ``outcome`` and ``usage`` are the trace's; without
    an ``outcome`` the view is live, ``running`` names the step in flight,
    and the incumbent (the best point the screen found feasible) is the best.
    ``extra_tokens`` are tokens spent outside the run (the compile), counted
    against the same token limit.
    """
    rows: list[IterationRow] = []
    points: list[CurvePoint] = []
    best_path: list[tuple[int, float]] = []
    violations: list[PolicyViolationRow] = []
    cumulative, tokens, best, incumbent = 0, 0, None, None
    for record in iterations:
        cumulative += int(record.get("evaluations", 0))
        tokens += int(record.get("tokens", 0))
        result, screen = record.get("result"), record.get("screen")
        decision = record.get("decision") or {}
        raw = (record.get("proposal") or {}).get("raw") or {}
        value = _first_objective(result, objective_name)
        feasibility = _screen_feasibility(screen)
        for v in record.get("violations") or []:
            violations.append(PolicyViolationRow(int(record["index"]), v["code"], v.get("path", ""), v["message"]))
        note = "; ".join(f"{v['code']}: {v['message']}" for v in record.get("violations") or [])
        if not note:
            note = (record.get("diagnosis") or {}).get("detail") or decision.get("rationale", "")
        rows.append(IterationRow(int(record["index"]), record["kind"],
                                 str(decision.get("solver_id") or raw.get("solver_id") or ""),
                                 int(record.get("evaluations", 0)), cumulative, value, feasibility,
                                 (screen or {}).get("verdict", ""), (record.get("diagnosis") or {}).get("kind", ""),
                                 int(record.get("tokens", 0)), str(note)))
        if value is not None:
            points.append(CurvePoint(cumulative, value, feasibility or "not_assessed",
                                     f"iteration-{int(record['index']):04d}"))
            if feasibility == "feasible" and (best is None or _better(value, best, sense)):
                best, incumbent = value, f"iteration-{int(record['index']):04d}"
        if best is not None and (not best_path or best_path[-1] != (cumulative, best)):
            best_path.append((cumulative, best))
    for ver in verifications:
        cumulative += int(ver.get("evaluations", 0))
        for alt in ver.get("alternates") or []:
            value = _first_objective(alt.get("result"), objective_name)
            if value is not None:
                points.append(CurvePoint(cumulative, value, "alternate",
                                         f"verify-{ver.get('attempt')}-{alt.get('solver_id')}"))
    if best_path and verifications:
        best_path.append((cumulative, best_path[-1][1]))
    if running is not None:
        rows.append(IterationRow(len(rows) + 1, "running", running, 0, cumulative, None, "", "", "", 0, "in progress"))

    report, source = None, "no candidate graded yet"
    if verifications:
        last = verifications[-1]
        report, source = last.get("report"), f"final verification, attempt {last.get('attempt')}"
    best_label = incumbent if outcome is None else outcome.get("best")
    if report is None and best_label and best_label.startswith("iteration-"):
        index = int(best_label.split("-")[1])
        report = next((r.get("screen") for r in iterations if int(r["index"]) == index), None)
        source = (f"screen of the incumbent {best_label}; the final verification runs when the loop ends"
                  if outcome is None else f"screen of {best_label} only: no final verification ran")
    statuses = tuple(
        StatusLine(name, (report or {}).get("statuses", {}).get(name, {}).get("outcome", "not run"),
                   (report or {}).get("statuses", {}).get(name, {}).get("reason", "no report graded this status"))
        for name in STATUS_NAMES)

    used = usage or {"iterations": len(iterations), "evaluations": cumulative, "tokens": tokens}
    meters = (
        Meter("Iterations", float(used.get("iterations", len(iterations))), float(limits.get("max_iterations", 0))),
        Meter("Evaluations", float(used.get("evaluations", cumulative)), float(limits.get("max_evaluations", 0))),
        Meter("Tokens", float(used.get("tokens", tokens)) + extra_tokens, float(limits.get("max_tokens", 0))),
    )
    best_result = None
    if best_label:
        if best_label.startswith("iteration-"):
            index = int(best_label.split("-")[1])
            best_result = next((r.get("result") for r in iterations if int(r["index"]) == index), None)
        else:
            for ver in verifications:
                for alt in ver.get("alternates") or []:
                    if best_label == f"verify-{ver.get('attempt')}-{alt.get('solver_id')}":
                        best_result = alt.get("result")
    return RunView(tuple(rows), tuple(points), tuple(best_path), best_label, (outcome or {}).get("status"),
                   (outcome or {}).get("stop_reason"), statuses, source, (report or {}).get("verdict"),
                   tuple((report or {}).get("codes", ())), tuple(violations), meters, best_result, objective_name,
                   sense, fingerprint, outcome is None)


def run_view_from_trace(trace: Mapping[str, Any], *, objective_name: str, sense: str,
                        extra_tokens: int = 0) -> RunView:
    """The view of a finished or replayed run, from its ``trace.json`` dict."""
    return run_view(trace["iterations"], limits=trace["limits"], objective_name=objective_name, sense=sense,
                    verifications=trace.get("verifications", ()), outcome=trace["outcome"], usage=trace["usage"],
                    fingerprint=trace.get("fingerprint"), extra_tokens=extra_tokens)


def view_at_step(trace: Mapping[str, Any], step: int, *, objective_name: str, sense: str) -> RunView:
    """The replayed run as it stood after ``step`` iterations (the final view at the last step)."""
    iterations = trace["iterations"]
    if step >= len(iterations):
        return run_view_from_trace(trace, objective_name=objective_name, sense=sense)
    return run_view(iterations[:max(step, 0)], limits=trace["limits"], objective_name=objective_name, sense=sense,
                    fingerprint=trace.get("fingerprint"))


# ----------------------------------------------------------------------------------------------- export
def report_document(view: RunView, *, problem: Mapping[str, Any], meta: Mapping[str, Any]) -> dict:
    """The verification report as a JSON-ready dict: run status, the four statuses, violations, budget use."""
    return {
        "format": "orbitopt.studio.report/1",
        **dict(meta),
        "problem_id": problem.get("id"),
        "problem": problem_text(problem),
        "run_status": view.status,
        "stop_reason": view.stop_reason,
        "verified": view.verified,
        "report_source": view.report_source,
        "verdict": view.verdict,
        "statuses": {s.name: {"outcome": s.outcome, "reason": s.reason} for s in view.statuses},
        "finding_codes": list(view.codes),
        "best": None if view.best is None else {k: view.best.get(k) for k in
                                                ("solver_id", "x", "objective_values", "termination")},
        "policy_violations": [asdict(v) for v in view.violations],
        "budget": [asdict(m) for m in view.meters],
        "iterations": [asdict(r) for r in view.rows],
    }


def report_html(doc: Mapping[str, Any]) -> str:
    """A readable, self-contained HTML page of :func:`report_document`'s output."""
    def esc(value: Any) -> str:
        return html.escape("—" if value is None else str(value))

    status_rows = "".join(
        f"<tr><td>{esc(STATUS_LABELS[name])}</td><td class='{esc(s['outcome'])}'>{esc(s['outcome'])}</td>"
        f"<td>{esc(s['reason'])}</td></tr>" for name, s in doc["statuses"].items())
    violation_rows = "".join(f"<tr><td>{v['iteration']}</td><td>{esc(v['code'])}</td><td>{esc(v['path'])}</td>"
                             f"<td>{esc(v['message'])}</td></tr>" for v in doc["policy_violations"]) \
        or "<tr><td colspan='4'>none recorded</td></tr>"
    iteration_rows = "".join(
        f"<tr><td>{r['index']}</td><td>{esc(r['kind'])}</td><td>{esc(r['solver'])}</td><td>{r['evaluations']}</td>"
        f"<td>{esc(r['objective'])}</td><td>{esc(r['feasibility'])}</td><td>{esc(r['note'])}</td></tr>"
        for r in doc["iterations"])
    meta = "".join(f"<tr><th>{esc(k)}</th><td>{esc(doc.get(k))}</td></tr>"
                   for k in ("problem_id", "problem_digest", "manifest_digest", "run_dir", "replay", "planner",
                             "provider", "seed"))
    budget = "".join(f"<tr><td>{esc(m['label'])}</td><td>{m['used']:g} / {m['limit']:g}</td></tr>"
                     for m in doc["budget"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Verification report {esc(doc.get('problem_id'))}</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", sans-serif; margin: 2rem; color: #111; background: #fff; }}
table {{ border-collapse: collapse; margin: 0.5rem 0 1.5rem; }}
td, th {{ border: 1px solid #ccc; padding: 4px 8px; text-align: left; vertical-align: top; }}
.passed {{ color: #0a7a4b; }} .failed {{ color: #b3261e; }} .not_assessed, .not\\ run {{ color: #666; }}
pre {{ background: #f4f4f4; padding: 0.75rem; }}
</style></head><body>
<h1>Verification report</h1>
<p><b>Run status:</b> {esc(doc['run_status'])} ({esc(RUN_STATUS_TEXT.get(doc['run_status'] or '', ''))})</p>
<p>Each status below is graded separately by the verification guard. Only a <code>verified</code> run
status means the guard passed the final answer.</p>
<table>{meta}</table>
<h2>Problem</h2><pre>{esc(doc['problem'])}</pre>
<h2>Statuses ({esc(doc['report_source'])})</h2>
<table><tr><th>Status</th><th>Outcome</th><th>Reason</th></tr>{status_rows}</table>
<p>Verdict: {esc(doc['verdict'])}. Finding codes: {esc(', '.join(doc['finding_codes']) or 'none')}.</p>
<h2>Best point</h2><pre>{esc(json.dumps(doc['best'], indent=2, sort_keys=True))}</pre>
<h2>Policy violations</h2>
<table><tr><th>Iteration</th><th>Code</th><th>Path</th><th>Message</th></tr>{violation_rows}</table>
<h2>Budget</h2><table>{budget}</table>
<h2>Iterations</h2>
<table><tr><th>#</th><th>Kind</th><th>Solver</th><th>Evaluations</th><th>Objective</th><th>Feasibility</th>
<th>Note</th></tr>{iteration_rows}</table>
</body></html>
"""
