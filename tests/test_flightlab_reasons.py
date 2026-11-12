"""FL-503: every reason code emitted under ``orbitopt/flightlab`` has a
plain-language catalog entry (EN + 中文 + suggested fix).

The scan is static (AST) so it covers code paths no test flies -- the
optimizer's INFEASIBLE_* codes, the re-fly backend errors, the recovery
chain once it lands. Two sources are collected from every module:

* every string literal that looks like a reason code (UPPER_SNAKE_CASE),
  wherever it appears -- module constants, dict values, ``Check(...)``
  arguments -- except ``__all__`` entries and the explicit non-code
  allow-list below (env var names, frame ids);
* every string literal passed to a ``reason_code=`` / ``fail_code=`` /
  ``*_code=`` keyword, in any case (the adapter emits ``max_q_exceeded``).

A reason code built at runtime (f-string, concatenation) in a code keyword
is refused outright: it would be invisible to this test.
"""
from __future__ import annotations

import ast
import functools
import re
from pathlib import Path

import pytest

from orbitopt.flightlab import reasons
from orbitopt.flightlab.reasons import CATALOG, explain, explain_all, lookup, normalize_code

FLIGHTLAB = Path(reasons.__file__).resolve().parent
CODE_RE = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")

#: UPPER_SNAKE literals under flightlab/ that are *not* reason codes.
NOT_REASON_CODES = frozenset({
    "GIT_SHA",              # env var (provenance)
    "XDG_CACHE_HOME",       # env var (designer cache root)
    "LAUNCH_ECI",           # frame id
    "ECEF",
})
NOT_REASON_PREFIXES = ("ORBITOPT_",)  # env vars

#: Codes the task names explicitly (FL-503 scope).
REQUIRED_CODES = ("PERIAPSIS_IN_ATMOSPHERE", "MAX_Q_EXCEEDED",
                  "INSUFFICIENT_PROPELLANT_BOOSTBACK", "LANDING_INFEASIBLE")


def _is_code_keyword(name: str | None) -> bool:
    return bool(name) and (name == "reason_code" or name.endswith("_code") or name == "code")


def _all_names(tree: ast.AST) -> set[int]:
    """ids of string nodes inside ``__all__ = [...]`` assignments."""
    skip: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets) and node.value:
                skip.update(id(n) for n in ast.walk(node.value))
    return skip


def _scan(path: Path) -> tuple[set[str], list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    skip = _all_names(tree)
    codes: set[str] = set()
    dynamic: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            if CODE_RE.match(node.value) and node.value not in NOT_REASON_CODES \
                    and not node.value.startswith(NOT_REASON_PREFIXES):
                codes.add(node.value)
        if isinstance(node, ast.keyword) and _is_code_keyword(node.arg):
            _keyword_literals(node.value, codes, dynamic, f"{path.relative_to(FLIGHTLAB)}")
    return codes, dynamic


def _keyword_literals(expr: ast.AST, codes: set[str], dynamic: list[str], where: str) -> None:
    """String literals a code keyword can evaluate to: constants and the
    branches of ``x if c else y`` / ``a or b`` (not, e.g., the key of a
    ``doc.get("reason_code")`` call)."""
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        if expr.value:
            codes.add(expr.value)
    elif isinstance(expr, ast.IfExp):
        _keyword_literals(expr.body, codes, dynamic, where)
        _keyword_literals(expr.orelse, codes, dynamic, where)
    elif isinstance(expr, ast.BoolOp):
        for value in expr.values:
            _keyword_literals(value, codes, dynamic, where)
    elif isinstance(expr, (ast.JoinedStr, ast.BinOp)):
        dynamic.append(f"{where}:{expr.lineno}")


@functools.lru_cache(maxsize=1)
def _emitted_codes() -> tuple[dict[str, list[str]], list[str]]:
    where: dict[str, list[str]] = {}
    dynamic: list[str] = []
    for path in sorted(FLIGHTLAB.rglob("*.py")):
        if path.name == "reasons.py":
            continue
        codes, dyn = _scan(path)
        dynamic += dyn
        for code in codes:
            where.setdefault(code, []).append(str(path.relative_to(FLIGHTLAB)))
    return where, dynamic


def test_scan_finds_the_known_emitters():
    """Guard the scanner itself: it must see codes from each emitter."""
    where, _ = _emitted_codes()
    for code in ("PERIAPSIS_IN_ATMOSPHERE", "INSERTION_APOAPSIS_OUT_OF_TOLERANCE",
                 "max_q_exceeded", "negative_mass", "REFLY_BACKEND_MISSING",
                 "INFEASIBLE_SIMULATION", "INFEASIBLE_RAAN_OUT_OF_TOLERANCE"):
        assert code in where, code


def test_every_reason_code_used_in_flightlab_is_in_the_catalog():
    where, _ = _emitted_codes()
    missing = {code: files for code, files in sorted(where.items()) if lookup(code) is None}
    assert not missing, (
        "reason codes without a catalog entry in orbitopt/flightlab/reasons.py "
        f"(add one, or allow-list a non-code identifier in this test): {missing}")


def test_no_runtime_built_reason_codes():
    _, dynamic = _emitted_codes()
    assert not dynamic, f"f-string reason codes are invisible to the catalog test: {dynamic}"


def test_required_codes_and_entry_quality():
    for code in REQUIRED_CODES:
        assert code in CATALOG, code
    for code, entry in CATALOG.items():
        assert entry.code == code and CODE_RE.match(code) or code == "LANDED", code
        for text in (entry.title_en, entry.title_zh, entry.explanation_en, entry.explanation_zh,
                     entry.fix_en, entry.fix_zh):
            assert text.strip(), code
        assert re.search(r"[一-鿿]", entry.title_zh + entry.explanation_zh), code
        assert entry.severity in ("fail", "missing", "info"), code


def test_lookup_is_case_insensitive_and_prefix_aware():
    assert normalize_code("max_q_exceeded") == "MAX_Q_EXCEEDED"
    assert lookup("max_q_exceeded") is CATALOG["MAX_Q_EXCEEDED"]
    assert lookup("SAMPLE_ERROR:ValueError") is CATALOG["SAMPLE_ERROR"]
    assert lookup("") is None
    unknown = explain("synthetic_fixture_below_target_by_design")
    assert not unknown.known and "synthetic_fixture" in unknown.title_en
    assert "Fix" in CATALOG["PERIAPSIS_IN_ATMOSPHERE"].text("en")
    assert "建议" in CATALOG["PERIAPSIS_IN_ATMOSPHERE"].text("zh")
    deduped = explain_all(["max_q_exceeded", "MAX_Q_EXCEEDED", "", "LANDING_INFEASIBLE"])
    assert [e.code for e in deduped] == ["MAX_Q_EXCEEDED", "LANDING_INFEASIBLE"]


@pytest.mark.slow
def test_flown_gate_b_run_failures_are_all_explained(tmp_path, monkeypatch):
    """Runtime cross-check on a real (stub-flown) Gate-B run."""
    from flightlab_compare_fixture import fly_stub_run
    from orbitopt.flightlab.contract import read_run

    monkeypatch.setenv("ORBITOPT_CACHE_DIR", str(tmp_path / "cache"))
    run = read_run(fly_stub_run(tmp_path / "run", tmp_path, monkeypatch))
    failed = [c.reason_code for c in run.verification.checks if not c.passed]
    assert failed
    assert all(lookup(code) is not None for code in failed), failed


@pytest.mark.parametrize("code", sorted(CATALOG))
def test_catalog_bilingual_text_renders(code):
    entry = CATALOG[code]
    block = entry.bilingual()
    assert entry.title_en in block and entry.title_zh in block
