"""Check that the generated reference tracks the published interfaces."""

import argparse
import ast
import re
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = runpy.run_path(str(ROOT / "scripts" / "generate_api_reference.py"))


def test_public_modules_and_definitions_have_reference_pages():
    pages = {page["url"]: page["content"] for page in GENERATOR["pages"]()}
    for module, path in GENERATOR["public_modules"]():
        assert f"[`{module}`]({{BASE}}{GENERATOR['module_url'](module)})" in pages["/api/python/"]
        content = pages[GENERATOR["module_url"](module)]
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) or node.name.startswith("_"):
                continue
            assert f"## `{node.name}`" in content, (module, node.name)
            if isinstance(node, ast.ClassDef):
                for method in node.body:
                    if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) and not method.name.startswith("_"):
                        assert f"### `{node.name}.{method.name}`" in content, (module, node.name, method.name)


def test_reference_includes_signatures_fields_and_reexports():
    pages = {page["url"]: page["content"] for page in GENERATOR["pages"]()}
    lambert = pages["/api/python/orbitopt.lambert.gpu_batch/"]
    assert "def solve_lambert_batch(r1_vec, r2_vec, tof, mu," in lambert
    assert "Relative TOF tolerance" in lambert
    assert "Returns" in lambert
    contract = pages["/api/python/orbitopt.flightlab.contract/"]
    assert "## `FlightTarget`" in contract
    assert "| `mission` | <code>Mission</code> | <code>Required / unset</code> |" in contract
    assert "|---|---|---|\n| `mission`" in contract
    entry = pages["/api/python/orbitopt.entry/"]
    assert "@property\ndef ballistic_coefficient(self) -> float" in entry
    state = pages["/api/python/orbitopt.viz.state/"]
    assert "| `scene_changed` | <code>Signal(object)</code> |" in state
    flightlab = pages["/api/python/orbitopt.flightlab/"]
    assert "`simulate`" in flightlab
    assert "/api/python/orbitopt.flightlab.simulate/" in flightlab
    assert "`build_target`" in flightlab


def test_cli_reference_matches_argparse_commands_and_options():
    from orbitopt.cli import build_parser

    actual = {}

    def walk(parser, name):
        actual[name] = []
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for child_name, child in action.choices.items():
                    walk(child, f"{name} {child_name}")
            elif action.dest != "help":
                actual[name].append(tuple(action.option_strings or [action.dest]))

    walk(build_parser(), "orbitopt")
    generated = GENERATOR["cli_commands"]()
    assert set(generated) == set(actual)
    for command, options in actual.items():
        assert [tuple(arg["names"]) for arg in generated[command]["arguments"]] == options


def test_schema_reference_includes_fields_constraints_and_definitions():
    pages = {page["url"]: page["content"] for page in GENERATOR["pages"]()}
    for path in GENERATOR["schema_files"]():
        assert f"/api/schemas/{path.stem}/" in pages
    scene = pages["/api/schemas/scene-1.0/"]
    assert "`$.schemaVersion` | Yes" in scene
    assert "array of reference: #/$defs/body" in scene
    assert "## Definition: vec3" in scene
    assert "minItems: 3" in scene


def test_generated_reference_is_english_and_internal_links_resolve():
    pages = GENERATOR["pages"]()
    urls = {page["url"] for page in pages}
    assert len(urls) == len(pages)
    for page in pages:
        assert not re.search(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", page["content"]), page["url"]
        for target in re.findall(r"\{BASE\}(/api/[^)\s]+)", page["content"]):
            assert target in urls, (page["url"], target)
