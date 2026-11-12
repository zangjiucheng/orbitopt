#!/usr/bin/env python3
"""Build Jekyll API pages from Python source and packaged JSON Schemas.

The generator parses source instead of importing orbitopt, so optional compute
and GUI dependencies are not needed to build the documentation.  Its JSON
output is consumed by docs/_plugins/api_reference.rb.
"""

from __future__ import annotations

import ast
import html
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "orbitopt"
REPO = "https://github.com/zangjiucheng/orbitopt/blob/main/"
GROUPS = {
    "root": "Core interfaces",
    "core": "Compute backend",
    "dynamics": "Dynamics",
    "flightlab": "Flight Lab",
    "lambert": "Lambert solvers",
    "optimize": "Optimization",
    "problems": "Optimization problems",
    "verify": "Verification",
    "viz": "Visualization and Mission Control",
}


def public_modules():
    """Yield module names and paths, excluding private files and packages."""
    for path in sorted(PACKAGE.rglob("*.py")):
        parts = path.relative_to(PACKAGE).parts
        if any(part.startswith("_") and part != "__init__.py" for part in parts):
            continue
        stem = path.relative_to(PACKAGE).with_suffix("").parts
        suffix = stem[:-1] if stem[-1] == "__init__" else stem
        yield ".".join(("orbitopt", *suffix)), path


def module_url(name):
    return f"/api/python/{name}/"


def source_link(path, line=None):
    url = REPO + path.relative_to(ROOT).as_posix()
    return url + (f"#L{line}" if line else "")


def clean_doc(text):
    # Translate the few bilingual phrases in upstream docstrings; runtime UI
    # strings and the package source are not altered by the docs build.
    replacements = {
        "EN + 中文": "English and Chinese",
        "支持 Escape 关闭": "Escape closes the popup",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r":(?:class|func|mod|meth|attr):`([^`]+)`", r"`\1`", text)


def description(node):
    """Render the summary and retain remaining source notes on demand."""
    doc = ast.get_docstring(node)
    if not doc:
        if isinstance(node, ast.ClassDef) and any(
            (ast.unparse(decorator).split("(", 1)[0].split(".")[-1] == "dataclass")
            for decorator in node.decorator_list
        ):
            return "Data class. Its declared fields and defaults are listed below; the source has no class docstring.\n"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and len(node.body) == 1:
            statement = node.body[0]
            if isinstance(statement, ast.Return) and statement.value is not None:
                value = ast.unparse(statement.value)
                if len(value) <= 200:
                    return f"Returns <code>{html.escape(value)}</code>. The source has no docstring for this member.\n"
        return "*No source docstring is available; see the signature and source.*\n"
    doc = clean_doc(doc).replace("``", "`")
    summary, _, rest = doc.partition("\n\n")
    summary = html.escape(" ".join(summary.split()))
    rendered = summary + "\n"
    if rest.strip():
        rendered += (
            '\n<details><summary>Parameters and notes</summary>\n'
            f'<pre>{html.escape(rest.strip())}</pre>\n</details>\n'
        )
    return rendered


def signature(node):
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args = ast.unparse(node.args)
    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    decorators = "".join(f"@{ast.unparse(decorator)}\n" for decorator in node.decorator_list)
    return f"{decorators}{prefix} {node.name}({args}){ret}"


def fields(node):
    """Return declared public annotated class fields and defaults."""
    result = []
    for item in node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            if not item.target.id.startswith("_"):
                default = ast.unparse(item.value) if item.value is not None else None
                result.append((item.target.id, ast.unparse(item.annotation), default))
    return result


def class_attributes(node):
    """Return public class assignments such as constants and Qt signals."""
    result = []
    for item in node.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    result.append((target.id, ast.unparse(item.value)))
    return result


def public_definitions(tree):
    return [
        item for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and not item.name.startswith("_")
    ]


def static_value(node, scope):
    """Evaluate only simple literal lists used to assemble __all__."""
    if isinstance(node, ast.Name):
        return scope.get(node.id)
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [static_value(item, scope) for item in node.elts]
        return values if all(isinstance(item, str) for item in values) else None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = static_value(node.left, scope), static_value(node.right, scope)
        if isinstance(left, list) and isinstance(right, list):
            return left + right
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "list" and len(node.args) == 1):
        return static_value(node.args[0], scope)
    return None


def exports(tree, module, path):
    """Find explicitly exported aliases, including simple lazy __all__ lists."""
    scope = {}
    origins = {}
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    for item in tree.body:
        if isinstance(item, ast.ImportFrom):
            target = ("." * item.level) + (item.module or "")
            try:
                target = importlib.util.resolve_name(target, package) if item.level else target
            except ImportError:
                continue
            for alias in item.names:
                origins[alias.asname or alias.name] = f"{target}.{alias.name}"
        elif isinstance(item, (ast.Assign, ast.AnnAssign)):
            targets = item.targets if isinstance(item, ast.Assign) else [item.target]
            value = static_value(item.value, scope) if item.value else None
            for target in targets:
                if isinstance(target, ast.Name) and value is not None:
                    scope[target.id] = value
        elif (isinstance(item, ast.AugAssign) and isinstance(item.target, ast.Name)
              and isinstance(item.op, ast.Add)):
            value = static_value(item.value, scope)
            current = scope.get(item.target.id)
            if isinstance(current, list) and isinstance(value, list):
                scope[item.target.id] = current + value
    # These exports are resolved by module __getattr__ rather than an import.
    if module == "orbitopt":
        origins.update({name: f"orbitopt.core.gpu.{name}" for name in scope.get("__all__", [])})
    if module == "orbitopt.flightlab":
        for name in scope.get("_LAZY", []):
            origins[name] = f"orbitopt.flightlab.target_export.{name}"
        for name in scope.get("_SIM_LAZY", []):
            origins[name] = f"orbitopt.flightlab.simulate.{name}"
    return [(name, origins.get(name)) for name in scope.get("__all__", [])]


def module_page(module, path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    page = [f"# `{module}`\n", f"[Source]({source_link(path)}) · [Python API index]({{BASE}}/api/python/)\n"]
    if ast.get_docstring(tree):
        page.append(description(tree))
    definitions = public_definitions(tree)
    names = {item.name for item in definitions}
    aliases = [(name, origin) for name, origin in exports(tree, module, path) if name not in names]
    if aliases:
        page.append("## Re-exported names\n")
        for name, origin in aliases:
            if origin and origin.startswith("orbitopt."):
                owner = origin.rpartition(".")[0]
                page.append(f"- `{name}` — [{owner}]({{BASE}}{module_url(owner)})\n")
            else:
                page.append(f"- `{name}` — see the [source]({source_link(path)}).\n")
    constants = []
    for item in tree.body:
        if isinstance(item, ast.Assign):
            targets = item.targets
        elif isinstance(item, ast.AnnAssign):
            targets = [item.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id.isupper() and not target.id.startswith("_"):
                constants.append((target.id, item.lineno))
    if constants:
        page.append("## Constants\n")
        page.extend(f"- [`{name}`]({source_link(path, line)})\n" for name, line in constants)
    for item in definitions:
        page.append(f"\n## `{item.name}`\n{{: #{item.name.lower()} }}\n")
        page.append(f"[Source]({source_link(path, item.lineno)})\n")
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            page.append(f"```python\n{signature(item)}\n```\n")
            page.append(description(item))
            continue
        bases = ", ".join(ast.unparse(base) for base in item.bases)
        page.append(f"```python\nclass {item.name}{f'({bases})' if bases else ''}\n```\n")
        page.append(description(item))
        init = next((x for x in item.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name == "__init__"), None)
        if init:
            page.append(f"**Constructor**\n\n```python\n{signature(init)}\n```\n")
            if ast.get_docstring(init):
                page.append(description(init))
        declared = fields(item)
        if declared:
            field_table = ["**Declared fields**\n", "| Field | Type | Default |", "|---|---|---|"]
            for name, annotation, default in declared:
                annotation = html.escape(annotation).replace("|", "&#124;")
                default = html.escape(default).replace("|", "&#124;") if default is not None else "Required / unset"
                field_table.append(f"| `{name}` | <code>{annotation}</code> | <code>{default}</code> |")
            page.append("\n".join(field_table) + "\n")
        attributes = class_attributes(item)
        if attributes:
            attribute_table = ["**Class attributes and signals**\n", "| Name | Declared value |", "|---|---|"]
            for name, value in attributes:
                attribute_table.append(f"| `{name}` | <code>{html.escape(value).replace('|', '&#124;')}</code> |")
            page.append("\n".join(attribute_table) + "\n")
        for method in item.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) or method.name.startswith("_"):
                continue
            page.append(f"\n### `{item.name}.{method.name}`\n{{: #{item.name.lower()}-{method.name.lower()} }}\n")
            page.append(f"[Source]({source_link(path, method.lineno)})\n")
            page.append(f"```python\n{signature(method)}\n```\n")
            page.append(description(method))
    return "\n".join(page)


def schema_files():
    yield from sorted((PACKAGE / "schemas").glob("*.json"))
    yield from sorted((PACKAGE / "data" / "flightlab" / "schemas").glob("*.json"))


def schema_type(spec):
    if "$ref" in spec:
        return f"reference: {spec['$ref']}"
    kind = spec.get("type", "any")
    if isinstance(kind, list):
        kind = " or ".join(kind)
    if kind == "array" and isinstance(spec.get("items"), dict):
        kind += " of " + schema_type(spec["items"])
    constraints = []
    for key in ("const", "enum", "default", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minItems", "maxItems", "minLength", "maxLength", "pattern"):
        if key in spec:
            constraints.append(f"{key}: {json.dumps(spec[key], ensure_ascii=False)}")
    for key in ("oneOf", "anyOf", "allOf"):
        if key in spec:
            constraints.append(f"{key}: {len(spec[key])} alternatives; see schema")
    if "additionalProperties" in spec:
        constraints.append(f"additionalProperties: {json.dumps(spec['additionalProperties'])}")
    return "; ".join([kind, *constraints])


def schema_rows(spec, prefix="$", required=None):
    """Walk inline object/array fields; references and definitions stay explicit."""
    if not isinstance(spec, dict):
        return []
    rows = []
    required = set(spec.get("required", []))
    for name, child in spec.get("properties", {}).items():
        path = f"{prefix}.{name}"
        rows.append((path, name in required, child))
        rows.extend(schema_rows(child, path))
        if isinstance(child.get("items"), dict):
            rows.extend(schema_rows(child["items"], path + "[]"))
    return rows


def schema_page(path):
    spec = json.loads(path.read_text(encoding="utf-8"))
    page = [f"# {html.escape(spec.get('title', path.name))}\n", f"[Schema source]({source_link(path)}) · [Schema index]({{BASE}}/api/schemas/)\n"]
    if spec.get("description"):
        page.append(html.escape(spec["description"]) + "\n")
    page.append("Required applies within the containing object. References and alternatives are defined by the source schema.\n")
    for heading, section in [("Root fields", spec), *[(f"Definition: {name}", value) for name, value in spec.get("$defs", {}).items()]]:
        rows = schema_rows(section, "$" if heading == "Root fields" else heading.split(": ", 1)[1])
        page.append(f"\n## {heading}\n\n**Type:** {html.escape(schema_type(section))}\n\n")
        if heading != "Root fields" and section.get("description"):
            page.append(html.escape(section["description"]) + "\n\n")
        if not rows:
            continue
        page.append("| Field | Required | Type and constraints | Description |\n|---|---|---|---|\n")
        for name, is_required, child in rows:
            typ = html.escape(schema_type(child)).replace("|", "&#124;")
            desc = html.escape(child.get("description", "")).replace("|", "&#124;").replace("\n", " ")
            page.append(f"| `{name}` | {'Yes' if is_required else 'No'} | {typ} | {desc or 'See schema.'} |\n")
    return "".join(page)


def literal_or_source(node):
    if isinstance(node, ast.JoinedStr):
        return "".join(part.value if isinstance(part, ast.Constant) else "<mission IDs>" for part in node.values)
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return ast.unparse(node)


def cli_commands():
    """Read argparse declarations without importing optional dependencies."""
    tree = ast.parse((PACKAGE / "cli.py").read_text(encoding="utf-8"))
    builder = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "build_parser")
    commands = {"orbitopt": {"help": "Launch Mission Control by default.", "arguments": []}}
    owners = {"parser": "orbitopt"}
    subparsers = {}
    for statement in builder.body:
        call = statement.value if isinstance(statement, (ast.Assign, ast.Expr)) else None
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            continue
        if not isinstance(call.func.value, ast.Name):
            continue
        receiver, method = call.func.value.id, call.func.attr
        target = statement.targets[0].id if isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name) else None
        kwargs = {kw.arg: literal_or_source(kw.value) for kw in call.keywords if kw.arg}
        if method == "add_subparsers" and target and receiver in owners:
            subparsers[target] = owners[receiver]
        elif method == "add_mutually_exclusive_group" and target and receiver in owners:
            owners[target] = owners[receiver]
        elif method == "add_parser" and receiver in subparsers and call.args:
            name = literal_or_source(call.args[0])
            command = f"{subparsers[receiver]} {name}"
            commands[command] = {"help": kwargs.get("help", ""), "arguments": []}
            if target:
                owners[target] = command
        elif method == "add_argument" and receiver in owners:
            names = [literal_or_source(arg) for arg in call.args]
            commands[owners[receiver]]["arguments"].append({"names": names, **kwargs})
    return commands


def cli_pages():
    commands = cli_commands()
    index = ["# Command-line reference\n\nThese commands and options are extracted from the argument parser. Run `orbitopt <command> --help` for the installed version.\n"]
    result = []
    for command, data in commands.items():
        suffix = command.removeprefix("orbitopt").strip().replace(" ", "/")
        url = f"/api/cli/{suffix}/" if suffix else "/api/cli/orbitopt/"
        index.append(f"- [`{command}`]({{BASE}}{url}) — {html.escape(str(data['help']))}\n")
        content = [f"# `{command}`\n\n{html.escape(str(data['help']))}\n\n[Command index]({{BASE}}/api/cli/)\n"]
        if data["arguments"]:
            content.append("\n| Argument | Requirement / default | Description |\n|---|---|---|\n")
        for arg in data["arguments"]:
            names = ", ".join(f"`{html.escape(str(name))}`" for name in arg["names"])
            flags = []
            if arg.get("required"):
                flags.append("required")
            elif arg["names"] and not str(arg["names"][0]).startswith("-"):
                flags.append("optional" if arg.get("nargs") in ("?", "*") else "required")
            if "default" in arg and arg["default"] is not None:
                flags.append("default: " + str(arg["default"]))
            if "choices" in arg:
                flags.append("choices: " + ", ".join(map(str, arg["choices"])))
            if "type" in arg:
                flags.append("type: " + str(arg["type"]))
            if "metavar" in arg:
                metavar = arg["metavar"]
                flags.append("value: " + (" ".join(map(str, metavar)) if isinstance(metavar, (list, tuple)) else str(metavar)))
            if arg.get("nargs"):
                flags.append("nargs: " + str(arg["nargs"]))
            detail = html.escape("; ".join(flags) or "optional").replace("|", "&#124;")
            help_text = html.escape(str(arg.get("help", ""))).replace("|", "&#124;").replace("\n", " ")
            content.append(f"| {names} | {detail} | {help_text or 'See command help.'} |\n")
        if command == "orbitopt flight export-target":
            content.append("\nExactly one of `--date` and `--window` is required.\n")
        result.append({"url": url, "title": command, "content": "".join(content)})
    result.append({"url": "/api/cli/", "title": "Command-line reference", "content": "\n".join(index)})
    return result


def pages():
    modules = list(public_modules())
    result = []
    index = ["# Python API\n\nPublic names are grouped by module. Signatures and descriptions come from the current source; missing docstrings are identified explicitly. Private names are omitted. Re-exports link to their defining module.\n"]
    for group, title in GROUPS.items():
        matching = [(name, path) for name, path in modules if (name.split(".")[1] if "." in name else "root") == group or (group == "root" and len(name.split(".")) == 2 and name.split(".")[1] not in GROUPS)]
        if not matching:
            continue
        index.append(f"\n## {title}\n")
        for name, path in matching:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            doc = ast.get_docstring(tree)
            summary = " ".join(clean_doc(doc).split("\n\n", 1)[0].split()) if doc else ""
            summary = html.escape(summary.replace("``", "`"))
            index.append(f"- [`{name}`]({{BASE}}{module_url(name)})" + (f" — {summary}" if summary else "") + "\n")
    result.append({"url": "/api/python/", "title": "Python API", "content": "\n".join(index)})
    for name, path in modules:
        result.append({"url": module_url(name), "title": name, "content": module_page(name, path)})
    schema_index = ["# Schemas\n\nField tables are derived from the packaged JSON Schemas. The source schema remains authoritative for references, variants and validation rules.\n"]
    for path in schema_files():
        url = f"/api/schemas/{path.stem}/"
        spec = json.loads(path.read_text(encoding="utf-8"))
        schema_index.append(f"- [{html.escape(spec.get('title', path.name))}]({{BASE}}{url}) (`{path.name}`)\n")
        result.append({"url": url, "title": spec.get("title", path.name), "content": schema_page(path)})
    result.append({"url": "/api/schemas/", "title": "Schemas", "content": "\n".join(schema_index)})
    result.extend(cli_pages())
    return result


if __name__ == "__main__":
    print(json.dumps(pages(), ensure_ascii=False))
