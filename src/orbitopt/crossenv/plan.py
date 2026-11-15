"""The cross-environment plan: which stage runs where, in which order.

A plan (``kind`` ``orbitopt.crossenv.plan``, ``schema_version`` ``1.0``,
validated against ``orbitopt/schemas/crossenv-plan-1.0.json``) names the
mission scenario every stage shares, maps environment labels to Python
interpreters, and lists the stages in execution order. Interpreter paths may
carry ``${VAR}`` / ``${VAR:-default}`` expansions so one packaged plan serves
developer machines and CI:

    environments:
      tudat: ${ORBITOPT_TUDAT_PYTHON:-python}
      gfold: ${ORBITOPT_GFOLD_PYTHON:-python}

``load_plan`` expands them, rejects unknown runners, duplicate stage ids or
``consumes`` references to stages that have not run yet (the plan is a DAG in
list order), and resolves every relative path against the repository root,
so a stage subprocess works from any working directory. The orchestrator
serialises the resolved plan to ``<out>/plan.resolved.json`` (absolute paths
only) and hands subprocesses exactly that file.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

from orbitopt.crossenv.envelope import CrossEnvError

PLAN_KIND = "orbitopt.crossenv.plan"
PLAN_SCHEMA_VERSION = "1.0"
PLAN_SCHEMA_FILE = "crossenv-plan-1.0.json"
RESOLVED_PLAN_FILE = "plan.resolved.json"

#: Runner names the stage subprocesses dispatch on (``stages.STAGE_RUNNERS``).
KNOWN_RUNNERS = ("ascent", "recovery")

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class PlanError(CrossEnvError):
    """A plan that cannot be executed as written: schema violation, unknown
    runner, dangling ``consumes``, undefined environment or a missing
    scenario file."""


def repo_root(start: Path | None = None) -> Path:
    """The repository root for resolving config-relative paths: the nearest
    ancestor carrying both ``pyproject.toml`` and ``src/orbitopt`` (true for
    any editable install and any plain checkout), else the working directory
    -- the same fallback the other config loaders accept."""
    here = (start or Path(__file__).resolve()).parent
    for cand in (here, *here.parents):
        if (cand / "pyproject.toml").is_file() and (cand / "src" / "orbitopt").is_dir():
            return cand
    return Path.cwd()


def resolve_path(text) -> Path:
    """Absolute paths pass through; relative ones resolve against the repo
    root, never against the caller's cwd."""
    p = Path(str(text)).expanduser()
    return p if p.is_absolute() else (repo_root() / p)


def expand_env_vars(text: str) -> str:
    """``${NAME}`` / ``${NAME:-default}`` substitution from the process
    environment; an unset variable without a default raises ``PlanError`` --
    silently running a stage on the wrong interpreter is the exact failure
    mode E5-03 exists to prevent."""
    missing: list[str] = []

    def repl(match: re.Match) -> str:
        name, default = match.group(1), match.group(2)
        value = os.environ.get(name)
        if value is None:
            if default is None:
                missing.append(name)
                return match.group(0)
            return default
        return value

    out = _ENV_PATTERN.sub(repl, str(text))
    if missing:
        raise PlanError(f"plan references unset environment variable(s): "
                        f"{', '.join(sorted(set(missing)))} (use ${{{missing[0]}:-default}} "
                        "to give one)")
    return out


@dataclass(frozen=True)
class CrossEnvStage:
    """One planned leg: ``runner`` decides what runs, ``env`` names the
    interpreter that runs it, ``consumes`` lists the stage ids whose sealed
    exchanges must validate before this stage starts."""

    id: str
    runner: str
    env: str
    consumes: tuple[str, ...] = ()
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CrossEnvPlan:
    """A validated plan with every path absolute and every interpreter path
    expanded. ``source`` records the file it was loaded from (the resolved
    form has none)."""

    mission_id: str
    scenario: Path
    environments: Mapping[str, str]
    stages: tuple[CrossEnvStage, ...]
    out: Path | None = None
    source: Path | None = None

    def stage(self, stage_id: str) -> CrossEnvStage:
        for st in self.stages:
            if st.id == stage_id:
                return st
        raise PlanError(f"plan {self.mission_id!r} has no stage {stage_id!r} "
                        f"(stages: {', '.join(s.id for s in self.stages)})")


def _load_yaml(path: Path) -> dict:
    import yaml

    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PlanError(f"no such file: {str(path)!r}") from exc
    except yaml.YAMLError as exc:
        raise PlanError(f"{str(path)!r} is not valid YAML: {exc}") from exc
    if not isinstance(doc, dict):
        raise PlanError(f"{str(path)!r} must hold a mapping at top level")
    return doc


def _validate_plan_doc(doc: Mapping[str, Any], *, source: str) -> None:
    import jsonschema

    try:
        schema = json.loads(resources.files("orbitopt").joinpath(
            "schemas", PLAN_SCHEMA_FILE).read_text(encoding="utf-8"))
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise PlanError(f"no packaged schema orbitopt/schemas/{PLAN_SCHEMA_FILE}") from exc
    try:
        jsonschema.validate(instance=dict(doc), schema=schema)
    except jsonschema.ValidationError as exc:
        raise PlanError(f"plan {source!r} violates {PLAN_SCHEMA_FILE}: {exc.message}") from exc


def load_plan(path, *, env_overrides: Mapping[str, str] | None = None) -> CrossEnvPlan:
    """Parse, schema-check and resolve a plan file (YAML or the JSON written
    by :func:`write_resolved_plan`). ``env_overrides`` (``NAME=python`` from
    the CLI ``--env``) win over the plan's ``environments`` map."""
    src = Path(str(path)).expanduser().resolve()
    doc = _load_json_or_yaml(src)
    _validate_plan_doc(doc, source=str(src))
    if doc.get("kind") != PLAN_KIND:
        raise PlanError(f"not a {PLAN_KIND!r} plan (kind: {doc.get('kind')!r})")
    version = doc.get("schema_version")
    if version != PLAN_SCHEMA_VERSION:
        raise PlanError(f"no plan schema for version {version!r} (this orbitopt implements "
                        f"{PLAN_SCHEMA_VERSION!r})")

    mission = doc["mission"]["id"]
    scenario = resolve_path(doc["scenario"])
    if not scenario.is_file():
        raise PlanError(f"plan {str(src)!r}: scenario {str(scenario)!r} does not exist")

    environments: dict[str, str] = {}
    for name, value in doc["environments"].items():
        environments[str(name)] = expand_env_vars(str(value))
    for name, value in (env_overrides or {}).items():
        environments[str(name)] = expand_env_vars(str(value))

    stages: list[CrossEnvStage] = []
    seen: set[str] = set()
    for raw in doc["stages"]:
        stage = CrossEnvStage(
            id=str(raw["id"]), runner=str(raw["runner"]), env=str(raw["env"]),
            consumes=tuple(str(c) for c in raw.get("consumes", ())),
            options=dict(raw.get("options", {})))
        if stage.id in seen:
            raise PlanError(f"plan {str(src)!r}: duplicate stage id {stage.id!r}")
        if stage.runner not in KNOWN_RUNNERS:
            raise PlanError(f"plan {str(src)!r}: stage {stage.id!r} names unknown runner "
                            f"{stage.runner!r} (known: {', '.join(KNOWN_RUNNERS)})")
        if stage.env not in environments:
            raise PlanError(f"plan {str(src)!r}: stage {stage.id!r} selects environment "
                            f"{stage.env!r}, undefined in the plan's environments map "
                            f"(known: {', '.join(sorted(environments))})")
        for dep in stage.consumes:
            if dep not in seen:
                raise PlanError(f"plan {str(src)!r}: stage {stage.id!r} consumes {dep!r}, "
                                "which is not a stage that runs before it")
        seen.add(stage.id)
        stages.append(stage)

    out = resolve_path(doc["out"]) if doc.get("out") else None
    return CrossEnvPlan(mission_id=str(mission), scenario=scenario,
                        environments=environments, stages=tuple(stages),
                        out=out, source=src)


def _load_json_or_yaml(src: Path) -> dict:
    if src.suffix.lower() == ".json":
        try:
            doc = json.loads(src.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise PlanError(f"no such file: {str(src)!r}") from exc
        except json.JSONDecodeError as exc:
            raise PlanError(f"{str(src)!r} is not valid JSON: {exc}") from exc
        if not isinstance(doc, dict):
            raise PlanError(f"{str(src)!r} must hold a mapping at top level")
        return doc
    return _load_yaml(src)


def to_resolved_dict(plan: CrossEnvPlan) -> dict:
    """The JSON form written to ``plan.resolved.json``: same schema, but
    every path absolute and every variable expanded, so a stage subprocess
    can reload it without the plan's author's environment."""
    return {
        "kind": PLAN_KIND,
        "schema_version": PLAN_SCHEMA_VERSION,
        "mission": {"id": plan.mission_id},
        "scenario": str(plan.scenario),
        **({"out": str(plan.out)} if plan.out else {}),
        "environments": dict(plan.environments),
        "stages": [{"id": s.id, "runner": s.runner, "env": s.env,
                    **({"consumes": list(s.consumes)} if s.consumes else {}),
                    **({"options": dict(s.options)} if s.options else {})}
                   for s in plan.stages],
    }


def write_resolved_plan(plan: CrossEnvPlan, path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(to_resolved_dict(plan), indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    return out


def run_stage_subprocess(plan: CrossEnvPlan, stage: CrossEnvStage, out_dir,
                         resolved_plan: Path) -> subprocess.CompletedProcess:
    """Launch one stage in its declared environment: the interpreter from the
    plan runs ``python -m orbitopt.crossenv stage``. ``PYTHONPATH`` gets the
    source tree of this very ``orbitopt`` prepended when running from a
    checkout, so the subprocess exercises the same code the orchestrator
    runs, not an older installed copy."""
    env = dict(os.environ)
    pkg = Path(__file__).resolve()
    src_dir = pkg.parents[2]  # .../src when running from a source checkout
    if (src_dir / "orbitopt" / "crossenv").is_dir():
        prior = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(src_dir) + (os.pathsep + prior if prior else "")
    cmd = [plan.environments[stage.env], "-m", "orbitopt.crossenv", "stage",
           "--plan", str(resolved_plan), "--stage", stage.id, "--out", str(Path(out_dir))]
    return subprocess.run(cmd, env=env, cwd=str(repo_root()), capture_output=True,
                          text=True, check=False)
