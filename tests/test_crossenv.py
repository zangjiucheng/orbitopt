"""E5-03 cross-environment orchestration (task #270).

Fast half: the exchange envelope is sealed, re-validated, and fails closed on
tampering / foreign schemas / escaping paths; the plan loader enforces its
schema and DAG rules; the CLI registers. The slow test orchestrates the real
packaged mission across two subprocesses of the *same* interpreter (proves the
subprocess protocol, sealing and joining without needing two environments).
The ``cross_env`` test -- run by CI's cross-env job with
``ORBITOPT_TUDAT_PYTHON`` / ``ORBITOPT_GFOLD_PYTHON`` set -- is the acceptance
evidence: one command, two real environments, provenance shows tudatpy in the
ascent leg and gfold 0.3.2 in the landing leg, and the mission lands.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from orbitopt.crossenv.envelope import (CrossEnvError, file_digest, read_envelope,
                                        read_payloads, run_dir_digest, write_envelope)
from orbitopt.crossenv.orchestrator import run_plan, verify_exchange
from orbitopt.crossenv.plan import PlanError, load_plan

REPO = Path(__file__).resolve().parents[1]
FROZEN_ASCENT = REPO / "tests" / "data" / "mission" / "reference" / "e2-04-frozen-mission" / "ascent"
PACKAGED_PLAN = REPO / "src" / "orbitopt" / "data" / "crossenv_e2e_plan.yaml"


def _seal_sample(tmp_path: Path) -> Path:
    """A realistic exchange: a copied frozen E2-04 ascent run, its SECO
    handoff state and a wrapped mission report, sealed by the stage writer."""
    from orbitopt.flightlab.contract import SCHEMA_VERSION, read_run
    from orbitopt.mission.ascent import ascent_handoff_state
    from orbitopt.mission.contract import state_to_dict

    ex = tmp_path / "exchange"
    shutil.copytree(FROZEN_ASCENT, ex / "ascent-run")
    seco = state_to_dict(ascent_handoff_state(read_run(FROZEN_ASCENT)))
    (ex / "states").mkdir(parents=True, exist_ok=True)
    (ex / "states" / "seco-handoff.json").write_text(json.dumps(seco), encoding="utf-8")
    (ex / "reports").mkdir(parents=True, exist_ok=True)
    report = {"kind": "orbitopt.crossenv.report", "schema_version": "1.0", "report": {"x": 1}}
    (ex / "reports" / "mission-report.json").write_text(json.dumps(report), encoding="utf-8")
    write_envelope(ex, stage="ascent", runner="ascent", env_label="tudat", payloads=[
        {"name": "ascent-run", "type": "flight-run", "schema_version": SCHEMA_VERSION,
         "path": "ascent-run", "digest_kind": "run-dir"},
        {"name": "seco-handoff", "type": "mission-handoff-state", "schema_version": "1.0",
         "path": "states/seco-handoff.json", "digest_kind": "file"},
        {"name": "mission-report", "type": "report", "schema_version": "1.0",
         "report_kind": "orbitopt.crossenv.report", "path": "reports/mission-report.json",
         "digest_kind": "file"},
    ])
    return ex


class TestEnvelope:
    def test_roundtrip_loads_native_readers(self, tmp_path):
        from orbitopt.flightlab.contract import FlightRun
        from orbitopt.mission.contract import HandoffState as MissionState

        ex = _seal_sample(tmp_path)
        doc = read_envelope(ex)
        assert doc["kind"] == "orbitopt.crossenv.exchange"
        assert doc["schema_version"] == "1.0"
        payloads = read_payloads(ex)
        assert isinstance(payloads["ascent-run"], FlightRun)
        assert isinstance(payloads["seco-handoff"], MissionState)
        assert payloads["mission-report"]["kind"] == "orbitopt.crossenv.report"

    def test_provenance_records_environment_matrix(self, tmp_path):
        doc = read_envelope(_seal_sample(tmp_path))
        env = doc["environment"]
        assert env["python_executable"] == sys.executable
        for tool in ("numpy", "scipy", "astropy", "tudatpy", "gfold"):
            assert tool in env["toolchain"]

    def test_tampered_run_csv_is_rejected(self, tmp_path):
        ex = _seal_sample(tmp_path)
        victim = next((ex / "ascent-run").rglob("*.csv"))
        victim.write_bytes(victim.read_bytes() + b"# tampered\n")
        with pytest.raises(CrossEnvError, match="digest mismatch"):
            read_envelope(ex)

    def test_tampered_handoff_state_is_rejected(self, tmp_path):
        ex = _seal_sample(tmp_path)
        (ex / "states" / "seco-handoff.json").write_text('{"schema_version": "1.0"}',
                                                         encoding="utf-8")
        with pytest.raises(CrossEnvError, match="digest mismatch"):
            read_payloads(ex)

    def test_missing_artifact_is_rejected(self, tmp_path):
        ex = _seal_sample(tmp_path)
        shutil.rmtree(ex / "ascent-run")
        with pytest.raises(CrossEnvError, match="missing"):
            read_envelope(ex)

    def test_foreign_schema_version_is_rejected(self, tmp_path):
        ex = _seal_sample(tmp_path)
        manifest = ex / "exchange.json"
        doc = json.loads(manifest.read_text())
        doc["schema_version"] = "2.0"
        manifest.write_text(json.dumps(doc))
        with pytest.raises(CrossEnvError, match="no exchange schema"):
            read_envelope(ex)

    def test_foreign_kind_is_rejected(self, tmp_path):
        ex = _seal_sample(tmp_path)
        manifest = ex / "exchange.json"
        doc = json.loads(manifest.read_text())
        doc["kind"] = "orbitopt.something-else"
        manifest.write_text(json.dumps(doc))
        with pytest.raises(CrossEnvError, match="not a"):
            read_envelope(ex)

    def test_seal_refuses_escaping_and_missing_paths(self, tmp_path):
        (tmp_path / "outside.txt").write_text("x")
        with pytest.raises(CrossEnvError, match="inside the exchange"):
            write_envelope(tmp_path / "exchange", stage="a", runner="ascent", env_label="e",
                           payloads=[{"name": "run", "type": "flight-run",
                                      "schema_version": "1.0", "path": "../outside.txt",
                                      "digest_kind": "file"}])
        with pytest.raises(CrossEnvError, match="does not exist"):
            write_envelope(tmp_path / "exchange", stage="a", runner="ascent", env_label="e",
                           payloads=[{"name": "gone", "type": "report", "schema_version": "1.0",
                                      "report_kind": "k", "path": "nope.json",
                                      "digest_kind": "file"}])

    def test_run_dir_digest_tracks_content(self, tmp_path):
        import hashlib

        a, b = tmp_path / "a", tmp_path / "b"
        (a / "sub").mkdir(parents=True)
        (a / "run_manifest.json").write_text("{}")
        (a / "sub" / "track.csv").write_text("t,x\n0,1\n")
        shutil.copytree(a, b)
        assert run_dir_digest(a) == run_dir_digest(b)
        (b / "extra").write_text("x")
        assert run_dir_digest(a) != run_dir_digest(b)
        (b / "extra").unlink()
        (b / "sub" / "track.csv").write_text("t,x\n0,2\n")
        assert run_dir_digest(a) != run_dir_digest(b)
        assert run_dir_digest(a).startswith("sha256:")
        # a file digest of the same content must not collide with the domain prefix
        assert run_dir_digest(a) != "sha256:" + hashlib.sha256(b"{}").hexdigest()


class TestPlan:
    def _plan_doc(self, **over):
        doc = {"kind": "orbitopt.crossenv.plan", "schema_version": "1.0",
               "mission": {"id": "t"},
               "scenario": "src/orbitopt/data/mission_two_stage_leo_recovery.yaml",
               "environments": {"a": "python"},
               "stages": [{"id": "s1", "runner": "ascent", "env": "a"},
                          {"id": "s2", "runner": "recovery", "env": "a", "consumes": ["s1"]}]}
        doc.update(over)
        return doc

    def _write(self, tmp_path, doc, name="plan.yaml"):
        import yaml

        path = tmp_path / name
        path.write_text(yaml.safe_dump(doc), encoding="utf-8")
        return path

    def test_packaged_plan_loads(self):
        plan = load_plan(PACKAGED_PLAN)
        assert plan.mission_id == "e5-03-cross-env-two-stage-leo-rtls"
        assert [s.id for s in plan.stages] == ["ascent", "recovery"]
        assert plan.stages[1].consumes == ("ascent",)
        assert plan.scenario.is_file() and plan.scenario.is_absolute()

    def test_env_default_and_override(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ORBITOPT_CX_TEST_PY", raising=False)
        doc = self._plan_doc(environments={"a": "${ORBITOPT_CX_TEST_PY:-fallback}"})
        plan = load_plan(self._write(tmp_path, doc), env_overrides={"b": "override-python"})
        assert plan.environments["a"] == "fallback"
        assert plan.environments["b"] == "override-python"
        monkeypatch.setenv("ORBITOPT_CX_TEST_PY", "/set/python")
        assert load_plan(self._write(tmp_path, doc)).environments["a"] == "/set/python"

    def test_unset_variable_without_default_is_an_error(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ORBITOPT_CX_TEST_PY", raising=False)
        doc = self._plan_doc(environments={"a": "${ORBITOPT_CX_TEST_PY}"})
        with pytest.raises(PlanError, match="unset environment variable"):
            load_plan(self._write(tmp_path, doc))

    @pytest.mark.parametrize("mutate,match", [
        (lambda d: d["stages"][0].update(runner="teleport"), "is not one of"),
        (lambda d: d["stages"][1].update(consumes=["s9"]), "not a stage that runs before"),
        (lambda d: d["stages"][0].update(env="nope"), "undefined"),
        (lambda d: d.update(bogus=1), "crossenv-plan-1.0"),
        (lambda d: d.update(kind="orbitopt.nope"), "was expected"),
    ])
    def test_rejects_broken_plans(self, tmp_path, mutate, match):
        doc = self._plan_doc()
        mutate(doc)
        with pytest.raises(PlanError, match=match):
            load_plan(self._write(tmp_path, doc))

    def test_duplicate_stage_ids_rejected(self, tmp_path):
        doc = self._plan_doc()
        doc["stages"][1]["id"] = "s1"
        with pytest.raises(PlanError, match="duplicate"):
            load_plan(self._write(tmp_path, doc))


class TestCli:
    def test_main_cli_registers_crossenv(self):
        from orbitopt.cli import build_parser

        args = build_parser().parse_args(["crossenv", "run", "--plan", "p.yaml",
                                          "--env", "x=/py", "-o", "out"])
        assert args.command == "crossenv" and args.crossenv_command == "run"
        assert args.env == ["x=/py"]

    def test_verify_cli_roundtrip(self, tmp_path, capsys):
        from orbitopt.crossenv.__main__ import main

        ex = _seal_sample(tmp_path)
        assert main(["verify", str(ex)]) == 0
        assert "valid:" in capsys.readouterr().out


@pytest.mark.slow
class TestOrchestrationSingleInterpreter:
    """Full orchestration with both environments bound to the current
    interpreter: proves the subprocess protocol, per-stage sealing, the
    orchestrator's independent re-verification and the joined report. The
    mission verdict depends on the landing stack being present; the
    environment integrity assertions never do."""

    def test_packaged_plan_end_to_end(self, tmp_path):
        report = run_plan(PACKAGED_PLAN, out=tmp_path / "cx",
                          env_overrides={"tudat": sys.executable, "gfold": sys.executable},
                          verbose=False)
        assert report["overall"]["exchanges_verified"] is True
        assert [s["id"] for s in report["stages"]] == ["ascent", "recovery"]
        for stage in report["stages"]:
            assert stage["status"] == "completed"
            assert stage["exit_code"] == 0
            assert stage["exchange_digest"].startswith("sha256:")
        # report file exists, digests re-verify outside the orchestrator
        report_file = tmp_path / "cx" / "crossenv.mission-report.json"
        assert report_file.is_file()
        assert json.loads(report_file.read_text())["kind"] == "orbitopt.crossenv.mission-report"
        verify_exchange(tmp_path / "cx" / "ascent" / "exchange")
        verify_exchange(tmp_path / "cx" / "recovery" / "exchange")
        for name in ("ascent-run", "seco-handoff", "separation-handoff"):
            assert name in report["stages"][0]["payloads"]
        for name in ("recovery-run", "recovery-handoff-report", "recovery-final-handoff"):
            assert name in report["stages"][1]["payloads"]
        try:
            import importlib.metadata

            landed = importlib.metadata.version("gfold") == "0.3.2"
        except importlib.metadata.PackageNotFoundError:
            landed = False
        if landed:
            assert report["overall"]["mission_ok"] is True


@pytest.mark.cross_env
@pytest.mark.slow
class TestCrossEnvironmentEndToEnd:
    """The acceptance test: the two real environments. Requires
    ORBITOPT_TUDAT_PYTHON (conda 3.10, tudatpy) and ORBITOPT_GFOLD_PYTHON
    (3.11+ venv, gfold 0.3.2) -- CI's cross-env job provides them."""

    tudat_py = os.environ.get("ORBITOPT_TUDAT_PYTHON", "")
    gfold_py = os.environ.get("ORBITOPT_GFOLD_PYTHON", "")

    @pytest.mark.skipif(not (tudat_py and gfold_py),
                        reason="set ORBITOPT_TUDAT_PYTHON and ORBITOPT_GFOLD_PYTHON")
    def test_one_command_two_environments(self, tmp_path):
        report = run_plan(PACKAGED_PLAN, out=tmp_path / "cx",
                          env_overrides={"tudat": self.tudat_py, "gfold": self.gfold_py},
                          verbose=False)
        ascent_env = report["environment_matrix"]["tudat"]
        recovery_env = report["environment_matrix"]["gfold"]
        assert ascent_env["python_executable"] != recovery_env["python_executable"]
        assert ascent_env["python"] != recovery_env["python"], \
            "the whole point of E5-03: the legs really ran in different Pythons"
        assert ascent_env["toolchain"]["gfold"] == "absent"
        assert recovery_env["toolchain"]["gfold"] == "0.3.2"
        assert report["overall"]["exchanges_verified"] is True
        assert report["overall"]["mission_ok"] is True, \
            f"mission verdict: {report['stages']}"
