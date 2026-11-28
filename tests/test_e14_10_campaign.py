"""E14-10 campaign runner, end to end on three tiny instances in pipeline-smoke mode (no server, no model).

It checks what the qualification report rests on: every shard file carries the frozen digests and seals its content,
a scored shard refuses to run without a matching freeze, the analysis refuses pipeline-smoke rows unless told it is a
pipeline test, and it refuses shards run under different digests. Mock providers stand in for a model here, so no
number this test computes says anything about any model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCH))

import bench_e14_end_to_end as E2E  # noqa: E402
import e14_analysis as AN  # noqa: E402
import e14_common as C  # noqa: E402

INSTANCES = ["cqp-worked-d2", "lp-wyndor-d2", "knapsack-p01-n10"]
ROW_KEYS = {"arm_id", "instance_id", "family", "problem_class", "group", "split", "seed", "mode", "problem_digest",
            "status", "stop", "success", "gap", "max_violation", "n_evaluations", "n_evaluations_independent",
            "budget_evaluations", "within_budget", "budget_typed", "wall_clock_s", "llm_calls", "tokens_in",
            "tokens_out", "cost_usd", "verification", "row_digest", "transcript_digest"}


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    out = tmp_path_factory.mktemp("campaign")
    freeze = out / "freeze.json"
    shards = E2E.plan(INSTANCES, [0])
    scored = next(s for s in shards if s["kind"] == "cells" and s["split"] == "scoring")
    with pytest.raises(C.FreezeError, match="no freeze"):
        E2E.run_shard(scored, out, out / "runs", E2E.SMOKE, freeze, ["test"])
    assert E2E.main(["freeze", "--freeze-path", str(freeze)]) == 0
    for shard in shards:
        E2E.run_shard(shard, out, out / "runs", E2E.SMOKE, freeze, ["test"])
    return out, shards, freeze


def test_every_shard_is_sealed_under_the_frozen_digests(campaign):
    out, shards, freeze = campaign
    frozen = C.load_freeze(freeze)
    docs = AN.load_shards(out)
    assert {d["shard"]["id"] for d in docs} == {s["id"] for s in shards}
    for doc in docs:
        assert C.seal_ok({k: v for k, v in doc.items() if k != "_file"})  # load_shards adds the file name
        assert doc["mode"] == E2E.SMOKE
        assert doc["spec_digest"] == C.SPEC.spec_digest() == frozen["spec_digest"]
        assert doc["protocol_digest"] == frozen["protocol_digest"] and doc["code_digest"] == frozen["code_digest"]
        assert doc["freeze_digest"] == frozen["content_digest"]


def test_rows_follow_the_schema_and_every_arm_is_labelled(campaign):
    out, _, _ = campaign
    rows = [r for d in AN.load_shards(out) for r in d["rows"]]
    assert {r["arm_id"] for r in rows} == set(C.ALL_ARMS)
    for row in rows:
        assert ROW_KEYS <= row.keys(), sorted(ROW_KEYS - row.keys())
        assert row["mode"] == E2E.SMOKE
        assert row["row_digest"] == C.row_digest(row)
    for row in rows:
        if row["arm_id"] in C.RULE_ARMS:
            assert row["tokens_in"] == row["tokens_out"] == 0 and row["cost_usd"] == 0.0
            assert row["n_evaluations"] == row["n_evaluations_independent"] <= row["budget_evaluations"]
            assert row["replay"]["equal"] is True


def test_a_rerun_skips_finished_shards(campaign):
    out, shards, freeze = campaign
    status = E2E.run_shard(shards[0], out, out / "runs", E2E.SMOKE, freeze, ["test"])
    assert status.startswith("skipped")


def test_analysis_refuses_smoke_rows_unless_told_and_labels_them(campaign):
    out, _, _ = campaign
    with pytest.raises(RuntimeError, match="pipeline-smoke"):
        AN.analyze(out)
    analysis = AN.analyze(out, allow_smoke=True)
    assert analysis["label"].startswith("PIPELINE SMOKE")
    assert "not evidence" in AN.render_tables(json.loads((out / "analysis.json").read_text()))
    faults = analysis["safety"]["false_verification_pass_faults"]["by_path"]
    assert set(faults) == {"verify_result", "verify_run"}
    assert all(v["false_pass"] == 0 and v["reason_named"] == v["graded"] for v in faults.values())


def test_collect_writes_a_manifest_that_catches_a_changed_byte(campaign, tmp_path):
    out, shards, _ = campaign
    AN.collect(out, shards)
    assert AN.verify(out) == []
    copy = tmp_path / "copy"
    (copy / "shards").mkdir(parents=True)
    for name in ("manifest.json", "cells.csv"):
        (copy / name).write_bytes((out / name).read_bytes())
    for path in (out / "shards").glob("*.json"):
        (copy / "shards" / path.name).write_bytes(path.read_bytes())
    victim = next((copy / "shards").glob("cells.scoring.B.*.json"))
    victim.write_bytes(victim.read_bytes().replace(b'"seed": 0', b'"seed": 1', 1))
    assert any(victim.name in p for p in AN.verify(copy))


def test_analysis_refuses_shards_run_under_different_digests(campaign, tmp_path):
    out, _, _ = campaign
    mixed = tmp_path / "mixed"
    (mixed / "shards").mkdir(parents=True)
    for path in (out / "shards").glob("*.json"):
        (mixed / "shards" / path.name).write_bytes(path.read_bytes())
    victim = next((mixed / "shards").glob("cells.scoring.A.*.json"))
    doc = json.loads(victim.read_text())
    doc["spec_digest"] = "sha256:" + "0" * 64
    C.write_json_atomic(victim, C.sealed(doc))  # resealed: only the digest check can catch it
    with pytest.raises(AN.MixedDigestError, match="spec_digest"):
        AN.analyze(mixed, allow_smoke=True)


def test_a_scored_shard_refuses_a_freeze_whose_digests_moved(campaign, tmp_path):
    out, shards, freeze = campaign
    moved = tmp_path / "freeze.json"
    doc = json.loads(freeze.read_text())
    doc["protocol_digest"] = "sha256:" + "1" * 64
    C.write_json_atomic(moved, C.sealed(doc))
    scored = next(s for s in shards if s["kind"] == "cells" and s["split"] == "scoring")
    with pytest.raises(C.FreezeError, match="protocol_digest"):
        E2E.run_shard(scored, tmp_path / "elsewhere", tmp_path / "runs", E2E.SMOKE, moved, ["test"])


def test_scored_plan_cannot_be_narrowed_outside_smoke_mode(capsys):
    with pytest.raises(SystemExit):
        E2E.main(["plan", "--instances", "lp-wyndor-d2"])
    assert "pipeline-smoke only" in capsys.readouterr().err


def test_provider_arms_are_not_run_without_credentials(monkeypatch):
    for var in ("E14_LLM_BASE_URL", "E14_LLM_MODEL", "E14_LLM_API_KEY", "E14_LLM_KIND"):
        monkeypatch.delenv(var, raising=False)
    assert C.provider_from_env() is None
    monkeypatch.setenv("E14_LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("E14_LLM_MODEL", "some-model")
    assert C.provider_from_env() is None  # an openai-compatible endpoint without a key is not a provider


def test_the_provider_from_the_e14_variables_talks_to_an_openai_compatible_endpoint(monkeypatch):
    """The path the provider arms take (E14_LLM_KIND, E14_LLM_BASE_URL, E14_LLM_MODEL, E14_LLM_API_KEY), against
    a fake OpenAI-compatible server on 127.0.0.1: no network, no real model, no key outside this test."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from orbitopt.agent.providers import Message

    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, dict(self.headers), body))
            data = json.dumps({"id": "c1", "model": "deepseek-chat", "object": "chat.completion",
                               "choices": [{"index": 0, "finish_reason": "stop",
                                            "message": {"role": "assistant", "content": "ok"}}],
                               "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    try:
        monkeypatch.setenv("NO_PROXY", "*")
        monkeypatch.setenv("E14_LLM_KIND", "openai_compat")
        monkeypatch.setenv("E14_LLM_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}/v1")
        monkeypatch.setenv("E14_LLM_MODEL", "deepseek-chat")
        monkeypatch.setenv("E14_LLM_API_KEY", "sk-test-not-a-key")
        provider = C.provider_from_env()
        assert provider is not None
        reply = provider.generate([Message("user", "Say ok.")], max_tokens=8, seed=3)
    finally:
        server.shutdown()
        server.server_close()
    path, headers, body = seen[0]
    assert path == "/v1/chat/completions"
    assert headers["Authorization"] == "Bearer sk-test-not-a-key"
    assert body["model"] == "deepseek-chat" and body["seed"] == 3
    assert reply.text == "ok"
