"""E14-08 LLM provider abstraction: no API key, no network, no real model.

The HTTP providers run against a fake OpenAI-compatible endpoint served by
``http.server`` on 127.0.0.1.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from orbitopt.agent.providers import (
    BudgetExceededError,
    ChatRequest,
    LLMProvider,
    LocalProvider,
    Message,
    MockProvider,
    MockReply,
    OpenAICompatProvider,
    ProviderError,
    ProviderUnavailableError,
    RecordingProvider,
    ReplayMismatchError,
    ReplayProvider,
    StructuredOutputError,
    ToolCall,
    ToolCallError,
    ToolSpec,
    UsageMeter,
    make_provider,
)

REPO = Path(__file__).resolve().parents[1]
KEY_ENV = "ORBITOPT_TEST_LLM_KEY"
SECRET = "sk-test-0123456789-do-not-leak"
SCHEMA = {"type": "object", "properties": {"dv_mps": {"type": "number"}},
          "required": ["dv_mps"], "additionalProperties": False}
TOOL = ToolSpec("propagate", "Propagate an orbit.",
                {"type": "object", "properties": {"days": {"type": "integer", "minimum": 0}},
                 "required": ["days"]})
ASK = [Message("system", "You are a mission planner."), Message("user", "Plan the burn.")]


class FakeEndpoint:
    """Scripted OpenAI-compatible server; records every request it receives."""

    def __init__(self):
        self.replies = deque()
        self.requests = []
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                endpoint.requests.append({"path": self.path, "headers": dict(self.headers),
                                          "body": json.loads(body)})
                status, payload, delay = endpoint.replies.popleft() if endpoint.replies else (
                    500, {"error": "no scripted reply"}, 0)
                if callable(payload):
                    payload = payload(dict(self.headers))
                time.sleep(delay)
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address):
                pass  # the client hung up on a deliberately slow reply

        self.server = Server(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01},
                         daemon=True).start()

    def reply(self, status=200, payload=None, delay=0.0):
        self.replies.append((status, payload, delay))


def completion(content=None, tool_calls=None, usage=(5, 7), finish="stop"):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [
            {"id": f"call_{i}", "type": "function",
             "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}
            for i, (name, args) in enumerate(tool_calls)]
    return {"id": "chatcmpl-1", "model": "fake-model", "object": "chat.completion",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1],
                      "total_tokens": sum(usage)}}


@pytest.fixture(autouse=True)
def offline_env(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv(KEY_ENV, SECRET)


@pytest.fixture
def endpoint():
    fake = FakeEndpoint()
    yield fake
    fake.server.shutdown()
    fake.server.server_close()


def http_provider(endpoint, **kwargs):
    options = {"api_key_env": KEY_ENV, "backoff_s": 0.0, "max_retries": 2, "timeout_s": 5.0}
    options.update(kwargs)
    return OpenAICompatProvider(endpoint.base_url, "fake-model", **options)


# OpenAI-compatible provider against the fake endpoint.

def test_openai_compat_generate_sends_chat_completion_and_reads_usage(endpoint):
    endpoint.reply(payload=completion("Burn at perigee.", usage=(11, 3)))
    response = http_provider(endpoint).generate(ASK, max_tokens=64, seed=7)

    sent = endpoint.requests[0]
    assert sent["path"] == "/v1/chat/completions"
    assert sent["headers"]["Authorization"] == f"Bearer {SECRET}"
    assert sent["body"]["model"] == "fake-model"
    assert sent["body"]["max_tokens"] == 64 and sent["body"]["seed"] == 7
    assert sent["body"]["temperature"] == 0.0
    assert sent["body"]["messages"] == [m.to_dict() for m in ASK]
    assert response.text == "Burn at perigee."
    assert (response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total) == (11, 3, 14)
    assert response.usage.cost_usd is None
    assert (response.model, response.finish_reason, response.raw_id) == ("fake-model", "stop", "chatcmpl-1")
    assert response.latency_s >= 0.0


def test_openai_compat_structured_output_uses_json_schema_response_format(endpoint):
    endpoint.reply(payload=completion('{"dv_mps": 12.5}'))
    response = http_provider(endpoint).generate_structured(ASK, SCHEMA, max_tokens=64)

    body = endpoint.requests[0]["body"]
    assert body["response_format"] == {"type": "json_schema",
                                       "json_schema": {"name": "response", "schema": SCHEMA}}
    assert response.parsed == {"dv_mps": 12.5}


def test_local_provider_sends_no_key_and_instructs_for_json(endpoint):
    endpoint.reply(payload=completion('```json\n{"dv_mps": 3}\n```'))
    provider = LocalProvider("qwen-local", endpoint.base_url, max_retries=0)
    response = provider.generate_structured(ASK, SCHEMA, max_tokens=64)

    sent = endpoint.requests[0]
    assert "Authorization" not in sent["headers"]
    assert "response_format" not in sent["body"]
    system = sent["body"]["messages"][0]
    assert system["role"] == "system" and system["content"].startswith("You are a mission planner.")
    assert '"dv_mps"' in system["content"] and len(sent["body"]["messages"]) == 2
    assert response.parsed == {"dv_mps": 3}
    assert LocalProvider("m").base_url == "http://localhost:11434/v1"


def test_openai_compat_tool_calls_are_parsed_and_validated(endpoint):
    endpoint.reply(payload=completion(None, tool_calls=[("propagate", {"days": 3})], finish="tool_calls"))
    response = http_provider(endpoint).call_tools(ASK, [TOOL], max_tokens=64)

    assert endpoint.requests[0]["body"]["tools"] == [{"type": "function", "function": {
        "name": "propagate", "description": "Propagate an orbit.",
        "parameters": TOOL.parameters_schema}}]
    assert response.tool_calls == (ToolCall("call_0", "propagate", {"days": 3}),)
    assert response.finish_reason == "tool_calls"


def test_tool_call_round_trip_message_is_sent_in_openai_wire_format(endpoint):
    endpoint.reply(payload=completion("Done."))
    history = ASK + [Message("assistant", None, tool_calls=[ToolCall("c1", "propagate", {"days": 1})]),
                     Message("tool", '{"ok": true}', tool_call_id="c1")]
    http_provider(endpoint).generate(history, max_tokens=8)
    sent = endpoint.requests[0]["body"]["messages"]
    assert sent[2]["tool_calls"] == [{"id": "c1", "type": "function",
                                      "function": {"name": "propagate", "arguments": '{"days": 1}'}}]
    assert sent[3] == {"role": "tool", "content": '{"ok": true}', "tool_call_id": "c1"}


def test_openai_compat_retries_429_then_succeeds(endpoint):
    endpoint.reply(429, {"error": "slow down"})
    endpoint.reply(payload=completion("ok"))
    assert http_provider(endpoint).generate(ASK, max_tokens=8).text == "ok"
    assert len(endpoint.requests) == 2


def test_openai_compat_retries_a_timeout_then_succeeds(endpoint):
    endpoint.reply(payload=completion("late"), delay=1.0)
    endpoint.reply(payload=completion("on time"))
    response = http_provider(endpoint, timeout_s=0.3).generate(ASK, max_tokens=8)
    assert response.text == "on time"


def test_openai_compat_persistent_500_stops_at_the_retry_cap(endpoint):
    for _ in range(5):
        endpoint.reply(500, {"error": "boom"})
    with pytest.raises(ProviderUnavailableError, match="after 3 attempts; last error: HTTP 500"):
        http_provider(endpoint, max_retries=2).generate(ASK, max_tokens=8)
    assert len(endpoint.requests) == 3


def test_openai_compat_does_not_retry_client_errors(endpoint):
    endpoint.reply(400, {"error": "bad request"})
    with pytest.raises(ProviderError, match="HTTP 400") as caught:
        http_provider(endpoint).generate(ASK, max_tokens=8)
    assert not isinstance(caught.value, ProviderUnavailableError)
    assert len(endpoint.requests) == 1


def test_unreachable_server_is_unavailable_after_bounded_retries():
    provider = OpenAICompatProvider("http://127.0.0.1:9/v1", "m", api_key_env=None,
                                    max_retries=1, backoff_s=0.0, timeout_s=1.0)
    with pytest.raises(ProviderUnavailableError, match="after 2 attempts"):
        provider.generate(ASK, max_tokens=8)


def test_malformed_json_gets_one_repair_retry_then_structured_output_error(endpoint):
    endpoint.reply(payload=completion("{dv_mps: 12"))
    endpoint.reply(payload=completion('{"dv_mps": "fast"}'))
    with pytest.raises(StructuredOutputError) as caught:
        http_provider(endpoint).generate_structured(ASK, SCHEMA, max_tokens=64)

    assert len(endpoint.requests) == 2
    repair = endpoint.requests[1]["body"]["messages"]
    assert repair[-2] == {"role": "assistant", "content": "{dv_mps: 12"}
    assert repair[-1]["role"] == "user" and "not valid JSON" in repair[-1]["content"]
    assert caught.value.raw_text == '{"dv_mps": "fast"}'
    assert caught.value.errors == ("dv_mps: 'fast' is not of type 'number'",)
    assert caught.value.usage.total == 2 * (5 + 7)  # both failed replies are accounted for


def test_repair_retry_success_returns_validated_value_and_summed_usage(endpoint):
    endpoint.reply(payload=completion('{"dv_mps": 1, "extra": 2}', usage=(10, 4)))
    endpoint.reply(payload=completion('{"dv_mps": 1}', usage=(20, 3)))
    response = http_provider(endpoint).generate_structured(ASK, SCHEMA, max_tokens=64)
    assert response.parsed == {"dv_mps": 1}
    assert (response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total) == (30, 7, 37)


def test_dotted_tool_ids_cross_the_wire_as_function_names_and_come_back(endpoint):
    erf = ToolSpec("math.erf.v1", "Error function.", {"type": "object", "properties": {"x": {"type": "number"}},
                                                       "required": ["x"], "additionalProperties": False})
    endpoint.reply(payload=completion(None, tool_calls=[("math-erf-v1", {"x": 0.5})], finish="tool_calls"))
    history = ASK + [Message("assistant", None, tool_calls=[ToolCall("c0", "math.erf.v1", {"x": 0.1})]),
                     Message("tool", '{"value": 0.11}', tool_call_id="c0")]
    response = http_provider(endpoint).call_tools(history, [erf], max_tokens=64)
    body = endpoint.requests[0]["body"]
    assert body["tools"][0]["function"]["name"] == "math-erf-v1"
    assert body["messages"][2]["tool_calls"][0]["function"]["name"] == "math-erf-v1"
    assert response.tool_calls == (ToolCall("call_0", "math.erf.v1", {"x": 0.5}),)
    clash = ToolSpec("math-erf-v1", "Another.", erf.parameters_schema)
    with pytest.raises(ProviderError, match="same function name"):
        http_provider(endpoint).call_tools(ASK, [erf, clash], max_tokens=64)
    with pytest.raises(ProviderError, match="cannot be sent"):
        http_provider(endpoint).call_tools(ASK, [ToolSpec("x" * 65, "Long.", erf.parameters_schema)], max_tokens=64)


def test_invalid_tool_calls_still_report_their_tokens(endpoint):
    endpoint.reply(payload=completion(None, tool_calls=[("rm_rf", {"path": "/"})], usage=(11, 2)))
    with pytest.raises(ToolCallError) as caught:
        http_provider(endpoint).call_tools(ASK, [TOOL], max_tokens=64)
    assert caught.value.usage.total == 13
    endpoint.reply(payload=completion(None, tool_calls=[("propagate", "[1, 2]")], usage=(3, 4)))
    with pytest.raises(ToolCallError) as caught:
        http_provider(endpoint).call_tools(ASK, [TOOL], max_tokens=64)
    assert caught.value.usage.total == 7


def test_unknown_tool_name_from_the_model_is_rejected(endpoint):
    endpoint.reply(payload=completion(None, tool_calls=[("rm_rf", {"path": "/"})]))
    with pytest.raises(ToolCallError, match="invalid tool calls") as caught:
        http_provider(endpoint).call_tools(ASK, [TOOL], max_tokens=64)
    assert "unknown tool 'rm_rf'" in caught.value.errors[0]


def test_tool_arguments_that_are_not_json_are_rejected(endpoint):
    endpoint.reply(payload=completion(None, tool_calls=[("propagate", "{days: 3")]))
    with pytest.raises(ToolCallError, match="not a JSON object"):
        http_provider(endpoint).call_tools(ASK, [TOOL], max_tokens=64)


def test_response_without_usage_is_an_error_not_silent_zero(endpoint):
    payload = completion("ok")
    del payload["usage"]
    endpoint.reply(payload=payload)
    with pytest.raises(ProviderError, match="malformed response"):
        http_provider(endpoint).generate(ASK, max_tokens=8)


def test_api_key_never_appears_in_repr_or_errors(endpoint):
    provider = http_provider(endpoint, max_retries=1)
    assert SECRET not in repr(provider) and KEY_ENV in repr(provider)
    assert all(SECRET not in str(value) for value in vars(provider).values())

    echo = lambda headers: {"error": f"invalid key {headers['Authorization']}"}  # noqa: E731
    endpoint.reply(401, echo)
    with pytest.raises(ProviderError) as client_error:
        provider.generate(ASK, max_tokens=8)
    endpoint.reply(500, echo)
    endpoint.reply(500, echo)
    with pytest.raises(ProviderUnavailableError) as outage:
        provider.generate(ASK, max_tokens=8)
    endpoint.reply(200, lambda headers: f"not json {headers['Authorization']}".encode())
    with pytest.raises(ProviderError) as garbage:
        provider.generate(ASK, max_tokens=8)
    for error in (client_error.value, outage.value, garbage.value):
        assert SECRET not in str(error) and "[REDACTED]" in str(error)
        assert error.__cause__ is None


def test_missing_api_key_names_the_variable(monkeypatch, endpoint):
    monkeypatch.delenv(KEY_ENV)
    with pytest.raises(ProviderError, match=f"environment variable {KEY_ENV} is not set"):
        http_provider(endpoint).generate(ASK, max_tokens=8)
    assert endpoint.requests == []


# Mock provider and failure injection.

def test_mock_serves_script_in_order_with_whitespace_token_rule():
    mock = MockProvider(["two words", MockReply(tool_calls=[ToolCall("c", "propagate", {"days": 2})])])
    first = mock.generate([Message("user", "count these four words")], max_tokens=8)
    assert (first.text, first.usage.prompt_tokens, first.usage.completion_tokens) == ("two words", 4, 2)
    second = mock.call_tools(ASK, [TOOL], max_tokens=8)
    # name (1) + '{"days": 2}' (2 words); prompt is 5 + 3 words
    assert (second.usage.prompt_tokens, second.usage.completion_tokens, second.usage.total) == (8, 3, 11)
    assert second.finish_reason == "tool_calls" and second.raw_id == "mock-2"
    assert [r.tools for r in mock.requests] == [(), (TOOL,)]
    with pytest.raises(ProviderError, match="script exhausted"):
        mock.generate(ASK, max_tokens=8)


def test_mock_script_can_be_a_function_of_the_request():
    mock = MockProvider(lambda request: f"echo {request.messages[-1].content}")
    assert mock.generate(ASK, max_tokens=8).text == "echo Plan the burn."


def test_mock_failure_injection():
    mock = MockProvider([
        ProviderUnavailableError("simulated timeout"),
        "{not json", '{"also": "wrong"}',
        MockReply(tool_calls=[ToolCall("c", "launch_nukes", {})]),
        MockReply(tool_calls=[ToolCall("c", "propagate", {"days": -1})]),
    ])
    with pytest.raises(ProviderUnavailableError, match="simulated timeout"):
        mock.generate(ASK, max_tokens=8)
    with pytest.raises(StructuredOutputError) as structured:
        mock.generate_structured(ASK, SCHEMA, max_tokens=8)
    assert structured.value.raw_text == '{"also": "wrong"}' and len(mock.requests) == 3
    with pytest.raises(ToolCallError, match="invalid tool calls") as unknown:
        mock.call_tools(ASK, [TOOL], max_tokens=8)
    assert "unknown tool 'launch_nukes'" in unknown.value.errors[0]
    with pytest.raises(ToolCallError) as bad_args:
        mock.call_tools(ASK, [TOOL], max_tokens=8)
    assert "-1 is less than the minimum of 0" in bad_args.value.errors[0]


def test_every_provider_satisfies_the_protocol(tmp_path):
    cassette = tmp_path / "c.jsonl"
    mock = MockProvider([])
    for provider in (mock, RecordingProvider(mock, cassette), ReplayProvider(cassette),
                     OpenAICompatProvider("http://127.0.0.1:9/v1", "m"), LocalProvider("m")):
        assert isinstance(provider, LLMProvider)


# Record and replay.

def run_session(provider):
    """A fixed agent-like session: text, structured with repair, tools, an outage."""
    results = [provider.generate(ASK, max_tokens=32, seed=1),
               provider.generate_structured(ASK, SCHEMA, max_tokens=32),
               provider.call_tools(ASK, [TOOL], max_tokens=32)]
    with pytest.raises(ProviderUnavailableError) as outage:
        provider.generate([Message("user", "again")], max_tokens=8)
    return [r.to_dict() for r in results], str(outage.value)


def test_record_then_replay_round_trip_is_byte_for_byte(tmp_path, endpoint):
    endpoint.reply(payload=completion("Burn at perigee."))
    endpoint.reply(payload=completion('{"dv_mps": "x"}'))
    endpoint.reply(payload=completion('{"dv_mps": 4.25}'))
    endpoint.reply(payload=completion(None, tool_calls=[("propagate", {"days": 2})]))
    for _ in range(3):
        endpoint.reply(503, {"error": "overloaded"})
    cassette = tmp_path / "run.jsonl"

    recorded = run_session(RecordingProvider(http_provider(endpoint), cassette))
    requests_made = len(endpoint.requests)
    replay = ReplayProvider(cassette)
    replayed = run_session(replay)

    assert len(endpoint.requests) == requests_made  # replay made no HTTP call
    assert json.dumps(replayed, sort_keys=True) == json.dumps(recorded, sort_keys=True)
    assert recorded[0][1]["parsed"] == {"dv_mps": 4.25}
    replay.assert_exhausted()
    lines = cassette.read_text().splitlines()
    assert json.loads(lines[0]) == {"format": "orbitopt.llm-cassette", "version": 1,
                                    "provider": "openai_compat", "model": "fake-model"}
    assert len(lines) == 1 + 5


def test_replay_fails_loudly_on_an_unrecorded_request(tmp_path):
    cassette = tmp_path / "run.jsonl"
    RecordingProvider(MockProvider(["hello"]), cassette).generate(ASK, max_tokens=8)
    replay = ReplayProvider(cassette)
    with pytest.raises(ReplayMismatchError, match="was not recorded") as caught:
        replay.generate(ASK, max_tokens=9)
    assert not isinstance(caught.value, ProviderError)
    assert replay.generate(ASK, max_tokens=8).text == "hello"
    with pytest.raises(ReplayMismatchError):
        replay.generate(ASK, max_tokens=8)  # each recording is served once


def test_assert_exhausted_reports_unconsumed_entries(tmp_path):
    cassette = tmp_path / "run.jsonl"
    recorder = RecordingProvider(MockProvider(["a", "b"]), cassette)
    recorder.generate(ASK, max_tokens=8)
    recorder.generate(ASK, max_tokens=8)
    replay = ReplayProvider(cassette)
    assert replay.generate(ASK, max_tokens=8).text == "a"
    with pytest.raises(ReplayMismatchError, match="1 recorded entries were never requested"):
        replay.assert_exhausted()
    assert replay.generate(ASK, max_tokens=8).text == "b"
    replay.assert_exhausted()


def test_request_digest_ignores_dict_ordering_and_provider_config():
    reordered = {"additionalProperties": False, "required": ["dv_mps"],
                 "properties": {"dv_mps": {"type": "number"}}, "type": "object"}
    a = ChatRequest(tuple(ASK), 32, 0, response_schema=SCHEMA)
    b = ChatRequest(tuple(ASK), 32, 0.0, response_schema=reordered)
    assert a.digest() == b.digest()
    assert a.digest() != ChatRequest(tuple(ASK), 33, response_schema=SCHEMA).digest()


def test_cassette_never_contains_credentials(tmp_path, endpoint):
    endpoint.reply(payload=completion("ok"))
    cassette = tmp_path / "run.jsonl"
    RecordingProvider(http_provider(endpoint), cassette).generate(ASK, max_tokens=8)
    assert endpoint.requests[0]["headers"]["Authorization"] == f"Bearer {SECRET}"
    text = cassette.read_text()
    assert SECRET not in text and "Authorization" not in text and KEY_ENV not in text


def test_replay_rejects_an_unknown_cassette_version(tmp_path):
    cassette = tmp_path / "old.jsonl"
    cassette.write_text(json.dumps({"format": "orbitopt.llm-cassette", "version": 99}) + "\n")
    with pytest.raises(ValueError, match="not a orbitopt.llm-cassette v1 cassette"):
        ReplayProvider(cassette)
    with pytest.raises(ValueError):
        RecordingProvider(MockProvider([]), cassette)


# Usage metering.

def test_meter_blocks_the_call_that_would_exceed_max_calls():
    meter = UsageMeter(max_calls=2)
    mock = MockProvider(["a", "b", "c"], meter=meter)
    mock.generate(ASK, max_tokens=8)
    mock.generate(ASK, max_tokens=8)
    with pytest.raises(BudgetExceededError) as caught:
        mock.generate(ASK, max_tokens=8)
    assert len(mock.requests) == 2  # raised before the third call was made
    assert (caught.value.kind, caught.value.limit, caught.value.used) == ("calls", 2, 3)


def test_meter_raises_after_the_call_that_crosses_max_tokens_with_overrun():
    meter = UsageMeter(max_tokens=10)
    mock = MockProvider(["one two three", "four five six seven"], meter=meter)
    mock.generate([Message("user", "a b")], max_tokens=8)  # 2 + 3 = 5 tokens
    with pytest.raises(BudgetExceededError) as caught:
        mock.generate([Message("user", "a b")], max_tokens=8)  # 2 + 4 more = 11
    error = caught.value
    assert (error.kind, error.limit, error.used, error.overrun) == ("tokens", 10, 11, 1)
    assert error.response.text == "four five six seven"
    assert meter.usage.total == 11 and meter.calls == 2
    with pytest.raises(BudgetExceededError, match="token budget exhausted"):
        mock.generate([Message("user", "a b")], max_tokens=8)
    assert len(mock.requests) == 2


def test_meter_counts_repair_retries_as_calls():
    meter = UsageMeter()
    mock = MockProvider(["nope", '{"dv_mps": 2}'], meter=meter)
    response = mock.generate_structured(ASK, SCHEMA, max_tokens=8)
    assert meter.calls == 2 and meter.usage.total == response.usage.total


# Factory and import isolation.

def test_make_provider_builds_each_kind(tmp_path):
    assert isinstance(make_provider({"kind": "mock", "script": ["hi"]}), MockProvider)
    local = make_provider({"kind": "local", "model": "m", "base_url": "http://127.0.0.1:1/v1"})
    assert isinstance(local, LocalProvider) and local.api_key_env is None
    remote = make_provider({"kind": "openai_compat", "base_url": "https://example.invalid/v1", "model": "m"})
    assert remote.api_key_env == "OPENAI_API_KEY"
    cassette = tmp_path / "c.jsonl"
    recorder = make_provider({"kind": "mock", "script": ["hi"], "record": str(cassette)})
    assert isinstance(recorder, RecordingProvider)
    recorder.generate(ASK, max_tokens=8)
    assert isinstance(make_provider({"kind": "replay", "cassette": cassette}), ReplayProvider)
    with pytest.raises(ValueError, match="provider kind must be one of"):
        make_provider({"kind": "anthropic"})


CORE_PROBE = r"""
import json, sys
import orbitopt, orbitopt.cli, orbitopt.missions, orbitopt.scene_format, orbitopt.__main__
print(json.dumps(sorted(m for m in sys.modules if m.startswith("orbitopt.agent"))))
"""

AGENT_PROBE = r"""
import json, sys
import jsonschema
baseline = set(sys.modules)
import orbitopt.agent, orbitopt.agent.providers
foreign = sorted(m for m in set(sys.modules) - baseline
                 if m.split(".")[0] not in sys.stdlib_module_names
                 and m not in ("orbitopt", "orbitopt.agent") and not m.startswith("orbitopt.agent."))
print(json.dumps(foreign))
"""


def run_probe(source):
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    result = subprocess.run([sys.executable, "-c", source], env=env, cwd=REPO,
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_core_cli_imports_do_not_import_the_agent_package():
    assert run_probe(CORE_PROBE) == []


def test_agent_package_imports_only_stdlib_and_jsonschema():
    assert run_probe(AGENT_PROBE) == []
