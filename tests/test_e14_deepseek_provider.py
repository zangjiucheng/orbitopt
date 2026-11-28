"""The DeepSeek preset against a local server that answers like DeepSeek (tests/fake_deepseek.py): no network."""
from __future__ import annotations

import json

import pytest
from fake_deepseek import FakeDeepSeek

from orbitopt import llm_env
from orbitopt.agent.providers import (DeepSeekProvider, Message, OpenAICompatProvider, ProviderAccountError,
                                      ProviderError, ProviderUnavailableError, RecordingProvider, ReplayProvider,
                                      make_provider)
from orbitopt.agent.providers.openai_compat import MAX_RETRY_AFTER_S, parse_usage

KEY = "sk-test-deepseek-not-a-key"
SCHEMA = {"type": "object", "required": ["x"], "properties": {"x": {"type": "number"}}}
ASK = [Message("system", "You propose a number."), Message("user", "Give x near 2.")]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("LLM_API_KEY", KEY)


@pytest.fixture
def server():
    replies = []
    fake = FakeDeepSeek(lambda body: replies.pop(0) if replies else '{"x": 2}', KEY)
    fake.replies = replies
    yield fake
    fake.close()


@pytest.fixture
def sleeps(monkeypatch):
    seen = []
    monkeypatch.setattr("orbitopt.agent.providers.openai_compat.time.sleep", seen.append)
    return seen


def deepseek(server, **kwargs):
    return DeepSeekProvider("deepseek-flash", server.base_url, **{"backoff_s": 0.5, "max_retries": 2, **kwargs})


def test_request_speaks_deepseek_json_mode_without_seed_and_with_thinking_off(server):
    reply = deepseek(server).generate_structured(ASK, SCHEMA, max_tokens=64, seed=7)
    sent = server.requests[0]
    body = sent["body"]
    assert sent["path"] == "/chat/completions" and sent["headers"]["Authorization"] == f"Bearer {KEY}"
    assert body["response_format"] == {"type": "json_object"} and body["thinking"] == {"type": "disabled"}
    assert "seed" not in body and body["temperature"] == 0.0 and body["max_tokens"] == 64
    assert '"x"' in body["messages"][0]["content"]  # the schema travels in the system instruction
    assert reply.parsed == {"x": 2} and reply.attempts == 1
    assert KEY not in json.dumps(body)


def test_the_generic_json_schema_mode_is_what_deepseek_rejects(server):
    generic = OpenAICompatProvider(server.base_url, "deepseek-flash", max_retries=0)
    with pytest.raises(ProviderError, match="HTTP 400.*response_format"):
        generic.generate_structured(ASK, SCHEMA, max_tokens=64)


def test_the_response_names_the_model_that_answered_not_the_alias_asked_for(server):
    reply = DeepSeekProvider("deepseek-chat", server.base_url).generate(ASK, max_tokens=16)
    assert server.requests[0]["body"]["model"] == "deepseek-chat"
    assert reply.model == "deepseek-flash"


def test_usage_carries_cache_hits_and_misses(server):
    provider = deepseek(server)
    first = provider.generate_structured(ASK, SCHEMA, max_tokens=64)
    second = provider.generate_structured(ASK[:1] + [Message("user", "Give x near 3.")], SCHEMA, max_tokens=64)
    assert first.usage.cache_hit_tokens == 0 and first.usage.cache_miss_tokens == first.usage.prompt_tokens
    assert second.usage.cache_hit_tokens > 0  # the shared system prompt
    assert second.usage.cache_hit_tokens + second.usage.cache_miss_tokens == second.usage.prompt_tokens
    assert (first.usage + second.usage).cache_hit_tokens == second.usage.cache_hit_tokens


def test_parse_usage_reads_nested_details_and_derives_the_miss():
    usage = parse_usage({"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
                         "prompt_tokens_details": {"cached_tokens": 6},
                         "completion_tokens_details": {"reasoning_tokens": 3}})
    assert (usage.cache_hit_tokens, usage.cache_miss_tokens, usage.reasoning_tokens) == (6, 4, 3)
    plain = parse_usage({"prompt_tokens": 10, "completion_tokens": 4})
    assert plain.cache_hit_tokens is None and plain.to_dict() == {"prompt_tokens": 10, "completion_tokens": 4,
                                                                  "total": 14, "cost_usd": None}


def test_blank_json_mode_content_is_repaired_and_both_calls_are_counted(server):
    server.replies += ["   ", '{"x": 2.5}']
    reply = deepseek(server).generate_structured(ASK, SCHEMA, max_tokens=64)
    assert len(server.requests) == 2 and reply.parsed == {"x": 2.5}
    assert reply.usage.prompt_tokens > len(" ".join(m.content for m in ASK).split())  # two prompts summed
    assert "did not validate" in server.requests[1]["body"]["messages"][-1]["content"]


def test_429_honours_retry_after_and_records_the_attempts(server, sleeps):
    server.fail(429, {"Retry-After": "3"})
    reply = deepseek(server).generate(ASK, max_tokens=16)
    assert reply.attempts == 2 and sleeps == [3.0]


def test_retry_after_is_capped_and_retries_are_bounded(server, sleeps):
    server.fail(429, {"Retry-After": "86400"}, times=5)
    with pytest.raises(ProviderUnavailableError, match="after 3 attempts"):
        deepseek(server).generate(ASK, max_tokens=16)
    assert len(server.requests) == 3 and sleeps == [MAX_RETRY_AFTER_S, MAX_RETRY_AFTER_S]


@pytest.mark.parametrize("status", [500, 503, 504])
def test_server_errors_are_retried_then_succeed(server, sleeps, status):
    server.fail(status)
    assert deepseek(server).generate(ASK, max_tokens=16).attempts == 2
    assert sleeps == [0.5]


def test_a_reply_cut_for_lack_of_resources_is_retried(server, sleeps):
    server.replies += [("", "insufficient_system_resource"), '{"x": 2}']
    reply = deepseek(server).generate_structured(ASK, SCHEMA, max_tokens=16)
    assert reply.attempts == 2 and reply.parsed == {"x": 2}


@pytest.mark.parametrize("status, error", [(402, ProviderAccountError), (400, ProviderError), (422, ProviderError)])
def test_client_errors_fail_at_once(server, sleeps, status, error):
    server.fail(status)
    with pytest.raises(error, match=f"HTTP {status}"):
        deepseek(server).generate(ASK, max_tokens=16)
    assert len(server.requests) == 1 and sleeps == []


def test_a_wrong_key_is_an_account_error_and_never_echoed(server, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-wrong")
    with pytest.raises(ProviderAccountError) as caught:
        deepseek(server).generate(ASK, max_tokens=16)
    assert "HTTP 401" in str(caught.value) and "sk-wrong" not in str(caught.value)


def test_the_cassette_keeps_the_answering_model_cache_usage_and_attempts(server, sleeps, tmp_path):
    server.fail(503)
    cassette = tmp_path / "c.jsonl"
    live = RecordingProvider(DeepSeekProvider("deepseek-chat", server.base_url, backoff_s=0.0), cassette)
    reply = live.generate_structured(ASK, SCHEMA, max_tokens=64, seed=3)
    entry = json.loads(cassette.read_text().splitlines()[1])["response"]
    assert entry["model"] == "deepseek-flash" and entry["attempts"] == 2
    assert {"cache_hit_tokens", "cache_miss_tokens", "reasoning_tokens"} <= entry["usage"].keys()
    assert KEY not in cassette.read_text()
    again = ReplayProvider(cassette).generate_structured(ASK, SCHEMA, max_tokens=64, seed=3)
    assert again.to_dict() == reply.to_dict()


def test_extra_body_cannot_replace_a_field_the_client_sets(server):
    provider = OpenAICompatProvider(server.base_url, "deepseek-flash", extra_body={"temperature": 1.0})
    with pytest.raises(ProviderError, match="extra_body may not replace"):
        provider.generate(ASK, max_tokens=16)


def test_llm_kind_deepseek_needs_only_a_model_and_the_key():
    env = {"LLM_KIND": "deepseek", "LLM_MODEL": "deepseek-flash", "LLM_API_KEY": KEY}
    spec = llm_env.provider_spec(env)
    assert spec == {"kind": "deepseek", "model": "deepseek-flash", "api_key_env": "LLM_API_KEY"}
    provider = llm_env.provider_from_env(env)
    assert isinstance(provider, DeepSeekProvider) and provider.base_url == "https://api.deepseek.com"
    assert isinstance(make_provider({**spec, "base_url": "http://127.0.0.1:9"}), DeepSeekProvider)
    with pytest.raises(ValueError, match="thinking"):
        DeepSeekProvider(thinking="sometimes")
