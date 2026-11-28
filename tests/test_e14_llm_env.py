"""One provider configuration for the campaign, the CLI and Mission Studio: the LLM_* variables and a .env reader.

No network and no real key: every key here is a made-up string, and the tests check it never reaches an output.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from orbitopt.agent.providers import LocalProvider, OpenAICompatProvider
from orbitopt import llm_env

BENCH = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCH))

FAKE_KEY = "sk-test-env-not-a-key"


def test_no_model_means_no_provider():
    assert llm_env.provider_spec({}) is None
    assert llm_env.provider_spec({"LLM_BASE_URL": "https://x.example/v1"}) is None
    assert llm_env.provider_spec({"LLM_MODEL": "m"}) is None  # openai_compat needs an endpoint


def test_the_spec_names_the_key_variable_never_the_key():
    env = {"LLM_MODEL": "m", "LLM_BASE_URL": "https://x.example/v1", "LLM_API_KEY": FAKE_KEY, "LLM_TIMEOUT_S": "90"}
    spec = llm_env.provider_spec(env)
    assert spec == {"kind": "openai_compat", "model": "m", "base_url": "https://x.example/v1",
                    "api_key_env": "LLM_API_KEY", "timeout_s": 90.0}
    assert FAKE_KEY not in repr(spec)
    assert not llm_env.key_missing(spec, env)
    assert llm_env.key_missing(spec, {k: v for k, v in env.items() if k != "LLM_API_KEY"})
    provider = llm_env.provider_from_env(env)
    assert isinstance(provider, OpenAICompatProvider) and provider.timeout_s == 90.0
    assert FAKE_KEY not in repr(provider)


def test_a_keyed_kind_without_its_key_is_not_a_provider():
    assert llm_env.provider_from_env({"LLM_MODEL": "m", "LLM_BASE_URL": "https://x.example/v1"}) is None


def test_local_needs_only_a_model_and_takes_no_key():
    spec = llm_env.provider_spec({"LLM_KIND": "local", "LLM_MODEL": "qwen"})
    assert spec == {"kind": "local", "model": "qwen"}
    assert isinstance(llm_env.provider_from_env({"LLM_KIND": "local", "LLM_MODEL": "qwen"}), LocalProvider)


@pytest.mark.parametrize("env, message", [({"LLM_KIND": "openai", "LLM_MODEL": "m"}, "LLM_KIND"),
                                          ({"LLM_KIND": "local", "LLM_MODEL": "m", "LLM_TIMEOUT_S": "soon"},
                                           "LLM_TIMEOUT_S")])
def test_a_bad_setting_is_an_error_not_a_silent_mock(env, message):
    with pytest.raises(llm_env.ProviderConfigError, match=message):
        llm_env.provider_spec(env)


def test_env_file_sets_only_what_is_unset_and_strips_one_pair_of_quotes(tmp_path):
    path = tmp_path / ".env"
    path.write_text("# provider\n\nLLM_MODEL = 'deepseek-flash'\nLLM_BASE_URL=\"https://api.example\"\n"
                    f"LLM_API_KEY={FAKE_KEY}\nLLM_KIND=local\nEMPTY=\n")
    env = {"LLM_KIND": "openai_compat"}
    assert llm_env.load_env_file(path, env) == ("LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY", "EMPTY")
    assert env == {"LLM_KIND": "openai_compat", "LLM_MODEL": "deepseek-flash", "LLM_BASE_URL": "https://api.example",
                   "LLM_API_KEY": FAKE_KEY, "EMPTY": ""}


@pytest.mark.parametrize("line", [f"export LLM_API_KEY={FAKE_KEY}", f"{FAKE_KEY}", "LLM_API_KEY"])
def test_env_file_refuses_anything_but_key_value_without_quoting_the_line(tmp_path, line):
    path = tmp_path / ".env"
    path.write_text(f"LLM_MODEL=m\n{line}\n")
    env: dict = {}
    with pytest.raises(ValueError) as caught:
        llm_env.load_env_file(path, env)
    assert ":2:" in str(caught.value) and FAKE_KEY not in str(caught.value)


def test_env_file_expands_nothing(tmp_path):
    path = tmp_path / ".env"
    path.write_text("LLM_MODEL=$HOME\nLLM_BASE_URL=`id`\n")
    env: dict = {}
    llm_env.load_env_file(path, env)
    assert env == {"LLM_MODEL": "$HOME", "LLM_BASE_URL": "`id`"}


def test_cli_provider_env_and_env_file(tmp_path, monkeypatch, capsys):
    from orbitopt.agent.commands import CommandError, build_provider
    from orbitopt.cli import main

    for name in llm_env.VARIABLES:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(CommandError, match="LLM_MODEL"):
        build_provider("env")
    path = tmp_path / ".env"
    path.write_text(f"LLM_MODEL=m\nLLM_BASE_URL=http://127.0.0.1:9/v1\nLLM_API_KEY={FAKE_KEY}\n")
    monkeypatch.setattr(os, "environ", dict(os.environ))  # main writes into os.environ; keep it local
    main(["--env-file", str(path), "missions"])
    assert os.environ["LLM_MODEL"] == "m"
    provider = build_provider("env")
    assert isinstance(provider, OpenAICompatProvider) and provider.api_key_env == "LLM_API_KEY"
    assert FAKE_KEY not in capsys.readouterr().err


def test_campaign_env_file_prints_a_count_never_a_value(tmp_path, monkeypatch, capsys):
    import bench_e14_end_to_end as E2E

    monkeypatch.setattr(os, "environ", dict(os.environ))
    for name in llm_env.VARIABLES:
        os.environ.pop(name, None)
    path = tmp_path / ".env"
    path.write_text(f"LLM_MODEL=m\nLLM_BASE_URL=http://127.0.0.1:9/v1\nLLM_API_KEY={FAKE_KEY}\n")
    assert E2E.main(["plan", "--env-file", str(path)]) == 0
    out = capsys.readouterr()
    assert "3 variables set" in out.err and FAKE_KEY not in out.out + out.err
    assert E2E.provider_record() == {"kind": "openai_compat", "model": "m", "base_url": "http://127.0.0.1:9/v1",
                                     "api_key_env": "LLM_API_KEY", "temperature": 0.0}
