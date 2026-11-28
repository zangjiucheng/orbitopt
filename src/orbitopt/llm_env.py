"""The one provider configuration that the campaign, the CLI and Mission Studio read: the ``LLM_*`` variables.

    LLM_KIND       openai_compat (default) or local: a make_provider kind
    LLM_MODEL      the model id; without it no provider is configured
    LLM_BASE_URL   the endpoint; required for openai_compat, local defaults to Ollama
    LLM_API_KEY    the key (openai_compat); the provider reads it at call time and never stores it
    LLM_TIMEOUT_S  optional per-request timeout in seconds

Sampling settings (temperature, reply length) are not configured here: each caller fixes them, and the campaign
freezes them in its protocol. ``load_env_file`` reads a ``.env`` file into the environment for a caller that is
asked to; it never overrides a variable that is already set and never prints a value.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping, MutableMapping

KIND, MODEL, BASE_URL, API_KEY, TIMEOUT_S = "LLM_KIND", "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY", "LLM_TIMEOUT_S"
VARIABLES = (KIND, MODEL, BASE_URL, API_KEY, TIMEOUT_S)
KINDS = ("openai_compat", "local")
KEYED_KINDS = ("openai_compat",)
"""Kinds that send ``Authorization: Bearer $LLM_API_KEY``."""


class ProviderConfigError(ValueError):
    """The ``LLM_*`` variables are set but do not describe a provider (an unknown kind, a bad timeout)."""


def provider_spec(environ: Mapping[str, str] | None = None) -> dict | None:
    """The ``make_provider`` spec the ``LLM_*`` variables describe, or None when no provider is configured.

    The spec names the key variable (``api_key_env``), never the key, so it can be logged or recorded. Whether the
    key is set is ``key_missing``'s question.
    """
    env = os.environ if environ is None else environ
    model = env.get(MODEL, "").strip()
    kind = env.get(KIND, "").strip() or "openai_compat"
    base_url = env.get(BASE_URL, "").strip()
    if kind not in KINDS:
        raise ProviderConfigError(f"{KIND}={kind!r}: must be one of {list(KINDS)}")
    if not model or (kind == "openai_compat" and not base_url):
        return None
    spec: dict = {"kind": kind, "model": model}
    if base_url:
        spec["base_url"] = base_url
    if kind in KEYED_KINDS:
        spec["api_key_env"] = API_KEY
    timeout = env.get(TIMEOUT_S, "").strip()
    if timeout:
        try:
            spec["timeout_s"] = float(timeout)
        except ValueError:
            raise ProviderConfigError(f"{TIMEOUT_S}={timeout!r}: not a number of seconds") from None
    return spec


def key_missing(spec: Mapping, environ: Mapping[str, str] | None = None) -> bool:
    """True when ``spec`` needs a key and its variable is empty or unset."""
    env = os.environ if environ is None else environ
    name = spec.get("api_key_env")
    return bool(name) and not env.get(name)


def provider_from_env(environ: Mapping[str, str] | None = None):
    """The provider the ``LLM_*`` variables configure, or None when none is (a keyed kind without a key included)."""
    from orbitopt.agent.providers import make_provider

    spec = provider_spec(environ)
    if spec is None or key_missing(spec, environ):
        return None
    return make_provider(spec)


_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def load_env_file(path: str | Path, environ: MutableMapping[str, str] | None = None) -> tuple[str, ...]:
    """Set each ``KEY=VALUE`` line of ``path`` that the environment does not already set; return the names set.

    The format is deliberately small: blank lines and ``#`` comments are skipped, one pair of matching single or
    double quotes around a value is removed, and nothing is expanded or executed (no ``export``, no ``$VAR``). A
    line that is not ``KEY=VALUE`` raises ``ValueError`` naming its line number, never its text, because the file
    may hold a key.
    """
    env = os.environ if environ is None else environ
    added = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _LINE.match(line)
        if match is None:
            raise ValueError(f"{path}:{number}: not a KEY=VALUE line")
        name, value = match.groups()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if name not in env:
            env[name] = value
            added.append(name)
    return tuple(added)
