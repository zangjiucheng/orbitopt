"""LLM provider abstraction: one interface for structured generation and tool calls.

Imports only the standard library and ``jsonschema``. See
``docs/agent/providers.md`` for configuration, recording and replay.
"""

from __future__ import annotations

from orbitopt.agent.providers.base import (
    BudgetExceededError,
    ChatProvider,
    ChatRequest,
    LLMProvider,
    Message,
    ProviderAccountError,
    ProviderError,
    ProviderResponse,
    ProviderUnavailableError,
    StructuredOutputError,
    ToolCall,
    ToolCallError,
    ToolSpec,
    Usage,
    UsageMeter,
)
from orbitopt.agent.providers.deepseek import DeepSeekProvider
from orbitopt.agent.providers.local import LocalProvider
from orbitopt.agent.providers.mock import (
    MockProvider,
    MockReply,
    RecordingProvider,
    ReplayMismatchError,
    ReplayProvider,
)
from orbitopt.agent.providers.openai_compat import OpenAICompatProvider

__all__ = [
    "BudgetExceededError", "ChatProvider", "ChatRequest", "DeepSeekProvider", "LLMProvider", "LocalProvider",
    "Message", "MockProvider", "MockReply", "OpenAICompatProvider", "ProviderAccountError", "ProviderError",
    "ProviderResponse", "ProviderUnavailableError", "RecordingProvider", "ReplayMismatchError",
    "ReplayProvider", "StructuredOutputError", "ToolCall", "ToolCallError", "ToolSpec", "Usage",
    "UsageMeter", "make_provider",
]

PROVIDER_KINDS = {
    "mock": MockProvider,
    "replay": ReplayProvider,
    "openai_compat": OpenAICompatProvider,
    "local": LocalProvider,
    "deepseek": DeepSeekProvider,
}


def make_provider(spec: dict) -> ChatProvider:
    """Build a provider from a config dict.

    ``kind`` selects the class in ``PROVIDER_KINDS``; the remaining keys are
    its constructor arguments. An optional ``record`` key names a cassette
    path and wraps the provider in a ``RecordingProvider``. Example:
    ``{"kind": "local", "model": "qwen2.5:7b", "record": "runs/x.jsonl"}``.
    """
    options = dict(spec)
    kind = options.pop("kind", None)
    record = options.pop("record", None)
    if kind not in PROVIDER_KINDS:
        raise ValueError(f"provider kind must be one of {sorted(PROVIDER_KINDS)}, got {kind!r}")
    provider = PROVIDER_KINDS[kind](**options)
    return RecordingProvider(provider, record) if record else provider
