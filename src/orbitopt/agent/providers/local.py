"""Provider preset for a local OpenAI-compatible server (Ollama, llama.cpp, vLLM)."""

from __future__ import annotations

from orbitopt.agent.providers.openai_compat import OpenAICompatProvider

DEFAULT_LOCAL_BASE_URL = "http://localhost:11434/v1"


class LocalProvider(OpenAICompatProvider):
    """``OpenAICompatProvider`` defaulting to a local server with no API key.

    Defaults differ only in configuration: Ollama's base URL, no
    ``Authorization`` header, a longer timeout for slow local inference, and
    instruct-and-validate structured output, which every server supports.
    """

    name = "local"

    def __init__(self, model: str, base_url: str = DEFAULT_LOCAL_BASE_URL, *,
                 timeout_s: float = 300.0, structured_output: str = "instruction", **kwargs):
        kwargs.setdefault("api_key_env", None)
        super().__init__(base_url, model, timeout_s=timeout_s, structured_output=structured_output, **kwargs)
