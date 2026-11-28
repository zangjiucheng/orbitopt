"""Provider preset for DeepSeek's OpenAI-compatible API (https://api-docs.deepseek.com, read 2026-11-28).

Where DeepSeek differs from the OpenAI dialect ``OpenAICompatProvider`` speaks by default:

- ``response_format`` takes ``text`` or ``json_object`` only, so the schema goes in a system instruction and the
  reply is validated locally, with the usual repair round (``structured_output="json_object"``). JSON mode may
  answer with blank content; that fails validation and is repaired like any invalid reply.
- Thinking is on by default, and in thinking mode ``temperature`` has no effect. The preset turns it off
  (``thinking: {"type": "disabled"}``) so that the temperature a caller sets is honoured. Thinking with tools would
  also require every earlier ``reasoning_content`` to be sent back, which this client does not do.
- No ``seed`` parameter is documented, so none is sent; the seed stays in the canonical request and its digest.
- ``usage`` reports ``prompt_cache_hit_tokens`` and ``prompt_cache_miss_tokens``, which ``parse_usage`` reads.
- 429 (rate or concurrency limit), 500, 503 and 504 are retried with backoff; 401 (key) and 402 (balance) raise
  ``ProviderAccountError``; 400 and 422 fail at once. A non-streaming reply may start with keep-alive blank lines,
  which JSON parsing skips. ``finish_reason`` ``insufficient_system_resource`` is retried.
- The response's ``model`` names the model that answered; an alias such as ``deepseek-chat`` answers as
  ``deepseek-flash``.
"""

from __future__ import annotations

from orbitopt.agent.providers.openai_compat import OpenAICompatProvider

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-flash"


class DeepSeekProvider(OpenAICompatProvider):
    """``OpenAICompatProvider`` with DeepSeek's endpoint, JSON mode, no seed and thinking off (``thinking``)."""

    name = "deepseek"

    def __init__(self, model: str = DEEPSEEK_MODEL, base_url: str = DEEPSEEK_BASE_URL, *,
                 thinking: str = "disabled", timeout_s: float = 120.0, **kwargs):
        if thinking not in ("disabled", "enabled"):
            raise ValueError(f"thinking must be 'disabled' or 'enabled', got {thinking!r}")
        kwargs.setdefault("api_key_env", "LLM_API_KEY")
        super().__init__(base_url, model, timeout_s=timeout_s, structured_output="json_object", send_seed=False,
                         extra_body={"thinking": {"type": thinking}}, **kwargs)
