"""Provider for any OpenAI-compatible ``/chat/completions`` endpoint, stdlib only.

The API key is read from an environment variable at call time. It is never
stored on the provider, shown in ``repr`` or included in an error message;
response bodies quoted in errors are redacted. Retries are bounded: only
HTTP 429, HTTP 5xx, timeouts and connection failures are retried, at most
``max_retries`` times with capped exponential backoff.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request

from orbitopt.agent.providers.base import (
    ChatProvider,
    ChatRequest,
    Message,
    ProviderError,
    ProviderResponse,
    ProviderUnavailableError,
    ToolCall,
    ToolCallError,
    Usage,
    UsageMeter,
    canonical_json,
)

_RETRYABLE_TRANSPORT = (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError)


def json_instruction(schema: dict) -> str:
    """System instruction used when the server cannot enforce a JSON Schema."""
    return ("Reply with only a JSON value, with no surrounding prose, that validates "
            f"against this JSON Schema:\n{canonical_json(schema)}")


_FUNCTION_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
"""What an OpenAI-compatible server accepts as a function name."""


def wire_name(name: str) -> str:
    """The function name sent for a tool named ``name``: each '.' becomes '-'.

    Tool ids such as ``math.erf.v1`` contain dots, which the function-name
    pattern rejects, and never contain '-', so the mapping is reversible for
    every tool id. Raises ``ProviderError`` for a name that still does not fit.
    """
    wire = name.replace(".", "-")
    if not _FUNCTION_NAME.match(wire):
        raise ProviderError(f"tool name {name!r} cannot be sent as a function name (1-64 of [a-zA-Z0-9_-])")
    return wire


def _wire_message(message: Message) -> dict:
    data = {"role": message.role, "content": message.content}
    if message.tool_calls:
        data["tool_calls"] = [
            {"id": call.id, "type": "function",
             "function": {"name": wire_name(call.name), "arguments": json.dumps(call.arguments, sort_keys=True)}}
            for call in message.tool_calls]
    if message.tool_call_id is not None:
        data["tool_call_id"] = message.tool_call_id
    return data


def _redact(text: str, secret: str | None) -> str:
    return text.replace(secret, "[REDACTED]") if secret else text


class OpenAICompatProvider(ChatProvider):
    """``POST {base_url}/chat/completions`` with stdlib ``urllib``.

    With ``supports_json_schema=True`` structured output is requested through
    ``response_format`` ``json_schema``; otherwise a system instruction asks
    for JSON. Either way the reply is validated locally. ``api_key_env=None``
    sends no ``Authorization`` header. The server must return a ``usage``
    block; a response without one is a ``ProviderError``, because silent zero
    usage would defeat budget enforcement. Tool names cross the wire
    through ``wire_name`` and come back mapped to the names offered.
    """

    name = "openai_compat"

    def __init__(self, base_url: str, model: str, *, api_key_env: str | None = "OPENAI_API_KEY",
                 timeout_s: float = 60.0, supports_json_schema: bool = True,
                 max_retries: int = 3, backoff_s: float = 1.0, max_backoff_s: float = 8.0,
                 meter: UsageMeter | None = None):
        super().__init__(meter=meter)
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        self.supports_json_schema = supports_json_schema
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.max_backoff_s = max_backoff_s

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(base_url={self.base_url!r}, model={self.model!r}, "
                f"api_key_env={self.api_key_env!r})")

    def _api_key(self) -> str | None:
        if self.api_key_env is None:
            return None
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ProviderError(f"{self.name}: environment variable {self.api_key_env} is not set")
        return key

    def request_body(self, request: ChatRequest) -> dict:
        """The JSON body sent for ``request`` (no credentials; those go in headers)."""
        messages = list(request.messages)
        body = {"model": self.model, "temperature": request.temperature,
                "max_tokens": request.max_tokens}
        if request.seed is not None:
            body["seed"] = request.seed
        if request.response_schema is not None:
            if self.supports_json_schema:
                body["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": "response", "schema": request.response_schema}}
            else:
                instruction = json_instruction(request.response_schema)
                if messages and messages[0].role == "system":
                    messages[0] = Message("system", f"{messages[0].content}\n\n{instruction}")
                else:
                    messages.insert(0, Message("system", instruction))
        if request.tools:
            _tool_names(request)
            body["tools"] = [{"type": "function", "function": {
                "name": wire_name(tool.name), "description": tool.description,
                "parameters": tool.parameters_schema}} for tool in request.tools]
        body["messages"] = [_wire_message(m) for m in messages]
        return body

    def complete(self, request):
        key = self._api_key()
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        data = json.dumps(self.request_body(request)).encode("utf-8")
        attempts = self.max_retries + 1
        last_error = ""
        for attempt in range(attempts):
            if attempt:
                time.sleep(min(self.backoff_s * 2 ** (attempt - 1), self.max_backoff_s))
            started = time.monotonic()
            try:
                http_request = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(http_request, timeout=self.timeout_s) as reply:
                    payload = reply.read()
            except urllib.error.HTTPError as exc:
                detail = _redact(exc.read(500).decode("utf-8", "replace"), key)
                if exc.code == 429 or exc.code >= 500:
                    last_error = f"HTTP {exc.code}: {detail}"
                    continue
                raise ProviderError(f"{self.name}: HTTP {exc.code} from {url}: {detail}") from None
            except _RETRYABLE_TRANSPORT as exc:
                last_error = _redact(f"{type(exc).__name__}: {exc}", key)
                continue
            return self._parse(payload, time.monotonic() - started, key, _tool_names(request))
        raise ProviderUnavailableError(
            f"{self.name}: POST {url} failed after {attempts} attempts; last error: {last_error}")

    def _parse(self, payload: bytes, latency_s: float, key: str | None, names: dict[str, str]) -> ProviderResponse:
        try:
            data = json.loads(payload)
            choice = data["choices"][0]
            message = choice["message"]
            usage = data["usage"]
            prompt = int(usage["prompt_tokens"])
            completion = int(usage.get("completion_tokens") or 0)
            total = int(usage.get("total_tokens") or prompt + completion)
            cost = usage.get("cost")
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            preview = _redact(payload[:300].decode("utf-8", "replace"), key)
            raise ProviderError(f"{self.name}: malformed response ({type(exc).__name__}: {exc}); "
                                f"body starts {preview!r}") from None
        usage = Usage(prompt, completion, total, None if cost is None else float(cost))
        calls = tuple(self._tool_call(index, raw, message.get("content"), names, usage)
                      for index, raw in enumerate(message.get("tool_calls") or ()))
        return ProviderResponse(
            text=message.get("content"), parsed=None, tool_calls=calls, usage=usage,
            model=data.get("model") or self.model, finish_reason=choice.get("finish_reason"),
            latency_s=latency_s, raw_id=data.get("id"))

    def _tool_call(self, index: int, raw: dict, text: str | None, names: dict[str, str], usage: Usage) -> ToolCall:
        function = raw.get("function") or {}
        name = function.get("name") or ""
        name = names.get(name, name)  # an unknown name stays as sent, for call_tools to reject
        arguments = function.get("arguments") or "{}"
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError as exc:
            parsed = exc
        if not isinstance(parsed, dict):
            raise ToolCallError(f"{self.name}: tool call arguments are not a JSON object",
                                raw_text=text, usage=usage,
                                errors=[f"{name}: arguments {str(arguments)[:200]!r} are not a JSON object"])
        return ToolCall(raw.get("id") or f"call_{index}", name, parsed)


def _tool_names(request: ChatRequest) -> dict[str, str]:
    """Wire name to offered tool name; ``ProviderError`` when two tools would share a wire name."""
    names = {wire_name(tool.name): tool.name for tool in request.tools}
    if len(names) != len(request.tools):
        raise ProviderError("two tool names map to the same function name: "
                            f"{sorted(tool.name for tool in request.tools)}")
    return names
