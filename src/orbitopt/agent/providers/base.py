"""Provider-neutral request and response types, typed errors and usage metering.

Every provider reduces to one primitive, ``ChatProvider.complete``, which
takes a canonical ``ChatRequest`` and returns a raw ``ProviderResponse``.
``generate``, ``generate_structured`` and ``call_tools`` are implemented once
on top of it, so schema validation, the structured-output repair retry and
tool-call checks behave the same for the mock, replay and HTTP providers.
Recording and replay also work at that primitive, which makes a replayed run
go through exactly the validation path the recorded run went through.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

import jsonschema

ROLES = ("system", "user", "assistant", "tool")
STRUCTURED_REPAIR_RETRIES = 1


class ProviderError(Exception):
    """A provider call failed for a reason that retrying will not fix."""


class ProviderUnavailableError(ProviderError):
    """The provider could not be reached or kept failing after bounded retries."""


class StructuredOutputError(ProviderError):
    """The model output did not validate, even after the repair retry.

    ``raw_text`` is the last raw reply and ``errors`` the validation errors
    against it. No unvalidated data is ever returned in place of this error.
    """

    def __init__(self, message: str, *, raw_text: str | None, errors: Sequence[str]):
        super().__init__(message)
        self.raw_text = raw_text
        self.errors = tuple(errors)


class ToolCallError(StructuredOutputError):
    """The model called an unknown tool or passed arguments that fail its schema."""


class BudgetExceededError(ProviderError):
    """A ``UsageMeter`` limit was reached.

    ``kind`` is ``"calls"`` or ``"tokens"``; ``overrun`` is ``used - limit``.
    For a token overrun, ``response`` holds the call that crossed the limit.
    """

    def __init__(self, message: str, *, kind: str, limit: int, used: int,
                 response: ProviderResponse | None = None):
        super().__init__(message)
        self.kind = kind
        self.limit = limit
        self.used = used
        self.overrun = used - limit
        self.response = response


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, data: dict) -> ToolCall:
        return cls(data["id"], data["name"], dict(data["arguments"]))


@dataclass(frozen=True)
class Message:
    """One chat message; ``tool_call_id`` links a ``tool`` reply to its call."""

    role: str
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

    def __post_init__(self):
        if self.role not in ROLES:
            raise ValueError(f"message role must be one of {ROLES}, got {self.role!r}")
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("a tool message needs the tool_call_id it answers")
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))

    def to_dict(self) -> dict:
        data = {"role": self.role, "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        return data


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may call; ``parameters_schema`` is a JSON Schema object."""

    name: str
    description: str
    parameters_schema: dict

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description,
                "parameters_schema": self.parameters_schema}


@dataclass(frozen=True)
class Usage:
    """Token counts for one or more calls; ``cost_usd`` is None when unknown."""

    prompt_tokens: int
    completion_tokens: int
    total: int
    cost_usd: float | None = None

    def __add__(self, other: Usage) -> Usage:
        costs = [c for c in (self.cost_usd, other.cost_usd) if c is not None]
        return Usage(self.prompt_tokens + other.prompt_tokens,
                     self.completion_tokens + other.completion_tokens,
                     self.total + other.total,
                     sum(costs) if costs else None)

    def to_dict(self) -> dict:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "total": self.total, "cost_usd": self.cost_usd}


@dataclass(frozen=True)
class ProviderResponse:
    """The result of one provider call.

    ``parsed`` is set only by ``generate_structured`` and only after the value
    validated against the requested schema.
    """

    text: str | None
    parsed: Any
    tool_calls: tuple[ToolCall, ...]
    usage: Usage
    model: str
    finish_reason: str | None
    latency_s: float
    raw_id: str | None

    def to_dict(self) -> dict:
        return {"text": self.text, "parsed": self.parsed,
                "tool_calls": [call.to_dict() for call in self.tool_calls],
                "usage": self.usage.to_dict(), "model": self.model,
                "finish_reason": self.finish_reason, "latency_s": self.latency_s,
                "raw_id": self.raw_id}

    @classmethod
    def from_dict(cls, data: dict) -> ProviderResponse:
        return cls(text=data["text"], parsed=data["parsed"],
                   tool_calls=tuple(ToolCall.from_dict(c) for c in data["tool_calls"]),
                   usage=Usage(**data["usage"]), model=data["model"],
                   finish_reason=data["finish_reason"], latency_s=data["latency_s"],
                   raw_id=data["raw_id"])


def canonical_json(value: Any) -> str:
    """Serialize with sorted keys and no insignificant whitespace."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def request_digest(request_dict: dict) -> str:
    """Digest of a canonical request dict; independent of dict ordering."""
    return "sha256:" + hashlib.sha256(canonical_json(request_dict).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ChatRequest:
    """The canonical, provider-neutral request a provider's ``complete`` receives.

    It holds no provider configuration (endpoint, model, credentials), so its
    digest is stable across providers and never covers a secret.
    """

    messages: tuple[Message, ...]
    max_tokens: int
    temperature: float = 0.0
    seed: int | None = None
    response_schema: dict | None = None
    tools: tuple[ToolSpec, ...] = ()

    def to_dict(self) -> dict:
        return {"messages": [m.to_dict() for m in self.messages],
                "max_tokens": self.max_tokens, "temperature": float(self.temperature),
                "seed": self.seed, "response_schema": self.response_schema,
                "tools": [t.to_dict() for t in self.tools]}

    def digest(self) -> str:
        return request_digest(self.to_dict())


def schema_errors(value: Any, schema: dict) -> list[str]:
    """Validation errors of ``value`` against ``schema``, sorted for determinism."""
    validator = jsonschema.validators.validator_for(schema)(schema)
    found = [f"{'/'.join(map(str, e.absolute_path)) or '$'}: {e.message}"
             for e in validator.iter_errors(value)]
    return sorted(found)


def parse_structured(text: str | None, schema: dict) -> tuple[Any, list[str]]:
    """Parse a JSON reply (one surrounding code fence allowed) and validate it."""
    body = (text or "").strip()
    if body.startswith("```") and body.endswith("```") and "\n" in body:
        body = body[body.index("\n") + 1:-3].strip()
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        return None, [f"$: reply is not valid JSON ({exc.msg} at char {exc.pos})"]
    return value, schema_errors(value, schema)


def tool_call_errors(calls: Sequence[ToolCall], tools: dict[str, ToolSpec]) -> list[str]:
    """Errors for calls to unknown tools or with arguments that fail the tool schema."""
    errors = []
    for call in calls:
        spec = tools.get(call.name)
        if spec is None:
            errors.append(f"{call.id}: unknown tool {call.name!r}; offered {sorted(tools)}")
            continue
        errors.extend(f"{call.id} ({call.name}) {error}"
                      for error in schema_errors(call.arguments, spec.parameters_schema))
    return errors


def repair_prompt(errors: Sequence[str]) -> str:
    listed = "\n".join(f"- {error}" for error in errors)
    return ("Your previous reply did not validate against the required JSON Schema:\n"
            f"{listed}\nReply again with only the corrected JSON value.")


class UsageMeter:
    """Accumulates usage across calls and enforces optional limits.

    ``before_call`` reserves a call slot and raises ``BudgetExceededError``
    before a call that would exceed ``max_calls`` (or when ``max_tokens`` is
    already spent). ``record`` adds a call's usage and raises after the call
    whose tokens crossed ``max_tokens``, reporting the overrun. The meter is
    thread-safe; it supplies accounting, not policy.
    """

    def __init__(self, *, max_tokens: int | None = None, max_calls: int | None = None):
        self.max_tokens = max_tokens
        self.max_calls = max_calls
        self.calls = 0
        self.usage = Usage(0, 0, 0)
        self._lock = threading.Lock()

    def before_call(self) -> None:
        with self._lock:
            if self.max_calls is not None and self.calls >= self.max_calls:
                raise BudgetExceededError(
                    f"call budget exhausted: {self.calls} of {self.max_calls} calls used",
                    kind="calls", limit=self.max_calls, used=self.calls + 1)
            if self.max_tokens is not None and self.usage.total >= self.max_tokens:
                raise BudgetExceededError(
                    f"token budget exhausted: {self.usage.total} of {self.max_tokens} tokens used",
                    kind="tokens", limit=self.max_tokens, used=self.usage.total)
            self.calls += 1

    def record(self, usage: Usage, response: ProviderResponse | None = None) -> None:
        with self._lock:
            self.usage = self.usage + usage
            if self.max_tokens is not None and self.usage.total > self.max_tokens:
                raise BudgetExceededError(
                    f"token budget exceeded: {self.usage.total} of {self.max_tokens} tokens "
                    f"used (overrun {self.usage.total - self.max_tokens})",
                    kind="tokens", limit=self.max_tokens, used=self.usage.total,
                    response=response)


@runtime_checkable
class LLMProvider(Protocol):
    """What the agent runtime and the compiler depend on."""

    name: str

    def generate(self, messages: Sequence[Message], *, max_tokens: int,
                 temperature: float = 0.0, seed: int | None = None) -> ProviderResponse: ...

    def generate_structured(self, messages: Sequence[Message], schema: dict, *,
                            max_tokens: int, temperature: float = 0.0,
                            seed: int | None = None) -> ProviderResponse: ...

    def call_tools(self, messages: Sequence[Message], tools: Sequence[ToolSpec], *,
                   max_tokens: int, temperature: float = 0.0,
                   seed: int | None = None) -> ProviderResponse: ...


class ChatProvider:
    """Base class implementing ``LLMProvider`` on top of ``complete``.

    Subclasses implement ``complete``. An optional ``meter`` counts every
    ``complete`` call this provider makes, repair retries included.
    """

    name = "provider"

    def __init__(self, *, meter: UsageMeter | None = None):
        self.meter = meter

    def complete(self, request: ChatRequest) -> ProviderResponse:
        """Send one canonical request and return the raw response (no validation)."""
        raise NotImplementedError

    def _send(self, request: ChatRequest) -> ProviderResponse:
        if self.meter is not None:
            self.meter.before_call()
        response = self.complete(request)
        if self.meter is not None:
            self.meter.record(response.usage, response)
        return response

    def generate(self, messages, *, max_tokens, temperature=0.0, seed=None):
        """Plain text generation."""
        return self._send(ChatRequest(tuple(messages), max_tokens, temperature, seed))

    def generate_structured(self, messages, schema, *, max_tokens, temperature=0.0, seed=None):
        """Generate a JSON value that validates against ``schema``.

        A reply that fails to parse or validate is sent back with its errors
        once (``STRUCTURED_REPAIR_RETRIES``). The returned response carries the
        validated ``parsed`` value and the usage and latency summed over all
        attempts. Persistent failure raises ``StructuredOutputError``.
        """
        jsonschema.validators.validator_for(schema).check_schema(schema)
        messages = tuple(messages)
        attempts = []
        for _ in range(1 + STRUCTURED_REPAIR_RETRIES):
            response = self._send(ChatRequest(messages, max_tokens, temperature, seed,
                                              response_schema=schema))
            attempts.append(response)
            value, errors = parse_structured(response.text, schema)
            if not errors:
                total = attempts[0].usage
                for earlier in attempts[1:]:
                    total = total + earlier.usage
                return replace(response, parsed=value, usage=total,
                               latency_s=sum(a.latency_s for a in attempts))
            messages += (Message("assistant", response.text), Message("user", repair_prompt(errors)))
        raise StructuredOutputError(
            f"{self.name}: structured output failed validation after {len(attempts)} attempts",
            raw_text=response.text, errors=errors)

    def call_tools(self, messages, tools, *, max_tokens, temperature=0.0, seed=None):
        """Offer ``tools`` to the model and return its reply with validated tool calls.

        A reply may contain text and no tool calls. A call to a tool that was
        not offered, or with arguments failing that tool's parameters schema,
        raises ``ToolCallError``.
        """
        tools = tuple(tools)
        by_name = {tool.name: tool for tool in tools}
        if len(by_name) != len(tools):
            raise ValueError("tool names must be unique")
        for tool in tools:
            jsonschema.validators.validator_for(tool.parameters_schema).check_schema(
                tool.parameters_schema)
        response = self._send(ChatRequest(tuple(messages), max_tokens, temperature, seed,
                                          tools=tools))
        errors = tool_call_errors(response.tool_calls, by_name)
        if errors:
            raise ToolCallError(f"{self.name}: invalid tool calls", raw_text=response.text,
                                errors=errors)
        return response
