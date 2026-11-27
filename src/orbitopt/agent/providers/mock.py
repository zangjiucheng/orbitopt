"""Deterministic providers: scripted mock, cassette recorder and cassette replay.

Cassette format (``orbitopt.llm-cassette`` version 1) is JSON Lines, one
object per line, each serialized with sorted keys:

* line 1, the header: ``{"format": "orbitopt.llm-cassette", "version": 1,
  "provider": <name>, "model": <model or null>}``;
* every later line, one ``complete`` call in call order: ``{"digest",
  "request", "response"}``, or ``{"digest", "request", "error"}`` when the
  call raised a ``ProviderError`` (``error`` holds ``type``, ``message`` and,
  for output errors, ``raw_text`` and ``errors``).

``digest`` is ``request_digest(request)`` over the canonical request, which
contains messages, sampling settings, schema and tools but never the
endpoint, model or credentials, so a cassette holds no secret.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from orbitopt.agent.providers.base import (
    ChatProvider,
    ChatRequest,
    ProviderError,
    ProviderResponse,
    ProviderUnavailableError,
    StructuredOutputError,
    ToolCall,
    ToolCallError,
    UsageMeter,
    Usage,
    canonical_json,
    request_digest,
)

CASSETTE_FORMAT = "orbitopt.llm-cassette"
CASSETTE_VERSION = 1
_RECORDED_ERRORS = {cls.__name__: cls for cls in
                    (ProviderError, ProviderUnavailableError, StructuredOutputError, ToolCallError)}


@dataclass(frozen=True)
class MockReply:
    """A scripted model reply; the mock fills in usage, model and ids."""

    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


ScriptItem = Union[str, MockReply, ProviderResponse, Exception]


def mock_usage(request: ChatRequest, reply: MockReply) -> Usage:
    """The mock's token rule: whitespace-separated words.

    Prompt tokens are the words of every message content. Completion tokens
    are the words of the reply text plus, per tool call, the words of its name
    and of ``json.dumps(arguments, sort_keys=True)``.
    """
    prompt = sum(len((m.content or "").split()) for m in request.messages)
    completion = len((reply.text or "").split()) + sum(
        len(call.name.split()) + len(json.dumps(call.arguments, sort_keys=True).split())
        for call in reply.tool_calls)
    return Usage(prompt, completion, prompt + completion)


class MockProvider(ChatProvider):
    """Scripted deterministic provider for tests and offline CI.

    ``script`` is a sequence served in order, or a function of the
    ``ChatRequest``. Each item is a ``str`` (text reply), a ``MockReply``, a
    full ``ProviderResponse`` (returned as is) or an exception instance, which
    is raised to inject a failure, e.g. ``ProviderUnavailableError("timeout")``.
    Malformed JSON and unknown tool names are injected as ordinary replies.
    Running past the end of a sequence raises ``ProviderError``. Every request
    received is appended to ``requests``.
    """

    name = "mock"

    def __init__(self, script: Sequence[ScriptItem] | Callable[[ChatRequest], ScriptItem], *,
                 model: str = "mock", meter: UsageMeter | None = None):
        super().__init__(meter=meter)
        self.model = model
        self._respond = script if callable(script) else None
        self._queue = deque(() if callable(script) else script)
        self.requests: list[ChatRequest] = []

    def complete(self, request):
        self.requests.append(request)
        if self._respond is not None:
            item = self._respond(request)
        elif self._queue:
            item = self._queue.popleft()
        else:
            raise ProviderError(f"mock script exhausted after {len(self.requests) - 1} replies")
        if isinstance(item, Exception):
            raise item
        if isinstance(item, ProviderResponse):
            return item
        reply = MockReply(text=item) if isinstance(item, str) else item
        return ProviderResponse(
            text=reply.text, parsed=None, tool_calls=tuple(reply.tool_calls),
            usage=mock_usage(request, reply), model=self.model,
            finish_reason="tool_calls" if reply.tool_calls else "stop", latency_s=0.0,
            raw_id=f"mock-{len(self.requests)}")


class ReplayMismatchError(AssertionError):
    """A replayed run diverged from its cassette.

    Deliberately not a ``ProviderError``: code that handles provider failures
    must not swallow a replay divergence.
    """


def _error_record(exc: ProviderError) -> dict:
    record = {"type": type(exc).__name__, "message": str(exc)}
    if isinstance(exc, StructuredOutputError):
        record.update(raw_text=exc.raw_text, errors=list(exc.errors))
    return record


def _error_from_record(record: dict) -> ProviderError:
    cls = _RECORDED_ERRORS[record["type"]]
    if issubclass(cls, StructuredOutputError):
        return cls(record["message"], raw_text=record["raw_text"], errors=record["errors"])
    return cls(record["message"])


class RecordingProvider(ChatProvider):
    """Wraps a provider and appends every ``complete`` call to a JSONL cassette.

    Both responses and ``ProviderError`` failures are recorded, so a replay
    reproduces retries, repairs and outages exactly. Appending to an existing
    cassette requires a matching format version.
    """

    def __init__(self, inner: ChatProvider, cassette: str | Path, *,
                 meter: UsageMeter | None = None):
        super().__init__(meter=meter)
        self.inner = inner
        self.name = inner.name
        self.model = getattr(inner, "model", None)
        self.path = Path(cassette)
        self._lock = threading.Lock()
        if self.path.exists() and self.path.stat().st_size:
            _read_header(self.path)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._append({"format": CASSETTE_FORMAT, "version": CASSETTE_VERSION,
                          "provider": inner.name, "model": getattr(inner, "model", None)})

    def _append(self, record: dict) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(record) + "\n")

    def complete(self, request):
        record = {"digest": request.digest(), "request": request.to_dict()}
        try:
            response = self.inner.complete(request)
        except ProviderError as exc:
            self._append({**record, "error": _error_record(exc)})
            raise
        self._append({**record, "response": response.to_dict()})
        return response


def _read_header(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        header = json.loads(handle.readline())
    if header.get("format") != CASSETTE_FORMAT or header.get("version") != CASSETTE_VERSION:
        raise ValueError(f"{path}: not a {CASSETTE_FORMAT} v{CASSETTE_VERSION} cassette "
                         f"(header {header})")
    return header


class ReplayProvider(ChatProvider):
    """Serves a cassette back without any model call.

    Each request is matched by digest; identical requests are served in
    recorded order. A request with no remaining recorded entry raises
    ``ReplayMismatchError``; ``assert_exhausted`` raises it when recorded
    entries were never requested. ``name`` and ``model`` are the recorded
    provider's, from the cassette header, so a replayed run names the
    provider that answered it.
    """

    name = "replay"

    def __init__(self, cassette: str | Path, *, meter: UsageMeter | None = None):
        super().__init__(meter=meter)
        self.path = Path(cassette)
        self.header = _read_header(self.path)
        self.name = self.header.get("provider") or ReplayProvider.name
        self.model = self.header.get("model")
        self._entries: dict[str, deque] = {}
        with self.path.open(encoding="utf-8") as handle:
            lines = handle.read().splitlines()[1:]
        for number, line in enumerate(lines, start=2):
            if not line.strip():
                continue
            entry = json.loads(line)
            digest = request_digest(entry["request"])
            if digest != entry["digest"]:
                raise ValueError(f"{self.path}:{number}: digest does not match its request")
            if ("response" in entry) == ("error" in entry):
                raise ValueError(f"{self.path}:{number}: entry needs exactly one of response/error")
            self._entries.setdefault(digest, deque()).append(entry)
        self._lock = threading.Lock()

    def complete(self, request):
        digest = request.digest()
        with self._lock:
            queue = self._entries.get(digest)
            if not queue:
                last = request.messages[-1].content if request.messages else None
                raise ReplayMismatchError(
                    f"{self.path}: request {digest} was not recorded (or all its recordings "
                    f"were served); last message: {(last or '')[:200]!r}")
            entry = queue.popleft()
        if "error" in entry:
            raise _error_from_record(entry["error"])
        return ProviderResponse.from_dict(entry["response"])

    def remaining(self) -> int:
        """Number of recorded entries not yet served."""
        with self._lock:
            return sum(len(queue) for queue in self._entries.values())

    def assert_exhausted(self) -> None:
        """Raise ``ReplayMismatchError`` if any recorded entry was never served."""
        left = self.remaining()
        if left:
            raise ReplayMismatchError(f"{self.path}: {left} recorded entries were never requested")
