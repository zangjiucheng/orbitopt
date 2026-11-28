"""The append-only, hash-chained record of every tool call, and its JSON form.

Each ``CallRecord`` holds the tool id (which carries the version), the
canonical arguments, the canonical result or the error, the declared
evaluations charged, the wall time, the caller id and a sequence number. Its
``digest`` is ``canonical_digest`` of all of that plus the previous record's
digest, so editing, dropping or reordering any record breaks ``verify``.
``canonical_digest`` quantises numbers to 12 significant digits, so a digest
does not change with last-bit noise from another machine (the E12
rendezvous digests hashed exact float reprs and did). Arguments and results
are stored exactly: JSON float text round-trips a double bit for bit, so a
replay runs on exactly the recorded inputs.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

from orbitopt.optimization import canonical_digest

from .spec import REFUSALS, ToolErrorCode

LOG_FORMAT = "orbitopt-tool-calllog-1"
GENESIS = ""
"""``prev_digest`` of the first record."""

UNCHARGED = frozenset({ToolErrorCode.UNKNOWN_TOOL, ToolErrorCode.NOT_ALLOWED, ToolErrorCode.BUDGET_EXCEEDED})
"""Codes of calls refused before the budget was charged; every other call counts against its caller."""

Outcome = Literal["ok", "error", "refused"]


def to_json_value(value: Any) -> Any:
    """A JSON-native copy of ``value``; raise ``TypeError`` or ``ValueError`` if there is none.

    Mappings need string keys, tuples become lists, float and int subclasses
    (numpy scalars among them) become ``float`` and ``int``. Booleans stay
    booleans. NaN and infinities are rejected.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite number {value!r}")
        return float(value)
    if isinstance(value, Mapping):
        if not all(isinstance(k, str) for k in value):
            raise TypeError("mapping keys must be strings")
        return {k: to_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_value(v) for v in value]
    raise TypeError(f"{type(value).__name__} is not a JSON value")


@dataclass(frozen=True)
class CallRecord:
    """One logged call. ``error`` is ``None`` on success, else ``{"code", "message"}``."""

    seq: int
    tool_id: str
    caller: str
    args: Any
    result: dict | None
    error: dict | None
    evaluations: float
    wall_s: float
    prev_digest: str
    digest: str

    @property
    def code(self) -> ToolErrorCode | None:
        return None if self.error is None else ToolErrorCode(self.error["code"])

    @property
    def outcome(self) -> Outcome:
        """``ok``; ``error`` when the tool or its arguments failed; ``refused`` when policy stopped the call."""
        if self.error is None:
            return "ok"
        return "refused" if self.code in REFUSALS else "error"

    def to_dict(self) -> dict:
        return asdict(self)

    def body(self) -> dict:
        """Every field except ``digest``: what the digest covers."""
        data = self.to_dict()
        data.pop("digest")
        return data


def record_digest(body: dict) -> str:
    """The digest of a record body (every field but ``digest``)."""
    return canonical_digest(body)


class CallLog:
    """Append-only list of ``CallRecord``; safe to append from several threads."""

    def __init__(self, records: tuple[CallRecord, ...] | list[CallRecord] = ()):
        self._records: list[CallRecord] = []
        self._usage: dict[str, tuple[int, float]] = {}
        self._lock = threading.Lock()
        for record in records:
            self._add(record)

    def _add(self, record: CallRecord) -> None:
        self._records.append(record)
        if record.code not in UNCHARGED:
            calls, evaluations = self._usage.get(record.caller, (0, 0.0))
            self._usage[record.caller] = (calls + 1, evaluations + record.evaluations)

    @property
    def records(self) -> tuple[CallRecord, ...]:
        return tuple(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def append(self, *, tool_id: str, caller: str, args: Any, result: dict | None, error: dict | None,
               evaluations: float, wall_s: float) -> CallRecord:
        """Seal and append one record; ``args`` and ``result`` must already be JSON values."""
        with self._lock:
            body = {"seq": len(self._records), "tool_id": tool_id, "caller": caller, "args": args,
                    "result": result, "error": error, "evaluations": float(evaluations), "wall_s": float(wall_s),
                    "prev_digest": self._records[-1].digest if self._records else GENESIS}
            record = CallRecord(**body, digest=record_digest(body))
            self._add(record)
            return record

    def usage(self, caller: str) -> tuple[int, float]:
        """Calls charged to ``caller`` and the evaluations they declared.

        Every call except one refused for the allowlist or the budget is
        charged, whatever its outcome.
        """
        return self._usage.get(caller, (0, 0.0))

    def verify(self) -> list[str]:
        """Integrity problems: a sequence gap, a broken chain or a digest that does not match its record."""
        problems = []
        prev = GENESIS
        for i, record in enumerate(self._records):
            if record.seq != i:
                problems.append(f"record {i} carries sequence number {record.seq}")
            if record.prev_digest != prev:
                problems.append(f"record {i} does not chain to the record before it")
            if record_digest(record.body()) != record.digest:
                problems.append(f"record {i} does not match its digest")
            prev = record.digest
        return problems

    def to_json(self) -> str:
        return json.dumps({"format": LOG_FORMAT, "records": [r.to_dict() for r in self._records]},
                          sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> "CallLog":
        """Load a log written by ``to_json``; raise ``ValueError`` if the document is not one.

        Only the shape is checked here: every record has exactly the record
        fields and a known error code. Integrity is what ``verify`` and
        ``replay`` check.
        """
        data = json.loads(text)
        if not isinstance(data, dict) or data.get("format") != LOG_FORMAT:
            raise ValueError(f"not a {LOG_FORMAT} document")
        records = []
        for i, item in enumerate(data.get("records", ())):
            try:
                record = CallRecord(**item)
                if record.error is not None:
                    ToolErrorCode(record.error["code"])
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"record {i} is malformed: {exc}") from None
            records.append(record)
        return cls(records)


ReplayStatus = Literal["exact", "quantised", "within_tolerance", "not_executed", "mismatch"]


@dataclass(frozen=True)
class ReplayEntry:
    """How one record replayed.

    ``exact``: the result is bit for bit the recorded one. ``quantised``:
    equal at the 12 significant digits of the canonical form, as expected
    from another machine's libm. ``within_tolerance``: within the tool's
    declared ``replay_tolerance``. ``not_executed``: a refused call, logged
    but not re-run. ``mismatch``: anything else, explained in ``detail``.
    """

    seq: int
    tool_id: str
    status: ReplayStatus
    detail: str = ""


@dataclass(frozen=True)
class ReplayReport:
    """Integrity problems of the log plus one ``ReplayEntry`` per record."""

    integrity: tuple[str, ...]
    entries: tuple[ReplayEntry, ...]

    @property
    def mismatches(self) -> tuple[ReplayEntry, ...]:
        return tuple(e for e in self.entries if e.status == "mismatch")

    @property
    def ok(self) -> bool:
        return not self.integrity and not self.mismatches
