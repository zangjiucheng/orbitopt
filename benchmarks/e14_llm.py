"""The provider arms' plumbing: per-cell caps, the campaign-wide hard stop, resumable cassettes and token cost.

A provider cell talks to the model through one stack, outermost first:

    ResumingProvider   cassette: serves every call the cell's cassette already holds, records every new one
      (meter)          CellMeter: the spec's per-cell call cap and token budget, checked before every call
    GuardedProvider    Ledger: the campaign-wide hard stop on calls and estimated spend, checked before every
                       call that would reach the network, shared by every process writing one output directory
    live provider      e.g. DeepSeekProvider, with its own bounded HTTP retries

A call that a cassette already holds never reaches the guard or the network, so rerunning an interrupted shard
repeats no paid call. The replay check serves the same cell from its cassette through ``ReplayProvider`` with a
fresh ``CellMeter``.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

PRICE_SOURCE = "https://api-docs.deepseek.com/quick_start/pricing, read 2026-11-28"
PRICES_USD_PER_MTOK = {
    "deepseek-flash": {"cache_hit": 0.006, "cache_miss": 0.30, "output": 1.20},
    "deepseek-v4-pro": {"cache_hit": 0.044, "cache_miss": 1.32, "output": 3.96},
}
"""DeepSeek's list (peak) prices. Off-peak prices are half; a cost here is the peak, so it never under-states."""

DEFAULT_MAX_CALLS = 40
DEFAULT_MAX_USD = 0.50
ENV_MAX_CALLS, ENV_MAX_USD = "LLM_CAMPAIGN_MAX_CALLS", "LLM_CAMPAIGN_MAX_USD"
CHARS_PER_TOKEN_FLOOR = 2.0
"""Reservation rule: a prompt is charged as at most one token per two characters of its request body, all at the
cache-miss price, plus ``max_tokens`` at the output price. Real prompts run about 3 to 4 characters per token."""


class CampaignHardStop(BaseException):
    """The campaign must stop: a hard stop was reached or the account was refused.

    A ``BaseException``, so that no ``except Exception`` in a cell, the compiler or the runtime can record it as a
    cell's own failure: the shard is left unwritten and a rerun resumes it from its cassettes.
    """


def cost_usd(usage, model: str | None) -> float | None:
    """The list price of one call's usage, or None when the model has no pinned price."""
    price = PRICES_USD_PER_MTOK.get(model or "")
    if price is None:
        return None
    hit = usage.cache_hit_tokens or 0
    miss = usage.cache_miss_tokens if usage.cache_miss_tokens is not None else usage.prompt_tokens - hit
    return (hit * price["cache_hit"] + miss * price["cache_miss"]
            + usage.completion_tokens * price["output"]) / 1e6


def limits_from(max_calls: int | None, max_usd: float | None, environ=os.environ) -> tuple[int, float]:
    """The hard stop: an explicit argument, else the environment, else the conservative default."""
    calls = max_calls if max_calls is not None else int(environ.get(ENV_MAX_CALLS) or DEFAULT_MAX_CALLS)
    usd = max_usd if max_usd is not None else float(environ.get(ENV_MAX_USD) or DEFAULT_MAX_USD)
    return calls, usd


# --------------------------------------------------------------------------------------------- the hard stop
class Ledger:
    """Calls and estimated spend of the whole campaign, in an append-only JSON Lines file under an exclusive lock.

    ``reserve`` adds a call's worst-case cost before it is made and refuses one that would pass either limit;
    ``settle`` replaces the reservation by the call's priced usage. A reservation whose process died is never
    settled and stays counted, so the totals only ever over-state what was spent.
    """

    def __init__(self, path: Path, *, max_calls: int, max_usd: float, model: str):
        if model not in PRICES_USD_PER_MTOK:
            raise ValueError(f"no pinned price for model {model!r}: the hard stop cannot estimate spend")
        self.path, self.max_calls, self.max_usd, self.model = Path(path), max_calls, max_usd, model
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch()
        self._offset = 0
        self._open: dict[str, float] = {}
        self._settled = {"calls": 0, "usd": 0.0, "attempts": 0, "tokens": 0}

    def _refresh(self, handle) -> None:
        handle.seek(self._offset)
        for line in handle.read().splitlines():
            entry = json.loads(line)
            if entry["op"] == "reserve":
                self._open[entry["id"]] = entry["usd"]
            else:
                self._open.pop(entry["id"], None)
                for k in self._settled:
                    self._settled[k] += entry[k]
        self._offset = handle.tell()

    def _locked(self, write: dict | None = None, check: float | None = None) -> dict:
        with self.path.open("r+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                self._refresh(handle)
                totals = self.totals_unlocked()
                if check is not None and (totals["calls"] + 1 > self.max_calls
                                          or totals["usd"] + check > self.max_usd):
                    raise CampaignHardStop(
                        f"hard stop: {totals['calls']} calls and ${totals['usd']:.4f} committed; the next call "
                        f"(up to ${check:.4f}) would pass {self.max_calls} calls or ${self.max_usd:.2f}")
                if write is not None:
                    handle.seek(0, os.SEEK_END)
                    handle.write(json.dumps(write, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    self._refresh(handle)
                return self.totals_unlocked()
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def totals_unlocked(self) -> dict:
        return {"calls": self._settled["calls"] + len(self._open),
                "usd": self._settled["usd"] + sum(self._open.values()),
                "attempts": self._settled["attempts"], "tokens": self._settled["tokens"],
                "open": len(self._open)}

    def totals(self) -> dict:
        return self._locked()

    def worst_case_usd(self, request) -> float:
        price = PRICES_USD_PER_MTOK[self.model]
        prompt = len(json.dumps(request.to_dict())) / CHARS_PER_TOKEN_FLOOR
        return (prompt * price["cache_miss"] + request.max_tokens * price["output"]) / 1e6

    def reserve(self, request) -> str:
        rid = uuid.uuid4().hex
        usd = self.worst_case_usd(request)
        self._locked({"op": "reserve", "id": rid, "usd": usd, "utc": time.time()}, check=usd)
        return rid

    def settle(self, rid: str, response=None) -> None:
        usage = None if response is None else response.usage
        usd = 0.0 if usage is None else cost_usd(usage, response.model)
        if usd is None:  # an unpriced model answered: charge it at the configured model's price
            usd = cost_usd(usage, self.model)
        self._locked({"op": "settle", "id": rid, "calls": 1, "usd": usd,
                      "attempts": 1 if response is None else response.attempts,
                      "tokens": 0 if usage is None else usage.total,
                      "model": None if response is None else response.model, "utc": time.time()})


def guarded(live, ledger: Ledger):
    """``live`` behind the hard stop: every network call is reserved first and settled after."""
    from orbitopt.agent.providers import ChatProvider, ProviderAccountError, ProviderError

    class GuardedProvider(ChatProvider):
        name = live.name
        model = getattr(live, "model", None)

        def complete(self, request):
            rid = ledger.reserve(request)
            try:
                response = live.complete(request)
            except ProviderAccountError as exc:
                ledger.settle(rid)
                raise CampaignHardStop(f"the provider refused the account: {exc}") from None
            except ProviderError:
                ledger.settle(rid)
                raise
            ledger.settle(rid, response)
            return response

    return GuardedProvider()


# ------------------------------------------------------------------------------------------------ per cell
def cell_meter(max_calls: int, max_tokens: int):
    """A ``UsageMeter`` that also remembers which limit cut the cell and the model of every reply."""
    from orbitopt.agent.providers import BudgetExceededError, UsageMeter

    class CellMeter(UsageMeter):
        def __init__(self):
            super().__init__(max_tokens=max_tokens, max_calls=max_calls)
            self.cut: str | None = None
            self.replies: list[tuple[str, object]] = []

        def before_call(self):
            try:
                super().before_call()
            except BudgetExceededError as exc:
                self.cut = exc.kind
                raise

        def record(self, usage, response=None):
            if response is not None:
                self.replies.append((response.model, usage))
            try:
                super().record(usage, response)
            except BudgetExceededError as exc:
                self.cut = exc.kind
                raise

    return CellMeter()


def resuming(inner, cassette: Path, meter):
    """``inner`` behind a cassette: a call the cassette holds is served from it, any other is made and recorded."""
    from orbitopt.agent.providers import RecordingProvider, ReplayProvider

    cassette = Path(cassette)
    recorded = ReplayProvider(cassette) if cassette.exists() and cassette.stat().st_size else None

    class ResumingProvider(RecordingProvider):
        def complete(self, request):
            if recorded is not None and recorded.pending(request):
                return recorded.complete(request)
            return super().complete(request)

    return ResumingProvider(inner, cassette, meter=meter)


@dataclass
class LLMContext:
    """What a shard needs to run the provider arms live: the provider and the campaign's hard stop."""
    live: object
    ledger: Ledger | None
    spec: dict = field(default_factory=dict)

    def cell(self, cassette: Path, max_calls: int, max_tokens: int):
        inner = self.live if self.ledger is None else guarded(self.live, self.ledger)
        return resuming(inner, cassette, cell_meter(max_calls, max_tokens))


def context_from_env(out: Path, max_calls: int | None = None, max_usd: float | None = None) -> LLMContext | None:
    """The live context of the LLM_* variables, with the hard stop's ledger in ``out``; None without a provider."""
    from orbitopt.llm_env import provider_from_env, provider_spec
    live = provider_from_env()
    if live is None:
        return None
    calls, usd = limits_from(max_calls, max_usd)
    return LLMContext(live, Ledger(Path(out) / "llm_ledger.jsonl", max_calls=calls, max_usd=usd, model=live.model),
                      provider_spec() or {})


def llm_fields(meter) -> dict:
    """A row's LLM accounting from the cell's meter: every call, repairs and refused calls included."""
    usage = meter.usage
    costs = [cost_usd(u, m) for m, u in meter.replies]
    return {"llm_calls": meter.calls, "tokens_in": usage.prompt_tokens, "tokens_out": usage.completion_tokens,
            "cache_hit_tokens": usage.cache_hit_tokens, "cache_miss_tokens": usage.cache_miss_tokens,
            "cost_usd": math.fsum(c for c in costs if c is not None),
            "cost_priced": all(c is not None for c in costs),
            "llm_models": sorted({m for m, _ in meter.replies}), "llm_cut": meter.cut,
            "llm_over_budget": meter.max_tokens is not None and usage.total > meter.max_tokens}
