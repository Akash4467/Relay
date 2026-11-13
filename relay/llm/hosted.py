"""The pipeline's LLM callable for a hosted model: reserve -> call -> settle (CLAUDE.md hard rule 9), with the
retry policy of DESIGN §15:
- at most `max_retries` transport retries per call, on 5xx, timeouts and per-minute 429s;
- backoff = max(retry-after, uniform(0, min(cap, base * 2**k))) (full jitter, never shorter than retry-after);
- never retry a per-day 429 (the daily budget is gone: GovernorReject-like DailyExhausted) or a 4xx;
- never retry past the request's deadline;
- a shared budget: retries in the last minute <= 10% of first attempts in that minute, but at least 1, so a quiet
  system can still retry once (100 first attempts -> 10 retries);
- every retry reserves through the governor again.
"""
from __future__ import annotations

import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from relay.governor.governor import Governor
from relay.llm.client import ChatClient, LLMError
from relay.llm.tokenizer import count

MESSAGE_OVERHEAD_TOKENS = 4  # role/format tokens per chat message


class GovernorReject(Exception):
    """No capacity for this call within the allowed wait."""


class DailyExhausted(Exception):
    """The provider (or the governor) says the daily budget for this model is gone."""

    def __init__(self, model: str, reset_in_s: float):
        super().__init__(f"{model}: daily budget exhausted, resets in {reset_in_s:.0f}s")
        self.model, self.reset_in_s = model, reset_in_s


class RetryBudget:
    """Rolling one-minute window of first attempts and retries, shared by all requests."""

    def __init__(self, fraction: float = 0.10, window_s: float = 60.0, clock=time.monotonic):
        self.fraction, self.window_s, self.clock = fraction, window_s, clock
        self.initial: deque[float] = deque()
        self.retries: deque[float] = deque()
        self.lock = threading.Lock()

    def _trim(self, now: float) -> None:
        for q in (self.initial, self.retries):
            while q and q[0] < now - self.window_s:
                q.popleft()

    def first_attempt(self) -> None:
        with self.lock:
            now = self.clock()
            self._trim(now)
            self.initial.append(now)

    def try_retry(self) -> bool:
        with self.lock:
            now = self.clock()
            self._trim(now)
            if len(self.retries) >= max(1, int(self.fraction * len(self.initial))):
                return False
            self.retries.append(now)
            return True


@dataclass
class HostedLLM:
    client: ChatClient
    governor: Governor
    model: dict                      # config models.primary / models.fallback
    output_reserve: int              # output_p95_tokens * (1 + safety_margin), reasoning included
    request_overhead: int = 0        # config reservation.request_overhead_tokens
    max_wait_s: float = 20.0         # longest governor wait accepted for one call
    question_id: str | None = None
    deadline: float | None = None    # monotonic time after which no call or retry starts
    max_retries: int = 0             # config retry.max_transport_retries_per_call
    backoff_base_s: float = 0.5
    backoff_cap_s: float = 8.0
    budget: RetryBudget | None = None
    sleep: object = time.sleep
    rng: random.Random = field(default_factory=random.Random)
    calls: list[dict] = field(default_factory=list)
    retries: int = 0

    def _left(self) -> float:
        return float("inf") if self.deadline is None else self.deadline - time.monotonic()

    def __call__(self, messages: list[dict]) -> str:
        if self.budget:
            self.budget.first_attempt()
        for attempt in range(self.max_retries + 1):
            try:
                return self._attempt(messages, attempt)
            except LLMError as e:
                per_day = e.kind == "rate_limited" and "per day" in (e.call.error or "")
                if per_day:
                    self.governor.on_daily_429(self.model["id"], e.call.retry_after_s)
                    raise DailyExhausted(self.model["id"], self.governor.daily_exhausted_for_s(self.model["id"])) from e
                retryable = e.kind in ("rate_limited", "upstream_error", "timeout")
                if not retryable or attempt == self.max_retries:
                    raise
                backoff = max(e.call.retry_after_s or 0.0,
                              self.rng.uniform(0, min(self.backoff_cap_s, self.backoff_base_s * 2 ** attempt)))
                if backoff >= self._left() or (self.budget and not self.budget.try_retry()):
                    raise
                self.retries += 1
                self.calls[-1]["backoff_s"] = round(backoff, 3)
                self.sleep(backoff)
        raise AssertionError("unreachable")

    def _attempt(self, messages: list[dict], attempt: int) -> str:
        if self.governor.daily_exhausted_for_s(self.model["id"]) > 0:
            raise DailyExhausted(self.model["id"], self.governor.daily_exhausted_for_s(self.model["id"]))
        tokens_in = self.request_overhead + sum(count(m["content"]) + MESSAGE_OVERHEAD_TOKENS for m in messages)
        reserve = tokens_in + self.output_reserve
        t0 = time.perf_counter_ns()
        wait = self.governor.wait_estimate(self.model["id"], reserve)
        if wait > min(self.max_wait_s, self._left()):
            raise GovernorReject(f"{self.model['id']}: estimated wait {wait:.1f}s exceeds what the request allows")
        if wait > 0:
            self.sleep(wait)
        res = self.governor.try_reserve(self.model["id"], reserve)
        if res is None:
            raise GovernorReject(f"{self.model['id']}: no capacity after waiting {wait:.1f}s")
        record = {"model": self.model["id"], "attempt": attempt, "tokens_in_local": tokens_in, "reserved": reserve,
                  "governor_wait_ms": round((time.perf_counter_ns() - t0) / 1e6, 1)}
        try:
            call = self.client.complete(self.model["id"], messages, self.model["max_tokens"], self.model["temperature"],
                                        self.model["reasoning_effort"], self.question_id)
        except LLMError as e:
            if e.kind == "rate_limited" and "per day" not in (e.call.error or ""):
                self.governor.on_429(self.model["id"], e.call.retry_after_s)
            self.governor.settle(res, res.tokens, e.call.headers)  # unknown usage: keep the reservation
            self.calls.append(record | {"status": e.kind, "http": e.call.status, "error": e.call.error,
                                        "total_ms": e.call.ms(e.call.t_last_ns)})
            raise
        usage = call.usage or {}
        actual = usage.get("total_tokens") or tokens_in + count(call.text)
        overrun = self.governor.settle(res, actual, call.headers)
        self.calls.append(record | {
            "status": "ok", "http": call.status, "usage": call.usage, "usage_estimated": not call.usage,
            "usage_source": call.usage_source,
            "prompt_tokens_local_error": usage["prompt_tokens"] - tokens_in if "prompt_tokens" in usage else None,
            "tokens_in": usage.get("prompt_tokens", tokens_in), "tokens_out": usage.get("completion_tokens", count(call.text)),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "reservation_overrun_tokens": overrun, "finish_reason": call.finish_reason,
            "first_byte_ms": call.ms(call.t_first_byte_ns), "model_ttft_ms": call.ms(call.t_first_content_ns),
            "total_ms": call.ms(call.t_last_ns), "max_gap_ms": round(call.max_gap_ms, 1),
            "rate_headers": call.headers})
        return call.text
