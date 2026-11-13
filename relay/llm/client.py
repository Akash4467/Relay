"""OpenAI-compatible streaming client for Groq and the stand-in (DESIGN §11). Same code for both providers;
only the base URL and key differ.

Request (Groq docs, read 2026-11-12): `max_completion_tokens` (max_tokens is deprecated), `reasoning_effort`,
`include_reasoning: false` (hides gpt-oss reasoning; `reasoning_format` is not supported for gpt-oss).
Records per call: monotonic send / first byte / first content / last times, the largest gap between chunks,
rate-limit headers, usage, finish reason. Every call is appended to results/raw/{real,standin}/{date}.jsonl
without any request header, so the API key never reaches a log.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

RAW_DIR = Path("results/raw")
RATE_HEADERS = ("x-ratelimit-limit-requests", "x-ratelimit-remaining-requests", "x-ratelimit-reset-requests",
                "x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens", "x-ratelimit-reset-tokens",
                "retry-after")


class LLMError(Exception):
    def __init__(self, kind: str, call: "Call"):
        super().__init__(f"{kind}: HTTP {call.status} {call.error or ''}".strip())
        self.kind, self.call = kind, call  # kind: rate_limited | upstream_error | timeout | bad_request


@dataclass
class Call:
    model: str
    started_at: str
    t_send_ns: int
    status: int | None = None
    t_first_byte_ns: int | None = None
    t_first_content_ns: int | None = None
    t_last_ns: int | None = None
    max_gap_ms: float = 0.0
    text: str = ""
    finish_reason: str | None = None
    usage: dict | None = None             # provider-reported; None if the stream carried none
    usage_source: str | None = None       # "usage" | "x_groq.usage": where the stream carried it
    headers: dict = field(default_factory=dict)
    error: str | None = None

    def ms(self, t_ns: int | None) -> float | None:
        return None if t_ns is None else round((t_ns - self.t_send_ns) / 1e6, 1)

    @property
    def retry_after_s(self) -> float | None:
        try:
            return float(self.headers["retry-after"])
        except (KeyError, ValueError):
            return None


def from_env(timeouts_ms: dict, raw_dir: Path | None = RAW_DIR) -> "ChatClient":
    """LLM_PROVIDER=groq for real calls; anything else (the default) is the stand-in, so a missing variable
    never spends real quota. The key comes from the environment (uv run --env-file .env), never from code."""
    provider = os.environ.get("LLM_PROVIDER", "standin")
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY is not set: run with `uv run --env-file .env ...`")
    default = "https://api.groq.com/openai/v1" if provider == "groq" else "http://localhost:8001/v1"
    return ChatClient(os.environ.get("GROQ_BASE_URL") or default, key, timeouts_ms,
                      "real" if provider == "groq" else "standin", raw_dir)


class ChatClient:
    def __init__(self, base_url: str, api_key: str, timeouts_ms: dict, source: str = "real",
                 raw_dir: Path | None = RAW_DIR, transport: httpx.BaseTransport | None = None):
        if source not in ("real", "standin"):
            raise ValueError("source must be 'real' or 'standin'")
        self.base_url, self.source, self.raw_dir = base_url.rstrip("/"), source, raw_dir
        self.total_ms = timeouts_ms["total_call_ms"]
        # shortcut: the first-byte limit also bounds the gap between chunks (one httpx read timeout);
        # split them when the stand-in tests mid-stream stalls (phase 5)
        timeout = httpx.Timeout(connect=timeouts_ms["connect_ms"] / 1000, read=timeouts_ms["first_byte_ms"] / 1000,
                                write=10.0, pool=10.0)
        self.http = httpx.Client(timeout=timeout, transport=transport,
                                 headers={"Authorization": f"Bearer {api_key}"})

    def complete(self, model: str, messages: list[dict], max_tokens: int, temperature: float,
                 reasoning_effort: str, question_id: str | None = None) -> Call:
        body = {"model": model, "messages": messages, "stream": True, "max_completion_tokens": max_tokens,
                "temperature": temperature, "reasoning_effort": reasoning_effort, "include_reasoning": False}
        headers = {"X-Relay-Question-Id": question_id} if question_id and self.source == "standin" else {}
        call = Call(model, datetime.now(timezone.utc).isoformat(timespec="milliseconds"), time.perf_counter_ns())
        try:
            with self.http.stream("POST", f"{self.base_url}/chat/completions", json=body, headers=headers) as resp:
                call.status = resp.status_code
                call.headers = {k: resp.headers[k] for k in RATE_HEADERS if k in resp.headers}
                if resp.status_code != 200:
                    call.error = resp.read().decode("utf-8", "replace")[:2000]
                    call.t_last_ns = time.perf_counter_ns()
                    kind = "rate_limited" if resp.status_code == 429 else \
                        "upstream_error" if resp.status_code >= 500 else "bad_request"
                    raise LLMError(kind, call)
                self._read_stream(resp, call)
        except httpx.TimeoutException as e:
            call.error, call.t_last_ns = f"timeout: {type(e).__name__}", time.perf_counter_ns()
            raise LLMError("timeout", call) from e
        except httpx.TransportError as e:
            call.error, call.t_last_ns = f"transport: {type(e).__name__}", time.perf_counter_ns()
            raise LLMError("upstream_error", call) from e
        finally:
            self._log(body, call)
        return call

    def _read_stream(self, resp: httpx.Response, call: Call) -> None:
        last = None
        for line in resp.iter_lines():
            now = time.perf_counter_ns()
            if call.t_first_byte_ns is None:
                call.t_first_byte_ns = now
            if last is not None:
                call.max_gap_ms = max(call.max_gap_ms, (now - last) / 1e6)
            last = now
            if (now - call.t_send_ns) / 1e6 > self.total_ms:
                call.error = "timeout: total call time exceeded"
                raise LLMError("timeout", call)
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            # usage location is not documented for streams: accept the OpenAI place or Groq's x_groq
            if chunk.get("usage"):
                call.usage, call.usage_source = chunk["usage"], "usage"
            elif (chunk.get("x_groq") or {}).get("usage"):
                call.usage, call.usage_source = chunk["x_groq"]["usage"], "x_groq.usage"
            for choice in chunk.get("choices", []):
                piece = (choice.get("delta") or {}).get("content") or ""
                if piece and call.t_first_content_ns is None:
                    call.t_first_content_ns = now
                call.text += piece
                call.finish_reason = choice.get("finish_reason") or call.finish_reason
        call.t_last_ns = time.perf_counter_ns()

    def _log(self, body: dict, call: Call) -> None:
        if self.raw_dir is None:
            return
        out = self.raw_dir / self.source / f"{call.started_at[:10]}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        record = {"request": body, **asdict(call),
                  "first_byte_ms": call.ms(call.t_first_byte_ns), "first_content_ms": call.ms(call.t_first_content_ns),
                  "total_ms": call.ms(call.t_last_ns)}
        with out.open("a", encoding="utf-8") as f:  # append only: raw results are a deliverable (CLAUDE.md rule 5)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
