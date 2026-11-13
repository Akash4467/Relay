"""Front door (DESIGN §8–9): POST /ask (SSE), GET /health. One uvicorn worker, one event loop.

    LLM_PROVIDER=standin uv run --env-file .env uvicorn relay.main:app --port 8000 --workers 1

Admission (§9): a request gets a deadline (its deadline_ms, else the default patience). It is rejected on arrival
when the queue is full or its estimated wait exceeds the deadline; it waits FIFO for one of `workers` slots and is
dropped if the deadline passes first. Work (retrieval, LLM through the governor, NLI verification) runs in a bounded
thread pool, never on the event loop. When the daily budget is known to be gone, the reply is immediate.
Events: meta (admitted), content (the verified answer or the honest refusal/busy/unavailable text), done.
Progress events are not answer tokens: first_token_ms is the first content event.
Every request appends one trace line to results/raw/traces/{date}.jsonl. Cache and single-flight are not built.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from relay.config import app_config, retrieval_settings
from relay.governor.governor import Governor
from relay.llm import prompts
from relay.llm.client import LLMError, from_env
from relay.llm.hosted import DailyExhausted, GovernorReject, HostedLLM, RetryBudget
from relay.pipeline import Verifier, answer_from_evidence
from relay.retrieval.index_build import load_model
from relay.retrieval.rerank import load_reranker
from relay.retrieval.search import Index, search
from relay.verify import nli

TRACE_DIR = Path(os.environ.get("TRACE_DIR", "results/raw/traces"))


class AskRequest(BaseModel):
    question: str
    request_id: str | None = None
    question_id: str | None = None
    deadline_ms: int | None = None
    retrieval_setting: str | None = None
    scenario: str = "dev"


class State:
    """Everything loaded once at startup, plus the live admission counters."""

    def __init__(self):
        self.cfg = app_config()
        self.settings = retrieval_settings()
        self.default_setting = os.environ.get("RETRIEVAL_SETTING", "B")
        self.client = from_env(self.cfg["timeouts"])
        if self.client.source == "real" and os.environ.get("ALLOW_REAL") != "1":
            raise SystemExit("refusing to serve real Groq without ALLOW_REAL=1 (every real call needs approval)")
        embedder = load_model()
        self.indexes = {size: Index(size, embedder) for size in {s.piece_tokens for s in self.settings.values()}}
        self.reranker = load_reranker() if any(s.rerank for s in self.settings.values()) else None
        ver = self.cfg["verification"]
        self.verifier = Verifier(nli.load_nli(ver["nli_model"]), ver["entail_min"], ver["entail_min_flagged"])
        self.governor = Governor.from_config()
        self.budget = RetryBudget(self.cfg["retry"]["budget_frac_of_initial_per_min"])
        q = self.cfg["queue"]
        self.workers, self.max_queue = q["semaphore_inflight_hosted_calls"], q["max_length"]
        self.executor = ThreadPoolExecutor(max_workers=self.workers)
        self.slots = asyncio.Semaphore(self.workers)
        self.waiting = self.busy = self.peak_waiting = 0
        self.service_ema_s = 5.0  # initial guess, updated from finished requests
        primary = self.cfg["models"]["primary"]
        self.primary = primary
        self.output_reserve = int(self.cfg["reservation"]["output_p95_tokens"][primary["id"]]
                                  * (1 + self.cfg["reservation"]["safety_margin"]))
        self.typical_reserve = 2500 + self.output_reserve  # prompt + evidence at the 1,600-token budget + output
        self.trace_lock = threading.Lock()

    def estimate_wait_s(self) -> float:
        """From the binding resource: free slots (FIFO position / workers x service time) or primary-model tokens
        for everything queued ahead plus this request."""
        slot_wait = 0.0 if self.busy < self.workers else (self.waiting + 1) / self.workers * self.service_ema_s
        # tokens for the queue can exceed one minute's capacity: they still drain at the refill rate (finite wait)
        tpm = self.governor.buckets[self.primary["id"]]["tpm"]
        need = (self.waiting + 1) * self.typical_reserve
        token_wait = max(0.0, need - tpm.available(self.governor.clock())) * tpm.period_s / tpm.capacity
        return max(slot_wait, token_wait)

    def retry_in_s(self) -> int:
        return max(1, min(3600, round(self.estimate_wait_s())))

    def trace(self, record: dict) -> None:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self.trace_lock, (TRACE_DIR / f"{record['start_time'][:10]}.jsonl").open("a", encoding="utf-8") as f:
            f.write(line)


STATE: State | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global STATE
    STATE = State()
    yield
    STATE.executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(lifespan=lifespan)


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def ms_since(t0_ns: int) -> float:
    return round((time.perf_counter_ns() - t0_ns) / 1e6, 1)


def work(s: State, req: AskRequest, deadline: float, rec: dict) -> tuple[str, dict]:
    """Runs in the thread pool. Returns (reply text, done event) and fills rec with timings and tokens."""
    setting = s.settings[req.retrieval_setting or s.default_setting]
    t0 = time.perf_counter_ns()
    ev = search(req.question, setting, s.cfg["pressure"]["L0"]["context_cap"], s.indexes[setting.piece_tokens],
                s.reranker if setting.rerank else None)
    rec.update(retrieval_ms=ms_since(t0), context_tokens=ev.tokens, retrieval_policy=ev.policy)
    retry = s.cfg["retry"]
    llm = HostedLLM(s.client, s.governor, s.primary, s.output_reserve,
                    request_overhead=s.cfg["reservation"]["request_overhead_tokens"], question_id=req.question_id,
                    deadline=deadline, max_wait_s=max(0.0, deadline - time.monotonic()),
                    max_retries=retry["max_transport_retries_per_call"], backoff_base_s=retry["backoff_base_ms"] / 1000,
                    backoff_cap_s=retry["backoff_cap_ms"] / 1000, budget=s.budget)
    done: dict = {"model": s.primary["id"]}
    try:
        result = answer_from_evidence(req.question, ev, llm, s.verifier, set(s.indexes[setting.piece_tokens].chunks))
        text = result.text
        done |= {"status": "ok", "tier": "normal" if result.status == "answered" else "refused",
                 "reason": result.refusal_reason, "citations": result.citations}
        rec.update(verification_ms=result.verification_ms, answer_status=result.status)
    except DailyExhausted as e:
        text = prompts.unavailable_text(e.reset_in_s)
        done |= {"status": "rate_limited", "tier": "unavailable", "reason": "daily_budget_exhausted"}
    except GovernorReject:
        text = prompts.busy_text(s.retry_in_s())
        done |= {"status": "rate_limited", "tier": "unavailable", "reason": "local_governor_reject"}
    except LLMError as e:
        kind = {"rate_limited": ("rate_limited", "upstream_429"), "timeout": ("timeout", "internal_timeout")}
        status, reason = kind.get(e.kind, ("error", "upstream_error"))
        text = prompts.busy_text(10) if status != "error" else "The assistant hit an internal error. Please try again."
        done |= {"status": status, "tier": "unavailable", "reason": reason}
    ok_calls = [c for c in llm.calls if c["status"] == "ok"]
    rec.update(
        llm_calls=len(llm.calls), initial_attempts=sum(c["attempt"] == 0 for c in llm.calls), retries=llm.retries,
        redraft_attempts=max(0, sum(c["attempt"] == 0 for c in llm.calls) - 1), fallback_generations=0,
        governor_wait_ms=round(sum(c["governor_wait_ms"] for c in llm.calls), 1),
        model_ttft_ms=ok_calls[0]["model_ttft_ms"] if ok_calls else None,
        first_byte_ms=ok_calls[0]["first_byte_ms"] if ok_calls else None,
        generation_ms=round(sum(c.get("total_ms") or 0 for c in llm.calls), 1),
        tokens_in=sum(c["tokens_in"] for c in ok_calls), tokens_out=sum(c["tokens_out"] for c in ok_calls),
        reasoning_tokens=sum(c.get("reasoning_tokens") or 0 for c in ok_calls),
        reservation_overrun_tokens=sum(c["reservation_overrun_tokens"] for c in ok_calls),
        model=s.primary["id"] if llm.calls else None, calls=llm.calls)
    return text, done


@app.post("/ask")
async def ask(req: AskRequest):
    s = STATE
    arrival = time.perf_counter_ns()
    deadline = time.monotonic() + (req.deadline_ms or s.cfg["deadlines"]["default_patience_ms"]) / 1000
    rec = {"request_id": req.request_id or uuid.uuid4().hex, "scenario": req.scenario,
           "source": "real" if s.client.source == "real" else "stand-in",
           "start_time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
           "question_id": req.question_id, "retrieval_setting": req.retrieval_setting or s.default_setting,
           "deadline_ms": req.deadline_ms, "queue_length_at_arrival": s.waiting, "busy_at_arrival": s.busy,
           "degradation_level": "L0", "cache_hit": False, "retries": 0, "tokens_in": 0, "tokens_out": 0}

    def finish(text: str, done: dict) -> None:
        rec.update(status=done["status"], failure_reason=done.get("reason") if done["status"] != "ok" else None,
                   answer_tier=done.get("tier"), reply_reason=done.get("reason"), total_ms=ms_since(arrival))
        s.trace(rec)

    async def immediate(text: str, done: dict):
        rec["first_token_ms"] = ms_since(arrival)
        yield sse("meta", {"request_id": rec["request_id"], "admitted": False})
        yield sse("content", {"text": text})
        yield sse("done", done)
        finish(text, done)

    reset = s.governor.daily_exhausted_for_s(s.primary["id"])
    if reset > 0:
        rec.update(admitted=False, queue_wait_ms=0.0)
        return StreamingResponse(immediate(prompts.unavailable_text(reset), {
            "status": "rate_limited", "tier": "unavailable", "reason": "daily_budget_exhausted"}),
            media_type="text/event-stream")
    remaining = deadline - time.monotonic()
    if s.waiting >= s.max_queue or s.estimate_wait_s() > remaining:
        rec.update(admitted=False, queue_wait_ms=0.0, estimated_wait_s=round(s.estimate_wait_s(), 2))
        return StreamingResponse(immediate(prompts.busy_text(s.retry_in_s()), {
            "status": "rate_limited", "tier": "unavailable", "reason": "queue_rejected"}),
            media_type="text/event-stream")

    async def stream():
        s.waiting += 1
        s.peak_waiting = max(s.peak_waiting, s.waiting)
        t_queue = time.perf_counter_ns()
        try:
            await asyncio.wait_for(s.slots.acquire(), timeout=max(0.0, deadline - time.monotonic()))
        except asyncio.TimeoutError:
            s.waiting -= 1
            rec.update(admitted=True, queue_wait_ms=ms_since(t_queue))
            async for chunk in immediate(prompts.busy_text(s.retry_in_s()), {
                    "status": "timeout", "tier": "unavailable", "reason": "deadline_expired"}):
                yield chunk
            return
        s.waiting -= 1
        s.busy += 1
        rec.update(admitted=True, queue_wait_ms=ms_since(t_queue))
        yield sse("meta", {"request_id": rec["request_id"], "admitted": True})
        gone = {"client": False}
        fut = asyncio.get_running_loop().run_in_executor(s.executor, work, s, req, deadline, rec)

        def released(f):
            s.busy -= 1
            s.slots.release()
            if not f.cancelled() and f.exception() is None:
                s.service_ema_s = 0.8 * s.service_ema_s + 0.2 * (ms_since(t_queue) - rec["queue_wait_ms"]) / 1000
            if gone["client"]:  # the user gave up: the work still finished, record what it cost
                rec["client_disconnected"] = True
                if not f.cancelled() and f.exception() is None:
                    finish(*f.result())
                else:
                    finish("", {"status": "error", "reason": "work_failed_after_disconnect"})
        fut.add_done_callback(released)
        try:
            text, done = await asyncio.shield(fut)
        except asyncio.CancelledError:
            gone["client"] = True
            raise
        except Exception as e:  # a bug, not a provider outcome: still answer and record it
            text, done = "The assistant hit an internal error. Please try again.", {
                "status": "error", "tier": "unavailable", "reason": f"internal: {type(e).__name__}"}
        rec["first_token_ms"] = ms_since(arrival)
        yield sse("content", {"text": text})
        yield sse("done", done | {"request_id": rec["request_id"]})
        finish(text, done)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/admin/reset")
async def admin_reset():
    """Start a load scenario like a fresh day: new governor and retry budget, peak queue cleared.
    Only against the stand-in: a real provider's quota is never reset by pretending."""
    s = STATE
    if s.client.source != "standin":
        return {"reset": False, "reason": "only allowed against the stand-in"}
    s.governor = Governor.from_config()
    s.budget = RetryBudget(s.cfg["retry"]["budget_frac_of_initial_per_min"])
    s.peak_waiting, s.service_ema_s = s.waiting, 5.0
    return {"reset": True}


@app.get("/health")
async def health():
    s = STATE
    return {"indexes": sorted(s.indexes), "source": s.client.source, "queue_waiting": s.waiting, "busy": s.busy,
            "peak_waiting": s.peak_waiting, "service_ema_s": round(s.service_ema_s, 2),
            "daily_exhausted_for_s": s.governor.daily_exhausted_for_s(s.primary["id"]),
            "primary_tpm_available": round(s.governor.buckets[s.primary["id"]]["tpm"].available(s.governor.clock()))}
