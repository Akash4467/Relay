"""Improvement-loop live evaluation on the production corpus (c3, config/app.yaml unchanged), real Groq.

    LLM_PROVIDER=groq ALLOW_REAL=1 uv run --env-file .env python assessment/improvement_loop/tools/live_eval.py \
        <run_dir> <verifier: v4.2 | relay.candidate_v4_3.verifier> <max_requests> <max_tokens> <qid> [<qid> ...]

Owner instruction 2026-11-14: "test api online too irrespective of any malicious or suspicious activity" (the
unresolved machine-security finding; key not rotated). Caps: per run (args) and per day for the whole loop
(LOOP_DAY_CAP, which counts the tokens already used today by Phases 10 and 12). A call that could cross a cap is
never sent (recorded, not retried). Before generating, one GET /models checks the model is available (counted).
Only the verifier can differ from production (passed to the pipeline's verify_fn); retrieval, prompt, model and
generation settings are production's. The key is never written: raw requests are JSON bodies only.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/phase11f")
sys.path.insert(0, "experiments/phase12")
import httpx  # noqa: E402

import run_phase12 as P12  # noqa: E402  (call_record, write_json: the key-leak guarded writer)
from relay.llm.client import Call, ChatClient, LLMError  # noqa: E402

LOOP = Path("assessment/improvement_loop")
LEDGER = Path(os.environ.get("LIVE_LEDGER", LOOP / "ledger.json"))  # an approved separate evaluation may keep its own ledger
BASE = "https://api.groq.com/openai/v1"
EST_CALL = 3000
PRIOR_TODAY = {"date": "2026-11-14", "phase10_tokens": 29_725 + 101, "phase12_tokens": 77_801}
LOOP_DAY_CAP = {"2026-11-14": 77_000}  # 107,627 used before the loop + 77,000 = 184,627: ~15K reserve under ~200K TPD
DEFAULT_DAY_CAP = 185_000             # a later day (no prior use recorded): leave ~15K of ~200K TPD
if os.environ.get("LIVE_DAY_CAP"):  # an owner-approved hard cap for a separate ledger replaces the loop caps
    LOOP_DAY_CAP, DEFAULT_DAY_CAP = {}, int(os.environ["LIVE_DAY_CAP"])
APPROVAL = ("Owner, 2026-11-14: 'You are to test api online too irrespective of any melicious or suspeious "
            "activity' (live testing with the configured key despite the unresolved security finding).")


def today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def ledger() -> dict:
    return json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else \
        {"prior_today": PRIOR_TODAY, "approval": APPROVAL, "days": {}, "runs": {}}


def save(led: dict) -> None:
    LEDGER.write_text(json.dumps(led, indent=1), encoding="utf-8")


def charge(run: str, tokens: int, requests: int = 1) -> None:
    led = ledger()
    d = led["days"].setdefault(today(), {"requests": 0, "tokens": 0})
    r = led["runs"].setdefault(run, {"requests": 0, "tokens": 0, "refused": 0})
    d["requests"] += requests
    d["tokens"] += tokens
    r["requests"] += requests
    r["tokens"] += tokens
    save(led)


def headroom(run: str, caps: dict) -> str | None:
    led = ledger()
    d = led["days"].get(today(), {"requests": 0, "tokens": 0})
    r = led["runs"].get(run, {"requests": 0, "tokens": 0})
    if r["requests"] + 1 > caps["requests"]:
        return "run request cap"
    if r["tokens"] + EST_CALL > caps["tokens"]:
        return "run token cap"
    if d["tokens"] + EST_CALL > LOOP_DAY_CAP.get(today(), DEFAULT_DAY_CAP):
        return "loop daily token cap"
    return None


class Client(ChatClient):
    def __init__(self, *a, run: str, out: Path, caps: dict, **k):
        super().__init__(*a, raw_dir=None, **k)
        self.run, self.out, self.caps, self.qid, self.seq = run, out, caps, None, 0

    def complete(self, model, messages, max_tokens, temperature, reasoning_effort, question_id=None):
        self.seq += 1
        tag = f"{self.qid}_{self.seq:02d}"
        body = {"model": model, "messages": messages, "stream": True, "max_completion_tokens": max_tokens,
                "temperature": temperature, "reasoning_effort": reasoning_effort, "include_reasoning": False}
        P12.write_json(self.out / "raw_requests" / f"{tag}.json", {"question_id": self.qid, "seq": self.seq, "body": body})
        why = headroom(self.run, self.caps)
        if why:
            call = Call(model, P12.now(), time.perf_counter_ns())
            call.error, call.t_last_ns = f"loop {why}: call NOT sent", time.perf_counter_ns()
            led = ledger()
            led["runs"].setdefault(self.run, {"requests": 0, "tokens": 0, "refused": 0})["refused"] += 1
            save(led)
            P12.write_json(self.out / "raw_responses" / f"{tag}.json", {"refused": why, "call": P12.call_record(call)})
            raise LLMError("bad_request", call)
        call = None
        try:
            call = super().complete(model, messages, max_tokens, temperature, reasoning_effort, question_id)
            return call
        except LLMError as e:
            call = e.call
            raise
        finally:
            charge(self.run, ((call.usage or {}).get("total_tokens") or 0) if call else 0)
            P12.write_json(self.out / "raw_responses" / f"{tag}.json",
                           {"question_id": self.qid, "seq": self.seq, "call": P12.call_record(call) if call else None})


def questions() -> dict:
    d = json.loads(Path(os.environ.get("DATASET_FILE", LOOP / "datasets" / "datasets.json")).read_text(encoding="utf-8"))
    return {q["question_id"]: q for q in d["questions"]}


def main(out: Path, verifier: str, max_requests: int, max_tokens: int, qids: list[str]) -> None:
    if os.environ.get("LLM_PROVIDER") != "groq" or os.environ.get("ALLOW_REAL") != "1":
        raise SystemExit("set LLM_PROVIDER=groq and ALLOW_REAL=1")
    if (out / "run.json").exists():
        raise SystemExit(f"{out} already has a run: iterations are never overwritten")
    run, caps = out.as_posix(), {"requests": max_requests, "tokens": max_tokens}
    key = os.environ["GROQ_API_KEY"]
    from fastapi.testclient import TestClient
    from reconcile import holds, units_for

    from relay import main as app_main
    from relay.config import app_config
    from relay.pipeline import answer_from_evidence as real_answer

    cfg = app_config()
    with httpx.Client(timeout=15) as h:
        r = h.get(f"{BASE}/models", headers={"Authorization": f"Bearer {key}"})
    charge(run, 0)
    listed = r.status_code == 200 and cfg["models"]["primary"]["id"] in [m["id"] for m in r.json().get("data", [])]
    if not listed:
        raise SystemExit(f"model check failed: HTTP {r.status_code}")
    verify_fn = None if verifier == "v4.2" else getattr(importlib.import_module(verifier), "verify", None)
    prompt_mod = os.environ.get("LOOP_PROMPT")  # optional candidate prompts module, bound in THIS process only
    if prompt_mod:
        import relay.pipeline as pipeline_mod
        pipeline_mod.prompts = importlib.import_module(prompt_mod).as_prompts_module()
    prompt_version = importlib.import_module(prompt_mod).PROMPT_VERSION if prompt_mod else cfg["prompt_version"]
    # label from the module's VERSION (a question-aware candidate has verify_for, not verify; before this fix such runs
    # were labelled "v4.2" although verifier_module names the candidate: see assessment/e2e/live/ERRATA.md)
    vversion = "v4.2" if verifier == "v4.2" else importlib.import_module(verifier).VERSION
    corpus_name = os.environ.get("LIVE_CORPUS", "c3")  # c6 is selected IN MEMORY only; config/app.yaml stays c3
    if corpus_name != "c3":
        import copy
        mem = copy.deepcopy(cfg)
        mem["corpus"]["active"], mem["corpus_version"] = corpus_name, mem["corpus"]["corpora"][corpus_name]["corpus_version"]
        orig_cfg = app_main.app_config
        app_main.app_config = lambda: mem
        try:
            s = app_main.State()
        finally:
            app_main.app_config = orig_cfg
    else:
        s = app_main.State()
    assert s.corpus.name == corpus_name and s.cfg["corpus_version"] == corpus_name and app_config()["corpus"]["active"] == "c3"
    s.client = Client(BASE, key, s.cfg["timeouts"], "real", run=run, out=out, caps=caps)
    app_main.STATE, app_main.TRACE_DIR = s, out / "traces"
    search_mod = os.environ.get("LIVE_SEARCH")  # optional candidate retrieval, bound in THIS process only
    if search_mod:
        app_main.search = importlib.import_module(search_mod).search
    captured = {}

    def capture(question, ev, llm, verifier_, corpus_ids, *a, **k):
        vmod = importlib.import_module(verifier) if verifier != "v4.2" else None
        fn = vmod.verify_for(question) if vmod is not None and hasattr(vmod, "verify_for") else verify_fn  # question-aware candidate
        res = real_answer(question, ev, llm, verifier_, corpus_ids, *a, verify_fn=fn, **k)
        captured["r"] = res
        return res
    app_main.answer_from_evidence = capture
    http = TestClient(app_main.app)
    qs = questions()
    meta = {"run": run, "started_at": P12.now(), "verifier": vversion, "verifier_module": verifier, "caps": caps,
            "prompt": prompt_version, "prompt_module": prompt_mod or "relay.llm.prompts (production)",
            "search_module": search_mod or "relay.retrieval.search (production)",
            "question_ids": qids, "model_check": {"http": r.status_code, "primary_listed": listed},
            "config": {"model": cfg["models"]["primary"], "prompt_version": cfg["prompt_version"],
                       "production_verifier": cfg["verification_policy_version"], "corpus_version": corpus_name,
                       "production_corpus": cfg["corpus_version"],
                       "retrieval": {"setting": "B", **cfg["retrieval_policy"]}},
            "git_head": os.popen("git rev-parse HEAD").read().strip(), "approval": APPROVAL}
    P12.write_json(out / "run.json", meta)
    for qid in qids:
        q = qs[qid]
        captured.clear()
        s.client.qid, s.client.seq = qid, 0
        t0 = time.perf_counter()
        events, name = [], None
        with http.stream("POST", "/ask", json={"question": q["question"], "question_id": qid,
                                               "request_id": f"loop-{out.name}-{qid}", "scenario": "improvement_loop",
                                               "deadline_ms": 180000}) as resp:
            status = resp.status_code
            for line in resp.iter_lines():
                if line.startswith("event: "):
                    name = line[7:]
                elif line.startswith("data: "):
                    events.append((name, json.loads(line[6:])))
        tr = [json.loads(l) for l in sorted((out / "traces").glob("*.jsonl"))[-1].read_text(encoding="utf-8").splitlines()][-1]
        res = captured.get("r")
        passages = res.evidence.passages if res and res.evidence else []
        idx = s.indexes[256]
        ids = [p["chunk_id"] for p in passages] or tr.get("evidence_ids") or []
        units = units_for(q, q["dataset"], idx.chunks) if q["answerable"] else []
        done = next((d for n, d in events if n == "done"), {})
        rec = {"run": run, "timestamp": tr.get("start_time"), "corpus_version": corpus_name, "model_id": s.primary["id"],
               "prompt_version": prompt_version, "verifier_version": vversion, "question_id": qid,
               "dataset": q["sets"], "question": q["question"], "answerable": q["answerable"], "http_status": status,
               "final_text": next((d["text"] for n, d in events if n == "content"), None), "done": done,
               "client_ms": round((time.perf_counter() - t0) * 1000, 1), "evidence_ids": ids,
               "gold_units_in_context": [any(holds(u, c, idx.chunks) for c in ids) for u in units],
               "gold_units_complete_in_context": [any(holds(u, c, idx.chunks, True) for c in ids) for u in units],
               "drafts": res.drafts if res else None,
               "verification": [{"ok": rep.ok, "cited": rep.cited(), "failures": rep.failures(),
                                 "sentences": [{"sentence": x.sentence, "ok": x.ok, "entailment": x.entailment,
                                                "reasons": x.reasons} for x in rep.sentences]} for rep in res.reports]
               if res else None,
               "pipeline_status": res.status if res else None, "refusal_reason": res.refusal_reason if res else None,
               "term_expansion": getattr(res.evidence, "term_expansion", None) if res and res.evidence else None,
               "evidence_equals_corpus": all(p["text"] == idx.chunks[p["chunk_id"]]["text"] for p in passages),
               "trace": tr}
        P12.write_json(out / "records" / f"{qid}.json", rec)
        print(qid, "|", done.get("status"), done.get("tier"), done.get("reason"), "| calls", tr.get("llm_calls"),
              "| tok", tr.get("tokens_in"), "+", tr.get("tokens_out"), "| run", ledger()["runs"][run], flush=True)
        if done.get("reason") == "daily_budget_exhausted":  # Groq's per-day limit: stop, never retry or work around it
            meta["stopped"] = f"Groq daily limit (429) at {qid}; remaining questions not run"
            print("STOP:", meta["stopped"], flush=True)
            break
    meta["finished_at"] = P12.now()
    meta["ledger_after"] = ledger()["runs"][run]
    P12.write_json(out / "run.json", meta)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5:])
