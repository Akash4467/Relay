"""Phase 12: controlled live Groq evaluation (owner approval 2026-11-14: "Accept risk again" with the current key;
token cap raised by the owner to 100,000 across 12A + 12B).

    uv run --env-file .env python experiments/phase12/run_phase12.py connectivity      # 12A, <= 5 provider requests
    uv run python experiments/phase12/run_phase12.py prepare                           # 12B questions + manifest (no API)
    LLM_PROVIDER=groq ALLOW_REAL=1 uv run --env-file .env python experiments/phase12/run_phase12.py paired   # 12B

Hard caps (cumulative, from experiments/phase12/ledger.json; no override): 12A 5 requests; 12B 24 initial generation
calls and 48 provider requests in total (redrafts and transport retries included); 100,000 provider tokens across both.
A call that could cross a cap is never sent: it fails as an LLMError whose error says so (recorded, not retried).
c3/c6 are selected in memory only (config/app.yaml stays c3). The key is read from the environment and never written:
raw requests are the JSON bodies only (no headers); responses keep only the rate-limit headers.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/phase11f")
import httpx  # noqa: E402

from relay.config import app_config  # noqa: E402
from relay.llm.client import ChatClient, LLMError  # noqa: E402

HERE = Path("experiments/phase12")
A, B = HERE / "phase12a", HERE / "phase12b"
LEDGER = HERE / "ledger.json"
BASE = "https://api.groq.com/openai/v1"
EXPERIMENT_ID = "phase12-live-c3-c6-2026-11-14"
CAPS = {"connectivity_requests": 5, "initial_generation_calls": 24, "generation_requests": 48, "total_tokens": 100_000}
EST_CALL = 3000  # worst-case tokens per pipeline call (Phase 10 max 2,482): a call is refused if this could cross the cap
RISK_NOTE = ("Owner re-accepted the risk on 2026-11-14 for Phase 12 ('Accept risk again'): the current key is used "
             "although the earlier machine-security finding is unresolved (no full scan, key not rotated). Token cap "
             "raised by the owner from 30,000 to 100,000 ('Raise cap to 100,000').")
SECRET_MARKERS = ("gsk_", "Bearer ", "Authorization")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def ledger() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text(encoding="utf-8"))
    return {"caps": CAPS, "connectivity_requests": 0, "initial_generation_calls": 0, "generation_requests": 0,
            "tokens_12a": 0, "tokens_12b": 0, "refused_calls": 0, "note": RISK_NOTE}


def save_ledger(led: dict) -> None:
    LEDGER.write_text(json.dumps(led, indent=1), encoding="utf-8")


def tokens_used(led: dict) -> int:
    return led["tokens_12a"] + led["tokens_12b"]


def key() -> str:
    k = os.environ.get("GROQ_API_KEY")
    if not k or k == "dummy":
        raise SystemExit("GROQ_API_KEY not set: run with `uv run --env-file .env ...`")
    return k


def write_json(path: Path, obj) -> None:
    text = json.dumps(obj, indent=1, ensure_ascii=False)
    k = os.environ.get("GROQ_API_KEY") or "\0"
    assert k not in text and "Bearer " not in text, f"secret would be written to {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def call_record(call) -> dict:
    d = asdict(call)
    d |= {"first_byte_ms": call.ms(call.t_first_byte_ns), "first_content_ms": call.ms(call.t_first_content_ns),
          "total_ms": call.ms(call.t_last_ns)}
    return d


# ---------------------------------------------------------------- 12A
def connectivity() -> dict:
    cfg = app_config()
    model = cfg["models"]["primary"]
    from relay.llm import prompts
    from relay.llm.tokenizer import count
    out, led = {}, ledger()

    def run(name, fn):
        led_now = ledger()
        if led_now["connectivity_requests"] >= CAPS["connectivity_requests"]:
            out[name] = {"skipped": "connectivity cap reached"}
            return
        t0 = time.perf_counter()
        try:
            out[name] = fn()
        except LLMError as e:
            out[name] = {"llm_error": e.kind, "call": call_record(e.call)}
            charge_a((e.call.usage or {}).get("total_tokens") or 0)
        out[name]["client_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        print(name, json.dumps(out[name], ensure_ascii=False)[:300])

    def charge_a(tokens):
        l = ledger()
        l["connectivity_requests"] += 1
        l["tokens_12a"] += tokens
        save_ledger(l)

    def complete(client_key, msgs, max_tokens, timeouts=None):
        c = ChatClient(BASE, client_key, timeouts or cfg["timeouts"], "real", None)
        call = c.complete(model["id"], msgs, max_tokens, model["temperature"], model["reasoning_effort"])
        charge_a((call.usage or {}).get("total_tokens") or 0)
        return call

    def models():
        with httpx.Client(timeout=15) as h:
            r = h.get(f"{BASE}/models", headers={"Authorization": f"Bearer {key()}"})
        charge_a(0)
        ids = [m["id"] for m in r.json().get("data", [])] if r.status_code == 200 else []
        return {"http": r.status_code, "n_models": len(ids), "primary_listed": model["id"] in ids,
                "gpt_oss_models": sorted(i for i in ids if "gpt-oss" in i)}

    def streaming():
        call = complete(key(), [{"role": "user", "content": "Reply with exactly one word: pong"}], 64)
        return {"call": call_record(call), "streamed_content": bool(call.text), "usage_present": call.usage is not None}

    def accounting():  # a real p4 prompt over three real c3 passages; provider prompt tokens vs the local estimate
        from relay.llm.hosted import MESSAGE_OVERHEAD_TOKENS
        chunks = [json.loads(l) for l in open("data/processed/chunks_256.jsonl", encoding="utf-8")]
        passages = [c for c in chunks if c["chunk_id"] in ("AIAYN:T3:r01", "AIAYN:T3:r02", "AIAYN:3.2.2:c013")]
        msgs = prompts.messages("How many attention heads does the base Transformer use?", passages, True)
        local = cfg["reservation"]["request_overhead_tokens"] + sum(count(m["content"]) + MESSAGE_OVERHEAD_TOKENS
                                                                     for m in msgs)
        call = complete(key(), msgs, 300)
        pt = (call.usage or {}).get("prompt_tokens")
        return {"call": call_record(call), "passages": [p["chunk_id"] for p in passages], "local_prompt_estimate": local,
                "provider_prompt_tokens": pt, "relative_error": round((pt - local) / pt, 3) if pt else None}

    def bad_key():
        return {"call": call_record(complete("gsk_invalid_test_key_not_real", [{"role": "user", "content": "pong"}], 16))}

    def timeout():
        return {"call": call_record(complete(key(), [{"role": "user", "content": "pong"}], 16,
                                             cfg["timeouts"] | {"first_byte_ms": 1, "connect_ms": 3000}))}

    run("1_auth_and_model_availability", models)
    run("2_streaming_completion", streaming)
    run("3_token_accounting_p4_prompt", accounting)
    run("4_invalid_key_rejected", bad_key)
    run("5_timeout_enforced", timeout)
    g = gate(out)
    led = ledger()
    write_json(A / "connectivity_results.json", {"experiment_id": EXPERIMENT_ID, "phase": "12A", "run_at": now(),
                                                 "model_id": model["id"], "note": RISK_NOTE, "results": out, "gate": g,
                                                 "ledger_after": led})
    return {"gate": g}


def gate(out: dict) -> dict:
    def get(name, *path):
        d = out.get(name, {})
        for p in path:
            d = (d or {}).get(p) if isinstance(d, dict) else None
        return d
    acc = out.get("3_token_accounting_p4_prompt", {})
    checks = {
        "auth_ok_and_primary_model_listed": get("1_auth_and_model_availability", "http") == 200
        and get("1_auth_and_model_availability", "primary_listed") is True,
        "streaming_200_with_content_and_usage": get("2_streaming_completion", "call", "status") == 200
        and bool(get("2_streaming_completion", "streamed_content")) and bool(get("2_streaming_completion", "usage_present")),
        "rate_limit_headers_returned": bool(get("2_streaming_completion", "call", "headers")),
        "token_accounting_within_15pct": acc.get("relative_error") is not None and abs(acc["relative_error"]) <= 0.15,
        "p4_prompt_completed": get("3_token_accounting_p4_prompt", "call", "status") == 200,
        "invalid_key_rejected_401": get("4_invalid_key_rejected", "call", "status") == 401,
        "timeout_raised_as_timeout": get("5_timeout_enforced", "llm_error") == "timeout",
    }
    return {"checks": checks, "passed": all(checks.values())}


# ---------------------------------------------------------------- 12B questions
ORDINARY = ["q01", "q02", "q07", "q15"]
TABLE_GAINED = ["t01", "t05", "t13"]
TARGETED = ["q11", "t10", "t29"]
UNANSWERABLE = ["q16", "q27"]
SELECTION_RULE = {
    "ordinary_answerable": "original set, answerable, category direct/multi_section/cross_paper, every gold unit "
                           "retrieved on BOTH corpora in Phase 11F: first two direct (q01, q02), first multi_section "
                           "(q07), first cross_paper (q15)",
    "table_heavy_c6_more_complete": "Phase 7 table set, answerable, complete_evidence == 'gained' (complete gold rows in "
                                    "context on c6, not on c3) in experiments/phase11f/per_question_comparison.json: the "
                                    "first such question of each of the first three distinct tables (HNET T6 t01, "
                                    "MOSHI T3 t05, MOSHI T5 t13)",
    "targeted": "q11, t10, t29 as required by the Phase 12 brief",
    "unanswerable": "original set refusal questions on two different papers not run live in Phase 10: q16 (AIAYN), q27 (HNET)",
}


def prepare() -> None:
    orig = {json.loads(l)["question_id"]: json.loads(l) for l in open("eval/questions.jsonl", encoding="utf-8")}
    table = {q["question_id"]: q for q in json.loads(Path("eval/phase7/table_question_set.json")
                                                     .read_text(encoding="utf-8"))["questions"]}
    pq = {r["question_id"]: r for r in json.loads(Path("experiments/phase11f/per_question_comparison.json")
                                                  .read_text(encoding="utf-8"))["questions"]}
    groups = [("ordinary_answerable", ORDINARY), ("table_heavy_c6_more_complete", TABLE_GAINED),
              ("targeted", TARGETED), ("unanswerable", UNANSWERABLE)]
    qs, order = [], 0
    for grp, ids in groups:
        for qid in ids:
            src = orig.get(qid) or table[qid]
            ds = "original" if qid in orig else "table"
            qs.append({"question_id": qid, "group": grp, "dataset": ds, "question": src["question"],
                       "answerable": src.get("answerable", True), "expected_answer": src.get("expected_answer"),
                       "gold": src.get("gold"), "evidence": src.get("evidence"),
                       "paper": src.get("paper"), "table_id": src.get("table_id"),
                       "label_provenance": ("eval/questions.jsonl: project labels, gold quotes located by the assistant; "
                                            "NOT independently reviewed") if ds == "original" else
                       ("eval/phase7/table_question_set.json: PROVISIONAL assistant-authored labels; NOT independently "
                        "reviewed"),
                       "phase11f_offline": {k: pq.get(qid, {}).get(k) for k in ("complete_evidence", "c3_found", "c6_found",
                                                                               "units")},
                       "corpus_order": ["c3", "c6"] if order % 2 == 0 else ["c6", "c3"]})
            order += 1
    write_json(B / "questions.json", {"experiment_id": EXPERIMENT_ID, "fixed_at": now(), "selection_rule": SELECTION_RULE,
                                      "n": len(qs), "questions": qs})
    cfg = app_config()
    write_json(B / "experiment_manifest.json", {
        "experiment_id": EXPERIMENT_ID, "created_at": now(), "owner_approval": RISK_NOTE, "caps": CAPS,
        "est_tokens_per_call_for_cap_guard": EST_CALL,
        "model": cfg["models"]["primary"], "prompt_version": cfg["prompt_version"],
        "verifier": {"version": cfg["verification_policy_version"], **cfg["verification"]},
        "retrieval": {"setting": "B", "policy": cfg["retrieval_policy"],
                      "context_cap": cfg["pressure"]["L0"]["context_cap"]},
        "corpora": {n: cfg["corpus"]["corpora"][n] for n in ("c3", "c6")},
        "config_on_disk": {"corpus.active": cfg["corpus"]["active"], "corpus_version": cfg["corpus_version"]},
        "corpus_selection": "in memory: deep copy of app_config() with corpus.active/corpus_version set, one main.State "
                            "per corpus; both States share ONE governor and retry budget (one Groq account)",
        "execution": "sequential; per question both corpora back to back, order alternating c3-first / c6-first",
        "outcome_categories": ["correct_answer", "partially_correct", "incorrect", "unsupported", "correct_abstention",
                               "false_refusal", "invalid_citation", "provider_failure", "incomplete_budget_or_timeout"],
        "excluded_inputs": "the seven unknown-provenance artifacts and the three unknown-provenance stand-in fixtures "
                           "(experiments/phase11f/fixture_provenance_audit.json) are not read",
        "git_head": os.popen("git rev-parse HEAD").read().strip()})
    print("prepared", len(qs), "questions:", [q["question_id"] for q in qs])


# ---------------------------------------------------------------- 12B paired run
class BudgetedClient(ChatClient):
    """Refuses a call that could cross a cap (never sent); charges every attempt; writes raw request / response."""

    def __init__(self, *a, corpus_name: str, **k):
        super().__init__(*a, raw_dir=None, **k)
        self.corpus_name, self.qid, self.seq = corpus_name, None, 0

    def complete(self, model, messages, max_tokens, temperature, reasoning_effort, question_id=None):
        led = ledger()
        self.seq += 1
        tag = f"{self.qid}_{self.seq:02d}"
        body = {"model": model, "messages": messages, "stream": True, "max_completion_tokens": max_tokens,
                "temperature": temperature, "reasoning_effort": reasoning_effort, "include_reasoning": False}
        write_json(B / "raw_requests" / self.corpus_name / f"{tag}.json",
                   {"experiment_id": EXPERIMENT_ID, "corpus_version": self.corpus_name, "question_id": self.qid,
                    "seq": self.seq, "at": now(), "body": body})
        why = None
        if led["generation_requests"] + 1 > CAPS["generation_requests"]:
            why = "generation request cap"
        elif tokens_used(led) + EST_CALL > CAPS["total_tokens"]:
            why = "total token cap"
        if why:
            from relay.llm.client import Call
            call = Call(model, now(), time.perf_counter_ns())
            call.error, call.t_last_ns = f"phase12 {why}: call NOT sent", time.perf_counter_ns()
            led["refused_calls"] += 1
            save_ledger(led)
            write_json(B / "raw_responses" / self.corpus_name / f"{tag}.json", {"refused": why, "call": call_record(call)})
            raise LLMError("bad_request", call)
        call = None
        try:
            call = super().complete(model, messages, max_tokens, temperature, reasoning_effort, question_id)
            return call
        except LLMError as e:
            call = e.call
            raise
        finally:
            led = ledger()
            led["generation_requests"] += 1
            led["tokens_12b"] += ((call.usage or {}).get("total_tokens") or 0) if call else 0
            save_ledger(led)
            write_json(B / "raw_responses" / self.corpus_name / f"{tag}.json",
                       {"experiment_id": EXPERIMENT_ID, "corpus_version": self.corpus_name, "question_id": self.qid,
                        "seq": self.seq, "call": call_record(call) if call else None})


def config_for(name: str) -> dict:
    cfg = copy.deepcopy(app_config())
    cfg["corpus"]["active"], cfg["corpus_version"] = name, cfg["corpus"]["corpora"][name]["corpus_version"]
    return cfg


def paired() -> None:
    if os.environ.get("LLM_PROVIDER") != "groq" or os.environ.get("ALLOW_REAL") != "1":
        raise SystemExit("set LLM_PROVIDER=groq and ALLOW_REAL=1")
    a = json.loads((A / "connectivity_results.json").read_text(encoding="utf-8"))
    if not a["gate"]["passed"]:
        raise SystemExit("Phase 12A gate did not pass: 12B not started")
    from fastapi.testclient import TestClient
    from reconcile import holds, units_for

    from relay import main
    from relay.llm import prompts
    from relay.pipeline import answer_from_evidence as real_answer

    qs = json.loads((B / "questions.json").read_text(encoding="utf-8"))["questions"]
    states = {}
    for name in ("c3", "c6"):
        orig = main.app_config
        main.app_config = lambda c=config_for(name): c
        try:
            s = main.State()
        finally:
            main.app_config = orig
        assert s.corpus.name == name and s.cfg["corpus_version"] == name
        s.client = BudgetedClient(BASE, key(), s.cfg["timeouts"], "real", corpus_name=name)
        states[name] = s
    states["c6"].governor, states["c6"].budget = states["c3"].governor, states["c3"].budget  # one account
    assert app_config()["corpus"]["active"] == "c3"  # disk untouched
    captured = {}

    def capture(question, ev, llm, verifier, corpus_ids, *a, **k):
        r = real_answer(question, ev, llm, verifier, corpus_ids, *a, **k)
        captured["r"] = r
        return r
    main.answer_from_evidence = capture
    http = TestClient(main.app)
    paired_rows = []
    for q in qs:
        row = {"question_id": q["question_id"], "group": q["group"], "order": q["corpus_order"]}
        for name in q["corpus_order"]:
            s = states[name]
            led = ledger()
            if led["initial_generation_calls"] + 1 > CAPS["initial_generation_calls"] or led["generation_requests"] + 2 > CAPS["generation_requests"] \
                    or tokens_used(led) + 2 * EST_CALL > CAPS["total_tokens"]:
                rec = {"skipped": "budget: not enough headroom for an initial call plus one redraft", "ledger": led}
                write_json(B / name / f"{q['question_id']}.json", rec)
                row[name] = rec
                print(q["question_id"], name, "SKIPPED (budget)")
                continue
            main.STATE, main.TRACE_DIR = s, B / name / "traces"
            s.client.qid, s.client.seq = q["question_id"], 0
            captured.clear()
            t0 = time.perf_counter()
            events, ev_name = [], None
            with http.stream("POST", "/ask", json={"question": q["question"], "question_id": q["question_id"],
                                                   "request_id": f"p12-{name}-{q['question_id']}", "scenario": "phase12_live",
                                                   "deadline_ms": 180000}) as resp:
                http_status = resp.status_code
                for line in resp.iter_lines():
                    if line.startswith("event: "):
                        ev_name = line[7:]
                    elif line.startswith("data: "):
                        events.append((ev_name, json.loads(line[6:])))
            client_ms = round((time.perf_counter() - t0) * 1000, 1)
            tr = [json.loads(l) for l in sorted((B / name / "traces").glob("*.jsonl"))[-1]
                  .read_text(encoding="utf-8").splitlines()][-1]
            led = ledger()
            led["initial_generation_calls"] += 1 if tr.get("llm_calls") else 0  # the redraft is not an initial call
            save_ledger(led)
            r = captured.get("r")
            ev = r.evidence if r else None
            idx = s.indexes[256]
            passages = ev.passages if ev else []
            # identity: every evidence chunk the verifier got is this corpus's original chunk; prompts carry the original question
            reqs = sorted((B / "raw_requests" / name).glob(f"{q['question_id']}_*.json"))
            # each request's user messages joined (a redraft carries the question in its first user message)
            user_msgs = [" ".join(m["content"] for m in json.loads(p.read_text(encoding="utf-8"))["body"]["messages"]
                                   if m["role"] == "user") for p in reqs]
            identity = {"evidence_equals_selected_corpus": all(p["text"] == idx.chunks[p["chunk_id"]]["text"] for p in passages),
                        "original_question_in_every_prompt": all(q["question"] in m for m in user_msgs),
                        "expanded_query_not_in_prompt": not (tr.get("retrieval_query") and any(
                            tr["retrieval_query"] in m for m in user_msgs))}
            units = units_for(q, q["dataset"], idx.chunks) if q["answerable"] else []
            ids = [p["chunk_id"] for p in passages]
            done = next((d for n, d in events if n == "done"), {})
            rec = {"experiment_id": EXPERIMENT_ID, "timestamp": tr.get("start_time"), "corpus_version": name,
                   "model_id": s.primary["id"], "prompt_version": s.cfg["prompt_version"] if prompts.PROMPT_VERSION ==
                   s.cfg["prompt_version"] else "MISMATCH", "verifier_version": s.cfg["verification_policy_version"],
                   "retrieval_config": {"setting": "B", **s.cfg["retrieval_policy"]}, "question_id": q["question_id"],
                   "group": q["group"], "question": q["question"], "answerable": q["answerable"],
                   "http_status": http_status, "final_text": next((d["text"] for n, d in events if n == "content"), None),
                   "done": done, "client_ms": client_ms,
                   "evidence_ids": ids, "retrieval_query": tr.get("retrieval_query"),
                   "query_expansion": tr.get("query_expansion"), "caption_sharing": tr.get("caption_sharing"),
                   "gold_units_in_context": [any(holds(u, c, idx.chunks) for c in ids) for u in units],
                   "gold_units_complete_in_context": [any(holds(u, c, idx.chunks, True) for c in ids) for u in units],
                   "drafts": r.drafts if r else None,
                   "verification": [{"ok": rep.ok, "cited": rep.cited(), "failures": rep.failures()} for rep in r.reports]
                   if r else None,
                   "pipeline_status": r.status if r else None, "refusal_reason": r.refusal_reason if r else None,
                   "identity_checks": identity, "trace": tr,
                   "provider_token_usage": [c.get("usage") for c in tr.get("calls", [])]}
            write_json(B / name / f"{q['question_id']}.json", rec)
            row[name] = {k: rec[k] for k in ("done", "pipeline_status", "refusal_reason", "evidence_ids",
                                             "gold_units_in_context", "gold_units_complete_in_context")}
            print(q["question_id"], name, "|", done.get("status"), done.get("tier"), done.get("reason"), "| calls",
                  tr.get("llm_calls"), "| tok", tr.get("tokens_in"), "+", tr.get("tokens_out"), "| identity",
                  all(identity.values()), "| ledger", tokens_used(ledger()), ledger()["generation_requests"])
        paired_rows.append(row)
    write_json(B / "paired_results.json", {"experiment_id": EXPERIMENT_ID, "finished_at": now(), "rows": paired_rows,
                                           "ledger_after": ledger()})


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd not in ("connectivity", "prepare", "paired"):
        raise SystemExit("usage: run_phase12.py connectivity|prepare|paired")
    HERE.mkdir(parents=True, exist_ok=True)
    if not LEDGER.exists():
        save_ledger(ledger())
    if cmd == "connectivity":
        print(connectivity())
    elif cmd == "prepare":
        prepare()
    else:
        paired()
