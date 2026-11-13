"""One request end to end: question -> retrieval -> structural check -> LLM -> citation/evidence verification
-> grounded answer or refusal. Prints the answer, the verifier's per-sentence decisions and where the time went,
and appends a summary to results/raw/e2e/{date}.jsonl.

    LLM_PROVIDER=groq uv run --env-file .env python scripts/ask.py --question-id q05
    LLM_PROVIDER=groq uv run --env-file .env python scripts/ask.py --question "What is Mimi?"

Each run is 1 hosted call, 2 if the verifier asks for a redraft. Test-split questions are refused: the test
split is measured once, after the retrieval setting is chosen.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from relay.config import app_config, retrieval_settings
from relay.eval.retrieval_recall import load_questions
from relay.governor.governor import Governor
from relay.llm.client import LLMError, from_env
from relay.llm.hosted import GovernorReject, HostedLLM
from relay.pipeline import Verifier, answer_from_evidence
from relay.retrieval.index_build import load_model
from relay.retrieval.rerank import load_reranker
from relay.retrieval.search import Index, search
from relay.verify import nli

E2E_DIR = Path("results/raw/e2e")


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--question-id")
    g.add_argument("--question")
    ap.add_argument("--setting", default="B")
    ap.add_argument("--no-redraft", action="store_true")
    args = ap.parse_args()

    qid, question = args.question_id, args.question
    if qid:
        q = {x["question_id"]: x for x in load_questions(None)}[qid]
        if q["split"] == "test":
            raise SystemExit(f"{qid} is a test-split question; the test split is not used before the setting is chosen")
        question = q["question"]

    cfg = app_config()
    setting = retrieval_settings()[args.setting]
    client = from_env(cfg["timeouts"])
    print(f"provider: {client.source}  setting: {args.setting}  nli: {cfg['verification']['nli_model']}")
    index = Index(setting.piece_tokens, load_model())
    reranker = load_reranker() if setting.rerank else None
    ver = cfg["verification"]
    verifier = Verifier(nli.load_nli(ver["nli_model"]), ver["entail_min"], ver["entail_min_flagged"])
    primary = cfg["models"]["primary"]
    reserve = int(cfg["reservation"]["output_p95_tokens"][primary["id"]] * (1 + cfg["reservation"]["safety_margin"]))
    llm = HostedLLM(client, Governor.from_config(), primary, reserve,
                    request_overhead=cfg["reservation"]["request_overhead_tokens"], question_id=qid)

    t0 = time.perf_counter_ns()
    ev = search(question, setting, cfg["pressure"]["L0"]["context_cap"], index, reranker)
    t_retrieved = time.perf_counter_ns()
    error = None
    try:
        result = answer_from_evidence(question, ev, llm, verifier, set(index.chunks), allow_redraft=not args.no_redraft)
    except (LLMError, GovernorReject) as e:
        result, error = None, str(e)
    t_done = time.perf_counter_ns()

    retrieval_ms = round((t_retrieved - t0) / 1e6, 1)
    llm_ms = round(sum(c.get("total_ms") or 0 for c in llm.calls), 1)
    total_ms = round((t_done - t0) / 1e6, 1)
    summary = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source": client.source,
        "question_id": qid, "question": question, "setting": args.setting, "policy": ev.policy,
        "passages": [p["chunk_id"] for p in ev.passages], "context_tokens": ev.tokens,
        "status": result.status if result else "error", "error": error,
        "refusal_reason": result.refusal_reason if result else None,
        "answer": result.text if result else None, "citations": result.citations if result else [],
        "llm_calls": llm.calls, "timing_ms": {"retrieval": retrieval_ms, "llm": llm_ms,
                                              "verification_and_rest": round(total_ms - retrieval_ms - llm_ms, 1),
                                              "total": total_ms},
        "verify": [[{"sentence": s.sentence, "ok": s.ok, "absence": s.absence, "entailment": s.entailment,
                     "context": s.context, "flags": s.flags, "reasons": s.reasons}
                    for s in r.sentences] for r in (result.reports if result else [])],
    }
    E2E_DIR.mkdir(parents=True, exist_ok=True)
    with (E2E_DIR / f"{summary['at'][:10]}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"\nQ: {question}\npassages ({ev.policy}, {ev.tokens} tokens): {summary['passages']}")
    for i, report in enumerate(summary["verify"]):
        print(f"\n-- draft {i + 1} verification")
        for s in report:
            ent = "-" if s["entailment"] is None else f"{s['entailment']:.2f}"
            print(f"  [{'ok ' if s['ok'] else 'NO '}] entail={ent} {s['sentence']}")
            for reason in s["reasons"]:
                print(f"         reason: {reason}")
    print(f"\n== {summary['status'].upper()}" + (f" ({summary['refusal_reason'] or error})" if summary["status"] != "answered" else ""))
    print(summary["answer"] or error)
    for c in llm.calls:
        print(f"\nLLM call: {c.get('status')} http={c.get('http')} in={c.get('tokens_in')} out={c.get('tokens_out')} "
              f"ttft={c.get('model_ttft_ms')}ms total={c.get('total_ms')}ms usage_estimated={c.get('usage_estimated')}")
    print(f"timing: {summary['timing_ms']}")


if __name__ == "__main__":
    main()
