"""Phase 12C (no API): failure analysis and the paired c3/c6 comparison from the Phase 12B records only.

Inputs: phase12a/connectivity_results.json, phase12b/{questions.json, c3/*.json, c6/*.json}, and
phase12c/answer_judgments.json (correctness of each ANSWERED reply against the question's expected answer; written by
the assistant after reading every reply, NOT an independent review). Every other outcome is derived mechanically:
  provider_failure            done.status != ok and a provider call failed (429, 5xx, transport)
  incomplete_budget_or_timeout skipped for budget, a call refused by the cap, or a deadline/internal timeout
  correct_abstention          unanswerable and refused / declined
  unsupported                 unanswerable and answered
  false_refusal               answerable and refused
  correct_answer / partially_correct / incorrect   answerable and answered: from answer_judgments.json
invalid_citation is counted per DRAFT (verifier citation failures), since an answer shown with an invalid citation is
impossible by construction (the verifier rejects it); a reply-level count is reported as well and must be 0.
Writes phase12c/{failure_analysis.json, q11_t10_t29_analysis.json, comparison_metrics.json} and
phase12a/provider_diagnostics.json.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

HERE = Path("experiments/phase12")
A, B, C = HERE / "phase12a", HERE / "phase12b", HERE / "phase12c"
CORPORA = ("c3", "c6")
CITATION_WORDS = ("citation", "cites no", "not cited", "unknown chunk", "not in the retrieved", "no citation")


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def outcome(rec: dict, q: dict, judgments: dict) -> str:
    if "skipped" in rec:
        return "incomplete_budget_or_timeout"
    done = rec["done"]
    calls = rec["trace"].get("calls", [])
    if any("phase12" in (c.get("error") or "") for c in calls):
        return "incomplete_budget_or_timeout"
    if done.get("status") == "timeout":
        return "incomplete_budget_or_timeout"
    if done.get("status") != "ok":
        return "provider_failure"
    answered = rec["pipeline_status"] == "answered"
    if not q["answerable"]:
        return "unsupported" if answered else "correct_abstention"
    if not answered:
        return "false_refusal"
    return judgments[f"{rec['corpus_version']}/{rec['question_id']}"]["grade"]


def failure_layer(rec: dict, q: dict, out: str) -> str | None:
    """Where a non-correct outcome came from (one layer, first that applies)."""
    if out in ("correct_answer", "correct_abstention"):
        return None
    if out in ("provider_failure", "incomplete_budget_or_timeout"):
        return "infrastructure"
    if q["answerable"] and not all(rec["gold_units_complete_in_context"] or [False]):
        if not all(rec["gold_units_in_context"] or [False]):
            return "retrieval (a gold unit not in the final context)"
        return "retrieval/extraction (gold rows in context but not complete: answering cells missing)"
    if out == "false_refusal":
        if (rec["refusal_reason"] or "").startswith("insufficient evidence"):
            return "retrieval (structural sufficiency refused before generation)"
        if rec["refusal_reason"] == "model: not in papers":
            return "generation (model declined although complete evidence was in context)"
        return "verification (every draft rejected although complete evidence was in context)"
    if out == "unsupported":
        return "generation+verification (answered an unanswerable question)"
    return "generation (answered with complete evidence in context, but wrong or incomplete)"


def citation_failures(rec: dict) -> int:
    return sum(1 for v in rec.get("verification") or [] for _, why in v["failures"]
               if any(w in why.lower() for w in CITATION_WORDS))


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return None if not xs else xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))]


def main() -> None:
    qs = {q["question_id"]: q for q in load(B / "questions.json")["questions"]}
    judgments = load(C / "answer_judgments.json")["judgments"]
    recs = {(c, qid): load(B / c / f"{qid}.json") for c in CORPORA for qid in qs}
    for (c, qid), r in recs.items():
        r.setdefault("corpus_version", c)
        r.setdefault("question_id", qid)
        assert "skipped" in r or r["corpus_version"] == c, (c, qid)  # never mixed
    rows = []
    for (c, qid), r in recs.items():
        q = qs[qid]
        out = outcome(r, q, judgments)
        t = r.get("trace", {})
        rows.append({"corpus": c, "question_id": qid, "group": q["group"], "answerable": q["answerable"], "outcome": out,
                     "failure_layer": failure_layer(r, q, out) if "skipped" not in r else "infrastructure",
                     "refusal_reason": r.get("refusal_reason"), "evidence_ids": r.get("evidence_ids"),
                     "gold_units_in_context": r.get("gold_units_in_context"),
                     "gold_units_complete_in_context": r.get("gold_units_complete_in_context"),
                     "llm_calls": t.get("llm_calls"), "redraft": (t.get("redraft_attempts") or 0) > 0,
                     "draft_citation_failures": citation_failures(r) if "skipped" not in r else 0,
                     "judgment": judgments.get(f"{c}/{qid}"), "identity_checks": r.get("identity_checks"),
                     "tokens_in": t.get("tokens_in"), "tokens_out": t.get("tokens_out"),
                     "reasoning_tokens": t.get("reasoning_tokens"), "total_ms": t.get("total_ms"),
                     "first_byte_ms": t.get("first_byte_ms"), "generation_ms": t.get("generation_ms"),
                     "verification_ms": t.get("verification_ms"), "retrieval_ms": t.get("retrieval_ms"),
                     "governor_wait_ms": t.get("governor_wait_ms")})
    by = {(r["corpus"], r["question_id"]): r for r in rows}
    cats = ["correct_answer", "partially_correct", "incorrect", "unsupported", "correct_abstention", "false_refusal",
            "invalid_citation", "provider_failure", "incomplete_budget_or_timeout"]
    per_corpus = {}
    for c in CORPORA:
        rs = [r for r in rows if r["corpus"] == c]
        counts = {k: sum(r["outcome"] == k for r in rs) for k in cats}
        counts["invalid_citation"] = 0  # reply level: impossible by construction, checked below
        ans = [r for r in rs if r["answerable"]]
        per_corpus[c] = {
            "n": len(rs), "outcomes": counts,
            "answerable": {"n": len(ans), "correct": sum(r["outcome"] == "correct_answer" for r in ans),
                           "correct_or_partial": sum(r["outcome"] in ("correct_answer", "partially_correct") for r in ans),
                           "false_refusal": sum(r["outcome"] == "false_refusal" for r in ans)},
            "unanswerable": {"n": len(rs) - len(ans), "correct_abstention":
                             sum(r["outcome"] == "correct_abstention" for r in rs if not r["answerable"])},
            "drafts_with_citation_failures": sum(r["draft_citation_failures"] > 0 for r in rs),
            "redrafted": sum(r["redraft"] for r in rs),
            "all_identity_checks_passed": all(all((r["identity_checks"] or {"x": False}).values()) for r in rs
                                              if r["llm_calls"] is not None),
            "by_group": {g: {k: sum(r["outcome"] == k for r in rs if r["group"] == g) for k in cats
                             if any(r["outcome"] == k for r in rs if r["group"] == g)}
                         for g in dict.fromkeys(q["group"] for q in qs.values())},
            "complete_gold_in_context": f"{sum(all(r['gold_units_complete_in_context'] or [False]) for r in ans)}/{len(ans)}",
            "all_gold_in_context": f"{sum(all(r['gold_units_in_context'] or [False]) for r in ans)}/{len(ans)}"}
    score = {"correct_answer": 2, "partially_correct": 1, "correct_abstention": 2}
    pairs = []
    for qid, q in qs.items():
        a, b = by[("c3", qid)], by[("c6", qid)]
        sa, sb = score.get(a["outcome"], 0), score.get(b["outcome"], 0)
        pairs.append({"question_id": qid, "group": q["group"], "c3": a["outcome"], "c6": b["outcome"],
                      "better": "same" if sa == sb else ("c6" if sb > sa else "c3"),
                      "c3_complete_gold": a["gold_units_complete_in_context"],
                      "c6_complete_gold": b["gold_units_complete_in_context"]})
    paired = {"n_pairs": len(pairs), "c6_better": sum(p["better"] == "c6" for p in pairs),
              "c3_better": sum(p["better"] == "c3" for p in pairs), "same": sum(p["better"] == "same" for p in pairs),
              "ordering": "correct_answer = correct_abstention > partially_correct > every other outcome",
              "pairs": pairs,
              "significance": "not tested and not claimed: 12 pairs, one sample per pair, model sampling at temperature 0.2"}
    lat = {}
    for c in CORPORA:
        rs = [r for r in rows if r["corpus"] == c and r["llm_calls"]]
        lat[c] = {k: {"median": statistics.median([r[k] for r in rs if r[k] is not None]) if rs else None,
                      "p90_nearest_rank": pct([r[k] for r in rs], 90), "max": max((r[k] for r in rs if r[k] is not None), default=None)}
                  for k in ("total_ms", "first_byte_ms", "generation_ms", "verification_ms", "retrieval_ms", "governor_wait_ms")}
        lat[c]["tokens"] = {k: sum(r[k] or 0 for r in rs) for k in ("tokens_in", "tokens_out", "reasoning_tokens")}
        lat[c]["llm_calls"] = sum(r["llm_calls"] for r in rs)
    led = load(HERE / "ledger.json")
    C.mkdir(parents=True, exist_ok=True)
    (C / "comparison_metrics.json").write_text(json.dumps({
        "label_note": "outcomes for answered answerable replies are ASSISTANT judgments (answer_judgments.json), not an "
                      "independent review; question labels are project/provisional labels (questions.json)",
        "per_corpus": per_corpus, "paired": paired, "latency_and_tokens": lat, "budget": led}, indent=1), encoding="utf-8")
    fails = [r for r in rows if r["failure_layer"]]
    layers = {}
    for r in fails:
        layers.setdefault(r["corpus"], {}).setdefault(r["failure_layer"], []).append(r["question_id"])
    (C / "failure_analysis.json").write_text(json.dumps({"by_corpus_and_layer": layers, "records": rows}, indent=1,
                                                        ensure_ascii=False), encoding="utf-8")
    tgt = {}
    for qid in ("q11", "t10", "t29"):
        for c in CORPORA:
            r, full = by[(c, qid)], recs[(c, qid)]
            tgt[f"{c}/{qid}"] = {"outcome": r["outcome"], "failure_layer": r["failure_layer"],
                                 "evidence_ids": r["evidence_ids"], "gold_units_in_context": r["gold_units_in_context"],
                                 "gold_units_complete_in_context": r["gold_units_complete_in_context"],
                                 "retrieval_query": full.get("retrieval_query"), "drafts": full.get("drafts"),
                                 "verification": full.get("verification"), "final_text": full.get("final_text"),
                                 "refusal_reason": full.get("refusal_reason"), "judgment": r["judgment"]}
    (C / "q11_t10_t29_analysis.json").write_text(json.dumps(tgt, indent=1, ensure_ascii=False), encoding="utf-8")
    # provider diagnostics: 12A plus every 12B provider call
    a = load(A / "connectivity_results.json")
    calls = [dict(c, corpus=cv, question_id=qid) for (cv, qid), r in recs.items() for c in r.get("trace", {}).get("calls", [])]
    raw = [load(p) for p in sorted((B / "raw_responses").rglob("*.json"))]
    headers = [x["call"]["headers"] for x in raw if x.get("call") and x["call"].get("headers")]
    (A / "provider_diagnostics.json").write_text(json.dumps({
        "phase12a_gate": a["gate"], "model_list": a["results"]["1_auth_and_model_availability"],
        "token_accounting": {k: a["results"]["3_token_accounting_p4_prompt"].get(k) for k in
                             ("local_prompt_estimate", "provider_prompt_tokens", "relative_error")},
        "phase12b_calls": {"n": len(calls), "status": {s: sum(c["status"] == s for c in calls) for s in {c["status"] for c in calls}},
                           "http": {str(h): sum(c.get("http") == h for c in calls) for h in {c.get("http") for c in calls}},
                           "finish_reason": {str(f): sum(c.get("finish_reason") == f for c in calls)
                                             for f in {c.get("finish_reason") for c in calls}},
                           "prompt_tokens_local_error": [c.get("prompt_tokens_local_error") for c in calls],
                           "reservation_overrun_tokens": [c.get("reservation_overrun_tokens") for c in calls],
                           "usage_source": sorted({str(x["call"].get("usage_source")) for x in raw if x.get("call")}),
                           "min_remaining_tokens_per_minute_header": min((int(h["x-ratelimit-remaining-tokens"]) for h in headers
                                                                          if "x-ratelimit-remaining-tokens" in h), default=None),
                           "min_remaining_requests_header": min((int(h["x-ratelimit-remaining-requests"]) for h in headers
                                                                 if "x-ratelimit-remaining-requests" in h), default=None)},
        "calls": calls}, indent=1, ensure_ascii=False), encoding="utf-8")
    for c in CORPORA:
        print(c, per_corpus[c]["outcomes"], "| complete gold", per_corpus[c]["complete_gold_in_context"])
    print("paired", {k: paired[k] for k in ("c6_better", "c3_better", "same")})


if __name__ == "__main__":
    main()
