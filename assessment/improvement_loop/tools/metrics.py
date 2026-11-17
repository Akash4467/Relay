"""Metrics for a set of live runs (no API). usage: metrics.py <out.json> <judgments.json> <run dir> [<run dir> ...]
A run dir holds records/*.json (Phase 12 c3 records or live_eval records). When a question appears in several run
dirs, the LAST dir given wins (so a completion run can follow a run whose question was cut by the budget cap).

Outcomes: as in experiments/phase12/analyse_phase12.py (correct_answer / partially_correct / incorrect from the
judgments file for answered answerable replies; everything else mechanical). Claim-level entailment is recomputed by
replaying the production verifier v4.2 (or the record's verifier module) on every draft: the replay reproduced
33/33 recorded verdicts (iteration_001/offline_verifier_comparison.json), so it is the recorded decision.
Every rate is reported as {n, d}; nothing is pooled across questions without its denominator.
"""
from __future__ import annotations

import importlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, ".")

LOOP = Path("assessment/improvement_loop")


def frac(n: int, d: int) -> dict:
    return {"n": n, "d": d, "pct": round(100 * n / d, 1) if d else None}


def pct(xs: list, p: int):
    xs = sorted(x for x in xs if x is not None)
    return None if not xs else xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))]


def outcome(r: dict, answerable: bool, judgments: dict, run_label: str) -> str:
    calls = r.get("trace", {}).get("calls", [])
    if any("call NOT sent" in (c.get("error") or "") for c in calls) or (r.get("done") or {}).get("status") == "timeout":
        return "incomplete_budget_or_timeout"
    if (r.get("done") or {}).get("status") != "ok":
        return "provider_failure"
    answered = r["pipeline_status"] == "answered"
    if not answerable:
        return "unsupported" if answered else "correct_abstention"
    if not answered:
        return "false_refusal"
    return judgments[f"{run_label}/{r['question_id']}"]["grade"]


def main(out: Path, judgments_path: Path, runs: list[str]) -> dict:
    from relay.config import app_config
    from relay.verify import nli
    import os
    ds = {q["question_id"]: q for q in json.loads(Path(os.environ.get("DATASET_FILE", LOOP / "datasets" / "datasets.json")).read_text(encoding="utf-8"))["questions"]}
    judgments = json.loads(judgments_path.read_text(encoding="utf-8"))["judgments"]
    recs = {}
    for run in runs:
        for f in sorted(Path(run).glob("records/*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            if r["question_id"] in ds:
                recs[r["question_id"]] = (Path(run).name, r)
    cfg = app_config()
    t = cfg["verification"]["entail_min"]
    model = nli.load_nli(cfg["verification"]["nli_model"])
    proc = {"c3": "data/processed", "c6": "experiments/phase10/corpus/c6/processed"}
    by_corpus = {c: {json.loads(l)["chunk_id"]: json.loads(l) for l in open(f"{d}/chunks_256.jsonl", encoding="utf-8")}
                 for c, d in proc.items()}
    rows = []
    for qid, (label, r) in sorted(recs.items()):
        q = ds[qid]
        tr = r.get("trace", {})
        chunks = by_corpus[r.get("corpus_version", "c3")]
        vmod = r.get("verifier_version", "v4.2")
        verify = importlib.import_module("relay.verify.verifier" if vmod == "v4.2" else
                                         {"v4.3c": "relay.candidate_v4_3.verifier", "v4.4c": "relay.candidate_v4_4.verifier", "v4.6c": "relay.candidate_v4_6.verifier", "v4.7c": "relay.candidate_v4_6.verifier", "v4.8c": "relay.candidate_v4_6.verifier",  # v4.7c = v4.6c per sentence; its guard is answer-level
                                          "v4.5c": "relay.candidate_v4_5.verifier"}[vmod]).verify
        ent = []
        for d in r.get("drafts") or []:
            if d.strip() == "NOT_IN_PAPERS":
                continue
            rep = verify(d, [chunks[c] for c in r["evidence_ids"]], set(chunks), model, t, t)
            ent += [{"ok": s.ok, "entailment": s.entailment, "absence": s.absence} for s in rep.sentences]
        cited = [c for v in (r.get("verification") or []) if v["ok"] for c in v["cited"]]
        ok_calls = [c for c in tr.get("calls", []) if c.get("status") == "ok"]
        rows.append({
            "question_id": qid, "run": label, "sets": q["sets"], "answerable": q["answerable"],
            "outcome": outcome(r, q["answerable"], judgments, label),
            "answered": r.get("pipeline_status") == "answered", "refusal_reason": r.get("refusal_reason"),
            "redrafted": (tr.get("redraft_attempts") or 0) > 0, "llm_calls": tr.get("llm_calls"),
            "evidence_complete": all(r.get("gold_units_complete_in_context") or [False]) if q["answerable"] else None,
            "evidence_all_in_context": all(r.get("gold_units_in_context") or [False]) if q["answerable"] else None,
            "shown_citations": len(cited), "shown_citations_valid": sum(c in r["evidence_ids"] and c in chunks for c in cited),
            "sentences": ent, "ttft_ms": tr.get("model_ttft_ms"), "total_ms": tr.get("total_ms"),
            "governor_wait_ms": tr.get("governor_wait_ms"),
            "tokens_in": sum(c.get("tokens_in") or 0 for c in ok_calls), "tokens_out": sum(c.get("tokens_out") or 0 for c in ok_calls),
            "provider_errors": sum(c.get("status") not in ("ok",) and "call NOT sent" not in (c.get("error") or "")
                                   for c in tr.get("calls", [])),
            "timeouts": sum("timeout" in (c.get("error") or "") for c in tr.get("calls", [])),
            "cap_refused_calls": sum("call NOT sent" in (c.get("error") or "") for c in tr.get("calls", []))})

    def block(sel: list[dict]) -> dict:
        ans, una = [r for r in sel if r["answerable"]], [r for r in sel if not r["answerable"]]
        claims = [s for r in sel for s in r["sentences"] if not s["absence"] and s["entailment"] is not None]
        red = [r for r in sel if r["redrafted"]]
        return {
            "n_questions": len(sel),
            "correct_answer": frac(sum(r["outcome"] == "correct_answer" for r in ans), len(ans)),
            "partially_correct": frac(sum(r["outcome"] == "partially_correct" for r in ans), len(ans)),
            "wrong_answer": frac(sum(r["outcome"] == "incorrect" for r in ans), len(ans)),
            "false_refusal": frac(sum(r["outcome"] == "false_refusal" for r in ans), len(ans)),
            "correct_refusal": frac(sum(r["outcome"] == "correct_abstention" for r in una), len(una)),
            "unsupported_answer": frac(sum(r["outcome"] == "unsupported" for r in una), len(una)),
            "incomplete_budget_or_timeout": frac(sum(r["outcome"] == "incomplete_budget_or_timeout" for r in sel), len(sel)),
            "citation_validity_shown_answers": frac(sum(r["shown_citations_valid"] for r in sel),
                                                    sum(r["shown_citations"] for r in sel)),
            "evidence_complete_in_context": frac(sum(bool(r["evidence_complete"]) for r in ans), len(ans)),
            "claim_sentences_accepted": frac(sum(s["ok"] for s in claims), len(claims)),
            "claim_entailment_median": round(statistics.median(s["entailment"] for s in claims), 3) if claims else None,
            "redraft_success": frac(sum(r["answered"] for r in red), len(red)),
            "ttft_ms": {"n": sum(r["ttft_ms"] is not None for r in sel), "p50": pct([r["ttft_ms"] for r in sel], 50),
                        "p95": pct([r["ttft_ms"] for r in sel], 95)},
            "total_ms": {"n": sum(r["total_ms"] is not None for r in sel), "p50": pct([r["total_ms"] for r in sel], 50),
                         "p95": pct([r["total_ms"] for r in sel], 95)},
            "governor_wait_ms_p50": pct([r["governor_wait_ms"] for r in sel], 50),
            "tokens_per_question": {"in_p50": pct([r["tokens_in"] for r in sel if r["llm_calls"]], 50),
                                    "out_p50": pct([r["tokens_out"] for r in sel if r["llm_calls"]], 50),
                                    "in_total": sum(r["tokens_in"] for r in sel), "out_total": sum(r["tokens_out"] for r in sel)},
            "llm_calls": sum(r["llm_calls"] or 0 for r in sel),
            "provider_errors": sum(r["provider_errors"] for r in sel), "timeouts": sum(r["timeouts"] for r in sel),
            "cap_refused_calls": sum(r["cap_refused_calls"] for r in sel)}
    res = {"runs": runs, "judgments": str(judgments_path), "p95_note": "nearest-rank percentile",
           "all": block(rows), "by_set": {s: block([r for r in rows if s in r["sets"]])
                                         for s in ("A_fixed_regression", "B_failure_focused", "C_unanswerable")},
           "rows": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    return res


if __name__ == "__main__":
    r = main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:])
    for k, v in r["all"].items():
        print(k, v)
    for s, b in r["by_set"].items():
        print(s, {k: b[k] for k in ("n_questions", "correct_answer", "partially_correct", "wrong_answer", "false_refusal",
                                    "correct_refusal", "unsupported_answer")})
