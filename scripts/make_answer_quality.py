"""eval/answer_quality_results.csv and eval/answer_quality_report.md from the real answer-quality run
(results/raw/answer_quality/real_*.jsonl) and the grade sheets (eval/answer_quality_grades.csv,
eval/answer_quality_sentences.csv). A confirmed grade, when the owner has filled one in, overrides the assistant's.
Reads only; every number in the report is computed here.

    uv run python scripts/make_answer_quality.py
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

RAW_DIR = Path("results/raw/answer_quality")
GRADES = Path("eval/answer_quality_grades.csv")
SENTENCES = Path("eval/answer_quality_sentences.csv")
OUT_CSV = Path("eval/answer_quality_results.csv")
OUT_MD = Path("eval/answer_quality_report.md")
ANSWERABLE_LABELS = ["correct_full_cited", "correct_partially_cited", "partially_correct", "factually_wrong",
                     "false_refusal"]


def wilson(k: int, n: int, z: float = 1.96) -> str:
    if n == 0:
        return "-"
    p = k / n
    centre, half = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo, hi = (centre - half) / (1 + z * z / n), (centre + half) / (1 + z * z / n)
    return f"{100 * p:.1f}% ({max(0.0, 100 * lo):.0f}-{min(100.0, 100 * hi):.0f}%)"


def load(raw: Path | None = None) -> tuple[dict, list[dict], dict, list[dict]]:
    raw = raw or sorted(RAW_DIR.glob("real_*.jsonl"))[-1]
    recs = [json.loads(l) for l in raw.read_text(encoding="utf-8").splitlines()]
    head = next(r for r in recs if r["kind"] == "run")
    end = [r for r in recs if r["kind"] == "end"][-1]
    qs = [r for r in recs if r["kind"] == "question"]
    grades = {g["question_id"]: g for g in csv.DictReader(GRADES.open(encoding="utf-8"))}
    sents = list(csv.DictReader(SENTENCES.open(encoding="utf-8")))
    return head | {"raw_file": raw.as_posix(), "end": end}, qs, grades, sents


def label(g: dict) -> str:
    return g["confirmed_label"] or g["assistant_label"]


def rows(qs: list[dict], grades: dict) -> list[dict]:
    out = []
    for r in qs:
        g, fin = grades[r["question_id"]], r["final"] or {}
        lab = label(g)
        out.append({
            "question_id": r["question_id"], "question": r["question"], "category": r["category"],
            "answerable": r["answerable"], "split": r["split"], "expected_answer": r["expected_answer"],
            "gold_evidence": " | ".join(f"{e['section']}: {e['quote']}" for e in r["evidence"]),
            "retrieved_chunks": ";".join(p["chunk_id"] for p in r["retrieval"]["passages"]),
            "gold_in_context_any": r["retrieval"]["gold"]["any_in_context"],
            "final_status": fin.get("status"), "final_answer": fin.get("text"),
            "final_citations": ";".join(fin.get("citations") or []), "refusal_reason": fin.get("refusal_reason"),
            "drafts": len(r["drafts"]),
            "verifier_decisions": ";".join("declined" if d["verifier"] is None else
                                           ("accepted" if d["verifier"]["accepted"] else "rejected") for d in r["drafts"]),
            "should_answer": r["answerable"], "refusal_correct": None if fin.get("status") != "refused" else not r["answerable"],
            "grade": lab, "grade_source": "owner_confirmed" if g["confirmed_label"] else "assistant_graded",
            "failure_categories": g["failure_categories"], "rationale": g["assistant_rationale"],
            "calls": len(r["calls"]), "retries": r["retries"], "tokens_in": r["tokens"]["in"],
            "tokens_out": r["tokens"]["out"], "tokens_total": r["tokens"]["total"],
            "total_ms": r["timing_ms"]["total"], "retrieval_ms": r["timing_ms"]["retrieval"],
            "llm_ms": r["timing_ms"]["llm"], "verification_ms": r["timing_ms"]["verification"],
            "model": r["model"], "verification_policy_version": r["verification_policy_version"],
            "prompt_version": r["prompt_version"], "corpus_version": r["corpus_version"],
            "retrieval_setting": r["retrieval"]["setting"],
        })
    return out


def report(head: dict, out: list[dict], sents: list[dict]) -> str:
    ans = [r for r in out if r["answerable"]]
    una = [r for r in out if not r["answerable"]]
    n, c = len(out), Counter(r["grade"] for r in out)
    confirmed = sum(r["grade_source"] == "owner_confirmed" for r in out)
    delivered = [r for r in out if r["final_status"] == "answered"]
    tokens = sum(r["tokens_total"] for r in out)
    lat = sorted(r["total_ms"] for r in out)
    fails = Counter(f for r in out for f in r["failure_categories"].split(";") if f)
    drafts = {}
    for s in sents:
        drafts.setdefault((s["question_id"], s["draft_n"]), s)
    good = {"correct_full_cited", "correct_partially_cited"}
    vt = Counter()
    for (q, _), s in drafts.items():
        if s["draft_assistant_label"] == "model_declined":
            continue
        rejected = any(x["verifier_accepted"] == "False" for x in sents if (x["question_id"], x["draft_n"]) == (q, s["draft_n"]))
        kind = "correct" if s["draft_assistant_label"] in good else \
            "wrong" if s["draft_assistant_label"] == "factually_wrong" else "other"
        vt[(kind, "rejected" if rejected else "accepted")] += 1
    sup = Counter(s["assistant_citation_support"] for s in sents if s["assistant_citation_support"])
    full = sum(s["fullwidth_brackets"] == "True" for s in sents)
    L = [
        "# Answer-quality evaluation (real model)",
        "",
        f"Generated by scripts/make_answer_quality.py from `{head['raw_file']}` and the grade sheets; do not edit.",
        "",
        f"- Run: {head['started_at']} UTC, source **{head['source']}**, model {head['model']} "
        f"(reasoning_effort {head['reasoning_effort']}), retrieval setting B, verifier **{head['verification_policy_version']}** "
        f"(NLI {head['nli_model']}, threshold {head['entail_min']}), prompt {head['prompt_version']}, corpus {head['corpus_version']}.",
        f"- Set: {len(out)} questions ({len(ans)} answerable, {len(una)} unanswerable), set sha256 `{head['set_sha256'][:12]}…`, "
        f"questions sha256 `{head['questions_sha256'][:12]}…`.",
        f"- Budget: {head['end']['calls']} of {head['caps']['max_calls']} calls, {head['end']['tokens']:,} of "
        f"{head['caps']['max_tokens']:,} tokens; stopped by: {head['end']['stopped_by'] or 'nothing (completed)'}.",
        f"- **Grades: {n - confirmed} of {n} are assistant-graded and not yet confirmed by the owner.**",
        "",
        "## Results",
        "",
        "| Measure | Value (95% Wilson interval) |",
        "|---|---|",
        f"| Correct and cited (correct_full_cited / {n} asked) | {wilson(c['correct_full_cited'], n)} |",
        f"| Wrong (factually_wrong / {n} asked) | {wilson(c['factually_wrong'], n)} |",
        f"| Correctly refused (correct_refusal / {n} asked) | {wilson(c['correct_refusal'], n)} |",
        f"| Answerable answered fully correct ({len(ans)}) | {wilson(c['correct_full_cited'] + c['correct_partially_cited'], len(ans))} |",
        f"| Answerable falsely refused ({len(ans)}) | {wilson(c['false_refusal'], len(ans))} |",
        f"| Unanswerable correctly refused ({len(una)}) | {wilson(c['correct_refusal'], len(una))} |",
        f"| Replies delivered after verification | {len(delivered)} of {n}: "
        + ", ".join(f"{r['question_id']} graded {r['grade']}" for r in delivered) + " |",
        f"| Refusals (structural, model declined, verifier) | {sum(r['final_status'] == 'refused' for r in out)} of {n} |",
        f"| Tokens: total / per question / per delivered answer | {tokens:,} / {tokens / n:,.0f} / "
        f"{(tokens / len(delivered)) if delivered else 0:,.0f} |",
        f"| Total time per question p50 / max (ms) | {lat[len(lat) // 2]:,.0f} / {lat[-1]:,.0f} |",
        "",
        "Label counts: " + ", ".join(f"{k} {c[k]}" for k in ANSWERABLE_LABELS + ["correct_refusal"] if c[k]) + ".",
        "",
        "Failure categories (a question can have several): " + ", ".join(f"{k} {v}" for k, v in fails.most_common()) + ".",
        "",
        "## The verifier on real drafts (every draft graded as if it had been delivered)",
        "",
        f"- Correct drafts (fully or partly cited): rejected {vt[('correct', 'rejected')]}, accepted {vt[('correct', 'accepted')]}.",
        f"- Factually wrong drafts: rejected {vt[('wrong', 'rejected')]}, accepted {vt[('wrong', 'accepted')]}.",
        f"- Other drafts (partially correct, refusals): rejected {vt[('other', 'rejected')]}, accepted {vt[('other', 'accepted')]}.",
        f"- Sentences: {sum(sup.values())} graded; citation support " + ", ".join(f"{k} {v}" for k, v in sup.most_common()) +
        f"; {full} written with full-width brackets (【…】), which the citation parser does not read.",
        "",
        "## Per question",
        "",
        "| ID | Category | Answerable | Gold retrieved | Outcome | Drafts (verifier) | Grade | Failure | Tokens | ms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in out:
        L.append(f"| {r['question_id']} | {r['category']} | {'yes' if r['answerable'] else 'no'} | "
                 f"{'-' if r['gold_in_context_any'] is None else ('yes' if r['gold_in_context_any'] else 'no')} | "
                 f"{r['final_status']} | {r['verifier_decisions']} | {r['grade']} | {r['failure_categories'] or '-'} | "
                 f"{r['tokens_total']:,} | {r['total_ms']:,.0f} |")
    L += ["", "## Rationale per question (assistant-graded unless confirmed)", ""]
    L += [f"- **{r['question_id']}** ({r['grade']}): {r['rationale']}" for r in out]
    L += ["", "Per-sentence grades, verifier decisions and notes: eval/answer_quality_sentences.csv. "
          "Full drafts, retrieved passages and calls: the raw file above.", ""]
    return "\n".join(L)


def main() -> list[dict]:
    head, qs, grades, sents = load()
    out = rows(qs, grades)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    OUT_MD.write_text(report(head, out, sents), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(f"{OUT_CSV}: {len(main())} rows; {OUT_MD}")
