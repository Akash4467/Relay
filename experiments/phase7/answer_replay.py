"""Phase 7, stage 5 (offline): answer-quality comparison c3 vs c4 with saved drafts only (no generation, no API).

c3 = the historical live run (results/raw/answer_quality/real_2026-11-13T18-06-56Z.jsonl; verifier v4, setting B,
corpus c3), graded in eval/answer_quality_grades.csv (assistant-graded, unconfirmed). c4 = a COUNTERFACTUAL replay:
the same saved drafts, verified by frozen v4 against the c4 setting-B context of the same question (from
eval/phase7/raw/retrieval_rows.jsonl). The drafts were written from c3 contexts, so this shows whether c4 alone would
change a verdict on the same drafts; it is not the answer quality the model would reach with c4 (that is pending: it
needs new generation). Outcome rule as in the pipeline: first accepted draft is delivered; a saved redraft counts only
if v4 rejected draft 1 for the same reasons as live; otherwise "pending_needs_generation".
The repaired-table set has no model output at all: every row of eval/phase7/table_question_grades.csv is pending.
Writes eval/phase7/answer_quality_comparison.csv, eval/phase7/table_question_grades.csv,
eval/phase7/raw/answer_replay.jsonl (new files only)."""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from relay.config import app_config
from relay.verify import nli
from relay.verify.verifier import verify

cfg = app_config()
T = cfg["verification"]["entail_min"]
NLI = nli.load_nli(cfg["verification"]["nli_model"])
RUN = Path("results/raw/answer_quality/real_2026-11-13T18-06-56Z.jsonl")
C4 = {json.loads(l)["chunk_id"]: json.loads(l) for l in open("data/corpus/c4/processed/chunks_256.jsonl", encoding="utf-8")}
C3 = {json.loads(l)["chunk_id"]: json.loads(l) for l in open("data/processed/chunks_256.jsonl", encoding="utf-8")}
RETR = [json.loads(l) for l in open("eval/phase7/raw/retrieval_rows.jsonl", encoding="utf-8")]
ctx4 = {r["question_id"]: r["context"] for r in RETR if r["dataset"] == "original" and r["corpus"] == "c4" and r["setting"] == "B"}
grades = {g["question_id"]: g for g in csv.DictReader(open("eval/answer_quality_grades.csv", encoding="utf-8"))}
draft_labels = {(s["question_id"], s["draft_n"]): s["draft_assistant_label"]
                for s in csv.DictReader(open("eval/answer_quality_sentences.csv", encoding="utf-8"))}
OUTCOME = {"correct_full_cited": "1 correct answer with sufficient evidence", "correct_refusal": "2 correct refusal",
           "false_refusal": "3 false refusal", "correct_partially_cited": "4 unsupported answer (partly cited)",
           "partially_correct": "4 unsupported answer (incomplete)", "factually_wrong": "5 incorrect answer delivered"}
CAUSE = {"retrieval_miss": "6 retrieval failure", "verifier_false_rejection": "7 verification failure",
         "citation_error": "7 verification failure", "verifier_false_acceptance": "7 verification failure",
         "model_error": "8 generation failure", "model_declined": "8 generation failure"}

rows, raw = [], []
for r in map(json.loads, RUN.read_text(encoding="utf-8").splitlines()):
    if r.get("kind") != "question":
        continue
    q = r["question_id"]
    passages = [C4[c] for c in ctx4[q]]
    dec = []
    for d in r["drafts"]:
        if d["verifier"] is None:
            dec.append(None)
            continue
        rep = verify(d["text"], passages, set(C4), NLI, T, T)
        live = [s["reasons"] for s in d["verifier"]["sentences"]]
        cited = sorted({c for s in d["verifier"]["sentences"] for c in s["citations"]})
        dec.append({"n": d["n"], "c4_accepted": rep.ok, "c3_live_accepted": d["verifier"]["accepted"],
                    "same_reasons_as_live": [s.reasons for s in rep.sentences] == live,
                    "cited": cited, "cited_in_c4_context": [c for c in cited if c in ctx4[q]],
                    "cited_text_unchanged": all(c in C4 and c in C3 and C4[c]["text"] == C3[c]["text"] for c in cited),
                    "c4_reasons": [s.reasons for s in rep.sentences]})
    # pipeline outcome under c4
    c4_outcome, c4_grade = None, None
    for i, x in enumerate(dec):
        if x is None:
            c4_outcome, c4_grade = "refused (model declined)", "false_refusal" if r["answerable"] else "correct_refusal"
            break
        if x["c4_accepted"]:
            c4_outcome, c4_grade = f"delivered draft {x['n']}", draft_labels[(q, str(x["n"]))]
            break
        last = i == len(dec) - 1
        if i == 0 and len(dec) == 1:
            c4_outcome, c4_grade = ("draft 1 rejected; no saved redraft", "pending_needs_generation") \
                if r["final"]["status"] == "answered" else ("refused", "false_refusal" if r["answerable"] else "correct_refusal")
            break
        if i == 0 and not x["same_reasons_as_live"]:
            c4_outcome, c4_grade = "draft 1 rejected for other reasons; saved redraft not valid", "pending_needs_generation"
            break
        if last:
            c4_outcome, c4_grade = "refused (verification failed)", "false_refusal" if r["answerable"] else "correct_refusal"
    g = grades[q]
    c3_grade = g["confirmed_label"] or g["assistant_label"]
    causes = sorted({CAUSE[c] for c in g["failure_categories"].split(";") if c})
    rows.append({"question_id": q, "answerable": r["answerable"], "c3_context_B": ";".join(p["chunk_id"] for p in r["retrieval"]["passages"]),
                 "c4_context_B": ";".join(ctx4[q]), "context_identical": [p["chunk_id"] for p in r["retrieval"]["passages"]] == ctx4[q],
                 "c3_final_live": r["final"]["status"], "c3_grade": c3_grade, "c3_outcome": OUTCOME.get(c3_grade, c3_grade),
                 "c3_failure_causes": ";".join(causes), "c3_grade_source": "owner_confirmed" if g["confirmed_label"] else "assistant_graded (unconfirmed)",
                 "c4_replay_outcome": c4_outcome, "c4_replay_grade": c4_grade,
                 "c4_outcome": OUTCOME.get(c4_grade, "pending (needs new generation)"),
                 "verdicts_changed": sum(x is not None and x["c4_accepted"] != x["c3_live_accepted"] for x in dec),
                 "tokens_total_c3_live": r["tokens"]["total"], "tokens_c4": "not applicable (no generation)",
                 "note": "c4 column is a counterfactual replay of c3-era drafts, not new answers"})
    raw.append({"question_id": q, "drafts": dec, **{k: rows[-1][k] for k in ("c4_replay_outcome", "c4_replay_grade")}})

Path("eval/phase7/raw").mkdir(parents=True, exist_ok=True)
with open("eval/phase7/raw/answer_replay.jsonl", "x", encoding="utf-8") as f:
    for x in raw:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
with open("eval/phase7/answer_quality_comparison.csv", "x", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

TQ = json.loads(Path("eval/phase7/table_question_set.json").read_text(encoding="utf-8"))["questions"]
ev = {(r["corpus"], r["setting"], r["question_id"]): r for r in RETR if r["dataset"] == "table"}
with open("eval/phase7/table_question_grades.csv", "x", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["question_id", "table", "type", "answerable", "expected_answer", "evidence_complete_c3_B",
                "evidence_complete_c4_B", "answer_c3", "answer_c4", "grade_c3", "grade_c4", "grader", "human_review"])
    for q in TQ:
        a, b = ev[("c3", "B", q["question_id"])], ev[("c4", "B", q["question_id"])]
        w.writerow([q["question_id"], f"{q['paper']}:{q['table_id']}", q["type"], q["answerable"], q["expected_answer"],
                    a.get("evidence_complete", "n/a"), b.get("evidence_complete", "n/a"),
                    "PENDING: no model output", "PENDING: no model output", "pending", "pending", "none yet", "pending"])
for x in rows:
    print(f"{x['question_id']}: ctx same {x['context_identical']!s:5} | c3 {x['c3_grade']:<20} | c4 replay {x['c4_replay_outcome']} -> {x['c4_replay_grade']} | verdicts changed {x['verdicts_changed']}")
