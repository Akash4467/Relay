"""Builds assessment/improvement_loop/datasets/datasets.json ONCE (refuses to overwrite). Questions, expected answers
and gold are copied verbatim from the eval files; nothing is edited. Membership rules are written into the file."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("assessment/improvement_loop/datasets/datasets.json")
SETS = {
    "A_fixed_regression": ["q01", "q02", "q03", "q04", "q05", "q06", "q07", "q08", "q09", "q10", "q11", "q12", "q13",
                           "q14", "q15"],
    "B_failure_focused": ["q02", "q07", "q11", "q15", "t01", "t05", "t10", "t13", "t29"],
    "C_unanswerable": ["q16", "q20", "q22", "q27", "q30"],
}
RULES = {
    "A_fixed_regression": "every answerable question of the original project set (eval/questions.jsonl), including the "
                          "three whose gold is not retrieved offline (q06, q10, q14); never modified during the loop",
    "B_failure_focused": "every question that failed on c3 in the Phase 12 live run (experiments/phase12, c3 records); "
                         "new failures found during the loop are added to the failure registry, not to this frozen list",
    "C_unanswerable": "original-set refusal questions over all three papers: q16 (AIAYN cost in dollars), q20 (GPT-4 MMLU), "
                      "q22 (commercial assistant beating Moshi), q27 (H-Net GPU-hours), q30 (false premise: Moshi chunking)",
}
if OUT.exists():
    raise SystemExit("datasets.json exists: the datasets are frozen")
orig = {json.loads(l)["question_id"]: json.loads(l) for l in open("eval/questions.jsonl", encoding="utf-8")}
table = {q["question_id"]: q for q in json.loads(Path("eval/phase7/table_question_set.json").read_text(encoding="utf-8"))["questions"]}
qs = []
for qid in dict.fromkeys(q for ids in SETS.values() for q in ids):
    src, ds = (orig[qid], "original") if qid in orig else (table[qid], "table")
    qs.append({"question_id": qid, "sets": [s for s, ids in SETS.items() if qid in ids], "dataset": ds,
               "question": src["question"], "answerable": src.get("answerable", True),
               "expected_answer": src.get("expected_answer"), "category": src.get("category") or src.get("type"),
               "gold": src.get("gold"), "evidence": src.get("evidence"), "paper": src.get("paper"),
               "table_id": src.get("table_id"),
               "label_provenance": "eval/questions.jsonl: project labels, gold quotes located by the assistant; NOT "
                                   "independently reviewed" if ds == "original" else
                                   "eval/phase7/table_question_set.json: PROVISIONAL assistant-authored labels; NOT "
                                   "independently reviewed"})
src_hash = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ("eval/questions.jsonl", "eval/phase7/table_question_set.json")}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sets": SETS,
                           "rules": RULES, "source_sha256": src_hash, "n_unique": len(qs), "questions": qs},
                          indent=1, ensure_ascii=False), encoding="utf-8")
print(len(qs), "questions;", {k: len(v) for k, v in SETS.items()})
