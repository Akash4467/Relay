"""Applies the v4.7c paper-coverage guard to every answer SHOWN in every recorded live run (final accepted text,
re-parsed with the production citation parser; all its sentences were accepted). Rejection-only: the question is
which shown answers it would block. Writes replay_v47_guard.json."""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from relay.candidate_v4_7.verifier import uncovered  # noqa: E402
from relay.verify import citations  # noqa: E402
from relay.verify.verifier import Report  # noqa: E402

files = sorted(glob.glob("experiments/phase12/phase12b/c*/*.json") + glob.glob("assessment/improvement_loop/baseline/*/records/*.json")
               + glob.glob("assessment/improvement_loop/iterations/*/live/records/*.json") + glob.glob("assessment/e2e/live/*/records/*.json"))
rows = []
for f in files:
    r = json.loads(Path(f).read_text(encoding="utf-8"))
    if r.get("pipeline_status") != "answered":
        continue
    text = citations.normalize_brackets(r["drafts"][-1])
    sents = citations.check(text, set(r["evidence_ids"]), set(r["evidence_ids"]) | {c for c in r["evidence_ids"]})
    miss = uncovered(r["question"], Report(sents))
    rows.append({"file": f, "question_id": r["question_id"], "blocked": bool(miss), "uncovered": miss, "answer": text})
blocked = [x for x in rows if x["blocked"]]
Path("assessment/e2e/fix/replay_v47_guard.json").write_text(json.dumps({"n_shown_answers": len(rows), "blocked": blocked, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
print("shown answers:", len(rows), "| blocked by guard:", len(blocked))
for x in blocked:
    print("  BLOCKED", x["question_id"], x["uncovered"], "|", x["file"].split("/")[-3] if "records" in x["file"] else x["file"].split("/")[-2], "|", x["answer"][:120])
