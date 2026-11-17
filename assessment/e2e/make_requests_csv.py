"""Request-level CSV of one live run, built ONLY from its raw records (no value is estimated or filled in):
  raw_responses/<qid>_<seq>.json   one per generation call: status, timings, usage, finish reason
  raw_requests/<qid>_<seq>.json    the matching request body (model, max tokens, temperature)
  run.json                         the run's model-availability check (a GET, no tokens; the raw records hold no timing
                                   for it, so its timing columns are left empty)
  ../<run>_grades.json             the question's grade (graded per question, so every request of a question carries it)
usage: make_requests_csv.py <run dir> <out.csv>      e.g. assessment/e2e/live/E25_V48 assessment/e2e/live/E25_V48_requests.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

COLS = ["run", "request", "question_id", "call_in_question", "call_kind", "started_at", "http_status", "finish_reason",
        "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens", "first_byte_ms", "first_content_ms",
        "total_ms", "model", "max_completion_tokens", "temperature", "corpus", "retrieval", "prompt", "verifier_module",
        "question_grade"]


def main(run_dir: Path, out: Path) -> dict:
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    grades_path = run_dir.parent / f"{run_dir.name}_grades.json"
    grades = json.loads(grades_path.read_text(encoding="utf-8"))["grades"] if grades_path.exists() else {}
    common = {"run": run_dir.name, "corpus": meta["config"]["corpus_version"], "retrieval": meta.get("search_module", ""),
              "prompt": meta.get("prompt", ""), "verifier_module": meta.get("verifier_module", "")}
    rows = [dict(common, request=1, call_kind="model availability check (GET /models)",
                 http_status=meta["model_check"]["http"], total_tokens=0)]
    for f in sorted((run_dir / "raw_responses").glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        body = json.loads((run_dir / "raw_requests" / f.name).read_text(encoding="utf-8"))["body"]
        c = r["call"]
        u = c.get("usage") or {}
        rows.append(dict(common, request=len(rows) + 1, question_id=r["question_id"], call_in_question=r["seq"],
                         call_kind="draft" if r["seq"] == 1 else "redraft", started_at=c["started_at"],
                         http_status=c["status"], finish_reason=c.get("finish_reason"), prompt_tokens=u.get("prompt_tokens"),
                         completion_tokens=u.get("completion_tokens"),
                         reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                         total_tokens=u.get("total_tokens"), first_byte_ms=c.get("first_byte_ms"),
                         first_content_ms=c.get("first_content_ms"), total_ms=c.get("total_ms"), model=body["model"],
                         max_completion_tokens=body["max_completion_tokens"], temperature=body["temperature"],
                         question_grade=(grades.get(r["question_id"]) or {}).get("grade", "")))
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in COLS})
    return {"requests": len(rows), "generation_calls": len(rows) - 1,
            "total_tokens": sum(int(r.get("total_tokens") or 0) for r in rows),
            "http_200": sum(r.get("http_status") == 200 for r in rows)}


if __name__ == "__main__":
    print(main(Path(sys.argv[1]), Path(sys.argv[2])))
