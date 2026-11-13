"""results/retrieval_eval.csv: retrieval-only gold-passage coverage, one row per (report, setting), from the retrieval
evaluation reports in eval/reports/ (relay/eval/retrieval_recall.py). Reads the reports, never changes them.

Coverage is a property of the retrieval step for a question set, not of a request, so it lives here and never in
results/results.csv. Table 3b (scripts/make_tables.py) is generated from this file.

    uv run python scripts/make_retrieval_csv.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

OUT = Path("results/retrieval_eval.csv")
# report file -> what that measurement is (the c1 test run is the frozen one; the c3 test run is not)
REPORTS = {
    "eval/reports/retrieval_dev.json": "dev split, corpus c1 (before the index rebuild)",
    "eval/reports/retrieval_test_frozen.json": "frozen test result: test split run once, corpus c1, committed tree",
    "eval/reports/retrieval_dev_c3.json": "dev split, corpus c3 (current index)",
    "eval/reports/retrieval_test_c3.json": "post-rebuild diagnostic: test split re-measured on corpus c3, uncommitted "
                                           "tree; not the frozen test result",
}
COLUMNS = ["report", "measurement", "split", "corpus_version", "run_at", "git_commit", "git_dirty", "setting",
           "piece_tokens", "final_passages", "rerank", "answerable_questions", "any_gold_retrieved_pct",
           "all_gold_retrieved_pct", "quote_recall_pct", "mean_context_tokens", "max_context_tokens"]


def rows(reports: dict[str, str] | None = None) -> list[dict]:
    out = []
    for path, label in (reports or REPORTS).items():
        rep = json.loads(Path(path).read_text(encoding="utf-8"))
        meta = rep["meta"]
        for r in rep["results"]:
            out.append({"report": path, "measurement": label, "split": meta["split"],
                        "corpus_version": meta["corpus_version"], "run_at": meta["run_at"],
                        "git_commit": meta["git_commit"], "git_dirty": meta["git_dirty"], "setting": r["setting"],
                        "piece_tokens": r["piece_tokens"], "final_passages": r["final_passages"],
                        "rerank": meta["settings"][r["setting"]]["rerank"], "answerable_questions": r["answerable"],
                        "any_gold_retrieved_pct": r["any_gold_retrieved_pct"],
                        "all_gold_retrieved_pct": r["all_gold_retrieved_pct"], "quote_recall_pct": r["quote_recall_pct"],
                        "mean_context_tokens": r["mean_context_tokens"], "max_context_tokens": r["max_context_tokens"]})
    return out


def main(out: Path = OUT) -> list[dict]:
    data = rows()
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(data)
    return data


if __name__ == "__main__":
    print(f"{OUT}: {len(main())} rows")
