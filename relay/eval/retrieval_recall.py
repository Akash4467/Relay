"""Phase 1 done-check (DESIGN §20.1): do the gold passages come back for the questions? Local only, no LLM.

Dev split: run as often as needed while tuning. Test split: run once on a committed tree
(`--split test --out retrieval_test_frozen.json`); the script refuses to overwrite it.

For each retrieval setting: per answerable question, how many of its evidence quotes are covered by the final
passages, and the rank of the first gold chunk among the candidates. For unanswerable questions it records the
top relevance scores, which the refusal thresholds (§14.4) are later fitted on.
Writes eval/reports/retrieval_{split}.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from relay.config import app_config, retrieval_settings
from relay.retrieval.index_build import load_model
from relay.retrieval.rerank import load_reranker
from relay.retrieval.search import Index, search

QUESTIONS = Path("eval/questions.jsonl")
REPORTS = Path("eval/reports")


def load_questions(split: str | None = "dev", path: Path = QUESTIONS) -> list[dict]:
    qs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    return [q for q in qs if split is None or q["split"] == split]


def _norm(text: str) -> str:
    return " ".join(text.split())


def gold_chunks(question: dict, chunks: dict[str, dict]) -> list[set[str]]:
    """One set of chunk IDs per evidence quote: the chunks of that paper whose text contains the quote."""
    out = []
    for ev in question["evidence"]:
        paper, quote = ev["section"].split()[0], _norm(ev["quote"])
        ids = {cid for cid, c in chunks.items() if c["paper"] == paper and quote in _norm(c["text"])}
        if not ids:
            raise ValueError(f"{question['question_id']}: quote not in corpus: {quote!r}")
        out.append(ids)
    return out


def evaluate(questions: list[dict], setting_name: str, index: Index, cap: int, reranker=None) -> dict:
    setting = retrieval_settings()[setting_name]
    rows = []
    for q in questions:
        ev = search(q["question"], setting, cap, index, reranker)
        got = {p["chunk_id"] for p in ev.passages}
        row = {"question_id": q["question_id"], "answerable": q["answerable"], "tokens": ev.tokens,
               "policy": ev.policy, "named_papers": ev.named_papers,
               "top_dense_cos": round(ev.top_dense_cos, 4),
               "top_ce": None if ev.top_ce is None else round(ev.top_ce, 4),
               "passages": [p["chunk_id"] for p in ev.passages]}
        if q["answerable"]:
            gold = gold_chunks(q, index.chunks)
            cand = [c["chunk_id"] for c in ev.candidates]
            first = min((cand.index(cid) + 1 for ids in gold for cid in ids if cid in cand), default=None)
            row |= {"quotes": len(gold), "quotes_covered": sum(bool(ids & got) for ids in gold),
                    "first_gold_rank": first}
        rows.append(row)

    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    return {
        "setting": setting_name, "piece_tokens": setting.piece_tokens, "final_passages": setting.final_passages,
        "context_cap": min(cap, setting.context_budget),
        "answerable": len(ans),
        "any_gold_retrieved_pct": round(100 * sum(r["quotes_covered"] > 0 for r in ans) / len(ans), 1) if ans else None,
        "all_gold_retrieved_pct": round(100 * sum(r["quotes_covered"] == r["quotes"] for r in ans) / len(ans), 1) if ans else None,
        "quote_recall_pct": round(100 * sum(r["quotes_covered"] for r in ans) / sum(r["quotes"] for r in ans), 1) if ans else None,
        "mean_context_tokens": round(statistics.mean(r["tokens"] for r in rows), 1),
        "max_context_tokens": max(r["tokens"] for r in rows),
        "rows": rows,
        "unanswerable_top_scores": [(r["question_id"], r["top_dense_cos"], r["top_ce"]) for r in una],
    }


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def snapshot(split: str, settings: tuple[str, ...]) -> dict:
    """What the numbers were measured on, so a frozen report can be traced to exact code, config and questions."""
    return {
        "split": split,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "questions_sha256": hashlib.sha256(QUESTIONS.read_bytes()).hexdigest(),
        "corpus_version": app_config()["corpus_version"],
        "settings": {n: retrieval_settings()[n].model_dump() for n in settings},
        "retrieval_policy": app_config()["retrieval_policy"],
        "pressure_cap": app_config()["pressure"]["L0"]["context_cap"],
    }


def main(split: str = "dev", settings: tuple[str, ...] = ("A", "B", "C"), out: str | None = None) -> dict:
    out_path = REPORTS / (out or f"retrieval_{split}.json")
    if split == "test" and out_path.exists():
        # the test split is measured once; a rerun after changes would be tuning on it
        raise FileExistsError(f"{out_path} exists: the frozen test measurement is never overwritten")
    meta = snapshot(split, settings)
    if split == "test" and meta["git_dirty"]:
        raise RuntimeError("commit before the frozen test run, so the report points at the exact code")
    cap = meta["pressure_cap"]
    questions = load_questions(split)
    embedder = load_model()
    reranker = load_reranker() if any(retrieval_settings()[s].rerank for s in settings) else None
    indexes: dict[int, Index] = {}
    results = []
    for name in settings:
        size = retrieval_settings()[name].piece_tokens
        indexes.setdefault(size, Index(size, embedder))
        results.append(evaluate(questions, name, indexes[size], cap, reranker))
    REPORTS.mkdir(parents=True, exist_ok=True)
    report = {"meta": meta, "results": results}
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--out", default=None, help="report file name in eval/reports/")
    args = ap.parse_args()
    report = main(args.split, out=args.out)
    print(report["meta"])
    for r in report["results"]:
        print({k: v for k, v in r.items() if k not in ("rows", "unanswerable_top_scores")})
        for row in r["rows"]:
            if row["answerable"] and row["quotes_covered"] < row["quotes"]:
                print("   miss", row["question_id"], f"{row['quotes_covered']}/{row['quotes']}",
                      "first gold rank", row["first_gold_rank"], row["passages"])
        print("   unanswerable top scores (q, cos, ce):", r["unanswerable_top_scores"])
