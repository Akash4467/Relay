"""Phase 9A: retrieval evaluation of c4 vs c5 (both loaded from disk), identical configuration.

    uv run python experiments/phase9/evaluate_retrieval.py   # -> experiments/phase9/out/retrieval_eval_{UTC}.json

Datasets, always reported separately: the original 15-question set (eval/questions), the 32-question repaired-table
set (eval/phase7/table_question_set.json; labels PROVISIONAL, independent review pending), and 16 MOSHI Table 3 vs
Table 17 probes (t3_t17_queries.json + the 8 table-set questions about these tables). Settings A, B, C from
config/retrieval_settings.yaml; pressure L0 cap; production search() (cross-paper reserve included).

Per question: context, context tokens, duplicate table chunks (extra chunks of a table already in the context),
evidence lost to the context limit (in the top final_passages but dropped by the token budget), and:
- original: quotes in context, any / complete gold, MRR (first gold among the 20 candidates), lost to depth;
- table: correct table in context, table MRR, competing tables of the same paper in context, rows found, question
  complete; a gold row is located by its id strings or, where the set gives paper_cells (t13-t15, written against
  the corpus's phantom-inflated values), by the id strings with those cells substituted, in every corpus alike;
- T3/T17: first rank of each table, correct table ranked ahead, each table in context."""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")
from relay.config import app_config, retrieval_settings
from relay.eval.retrieval_recall import gold_chunks, load_questions
from relay.llm.tokenizer import count
from relay.retrieval import index_build
from relay.retrieval.context import fill, passage_block
from relay.retrieval.search import Index, named_papers, reserve, search

CAP = app_config()["pressure"]["L0"]["context_cap"]
RESERVE = app_config()["retrieval_policy"]["cross_paper_reserve_per_paper"]
SETTINGS = retrieval_settings()
ORIG = load_questions(None)
TABLE_QS = json.loads(Path("eval/phase7/table_question_set.json").read_text(encoding="utf-8"))["questions"]
PROBES = json.loads(Path("experiments/phase9/t3_t17_queries.json").read_text(encoding="utf-8"))
OUT = Path("experiments/phase9/out")


class MemIndex(Index):
    """An Index built in memory from a chunk list (the ablation's variants); same retrieval code as Index."""

    def __init__(self, chunks: list[dict], embedder):  # noqa: super().__init__ loads files
        self.chunks = {c["chunk_id"]: c for c in chunks}
        self.bm25 = index_build.build_bm25(chunks)
        self.bm25_ids = [c["chunk_id"] for c in chunks]
        self.faiss = index_build.build_faiss(index_build.embed([index_build.index_text(c) for c in chunks], embedder))
        self.faiss_ids = list(self.bm25_ids)
        self.embedder = embedder


def probe_queries() -> list[dict]:
    qs = [dict(q) for q in PROBES["queries"]] + [{"id": q["question_id"], "gold": q["table_id"], "question": q["question"]}
                                                  for q in TABLE_QS if q["paper"] == "MOSHI" and q["table_id"] in ("T3", "T17")]
    for q in qs:
        q["exclusion"], q["mentions_mushra_to_exclude"] = q["id"] in PROBES["exclusion"], q["id"] in PROBES["mentions_mushra_to_exclude"]
    return qs


def row_lines(c: dict) -> list[str]:
    return [l for l in c["text"].split("\n") if l.startswith("Configuration ")] if c.get("rows") else []


def table_key(c: dict):
    return (c["paper"], c["table_id"]) if c.get("rows") is not None and c["kind"] == "table" else None


def locators(gold_row: dict) -> list[list[str]]:
    """The row's id strings, and (if the set gives paper_cells) the same with each id string whose column has a
    paper_cell replaced by it."""
    ids = list(gold_row["id"])
    paper = {p.split(" = ")[0]: p for p in gold_row.get("paper_cells", [])}
    alt = [paper.get(i.split(" = ")[0], i) if " = " in i else i for i in ids]
    return [ids] if alt == ids else [ids, alt]


def row_status(gold_row: dict, lines: list[str]) -> tuple[bool, bool]:
    """(row found, row complete): complete = one line holds a locator, the needed cells and the paper's values."""
    need = gold_row["cells"] + gold_row.get("paper_cells", [])
    found = any(all(x in l for x in loc) for loc in locators(gold_row) for l in lines)
    done = any(all(x in l for x in loc + need) for loc in locators(gold_row) for l in lines)
    return found, done


def first_rank(ids: list[str], pred) -> int | None:
    return next((i + 1 for i, c in enumerate(ids) if pred(c)), None)


def retrieve(question: str, setting, index):
    """search() as in production, plus the ordered list that fill() saw (to measure what the budget dropped)."""
    ev = search(question, setting, CAP, index, RERANKER if setting.rerank else None)
    papers = named_papers(question)
    ordered = reserve(ev.candidates, papers, RESERVE) if len(papers) >= 2 else ev.candidates
    top = ordered[:setting.final_passages]
    kept, _ = fill(top, min(CAP, setting.context_budget))
    assert [c["chunk_id"] for c in kept] == [c["chunk_id"] for c in ev.passages], "fill() reconstruction differs"
    return ev.candidates, top, ev.passages


def evaluate(indexes: dict[int, Index], settings: str = "ABC") -> list[dict]:
    rows = []
    for name in settings:
        s = SETTINGS[name]
        index = indexes[s.piece_tokens]
        items = [("original", q["question_id"], q["question"], q) for q in ORIG] + \
                [("table", q["question_id"], q["question"], q) for q in TABLE_QS] + \
                [("t3_t17", q["id"], q["question"], q) for q in probe_queries()]
        for ds, qid, question, q in items:
            cand, top, final = retrieve(question, s, index)
            cids, tids, fids = [c["chunk_id"] for c in cand], {c["chunk_id"] for c in top}, {c["chunk_id"] for c in final}
            tabs = [table_key(c) for c in final if table_key(c)]
            r = {"dataset": ds, "setting": name, "question_id": qid, "context": [c["chunk_id"] for c in final],
                 "context_tokens": sum(count(passage_block(c)) + 2 for c in final),
                 "dropped_by_budget": [c["chunk_id"] for c in top if c["chunk_id"] not in fids],
                 "duplicate_table_chunks": len(tabs) - len(set(tabs))}
            if ds == "original":
                r["answerable"] = q["answerable"]
                if q["answerable"]:
                    gold = gold_chunks(q, index.chunks)
                    first = first_rank(cids, lambda c: any(c in g for g in gold))
                    r |= {"quotes": len(gold), "quotes_in": sum(bool(g & fids) for g in gold),
                          "any_gold": any(g & fids for g in gold), "complete": all(g & fids for g in gold),
                          "mrr": round(1 / first, 4) if first else 0.0,
                          "quotes_lost_to_budget": sum(bool(g & tids) and not (g & fids) for g in gold),
                          "quotes_lost_to_depth": sum(bool(g & set(cids)) and not (g & tids) for g in gold)}
            elif ds == "table":
                key = (q["paper"], q["table_id"])
                r["answerable"] = q["answerable"]
                lines = [l for c in final if table_key(c) == key for l in row_lines(c)]
                top_lines = [l for c in top if table_key(c) == key for l in row_lines(c)]
                status = [row_status(g, lines) for g in q["gold"]["rows"]] if q["answerable"] else []
                first = first_rank(cids, lambda c: c.startswith(f"{key[0]}:{key[1]}:"))
                competing = sorted({f"{k[0]}:{k[1]}" for k in tabs if k[0] == key[0] and k != key})
                r |= {"table_in": key in tabs, "table_mrr": round(1 / first, 4) if first else 0.0,
                      "competing_tables": competing, "competing_in": bool(competing),
                      "table_lost_to_budget": any(table_key(c) == key for c in top) and key not in tabs}
                if q["answerable"]:
                    r |= {"rows": len(status), "rows_in": sum(f for f, _ in status), "complete": all(d for _, d in status),
                          "complete_without_budget_limit": all(row_status(g, top_lines)[1] for g in q["gold"]["rows"])}
            else:
                gold, other = q["gold"], "T17" if q["gold"] == "T3" else "T3"
                rg = first_rank(cids, lambda c: c.startswith(f"MOSHI:{gold}:"))
                ro = first_rank(cids, lambda c: c.startswith(f"MOSHI:{other}:"))
                r |= {"gold": gold, "exclusion": q["exclusion"], "mentions_mushra_to_exclude": q["mentions_mushra_to_exclude"],
                      "gold_rank": rg, "other_rank": ro, "correct_ahead": rg is not None and (ro is None or rg < ro),
                      "gold_in": ("MOSHI", gold) in tabs, "other_in": ("MOSHI", other) in tabs}
            rows.append(r)
    return rows


def frac(sel, key):
    return f"{sum(bool(r[key]) for r in sel)}/{len(sel)}"


def summarise(rows: list[dict]) -> dict:
    out = {}
    for ds in ("original", "table", "t3_t17"):
        for st in sorted({r["setting"] for r in rows}):
            sel = [r for r in rows if r["dataset"] == ds and r["setting"] == st]
            if not sel:
                continue
            n_ctx = sum(len(r["context"]) for r in sel)
            d = {"context_tokens_mean": round(sum(r["context_tokens"] for r in sel) / len(sel), 1),
                 "duplicate_table_chunk_rate": round(sum(r["duplicate_table_chunks"] for r in sel) / n_ctx, 3),
                 "passages_dropped_by_budget": sum(len(r["dropped_by_budget"]) for r in sel)}
            if ds == "original":
                a = [r for r in sel if r["answerable"]]
                d |= {"any_gold": frac(a, "any_gold"), "complete": frac(a, "complete"),
                      "quote_recall": f"{sum(r['quotes_in'] for r in a)}/{sum(r['quotes'] for r in a)}",
                      "mrr": round(sum(r["mrr"] for r in a) / len(a), 3),
                      "quotes_lost_to_budget": sum(r["quotes_lost_to_budget"] for r in a),
                      "missing": [r["question_id"] for r in a if not r["any_gold"]]}
            elif ds == "table":
                a, u = [r for r in sel if r["answerable"]], [r for r in sel if not r["answerable"]]
                d |= {"table_in": frac(a, "table_in"), "table_mrr": round(sum(r["table_mrr"] for r in a) / len(a), 3),
                      "rows_in": f"{sum(r['rows_in'] for r in a)}/{sum(r['rows'] for r in a)}", "complete": frac(a, "complete"),
                      "complete_without_budget_limit": frac(a, "complete_without_budget_limit"),
                      "table_lost_to_budget": sum(r["table_lost_to_budget"] for r in a),
                      "competing_table_in_context": frac(a, "competing_in"),
                      "unanswerable_table_in": frac(u, "table_in")}
            else:
                ex = [r for r in sel if r["exclusion"]]
                d |= {"correct_ahead": frac(sel, "correct_ahead"), "gold_in": frac(sel, "gold_in"), "other_in": frac(sel, "other_in"),
                      "exclusion_correct_ahead": frac(ex, "correct_ahead"),
                      "wrong_first": [r["question_id"] for r in sel if not r["correct_ahead"]]}
            out[f"{ds} {st}"] = d
    return out


METRICS = ("any_gold", "quotes_in", "complete", "mrr", "table_in", "rows_in", "competing_in", "correct_ahead",
           "gold_in", "other_in", "table_mrr")
WORSE_IF_HIGHER = {"competing_in", "other_in"}


def changes(base: list[dict], other: list[dict], name: str) -> list[dict]:
    """Per question and setting: every metric that differs, with its direction for the question."""
    idx = {(r["dataset"], r["setting"], r["question_id"]): r for r in other}
    out = []
    for a in base:
        b = idx[(a["dataset"], a["setting"], a["question_id"])]
        for m in METRICS:
            if m in a and a[m] != b.get(m):
                up = (b[m] or 0) > (a[m] or 0)
                out.append({"dataset": a["dataset"], "setting": a["setting"], "q": a["question_id"], "metric": m,
                            "base": a[m], name: b[m], "direction": "regression" if up == (m in WORSE_IF_HIGHER) else "gain"})
    return out


def load_models():
    global RERANKER
    from relay.retrieval.rerank import load_reranker
    RERANKER = load_reranker()
    return index_build.load_model()


RERANKER = None

if __name__ == "__main__":
    embedder = load_models()
    corpora = {"c4": Path("data/corpus/c4"), "c5": Path("experiments/phase9/corpus/c5")}
    result = {"config": {"settings": {k: vars(v) if hasattr(v, "__dict__") else str(v) for k, v in SETTINGS.items()},
                         "context_cap": CAP, "table_set_labels": "PROVISIONAL (independent review pending)"},
              "corpora": {k: str(v) for k, v in corpora.items()}, "rows": {}, "summary": {}}
    for name, root in corpora.items():
        idx = {s: Index(s, embedder, root / "index", root / "processed") for s in (256, 512)}
        result["rows"][name] = evaluate(idx)
        result["summary"][name] = summarise(result["rows"][name])
    result["changes_c4_to_c5"] = changes(result["rows"]["c4"], result["rows"]["c5"], "c5")
    for k in result["summary"]["c4"]:
        print(f"{k}\n  c4 {result['summary']['c4'][k]}\n  c5 {result['summary']['c5'][k]}")
    for c in result["changes_c4_to_c5"]:
        print("  ", c)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"retrieval_eval_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}.json"
    with p.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False, default=str)
    print(p)
