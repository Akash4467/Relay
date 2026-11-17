"""Phase 8, investigations C and D: shared, experiment-only retrieval helpers. Nothing here is imported by the pipeline.

- variant_chunks(variant, size): c4's processed files re-chunked with the existing chunker, with exactly one change
  patched in-process ("base" = no change; must reproduce the stored c4 chunks).
- MemIndex: an Index built in memory (BM25 + FAISS) from such a chunk list.
- run(...): settings A/B/C on the original and the repaired-table sets, with a context-assembly policy, using the
  same metrics as phase 7 (quote coverage for the original set; table/row/evidence completeness for the table set)."""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from relay.config import app_config, retrieval_settings
from relay.eval.retrieval_recall import gold_chunks, load_questions
from relay.retrieval import chunker, index_build
from relay.retrieval.context import fill, passage_block
from relay.retrieval.fetch_papers import PAPERS
from relay.retrieval.search import Index, named_papers, reserve, search
from relay.llm.tokenizer import count

C4 = Path("data/corpus/c4/processed")
CAP = app_config()["pressure"]["L0"]["context_cap"]
SETTINGS = retrieval_settings()
ORIG = load_questions(None)
TABLE_QS = json.loads(Path("eval/phase7/table_question_set.json").read_text(encoding="utf-8"))["questions"]
PHANTOM = ("Perplexity = 142.2", "Perplexity = 140.3", "Perplexity = 136.8")


def _read(kind):
    return [json.loads(l) for l in (C4 / f"{kind}.jsonl").read_text(encoding="utf-8").splitlines()]


SECTIONS, PARAGRAPHS, TABLES, FIGURES = (_read(k) for k in ("sections", "paragraphs", "tables", "figures"))
TITLES = {(s["paper"], s["section_id"]): s["title"] for s in SECTIONS}
LABELS = {(t["paper"], t["table_id"]): t["label"] for t in TABLES}


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def variant_chunks(variant: str, size: int) -> list[dict]:
    """base | full_caption (every table chunk repeats the whole caption) | metadata (each table chunk gets a line with
    its section and LaTeX label) | whole_table (a table is split only beyond 2x the piece size)."""
    def chunk_all():
        return [c for code in PAPERS for c in chunker.chunk_paper(code, SECTIONS, PARAGRAPHS, TABLES, FIGURES, size, "c4")]
    if variant == "base":
        return chunk_all()
    if variant == "full_caption":
        with patched(chunker, "short_caption", lambda caption, size: caption):
            return chunk_all()
    if variant == "whole_table":
        original = chunker.table_chunks
        with patched(chunker, "table_chunks", lambda table, size: original(table, size * 2)):
            return chunk_all()
    if variant == "metadata":
        out = chunk_all()
        for c in out:
            if c.get("rows"):
                lines = c["text"].split("\n")
                meta = f"Source: section {c['section_id']} {c['section_title']}; label {LABELS[(c['paper'], c['table_id'])]}"
                c["text"] = "\n".join(lines[:2] + [meta] + lines[2:])
                c["n_tokens"] = count(c["text"])
        return out
    raise ValueError(variant)


class MemIndex(Index):
    def __init__(self, chunks: list[dict], embedder):  # noqa: super().__init__ loads files; this builds in memory
        self.chunks = {c["chunk_id"]: c for c in chunks}
        self.bm25 = index_build.build_bm25(chunks)
        self.bm25_ids = [c["chunk_id"] for c in chunks]
        self.faiss = index_build.build_faiss(index_build.embed([index_build.index_text(c) for c in chunks], embedder))
        self.faiss_ids = list(self.bm25_ids)
        self.embedder = embedder


def row_lines(c):
    return [l for l in c["text"].split("\n") if l.startswith("Configuration ")] if c.get("rows") else []


def assemble(question, setting, index, reranker, policy=None):
    """search() as in production; policy (or None) then re-picks the final passages from the same ranked list."""
    ev = search(question, setting, CAP, index, reranker if setting.rerank else None)
    if policy is None:
        return ev.passages, ev.candidates
    papers = named_papers(question)
    ordered = ev.candidates
    if len(papers) >= 2:
        ordered = reserve(ev.candidates, papers, app_config()["retrieval_policy"]["cross_paper_reserve_per_paper"])
    picked = policy(ordered, setting)
    passages, _ = fill(picked[:setting.final_passages], min(CAP, setting.context_budget))
    return passages, ev.candidates


def run(indexes: dict[int, Index], reranker, policy=None, settings="ABC") -> list[dict]:
    rows = []
    for name in settings:
        s = SETTINGS[name]
        index = indexes[s.piece_tokens]
        for q in ORIG + TABLE_QS:
            ds = "table" if q in TABLE_QS else "original"
            final, cand = assemble(q["question"], s, index, reranker, policy)
            tables_in = [(c["paper"], c.get("table_id")) for c in final if c.get("rows")]
            r = {"dataset": ds, "setting": name, "question_id": q["question_id"], "answerable": q["answerable"],
                 "context": [c["chunk_id"] for c in final],
                 "context_tokens": sum(count(passage_block(c)) + 2 for c in final),
                 "duplicate_table_chunks": len(tables_in) - len(set(tables_in)),
                 "corrupted_lines": sum(any(p in l for p in PHANTOM) for c in final for l in row_lines(c))}
            if ds == "original" and q["answerable"]:
                gold = gold_chunks(q, index.chunks)
                fin = {c["chunk_id"] for c in final}
                ids = [c["chunk_id"] for c in cand]
                first = min((ids.index(g) + 1 for g in set().union(*gold) if g in ids), default=None)
                r |= {"quotes": len(gold), "quotes_in": sum(bool(g & fin) for g in gold), "any_gold": any(g & fin for g in gold),
                      "mrr": 1 / first if first else 0.0}
            if ds == "table" and q["answerable"]:
                key = (q["paper"], q["table_id"])
                lines = [l for c in final if (c["paper"], c.get("table_id")) == key for l in row_lines(c)]
                found = [any(all(x in l for x in g["id"]) for l in lines) for g in q["gold"]["rows"]]
                done = [any(all(x in l for x in g["id"] + g["cells"] + g.get("paper_cells", [])) for l in lines)
                        for g in q["gold"]["rows"]]
                r |= {"table_in": any((c["paper"], c.get("table_id")) == key for c in final),
                      "rows": len(found), "rows_in": sum(found), "complete": all(done)}
            rows.append(r)
    return rows


def summarise(rows) -> dict:
    out = {}
    for ds in ("original", "table"):
        for st in sorted({r["setting"] for r in rows}):
            sel = [r for r in rows if r["dataset"] == ds and r["setting"] == st]
            ans = [r for r in sel if r["answerable"]]
            ctx = sum(len(r["context"]) for r in sel)
            d = {"context_tokens_mean": round(sum(r["context_tokens"] for r in sel) / len(sel), 1),
                 "duplicate_table_chunk_rate": round(sum(r["duplicate_table_chunks"] for r in sel) / ctx, 3) if ctx else 0}
            if ds == "original":
                d |= {"any_gold": f"{sum(r['any_gold'] for r in ans)}/{len(ans)}",
                      "quote_recall": f"{sum(r['quotes_in'] for r in ans)}/{sum(r['quotes'] for r in ans)}",
                      "mrr": round(sum(r["mrr"] for r in ans) / len(ans), 3),
                      "missing": [r["question_id"] for r in ans if not r["any_gold"]]}
            else:
                d |= {"table_in": f"{sum(r['table_in'] for r in ans)}/{len(ans)}",
                      "rows_in": f"{sum(r['rows_in'] for r in ans)}/{sum(r['rows'] for r in ans)}",
                      "complete": f"{sum(r['complete'] for r in ans)}/{len(ans)}"}
            out[f"{ds} {st}"] = d
    return out
