"""Phase 11A table-aware retrieval lab (REAL-PAPER OFFLINE; no API). One corpus snapshot for every configuration:
c6 (experiments/phase10/corpus/c6, read only). Retrieval limits from config/retrieval_settings.yaml: 20 candidates,
8 final passages (4 for C), 1,600-token evidence budget, L0 cap 2,500 - identical everywhere.

Configurations (each changes ONE thing relative to B unless stated):
  A  production search, setting A (no cross-encoder)          B  production search, setting B
  C  production search, setting C (512-token pieces)          D  B + symbolic query expansion (glossary.expand) for
                                                                 BM25, dense and cross-encoder
  E  B over a row-level index (each table row = caption + Columns + one row; mapped back to its source chunk, so
     context and citation IDs are the original chunks)
  F  B with the cross-encoder scoring an annotated view of table chunks (glossary.annotate); chunks unchanged
  G  the best justified combination, chosen from the D/E/F results (see HYBRIDS)
  +pack  context packing variant (Stage 4): a second chunk of a table already in the context is sent without its
         caption line (Columns line and row labels kept), so more rows fit the same 1,600 tokens
A, B, C must reproduce relay.retrieval.search.search exactly (asserted).
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/phase9")
sys.path.insert(0, "experiments/phase11")
import glossary  # noqa: E402
from evaluate_retrieval import MemIndex, row_lines, row_status  # noqa: E402
from relay.config import app_config, retrieval_settings  # noqa: E402
from relay.eval.retrieval_recall import gold_chunks, load_questions  # noqa: E402
from relay.llm.tokenizer import count  # noqa: E402
from relay.retrieval import search as S  # noqa: E402
from relay.retrieval.context import fill, passage_block  # noqa: E402
from relay.retrieval.paper_mentions import named_papers  # noqa: E402
from relay.retrieval.rrf import rrf  # noqa: E402

C6 = (Path("experiments/phase10/corpus/c6/index"), Path("experiments/phase10/corpus/c6/processed"))
CAP = app_config()["pressure"]["L0"]["context_cap"]
RESERVE = app_config()["retrieval_policy"]["cross_paper_reserve_per_paper"]
SETTINGS = retrieval_settings()
OUT = Path("experiments/phase11/out")
ORIG = load_questions(None)
TABLE_QS = json.loads(Path("eval/phase7/table_question_set.json").read_text(encoding="utf-8"))["questions"]
NEW_QS = json.loads(Path("experiments/phase11/new_cases.json").read_text(encoding="utf-8"))["questions"]
HYBRIDS: dict[str, dict] = {}  # filled after the D/E/F ablation (documented in the report)


class Lab:
    def __init__(self):
        from relay.retrieval.index_build import load_model
        from relay.retrieval.rerank import load_reranker
        self.model, self.reranker = load_model(), load_reranker()
        self.index = {s: S.Index(s, self.model, *C6) for s in (256, 512)}
        self.rows = self._row_index(self.index[256])

    def _row_index(self, base: S.Index) -> MemIndex:
        units = []
        for c in base.chunks.values():
            if c.get("kind") == "table" and c.get("rows"):
                head = c["text"].split("\n")[:2]
                for k, line in enumerate(row_lines(c)):
                    units.append(dict(c, chunk_id=f"{c['chunk_id']}#{k}", text="\n".join(head + [line]), source=c["chunk_id"]))
            else:
                units.append(dict(c, source=c["chunk_id"]))
        idx = MemIndex(units, self.model)
        idx.sources = {u["chunk_id"]: u["source"] for u in units}
        return idx

    def run(self, config: str, question: str) -> dict:
        spec = {"A": dict(setting="A"), "B": dict(setting="B"), "C": dict(setting="C"),
                "D": dict(setting="B", expand=True), "E": dict(setting="B", rows=True), "F": dict(setting="B", annotate=True),
                "B+pack": dict(setting="B", pack=True), "C+pack": dict(setting="C", pack=True)} | HYBRIDS
        o = spec[config]
        st = SETTINGS[o["setting"]]
        t0 = time.perf_counter()
        exp = glossary.expand(question) if o.get("expand") else {"original": question, "expanded": question, "terms": []}
        q = exp["expanded"]
        index = self.rows if o.get("rows") else self.index[st.piece_tokens]
        papers = named_papers(question)  # the ORIGINAL question decides the paper policy
        per_paper = len(papers) >= 2
        qvec = index.embed_query(q)
        if per_paper:
            k = max(1, st.candidates // len(papers))
            cands = sorted((c for p in papers for c in index.fused(q, qvec, k, p)), key=lambda c: -c["score_rrf"])
        else:
            cands = index.fused(q, qvec, st.candidates)
        fused_order = [c["chunk_id"] for c in cands]
        rr_ms = None
        if st.rerank:
            t1 = time.perf_counter()
            texts = [glossary.annotate(c) if o.get("annotate") else c["text"] for c in cands]
            scores = self.reranker.predict([(q, t) for t in texts])
            rr_ms = round((time.perf_counter() - t1) * 1000, 1)
            for c, s in zip(cands, scores):
                c["score_ce"] = float(s)
            cands = sorted(cands, key=lambda c: -c["score_ce"])
            if o.get("fuse_ce"):  # hybrid: reciprocal-rank fusion of the fused order and the cross-encoder order
                by = {c["chunk_id"]: c for c in cands}
                cands = [by[cid] for cid, _ in rrf([fused_order, [c["chunk_id"] for c in cands]])]
        if o.get("rows"):  # back to source chunks, best unit first, each chunk once
            seen, mapped = set(), []
            for c in cands:
                src = index.sources[c["chunk_id"]]
                if src not in seen:
                    seen.add(src)
                    mapped.append(dict(self.index[256].chunks[src], score_ce=c.get("score_ce"), score_rrf=c.get("score_rrf"),
                                       dense_cos=c.get("dense_cos")))
            cands = mapped
        ordered = S.reserve(cands, papers, RESERVE) if per_paper else cands
        top = ordered[:st.final_passages]
        final, tokens = pack_fill(top, min(CAP, st.context_budget)) if o.get("pack") else fill(top, min(CAP, st.context_budget))
        total_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {"ranked": [c["chunk_id"] for c in cands], "top": [c["chunk_id"] for c in top], "final": final,
                "final_ids": [c["chunk_id"] for c in final], "tokens": tokens, "rerank_ms": rr_ms, "total_ms": total_ms,
                "expansion": exp["terms"], "expanded_query": exp["expanded"] if exp["terms"] else None,
                "scores": {c["chunk_id"]: {k: c.get(k) for k in ("score_rrf", "dense_cos", "score_ce")} for c in cands}}


def pack_fill(top: list[dict], cap: int) -> tuple[list[dict], int]:
    """Stage 4: like fill() (rank order, skip what does not fit, never cut), but a later chunk of a table already in
    the context loses its caption line; its Columns line and row labels stay, so every value keeps its row/column."""
    kept, used, tables = [], 0, set()
    for c in top:
        key = (c["paper"], c.get("table_id")) if c.get("kind") == "table" else None
        p = c
        if key and key in tables:
            lines = c["text"].split("\n")
            p = dict(c, text="\n".join(lines[1:]), caption_shared_with=next(k["chunk_id"] for k in kept
                                                                            if (k["paper"], k.get("table_id")) == key))
        n = count(passage_block(p)) + 2
        if used + n <= cap:
            kept.append(p)
            used += n
            if key:
                tables.add(key)
    return kept, used


# ---------------------------------------------------------------- metrics
_E4_UNITS = __import__("re").compile(r"(?<=\d) (?:ms|min|s)\b")  # the only text e4 inserted (e4_changes.json)


def gold_chunks_c6(q: dict, chunks: dict) -> list[set[str]]:
    """Gold quotes of eval/questions.jsonl, located in c6. q09's quote was written against c3 text that had lost
    its siunitx units ("the 230 average"); it is located with e4's unit insertions undone. Labels are unchanged."""
    try:
        return gold_chunks(q, chunks)
    except ValueError:
        stripped = {cid: dict(c, text=_E4_UNITS.sub("", c["text"])) for cid, c in chunks.items()}
        return gold_chunks(q, stripped)


def chunk_rows(lab: Lab, cid: str, size: int) -> list[str]:
    return row_lines(lab.index[size].chunks[cid])


def table_gold(q: dict, lab: Lab, size: int) -> list[dict]:
    """Gold units: one per gold row (located by row_status in a chunk), a caption unit, or a prose quote unit."""
    units = []
    for g in q["gold"].get("rows", []):  # a row is only ever looked for in its own table's chunks
        units.append({"kind": "row", "row": g, "prefix": f"{q['paper']}:{q['table_id']}:"})
    if q["gold"].get("caption_only"):
        units.append({"kind": "caption", "prefix": f"{q['paper']}:{q['table_id']}:"})
    for qt in q["gold"].get("quotes", []):
        ids = gold_chunks({"question_id": q["question_id"], "evidence": [qt]}, lab.index[size].chunks)[0]
        units.append({"kind": "quote", "ids": ids})
    return units


def hit(unit: dict, cid: str, lab: Lab, size: int, complete: bool = False) -> bool:
    if unit["kind"] == "row":
        if not cid.startswith(unit["prefix"]):
            return False
        found, done = row_status(unit["row"], chunk_rows(lab, cid, size))
        return done if complete else found
    if unit["kind"] == "caption":
        return cid.startswith(unit["prefix"])
    return cid in unit["ids"]


def score(units: list[dict], r: dict, lab: Lab, size: int, final_passages: int) -> dict:
    ranks = [next((i + 1 for i, c in enumerate(r["ranked"]) if hit(u, c, lab, size)), None) for u in units]
    in_top = [any(hit(u, c, lab, size) for c in r["top"]) for u in units]
    in_final = [any(hit(u, c, lab, size) for c in r["final_ids"]) for u in units]
    complete = [any(hit(u, c, lab, size, complete=True) for c in r["final_ids"]) for u in units]
    first = min([x for x in ranks if x], default=None)
    n = len(units)
    return {"units": n, "recall@5": sum(1 for x in ranks if x and x <= 5) / n, "recall@8": sum(1 for x in ranks if x and x <= 8) / n,
            "mrr": round(1 / first, 4) if first else 0.0, "coverage": sum(in_final) / n, "all_gold": all(complete),
            "retained": sum(in_final), "in_top": sum(in_top), "gold_ranks": ranks}


def evaluate(lab: Lab, configs: list[str]) -> list[dict]:
    rows = []
    items = [("original", q) for q in ORIG] + [("table", q) for q in TABLE_QS] + [("new", q) for q in NEW_QS]
    for cfg in configs:
        size = 512 if cfg.startswith("C") else 256
        fp = SETTINGS["C" if cfg.startswith("C") else "B"].final_passages
        for ds, q in items:
            r = lab.run(cfg, q["question"])
            rec = {"config": cfg, "dataset": ds, "question_id": q["question_id"], "answerable": q["answerable"],
                   "category": q.get("category") or q.get("type"), "final": r["final_ids"], "tokens": r["tokens"],
                   "rerank_ms": r["rerank_ms"], "total_ms": r["total_ms"], "expansion": r["expansion"],
                   "expanded_query": r["expanded_query"]}
            if ds == "original" and q["answerable"]:
                units = [{"kind": "quote", "ids": g} for g in gold_chunks_c6(q, lab.index[size].chunks)]
                rec |= score(units, r, lab, size, fp)
            elif ds in ("table", "new") and q["answerable"]:
                rec |= score(table_gold(q, lab, size), r, lab, size, fp)
            elif ds in ("table", "new"):  # unanswerable: what reaches the context
                rec["table_in_context"] = any(c.startswith(f"{q['paper']}:{q['table_id']}:") for c in r["final_ids"])
            rows.append(rec)
    return rows


def summarise(rows: list[dict]) -> dict:
    out = {}
    for cfg in sorted({r["config"] for r in rows}):
        for ds in ("original", "original_nontable", "table", "new"):
            sel = [r for r in rows if r["config"] == cfg and "units" in r and
                   (r["dataset"] == ds or (ds == "original_nontable" and r["dataset"] == "original"
                                           and r["category"] != "numerical_table"))]
            if not sel:
                continue
            u = sum(r["units"] for r in sel)
            multi = [r for r in sel if r["units"] >= 2]
            out[f"{cfg}/{ds}"] = {
                "questions": len(sel), "gold_units": u,
                "recall@5": round(sum(r["recall@5"] * r["units"] for r in sel) / u, 3),
                "recall@8": round(sum(r["recall@8"] * r["units"] for r in sel) / u, 3),
                "mrr": round(statistics.mean(r["mrr"] for r in sel), 3),
                "coverage": round(sum(r["coverage"] * r["units"] for r in sel) / u, 3),
                "all_gold_questions": sum(r["all_gold"] for r in sel),
                "multi_unit_all_gold": f"{sum(r['all_gold'] for r in multi)}/{len(multi)}",
                "budget_retention": round(sum(r["retained"] for r in sel) / max(1, sum(r["in_top"] for r in sel)), 3),
                "rerank_ms_mean": round(statistics.mean(r["rerank_ms"] for r in sel), 1) if sel[0]["rerank_ms"] is not None else None,
                "total_ms_mean": round(statistics.mean(r["total_ms"] for r in sel), 1),
                "expanded_questions": sum(bool(r["expansion"]) for r in sel)}
    return out


def reproduce_production(lab: Lab) -> dict:
    """A, B, C through the lab must equal relay.retrieval.search.search on every question."""
    bad = []
    for cfg in "ABC":
        st = SETTINGS[cfg]
        for q in ORIG + TABLE_QS + NEW_QS:
            prod = [p["chunk_id"] for p in S.search(q["question"], st, CAP, lab.index[st.piece_tokens],
                                                     lab.reranker if st.rerank else None).passages]
            if prod != lab.run(cfg, q["question"])["final_ids"]:
                bad.append((cfg, q["question_id"]))
    return {"identical": not bad, "differences": bad}


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
