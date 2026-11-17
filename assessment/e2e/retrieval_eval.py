"""End-to-end assessment, part 1: retrieval A/B/C on the production corpus (c3) and the production search() path
(Configuration G flags from config/app.yaml), offline. No LLM, no NLI model. Writes assessment/e2e/retrieval/<label>/.

Question set: every labelled question in the three existing sets, verbatim (eval/questions.jsonl: 30;
eval/phase7/table_question_set.json: 32; experiments/phase11/new_cases.json: 12). Gold units and the matcher are the
Phase 11F ones (experiments/phase11f/reconcile.py: quote -> chunks holding the quote; table row -> chunks holding
the row id, 'complete' = also its answering cells). No label is edited. Facets are assigned by rule from each
question's existing category (see FACETS), never from results.

Per question x setting: the ranked candidate list (after fusion, and reranking for B/C), the final context (after
the token budget), and per gold unit: rank in the ranked list, rank in the BM25 and dense full rankings (with the
query actually used), in pool / in top final_passages / in context, complete in context, and a failure cause.
Metrics: Recall@k (k = the setting's final_passages; share of gold units held by the top k ranked), MRR (1 / rank of
the first chunk holding any gold unit), gold coverage (all units in context), complete coverage, latency, tokens.

usage: retrieval_eval.py <label> [--expand on|off] [--corpus c3|c6]   (default: production flags, production corpus)
--corpus selects a corpus IN MEMORY (relay.corpus.selected(cfg, name)); config/app.yaml is never changed.
"""
from __future__ import annotations

import ctypes
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/phase11f")
from reconcile import holds, quote_chunks, units_for  # noqa: E402
from relay import corpus  # noqa: E402
from relay.config import app_config, piece_sizes, retrieval_settings  # noqa: E402
from relay.retrieval.index_build import load_model  # noqa: E402
from relay.retrieval.rerank import load_reranker  # noqa: E402
from relay.retrieval.search import policy_flags, search, tokenize  # noqa: E402

OUT = Path("assessment/e2e/retrieval")
TERMINOLOGY = {"q14"}  # documented in the write-up ("Different words": byte-pair / word-piece never found)


def free_gb() -> float:
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + \
                   [(n, ctypes.c_ulonglong) for n in ("tp", "ap", "tpf", "apf", "tv", "av", "aev")]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ap / 2**30


def facets(q: dict, ds: str) -> list[str]:
    cat = q.get("category") or q.get("type") or ""
    f = ["answerable" if q.get("answerable", True) else "unanswerable"]
    if ds in ("table", "new") or cat == "numerical_table":
        f.append("table_heavy")
    if cat == "numerical_table" or "value" in cat or "comparison" in cat or cat == "similar_metrics":
        f.append("numerical")
    if cat == "cross_paper":
        f.append("cross_paper")
    if q["question_id"] in TERMINOLOGY or cat.startswith("symbolic_header"):
        f.append("terminology_mismatch")
    return f


def load_questions() -> list[tuple[str, dict]]:
    orig = [json.loads(l) for l in open("eval/questions.jsonl", encoding="utf-8")]
    table = json.loads(Path("eval/phase7/table_question_set.json").read_text(encoding="utf-8"))["questions"]
    new = json.loads(Path("experiments/phase11/new_cases.json").read_text(encoding="utf-8"))["questions"]
    return [("original", q) for q in orig] + [("table", q) for q in table] + [("new", q) for q in new]


def boundary_split(quote: str, paper: str, chunks: dict) -> bool:
    """A quote no single chunk holds, but two consecutive chunks of the same section hold when joined."""
    q = " ".join(quote.split())
    ids = sorted((c for c in chunks.values() if c["paper"] == paper and c.get("kind") != "table"),
                 key=lambda c: (c["section_id"], c["chunk_id"]))
    return any(a["section_id"] == b["section_id"] and q in " ".join((a["text"] + " " + b["text"]).split())
               for a, b in zip(ids, ids[1:]))


def cause(u: dict, info: dict, setting, q: dict, chunks: dict) -> str | None:
    if info["complete_in_context"]:
        return None
    if not info["holders"]:
        if u["kind"] == "quote":
            ev = next((e for e in q.get("evidence") or [] if quote_chunks(e["quote"], e["section"].split()[0], chunks) == u["ids"]), None)
            if ev and boundary_split(ev["quote"], ev["section"].split()[0], chunks):
                return "chunk_boundary"
        return "missing_or_incorrect_extraction"
    if info["in_context"] and not info["complete_in_context"]:
        return "table_structure_or_missing_row_cells"
    if info["rank"] is None:  # never in the candidate pool
        b, d = info["bm25_rank"], info["dense_rank"]
        pool = setting.candidates
        if b > pool and d > pool:
            return "terminology_mismatch (neither BM25 nor dense ranks it in the pool)"
        return "embedding_retrieval (BM25 found it, dense did not, fusion cut it)" if b <= pool < d else \
               "bm25_retrieval (dense found it, BM25 did not, fusion cut it)" if d <= pool < b else \
               "fusion_cut (both ranked it inside the pool, RRF did not)"
    if info["rank"] > setting.final_passages:
        return "reranking" if setting.rerank else "fusion_ranking"
    return "insufficient_context (in the top final_passages, dropped by the token budget)"


def main(label: str, expand_override: str | None, corpus_name: str | None = None) -> None:
    out = OUT / label
    if out.exists():
        raise SystemExit(f"{out} exists: results are never overwritten; use a new label")
    print(f"free memory {free_gb():.2f} GB", flush=True)
    if free_gb() < 1.0:
        raise SystemExit("less than 1.0 GB free: not started")
    cfg = app_config()
    flags = policy_flags(cfg)
    if expand_override:
        flags["expand"] = expand_override == "on"
    settings = retrieval_settings()
    cap = cfg["pressure"]["L0"]["context_cap"]
    c = corpus.selected(cfg, corpus_name) if corpus_name else corpus.selected(cfg)
    assert c.name == (corpus_name or "c3")
    corpus.validate(c, piece_sizes())
    model, rr = load_model(), load_reranker()
    indexes = corpus.load_indexes(c, piece_sizes(), model)
    print(f"models + indexes loaded; free memory {free_gb():.2f} GB", flush=True)
    qs = load_questions()
    for idx in indexes.values():  # warm up so latency excludes first-call costs
        search("warm up", settings["B"], cap, idx, rr, **flags)
    rows = []
    for ds, q in qs:
        for name, st in settings.items():
            index = indexes[st.piece_tokens]
            t0 = time.perf_counter_ns()
            ev = search(q["question"], st, cap, index, rr if st.rerank else None, **flags)
            ms = (time.perf_counter_ns() - t0) / 1e6
            query = ev.expansion["expanded"] if ev.expansion else q["question"]
            ranked = [x["chunk_id"] for x in ev.candidates]
            final = [p["chunk_id"] for p in ev.passages]
            row = {"dataset": ds, "question_id": q["question_id"], "setting": name, "facets": facets(q, ds),
                   "term_expansion": getattr(ev, "term_expansion", None),
                   "category": q.get("category") or q.get("type"), "answerable": q.get("answerable", True),
                   "question": q["question"], "retrieval_query": query if query != q["question"] else None,
                   "policy": ev.policy, "ranked": ranked, "final_context": final, "context_tokens": ev.tokens,
                   "latency_ms": round(ms, 1)}
            if row["answerable"]:
                sc = index.bm25.get_scores(tokenize(query))
                bm25 = [index.bm25_ids[i] for i in sorted(range(len(sc)), key=lambda i: -sc[i])]
                dense = [x for x, _ in index.dense_ranking(index.embed_query(query), len(index.chunks))]
                units = []
                for u in units_for(q, "original" if ds == "original" else "table", index.chunks):
                    holders = sorted(cid for cid in index.chunks if holds(u, cid, index.chunks))
                    info = {"kind": u["kind"], "unit": u["row"]["id"] if u["kind"] == "row" else
                            (sorted(u["ids"]) if u["kind"] == "quote" else u["prefix"]),
                            "holders": holders,
                            "rank": next((i + 1 for i, cid in enumerate(ranked) if cid in holders), None),
                            "bm25_rank": min((bm25.index(h) + 1 for h in holders), default=None),
                            "dense_rank": min((dense.index(h) + 1 for h in holders), default=None),
                            "in_context": any(h in final for h in holders),
                            "complete_in_context": any(holds(u, cid, index.chunks, True) for cid in final)}
                    if u["kind"] == "row":
                        info["gold_cells"] = u["row"].get("cells")
                    info["failure_cause"] = cause(u, info, st, q, index.chunks)
                    units.append(info)
                k = st.final_passages
                first = min((x["rank"] for x in units if x["rank"]), default=None)
                row |= {"units": units, "n_units": len(units),
                        "recall_at_k": sum(bool(x["rank"] and x["rank"] <= k) for x in units) / len(units) if units else None,
                        "rr": 1 / first if first else 0.0,
                        "all_gold_in_context": all(x["in_context"] for x in units) if units else False,
                        "all_gold_complete_in_context": all(x["complete_in_context"] for x in units) if units else False}
            else:
                row["target_table_in_context"] = any(cid.startswith(f"{q['paper']}:{q['table_id']}:") for cid in final) \
                    if q.get("table_id") else None
            rows.append(row)
        print(q["question_id"], {r["setting"]: (round(r.get("rr", 0), 2), r.get("all_gold_complete_in_context"))
                                 for r in rows[-3:]}, flush=True)
    out.mkdir(parents=True)
    with (out / "per_question.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"label": label, "flags": flags, "corpus": c.name, "context_cap": cap,
               "settings": {n: vars(s) for n, s in settings.items()}, "n_questions": len(qs), "by_setting": {}}
    for name in settings:
        rs = [r for r in rows if r["setting"] == name]
        summary["by_setting"][name] = {"all": block(rs)} | {f: block([r for r in rs if f in r["facets"]]) for f in
            ("table_heavy", "numerical", "cross_paper", "terminology_mismatch", "unanswerable")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    for name, b in summary["by_setting"].items():
        print(name, {k: v for k, v in b["all"].items() if k != "failure_causes"})
        print("   causes:", b["all"]["failure_causes"])


def block(rs: list[dict]) -> dict:
    ans = [r for r in rs if r["answerable"]]
    units = [u for r in ans for u in r["units"]]
    causes: dict = {}
    for u in units:
        if u["failure_cause"]:
            causes[u["failure_cause"]] = causes.get(u["failure_cause"], 0) + 1
    lat = sorted(r["latency_ms"] for r in rs)
    return {"n_questions": len(rs), "n_answerable": len(ans), "n_gold_units": len(units),
            "recall_at_k_mean": round(statistics.mean(r["recall_at_k"] for r in ans if r["recall_at_k"] is not None), 3) if ans else None,
            "mrr": round(statistics.mean(r["rr"] for r in ans), 3) if ans else None,
            "gold_coverage_questions": f"{sum(r['all_gold_in_context'] for r in ans)}/{len(ans)}",
            "complete_coverage_questions": f"{sum(r['all_gold_complete_in_context'] for r in ans)}/{len(ans)}",
            "gold_units_in_context": f"{sum(u['in_context'] for u in units)}/{len(units)}",
            "gold_units_complete_in_context": f"{sum(u['complete_in_context'] for u in units)}/{len(units)}",
            "unanswerable_target_table_in_context": f"{sum(bool(r.get('target_table_in_context')) for r in rs if not r['answerable'])}"
                                                   f"/{sum(r.get('target_table_in_context') is not None for r in rs if not r['answerable'])}",
            "latency_ms_p50": lat[len(lat) // 2] if lat else None, "latency_ms_p95": lat[min(len(lat) - 1, round(0.95 * (len(lat) - 1)))] if lat else None,
            "context_tokens_mean": round(statistics.mean(r["context_tokens"] for r in rs), 1) if rs else None,
            "failure_causes": causes}


if __name__ == "__main__":
    a = sys.argv[1:]
    exp = a[a.index("--expand") + 1] if "--expand" in a else None
    if "--search" in a:  # a candidate search module with production's search() signature
        import importlib
        search = importlib.import_module(a[a.index("--search") + 1]).search  # noqa: F811
    main(a[0], exp, a[a.index("--corpus") + 1] if "--corpus" in a else None)
