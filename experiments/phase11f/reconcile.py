"""Phase 11F metric reconciliation (REAL-PAPER OFFLINE; no API). Independently recomputes the Phase 11E c3/c6 metrics:

1. re-runs the production retrieval for every question on both corpora (search() with Configuration G from config,
   setting B; corpus paths from relay.corpus), keeping the full ranked candidate list;
2. matches gold evidence with code written here (not retrieval_lab.score), using the gold definitions as they are:
   - eval/questions.jsonl: one unit per evidence quote; a chunk holds it if its whitespace-normalised text contains the
     quote (q09's quote, written against c3's unit-less text, is also tried with e4's unit insertions removed);
   - table set / new cases: one unit per gold row, looked for only in that question's table chunks; FOUND = a row line
     contains every id string; COMPLETE = it also contains every required cell (both also tried with paper_cells
     substituted for t13-t15, as in phases 9-11); caption_only = any chunk of that table; quotes as above;
3. compares every question with the stored Phase 11E row (final evidence, gold ranks, coverage, complete);
4. computes the metrics with explicit numerators and denominators.
Writes experiments/phase11f/metric_reconciliation.json and per_question_comparison.json.
"""
from __future__ import annotations

import json
import re
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, ".")
from relay import corpus  # noqa: E402
from relay.config import app_config, piece_sizes, retrieval_settings  # noqa: E402
from relay.eval.retrieval_recall import load_questions  # noqa: E402
from relay.retrieval.index_build import load_model  # noqa: E402
from relay.retrieval.rerank import load_reranker  # noqa: E402
from relay.retrieval.search import policy_flags, search  # noqa: E402

OUT = Path("experiments/phase11f")
CFG = app_config()
B = retrieval_settings()["B"]
CAP = CFG["pressure"]["L0"]["context_cap"]
ORIG = load_questions(None)
TABLE = json.loads(Path("eval/phase7/table_question_set.json").read_text(encoding="utf-8"))["questions"]
NEW = json.loads(Path("experiments/phase11/new_cases.json").read_text(encoding="utf-8"))["questions"]
UNITS_RE = re.compile(r"(?<=\d) (?:ms|min|s)\b")
LABELS = {"original": "project set; gold quotes assistant-located; NOT independently reviewed",
          "table": "PROVISIONAL: assistant-written and assistant-labelled (eval/phase7)",
          "new": "PROVISIONAL: assistant-written (phase 11A)"}


def norm(t: str) -> str:
    return " ".join(t.split())


def quote_chunks(quote: str, paper: str, chunks: dict) -> set:
    q = norm(quote)
    hit = {cid for cid, c in chunks.items() if c["paper"] == paper and q in norm(c["text"])}
    return hit or {cid for cid, c in chunks.items() if c["paper"] == paper and q in norm(UNITS_RE.sub("", c["text"]))}


def row_lines(c: dict) -> list[str]:
    return [l for l in c["text"].split("\n") if l.startswith("Configuration ")]


def variants(row: dict) -> list[tuple[list, list]]:
    """(id strings, required cells) as written, and with paper_cells substituted (t13-t15)."""
    ids, cells = list(row["id"]), list(row["cells"])
    out = [(ids, cells + list(row.get("paper_cells", [])))]
    paper = {p.split(" = ")[0]: p for p in row.get("paper_cells", [])}
    if paper:
        out.append(([paper.get(i.split(" = ")[0], i) if " = " in i else i for i in ids], cells + list(paper.values())))
    return out


def units_for(q: dict, ds: str, chunks: dict) -> list[dict]:
    if ds == "original":
        return [{"kind": "quote", "ids": quote_chunks(e["quote"], e["section"].split()[0], chunks)} for e in q["evidence"]]
    prefix = f"{q['paper']}:{q['table_id']}:"
    u = [{"kind": "row", "row": r, "prefix": prefix} for r in q["gold"].get("rows", [])]
    if q["gold"].get("caption_only"):
        u.append({"kind": "caption", "prefix": prefix})
    for e in q["gold"].get("quotes", []):
        u.append({"kind": "quote", "ids": quote_chunks(e["quote"], e["section"].split()[0], chunks)})
    return u


def holds(u: dict, cid: str, chunks: dict, complete: bool = False) -> bool:
    if u["kind"] == "quote":
        return cid in u["ids"]
    if not cid.startswith(u["prefix"]):
        return False
    if u["kind"] == "caption":
        return True
    lines = row_lines(chunks[cid])
    for ids, cells in variants(u["row"]):
        need = ids + (cells if complete else [])
        if any(all(x in l for x in need) for l in lines):
            return True
    return False


def run_corpus(name: str, model, reranker) -> dict[tuple, dict]:
    c = corpus.selected(CFG, name)
    corpus.validate(c, piece_sizes())
    index = corpus.load_indexes(c, [256], model)[256]
    out = {}
    for ds, qs in (("original", ORIG), ("table", TABLE), ("new", NEW)):
        for q in qs:
            ev = search(q["question"], B, CAP, index, reranker, **policy_flags(CFG))
            ranked, final = [x["chunk_id"] for x in ev.candidates], [p["chunk_id"] for p in ev.passages]
            rec = {"answerable": q["answerable"], "final": final, "tokens": ev.tokens}
            if q["answerable"]:
                us = units_for(q, ds, index.chunks)
                ranks = [next((i + 1 for i, cid in enumerate(ranked) if holds(u, cid, index.chunks)), None) for u in us]
                rec |= {"units": len(us), "gold_ranks": ranks,
                        "found_in_context": [any(holds(u, cid, index.chunks) for cid in final) for u in us],
                        "complete_in_context": [any(holds(u, cid, index.chunks, True) for cid in final) for u in us]}
            else:
                rec["target_table_in_context"] = any(cid.startswith(f"{q['paper']}:{q['table_id']}:") for cid in final) \
                    if ds != "original" else None
            out[(ds, q["question_id"])] = rec
    return out


def frac(n, d) -> dict:
    return {"numerator": n, "denominator": d, "value": round(n / d, 3) if d else None}


def metrics(res: dict, ds: str) -> dict:
    rows = [r for (d, _), r in res.items() if d == ds and r["answerable"]]
    if ds == "original_nontable":
        rows = [r for (d, q), r in res.items() if d == "original" and r["answerable"]
                and next(x for x in ORIG if x["question_id"] == q)["category"] != "numerical_table"]
    units = sum(r["units"] for r in rows)
    r5 = sum(1 for r in rows for x in r["gold_ranks"] if x and x <= 5)
    r8 = sum(1 for r in rows for x in r["gold_ranks"] if x and x <= 8)
    rr = [Fraction(1, min(x for x in r["gold_ranks"] if x)) if any(r["gold_ranks"]) else Fraction(0) for r in rows]
    found = sum(sum(r["found_in_context"]) for r in rows)
    complete_units = sum(sum(r["complete_in_context"]) for r in rows)
    complete_q = sum(all(r["complete_in_context"]) for r in rows)
    return {"questions": len(rows), "gold_units": units, "recall@5": frac(r5, units), "recall@8": frac(r8, units),
            "mrr": {"sum_of_reciprocal_ranks": round(float(sum(rr)), 4), "questions": len(rows),
                    "value": round(float(sum(rr) / len(rows)), 3) if rows else None},
            "gold_units_found_in_context": frac(found, units),
            "gold_units_complete_in_context": frac(complete_units, units),
            "questions_with_complete_gold_in_context": frac(complete_q, len(rows))}


def main() -> None:
    stored = json.loads(Path("experiments/phase11e/corpus_comparison.json").read_text(encoding="utf-8"))
    srows = {(r["corpus"], r["dataset"], r["question_id"]): r for r in stored["rows"]}
    model, reranker = load_model(), load_reranker()
    res = {n: run_corpus(n, model, reranker) for n in ("c3", "c6")}
    # 1. agreement with the stored Phase 11E rows
    mismatches = []
    for n in res:
        for (ds, q), r in res[n].items():
            s = srows[(n, ds, q)]
            if r["final"] != s["final"]:
                mismatches.append((n, ds, q, "final evidence"))
            if r["answerable"]:
                if r["gold_ranks"] != s["gold_ranks"]:
                    mismatches.append((n, ds, q, f"gold ranks {s['gold_ranks']} -> {r['gold_ranks']}"))
                if all(r["complete_in_context"]) != s["all_gold"]:
                    mismatches.append((n, ds, q, "complete"))
                if abs(sum(r["found_in_context"]) / r["units"] - s["coverage"]) > 1e-9:
                    mismatches.append((n, ds, q, "coverage"))
    # 2. metrics with denominators, and the stored summary for comparison
    recon = {"procedure": "re-run production search (G from config, setting B) per corpus via relay.corpus; gold matched "
                          "by phase 11F code; metrics computed with explicit denominators",
             "labels": LABELS, "agreement_with_phase11e_rows": {"questions_compared": 2 * len(res["c3"]),
                                                                "mismatches": mismatches, "identical": not mismatches},
             "metrics": {}, "stored_phase11e_summary": stored["metrics"]}
    for n in res:
        recon["metrics"][n] = {ds: metrics(res[n], ds) for ds in ("original", "original_nontable", "table", "new")}
    # 3. per-question comparison c3 vs c6
    perq = []
    for (ds, q), a in res["c3"].items():
        b = res["c6"][(ds, q)]
        row = {"dataset": ds, "question_id": q, "labels": LABELS[ds], "answerable": a["answerable"],
               "excluded_unknown_provenance": False, "c3_final": a["final"], "c6_final": b["final"]}
        if a["answerable"]:
            ca, cb = all(a["complete_in_context"]), all(b["complete_in_context"])
            fa, fb = sum(a["found_in_context"]), sum(b["found_in_context"])
            ma = max([1 / min(x for x in a["gold_ranks"] if x)] if any(a["gold_ranks"]) else [0.0])
            mb = max([1 / min(x for x in b["gold_ranks"] if x)] if any(b["gold_ranks"]) else [0.0])
            row |= {"units": a["units"], "c3_gold_ranks": a["gold_ranks"], "c6_gold_ranks": b["gold_ranks"],
                    "c3_found": fa, "c6_found": fb, "c3_complete": ca, "c6_complete": cb,
                    "c3_reciprocal_rank": round(ma, 4), "c6_reciprocal_rank": round(mb, 4)}
            row["complete_evidence"] = "gained" if cb and not ca else "lost" if ca and not cb else "same"
            row["found_rows"] = "more" if fb > fa else "fewer" if fb < fa else "same"
            row["first_gold_rank"] = "better" if mb > ma + 1e-9 else "worse" if mb < ma - 1e-9 else "same"
        else:
            row |= {"c3_target_table_in_context": a["target_table_in_context"], "c6_target_table_in_context": b["target_table_in_context"]}
        perq.append(row)
    summary = {}
    for ds in ("original", "table", "new"):
        sel = [r for r in perq if r["dataset"] == ds and r["answerable"]]
        summary[ds] = {k: {v: sorted(r["question_id"] for r in sel if r[k] == v) for v in sorted({r[k] for r in sel})}
                       for k in ("complete_evidence", "found_rows", "first_gold_rank")}
        summary[ds]["identical_final_evidence"] = sorted(r["question_id"] for r in sel if r["c3_final"] == r["c6_final"])
    summary["unanswerable"] = {r["question_id"]: (r["c3_target_table_in_context"], r["c6_target_table_in_context"])
                               for r in perq if not r["answerable"]}
    (OUT / "metric_reconciliation.json").write_text(json.dumps(recon, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUT / "per_question_comparison.json").write_text(json.dumps({"summary": summary, "questions": perq}, indent=1,
                                                                 ensure_ascii=False), encoding="utf-8")
    print("agreement with phase 11E rows:", not mismatches, mismatches[:8])
    for n in res:
        for ds, m in recon["metrics"][n].items():
            print(n, f"{ds:18}", "R@5", m["recall@5"]["numerator"], "/", m["recall@5"]["denominator"], "=", m["recall@5"]["value"],
                  "| R@8", m["recall@8"]["value"], "| MRR", m["mrr"]["value"], "| found", m["gold_units_found_in_context"]["value"],
                  "| complete q", m["questions_with_complete_gold_in_context"]["numerator"], "/", m["questions"])
    print(json.dumps(summary, ensure_ascii=False)[:2500])


if __name__ == "__main__":
    main()
