"""Phase 11E c3 vs c6 comparison (REAL-PAPER OFFLINE retrieval; labels partly PROVISIONAL). Identical procedure for
both corpora: production search() with Configuration G (expansion + caption sharing), setting B, the Phase 11A metrics
(experiments/phase11/retrieval_lab.py, unchanged), the same question sets. Each corpus's paths come from
relay.corpus (validated against its manifest), not from hard-coded paths.
Adds the unanswerable cases and the mocked citation/evidence outcomes from {c3,c6}_regression.json.
Writes experiments/phase11e/corpus_comparison.json."""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/phase9")
sys.path.insert(0, "experiments/phase11")
sys.path.insert(0, "experiments/phase11b")
sys.path.insert(0, "experiments/phase11c")
import retrieval_lab as L  # noqa: E402
from compare import ProdLab  # noqa: E402
from compare_corpora import q11_stages  # noqa: E402
from relay import corpus  # noqa: E402
from relay.config import app_config, piece_sizes  # noqa: E402
from relay.retrieval import search as S  # noqa: E402

HERE = Path("experiments/phase11e")
LABELS = {"original": "eval/questions.jsonl: project set, gold quotes assistant-located; NOT independently reviewed",
          "table": "eval/phase7/table_question_set.json: PROVISIONAL (assistant-written and -labelled)",
          "new": "experiments/phase11/new_cases.json: PROVISIONAL (assistant-written)"}


def main() -> None:
    labs, shared, rows, summary, q11 = {}, {}, [], {}, {}
    for name in ("c3", "c6"):
        c = corpus.selected(app_config(), name)
        corpus.validate(c, piece_sizes())
        labs[name] = ProdLab((c.index_path, c.processed_path), shared.get("model"), shared.get("reranker"))
        shared |= {"model": labs[name].model, "reranker": labs[name].reranker}
        r = L.evaluate(labs[name], ["PROD-G"])
        for x in r:
            x["corpus"] = name
        rows += r
        summary[name] = {k.split("/", 1)[1]: v for k, v in L.summarise(r).items()}
        q11[name] = q11_stages(labs[name])
    by = {(r["corpus"], r["dataset"], r["question_id"]): r for r in rows}
    unanswerable = {name: {f"{ds}:{q}": r.get("table_in_context") for (c, ds, q), r in by.items()
                           if c == name and not r["answerable"]} for name in ("c3", "c6")}
    complete_lost = [f"{ds}:{q}" for (c, ds, q), r in by.items() if c == "c3" and r.get("all_gold") and not by[("c6", ds, q)].get("all_gold")]
    complete_gained = [f"{ds}:{q}" for (c, ds, q), r in by.items() if c == "c6" and r.get("all_gold") and not by[("c3", ds, q)].get("all_gold")]
    lat = {n: [] for n in labs}
    for q in [x["question"] for x in L.ORIG + L.TABLE_QS + L.NEW_QS]:
        for n in labs:  # interleaved
            t0 = time.perf_counter()
            S.search(q, L.SETTINGS["B"], L.CAP, labs[n].index[256], labs[n].reranker, expand=True, share_captions=True)
            lat[n].append((time.perf_counter() - t0) * 1000)
    validation = {}
    for n in ("c3", "c6"):
        reg = json.loads((HERE / f"{n}_regression.json").read_text(encoding="utf-8"))
        validation[n] = {r["scenario"]: {"outcome": r["outcome"], "reason": r["reason"]} for r in reg["e2e"]["rows"]}
    report = {
        "kind": "REAL-PAPER OFFLINE retrieval + MOCKED citation/evidence outcomes (scripted provider; not model performance)",
        "procedure": "production search(expand=True, share_captions=True), setting B, Phase 11A metrics, identical questions",
        "labels": LABELS,
        "metrics": {n: {ds: {k: summary[n][ds][k] for k in ("questions", "recall@5", "recall@8", "mrr", "coverage",
                                                             "all_gold_questions", "multi_unit_all_gold", "budget_retention")}
                        for ds in ("original", "original_nontable", "table", "new")} for n in summary},
        "complete_evidence": {"gained_in_c6": complete_gained, "lost_in_c6": complete_lost},
        "unanswerable_target_table_in_context": unanswerable,
        "citation_evidence_validation_mocked": validation,
        "q11": q11,
        "latency_ms": {n: {"median": round(statistics.median(v), 1), "p95": round(sorted(v)[int(0.95 * (len(v) - 1))], 1)} for n, v in lat.items()},
        "context_tokens": {n: {"mean": round(statistics.mean(r["tokens"] for r in rows if r["corpus"] == n), 1),
                               "max": max(r["tokens"] for r in rows if r["corpus"] == n)} for n in labs},
        "rows": rows}
    (HERE / "corpus_comparison.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    for n in ("c3", "c6"):
        for ds, m in report["metrics"][n].items():
            print(n, f"{ds:18}", m)
    print("complete gained:", complete_gained, "| lost:", complete_lost)
    print("latency:", report["latency_ms"], "| tokens:", report["context_tokens"])
    print("q11 r01 in context:", {n: q11[n]["AIAYN:T3:r01"]["in_final_context"] for n in q11},
          "| r02:", {n: q11[n]["AIAYN:T3:r02"]["in_final_context"] for n in q11})


if __name__ == "__main__":
    main()
