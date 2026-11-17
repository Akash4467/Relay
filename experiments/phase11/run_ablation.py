"""Phase 11A ablation runner (REAL-PAPER OFFLINE). usage: run_ablation.py [config ...]  (default: A B C D E F B+pack C+pack)
Writes experiments/phase11/out/ablation_{UTC}.json: reproduction check, per-question rows, summaries."""
import json
import sys

sys.path.insert(0, "experiments/phase11")
import retrieval_lab as L  # noqa: E402


def main() -> None:
    configs = sys.argv[1:] or ["A", "B", "C", "D", "E", "F", "B+pack", "C+pack"]
    for name in configs:  # hybrids are defined by name: "G:<flags>" e.g. G:expand+annotate
        if name.startswith("G"):
            flags = name.split(":", 1)[1].split("+")
            L.HYBRIDS[name] = {"setting": "B", **{f: True for f in flags}}
    lab = L.Lab()
    repro = L.reproduce_production(lab)
    rows = L.evaluate(lab, configs)
    summary = L.summarise(rows)
    L.OUT.mkdir(parents=True, exist_ok=True)
    out = L.OUT / f"ablation_{L.stamp()}.json"
    out.write_text(json.dumps({"corpus": "c6", "configs": configs, "reproduces_production": repro, "summary": summary,
                               "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print("reproduces production search (A, B, C):", repro["identical"], repro["differences"][:5])
    for k, v in summary.items():
        print(f"{k:24}", {x: v[x] for x in ("recall@5", "recall@8", "mrr", "coverage", "all_gold_questions",
                                             "multi_unit_all_gold", "budget_retention", "rerank_ms_mean", "total_ms_mean",
                                             "expanded_questions")})
    print(out.as_posix())


if __name__ == "__main__":
    main()
