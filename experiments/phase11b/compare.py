"""Phase 11B (REAL-PAPER OFFLINE): the production Configuration G against Phase 11A's experimental G, with the Phase 11A
harness, question sets (30 original, 32 table [provisional labels], 12 new [provisional, assistant-written]),
metrics, reranker, limits and budget.

  c6 (the Phase 11A snapshot): lab "G:expand+pack" vs production search(expand=True, share_captions=True) - the
      per-question final evidence must be identical (the transfer check);
  c3 (the production corpus):  production search with G off ("PROD-B") vs on ("PROD-G").
Writes experiments/phase11b/out/compare_{UTC}.json. No API, no corpus change.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/phase9")
sys.path.insert(0, "experiments/phase11")
import retrieval_lab as L  # noqa: E402
from relay.llm import prompts  # noqa: E402
from relay.retrieval import search as S  # noqa: E402
from relay.retrieval.paper_mentions import named_papers  # noqa: E402

OUT = Path("experiments/phase11b/out")
CORPORA = {"c6": L.C6, "c3": (Path("data/index"), Path("data/processed"))}
L.HYBRIDS["G:expand+pack"] = {"setting": "B", "expand": True, "pack": True}


class ProdLab(L.Lab):
    def __init__(self, dirs, model=None, reranker=None):  # noqa: super().__init__ builds c6 + a row index
        from relay.retrieval.index_build import load_model
        from relay.retrieval.rerank import load_reranker
        self.model, self.reranker = model or load_model(), reranker or load_reranker()
        self.index = {s: S.Index(s, self.model, *dirs) for s in (256, 512)}

    def run(self, config: str, question: str) -> dict:
        if not config.startswith("PROD"):
            return super().run(config, question)
        on = config == "PROD-G"
        st = L.SETTINGS["B"]
        index = self.index[st.piece_tokens]
        t0 = time.perf_counter()
        ev = S.search(question, st, L.CAP, index, self.reranker, expand=on, share_captions=on)
        retrieval_ms = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        msgs = prompts.messages(question, ev.passages, ev.caption_sharing)  # model-facing context construction
        prompt_ms = (time.perf_counter() - t1) * 1000
        papers = named_papers(question)
        ordered = S.reserve(ev.candidates, papers, L.RESERVE) if len(papers) >= 2 else ev.candidates
        return {"ranked": [c["chunk_id"] for c in ev.candidates], "top": [c["chunk_id"] for c in ordered[:st.final_passages]],
                "final": ev.passages, "final_ids": [p["chunk_id"] for p in ev.passages], "tokens": ev.tokens,
                "rerank_ms": None if not ev.candidates else round(retrieval_ms, 1),  # search() does not split it out
                "total_ms": round(retrieval_ms + prompt_ms, 1), "prompt_ms": round(prompt_ms, 2),
                "expansion": (ev.expansion or {}).get("terms", []),
                "expanded_query": (ev.expansion or {}).get("expanded") if ev.expansion and ev.expansion["terms"] else None,
                "prompt_chars": len(msgs[1]["content"])}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report, shared = {}, {}
    for corpus, dirs in CORPORA.items():
        lab = ProdLab(dirs, shared.get("model"), shared.get("reranker"))
        shared |= {"model": lab.model, "reranker": lab.reranker}
        configs = ["G:expand+pack", "PROD-G"] if corpus == "c6" else ["PROD-B", "PROD-G"]
        rows = L.evaluate(lab, configs)
        entry = {"configs": configs, "summary": L.summarise(rows), "rows": rows}
        if corpus == "c6":
            by = {(r["config"], r["dataset"], r["question_id"]): r for r in rows}
            diffs = [k[1:] for k in by if k[0] == "PROD-G" and by[k]["final"] != by[("G:expand+pack",) + k[1:]]["final"]]
            exp = [k[1:] for k in by if k[0] == "PROD-G" and by[k]["expansion"] != by[("G:expand+pack",) + k[1:]]["expansion"]]
            entry["transfer_check"] = {"questions": sum(1 for k in by if k[0] == "PROD-G"), "final_evidence_differs": diffs,
                                       "expansion_differs": exp, "identical": not diffs and not exp}
            print("c6 transfer check (production G == Phase 11A G):", entry["transfer_check"]["identical"], diffs[:5], exp[:5])
        report[corpus] = entry
        for k, v in entry["summary"].items():
            print(corpus, f"{k:24}", {x: v[x] for x in ("recall@5", "recall@8", "mrr", "coverage", "all_gold_questions",
                                                       "multi_unit_all_gold", "budget_retention", "total_ms_mean",
                                                       "expanded_questions")})
    out = OUT / f"compare_{L.stamp()}.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(out.as_posix())


if __name__ == "__main__":
    main()
