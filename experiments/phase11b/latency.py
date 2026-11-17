"""Phase 11B latency (REAL-PAPER OFFLINE, c3, setting B): production search() + model-facing prompt construction with
Configuration G off vs on, interleaved per question (B, G, B, G, ...) so machine load affects both alike. Generation
is not run (no API); verification time is unaffected by G (same evidence) and not measured here."""
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/phase11")
import retrieval_lab as L  # noqa: E402
from relay.llm import prompts  # noqa: E402
from relay.retrieval import search as S  # noqa: E402
from relay.retrieval.index_build import load_model  # noqa: E402
from relay.retrieval.rerank import load_reranker  # noqa: E402

index, rr, st = S.Index(256, load_model()), load_reranker(), L.SETTINGS["B"]
qs = [q["question"] for q in L.ORIG + L.TABLE_QS + L.NEW_QS]
S.search(qs[0], st, L.CAP, index, rr)  # warm-up
t = {"off": {"retrieval": [], "prompt": []}, "on": {"retrieval": [], "prompt": []}}
for q in qs:
    for mode in ("off", "on"):
        on = mode == "on"
        t0 = time.perf_counter()
        ev = S.search(q, st, L.CAP, index, rr, expand=on, share_captions=on)
        t1 = time.perf_counter()
        prompts.messages(q, ev.passages, ev.caption_sharing)
        t2 = time.perf_counter()
        t[mode]["retrieval"].append((t1 - t0) * 1000)
        t[mode]["prompt"].append((t2 - t1) * 1000)
out = {m: {k: {"mean": round(statistics.mean(v), 2), "median": round(statistics.median(v), 2),
               "p95": round(sorted(v)[int(0.95 * (len(v) - 1))], 2)} for k, v in d.items()} for m, d in t.items()}
out["questions"] = len(qs)
p = Path(f"experiments/phase11b/out/latency_{L.stamp()}.json")
p.write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out), p.as_posix())
