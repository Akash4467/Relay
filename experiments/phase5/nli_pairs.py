"""Phase 5, investigation A (offline): the exact NLI premise/hypothesis pairs behind every rejected sentence of the
live v5 run, built with the frozen verifier's own functions (premises, atomic_claims, plain_math, context rule).
Writes experiments/phase5/out/nli_pairs.json (new file each run). No LLM or API calls."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")
from relay.candidate_v5.verifier import verify as verify_v5
from relay.config import app_config
from relay.verify import nli
from relay.verify.citations import strip_citations
from relay.verify.claims import atomic_claims
from relay.verify.mathnorm import plain_math
from relay.verify.verifier import REFERENCE, premises

RUN = Path("results/live_validation/real_v5_2026-11-13T19-39-49Z.jsonl")
OUT = Path("experiments/phase5/out")
cfg = app_config()
T = cfg["verification"]["entail_min"]
M = nli.load_nli(cfg["verification"]["nli_model"])
CHUNK_IDS = {json.loads(l)["chunk_id"] for l in open("data/processed/chunks_256.jsonl", encoding="utf-8")}


def pairs_for(part: str, prem: list[str], ctx: list[str]) -> list[dict]:
    raw = [(p, part) for p in prem] + [(f"{c}\n{p}", part) for c in ctx for p in prem]
    scored = nli.scores(M, [(plain_math(a), plain_math(b)) for a, b in raw])
    return [{"premise": plain_math(a), "hypothesis": plain_math(b), "entail": round(e, 4), "contra": round(c, 4)}
            for (a, b), (e, c) in zip(raw, scored)]


records = []
for r in map(json.loads, RUN.read_text(encoding="utf-8").splitlines()):
    if r.get("kind") != "question":
        continue
    passages = r["retrieval"]["passages"]
    by_id = {p["chunk_id"]: p for p in passages}
    for d in r["drafts"]:
        rep = verify_v5(d["text"], passages, CHUNK_IDS, M, T, T)
        live = d["verifier"]
        if rep.ok != live["accepted"] or [s.ok for s in rep.sentences] != [s["accepted"] for s in live["sentences"]]:
            raise SystemExit(f"replay differs from the live decision: {r['question_id']} draft {d['n']}")
        prev = None
        for i, s in enumerate(rep.sentences, 1):
            rec = {"question_id": r["question_id"], "draft": d["n"], "sentence_n": i, "sentence": s.sentence,
                   "citations": s.citations, "accepted": s.ok, "entailment": s.entailment, "reasons": s.reasons,
                   "absence": s.absence, "context": s.context, "parts": []}
            if s.citations and not s.absence and all(c in by_id for c in s.citations):
                claim = strip_citations(s.sentence)
                cited = [by_id[c] for c in s.citations]
                prem = premises(cited, claim)
                parts = atomic_claims(claim)
                context = [s.context] if s.context else []
                for j, part in enumerate(parts):
                    ctx = context if j <= 1 else ([parts[1]] if REFERENCE.match(part) else [])
                    ps = pairs_for(part, prem, ctx)
                    best = max(ps, key=lambda x: x["entail"])
                    rec["parts"].append({"part": part, "is_whole_sentence": j == 0, "best": best, "n_pairs": len(ps),
                                         "all_pairs": ps})
                rec["cited_text"] = {c: by_id[c]["text"] for c in s.citations}
            records.append(rec)

OUT.mkdir(parents=True, exist_ok=True)
out = OUT / f"nli_pairs_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}.json"
with out.open("x", encoding="utf-8") as f:
    json.dump({"run": RUN.as_posix(), "threshold": T, "records": records}, f, indent=1, ensure_ascii=False)
print(out)
for rec in records:
    if rec["accepted"]:
        continue
    print(f"\n{rec['question_id']} d{rec['draft']} s{rec['sentence_n']} e={rec['entailment']} | {rec['reasons'][0][:90] if rec['reasons'] else ''}")
    for p in rec["parts"]:
        b = p["best"]
        print(f"   part ({'whole' if p['is_whole_sentence'] else 'clause'}) e={b['entail']} c={b['contra']} :: {p['part'][:110]}")
        print(f"      best premise: {b['premise'][:230]!r}")
