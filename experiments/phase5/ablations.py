"""Phase 5, investigation A (offline): one-change-at-a-time ablations against the exact premises v4/v5 built for
the rejected live sentences. Writes experiments/phase5/out/ablations_{UTC}.json. No LLM or API calls."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")
from relay.config import app_config
from relay.verify import nli
from relay.verify.mathnorm import plain_math
from relay.verify.verifier import premises

RUN = Path("results/live_validation/real_v5_2026-11-13T19-39-49Z.jsonl")
cfg = app_config()
M = nli.load_nli(cfg["verification"]["nli_model"])
R = {r["question_id"]: r for r in map(json.loads, RUN.read_text(encoding="utf-8").splitlines()) if r.get("kind") == "question"}


def prem(q, ids, claim=""):
    by = {p["chunk_id"]: p for p in R[q]["retrieval"]["passages"]}
    return premises([by[i] for i in ids], claim)


def score(prems, claim):
    s = nli.scores(M, [(plain_math(p), plain_math(claim)) for p in prems])
    i = max(range(len(s)), key=lambda k: s[k][0])
    return {"entail": round(s[i][0], 4), "contra": round(s[i][1], 4), "best_premise": plain_math(prems[i])[:160]}


TESTS = {
    "q02 named subject 'base Transformer'": ("q02", ["AIAYN:3.2.2:c014"], [
        ("original clause", "The base Transformer uses eight parallel attention heads."),
        ("drop 'base'", "The Transformer uses eight parallel attention heads."),
        ("8 for eight", "The base Transformer uses 8 parallel attention heads."),
        ("drop 'base', 8", "The Transformer uses 8 parallel attention heads."),
        ("passage's own subject", "In this work they employ h=8 parallel attention layers, or heads."),
        ("'base model'", "The base model uses 8 parallel attention heads.")]),
    "q12 table row label vs entity and metric names": ("q12", ["MOSHI:T8:r01"], [
        ("original", "Moshi with Inner Monologue achieves 26.6 % on Web Questions, 62.3 % on Llama Questions, and 22.8 % on Audio TriviaQA."),
        ("drop 'with Inner Monologue'", "Moshi achieves 26.6 % on Web Questions, 62.3 % on Llama Questions, and 22.8 % on Audio TriviaQA."),
        ("table's column names", "Moshi with Inner Monologue achieves 26.6 on Web Q., 62.3 on LlaMA Q., and 22.8 on Audio Trivia QA."),
        ("column names, no qualifier", "Moshi achieves 26.6 on Web Q., 62.3 on LlaMA Q., and 22.8 on Audio Trivia QA."),
        ("table's own wording", "The configuration Moshi has Web Q. = 26.6, LlaMA Q. = 62.3, Audio Trivia QA = 22.8."),
        ("one value", "Moshi achieves 26.6 on Web Q.")]),
    "q12 the without-IM sentence (accepted) for contrast": ("q12", ["MOSHI:T8:r02"], [
        ("accepted live", "Moshi without Inner Monologue achieves 9.2 % on Web Questions, 21.0 % on Llama Questions, and 7.3 % on Audio TriviaQA.")]),
    "q10 participle tail / inference": ("q10", ["HNET:3.1:c049"], [
        ("original", "H‑Net (2‑stage) is significantly better than H‑Net (1‑stage), confirming that adding a second dynamic chunking stage improves performance."),
        ("first clause only", "H‑Net (2‑stage) is significantly better than H‑Net (1‑stage)."),
        ("tail only", "Adding a second dynamic chunking stage improves performance."),
        ("passage's own tail", "H-Net (2-stage) is significantly better than H-Net (1-stage), validating that iterated dynamic chunking can potentially learn a nested hierarchy of useful features.")]),
    "q10 paraphrase of c049": ("q10", ["HNET:3.1:c049"], [
        ("original", "In the 1‑stage version, H‑Net already learns context‑dependent segmentations that improve over static heuristics."),
        ("closer wording", "H-Net (1-stage) learns to segment data in a context-dependent way that improves over strong heuristics."),
        ("'static' vs 'strong'", "H-Net (1-stage) learns context-dependent segmentations that improve over strong heuristics.")]),
    "q10 figure caption with an added conclusion": ("q10", ["HNET:1:c012"], [
        ("original", "H‑Net’s architecture includes a routing module that dynamically draws chunk boundaries and a downsampler that retains vectors at those boundaries, forming the basis of its hierarchical design."),
        ("drop the conclusion", "H‑Net’s architecture includes a routing module that dynamically draws chunk boundaries and a downsampler that retains vectors at those boundaries."),
        ("caption wording", "A chunking layer has a routing module for dynamically drawing chunk boundaries and a downsampler that selectively retains vectors based on boundary indicators.")]),
}

out = {}
for name, (q, ids, variants) in TESTS.items():
    prems = prem(q, ids, variants[0][1])
    out[name] = [{"variant": v, "claim": c, **score(prems, c)} for v, c in variants]
    print(f"\n== {name} ({len(prems)} premises)")
    for row in out[name]:
        print(f"   e={row['entail']:<7} c={row['contra']:<7} {row['variant']:<28} | best: {row['best_premise'][:85]!r}")
Path("experiments/phase5/out").mkdir(parents=True, exist_ok=True)
p = Path(f"experiments/phase5/out/ablations_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}.json")
with p.open("x", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print("\n", p)
