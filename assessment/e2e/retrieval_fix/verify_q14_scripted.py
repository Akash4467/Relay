"""Offline: does verifier v4.7c (with the q14 question) accept a CORRECT q14 answer over the R2 (c6, setting B)
context, and reject the known misleading one? Scripted drafts (hand-written, not model output)."""
import json
import sys

sys.path.insert(0, ".")
from relay.candidate_v4_7 import verifier as v47  # noqa: E402
from relay.config import app_config  # noqa: E402
from relay.verify import nli  # noqa: E402

Q14 = "How is the input segmented in the original Transformer's translation experiments compared with H-Net?"
C = {json.loads(l)["chunk_id"]: json.loads(l) for l in open("experiments/phase10/corpus/c6/processed/chunks_256.jsonl", encoding="utf-8")}
ctx = [r for r in map(json.loads, open("assessment/e2e/retrieval/" + (sys.argv[1] if len(sys.argv) > 1 else "candidate_R2_c6") + "/per_question.jsonl", encoding="utf-8"))
       if r["question_id"] == "q14" and r["setting"] == "B"][0]["final_context"]
print("R2 context:", ctx)
for cid in ctx:
    if cid.startswith("HNET"):
        print(cid, "|", C[cid]["text"][:300].replace("\n", " "))
cfg = app_config()["verification"]
m = nli.load_nli(cfg["nli_model"])
verify = v47.verify_for(Q14)
DRAFTS = {
 "correct_verbatim": "Sentences were encoded using byte-pair encoding, which has a shared source-target vocabulary of about 37000 tokens [AIAYN:5.1:c025]. "
                     "For English-French, the authors used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary [AIAYN:5.1:c025].",
 "correct_complete": "Sentences were encoded using byte-pair encoding, which has a shared source-target vocabulary of about 37000 tokens [AIAYN:5.1:c025]. "
                     "For English-French, the authors split tokens into a 32000 word-piece vocabulary [AIAYN:5.1:c025]. "
                     "H-Net (space) and H-Net (2-stage) are byte-level, while the Transformers use the Llama-3 tokenizer [HNET:3.2:c061].",
 "misleading_hnet_only": "All models but Transformer take raw byte inputs, while the Transformer uses the GPT-2 tokenizer [HNET:3.2:c061].",
}
for name, d in DRAFTS.items():
    rep = verify(d, [C[c] for c in ctx], set(C), m, cfg["entail_min"], cfg["entail_min_flagged"])
    print(f"{name}: {'ACCEPT' if rep.ok else 'reject'}", [(round(s.entailment or 0, 2), s.reasons[:1], s.flags[-1:]) for s in rep.sentences])
