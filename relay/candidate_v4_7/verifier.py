"""Candidate verifier v4.7c = v4.6c + a question-aware PAPER-COVERAGE guard. For evaluation only.

Root cause (assessment/e2e/live/E25_p7_v46_c6 q14, and B12_fix_c6 q14: a misleading answer shown to the user): the
question asks about the ORIGINAL Transformer's translation experiments and H-Net. The AIAYN passage that answers it
(5.1: byte-pair / word-piece vocabularies) is not retrieved, so the model answers with the H-Net paper's Transformer
baseline ("the Transformer uses the GPT-2 tokenizer [HNET:3.1:c053]"). Every sentence is supported by its citation,
so the per-sentence verifier accepts it; nothing checks that the answer covers the papers the question asks about.

Guard (rejection only; it can never accept anything v4.6c rejects): for every paper the QUESTION names
(relay.retrieval.paper_mentions.named_papers: "original Transformer"/"Vaswani" -> AIAYN, "Moshi"/"Mimi" -> MOSHI,
"H-Net"/"Dynamic Chunking" -> HNET), the answer must contain an accepted sentence that cites a passage of that
paper, or an accepted "The retrieved passages do not state ..." sentence that names that paper. Otherwise the first
sentence is rejected with a reason the redraft can act on. Questions naming no paper are unaffected.

The verifier API has no question, so this candidate is built per question: verify_for(question) returns a verify
function with the usual signature (an experiment harness binds it; production code is unchanged).
"""
from __future__ import annotations

from relay.candidate_v4_6 import verifier as v46
from relay.retrieval.index_build import PAPER_TITLES
from relay.retrieval.paper_mentions import named_papers

VERSION = "v4.7c"


def uncovered(question: str, rep) -> list[str]:
    papers = named_papers(question)
    if not papers or not rep.sentences:
        return []
    cited = {c.split(":", 1)[0] for s in rep.sentences if s.ok and not s.absence for c in s.citations}
    stated_absent = {p for s in rep.sentences if s.ok and s.absence for p in named_papers(s.sentence)}
    return [p for p in papers if p not in cited and p not in stated_absent]


def verify_for(question: str):
    def verify(answer: str, passages: list[dict], corpus_ids: set[str], nli_model, entail_min: float,
               entail_min_flagged: float):
        rep = v46.verify(answer, passages, corpus_ids, nli_model, entail_min, entail_min_flagged)
        missing = uncovered(question, rep)
        if missing and rep.ok:
            names = "; ".join(f"{PAPER_TITLES[p]} ({p})" for p in missing)
            rep.sentences[0].reasons.append(
                f"the question asks about {names}, but no sentence cites a passage from it: answer that part from "
                "that paper's passages, or write one sentence 'The retrieved passages do not state <missing part>.'")
        return rep
    verify.VERSION = VERSION
    return verify
