"""Candidate verifier v4.4c = frozen v4.2 + one change, for the improvement loop only (never the production default).

Root cause (assessment/improvement_loop/failure_registry.json, F-VER-PREMISE): the NLI premise for a cited text
passage is the whole ~256-token chunk. Sentences copied VERBATIM from that chunk are rejected (iteration_002 live:
'All models are trained on H100 GPUs.' 0.48 against MOSHI:4.4:c062; 'For each of these we use d_k=d_v=d_model/h=64.'
0.81 against AIAYN:3.2.2:c014). A sentence-pair NLI model is diluted by a long multi-fact premise.

Change: a cited text passage is ALSO offered as each of its sentences and each pair of adjacent sentences (the pair
resolves "For each of these" / "this" to the previous sentence), every one under the passage's header (paper title
| section). The whole-chunk premise, table premises, threshold 0.9 and every level-1 check (citations, absence,
numbers, formulas, table value owners) are unchanged: it is the frozen verify() run with this premises() bound.
"""
from __future__ import annotations

from types import FunctionType

from relay.retrieval.index_build import PAPER_TITLES
from relay.verify import verifier as v42
from relay.verify.sentences import split_sentences

VERSION = "v4.4c"


def _pieces(p: dict) -> list[str]:
    head = f"{PAPER_TITLES[p['paper']]} | {p['section_id']} {p['section_title']}"
    sents = [s for s in split_sentences(p["text"]) if s.strip()]
    windows = sents + [f"{a} {b}" for a, b in zip(sents, sents[1:])]
    return [f"{head}\n{w}" for w in windows]


def premises(cited: list[dict], claim: str = "") -> list[str]:
    out = v42.premises(cited, claim)
    for p in cited:
        if not (p.get("kind") == "table" and p.get("rows")):
            out += _pieces(p)
    return list(dict.fromkeys(out))


# the frozen v4.2 verify(), unchanged, resolving `premises` to the function above (no shared state is patched)
verify = FunctionType(v42.verify.__code__, {**v42.__dict__, "premises": premises}, "verify", v42.verify.__defaults__)
