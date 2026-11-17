"""Candidate verifier v4.3c = frozen v4.2 + one change, for the improvement loop only (never the production default).

Root cause it targets (assessment/improvement_loop/failure_registry.json, F-VER-SUBJ): the NLI model returns
NEUTRAL (entailment ~0, contradiction ~0) when a claim names the paper's own model ("The Transformer was trained on
one machine with 8 NVIDIA P100 GPUs") but the passage only says "we" / "our models" ("We trained our models on one
machine with 8 NVIDIA P100 GPUs"): 0.01 vs 0.99 for "The models were trained on ...". The section header naming the
paper does not link "we" to the model.

Change: the NLI premise for a cited TEXT passage starts with that paper's own self-identification sentence, quoted
verbatim from the paper (source chunk recorded below; a unit test checks it is verbatim in c3 and c6). Nothing else
changes: citation, absence, number, formula and table checks see the original chunks; table premises are untouched;
the threshold stays 0.9. It is the frozen verify() itself, run with this premises() bound in place of v4.2's.
"""
from __future__ import annotations

from types import FunctionType

from relay.verify import verifier as v42

VERSION = "v4.3c"
SELF_REFERENCE = {  # paper -> (source chunk id, verbatim sentence)
    "AIAYN": ("AIAYN:1:c003", "In this work we propose the Transformer, a model architecture eschewing recurrence "
                              "and instead relying entirely on an attention mechanism to draw global dependencies "
                              "between input and output."),
    "MOSHI": ("MOSHI:abstract:c001", "We introduce Moshi, a speech-text foundation model and full-duplex spoken "
                                     "dialogue framework."),
    "HNET": ("HNET:1:c005", "In this work, we introduce an end-to-end hierarchical network (H-Net) that compresses raw "
                            "data through a recursive, data-dependent dynamic chunking (DC) process (Figure 1)."),
}


def _with_self_reference(p: dict) -> dict:
    if p.get("kind") == "table" and p.get("rows"):
        return p
    return dict(p, text=f"{SELF_REFERENCE[p['paper']][1]}\n{p['text']}")


def premises(cited: list[dict], claim: str = "") -> list[str]:
    return v42.premises([_with_self_reference(p) for p in cited], claim)


# the frozen v4.2 verify(), unchanged, resolving `premises` to the function above (no shared state is patched)
verify = FunctionType(v42.verify.__code__, {**v42.__dict__, "premises": premises}, "verify", v42.verify.__defaults__)
