import json
from pathlib import Path

import pytest

from relay.candidate_r2 import search as r2

Q14 = "How is the input segmented in the original Transformer's translation experiments compared with H-Net?"


def test_expansion_fires_on_tokenization_wording_for_aiayn_or_no_paper():
    assert r2.term_expansion(Q14) == ["byte-pair encoding", "word-piece vocabulary"]
    assert r2.term_expansion("How was the text tokenized for training?") == ["byte-pair encoding", "word-piece vocabulary"]
    assert r2.term_expansion("Which subword units did Vaswani et al. use?") == ["byte-pair encoding", "word-piece vocabulary"]


def test_expansion_never_fires_for_other_papers_or_unrelated_words():
    assert r2.term_expansion("What limitation of tokenization motivates the Dynamic Chunking paper?") == []  # HNET only
    assert r2.term_expansion("How does Moshi's tokenizer encode audio?") == []
    assert r2.term_expansion("How many attention heads does the base Transformer use?") == []
    assert r2.term_expansion("How is byte-pair encoding and the word-piece vocabulary used?") == []  # already present


@pytest.mark.parametrize("path", ["data/processed/chunks_256.jsonl",
                                  "experiments/phase10/corpus/c6/processed/chunks_256.jsonl"])
def test_phrases_are_verbatim_in_their_source_passage(path):
    if not Path(path).exists():
        pytest.skip(f"{path} not built")
    chunks = {json.loads(l)["chunk_id"]: json.loads(l)["text"] for l in open(path, encoding="utf-8")}
    for t in r2.TERMS:
        for paper, (cid, phrases) in t["phrases"].items():
            assert cid.startswith(paper + ":") and all(p in chunks[cid] for p in phrases)
