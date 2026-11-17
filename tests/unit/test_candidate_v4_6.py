import json
from pathlib import Path

import pytest

from relay.candidate_v4_6 import verifier as v46

C6 = Path("experiments/phase10/corpus/c6/processed/chunks_256.jsonl")


@pytest.fixture(scope="module")
def C():
    if not C6.exists():
        pytest.skip("c6 not built")
    return {json.loads(l)["chunk_id"]: json.loads(l) for l in C6.open(encoding="utf-8")}


TRUE_QUOTES = [("This inherently sequential nature precludes parallelization within training examples, which becomes "
                "critical at longer sequence lengths, as memory constraints limit batching across examples.", "AIAYN:1:c002"),
               ("All models are trained on H100 GPUs, using FSDP and activation checkpointing.", "MOSHI:4.4:c062"),
               ("The authors trained their models on one machine with 8 NVIDIA P100 GPUs.", "AIAYN:5.2:c026")]
FALSE_QUOTES = [("The authors trained their models on one machine with 16 NVIDIA P100 GPUs.", "AIAYN:5.2:c026"),
                ("Each training step of the base models took about 1.0 seconds.", "AIAYN:5.2:c026"),
                ("Models are trained.", "MOSHI:4.4:c062")]  # too short to count as a quote
TRUE_ROWS = [("The authors also report that Moshi achieves Web Q. = 26.6, LlaMA Q. = 62.3, and Audio Trivia QA = 22.8", "MOSHI:T8:r01"),
             ("Moshi (no Inner Monologue) achieves Web Q. = 9.2, LlaMA Q. = 21.0, and Audio Trivia QA = 7.3", "MOSHI:T8:r02"),
             ("Configuration base: h = 8; BLEU / (dev) = 25.8", "AIAYN:T3:r01"),
             ("In Table 2, the Transformer (#FLOPs matched to GPT-3 XL) has Average / acc ↑ = 55.5 %", "HNET:T2:r11")]
FALSE_ROWS = [("Moshi (no Inner Monologue) achieves Web Q. = 26.6", "MOSHI:T8:r01"),
              ("Moshi achieves Web Q. = 9.2", "MOSHI:T8:r02"),
              ("Moshi (with Inner Monologue) achieves Web Q. = 9.2", "MOSHI:T8:r02"),
              ("The base configuration has PPL / (dev) = 25.8", "AIAYN:T3:r01"),
              ("The big configuration has h = 8", "AIAYN:T3:r01"),
              ("SpeechGPT (7B) has Web Q. = 26.6", "MOSHI:T8:r01"),
              ("The base configuration has a development perplexity of 25.8", "AIAYN:T3:r01")]


def test_quotes(C):
    assert all(v46._quote_ok(c, C[i]) for c, i in TRUE_QUOTES)
    assert not any(v46._quote_ok(c, C[i]) for c, i in FALSE_QUOTES)


def test_rows(C):
    P = [C[i] for i in ("MOSHI:T8:r01", "MOSHI:T8:r02", "AIAYN:T3:r01", "HNET:T2:r11")]
    assert all(v46._row_ok(c, C[i], P) for c, i in TRUE_ROWS)
    assert not any(v46._row_ok(c, C[i], P) for c, i in FALSE_ROWS)


class NeverEntails:
    def predict(self, pairs):
        return [[9.0, 0.0, 0.0] for _ in pairs]

    class config:
        id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}


def test_exact_support_only_overrides_a_pure_nli_rejection(C):
    p = C["MOSHI:4.4:c062"]
    ok = v46.verify("All models are trained on H100 GPUs, using FSDP and activation checkpointing [MOSHI:4.4:c062].",
                    [p], set(C), NeverEntails(), 0.9, 0.9)
    assert ok.ok and "exact support" in ok.sentences[0].flags[-1]
    # an unknown citation is a level-1 failure: never overridden
    bad = v46.verify("All models are trained on H100 GPUs, using FSDP and activation checkpointing [MOSHI:9:c999].",
                     [p], set(C), NeverEntails(), 0.9, 0.9)
    assert not bad.ok
    # a wrong value is not a quote: stays rejected
    wrong = v46.verify("All models are trained on A100 GPUs, using FSDP and activation checkpointing [MOSHI:4.4:c062].",
                       [p], set(C), NeverEntails(), 0.9, 0.9)
    assert not wrong.ok
