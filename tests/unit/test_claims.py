import pytest

from relay.verify.claims import atomic_claims


def test_single_claim_is_not_split():
    s = "In encoder-decoder attention the queries come from the decoder and the keys and values come from the encoder."
    assert atomic_claims(s) == [s]  # bare "and" inside a clause is not a connector


@pytest.mark.parametrize("sentence,parts", [
    ("All models use FineWeb-Edu, which was deduplicated against the benchmarks.",
     ["All models use FineWeb-Edu.", "This was deduplicated against the benchmarks."]),
    ("Mamba layers suit the encoder because they store complete key-value caches.",
     ["Mamba layers suit the encoder.", "They store complete key-value caches."]),
    ("The smoothing module is essential, and removing it doubles the training time.",
     ["The smoothing module is essential.", "Removing it doubles the training time."]),
    ("Recurrent models process positions one at a time, and this is why they generalize poorly.",
     ["Recurrent models process positions one at a time.", "They generalize poorly."]),
    ("Sparsity improves LlamaByte; it remains far worse than H-Net.",
     ["Sparsity improves LlamaByte.", "It remains far worse than H-Net."]),
])
def test_compound_sentences_keep_the_whole_and_each_clause(sentence, parts):
    assert atomic_claims(sentence) == [sentence] + parts


@pytest.mark.parametrize("sentence", [
    "Mimi uses 8 quantizers, a codebook of 2048 entries, and a frame rate of 12.5 Hz.",
    "The encoder uses self-attention, feed-forward layers, and residual connections around each sub-layer.",
    "This approach has existed since the 2017 paper was published.",
])
def test_lists_and_time_phrases_stay_whole(sentence):
    assert atomic_claims(sentence) == [sentence]


def test_short_fragments_are_not_split_off():
    s = "It uses 8 heads, so fast."
    assert atomic_claims(s) == [s]  # "Fast." is under 3 words, leaving a single clause
