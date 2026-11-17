"""Phase 11A tests. SYNTHETIC: hand-made chunks/questions. REAL-PAPER: corpus c6 and the real retrieval models
(offline; no API). Nothing here changes production retrieval, corpora or the verifier."""
import copy
import sys

import pytest

sys.path.insert(0, "experiments/phase11")
import glossary as g  # noqa: E402
import retrieval_lab as L  # noqa: E402

Q11 = "In the Table 3 ablations, what development BLEU did the model get with a single attention head, and what did the base configuration get?"


# ---- REAL-PAPER: the glossary is the papers' own definitions
def test_glossary_entries_are_verbatim_from_their_own_paper():
    assert g.check() == []


def test_every_meaning_word_comes_from_the_defining_sentence():
    for e in g.ENTRIES:
        assert g.meaning_words(e) <= {g.stem(w) for w in g.tokens(e["sentence"])}, e["symbol"]


# ---- SYNTHETIC: expansion is traceable and conservative
def test_q11_expands_to_h_equals_1_and_keeps_the_original():
    r = g.expand(Q11)
    assert r["original"] == Q11 and r["terms"] == ["h = 1"] and r["expanded"] == Q11 + " [h = 1]"
    assert r["matches"][0]["evidence"] == "In this work we employ h=8 parallel attention layers, or heads."


@pytest.mark.parametrize("question", ["What is Mimi's bitrate?", "How does H-Net chunk bytes?",
                                      "Which attention mechanism does Moshi use?"])
def test_questions_without_two_meaning_words_are_not_expanded(question):
    assert g.expand(question)["expanded"] == question


def test_a_symbol_is_never_carried_to_another_paper():
    assert g.expand("How many attention heads does the Moshi Temporal Transformer use?")["terms"] == []
    assert g.expand("In the Moshi paper, what is the codec frame rate of Mimi?")["terms"] == ["f_r"]


def test_expansion_only_emits_glossary_symbols():
    for q in (Q11, "dropout rate of 0.2 and label smoothing 0.0", "key dimension 64 and value dimension 32"):
        for t in g.expand(q)["terms"]:
            assert t.split(" = ")[0] in {e["symbol"] for e in g.ENTRIES}


def test_a_false_premise_number_is_passed_through_not_invented():
    """n11: 64 heads do not exist in Table 3; the expansion states what was asked (h = 64), nothing else."""
    assert g.expand("what BLEU does the variant with 64 attention heads achieve?")["terms"] == ["h = 64"]


# ---- SYNTHETIC: annotation and packing never alter evidence
T = {"chunk_id": "AIAYN:T3:r02", "paper": "AIAYN", "table_id": "T3", "kind": "table", "section_id": "6.2",
     "section_title": "Model Variations",
     "text": "Table 3: Variations.\nColumns: col1 | h | d_k | BLEU / (dev)\nConfiguration (A): h = 1; d_k = 512; BLEU / (dev) = 24.9",
     "rows": []}


def test_annotate_changes_only_the_cross_encoder_view():
    before = copy.deepcopy(T)
    view = g.annotate(T)
    assert T == before
    assert "h (parallel attention layers, or heads)" in view and "d_k (queries and keys of dimension)" in view
    assert view.split("\n")[2] == T["text"].split("\n")[2]  # row values untouched
    assert g.annotate(dict(T, kind="text")) == T["text"]
    assert g.annotate(dict(T, paper="HNET")) == T["text"]  # no HNET entries: unchanged


def test_pack_fill_shares_the_caption_but_keeps_columns_rows_and_ids():
    second = dict(T, chunk_id="AIAYN:T3:r03",
                  text="Table 3: Variations.\nColumns: col1 | h | d_k | BLEU / (dev)\nConfiguration (A): h = 4; d_k = 128; BLEU / (dev) = 25.5")
    kept, used = L.pack_fill([T, second], 1600)
    assert [k["chunk_id"] for k in kept] == ["AIAYN:T3:r02", "AIAYN:T3:r03"] and used <= 1600
    assert kept[0]["text"] == T["text"]
    assert kept[1]["text"].startswith("Columns: ") and "h = 4; d_k = 128; BLEU / (dev) = 25.5" in kept[1]["text"]
    assert kept[1]["caption_shared_with"] == "AIAYN:T3:r02"


def test_pack_fill_never_exceeds_the_cap():
    kept, used = L.pack_fill([T] * 3, 60)
    assert used <= 60


# ---- REAL-PAPER: row index and retrieval safety (real models)
@pytest.fixture(scope="module")
def lab():
    return L.Lab()


def test_row_units_hold_caption_columns_and_exactly_one_source_row(lab):
    for uid, src in lab.rows.sources.items():
        if uid == src:
            continue
        u, c = lab.rows.chunks[uid], lab.index[256].chunks[src]
        lines = u["text"].split("\n")
        assert lines[:2] == c["text"].split("\n")[:2] and len(lines) == 3 and lines[2] in c["text"].split("\n")


def test_blank_cells_stay_blank_in_row_units(lab):
    rvq = next(u for u in lab.rows.chunks.values() if u["chunk_id"].startswith("MOSHI:T4:") and "RVQGAN:" in u["text"])
    assert "causal = (blank)" in rvq["text"] and "causal = ✓" not in rvq["text"]


@pytest.mark.parametrize("config", ["A", "B", "C", "D", "E", "F", "B+pack"])
def test_every_configuration_returns_real_chunk_ids(lab, config):
    r = lab.run(config, Q11)
    size = 512 if config.startswith("C") else 256
    assert r["final_ids"] and set(r["final_ids"]) <= set(lab.index[size].chunks)
    assert r["final_ids"] == [p["chunk_id"] for p in r["final"]]


def test_configurations_A_B_C_reproduce_production_search(lab):
    from relay.retrieval import search as S
    for cfg in "ABC":
        st = L.SETTINGS[cfg]
        prod = [p["chunk_id"] for p in S.search(Q11, st, L.CAP, lab.index[st.piece_tokens],
                                                 lab.reranker if st.rerank else None).passages]
        assert prod == lab.run(cfg, Q11)["final_ids"]


def test_retrieved_passages_still_go_through_the_verifier(lab):
    """Any variant's evidence is ordinary chunks: verifier v4.2 checks answers against them unchanged."""
    from relay.verify.verifier import verify
    from tests.unit.test_verifier import FakeNLI
    r = lab.run("D", Q11)
    rep = verify("The single-head model reaches 24.9 BLEU [AIAYN:T3:r99].", r["final"], set(lab.index[256].chunks),
                 FakeNLI({}), 0.9, 0.9)
    assert not rep.ok  # a citation outside the evidence is still rejected


def test_a_count_of_configurations_is_not_bound_as_a_header_value():
    """Found in the ablation: 'the two SpeechTokenizer configurations ... codec frame rate' must not give f_r = 2."""
    q = "In the Moshi paper's audio quality evaluation, at what codec frame rate do the two SpeechTokenizer configurations run?"
    assert g.expand(q)["terms"] == ["f_r"]
    assert g.expand("dimension of the keys set to 16")["terms"] == ["d_k"]  # concept fires; 16 is 3 words away: not bound
    assert g.expand("the key dimension to 16")["terms"] == ["d_k = 16"]
