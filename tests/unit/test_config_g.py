"""Phase 11B, Configuration G in production: symbolic query expansion and prompt-only caption sharing.
Fakes only (fake embedder / cross-encoder / NLI); no model download, never Groq."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

from relay.config import app_config
from relay.llm import prompts
from relay.llm.tokenizer import count
from relay.pipeline import answer_from_evidence
from relay.retrieval import glossary
from relay.retrieval import index_build as ib
from relay.retrieval.context import fill, model_view, passage_block
from relay.retrieval.search import Evidence, Index, policy_flags, search
from tests.unit.test_index_build import CHUNKS, FakeModel
from tests.unit.test_pipeline import FakeLLM, verifier
from tests.unit.test_search import A, B
from tests.unit.test_rerank_context import FakeCrossEncoder

Q11 = "In the Table 3 ablations, what development BLEU did the model get with a single attention head, and what did the base configuration get?"


# ---------------------------------------------------------------- the glossary is the validated one
def test_glossary_data_is_the_validated_phase11a_file():
    a, b = Path("relay/retrieval/symbolic_glossary.json"), Path("experiments/phase11/glossary.json")
    assert hashlib.sha256(a.read_bytes()).hexdigest() == hashlib.sha256(b.read_bytes()).hexdigest()


def test_glossary_is_verified_against_the_production_corpus():
    assert glossary.check() == []  # every defining sentence is in c3 (data/processed), verbatim


def test_production_expand_equals_the_experimental_logic():
    sys.path.insert(0, "experiments/phase11")
    import glossary as experimental
    qs = [json.loads(l)["question"] for l in open("eval/questions.jsonl", encoding="utf-8")]
    qs += [q["question"] for q in json.load(open("eval/phase7/table_question_set.json", encoding="utf-8"))["questions"]]
    qs += [q["question"] for q in json.load(open("experiments/phase11/new_cases.json", encoding="utf-8"))["questions"]]
    assert all(experimental.expand(q) == glossary.expand(q) for q in qs)


# ---------------------------------------------------------------- query expansion
def test_aiayn_expansion_and_original_kept():
    r = glossary.expand(Q11)
    assert r["original"] == Q11 and r["terms"] == ["h = 1"] and r["expanded"] == Q11 + " [h = 1]"


def test_moshi_expansion():
    assert glossary.expand("In the Moshi paper, what is the codec frame rate of Mimi?")["terms"] == ["f_r"]
    assert glossary.expand("What is the audio sample rate of SemantiCodec in the Moshi paper?")["terms"] == ["f_s"]


def test_one_papers_glossary_never_applies_to_another():
    assert glossary.expand("How many attention heads does the Moshi Temporal Transformer use?")["terms"] == []
    assert glossary.expand("What codec frame rate does H-Net's dynamic chunking use?")["terms"] == []


@pytest.mark.parametrize("question", ["How does attention work?", "What is the frame of reference?",
                                      "Which dimension matters most?", "What rate of learning was used?"])
def test_one_meaning_word_is_too_ambiguous_to_expand(question):
    assert glossary.expand(question)["terms"] == []


def test_number_must_be_within_two_words():
    assert glossary.expand("the key dimension to 16")["terms"] == ["d_k = 16"]
    assert glossary.expand("dimension of the keys set to 16")["terms"] == ["d_k"]


def test_meaning_words_must_be_within_six_tokens():
    q = "In the Moshi paper's audio quality evaluation, at what codec frame rate do the two SpeechTokenizer configurations run?"
    assert glossary.expand(q)["terms"] == ["f_r"]  # not f_s ("audio ... rate" too far apart), not "= 2"


def test_glossary_entries_are_never_invented():
    for q in (Q11, "dropout rate of 0.2 and label smoothing 0.0", "stack of 8 identical layers and 4096 inner-layer dimensionality"):
        for m in glossary.expand(q)["matches"]:
            assert any(e["symbol"] == m["symbol"] and e["sentence"] == m["evidence"] for e in glossary.ENTRIES)


# ---------------------------------------------------------------- search(): flags, traceability, limits
@pytest.fixture
def index(tmp_path):
    proc, idx = tmp_path / "proc", tmp_path / "index"
    proc.mkdir()
    (proc / "chunks_256.jsonl").write_text("".join(json.dumps(c) + "\n" for c in CHUNKS), encoding="utf-8")
    ib.build(256, FakeModel(), proc, idx)
    return Index(256, FakeModel(), idx, proc)


def test_config_turns_g_on_and_absent_switches_are_off():
    assert policy_flags(app_config()) == {"expand": True, "share_captions": True}
    assert policy_flags({}) == {"expand": False, "share_captions": False}


def test_search_without_flags_is_unchanged(index):
    ev = search("how many attention heads", A, 2500, index)
    assert ev.expansion is None and ev.caption_sharing is False


def test_search_records_the_expansion_and_keeps_limits(index, monkeypatch):
    seen = {}
    real = index.embed_query
    monkeypatch.setattr(index, "embed_query", lambda q: seen.setdefault("embedded", q) and real(q))
    ce = FakeCrossEncoder()
    ev = search("how many attention heads with a single attention head", B, 2500, index, ce, expand=True)
    assert ev.expansion["original"] == "how many attention heads with a single attention head"
    assert ev.expansion["terms"] == ["h = 1"] and seen["embedded"] == ev.expansion["expanded"]
    assert len(ev.passages) <= B.final_passages and ev.tokens <= B.context_budget


def test_expansion_does_not_change_the_paper_policy(index):
    q = "Compare a single attention head in Vaswani et al. and Moshi"  # two papers: per-paper policy from the ORIGINAL
    assert search(q, A, 2500, index, expand=True).policy == search(q, A, 2500, index).policy == "per_paper"


# ---------------------------------------------------------------- caption sharing: model view vs verifier evidence
def table_chunk(cid, rows_text, caption="Table 3: Variations on the Transformer architecture. Unlisted values are identical."):
    label, cells = rows_text[len("Configuration "):].split(": ", 1)
    values = [{"metric": k.split(" / ")[0], "split": "dev" if "(dev)" in k else None, "column": k, "value": v}
              for k, v in (c.split(" = ") for c in cells.split("; "))]
    return {"chunk_id": cid, "paper": "AIAYN", "table_id": "T3", "kind": "table", "section_id": "6.2",
            "section_title": "Model Variations", "rows": [{"configuration": label, "values": values}],
            "text": f"{caption}\nColumns: col1 | h | d_k | BLEU / (dev) | f_r\n{rows_text}"}


R1 = table_chunk("AIAYN:T3:r01", "Configuration base: h = 8; d_k = 64; BLEU / (dev) = 25.8")
R2 = table_chunk("AIAYN:T3:r02", "Configuration (A): h = 1; d_k = 512; BLEU / (dev) = 24.9; f_r = 12.5Hz")
PROSE = {"chunk_id": "AIAYN:6.2:c031", "paper": "AIAYN", "section_id": "6.2", "section_title": "Model Variations",
         "kind": "text", "text": "We vary the number of attention heads."}


def test_caption_is_shared_and_everything_else_is_kept():
    view = model_view([R1, PROSE, R2])
    assert view[0] is R1 and view[1] is PROSE  # unchanged objects
    assert view[2]["text"] == "Columns: col1 | h | d_k | BLEU / (dev) | f_r\nConfiguration (A): h = 1; d_k = 512; BLEU / (dev) = 24.9; f_r = 12.5Hz"
    assert view[2]["chunk_id"] == "AIAYN:T3:r02" and passage_block(view[2]).startswith("[AIAYN:T3:r02]")


def test_first_chunk_of_each_table_keeps_its_caption_and_other_tables_are_untouched():
    other = dict(R2, chunk_id="MOSHI:T4:r01", paper="MOSHI", table_id="T4")
    view = model_view([R2, other])
    assert view[0] is R2 and view[1] is other


def test_model_view_never_mutates_evidence():
    before = copy.deepcopy([R1, R2])
    model_view([R1, R2])
    fill([R1, R2], 1600, share_captions=True)
    prompts.messages("q", [R1, R2], share_captions=True)
    assert [R1, R2] == before


def test_fill_returns_the_original_chunks_and_counts_the_model_view():
    kept, used = fill([R1, R2], 1600, share_captions=True)
    assert kept[0] is R1 and kept[1] is R2  # verifier-facing: complete, original objects
    assert used == sum(count(passage_block(v)) + 2 for v in model_view([R1, R2]))
    assert used < fill([R1, R2], 1600)[1]


@pytest.mark.parametrize("cap", [40, 60, 80, 120, 1600])
def test_token_limit_is_never_exceeded(cap):
    kept, used = fill([R1, R2, PROSE, R1], cap, share_captions=True)
    assert used <= cap and used == sum(count(passage_block(v)) + 2 for v in model_view(kept))


def test_more_rows_fit_the_same_budget_with_sharing():
    cap = sum(count(passage_block(v)) + 2 for v in model_view([R1, R2]))
    assert len(fill([R1, R2], cap)[0]) == 1 and len(fill([R1, R2], cap, share_captions=True)[0]) == 2


# ---------------------------------------------------------------- pipeline: prompt shows the view, verifier the chunks
class SpyVerify:
    def __init__(self):
        from relay.verify.verifier import verify
        self.real, self.seen = verify, []

    def __call__(self, draft, passages, *a, **k):
        self.seen.append(copy.deepcopy(passages))
        return self.real(draft, passages, *a, **k)


def ev_shared():
    return Evidence(passages=[R1, R2], top_dense_cos=0.8, top_ce=None, tokens=100, caption_sharing=True)


GOOD = "The base configuration reaches 25.8 BLEU [AIAYN:T3:r01]."


def test_prompt_shares_captions_but_verifier_gets_complete_chunks_including_redraft():
    llm, spy = FakeLLM("A model with one head reaches 24.9 BLEU [AIAYN:9.9:c999].", GOOD), SpyVerify()
    r = answer_from_evidence("base BLEU?", ev_shared(), llm, verifier({"The base configuration reaches 25.8 BLEU.": 0.95}),
                             {"AIAYN:T3:r01", "AIAYN:T3:r02"}, verify_fn=spy)
    for msgs in llm.calls:  # draft and redraft prompts: second chunk without its caption line
        user = msgs[1]["content"]
        assert user.count("Table 3: Variations") == 1 and "[AIAYN:T3:r02] (AIAYN §6.2 Model Variations)\nColumns:" in user
    assert len(spy.seen) == 2 and all(p == [R1, R2] for p in spy.seen)  # both verifications: original evidence
    assert r.status == "answered" and r.text == GOOD


def test_invalid_citation_is_still_rejected_with_sharing():
    r = answer_from_evidence("q", ev_shared(), FakeLLM("Reaches 24.9 BLEU [AIAYN:9.9:c999].") , verifier({}),
                             {"AIAYN:T3:r01", "AIAYN:T3:r02"}, allow_redraft=False)
    assert r.status == "refused" and "does not exist" in " ".join(w for rep in r.reports for _, w in rep.failures())


def test_false_absence_is_still_rejected_with_sharing():
    r = answer_from_evidence("q", ev_shared(),
                             FakeLLM(GOOD + " The retrieved passages do not state the BLEU of the base configuration."),
                             verifier({"The base configuration reaches 25.8 BLEU.": 0.95}),
                             {"AIAYN:T3:r01", "AIAYN:T3:r02"}, allow_redraft=False)
    assert r.status == "refused"


def test_prompt_version_bumped_for_the_new_rendering():
    assert prompts.PROMPT_VERSION == app_config()["prompt_version"] == "p4"
