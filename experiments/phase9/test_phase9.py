"""Phase 9A unit tests (no models, no network): uv run pytest -q experiments/phase9"""
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/phase9")
import build_c5  # noqa: E402
import evaluate_retrieval as ev  # noqa: E402
import extraction_e3 as e3m  # noqa: E402
import section_correction as sc  # noqa: E402
import table_chunk_metadata as tm  # noqa: E402

from relay.retrieval import latex_parse as lp


# --- e3 -------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("tex,want", [
    (r"$\phantom{1}42.2$", "$42.2$"),
    (r"\phantom{00}1.0", "1.0"),
    (r"100.00\hphantom{00}", "100.00"),
    (r"a\vphantom{X}b", "ab"),
    (r"\phantom{a\phantom{b}c}3", "3"),           # nested: removed with the parent
    (r"1\phantom{0}5", "15"),                      # visible digits on both sides survive
    (r"42.2 and 136.8", "42.2 and 136.8"),         # no phantom: unchanged
    (r"\phantom{1 42.2", r"\phantom{1 42.2"),      # unbalanced: left alone
    (r"\phantomx{1}", r"\phantomx{1}"),            # another command, not a phantom
    (r"gray \colorbox{Sand!20}{\phantom{x}}, ok", r"gray \colorbox{Sand!20}{}, ok"),
])
def test_strip_phantoms(tex, want):
    assert e3m.strip_phantoms(tex) == want


def test_e3_converts_and_restores():
    before = lp._L2T
    with e3m.e3():
        assert lp._L2T.latex_to_text(r"$\phantom{1}42.2$").strip() == "42.2"
    assert lp._L2T is before
    assert "142.2" in lp._L2T.latex_to_text(r"$\phantom{1}42.2$")  # e2 behaviour, pinned: the defect being fixed


def test_e3_restores_on_error():
    before = lp._L2T
    with pytest.raises(RuntimeError):
        with e3m.e3():
            raise RuntimeError
    assert lp._L2T is before


def test_pdf_row_matching():
    row = {"row_header": "[0, 1]", "cells": {"RQ": "(blank)", "Perplexity": "42.2", "x": ""}}
    assert e3m.printed_cells(row) == ["[0, 1]", "42.2"]
    lines = ["Perplexity", "[0, 1]", "42.2", "[0, 1]", "✓", "40.3"]
    assert e3m.row_in_lines(["[0, 1]", "42.2"], lines)
    assert not e3m.row_in_lines(["[0, 1]", "142.2"], lines)
    assert not e3m.row_in_lines(["[0, 1]", "40.3"], lines)  # must be consecutive


def test_table_region_stops_at_next_caption():
    pages = ["intro", "x\nTable 5: cap\n1\n2\nTable 6: other\n3"]
    page, lines = e3m.table_region(pages, 5)
    assert page == 2 and "1" in lines and "3" not in lines
    assert e3m.table_region(pages, 7) is None


def test_removed_words():
    assert e3m.removed_words("are gray x, unsafe", "are gray , unsafe") == [("gray", "x", ",")]
    assert e3m.removed_words("value 142.2 here", "value 42.2 here") == []  # a replacement is not a deletion


# --- section correction ----------------------------------------------------------------------------------------------
def _t(tid, label, sec):
    return {"paper": "MOSHI", "table_id": tid, "label": label, "section_id": sec, "caption": "c", "headers": ["h"],
            "rows": [{"row_header": "r", "cells": {"h": "1"}}], "row_texts": ["r: h = 1"]}


def test_correct_tables_only_table_3():
    tables = [_t("T3", "tab:mimi_ablations", "5.1"), _t("T17", "tab:mimi_ablations_appendix", "A"), _t("T4", "tab:mimi_baselines", "5.1")]
    fixed, rec = sc.correct_tables(tables)
    assert fixed[0]["section_id"] == "5.2" and fixed[0]["section_id_parsed"] == "5.1"
    assert fixed[1] == tables[1] and fixed[2] == tables[2]
    assert tables[0]["section_id"] == "5.1"  # input not mutated
    assert rec == [{"paper": "MOSHI", "table_id": "T3", "label": "tab:mimi_ablations", "section_id_original": "5.1",
                    "section_id_corrected": "5.2", "version": "s1"}]


def test_correct_tables_refuses_other_table():
    with pytest.raises(ValueError):
        sc.correct_tables([_t("T3", "tab:something_else", "5.1")])


def _c(cid, kind, sec, text, tid=None):
    c = {"chunk_id": cid, "paper": "MOSHI", "section_id": sec, "section_title": "old", "kind": kind, "text": text}
    return c | ({"table_id": tid, "rows": []} if tid else {})


SECTIONS = [{"paper": "MOSHI", "section_id": "5.1", "title": "Text Language Modeling"},
            {"paper": "MOSHI", "section_id": "5.2", "title": "Audio Tokenization"}]


def test_correct_chunks():
    chunks = [_c("MOSHI:5.1:c070", "caption", "5.1", "Table 3: Ablation"), _c("MOSHI:T3:r01", "table", "5.1", "Table 3: x", "T3"),
              _c("MOSHI:5.1:c071", "caption", "5.1", "Table 30: other"), _c("MOSHI:5.1:c072", "text", "5.1", "Table 3: prose"),
              _c("MOSHI:T17:r01", "table", "A", "Table 17: y", "T17")]
    out, changed = sc.correct_chunks(chunks, SECTIONS)
    assert changed == ["MOSHI:5.1:c070", "MOSHI:T3:r01"]
    assert [c["chunk_id"] for c in out] == [c["chunk_id"] for c in chunks]
    assert out[1]["section_id"] == "5.2" and out[1]["section_title"] == "Audio Tokenization" and out[1]["text"] == "Table 3: x"
    assert out[2] == chunks[2] and out[3] == chunks[3] and out[4] == chunks[4]


def test_citing_sections():
    paras = [{"paper": "MOSHI", "section_id": "5.2", "text": "Table 3 shows"}, {"paper": "MOSHI", "section_id": "5.1", "text": "Table 30 and Table 3.1"},
             {"paper": "HNET", "section_id": "5.2", "text": "Table 3"}]
    assert sc.citing_sections("MOSHI", "T3", paras) == {"5.2": 1}


# --- table-chunk metadata --------------------------------------------------------------------------------------------
def test_add_metadata():
    t = {"paper": "MOSHI", "table_id": "T3", "label": "tab:mimi_ablations", "rows": [{}] * 3}
    head = "Table 3: Cap.\nColumns: a | b"
    chunks = [_c("MOSHI:5.2:c001", "text", "5.2", "prose"),
              _c("MOSHI:T3:r01", "table", "5.2", head + "\nConfiguration Row 1: a\nConfiguration Row 2: b", "T3") | {"rows": [{}, {}]},
              _c("MOSHI:T3:r02", "table", "5.2", head + "\nConfiguration Row 3: c", "T3") | {"rows": [{}]}]
    out = tm.add_metadata(chunks, [t])
    assert out[0] == chunks[0]
    lines = out[1]["text"].split("\n")
    assert lines[:2] == head.split("\n") and lines[3:] == ["Configuration Row 1: a", "Configuration Row 2: b"]
    assert lines[2] == ("Source: Moshi: a speech-text foundation model for real-time dialogue (MOSHI); Table 3 "
                        "(tab:mimi_ablations); section 5.2 old; rows 1-2 of 3")
    assert out[2]["text"].split("\n")[2].endswith("rows 3-3 of 3")
    assert out[1]["n_tokens"] > 0 and chunks[1]["text"].count("\n") == 3  # input not mutated


def test_add_metadata_row_count_mismatch():
    t = {"paper": "MOSHI", "table_id": "T3", "label": "l", "rows": [{}] * 5}
    with pytest.raises(ValueError):
        tm.add_metadata([_c("MOSHI:T3:r01", "table", "5.1", "T\nColumns: a\nConfiguration x", "T3") | {"rows": [{}]}], [t])


# --- evaluator helpers -----------------------------------------------------------------------------------------------
T13_ROW = {"id": ["[0, 2, 2, 2, 2, 2, 2, 2]", "Perplexity = 136.8"], "cells": ["RQ-Transformer = ✓"], "paper_cells": ["Perplexity = 36.8"]}


def test_row_status_phantom_rows():
    c4 = ["Configuration [0, 2, 2, 2, 2, 2, 2, 2]: RQ-Transformer = ✓; Perplexity = 136.8"]
    c5 = ["Configuration [0, 2, 2, 2, 2, 2, 2, 2]: RQ-Transformer = ✓; Perplexity = 36.8"]
    assert ev.row_status(T13_ROW, c4) == (True, False)  # found, but the paper's value is not there
    assert ev.row_status(T13_ROW, c5) == (True, True)
    assert ev.row_status(T13_ROW, ["Configuration [0, 2, 2, 2, 2, 2, 2, 2]: RQ-Transformer = (blank); Perplexity = 135.4"]) == (False, False)


def test_row_status_plain_rows():
    row = {"id": ["Example Architectures = H-Net"], "cells": ["Autoregressive = ✓"]}
    assert ev.locators(row) == [["Example Architectures = H-Net"]]
    assert ev.row_status(row, ["Configuration Example Architectures = H-Net; Autoregressive = ✓"]) == (True, True)


def test_changes_direction():
    base = [{"dataset": "table", "setting": "B", "question_id": "t1", "table_in": False, "competing_in": False}]
    other = [{"dataset": "table", "setting": "B", "question_id": "t1", "table_in": True, "competing_in": True}]
    got = {c["metric"]: c["direction"] for c in ev.changes(base, other, "x")}
    assert got == {"table_in": "gain", "competing_in": "regression"}


def test_table_key():
    assert ev.table_key({"paper": "MOSHI", "table_id": "T3", "kind": "table", "rows": []}) == ("MOSHI", "T3")
    assert ev.table_key({"paper": "MOSHI", "kind": "caption", "text": "Table 3: x"}) is None


# --- built artifacts -------------------------------------------------------------------------------------------------
C5 = Path("experiments/phase9/corpus/c5")


def test_c5_manifest_hashes_match_files():
    m = json.loads((C5 / "manifest.json").read_text(encoding="utf-8"))
    assert m["corpus_version"] == "c5" and m["parent_corpus"] == "c4" and m["sources_equal_parent"]
    for rel, h in m["outputs_sha256"].items():
        assert hashlib.sha256((C5 / rel).read_bytes()).hexdigest() == h, rel


def test_c5_chunks_reproducible_from_processed():
    for s in (256, 512):
        stored = [json.loads(l) for l in (C5 / "processed" / f"chunks_{s}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert build_c5.chunk_corpus(C5 / "processed", "c5", s, True, True) == stored


def test_control_equals_c4():
    ctrl, c4 = Path("experiments/phase9/corpus/c4_control/processed"), Path("data/corpus/c4/processed")
    for f in sorted(c4.iterdir()):
        assert (ctrl / f.name).read_bytes() == f.read_bytes(), f.name
