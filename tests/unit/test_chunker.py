import json

import pytest

from relay.llm.tokenizer import count
from relay.retrieval import chunker as ch


def para(idx, text, page=None, sec="1", paper="AIAYN"):
    return {"paper": paper, "section_id": sec, "para_idx": idx, "text": text, "page": page}


SENT = "The model uses eight attention heads in every layer of the stack."  # 13 tokens


def test_small_paragraphs_pack_together_and_respect_boundaries():
    chunks = ch.pack_section([para(0, SENT), para(1, SENT), para(2, SENT)], size=30)
    assert [c["para_span"] for c in chunks] == [[0, 1], [2, 2]]
    assert chunks[0]["text"] == SENT + "\n\n" + SENT
    assert all(count(c["text"]) <= 30 for c in chunks)


def test_long_paragraph_splits_on_sentences():
    long = " ".join([SENT] * 5)
    chunks = ch.pack_section([para(0, long)], size=30)
    assert len(chunks) == 3
    assert all(c["para_span"] == [0, 0] for c in chunks)
    assert all(c["text"].endswith(".") for c in chunks)  # never cut mid-sentence
    assert " ".join(c["text"] for c in chunks) == long


def test_single_huge_sentence_is_cut_by_tokens():
    chunks = ch.pack_section([para(0, "word " * 100)], size=20)
    assert all(count(c["text"]) <= 20 for c in chunks)


def table(n_rows, caption="Short caption."):
    rows = [{"row_header": f"Model {i}", "cells": {"BLEU / (dev)": f"{i}.0"}} for i in range(n_rows)]
    return {"table_id": "T3", "caption": caption, "headers": ["Model", "BLEU / (dev)"], "rows": rows,
            "row_texts": [f"Model {i}: BLEU / (dev) = {i}.0" for i in range(n_rows)]}


def test_table_chunks_repeat_header_and_short_caption():
    chunks = ch.table_chunks(table(30), size=60)
    assert len(chunks) > 1
    for c in chunks:
        assert c["text"].startswith("Table 3: Short caption.\nColumns: Model | BLEU / (dev)\n")
        assert count(c["text"]) <= 60
    assert sum(len(c["rows"]) for c in chunks) == 30


def test_table_rows_name_their_configuration_and_are_structured():
    (c,) = ch.table_chunks(table(2), size=256)
    assert c["text"].endswith("Configuration Model 0: BLEU / (dev) = 0.0\nConfiguration Model 1: BLEU / (dev) = 1.0")
    assert c["rows"][1] == {"configuration": "Model 1",
                            "values": [{"metric": "BLEU", "split": "dev", "column": "BLEU / (dev)", "value": "1.0"}]}


@pytest.mark.parametrize("header,expected", [
    ("BLEU / (dev)", ("BLEU", "dev")), ("PPL / (test)", ("PPL", "test")), ("ABX (↓)", ("ABX (↓)", None)),
    ("Training Cost (FLOPs) / EN-DE", ("Training Cost (FLOPs) / EN-DE", None)), ("F-Edu / bpb ↓", ("F-Edu / bpb ↓", None)),
])
def test_metric_split(header, expected):
    assert ch.metric_split(header) == expected


def test_long_caption_shortened_to_first_sentence():
    cap = "Main results. " + "This sentence explains the table in a lot of detail. " * 10
    assert ch.short_caption(cap, 256) == "Main results."
    assert ch.short_caption("Main results.", 256) == "Main results."


DATA = {
    "sections": [{"paper": "AIAYN", "section_id": "1", "title": "Intro"},
                 {"paper": "AIAYN", "section_id": "2", "title": "Results"}],
    "paragraphs": [para(0, SENT, 1), para(1, SENT, 2), para(0, SENT, 3, sec="2")],
    "tables": [{"paper": "AIAYN", "table_id": "T1", "section_id": "2", "caption": "Scores. " + "Long words here. " * 30,
                "headers": ["M", "S"], "row_texts": ["a: S = 1", "b: S = 2"],
                "rows": [{"row_header": "a", "cells": {"S": "1"}}, {"row_header": "b", "cells": {"S": "2"}}]}],
    "figures": [{"paper": "AIAYN", "figure_id": "F1", "section_id": "1", "caption": "The architecture."}],
}


def test_chunk_paper_ids_kinds_pages_and_sections():
    chunks = ch.chunk_paper("AIAYN", DATA["sections"], DATA["paragraphs"], DATA["tables"], DATA["figures"],
                            size=256, corpus_version="cX")
    assert [(c["chunk_id"], c["kind"]) for c in chunks] == [
        ("AIAYN:1:c001", "text"), ("AIAYN:1:c002", "caption"),
        ("AIAYN:2:c003", "text"), ("AIAYN:2:c004", "caption"), ("AIAYN:T1:r01", "table")]
    first = chunks[0]
    assert (first["para_span"], first["page"], first["section_title"], first["corpus_version"]) == ([0, 1], 1, "Intro", "cX")
    assert chunks[1]["text"] == "Figure 1: The architecture."
    assert chunks[3]["text"].startswith("Table 1: Scores. Long words")  # full caption kept in its own chunk
    assert chunks[4]["text"].startswith("Table 1: Scores.\nColumns: M | S")
    # a section never leaks into another
    assert all(c["section_id"] == "2" for c in chunks[2:])


def test_chunk_ids_stable_across_rebuilds():
    args = (DATA["sections"], DATA["paragraphs"], DATA["tables"], DATA["figures"], 256, "c1")
    assert ch.chunk_paper("AIAYN", *args) == ch.chunk_paper("AIAYN", *args)


def test_main_writes_one_file_per_size(tmp_path):
    for kind, rows in DATA.items():
        (tmp_path / f"{kind}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = ch.main(tmp_path, sizes=[64, 256], corpus_version="c1")
    assert set(out) == {64, 256}
    for size in (64, 256):
        rows = [json.loads(l) for l in (tmp_path / f"chunks_{size}.jsonl").read_text().splitlines()]
        assert rows == out[size]
        assert all(r["piece_tokens"] == size and r["n_tokens"] <= size for r in rows)
