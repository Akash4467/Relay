import json

import pymupdf
import pytest

from relay.retrieval import pdf_validate as pv


def make_pdf(path, pages: list[str]) -> None:
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=10)
    doc.save(path)


def test_norm_words_drops_equations_joins_hyphenation_and_folds_unicode():
    assert pv.norm_words("The Trans-\nformer uses ﬁne 𝖢𝗁𝗎𝗇𝗄s.\n[EQ] x = y") == \
        ["the", "transformer", "uses", "fine", "chunks"]


def test_locate_by_opening_words_and_by_window():
    pages = [" alpha beta gamma ", " one two three four five six seven eight nine ten eleven twelve thirteen "]
    assert pv.locate("One two three four five six seven eight nine ten eleven twelve", pages) == 2
    # opening words differ (e.g. inline math typeset differently) but a later 8-word window matches
    assert pv.locate("X_1 two three four five six seven eight nine", pages) == 2
    assert pv.locate("nothing like this anywhere at all in the pages here", pages) is None
    assert pv.locate("[EQ] only an equation", pages) is None


def test_locate_does_not_match_partial_words():
    assert pv.locate("former uses", [" transformer uses "]) is None


@pytest.mark.parametrize("section,found", [
    ({"section_id": "3.2.1", "title": "Scaled Dot-Product Attention"}, True),
    ({"section_id": "3.2.2", "title": "Scaled Dot-Product Attention"}, False),
    ({"section_id": "A", "title": "Extra Results"}, True),
    ({"section_id": "abstract", "title": "Abstract"}, True),
    ({"section_id": "Acknowledgements", "title": "Acknowledgements"}, True),
])
def test_heading_found(section, found):
    pages = [" abstract we present ", " 3 2 1 scaled dot product attention ", " appendix a extra results acknowledgements "]
    assert pv.heading_found(section, pages) is found


def test_validate_sets_pages_and_reports(tmp_path):
    (tmp_path / "AIAYN").mkdir()
    make_pdf(tmp_path / "AIAYN" / "paper.pdf",
             ["1 Introduction\nRecurrent neural networks have been firmly established as state of the art.",
              "2 Background\nThe goal of reducing sequential computation also forms the foundation."])
    paras = [
        {"paper": "AIAYN", "text": "Recurrent neural networks have been firmly established as state of the art."},
        {"paper": "AIAYN", "text": "The goal of reducing sequential computation also forms the foundation."},
        {"paper": "AIAYN", "text": "A sentence that the PDF does not contain at all, anywhere."},
        {"paper": "MOSHI", "text": "Other paper, untouched."},
    ]
    sections = [{"paper": "AIAYN", "section_id": "1", "title": "Introduction"},
                {"paper": "AIAYN", "section_id": "2", "title": "Background"},
                {"paper": "AIAYN", "section_id": "3", "title": "Missing"}]
    report = pv.validate("AIAYN", paras, sections, tmp_path)
    assert [p.get("page") for p in paras] == [1, 2, None, None]
    assert "page" not in paras[3]
    assert report == {"paper": "AIAYN", "paragraphs": 3, "located": 2, "coverage_pct": 66.7,
                      "headings_missing": ["3"]}


def test_main_rewrites_paragraphs_with_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(pv, "PAPERS", {"AIAYN": ("1706.03762", 7)})
    raw, proc = tmp_path / "raw", tmp_path / "proc"
    (raw / "AIAYN").mkdir(parents=True)
    proc.mkdir()
    make_pdf(raw / "AIAYN" / "paper.pdf", ["1 Intro\nHello world from the paper text."])
    (proc / "paragraphs.jsonl").write_text(json.dumps({"paper": "AIAYN", "text": "Hello world from the paper"}) + "\n")
    (proc / "sections.jsonl").write_text(json.dumps({"paper": "AIAYN", "section_id": "1", "title": "Intro"}) + "\n")
    reports = pv.main(raw, proc)
    assert reports[0]["coverage_pct"] == 100.0
    assert json.loads((proc / "paragraphs.jsonl").read_text())["page"] == 1
    assert json.loads((proc / "pdf_validation.json").read_text()) == reports
