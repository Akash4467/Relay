"""Ingestion pipeline wired together. Grows as each ingestion unit is built."""
import gzip
import io
import json
import tarfile
from pathlib import Path

import httpx
import pytest

import pymupdf

from relay.retrieval import chunker as ch
from relay.retrieval import fetch_papers as fp
from relay.retrieval import index_build as ib
from relay.retrieval import latex_parse as lp
from relay.retrieval import pdf_validate as pv

TINY_MAIN = r"""\documentclass{article}
\newcommand{\model}{H-Net}
\begin{document}
\begin{abstract}We present \model{}.\end{abstract}
\input{sections/body}
\begin{thebibliography}{1}\bibitem{x} Some Ref.\end{thebibliography}
\end{document}
"""
TINY_BODY = r"""\section{Intro}\label{sec:intro}
\model{} uses 8 heads~\citep{v}. See \cref{tab:res}.

Second paragraph.
\section{Results}
\begin{table}\caption{Main results.}\label{tab:res}
\begin{tabular}{lc}\toprule Model & BLEU \\ \midrule Big & 28.4 \\ \bottomrule\end{tabular}\end{table}
\paragraph{Note.} As in \cref{sec:intro}.
"""


def _tarball() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, text in {"main.tex": TINY_MAIN, "sections/body.tex": TINY_BODY}.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _tiny_pdf() -> bytes:
    """What the tiny paper looks like typeset: page 1 abstract + intro, page 2 results."""
    doc = pymupdf.open()
    for text in ["Abstract\nWe present H-Net.\n1 Intro\nH-Net uses 8 heads. See Table 1.\nSecond paragraph.",
                 "2 Results\nTable 1: Main results.\nNote. As in Section 1."]:
        doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=10)
    return doc.tobytes()


def _fake_arxiv(requested: list[str], source: bytes, pdf: bytes = b"%PDF") -> httpx.Client:
    def handler(req: httpx.Request) -> httpx.Response:
        requested.append(str(req.url))
        return httpx.Response(200, content=source if "/e-print/" in str(req.url) else pdf)
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_all_three_papers_at_exact_versions(tmp_path):
    requested: list[str] = []
    manifests = fp.fetch_all(_fake_arxiv(requested, gzip.compress(b"\\begin{document}")), tmp_path)

    assert [m["paper"] for m in manifests] == ["AIAYN", "MOSHI", "HNET"]
    assert sorted(requested) == sorted([
        "https://arxiv.org/e-print/1706.03762v7", "https://arxiv.org/pdf/1706.03762v7",
        "https://arxiv.org/e-print/2410.00037v1", "https://arxiv.org/pdf/2410.00037v1",
        "https://arxiv.org/e-print/2507.07955v2", "https://arxiv.org/pdf/2507.07955v2",
    ])
    for code in fp.PAPERS:
        assert (tmp_path / code / "src" / "main.tex").exists()


def test_fetch_then_parse_writes_processed_records(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "processed"
    fp.fetch_all(_fake_arxiv([], _tarball()), raw)
    lp.main(raw, out)

    def read(kind):
        return [json.loads(l) for l in (out / f"{kind}.jsonl").read_text(encoding="utf-8").splitlines()]

    secs, paras, tables = read("sections"), read("paragraphs"), read("tables")
    assert {s["paper"] for s in secs} == set(fp.PAPERS)
    aiayn = [s for s in secs if s["paper"] == "AIAYN"]
    assert [s["section_id"] for s in aiayn] == ["abstract", "1", "2"]

    texts = [p["text"] for p in paras if p["paper"] == "AIAYN"]
    assert texts == [
        "We present H-Net.",
        "H-Net uses 8 heads. See Table 1.",
        "Second paragraph.",
        "Note. As in Section 1.",
    ]
    t = next(t for t in tables if t["paper"] == "AIAYN")
    assert (t["table_id"], t["section_id"], t["caption"]) == ("T1", "2", "Main results.")
    assert t["row_texts"] == ["Big: BLEU = 28.4"]
    assert "Some Ref." in (out / "bibliography_AIAYN.txt").read_text(encoding="utf-8")
    assert not any("Some Ref" in p["text"] for p in paras)  # bibliography never reaches retrieval


def test_fetch_parse_validate_attaches_pages(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "processed"
    fp.fetch_all(_fake_arxiv([], _tarball(), _tiny_pdf()), raw)
    lp.main(raw, out)
    reports = pv.main(raw, out)

    assert [(r["paper"], r["coverage_pct"], r["headings_missing"]) for r in reports] == \
        [(code, 100.0, []) for code in fp.PAPERS]
    paras = [json.loads(l) for l in (out / "paragraphs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [p["page"] for p in paras if p["paper"] == "AIAYN"] == [1, 1, 1, 2]

    chunks = ch.main(out, sizes=[256], corpus_version="c1")[256]
    aiayn = [(c["chunk_id"], c["kind"], c["page"]) for c in chunks if c["paper"] == "AIAYN"]
    assert aiayn == [("AIAYN:abstract:c001", "text", 1), ("AIAYN:1:c002", "text", 1),
                     ("AIAYN:2:c003", "text", 2), ("AIAYN:T1:r01", "table", None)]
    table = next(c for c in chunks if c["chunk_id"] == "AIAYN:T1:r01")
    assert table["text"] == "Table 1: Main results.\nColumns: Model | BLEU\nBig: BLEU = 28.4"

    from tests.unit.test_index_build import FakeModel  # deterministic embedder, no model download
    report = ib.build(256, FakeModel(), out, tmp_path / "index")
    assert report["chunks"] == len(chunks)
    ids = json.loads((tmp_path / "index" / "faiss_256.ids.json").read_text())
    assert ids == [c["chunk_id"] for c in chunks]


REAL = Path("data/raw")


@pytest.mark.skipif(not all((REAL / c / "src").exists() for c in fp.PAPERS), reason="papers not downloaded")
def test_real_papers_match_pdf_counts():
    """Table and figure counts and numbered sections checked by hand against the three PDFs (2026-11-12)."""
    parsed = {code: lp.parse_paper(code) for code in fp.PAPERS}
    assert {c: (len(p.tables), len(p.figures)) for c, p in parsed.items()} == {
        "AIAYN": (4, 5), "MOSHI": (19, 11), "HNET": (8, 16)}
    ids = {c: [s["section_id"] for s in p.sections] for c, p in parsed.items()}
    assert "3.2.2" in ids["AIAYN"] and "6.4" in ids["MOSHI"] and "2.2.1" in ids["HNET"]
    assert ids["MOSHI"][-3:] == ["F.1", "F.2", "F.3"] and ids["HNET"][-1] == "F"
    for p in parsed.values():
        for para in p.paragraphs:
            assert "??" not in para["text"] and "\\" not in para["text"].split("[EQ]")[0]
            assert not para["text"].startswith("[EQ]")  # equations stay inside their paragraph

    paragraphs = [para for p in parsed.values() for para in p.paragraphs]
    sections = [s for p in parsed.values() for s in p.sections]
    for code in fp.PAPERS:
        report = pv.validate(code, paragraphs, sections)
        assert report["headings_missing"] == []
        assert report["coverage_pct"] >= 95.0  # measured 2026-11-12: 100.0 / 100.0 / 98.8

    tables = [t for p in parsed.values() for t in p.tables]
    figures = [f for p in parsed.values() for f in p.figures]
    for size in (256, 512):
        chunks = [c for code in fp.PAPERS
                  for c in ch.chunk_paper(code, sections, paragraphs, tables, figures, size, "c1")]
        assert len({c["chunk_id"] for c in chunks}) == len(chunks)
        assert max(c["n_tokens"] for c in chunks) <= size
        for code in fp.PAPERS:  # chunking neither drops nor duplicates a word of paragraph text
            para_words = sum(len(p["text"].split()) for p in paragraphs if p["paper"] == code)
            chunk_words = sum(len(c["text"].split()) for c in chunks if c["paper"] == code and c["kind"] == "text")
            assert chunk_words == para_words
        assert sum(c["kind"] == "table" for c in chunks if c["chunk_id"].startswith("MOSHI:T19:")) >= 1
