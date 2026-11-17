"""Check parsed text against the PDFs and attach page numbers (DESIGN §6.3).

Adds "page" (1-based, or null) to every record in data/processed/paragraphs.jsonl and writes
data/processed/pdf_validation.json with per-paper coverage and any section heading not found.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pymupdf

from relay.retrieval.fetch_papers import PAPERS, RAW_DIR
from relay.retrieval.latex_parse import PROCESSED_DIR, write_jsonl

START_WORDS = 12
WINDOW = 8
SCAN_WORDS = 40


def norm_words(text: str) -> list[str]:
    text = re.sub(r"\[EQ\][^\n]*", " ", text)  # display equations are TeX here but typeset in the PDF
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)  # words hyphenated across PDF lines
    return re.findall(r"[a-z0-9]+", text)


def page_texts(pdf_path: Path) -> list[str]:
    """Each page as ' w1 w2 ... ' so a phrase match never cuts a word."""
    with pymupdf.open(pdf_path) as doc:
        return [" " + " ".join(norm_words(p.get_text())) + " " for p in doc]


def locate(text: str, pages: list[str]) -> int | None:
    """1-based page holding the paragraph's opening words; else any 8-word window from its first 40 words."""
    words = norm_words(text)
    if not words:
        return None
    phrases = [words[:START_WORDS]]
    if len(words) > WINDOW:
        phrases += [words[i:i + WINDOW] for i in range(0, min(len(words), SCAN_WORDS) - WINDOW + 1)]
    for phrase in phrases:
        needle = " " + " ".join(phrase) + " "
        for n, page in enumerate(pages, 1):
            if needle in page:
                return n
    return None


def heading_found(section: dict, pages: list[str]) -> bool:
    words = norm_words(section["title"])
    if section["section_id"][:1].isdigit() or "." in section["section_id"] or len(section["section_id"]) == 1:
        words = norm_words(section["section_id"]) + words  # numbered: "3 2 1 scaled dot product ..."
    needle = " " + " ".join(words) + " "
    return any(needle in p for p in pages)


def validate(code: str, paragraphs: list[dict], sections: list[dict], raw_dir: Path = RAW_DIR) -> dict:
    """Sets paragraph["page"] in place for this paper's paragraphs and returns the report."""
    pages = page_texts(raw_dir / code / "paper.pdf")
    paras = [p for p in paragraphs if p["paper"] == code]
    for p in paras:
        p["page"] = locate(p["text"], pages)
    located = sum(p["page"] is not None for p in paras)
    missing = [s["section_id"] for s in sections if s["paper"] == code and not heading_found(s, pages)]
    return {"paper": code, "paragraphs": len(paras), "located": located,
            "coverage_pct": round(100 * located / len(paras), 1) if paras else 0.0,
            "headings_missing": missing}


def main(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> list[dict]:
    def read(kind: str) -> list[dict]:
        return [json.loads(l) for l in (processed_dir / f"{kind}.jsonl").read_text(encoding="utf-8").splitlines()]

    paragraphs, sections = read("paragraphs"), read("sections")
    reports = [validate(code, paragraphs, sections, raw_dir) for code in PAPERS]
    write_jsonl(processed_dir / "paragraphs.jsonl", paragraphs)
    (processed_dir / "pdf_validation.json").write_text(json.dumps(reports, indent=2))
    return reports


if __name__ == "__main__":
    for r in main():
        print(r)
