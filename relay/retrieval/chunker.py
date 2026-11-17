"""Section-aware chunking (DESIGN §6.4): data/processed/*.jsonl -> data/processed/chunks_{size}.jsonl.

Text chunks pack whole paragraphs, falling back to whole sentences, and never cross a section.
Table rows are grouped under their caption and header line; figure captions are their own chunks.
IDs: {PAPER}:{section_id}:c{NNN} (NNN = order within the paper) and {PAPER}:T{n}:r{NN} for tables.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from relay.config import app_config, piece_sizes
from relay.llm.tokenizer import count, split_by_tokens
from relay.retrieval.fetch_papers import PAPERS
from relay.retrieval.latex_parse import PROCESSED_DIR, write_jsonl
from relay.verify.sentences import split_sentences


def _units(paragraphs: list[dict], size: int) -> list[tuple[int, str]]:
    """(para_idx, text) pieces of at most `size` tokens: whole paragraphs, else sentences, else token cuts."""
    units = []
    for p in paragraphs:
        if count(p["text"]) <= size:
            units.append((p["para_idx"], p["text"]))
            continue
        for sent in split_sentences(p["text"]):
            for piece in (split_by_tokens(sent, size) if count(sent) > size else [sent]):
                units.append((p["para_idx"], piece))
    return units


def pack_section(paragraphs: list[dict], size: int) -> list[dict]:
    """Greedy packing of units in order. A whole paragraph that does not fit starts a new chunk, so
    chunks follow paragraph boundaries whenever a paragraph is small enough to stay whole."""
    chunks: list[dict] = []
    cur: list[tuple[int, str]] = []

    def text_of(parts: list[tuple[int, str]]) -> str:
        out = ""
        for i, (idx, t) in enumerate(parts):
            out += t if i == 0 else (" " if idx == parts[i - 1][0] else "\n\n") + t
        return out

    def flush() -> None:
        if cur:
            chunks.append({"para_span": [cur[0][0], cur[-1][0]], "text": text_of(cur)})
            cur.clear()

    for idx, text in _units(paragraphs, size):
        if count(text_of(cur + [(idx, text)])) <= size:
            cur.append((idx, text))
            continue
        flush()
        cur.append((idx, text))
    flush()
    return chunks


def short_caption(caption: str, size: int) -> str:
    """Long captions would leave room for one row per chunk; row chunks then repeat only the first sentence."""
    if count(caption) <= size // 4:
        return caption
    return (split_sentences(caption) or [caption])[0]


_SPLIT = re.compile(r"\((dev|development|test|train|training|valid|validation)\)", re.I)


def metric_split(header: str) -> tuple[str, str | None]:
    """"BLEU / (dev)" -> ("BLEU", "dev"); "Training Cost (FLOPs) / EN-DE" -> (unchanged, None)."""
    parts = [p.strip() for p in header.split(" / ")]
    splits = [m.group(1).lower() for p in parts if (m := _SPLIT.fullmatch(p))]
    metric = " / ".join(p for p in parts if not _SPLIT.fullmatch(p)) or header
    return metric, (splits[0] if splits else None)


def structured_row(row: dict) -> dict:
    """One table row as explicit evidence: its configuration and one record per cell (column, metric, split, value).
    Rows of the same table stay separate records, so siblings are never read as competing values."""
    return {"configuration": row["row_header"],
            "values": [dict(zip(("metric", "split"), metric_split(col)), column=col, value=val)
                       for col, val in row["cells"].items() if val != ""]}  # e2 keeps empty cells' places; e1 has none


def table_chunks(table: dict, size: int) -> list[dict]:
    """Caption + header line + as many rows as fit; every chunk repeats the (short) caption and headers.
    Each row line names its configuration ("Configuration base: ..."), so a question about a configuration can
    match its row; the chunk also keeps the rows as structured records for the verifier."""
    n = table["table_id"][1:]
    head = f"Table {n}: {short_caption(table['caption'], size)}\nColumns: {' | '.join(table['headers'])}"
    lines = [f"Configuration {t}" for t in table["row_texts"]]
    groups: list[list[int]] = []
    for i, line in enumerate(lines):
        if groups and count(head + "\n" + "\n".join([lines[j] for j in groups[-1]] + [line])) <= size:
            groups[-1].append(i)
        else:
            groups.append([i])  # shortcut: a caption longer than the piece still keeps one row, so the chunk overflows
    return [{"text": head + "\n" + "\n".join(lines[i] for i in g), "rows": [structured_row(table["rows"][i]) for i in g]}
            for g in groups or [[]]]


def chunk_paper(code: str, sections: list[dict], paragraphs: list[dict], tables: list[dict],
                figures: list[dict], size: int, corpus_version: str) -> list[dict]:
    titles = {s["section_id"]: s["title"] for s in sections if s["paper"] == code}
    paras_by_sec: dict[str, list[dict]] = defaultdict(list)
    for p in paragraphs:
        if p["paper"] == code:
            paras_by_sec[p["section_id"]].append(p)
    figs_by_sec: dict[str, list[dict]] = defaultdict(list)
    for f in figures:
        if f["paper"] == code:
            figs_by_sec[f["section_id"]].append(f)

    out: list[dict] = []

    def add(section_id: str, kind: str, text: str, para_span: list[int] | None, page: int | None,
            chunk_id: str | None = None, rows: list[dict] | None = None) -> None:
        out.append({"chunk_id": chunk_id or f"{code}:{section_id}:c{len([c for c in out if c['kind'] != 'table']) + 1:03d}",
                    "paper": code, "section_id": section_id, "section_title": titles.get(section_id, ""),
                    "kind": kind, "piece_tokens": size, "para_span": para_span, "text": text,
                    "n_tokens": count(text), "page": page, "corpus_version": corpus_version}
                   | ({"table_id": chunk_id.split(":")[1], "rows": rows} if kind == "table" else {}))

    for s in (s for s in sections if s["paper"] == code):
        sid = s["section_id"]
        paras = paras_by_sec[sid]
        pages = {p["para_idx"]: p.get("page") for p in paras}
        for c in pack_section(paras, size):
            page = next((pages[i] for i in range(c["para_span"][0], c["para_span"][1] + 1) if pages.get(i)), None)
            add(sid, "text", c["text"], c["para_span"], page)
        for f in figs_by_sec[sid]:
            cap = f"Figure {f['figure_id'][1:]}: {f['caption']}"
            for piece in pack_section([{"para_idx": 0, "text": cap}], size):
                add(sid, "caption", piece["text"], None, None)
        for t in (t for t in tables if t["paper"] == code and t["section_id"] == sid):
            if short_caption(t["caption"], size) != t["caption"]:  # the full caption still gets retrieved
                for piece in pack_section([{"para_idx": 0, "text": f"Table {t['table_id'][1:]}: {t['caption']}"}], size):
                    add(sid, "caption", piece["text"], None, None)
            for k, c in enumerate(table_chunks(t, size), 1):
                add(sid, "table", c["text"], None, None, f"{code}:{t['table_id']}:r{k:02d}", c["rows"])
    return out


def main(processed_dir: Path = PROCESSED_DIR, sizes: list[int] | None = None,
         corpus_version: str | None = None) -> dict[int, list[dict]]:
    def read(kind: str) -> list[dict]:
        return [json.loads(l) for l in (processed_dir / f"{kind}.jsonl").read_text(encoding="utf-8").splitlines()]

    sections, paragraphs, tables, figures = (read(k) for k in ("sections", "paragraphs", "tables", "figures"))
    corpus_version = corpus_version or app_config()["corpus_version"]
    result = {}
    for size in sizes or piece_sizes():
        chunks = [c for code in PAPERS
                  for c in chunk_paper(code, sections, paragraphs, tables, figures, size, corpus_version)]
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "duplicate chunk IDs"
        write_jsonl(processed_dir / f"chunks_{size}.jsonl", chunks)
        result[size] = chunks
    return result


if __name__ == "__main__":
    for size, chunks in main().items():
        by_kind = defaultdict(int)
        for c in chunks:
            by_kind[c["kind"]] += 1
        over = sum(c["n_tokens"] > size for c in chunks)
        print(f"{size}: {len(chunks)} chunks {dict(by_kind)}, max {max(c['n_tokens'] for c in chunks)} tokens, "
              f"{over} over the piece size")
