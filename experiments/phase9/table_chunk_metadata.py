"""Phase 9A, investigation C: section-and-label metadata on table chunks (chunking version k1).

Each table chunk (kind "table") gets one line after its "Columns:" line:

    Source: <paper title> (<PAPER>); Table <n> (<latex label>); section <id> <title>; rows <a>-<b> of <N>

The chunk's first line already holds "Table <n>: <caption>" (the existing short caption), and its row lines are
unchanged. Added after chunking: chunk boundaries, IDs, order and every other chunk stay as they are; a table chunk
grows by the line's tokens (it may exceed the piece size; reported). Text, caption and figure chunks are untouched.
The section in the line is the chunk's own section_id/section_title, so it shows the corrected section only when
section_correction has been applied first."""
from __future__ import annotations

import copy
import sys

sys.path.insert(0, ".")
from relay.llm.tokenizer import count
from relay.retrieval.index_build import PAPER_TITLES

VERSION = "k1"


def metadata_line(chunk: dict, label: str | None, first: int, last: int, total: int) -> str:
    return (f"Source: {PAPER_TITLES[chunk['paper']]} ({chunk['paper']}); Table {chunk['table_id'][1:]} ({label or 'no label'}); "
            f"section {chunk['section_id']} {chunk['section_title']}; rows {first}-{last} of {total}")


def add_metadata(chunks: list[dict], tables: list[dict]) -> list[dict]:
    """Copies of the chunks with the metadata line in every table chunk. Rows are numbered over the table's chunks
    in their order (r01, r02, ...), which is the order of the table's rows."""
    info = {(t["paper"], t["table_id"]): t for t in tables}
    seen: dict[tuple[str, str], int] = {}
    out = []
    for c in chunks:
        c = copy.deepcopy(c)
        if c["kind"] == "table":
            key = (c["paper"], c["table_id"])
            t = info[key]
            first = seen.get(key, 0) + 1
            last = first + len(c["rows"]) - 1
            seen[key] = last
            lines = c["text"].split("\n")
            at = next(i for i, l in enumerate(lines) if l.startswith("Columns: ")) + 1
            c["text"] = "\n".join(lines[:at] + [metadata_line(c, t["label"], first, last, len(t["rows"]))] + lines[at:])
            c["n_tokens"] = count(c["text"])
        out.append(c)
    for key, n in seen.items():
        if n != len(info[key]["rows"]):
            raise ValueError(f"{key}: chunks hold {n} rows, table has {len(info[key]['rows'])}")
    return out
