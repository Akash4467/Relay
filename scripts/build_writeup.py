"""Assemble writeup/relay_writeup.md from the template and results/tables.md, so the report's tables are the
generated ones, never retyped (CLAUDE.md rule 4).

    uv run python scripts/build_writeup.py
"""
from __future__ import annotations

import re
from pathlib import Path

TEMPLATE = Path("writeup/relay_writeup.template.md")
TABLES = Path("results/tables.md")
OUT = Path("writeup/relay_writeup.md")
SECTIONS = {"TABLE1": "Table 1", "TABLE2A": "Table 2a", "TABLE2B": "Table 2b", "TABLE3": "Table 3: Retrieval settings", "TABLE3B": "Table 3b"}


def section(tables_md: str, title: str) -> str:
    """The '## <title>...' heading and its table, up to the next heading."""
    m = re.search(rf"^## {re.escape(title)}\b.*?(?=^## |\Z)", tables_md, re.S | re.M)
    if not m:
        raise ValueError(f"{title} not found in {TABLES}")
    return m.group(0).strip().replace("## ", "**", 1).replace("\n", "**\n", 1)


def main() -> str:
    tables = TABLES.read_text(encoding="utf-8")
    text = TEMPLATE.read_text(encoding="utf-8")
    for marker, title in SECTIONS.items():
        text = text.replace("{{" + marker + "}}", section(tables, title))
    if "{{" in text:
        raise ValueError("unfilled marker left in the write-up")
    OUT.write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    out = main()
    print(f"{OUT}: {len(out.split())} words")
