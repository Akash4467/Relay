"""Assemble writeup/relay_writeup.md from the template and results/tables.md, so the report's tables are the
generated ones, never retyped (CLAUDE.md rule 4).

    uv run python scripts/build_writeup.py
    uv run python scripts/build_writeup.py docs/architecture.template.md docs/architecture.md   # explainer
"""
from __future__ import annotations

import re
from pathlib import Path

TEMPLATE = Path("writeup/relay_writeup.template.md")
TABLES = Path("results/tables.md")
OUT = Path("writeup/relay_writeup.md")
SECTIONS = {"TABLE1": "Table 1", "TABLE2A": "Table 2a", "TABLE2B": "Table 2b", "TABLE3": "Table 3: Retrieval settings", "TABLE3B": "Table 3b",
            "TIERS": "Reply tiers", "COMPONENTS": "Component medians"}


def section(tables_md: str, title: str) -> str:
    """The '## <title>...' heading and its table, up to the next heading."""
    m = re.search(rf"^## {re.escape(title)}\b.*?(?=^## |\Z)", tables_md, re.S | re.M)
    if not m:
        raise ValueError(f"{title} not found in {TABLES}")
    return m.group(0).strip().replace("## ", "**", 1).replace("\n", "**\n", 1)


# optional generated blocks from other evidence files (only when a template uses the marker); never typed by hand
EXTRA = {"E2E": (Path("assessment/e2e/e2e_tables.md"), "Latest measurement")}  # from assessment/e2e/make_e2e_tables.py


def main(template: Path = TEMPLATE, out: Path = OUT) -> str:
    tables = TABLES.read_text(encoding="utf-8")
    text = template.read_text(encoding="utf-8")
    for marker, title in SECTIONS.items():
        text = text.replace("{{" + marker + "}}", section(tables, title))
    for marker, (path, title) in EXTRA.items():
        if "{{" + marker + "}}" in text:
            text = text.replace("{{" + marker + "}}", section(path.read_text(encoding="utf-8"), title))
    if "{{" in text:
        raise ValueError("unfilled marker left in the write-up")
    out.write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    import sys
    paths = [Path(a) for a in sys.argv[1:3]] if len(sys.argv) == 3 else [TEMPLATE, OUT]
    text = main(*paths)
    print(f"{paths[1]}: {len(text.split())} words")
