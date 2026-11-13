"""Render writeup/relay_writeup.md to PDF (A4, 10pt body, 7.5pt tables) with headless Chrome and print the page count,
so the 2-page limit is checked the same way every time. Reads the write-up only; writes the PDF next to it.

    uv run python scripts/render_writeup.py [--chrome PATH]
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import pymupdf
from markdown_it import MarkdownIt

SRC = Path("writeup/relay_writeup.md")
OUT = Path("writeup/relay_writeup.pdf")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
CSS = """@page{size:A4;margin:1.4cm 1.5cm}body{font:10pt/1.25 Arial,sans-serif}h1{font-size:12.5pt;margin:0 0 3pt}
h2{font-size:10.5pt;margin:6pt 0 1pt}p{margin:0 0 3pt}ol,ul{margin:0 0 3pt;padding-left:15pt}
table{border-collapse:collapse;font-size:7.5pt;margin:1pt 0 3pt;width:100%}td,th{border:.5pt solid #999;padding:.5pt 3pt}
table.num td{white-space:nowrap}"""


def render(src: Path = SRC, out: Path = OUT, chrome: str = CHROME) -> int:
    body = MarkdownIt("commonmark").enable("table").render(src.read_text(encoding="utf-8"))
    # latency tables: keep "p50 / p95 / p99 (N)" cells on one line; text-heavy tables (Table 1) still wrap
    body = re.sub(r"<table>(?=(?:(?!</table>).)*?\(ms\))", '<table class="num">', body, flags=re.S)
    html = out.with_suffix(".html")
    html.write_text(f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>",
                    encoding="utf-8")
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={out.resolve()}",
                    html.resolve().as_uri()], check=True, capture_output=True)
    html.unlink()
    return pymupdf.open(out).page_count


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--chrome", default=CHROME)
    a = ap.parse_args()
    print(f"{a.out}: {render(a.src, a.out, a.chrome)} pages")
