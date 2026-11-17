"""Build a versioned corpus in its own directory, never over an existing one or the default data/processed, data/index.

    uv run python scripts/build_corpus.py --version c4 --extraction e2 --out data/corpus/c4

Runs the existing stages with their existing configuration (latex_parse -> pdf_validate -> chunker -> index_build;
piece sizes from config/retrieval_settings.yaml; embedding model BAAI/bge-small-en-v1.5) and writes
{out}/processed/, {out}/index/ and {out}/manifest.json (corpus version, extraction version, source and code hashes,
chunking and embedding configuration, build time, output hashes). The default corpus and config are not touched;
a corpus is used only when its directory is passed explicitly (relay.retrieval.search.Index(..., index_dir, processed_dir)).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from relay.config import piece_sizes
from relay.retrieval import chunker, index_build, latex_parse, pdf_validate
from relay.retrieval.fetch_papers import PAPERS, RAW_DIR
from relay.retrieval.latex_parse import PROCESSED_DIR

DEFAULT_DIRS = {PROCESSED_DIR.resolve(), index_build.INDEX_DIR.resolve()}
CODE = ["relay/retrieval/latex_parse.py", "relay/retrieval/pdf_validate.py", "relay/retrieval/chunker.py",
        "relay/retrieval/index_build.py", "relay/llm/tokenizer.py", "config/retrieval_settings.yaml"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha256(root: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(f.relative_to(root).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


def build(version: str, extraction: str, out: Path) -> dict:
    out = Path(out)
    processed, index = out / "processed", out / "index"
    if out.exists():
        raise SystemExit(f"{out} exists: a corpus version is built once, never overwritten")
    if {processed.resolve(), index.resolve()} & DEFAULT_DIRS or out.resolve() in {p.parent for p in DEFAULT_DIRS}:
        raise SystemExit("refusing to build into the default corpus directories")
    processed.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    parsed = latex_parse.main(RAW_DIR, processed, extraction)
    pages = pdf_validate.main(RAW_DIR, processed)
    sizes = piece_sizes()
    chunks = chunker.main(processed, sizes, corpus_version=version)
    model = index_build.load_model()
    indexes = [index_build.build(s, model, processed, index) for s in sizes]
    manifest = {
        "corpus_version": version, "extraction": extraction, "built_at": started,
        "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
        "sources": {code: {"src_tree_sha256": tree_sha256(RAW_DIR / code / "src"),
                           **{f.name + "_sha256": sha256(f) for f in sorted((RAW_DIR / code).glob("*")) if f.is_file()}}
                    for code in PAPERS},
        "code_sha256": {c: sha256(Path(c)) for c in CODE},
        "chunking": {"piece_sizes": sizes, "chunks": {s: len(c) for s, c in chunks.items()}},
        "embedding": {"model": index_build.EMBED_MODEL, "query_prefix": index_build.QUERY_PREFIX, "indexes": indexes},
        "parsed": {code: {k: len(getattr(p, k)) for k in ("sections", "paragraphs", "tables", "figures")}
                   for code, p in parsed.items()},
        "pdf_validation": pages,
        "outputs_sha256": {f.relative_to(out).as_posix(): sha256(f) for f in sorted(out.rglob("*")) if f.is_file()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--extraction", required=True, choices=["e1", "e2"])
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    m = build(a.version, a.extraction, a.out)
    print(json.dumps({k: m[k] for k in ("corpus_version", "extraction", "chunking", "parsed")}, indent=1))
