"""Phase 9A: build the experimental corpus c5 (parent c4) under experiments/phase9/corpus/. Never overwrites.

    uv run python experiments/phase9/build_c5.py

1. Gates: extraction_e3.validate() and section_correction.validate() must pass.
2. Control: the production stages with c4's settings (latex_parse e2 -> pdf_validate -> chunker) into
   corpus/c4_control/processed; every file must equal data/corpus/c4/processed byte for byte, or the build stops.
3. c5: the same stages with extraction e3 (latex_parse wrapped by extraction_e3.e3()), then section metadata s1
   (Table 3 -> 5.2) and table-chunk metadata k1, then index_build into corpus/c5/index; manifest.json.
Chunking always uses the parsed sections (a table's section_id_parsed when present), so IDs, order and boundaries
equal the parent's; corrections and metadata are applied to the chunks afterwards (chunk_corpus)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/phase9")
import extraction_e3  # noqa: E402
import section_correction  # noqa: E402
import table_chunk_metadata  # noqa: E402

from relay.config import piece_sizes
from relay.retrieval import chunker, index_build, latex_parse, pdf_validate
from relay.retrieval.fetch_papers import PAPERS, RAW_DIR

ROOT = Path("experiments/phase9/corpus")
C4 = Path("data/corpus/c4")
PROD_CODE = ["relay/retrieval/latex_parse.py", "relay/retrieval/pdf_validate.py", "relay/retrieval/chunker.py",
             "relay/retrieval/index_build.py", "relay/llm/tokenizer.py", "config/retrieval_settings.yaml"]
EXP_CODE = ["experiments/phase9/extraction_e3.py", "experiments/phase9/section_correction.py",
            "experiments/phase9/table_chunk_metadata.py", "experiments/phase9/build_c5.py"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha256(root: Path) -> str:  # same definition as scripts/build_corpus.py, so source hashes compare with c4's
    h = hashlib.sha256()
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(f.relative_to(root).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def read(processed: Path, kind: str) -> list[dict]:
    return [json.loads(l) for l in (processed / f"{kind}.jsonl").read_text(encoding="utf-8").splitlines()]


def chunk_corpus(processed: Path, version: str, size: int, fix_sections: bool, metadata: bool) -> list[dict]:
    """Chunks of a processed corpus with the given experimental changes (the single definition used by the build
    and the ablation)."""
    sections, paragraphs, figures = (read(processed, k) for k in ("sections", "paragraphs", "figures"))
    tables = []
    for t in read(processed, "tables"):  # chunk with the parsed section so that IDs and order stay the parent's
        t = dict(t)
        t["section_id"] = t.pop("section_id_parsed", t["section_id"])
        tables.append(t)
    chunks = [c for code in PAPERS for c in chunker.chunk_paper(code, sections, paragraphs, tables, figures, size, version)]
    if fix_sections:
        chunks, _ = section_correction.correct_chunks(chunks, sections)
    if metadata:
        chunks = table_chunk_metadata.add_metadata(chunks, tables)
    assert len({c["chunk_id"] for c in chunks}) == len(chunks), "duplicate chunk IDs"
    return chunks


def parse_into(processed: Path, e3: bool) -> None:
    processed.mkdir(parents=True)
    if e3:
        with extraction_e3.e3():
            latex_parse.main(RAW_DIR, processed, "e2")
    else:
        latex_parse.main(RAW_DIR, processed, "e2")
    pdf_validate.main(RAW_DIR, processed)


def build() -> dict:
    control, c5 = ROOT / "c4_control", ROOT / "c5"
    for d in (control, c5):
        if d.exists():
            raise SystemExit(f"{d} exists: a corpus is built once, never overwritten")
    gates = {"e3": extraction_e3.validate(), "s1": section_correction.validate()}
    if not all(g["all_checks_passed"] for g in gates.values()):
        raise SystemExit(f"validation gate failed: { {k: g['checks'] for k, g in gates.items()} }")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sizes = piece_sizes()

    # 2. control: must reproduce c4 exactly
    parse_into(control / "processed", e3=False)
    for s in sizes:
        latex_parse.write_jsonl(control / "processed" / f"chunks_{s}.jsonl", chunk_corpus(control / "processed", "c4", s, False, False))
    stored = sorted(p.name for p in (C4 / "processed").iterdir())
    rebuilt = sorted(p.name for p in (control / "processed").iterdir())
    mismatch = [n for n in stored if n not in rebuilt or sha256(C4 / "processed" / n) != sha256(control / "processed" / n)]
    if mismatch or stored != rebuilt:
        raise SystemExit(f"control does not reproduce c4: {mismatch or (stored, rebuilt)}")

    # 3. c5
    processed, index = c5 / "processed", c5 / "index"
    parse_into(processed, e3=True)
    fixed, records = section_correction.correct_tables(read(processed, "tables"))
    latex_parse.write_jsonl(processed / "tables.jsonl", fixed)
    (processed / "section_corrections.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    counts = {}
    for s in sizes:
        chunks = chunk_corpus(processed, "c5", s, True, True)
        latex_parse.write_jsonl(processed / f"chunks_{s}.jsonl", chunks)
        tables = [c for c in chunks if c["kind"] == "table"]
        counts[s] = {"chunks": len(chunks), "table_chunks": len(tables),
                     "table_chunks_over_piece_size": sum(c["n_tokens"] > s for c in tables)}
    model = index_build.load_model()
    indexes = [index_build.build(s, model, processed, index) for s in sizes]
    manifest = {
        "corpus_version": "c5", "parent_corpus": "c4", "experimental": True, "built_at": started,
        "extraction": "e3", "section_metadata": section_correction.VERSION, "chunking": {
            "version": table_chunk_metadata.VERSION, "base": "relay.retrieval.chunker (c4 settings) + table-chunk metadata line",
            "piece_sizes": sizes, "overlap": "none (paragraph packing; tables split by rows)", "counts": counts},
        "embedding": {"model": index_build.EMBED_MODEL, "query_prefix": index_build.QUERY_PREFIX,
                      "index_text": "paper title | section_id section_title, newline, chunk text", "indexes": indexes},
        "section_corrections": records,
        "control": {"dir": str(control), "reproduces_c4_processed_byte_for_byte": True, "files": stored},
        "gates": {k: g["checks"] for k, g in gates.items()},
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "sources": {code: {"src_tree_sha256": tree_sha256(RAW_DIR / code / "src"),
                           **{f.name + "_sha256": sha256(f) for f in sorted((RAW_DIR / code).glob("*")) if f.is_file()}}
                    for code in PAPERS},
        "code_sha256": {c: sha256(Path(c)) for c in PROD_CODE + EXP_CODE},
        "outputs_sha256": {f.relative_to(c5).as_posix(): sha256(f) for f in sorted(c5.rglob("*")) if f.is_file()},
    }
    parent = json.loads((C4 / "manifest.json").read_text(encoding="utf-8"))
    manifest["sources_equal_parent"] = manifest["sources"] == parent["sources"]
    (c5 / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    m = build()
    print(json.dumps({k: m[k] for k in ("corpus_version", "parent_corpus", "extraction", "section_metadata", "chunking",
                                        "control", "sources_equal_parent")}, indent=1))
