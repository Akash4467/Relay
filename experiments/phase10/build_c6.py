"""Phase 10C: corpus c6 = extraction e4 (e2 marks + e3 phantom fix + siunitx units), c4 chunking (no c5 section fix,
no c5 metadata line), same embedding/index settings. Built once under experiments/phase10/corpus/c6, never
overwritten; c3, c4 and c5 are only read.

Gate before building: a full parse of all three papers under e3 and e4 differs ONLY where a siunitx command is,
and each such difference only inserts unit text (no word removed, no table/structure change).

    uv run python experiments/phase10/build_c6.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/phase9")
sys.path.insert(0, "experiments/phase10")
import build_c5  # noqa: E402
import extraction_e3  # noqa: E402
import extraction_e4  # noqa: E402
from relay.config import piece_sizes  # noqa: E402
from relay.retrieval import index_build, latex_parse, pdf_validate  # noqa: E402
from relay.retrieval.fetch_papers import PAPERS, RAW_DIR  # noqa: E402

OUT = Path("experiments/phase10/corpus/c6")
REPORT = Path("experiments/phase10/out/e4_changes.json")


def inserted_words(old: str, new: str) -> list[str] | None:
    """Words e4 added; None if anything else changed (a word removed or replaced)."""
    import difflib
    a, b = extraction_e3.words(old), extraction_e3.words(new)
    added = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        if op != "insert":
            return None
        added += b[j1:j2]
    return added


def validate() -> dict:
    changes, problems = [], []
    for code in PAPERS:
        with extraction_e3.e3():
            a = latex_parse.parse_paper(code, RAW_DIR, extraction="e2")
        b = extraction_e4.parse_paper_e4(code)
        d = extraction_e3.diff(a, b)
        if d["structure"] or d["tables"] or d["cells"] or d["sections"] or d["figures"]:
            problems.append({"paper": code, "structure": d["structure"], "tables": d["tables"][:5],
                             "cells": [{k: v for k, v in c.items() if k != "_rows"} for c in d["cells"][:5]],
                             "sections": len(d["sections"]), "figures": len(d["figures"])})
        for p in d["paragraphs"]:
            added = inserted_words(p["e2"]["text"], p["e3"]["text"])
            rec = {"paper": code, "paragraph": f"{p['e2']['section_id']}#{p['e2']['para_idx']}",
                   "inserted": added, "e3_excerpt": p["e2"]["text"][:200], "e4_excerpt": p["e3"]["text"][:200]}
            changes.append(rec)
            if added is None or not added:
                problems.append({"paper": code, "not_a_pure_unit_insertion": rec})
    units = sorted({w for c in changes for w in (c["inserted"] or [])})
    report = {"version": extraction_e4.VERSION, "changed_paragraphs": changes, "inserted_tokens": units,
              "unknown_unit_macros": sorted(set(extraction_e4.UNKNOWN)), "problems": problems,
              "all_checks_passed": not problems and len(changes) > 0}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report


def build() -> dict:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: a corpus is built once, never overwritten")
    gate = validate()
    if not gate["all_checks_passed"]:
        raise SystemExit(f"e4 gate failed: {gate['problems'][:3]}")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    processed, index = OUT / "processed", OUT / "index"
    processed.mkdir(parents=True)
    with extraction_e4.e4():
        latex_parse.main(RAW_DIR, processed, "e2")
    pdf_validate.main(RAW_DIR, processed)
    sizes, counts = piece_sizes(), {}
    for s in sizes:
        chunks = build_c5.chunk_corpus(processed, "c6", s, fix_sections=False, metadata=False)
        latex_parse.write_jsonl(processed / f"chunks_{s}.jsonl", chunks)
        counts[s] = {"chunks": len(chunks), "table_chunks": sum(c["kind"] == "table" for c in chunks)}
    model = index_build.load_model()
    indexes = [index_build.build(s, model, processed, index) for s in sizes]
    manifest = {
        "corpus_version": "c6", "parent_corpus": "c4", "experimental": True, "built_at": started,
        "extraction": "e4 = e2 + e3 (phantoms) + siunitx units", "section_metadata": "none (c4 placement)",
        "chunking": {"base": "relay.retrieval.chunker, c4 settings, no metadata line", "piece_sizes": sizes,
                     "counts": counts},
        "embedding": {"model": index_build.EMBED_MODEL, "query_prefix": index_build.QUERY_PREFIX, "indexes": indexes},
        "gate": {"changed_paragraphs": len(gate["changed_paragraphs"]), "inserted_tokens": gate["inserted_tokens"],
                 "unknown_unit_macros": gate["unknown_unit_macros"], "report": REPORT.as_posix()},
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "sources": {code: {"src_tree_sha256": build_c5.tree_sha256(RAW_DIR / code / "src")} for code in PAPERS},
        "code_sha256": {c: build_c5.sha256(Path(c)) for c in build_c5.PROD_CODE + [
            "experiments/phase9/extraction_e3.py", "experiments/phase10/extraction_e4.py", "experiments/phase10/build_c6.py"]},
        "outputs_sha256": {f.relative_to(OUT).as_posix(): build_c5.sha256(f) for f in sorted(OUT.rglob("*")) if f.is_file()},
    }
    parent = json.loads((build_c5.C4 / "manifest.json").read_text(encoding="utf-8"))
    manifest["sources_equal_parent"] = {k: v["src_tree_sha256"] for k, v in manifest["sources"].items()} == \
        {k: v["src_tree_sha256"] for k, v in parent["sources"].items()}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    if sys.argv[1:] == ["validate"]:
        r = validate()
        print(json.dumps({k: r[k] for k in ("inserted_tokens", "unknown_unit_macros", "all_checks_passed")}),
              len(r["changed_paragraphs"]), "paragraphs;", len(r["problems"]), "problems")
    else:
        m = build()
        print("c6 built:", m["chunking"]["counts"], "| sources equal c4:", m["sources_equal_parent"])
