"""Phase 9A: ablation of c5's three changes, one factor at a time, in memory (same evaluator as evaluate_retrieval).

    uv run python experiments/phase9/ablation.py   # -> experiments/phase9/out/ablation_{UTC}.json

Variants (processed source: e2 = corpus/c4_control/processed, which equals c4; e3 = corpus/c5/processed):
    c4            e2, no section fix, no metadata    must equal the stored c4 chunks
    e3_only       e3
    section_only  e2 + Table 3 -> 5.2 (changes Table 3's index header only: section id and title)
    metadata_only e2 + metadata line (the line names Table 3's parsed section, 5.1)
    section_meta  e2 + section fix + metadata (isolates e3 by difference with c5)
    c5            e3 + section fix + metadata        must equal the stored c5 chunks
Each variant is compared with c4 per question; c5 is also compared with section_meta (= the effect of e3 given the
other two changes)."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/phase9")
import evaluate_retrieval as ev  # noqa: E402
from build_c5 import chunk_corpus  # noqa: E402

E2, E3 = Path("experiments/phase9/corpus/c4_control/processed"), Path("experiments/phase9/corpus/c5/processed")
VARIANTS = {"c4": (E2, False, False), "e3_only": (E3, False, False), "section_only": (E2, True, False),
            "metadata_only": (E2, False, True), "section_meta": (E2, True, True), "c5": (E3, True, True)}
STORED = {"c4": Path("data/corpus/c4/processed"), "c5": E3}


def strip_version(chunks):
    return [{k: v for k, v in c.items() if k != "corpus_version"} for c in chunks]


if __name__ == "__main__":
    embedder = ev.load_models()
    out = {"variants": {k: {"processed": str(v[0]), "section_fix": v[1], "metadata": v[2]} for k, v in VARIANTS.items()},
           "reproduction": {}, "summary": {}, "changes_vs_c4": {}, "rows": {}}
    for name, (src, fix, meta) in VARIANTS.items():
        chunks = {s: chunk_corpus(src, name, s, fix, meta) for s in (256, 512)}
        if name in STORED:
            same = all(strip_version(chunks[s]) == strip_version(
                [json.loads(l) for l in (STORED[name] / f"chunks_{s}.jsonl").read_text(encoding="utf-8").splitlines()])
                for s in (256, 512))
            out["reproduction"][name] = same
            if not same:
                raise SystemExit(f"variant {name} does not reproduce the stored {name} chunks")
        rows = ev.evaluate({s: ev.MemIndex(chunks[s], embedder) for s in (256, 512)})
        out["rows"][name], out["summary"][name] = rows, ev.summarise(rows)
        print(f"### {name}", flush=True)
    for name in VARIANTS:
        if name != "c4":
            out["changes_vs_c4"][name] = ev.changes(out["rows"]["c4"], out["rows"][name], name)
    out["changes_c5_vs_section_meta"] = ev.changes(out["rows"]["section_meta"], out["rows"]["c5"], "c5")
    # the disk-built c5 index must give the same contexts as the in-memory c5 (latest evaluate_retrieval output)
    latest = sorted(Path("experiments/phase9/out").glob("retrieval_eval_*.json"))
    if latest:
        disk = json.loads(latest[-1].read_text(encoding="utf-8"))["rows"]
        out["memory_equals_disk"] = {k: [r["context"] for r in disk[k]] == [r["context"] for r in out["rows"][k]]
                                     for k in ("c4", "c5")}
    for k in out["summary"]["c4"]:
        print(k)
        for name in VARIANTS:
            print(f"  {name:13} {out['summary'][name][k]}")
    for name, ch in out["changes_vs_c4"].items():
        print(f"## {name} vs c4: {len(ch)} changes")
        for c in ch:
            print("  ", c)
    print("## c5 vs section_meta (effect of e3):")
    for c in out["changes_c5_vs_section_meta"]:
        print("  ", c)
    print("reproduction", out["reproduction"], "memory_equals_disk", out.get("memory_equals_disk"))
    p = Path(f"experiments/phase9/out/ablation_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}.json")
    with p.open("x", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=str)
    print(p)
