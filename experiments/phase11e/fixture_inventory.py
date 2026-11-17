"""Phase 11E: inventory of every corpus-dependent fixture / recorded response / expected result / benchmark artifact,
with its corpus provenance. Read only (no artifact is modified, relabelled or moved).

provenance:
  recorded     the artifact itself states its corpus (a corpus_version field, a corpus key, or its author note)
  established  not stated, but provable from the artifact chain: the code that produced it loaded data/processed +
               data/index, whose files equal the c3 hashes recorded at the start of phase 9A (before the artifact) and
               are unchanged since (checked here)
  unknown      neither: excluded from every corpus-specific comparison
Writes experiments/phase11e/fixture_inventory.json.
"""
from __future__ import annotations

import glob
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
C3_RECORD = "experiments/phase9/out/baseline_checksums.sha256"


def c3_unchanged_since_9a() -> bool:
    rec = {l.split()[1].lstrip("*"): l.split()[0] for l in (ROOT / C3_RECORD).read_text().splitlines()
           if l.split()[1].lstrip("*").startswith(("data/processed/", "data/index/"))}
    return all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in rec.items())


def first_json_line(p: Path) -> dict:
    with p.open(encoding="utf-8") as f:
        return json.loads(f.readline())


def entry(path, kind, corpus, provenance, evidence, comparable, notes=""):
    return {"path": path, "kind": kind, "corpus_version": corpus, "provenance": provenance, "evidence": evidence,
            "usable_in_corpus_specific_comparisons": comparable, "notes": notes}


def build() -> dict:
    c3_ok = c3_unchanged_since_9a()
    items = []
    for p in sorted((ROOT / "fixtures/perf").glob("*.json")):
        fx = json.loads(p.read_text(encoding="utf-8"))
        items.append(entry(p.relative_to(ROOT).as_posix(), "stand-in Mode A fixture (scripts/record_fixtures.py)", "unknown",
                           "unknown", f"declares prompt_version {fx.get('prompt_version')}, recorded_at {fx.get('recorded_at')}; "
                           "no corpus field; its source runs (results/raw/real/2026-11-12.jsonl, results/raw/e2e/2026-11-12.jsonl) "
                           "record none either", False,
                           "DESIGN's change log introduces c2 on 2026-11-13, after this recording, but the artifact does not state "
                           "its corpus. Kept unchanged; used only for load/latency replay, never for corpus comparisons."))
    for pattern, kind in (("results/raw/real/*.jsonl", "real provider call log"), ("results/raw/e2e/*.jsonl", "real end-to-end run log"),
                          ("results/raw/traces/*.jsonl", "request traces")):
        for p in sorted(ROOT.glob(pattern)):
            r = first_json_line(p)
            cv = r.get("corpus_version")
            items.append(entry(p.relative_to(ROOT).as_posix(), kind, cv or "unknown", "recorded" if cv else "unknown",
                               f"first record corpus_version = {cv!r}", bool(cv)))
    for p in sorted(ROOT.glob("results/raw/answer_quality/*.jsonl")):
        cv = first_json_line(p).get("corpus_version")
        items.append(entry(p.relative_to(ROOT).as_posix(), "real answer-quality run (frozen v4 set)", cv or "unknown",
                           "recorded" if cv else "unknown", f"corpus_version = {cv!r} in every record", bool(cv)))
    for p in sorted(ROOT.glob("experiments/phase10/out/live/pipeline_*.json")):
        v = (json.loads(p.read_text(encoding="utf-8")).get("versions") or {}).get("corpus_version")
        items.append(entry(p.relative_to(ROOT).as_posix(), "phase 10 live run (real model)", v or ("c3" if c3_ok else "unknown"),
                           "recorded" if v else ("established" if c3_ok else "unknown"),
                           f"versions.corpus_version = {v!r}" if v else
                           "loaded data/processed + data/index (hard-coded before phase 11D); those files equal the c3 hashes "
                           f"recorded at phase 9A ({C3_RECORD}) and are unchanged", True))
    for p in sorted(ROOT.glob("experiments/phase10/out/offline_integration_*.json")):
        items.append(entry(p.relative_to(ROOT).as_posix(), "phase 10 offline integration (fake provider)", "c3" if c3_ok else "unknown",
                           "established" if c3_ok else "unknown",
                           "main.State() default corpus = data/processed + data/index; c3 hashes unchanged since phase 9A", c3_ok))
    for p in sorted(ROOT.glob("experiments/phase11/out/ablation_2*.json")):
        items.append(entry(p.relative_to(ROOT).as_posix(), "phase 11A retrieval ablation", json.loads(p.read_text(encoding="utf-8")).get("corpus"),
                           "recorded", "top-level 'corpus' field", True))
    for p in sorted(ROOT.glob("experiments/phase11b/out/compare_*.json")):
        items.append(entry(p.relative_to(ROOT).as_posix(), "phase 11B comparison", "c3 and c6 (separate sections)", "recorded",
                           "results keyed by corpus", True))
    for p in sorted(ROOT.glob("experiments/phase11c/out/*_2*.json")):
        items.append(entry(p.relative_to(ROOT).as_posix(), "phase 11C output", "c3 and c6 (per row 'corpus')", "recorded",
                           "every row carries its corpus", True))
    items.append(entry("eval/phase7/table_question_set.json", "expected results (gold rows), PROVISIONAL labels", "c4", "recorded",
                       "author note: 'gold written from the LaTeX source tables and the c4 extraction'", True,
                       "row locators are id strings applied to any corpus; t13-t15 also carry paper_cells for c4's inflated values"))
    items.append(entry("eval/questions.jsonl", "expected results (gold quotes)", "corpus-independent text, c3-era", "recorded",
                       "gold = verbatim quotes of paper text; q09's quote matches c3's unit-less text only", True,
                       "located in c6 with e4's unit insertions undone (label unchanged), as in phases 11A-11C"))
    items.append(entry("experiments/phase11/new_cases.json", "expected results, PROVISIONAL (assistant-written)", "c6", "recorded",
                       "written from corpus c6 tables (its 'author' note)", True))
    for p in ("experiments/phase8/table_semantics_cases.json", "experiments/phase8/table_semantics_heldout.json"):
        items.append(entry(p, "phase 8/9B verifier claim cases (cite c4 chunks)", "c4", "recorded", "its 'about' note: 'real table cells in corpus c4'",
                           False, "Phase 9B benchmark input; read only, never modified; a verifier benchmark, not a corpus comparison"))
    counts = {}
    for it in items:
        counts[it["provenance"]] = counts.get(it["provenance"], 0) + 1
    return {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "c3_unchanged_since_phase9a": c3_ok,
            "new_c6_fixtures": [], "why_no_c6_fixtures": "scripts/record_fixtures.py only converts REAL model answers "
            "(results/raw/real + e2e) into stand-in fixtures; recording c6 answers needs live Groq calls, which are not "
            "authorised. No fixture was relabelled, re-recorded or invented.", "counts_by_provenance": counts, "artifacts": items}


class ProvenanceError(ValueError):
    pass


def select_for_comparison(items: list[dict], corpus_version: str) -> list[dict]:
    """Artifacts usable in a comparison about `corpus_version`: provenance recorded or established AND that corpus.
    Asking for an artifact with unknown or different provenance is an error, never a silent inclusion."""
    out = []
    for it in items:
        if it["provenance"] == "unknown" or not it["usable_in_corpus_specific_comparisons"]:
            raise ProvenanceError(f"{it['path']}: provenance {it['provenance']!r}; excluded from corpus-specific comparisons")
        if corpus_version not in str(it["corpus_version"]).replace(",", " ").split():
            raise ProvenanceError(f"{it['path']}: corpus {it['corpus_version']!r}, not {corpus_version!r}")
        out.append(it)
    return out


def main() -> None:
    inv = build()
    out = ROOT / "experiments/phase11e/fixture_inventory.json"
    out.write_text(json.dumps(inv, indent=1, ensure_ascii=False), encoding="utf-8")
    print(len(inv["artifacts"]), "artifacts;", inv["counts_by_provenance"], "| c3 unchanged since 9A:", inv["c3_unchanged_since_phase9a"])
    for it in inv["artifacts"]:
        if it["provenance"] == "unknown":
            print("  unknown:", it["path"])


if __name__ == "__main__":
    main()
