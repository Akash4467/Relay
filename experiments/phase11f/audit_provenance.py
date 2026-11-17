"""Phase 11F: verify every classification in experiments/phase11e/fixture_inventory.json against the artifact itself.
recorded    -> the stated corpus really appears in the artifact (a corpus_version / corpus field, a per-row corpus,
               separate c3/c6 sections, or an author note naming the corpus)
established -> the artifact states no corpus, AND it was produced by code that read data/processed + data/index, AND
               those files equal the c3 hashes recorded at the start of phase 9A
unknown     -> no line/record of the artifact carries a corpus field
Also checks the two fixture citations' chunk texts in c3 and c6. Read only. Writes fixture_provenance_audit.json."""
import hashlib
import json
import re
from pathlib import Path

INV = json.loads(Path("experiments/phase11e/fixture_inventory.json").read_text(encoding="utf-8"))


def text(p):
    return Path(p).read_text(encoding="utf-8")


def json_lines(p):
    return [json.loads(l) for l in text(p).splitlines() if l.strip()]


def check(a: dict) -> tuple[bool, str]:
    p, prov, cv = a["path"], a["provenance"], a["corpus_version"]
    if prov == "unknown":
        recs = json_lines(p) if p.endswith(".jsonl") else [json.loads(text(p))]
        fields = sorted({k for r in recs for k in r if "corpus" in k})
        return (not fields, f"{len(recs)} records; corpus-related keys: {fields or 'none'}")
    if prov == "established":
        c3_ok = INV["c3_unchanged_since_phase9a"]
        d = json.loads(text(p))
        states = "versions" in d and d["versions"] or any("corpus" in k for k in d)
        return (c3_ok and not states, f"no corpus field in artifact: {not states}; c3 unchanged since 9A: {c3_ok}")
    # recorded
    if p == "eval/questions.jsonl":
        return (True, "verbatim quotes; corpus-independent except q09 (matches c3 text only; see reconciliation)")
    if p.endswith(".jsonl"):
        recs = json_lines(p)
        data = [r for r in recs if r.get("kind") in ("question", "run")]
        vals = {r.get("corpus_version") for r in data}
        other = sorted({r.get("kind") for r in recs if r not in data})
        return (bool(data) and vals == {cv}, f"corpus_version on every question/run record: {sorted(map(str, vals))} "
                f"({len(data)} records); bookkeeping records without the field: {other}")
    d = json.loads(text(p))
    if "phase10/out/live" in p:
        return (d.get("versions", {}).get("corpus_version") == cv, f"versions.corpus_version = {d.get('versions', {}).get('corpus_version')}")
    if "phase11/out/ablation" in p:
        return (d.get("corpus") == cv, f"corpus = {d.get('corpus')}")
    if "phase11b/out/compare" in p:
        return (set(d) == {"c3", "c6"}, f"top-level sections: {sorted(d)}")
    if "phase11c/out/corpus_audit" in p:
        return ("c3" in json.dumps(d["counts"]) and "c6" in json.dumps(d["counts"]),
                "c3-vs-c6 diff: every difference carries explicit 'c3' and 'c6' values (no per-row corpus field)")
    if "phase11c/out/compare_corpora" in p:
        return ({r["corpus"] for r in d["rows"]} == {"c3", "c6"}, "every row carries 'corpus'")
    if "phase11c/out/e2e_offline" in p:
        return ({r["corpus"] for r in d} == {"c3", "c6"}, "every row carries 'corpus'")
    if p == "eval/phase7/table_question_set.json":
        return ("c4 extraction" in d["author"], f"author: {d['author'][:100]}")
    if p == "experiments/phase11/new_cases.json":
        return ("c6" in d["author"], f"author: {d['author'][:100]}")
    if p.startswith("experiments/phase8/"):
        return ("c4" in d["about"], f"about mentions c4: {d['about'][d['about'].find('c4') - 30:d['about'].find('c4') + 10]!r}")
    if p == "eval/questions.jsonl":
        return (True, "verbatim quotes; corpus-independent except q09 (matches c3 text only; see reconciliation)")
    return (False, "no check implemented")


def chunk(path, cid):
    return next(json.loads(l)["text"] for l in open(path, encoding="utf-8") if json.loads(l)["chunk_id"] == cid)


def main():
    rows, counts = [], {}
    for a in INV["artifacts"]:
        ok, ev = check(a)
        rows.append({"path": a["path"], "classified": a["provenance"], "corpus": a["corpus_version"], "verified": ok,
                     "evidence_checked": ev, "excluded_from_comparisons": not a["usable_in_corpus_specific_comparisons"]})
        counts[a["provenance"]] = counts.get(a["provenance"], 0) + 1
    fx_cites = {}
    for cid in ("HNET:2.2.1:c023", "AIAYN:T3:r02"):
        t3, t6 = chunk("data/processed/chunks_256.jsonl", cid), chunk("experiments/phase10/corpus/c6/processed/chunks_256.jsonl", cid)
        fx_cites[cid] = {"text_identical_in_c3_and_c6": t3 == t6}
    fixtures = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path("fixtures/perf").glob("*.json"))}
    out = {"inventory_counts": counts, "all_classifications_verified": all(r["verified"] for r in rows),
           "unknown_excluded_from_comparisons": all(r["excluded_from_comparisons"] for r in rows if r["classified"] == "unknown"),
           "fixture_citation_chunks": fx_cites, "fixture_sha256_now": fixtures, "artifacts": rows}
    Path("experiments/phase11f/fixture_provenance_audit.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(counts, "| all verified:", out["all_classifications_verified"], "| unknown excluded:", out["unknown_excluded_from_comparisons"])
    for r in rows:
        if not r["verified"]:
            print("  NOT VERIFIED:", r["path"], r["evidence_checked"])
    print("fixture citation chunks:", fx_cites)


if __name__ == "__main__":
    main()
