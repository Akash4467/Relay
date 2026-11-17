"""Phase 11C tests. REAL-PAPER OFFLINE: corpus integrity (c3 vs c6, read only) and the MOSHI cases on the stored c6
corpus. RESULT CHECKS: the latest comparison / mocked end-to-end outputs keep their key guarantees."""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, "experiments/phase11c")
import corpus_audit  # noqa: E402

C6 = Path("experiments/phase10/corpus/c6/processed")


def latest(pattern):
    return json.loads(Path(sorted(glob.glob(f"experiments/phase11c/out/{pattern}"))[-1]).read_text(encoding="utf-8"))


def test_corpus_audit_passes_with_no_unexplained_difference():
    r = corpus_audit.audit()
    assert r["passed"] and r["unexplained"] == []
    assert all(v["equal_to_c4_manifest"] and v["c6_src_tree_equals_c4"] for v in r["sources"].values())
    assert r["counts"]["paragraphs"]["c3"] == r["counts"]["paragraphs"]["c6"] == 363
    for size in (256, 512):
        c = r["counts"][f"chunks_{size}"]
        assert c["c3_ids_missing_in_c6"] == [] and c["duplicate_ids"] == [0, 0] and c["prose_ids_identical"]


def test_moshi_units_in_stored_c6():
    text = " ".join(json.loads(l)["text"] for l in (C6 / "chunks_256.jsonl").open(encoding="utf-8"))
    for s in ("160 ms—lower than the 230 ms average", "5 min", "12 s"):
        assert s in text, s


def test_moshi_table4_causal_marks_stay_on_their_rows():
    t = next(t for t in map(json.loads, (C6 / "tables.jsonl").open(encoding="utf-8")) if t["paper"] == "MOSHI" and t["table_id"] == "T4")
    by = {r["row_header"]: r["cells"] for r in t["rows"]}
    assert all(by[r]["causal"] == "✓" for r in ("Mimi, adv. loss only", "Same, downsampled at 16kHz", "Mimi, non adv. only"))
    assert all(by[r]["causal"] == "(blank)" for r in ("RVQGAN", "SemantiCodec"))
    assert by["Mimi, adv. loss only"]["MUSHRA (↑)"] == "81.0±1.3"
    rows = [r for c in map(json.loads, (C6 / "chunks_256.jsonl").open(encoding="utf-8")) if c.get("table_id") == "T4"
            and c["paper"] == "MOSHI" for r in c["rows"]]
    mimi = next(r for r in rows if r["configuration"] == "Mimi, adv. loss only")
    assert {v["column"]: v["value"] for v in mimi["values"]}["causal"] == "✓"


def test_no_question_loses_complete_evidence_from_c3_to_c6():
    d = latest("compare_corpora_*.json")
    by = {(r["corpus"], r["dataset"], r["question_id"]): r for r in d["rows"]}
    lost = [k for k, r in by.items() if k[0] == "c3" and r.get("all_gold") and not by[("c6",) + k[1:]].get("all_gold")]
    assert lost == []
    assert d["changes_c3_to_c6"]["unanswerable_table_in_context_changed"] == []


def test_q11_is_reported_as_not_fully_fixed_on_both_corpora():
    d = latest("compare_corpora_*.json")
    for c in ("c3", "c6"):
        assert d["q11"][c]["AIAYN:T3:r02"]["in_final_context"] and not d["q11"][c]["AIAYN:T3:r01"]["in_final_context"]


def test_mocked_end_to_end_guarantees_hold_on_both_corpora():
    r = latest("e2e_offline_*.json")
    assert len(r) == 30 and all(x["passed"] for x in r)
    refused = {"q18_refusal", "q20_refusal", "q30_refusal", "q20_fabricated", "q30_fabricated", "q12_false_absence",
               "invalid_citation", "unretrieved_citation"}
    assert all(x["outcome"] == "refused" for x in r if x["scenario"] in refused)
    assert all(x["checks"]["verifier_chunks_complete_and_unmodified"] for x in r)
