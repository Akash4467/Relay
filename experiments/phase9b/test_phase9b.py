"""Phase 9B unit tests: scoring, the freeze gate, the review package and the Phase 8 loader. No models are loaded."""
import ast
import csv
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import build_review_package as brp  # noqa: E402
import evaluate_verifiers as ev  # noqa: E402

REVIEW = Path("eval/phase9b_human_review")


def row(i, label, a=False, c=None, d_from="deterministic", d=False, cat="x"):
    return {"id": i, "label": label, "category": cat, "A": {"accepted": a}, "B1": {"accepted": a}, "B2": {"accepted": a},
            "C": {"accepted": bool(c), "abstained": c is None}, "D": {"accepted": d, "from": d_from}}


def test_score_counts_abstention_separately():
    rows = [row("s1", "supported", c=True), row("s2", "supported", c=None), row("s3", "supported", c=False),
            row("u1", "unsupported", c=True), row("u2", "unsupported", c=None), row("a1", "ambiguous", c=None),
            row("a2", "ambiguous", c=True)]
    s = ev.score(rows, "C")
    assert (s["TA"], s["FR"], s["FA"], s["TR"]) == (1, 1, 1, 0)
    assert (s["abstain_supported"], s["abstain_unsupported"]) == (1, 1)
    assert s["FAR"] == 0.5 and s["FRR"] == round(1 / 3, 3) and s["FRR_incl_abstain"] == round(2 / 3, 3)
    assert s["coverage"] == round(1 / 3, 3) and s["abstention_rate"] == round(3 / 7, 3)
    assert (s["ambiguous_safe_abstain"], s["ambiguous_unsafe_accept"]) == (1, 1)


def test_score_without_unsupported_cases_has_no_rate():
    assert ev.score([row("s1", "supported", a=True)], "A")["FAR"] is None


def test_c2g_outcomes():
    rows = [row("ok", "supported", d=True), row("bad", "unsupported", d=True),
            row("safe", "unsupported", d_from="rejected: unresolved mark claim"),
            row("lost", "supported", d_from="rejected: unresolved mark claim"), row("fb", "supported", d_from="v4", d=True)]
    o = ev.c2g_outcomes(rows)
    assert o["resolved_correct"] == ["ok"] and o["resolved_wrong"] == ["bad"]
    assert o["rejected_safely_unresolved"] == ["safe"] and o["rejected_wrongly_unresolved"] == ["lost"]
    assert o["fallback_v4"] == ["fb"] and o["unresolved"] == ["safe", "lost", "fb"]


def test_provisional_mode_refuses_heldout():
    with pytest.raises(SystemExit):
        ev.provisional_cases("heldout")
    assert len(ev.provisional_cases("development")) == 31


def test_real_manifest_is_not_frozen():
    with pytest.raises(SystemExit, match="not frozen"):
        ev.frozen_cases("heldout")



def test_c2g_outcomes_lists_ambiguous_apart():
    rows = [row("amb", "ambiguous", d=False), row("ok", "unsupported", d=False)]
    o = ev.c2g_outcomes(rows)
    assert o["resolved_on_ambiguous"] == ["amb"] and o["resolved_correct"] == ["ok"] and o["resolved_wrong"] == []


# ---- two-level review protocol: a throwaway review folder per test
TEMPLATES = REVIEW / "reviewer"


def fill_agree(r, rid="rev1"):
    """A review that agrees with the original label at both levels."""
    o = r["case_id"]
    lab = ev.originals()["development" if o.startswith("s") else "heldout"][o]["label"].upper()
    r |= {"table_only_outcome": lab, "whole_paper_outcome": lab, "supporting_evidence": "cell as printed",
          "source_page_or_section": f"p{r['pdf_page']}", "legend_or_definition_needed": "no", "confidence": "high",
          "explanation": "the printed cell decides it", "reviewer_id": rid, "review_date": "2026-11-20"}


INDEPENDENT = {"independent": True, "human": True, "model_assisted": False, "owner": False,
               "authored_cases_or_labels": False, "signed_declaration_on_file": True}


def attest(review, manifest, rid, **flags):
    """Record review_provenance for rid (independent human review unless flags say otherwise), bound to file hashes."""
    m = json.loads(manifest.read_text(encoding="utf-8"))
    files = {s: {"sha256": ev.sha256(review / "reviews" / rid / f"{s}.csv")} for s in ("development", "heldout")}
    m.setdefault("review_provenance", {})[rid] = INDEPENDENT | flags | {"files": files}
    manifest.write_text(json.dumps(m), encoding="utf-8")


def make_review(tmp_path, reviewers, excluded=(), attested=True):
    """reviewers: {id: fill(row, id)}; each is attested as independent human review unless attested=False.
    Returns (review_dir, bench_dir, manifest_path)."""
    review = tmp_path / "review"
    (review / "reviews").mkdir(parents=True)
    (review / "disagreement_log.csv").write_text(",".join(ev.LOG_FIELDS) + "\n", encoding="utf-8")
    for rid, fill in reviewers.items():
        d = review / "reviews" / rid
        d.mkdir()
        for split in ("development", "heldout"):
            with (TEMPLATES / f"template_{split}.csv").open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            for r in rows:
                fill(r, rid)
            with (d / f"{split}.csv").open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    manifest = tmp_path / "benchmark_manifest.json"
    manifest.write_text(json.dumps({"status": "awaiting_independent_review", "min_reviewers": 1,
                                    "excluded_reviewer_ids": list(excluded)}), encoding="utf-8")
    if attested:
        for rid in reviewers:
            attest(review, manifest, rid)
    return review, tmp_path / "bench", manifest


def review_rows(split="development"):
    with (TEMPLATES / f"template_{split}.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        fill_agree(r)
    return rows


def errors(rows, split="development", rid="rev1", manifest=None):
    return ev.validate_review(rows, split, rid, manifest or {})


def test_a_complete_review_is_valid():
    assert errors(review_rows()) == [] and errors(review_rows("heldout"), "heldout") == []


def test_incomplete_review_is_rejected():
    rows = review_rows()
    rows[3]["confidence"] = ""
    assert any("incomplete review" in e for e in errors(rows))


@pytest.mark.parametrize("rid", ["assistant", "Claude", "claude-opus", "self", "developer"])
def test_self_review_is_rejected(rid):
    rows = review_rows()
    for r in rows:
        r["reviewer_id"] = rid
    assert any("excluded" in e for e in errors(rows, rid=rid))


def test_manifest_can_exclude_the_case_author():
    assert any("excluded" in e for e in errors(review_rows(), manifest={"excluded_reviewer_ids": ["rev1"]}))


def test_splits_stay_separate_and_claims_unedited():
    dev, held = review_rows(), review_rows("heldout")
    assert any("exactly once" in e for e in errors(dev + held[:1]))
    dev[0]["claim"] += " (reworded)"
    assert any("claim text was edited" in e for e in errors(dev))


def test_outcome_levels_are_enforced():
    rows = review_rows()
    rows[0]["table_only_outcome"] = "SUPPORTED_ELSEWHERE"
    rows[1] |= {"table_only_outcome": "UNSUPPORTED", "whole_paper_outcome": "SUPPORTED"}
    rows[2] |= {"table_only_outcome": "SUPPORTED", "whole_paper_outcome": "SUPPORTED_ELSEWHERE"}
    rows[3]["reviewer_id"] = "someone-else"
    e = errors(rows)
    assert any("table_only_outcome must be" in x for x in e) and any("use SUPPORTED_ELSEWHERE" in x for x in e)
    assert any("contradicts" in x for x in e) and any("folder name" in x for x in e)


def test_freeze_refuses_without_reviews(tmp_path):
    review, bench, manifest = make_review(tmp_path, {})
    with pytest.raises(SystemExit, match="at least 1 required"):
        ev.freeze(review, bench, manifest)


def test_freeze_refuses_self_review_folder(tmp_path):
    review, bench, manifest = make_review(tmp_path, {"assistant": fill_agree})
    with pytest.raises(SystemExit, match="excluded"):
        ev.freeze(review, bench, manifest)
    assert not bench.exists()


def test_agreeing_review_freezes_and_is_tamper_checked(tmp_path):
    review, bench, manifest = make_review(tmp_path, {"rev1": fill_agree})
    assert ev.write_disagreement_log(review, manifest) == []
    m = ev.freeze(review, bench, manifest)
    assert m["status"] == "frozen" and m["reviewers"] == ["rev1"] and len(m["benchmark_files"]["heldout"]["sha256"]) == 64
    held = ev.frozen_cases("heldout", manifest)
    assert {c["id"] for c in held} == set(ev.originals()["heldout"])
    assert all(c["label"] == c["label_phase8"] and c["label_paper"] == c["label"] for c in held)
    with pytest.raises(SystemExit, match="already frozen"):
        ev.freeze(review, bench, manifest)
    path = bench / "development_v1.json"
    path.write_text(path.read_text(encoding="utf-8").replace('"supported"', '"unsupported"', 1), encoding="utf-8")
    with pytest.raises(SystemExit, match="frozen hash"):
        ev.frozen_cases("development", manifest)


def test_new_version_keeps_the_previous_one(tmp_path):
    review, bench, manifest = make_review(tmp_path, {"rev1": fill_agree})
    ev.freeze(review, bench, manifest)
    m = ev.freeze(review, bench, manifest, version="v2")
    assert m["version"] == "v2" and m["previous_versions"][0]["version"] == "v1"
    assert (bench / "development_v1.json").exists() and (bench / "development_v2.json").exists()


def test_frozen_cases_rejects_a_file_from_the_other_split(tmp_path):
    review, bench, manifest = make_review(tmp_path, {"rev1": fill_agree})
    m = ev.freeze(review, bench, manifest)
    m["benchmark_files"]["heldout"] = m["benchmark_files"]["development"]
    manifest.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(SystemExit, match="exactly the heldout cases"):
        ev.frozen_cases("heldout", manifest)


def h15_differs(outcome_table, outcome_paper):
    def fill(r, rid):
        fill_agree(r, rid)
        if r["case_id"] == "h15":
            r |= {"table_only_outcome": outcome_table, "whole_paper_outcome": outcome_paper, "explanation": "prose"}
    return fill


def test_disagreements_need_adjudication_no_majority_vote(tmp_path):
    review, bench, manifest = make_review(tmp_path, {"rev1": fill_agree, "rev2": fill_agree,
                                                     "rev3": h15_differs("AMBIGUOUS", "SUPPORTED_ELSEWHERE")})
    rows = ev.write_disagreement_log(review, manifest)
    kinds = sorted((r["case_id"], r["level"], r["kind"]) for r in rows)
    assert kinds == [("h15", "table_only", "between_reviewers"), ("h15", "table_only", "reviewer_vs_original"),
                     ("h15", "whole_paper", "between_reviewers")]
    assert all(r["working_outcome"] == "AMBIGUOUS" and r["status"] == "unresolved" for r in rows)
    assert all("rev3" in r["evidence"] and "rev1" in r["evidence"] for r in rows)
    with pytest.raises(SystemExit, match="not adjudicated"):  # 2 of 3 agree: still no automatic choice
        ev.freeze(review, bench, manifest)
    adjudicate(review, {"table_only": "AMBIGUOUS", "whole_paper": "SUPPORTED_ELSEWHERE"}, "adj1")
    ev.freeze(review, bench, manifest)
    h15 = next(c for c in ev.frozen_cases("heldout", manifest) if c["id"] == "h15")
    assert (h15["table_only_outcome"], h15["whole_paper_outcome"]) == ("AMBIGUOUS", "SUPPORTED_ELSEWHERE")
    assert (h15["label"], h15["label_paper"], h15["label_phase8"]) == ("ambiguous", "supported", "unsupported")
    assert set(h15["reviews"]) == {"rev1", "rev2", "rev3"}


def adjudicate(review, outcomes, who):
    path = review / "disagreement_log.csv"
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r |= {"adjudicated_outcome": outcomes[r["level"]], "adjudicator_id": who, "adjudication_date": "2026-11-21",
              "adjudication_reason": "discussed", "status": "adjudicated"}
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ev.LOG_FIELDS)
        w.writeheader()
        w.writerows(rows)


def test_excluded_adjudicator_is_refused(tmp_path):
    review, bench, manifest = make_review(tmp_path, {"rev1": h15_differs("SUPPORTED", "SUPPORTED")})
    ev.write_disagreement_log(review, manifest)
    adjudicate(review, {"table_only": "SUPPORTED", "whole_paper": "SUPPORTED"}, "claude")
    with pytest.raises(SystemExit, match="adjudicator 'claude' is excluded"):
        ev.freeze(review, bench, manifest)


def test_changing_reviews_after_adjudication_is_refused(tmp_path):
    review, bench, manifest = make_review(tmp_path, {"rev1": h15_differs("AMBIGUOUS", "AMBIGUOUS")})
    ev.write_disagreement_log(review, manifest)
    adjudicate(review, {"table_only": "AMBIGUOUS", "whole_paper": "AMBIGUOUS"}, "adj1")
    p = review / "reviews" / "rev1" / "heldout.csv"
    p.write_text(p.read_text(encoding="utf-8").replace("AMBIGUOUS,AMBIGUOUS", "SUPPORTED,SUPPORTED", 1), encoding="utf-8")
    with pytest.raises(SystemExit, match="reviews changed after adjudication"):
        ev.write_disagreement_log(review, manifest)
    attest(review, manifest, "rev1")  # re-attest the edited files, so the stale-log check is what refuses
    with pytest.raises(SystemExit, match="stale"):
        ev.freeze(review, bench, manifest)


def test_load_phase8_keeps_original_checker_source():
    tree = ast.parse(ev.P8_SCRIPT.read_text(encoding="utf-8"))
    original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "deterministic_v2")
    ns = ev.load_phase8({})
    assert set(ev.P8_NAMES) <= set(ns)
    assert ns["deterministic_v2"].__code__.co_firstlineno == original.lineno


# ---- review package
def test_latex_cell_follows_multirow_and_empty_cells():
    rows = [(10, ["A", "\\multirow{2}{*}{\\cmark}", "x"]), (11, ["", "", "y"]), (12, ["", "", "z"])]
    assert brp.latex_cell(rows, 1, 1) == ("\\multirow{2}{*}{\\cmark}", 10, "inherited from the \\multirow at line 10")
    assert brp.latex_cell(rows, 2, 1) == ("", 12, "empty LaTeX cell")
    assert brp.latex_cell(rows, 0, 2) == ("x", 10, "own cell")


def test_split_cells_drops_row_end_and_comments():
    assert brp.split_cells(r" & \checkmark &  & 0.8 & 98.40\hphantom{0}   \\  %") == \
        ["", "\\checkmark", "", "0.8", "98.40\\hphantom{0}"]


def test_package_keeps_labels_and_leaves_review_columns_empty():
    p8 = {s: {c["id"]: c for c in json.loads(p.read_text(encoding="utf-8"))["cases"]} for s, p in ev.P8_CASES.items()}
    for split, n in (("development", 31), ("heldout", 16)):
        with (REVIEW / f"{split}_review.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == n and {r["case_id"] for r in rows} == set(p8[split])
        for r in rows:
            assert r["current_label"] == p8[split][r["case_id"]]["label"]
            assert r["claim"] == p8[split][r["case_id"]]["claim"]
            assert all(r[k] == "" for k in brp.REVIEW_COLUMNS)
            assert r["pdf_page"] and r["cell_latex"] and r["column"]


def test_package_macros_match_paper_sources():
    assert "\\newcommand{\\xmark}{}" in Path("data/raw/MOSHI/src/main.tex").read_text(encoding="utf-8")
    assert "\\newcommand{\\xmark}{\\ding{55}}" in Path("data/raw/HNET/src/main.tex").read_text(encoding="utf-8")



def test_original_case_files_are_unchanged():
    m = json.loads(ev.MANIFEST.read_text(encoding="utf-8"))
    for f in ("eval/phase9b_human_review/development_review.csv", "eval/phase9b_human_review/heldout_review.csv",
              "experiments/phase8/table_semantics_cases.json", "experiments/phase8/table_semantics_heldout.json"):
        assert ev.sha256(Path(f)) == m["pre_review_sha256_v1"][f], f


def test_reviewer_templates_are_blind():
    for split, n in (("development", 31), ("heldout", 16)):
        text = (TEMPLATES / f"template_{split}.csv").read_text(encoding="utf-8")
        rows = list(csv.DictReader(text.splitlines()))
        assert len(rows) == n and list(rows[0]) == brp.REVIEWER_EVIDENCE + brp.REVIEWER_INPUT
        assert not set(brp.HIDDEN) & set(rows[0])
        assert all(r[k] == "" for r in rows for k in brp.REVIEWER_INPUT)
        assert "unsupported" not in text.lower() and "phase 8" not in text.lower()
        orig = ev.originals()[split]
        assert all(r["claim"] == orig[r["case_id"]]["claim"] for r in rows)
    assert "unsupported" not in (TEMPLATES / "table_evidence.md").read_text(encoding="utf-8").lower()


def test_printed_on_page_follows_each_papers_macros():
    assert brp.printed_on_page("MOSHI", r"\xmark", "(blank)").startswith("nothing (blank)")
    assert brp.printed_on_page("HNET", r"\multirow{2}{*}{\xmark}", "✗") == "a cross (✗)"
    assert brp.printed_on_page("MOSHI", "(empty)", "(empty)") == "nothing (blank): empty LaTeX cell"
    assert brp.printed_on_page("MOSHI", r"\checkmark", "✓") == "a check (✓)"


def test_write_new_never_changes_an_existing_file(tmp_path):
    p = tmp_path / "f.csv"
    brp.write_new(p, "a\r\n")
    brp.write_new(p, "a\r\n")
    with pytest.raises(SystemExit):
        brp.write_new(p, "b\r\n")
    assert p.read_bytes() == b"a\r\n"


# ---- model-assisted audit exclusion
AUDIT_ID = "chatgpt_model_assisted_audit_NOT_INDEPENDENT"


def with_id(rows, rid):
    for r in rows:
        r["reviewer_id"] = rid
    return rows


@pytest.mark.parametrize("split", ["development", "heldout"])
def test_model_audit_id_is_rejected(split):
    assert any("excluded" in e for e in errors(with_id(review_rows(split), AUDIT_ID), split, rid=AUDIT_ID))


@pytest.mark.parametrize("split", ["development", "heldout"])
@pytest.mark.parametrize("folder", ["Akash", "reviewer-jdoe", "x"])
def test_renaming_the_folder_does_not_bypass_the_exclusion(split, folder):
    e = errors(with_id(review_rows(split), AUDIT_ID), split, rid=folder)
    assert any(f"reviewer '{AUDIT_ID}' in the file is excluded" in x for x in e)


@pytest.mark.parametrize("variant", ["ChatGPT-review", "gpt_model-assisted", "audit_not_independent"])
def test_variants_of_model_audit_ids_are_rejected(variant):
    assert ev.excluded(variant, {})


@pytest.mark.parametrize("split", ["development", "heldout"])
def test_the_stored_model_audit_files_are_rejected_wherever_they_go(split):
    with (Path("experiments/phase9b/model_audit") / f"{split}.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))  # read only; the audit files are never modified
    for folder in (AUDIT_ID, "Akash", "renamed"):
        assert any("excluded" in e for e in ev.validate_review(rows, split, folder, {}))


def test_model_audit_copied_into_reviews_cannot_be_frozen(tmp_path):
    def fill(r, rid):
        fill_agree(r, rid)
        r["reviewer_id"] = AUDIT_ID
    review, bench, manifest = make_review(tmp_path, {"renamed-folder": fill})
    with pytest.raises(SystemExit, match="excluded"):
        ev.freeze(review, bench, manifest)
    assert not bench.exists()


def test_a_legitimate_reviewer_id_passes_the_independence_check():
    for rid in ("rev1", "reviewer-jdoe", "Akash"):
        assert not ev.excluded(rid, {})
        assert not [e for e in errors(with_id(review_rows(), rid), rid=rid) if "excluded" in e]


# ---- independent human review is required to freeze
def owner_model_assisted(review, manifest, rid):
    attest(review, manifest, rid, independent=False, model_assisted=True, owner=True)


def test_owner_model_assisted_review_alone_cannot_freeze(tmp_path):
    review, bench, manifest = make_review(tmp_path, {"Akash": fill_agree}, attested=False)
    owner_model_assisted(review, manifest, "Akash")
    with pytest.raises(SystemExit, match="genuinely independent human reviewer is required"):
        ev.freeze(review, bench, manifest)
    assert not bench.exists()


def test_the_real_akash_provenance_does_not_qualify(tmp_path):
    real = json.loads(ev.MANIFEST.read_text(encoding="utf-8"))["review_provenance"]["Akash"]
    review, bench, manifest = make_review(tmp_path, {"Akash": fill_agree}, attested=False)
    m = json.loads(manifest.read_text(encoding="utf-8"))
    m["review_provenance"] = {"Akash": real | {"files": {s: {"sha256": ev.sha256(review / "reviews" / "Akash" / f"{s}.csv")}
                                                       for s in ("development", "heldout")}}}
    manifest.write_text(json.dumps(m), encoding="utf-8")
    ok, why = ev.independent_reviewers(review, ev.load_reviews(review, m), m)
    assert ok == [] and "does not establish independent human review" in why["Akash"]


@pytest.mark.parametrize("flags", [{"authored_cases_or_labels": True}, {"model_assisted": True}, {"owner": True},
                                   {"human": False}, {"independent": "yes"}])
def test_any_disqualifying_provenance_flag_blocks(tmp_path, flags):
    review, bench, manifest = make_review(tmp_path, {"rev1": fill_agree}, attested=False)
    attest(review, manifest, "rev1", **flags)
    with pytest.raises(SystemExit, match="genuinely independent"):
        ev.freeze(review, bench, manifest)


def test_no_provenance_does_not_qualify(tmp_path):
    review, bench, manifest = make_review(tmp_path, {"rev1": fill_agree}, attested=False)
    with pytest.raises(SystemExit, match="no review_provenance entry"):
        ev.freeze(review, bench, manifest)


def test_independent_reviewer_satisfies_the_gate_alongside_the_owner(tmp_path):
    review, bench, manifest = make_review(tmp_path, {"Akash": fill_agree, "rev1": fill_agree}, attested=False)
    owner_model_assisted(review, manifest, "Akash")
    attest(review, manifest, "rev1")
    m = ev.freeze(review, bench, manifest)
    assert m["status"] == "frozen" and m["independent_reviewers"] == ["rev1"] and m["reviewers"] == ["Akash", "rev1"]


def test_attestation_is_bound_to_the_submitted_files(tmp_path):
    review, bench, manifest = make_review(tmp_path, {"rev1": fill_agree})
    p = review / "reviews" / "rev1" / "development.csv"
    p.write_text(p.read_text(encoding="utf-8").replace("cell as printed", "cell as printed!", 1), encoding="utf-8")
    with pytest.raises(SystemExit, match="do not match the submitted files"):
        ev.freeze(review, bench, manifest)


def test_min_independent_reviewers_cannot_be_lowered_below_one(tmp_path):
    review, bench, manifest = make_review(tmp_path, {"rev1": fill_agree}, attested=False)
    m = json.loads(manifest.read_text(encoding="utf-8"))
    manifest.write_text(json.dumps(m | {"min_independent_reviewers": 0}), encoding="utf-8")
    with pytest.raises(SystemExit, match="genuinely independent"):
        ev.freeze(review, bench, manifest)


def test_model_audit_relabelled_under_a_new_folder_and_id_cannot_qualify(tmp_path):
    """The stored audit copied in under a clean id, with a forged independent attestation, is still refused."""
    review, bench, manifest = make_review(tmp_path, {}, attested=False)
    d = review / "reviews" / "indep-reviewer"
    d.mkdir()
    for split in ("development", "heldout"):
        with (Path("experiments/phase9b/model_audit") / f"{split}.csv").open(encoding="utf-8", newline="") as f:
            rows = with_id(list(csv.DictReader(f)), "indep-reviewer")  # read only; audit files untouched
        with (d / f"{split}.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    attest(review, manifest, "indep-reviewer")
    m = json.loads(manifest.read_text(encoding="utf-8"))
    ok, why = ev.independent_reviewers(review, ev.load_reviews(review, m), m)
    assert ok == [] and why["indep-reviewer"] == "submission is a copy of the model-assisted audit"
    with pytest.raises(SystemExit, match="genuinely independent"):
        ev.freeze(review, bench, manifest)


def test_excluded_ids_cannot_qualify_even_with_an_independent_attestation(tmp_path):
    def fill(r, rid):
        fill_agree(r, rid)
        r["reviewer_id"] = AUDIT_ID
    review, bench, manifest = make_review(tmp_path, {"renamed": fill})
    with pytest.raises(SystemExit, match="excluded"):
        ev.freeze(review, bench, manifest)


def test_independent_reviewer_needs_their_own_signed_declaration(tmp_path):
    review, bench, manifest = make_review(tmp_path, {"rev1": fill_agree}, attested=False)
    attest(review, manifest, "rev1", signed_declaration_on_file=False)
    with pytest.raises(SystemExit, match="signed independence declaration is not on file"):
        ev.freeze(review, bench, manifest)
    assert not bench.exists()


def test_the_real_rahul_entry_waits_for_his_signed_declaration():
    m = json.loads(ev.MANIFEST.read_text(encoding="utf-8"))
    ok, why = ev.independent_reviewers(ev.REVIEW, ev.load_reviews(ev.REVIEW, m), m)
    assert ok == [] and why == {"rahul": "the reviewer's own signed independence declaration is not on file"}
