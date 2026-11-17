"""Phase 11E tests. PROVENANCE: the fixture inventory and its comparison guard. CONFIG: explicit c3/c6 selection
without touching config/app.yaml, G + v4.2 + p4, rollback. RESULT CHECKS: the recorded MOCKED regression outputs
({c3,c6}_regression.json) keep their guarantees (they are scripted-provider runs, not model performance)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "experiments/phase11e")
import fixture_inventory as FI  # noqa: E402
from relay import corpus  # noqa: E402
from relay.config import app_config, piece_sizes  # noqa: E402
from relay.llm import prompts  # noqa: E402

HERE = Path("experiments/phase11e")


@pytest.fixture(scope="module")
def inv():
    return FI.build()


# ---- fixture provenance
def test_every_artifact_has_an_explicit_provenance(inv):
    assert inv["artifacts"] and all(a["provenance"] in ("recorded", "established", "unknown") for a in inv["artifacts"])
    assert all(a["corpus_version"] for a in inv["artifacts"])


def test_stand_in_fixtures_are_unknown_and_excluded(inv):
    fx = [a for a in inv["artifacts"] if a["path"].startswith("fixtures/perf/")]
    assert len(fx) == 3 and all(a["corpus_version"] == "unknown" and not a["usable_in_corpus_specific_comparisons"] for a in fx)


def test_no_fixture_was_relabelled_or_created_for_c6(inv):
    assert inv["new_c6_fixtures"] == []
    assert not any(a["corpus_version"] == "c6" and a["path"].startswith("fixtures/") for a in inv["artifacts"])
    assert subprocess.run(["git", "diff", "--quiet", "--", "fixtures"]).returncode == 0  # fixtures unchanged vs HEAD


def test_comparison_guard_rejects_unknown_and_mismatched_provenance(inv):
    unknown = [a for a in inv["artifacts"] if a["provenance"] == "unknown"]
    with pytest.raises(FI.ProvenanceError, match="unknown"):
        FI.select_for_comparison(unknown[:1], "c3")
    c4 = [a for a in inv["artifacts"] if a["corpus_version"] == "c4" and a["usable_in_corpus_specific_comparisons"]]
    with pytest.raises(FI.ProvenanceError, match="not 'c6'"):
        FI.select_for_comparison(c4[:1], "c6")
    recorded_c3 = [a for a in inv["artifacts"] if a["corpus_version"] == "c3" and a["provenance"] == "recorded"]
    assert FI.select_for_comparison(recorded_c3, "c3") == recorded_c3


def test_established_c3_provenance_rests_on_unchanged_c3(inv):
    assert inv["c3_unchanged_since_phase9a"] is True
    assert all(a["corpus_version"] == "c3" for a in inv["artifacts"] if a["provenance"] == "established")


# ---- configuration: explicit selection, defaults, rollback
def test_disk_config_stays_c3_with_g_v42_p4():
    cfg = app_config()
    assert cfg["corpus"]["active"] == cfg["corpus_version"] == "c3"
    assert cfg["retrieval_policy"]["symbolic_query_expansion"] and cfg["retrieval_policy"]["prompt_caption_sharing"]
    assert cfg["verification_policy_version"] == "v4.2" and cfg["verification"]["entail_min"] == 0.9
    assert cfg["prompt_version"] == prompts.PROMPT_VERSION == "p4"


def test_explicit_selection_and_rollback_without_touching_the_file():
    import regression as R
    six, three = R.config_for("c6"), R.config_for("c3")
    assert corpus.selected(six).name == "c6" and corpus.validate(corpus.selected(six), piece_sizes())["corpus"] == "c6"
    assert corpus.selected(three).name == "c3" and corpus.validate(corpus.selected(three), piece_sizes())["corpus"] == "c3"
    assert app_config()["corpus"]["active"] == "c3"  # in-memory selection only


# ---- recorded mocked regression outputs
@pytest.mark.parametrize("name", ["c3", "c6"])
def test_mocked_regression_passed_for_each_corpus(name):
    r = json.loads((HERE / f"{name}_regression.json").read_text(encoding="utf-8"))
    assert r["corpus"] == name and r["passed"] and r["config_on_disk_active"] == "c3"
    assert all(r["startup"]["checks"].values())
    assert r["phase10_suite"]["passed"] == r["phase10_suite"]["scenarios"] == 31
    assert r["e2e"]["passed"] == r["e2e"]["scenarios"] == 15
    assert r["e2e"]["corpus_identity"]["q09 evidence contains '230 ms'"] is (name == "c6")
    assert r["fail_closed"]["passed"] and r["fail_closed"]["no_fallback_to_c3"]
    assert all(row["checks"]["verifier_evidence_is_selected_corpus_text"] and row["checks"]["original_question_in_prompt"]
               for row in r["e2e"]["rows"])
    assert all(row["checks"]["trace_corpus_version"] for row in r["e2e"]["rows"])


@pytest.mark.parametrize("name", ["c3", "c6"])
def test_unsupported_and_invalid_citations_refused_on_each_corpus(name):
    rows = {r["scenario"]: r for r in json.loads((HERE / f"{name}_regression.json").read_text(encoding="utf-8"))["e2e"]["rows"]}
    for s in ("q18_refusal", "q20_refusal", "q30_refusal", "q20_fabricated", "q30_fabricated", "q12_false_absence",
              "invalid_citation", "unretrieved_citation"):
        assert rows[s]["outcome"] == "refused", s
