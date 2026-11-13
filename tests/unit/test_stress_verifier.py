import collections
import json
from pathlib import Path

import pytest

from relay.eval import stress_verifier as sv


def claim(entail, contra=0.0, flags=()):
    return {"entail": entail, "contra": contra, "flags": list(flags)}


def row(pid, supported, *claims, split="calibration", cat="c"):
    return {"pair_id": pid, "supported": supported, "claims": list(claims), "split": split, "category": cat,
            "ms_whole": 10.0, "ms_atomic": 10.0 * len(claims)}


def test_policies():
    c = claim(0.3, contra=0.1)
    assert not sv.p1(c, 0.5) and sv.p1(c, 0.3)
    assert sv.p2(c, tc=0.5, tf=0.8)                                # unflagged, not contradicted
    assert not sv.p2(claim(0.9, contra=0.6), 0.5, 0.8)             # contradicted
    assert not sv.p2(claim(0.6, flags=["x"]), 0.5, 0.8)            # flagged, weak support
    assert sv.p2(claim(0.85, flags=["x"]), 0.5, 0.8)


def test_atomic_needs_every_clause_whole_looks_at_the_sentence_only():
    r = row("u", False, claim(0.9), claim(0.9), claim(0.05))  # sentence + supported clause + invented clause
    assert sv.decide(r, sv.p1, "whole", {"t": 0.5})
    assert not sv.decide(r, sv.p1, "atomic", {"t": 0.5})


def test_metrics_and_categories():
    rows = [row("s1", True, claim(0.9), cat="num"), row("s2", True, claim(0.1), cat="num"),
            row("u1", False, claim(0.9), cat="add")]
    m = sv.metrics(rows, [True, False, True])
    assert (m["false_acceptances"], m["false_rejections"], m["accuracy_pct"]) == (1, 1, 33.3)
    assert m["false_acceptance_rate_pct"] == 100.0 and m["false_rejection_rate_pct"] == 50.0
    assert m["by_category"]["num"] == {"supported": 2, "false_rejections": 1, "unsupported": 0, "false_acceptances": 0}


def test_calibration_weights_false_acceptance_double_and_prefers_strict_on_ties():
    rows = [row("s", True, claim(0.7)), row("u", False, claim(0.4))]
    assert sv.calibrate(rows, sv.p1, "whole") == {"t": 0.7}  # every t in (0.4, 0.7] costs 0; the strictest wins
    p = sv.calibrate([row("s", True, claim(0.7, 0.05)), row("u", False, claim(0.1, 0.3))], sv.p2, "whole")
    assert p["tc"] <= 0.3  # the contradiction threshold must catch u


def test_evaluate_uses_only_calibration_rows_for_thresholds():
    rows = [row("s", True, claim(0.7)), row("u", False, claim(0.4)),
            row("hs", True, claim(0.2), split="heldout"), row("hu", False, claim(0.95), split="heldout")]
    out = sv.evaluate(rows)
    assert out["P1-whole"]["params"] == {"t": 0.7}
    assert out["P1-whole"]["heldout_decisions"] == {"hs": False, "hu": True}  # held-out never moves thresholds
    assert set(out) >= {"P1-whole", "P2-whole", "P1-atomic", "P2-atomic", "latency_ms_per_answer_est_atomic"}


def test_report_is_never_overwritten(tmp_path):
    report = tmp_path / "r.json"
    report.write_text("{}")
    with pytest.raises(FileExistsError):
        sv.main(models=(), report=report)


WANT = {"exact": (3, 2), "semantic_paraphrase": (4, 2), "technical_paraphrase": (3, 2), "implicit_entity": (3, 2),
        "numerical": (2, 4), "attribution": (2, 3), "experimental_added": (3, 5)}


@pytest.mark.parametrize("path", ["eval/verifier_stress.jsonl", "eval/verifier_stress_v2.jsonl"])
def test_stress_set_shape(path):
    pairs = sv.load_pairs(Path(path))
    assert len(pairs) == 40 and len({p["pair_id"] for p in pairs}) == 40
    assert collections.Counter(p["supported"] for p in pairs) == {True: 20, False: 20}
    split = collections.Counter((p["split"], p["supported"]) for p in pairs)
    assert split[("calibration", True)] + split[("calibration", False)] == 25
    assert abs(split[("heldout", True)] - split[("heldout", False)]) <= 1
    got = collections.Counter((p["category"], p["supported"]) for p in pairs)
    assert {c: (got[(c, True)], got[(c, False)]) for c in WANT} == WANT


def test_round_two_is_fresh():
    old = {p["claim"] for p in sv.load_pairs(Path("eval/verifier_stress.jsonl"))}
    assert not old & {p["claim"] for p in sv.load_pairs(Path("eval/verifier_stress_v2.jsonl"))}


@pytest.mark.parametrize("path", ["eval/verifier_stress.jsonl", "eval/verifier_stress_v2.jsonl"])
def test_every_evidence_chunk_exists(path):
    chunks = Path("data/processed/chunks_256.jsonl")
    if not chunks.exists():
        pytest.skip("corpus not built")
    ids = {json.loads(l)["chunk_id"] for l in chunks.read_text(encoding="utf-8").splitlines()}
    for p in sv.load_pairs(Path(path)):
        assert set(p["evidence"]) <= ids, p["pair_id"]
