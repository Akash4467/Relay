import collections
import json
from pathlib import Path

import pytest

from relay.eval import stress_verifier as sv


def row(pid, supported, entail, contra=0.0, flags=(), split="calibration", cat="c"):
    return {"pair_id": pid, "supported": supported, "entail": entail, "contra": contra, "flags": list(flags),
            "split": split, "category": cat, "ms": 10.0}


def test_policies():
    r = row("a", True, entail=0.3, contra=0.1)
    assert not sv.p1(r, 0.5) and sv.p1(r, 0.3)
    assert sv.p2(r, tc=0.5, tf=0.8)                                  # unflagged, not contradicted
    assert not sv.p2(row("b", True, 0.9, contra=0.6), 0.5, 0.8)      # contradicted
    assert not sv.p2(row("c", True, 0.6, flags=["x"]), 0.5, 0.8)     # flagged, weak support
    assert sv.p2(row("d", True, 0.85, flags=["x"]), 0.5, 0.8)


def test_metrics_and_categories():
    rows = [row("s1", True, 0.9, cat="num"), row("s2", True, 0.1, cat="num"), row("u1", False, 0.9, cat="add")]
    m = sv.metrics(rows, [True, False, True])
    assert (m["false_acceptances"], m["false_rejections"], m["accuracy_pct"]) == (1, 1, 33.3)
    assert m["false_acceptance_rate_pct"] == 100.0 and m["false_rejection_rate_pct"] == 50.0
    assert m["by_category"]["num"] == {"supported": 2, "false_rejections": 1, "unsupported": 0, "false_acceptances": 0}


def test_calibration_weights_false_acceptance_double_and_prefers_strict_on_ties():
    # t <= 0.4 accepts u (1 FA = cost 2); t in (0.4, 0.7] rejects u, accepts s (cost 0)
    rows = [row("s", True, 0.7), row("u", False, 0.4)]
    assert sv.calibrate_p1(rows) == {"t": 0.7}  # every t in (0.4, 0.7] costs 0; the strictest wins
    p = sv.calibrate_p2([row("s", True, 0.7, contra=0.05), row("u", False, 0.1, contra=0.3)])
    assert p["tc"] <= 0.3  # the contradiction threshold must catch u


def test_evaluate_uses_only_calibration_rows_for_thresholds():
    rows = [row("s", True, 0.7), row("u", False, 0.4),
            row("hs", True, 0.2, split="heldout"), row("hu", False, 0.95, split="heldout")]
    out = sv.evaluate(rows)
    assert out["P1"]["params"] == {"t": 0.7}
    assert out["P1"]["heldout_decisions"] == {"hs": False, "hu": True}  # held-out results do not move thresholds
    assert out["latency_ms_per_answer_est"]["mean"] == 40.0


def test_report_is_never_overwritten(tmp_path):
    report = tmp_path / "r.json"
    report.write_text("{}")
    with pytest.raises(FileExistsError):
        sv.main(models=(), report=report)


def test_stress_set_shape():
    pairs = sv.load_pairs()
    assert len(pairs) == 40 and len({p["pair_id"] for p in pairs}) == 40
    assert collections.Counter(p["supported"] for p in pairs) == {True: 20, False: 20}
    split = collections.Counter((p["split"], p["supported"]) for p in pairs)
    assert split[("calibration", True)] + split[("calibration", False)] == 25
    assert abs(split[("heldout", True)] - split[("heldout", False)]) <= 1
    want = {"exact": (3, 2), "semantic_paraphrase": (4, 2), "technical_paraphrase": (3, 2), "implicit_entity": (3, 2),
            "numerical": (2, 4), "attribution": (2, 3), "experimental_added": (3, 5)}
    got = collections.Counter((p["category"], p["supported"]) for p in pairs)
    assert {c: (got[(c, True)], got[(c, False)]) for c in want} == want


def test_every_evidence_chunk_exists():
    path = Path("data/processed/chunks_256.jsonl")
    if not path.exists():
        pytest.skip("corpus not built")
    ids = {json.loads(l)["chunk_id"] for l in path.read_text(encoding="utf-8").splitlines()}
    for p in sv.load_pairs():
        assert set(p["evidence"]) <= ids, p["pair_id"]
