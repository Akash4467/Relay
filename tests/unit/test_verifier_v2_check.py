import collections
import json
from pathlib import Path

import pytest

from relay.eval import verifier_v2_check as v2

PAIRS = [json.loads(l) for l in v2.PAIRS.read_text(encoding="utf-8").splitlines()]
# evidence behind q05 (routing), q11 (Table 3) and q16 (training cost): never reused as fresh pairs
REAL_QUESTION_EVIDENCE = ("HNET:2.2.1:", "AIAYN:T3:", "AIAYN:T2:", "AIAYN:6.1:")


def test_shape_split_and_categories():
    assert len(PAIRS) == 20 and len({p["pair_id"] for p in PAIRS}) == 20
    split = collections.Counter((p["split"], p["supported"]) for p in PAIRS)
    assert split == {("calibration", True): 6, ("calibration", False): 6, ("heldout", True): 4, ("heldout", False): 4}
    assert {p["category"] for p in PAIRS} == {"table_row", "comparison", "latex", "pronoun", "absence", "added"}


def test_fresh_against_real_questions_and_earlier_stress_sets():
    earlier = set()
    for path in ("eval/verifier_stress.jsonl", "eval/verifier_stress_v2.jsonl"):
        earlier |= {json.loads(l)["claim"] for l in Path(path).read_text(encoding="utf-8").splitlines()}
    for p in PAIRS:
        assert not any(e in p["answer"] for e in earlier), p["pair_id"]
        assert not any(c.startswith(REAL_QUESTION_EVIDENCE) for c in p["evidence"]), p["pair_id"]


def test_rejected_context_case_is_present():
    p3 = next(p for p in PAIRS if p["pair_id"] == "p3")
    assert not p3["supported"] and " It " in p3["answer"]


def test_every_evidence_chunk_exists():
    path = Path("data/processed/chunks_256.jsonl")
    if not path.exists():
        pytest.skip("corpus not built")
    ids = {json.loads(l)["chunk_id"] for l in path.read_text(encoding="utf-8").splitlines()}
    for p in PAIRS:
        assert set(p["evidence"]) <= ids and all(f"[{c}]" in p["answer"] for c in p["evidence"]), p["pair_id"]


def test_cached_nli_computes_each_pair_once():
    class Model:
        config = object()
        calls = 0

        def predict(self, pairs):
            Model.calls += len(pairs)
            return [[0.0, 1.0, 0.0] for _ in pairs]
    c = v2.CachedNLI(Model())
    c.predict([("p", "h"), ("p", "h2")])
    c.predict([("p", "h"), ("p", "h3")])
    assert Model.calls == 3


def test_report_is_never_overwritten(tmp_path):
    r = tmp_path / "r.json"
    r.write_text("{}")
    with pytest.raises(FileExistsError):
        v2.main(r)
