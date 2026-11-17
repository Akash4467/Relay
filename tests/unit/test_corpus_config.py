"""Phase 11D: configuration-driven corpus selection and startup integrity checks.
Failure cases use small fake corpora in tmp_path; selection and integrity of the real c3/c6 read their files only."""
import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from relay import corpus
from relay.config import app_config, piece_sizes

REPO = corpus.REPO_ROOT
SIZES = [256]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- the real configuration
def test_default_selects_c3_with_repo_root_paths():
    c = corpus.selected(app_config())
    assert (c.name, c.corpus_version) == ("c3", "c3") and app_config()["corpus_version"] == "c3"
    assert c.processed_path == REPO / "data" / "processed" and c.index_path == REPO / "data" / "index"


def test_explicit_candidate_selects_c6():
    c = corpus.selected(app_config(), "c6")
    assert c.corpus_version == "c6" and c.processed_path == REPO / "experiments/phase10/corpus/c6/processed"


def test_both_corpora_pass_the_integrity_check():
    for name in ("c3", "c6"):
        r = corpus.validate(corpus.selected(app_config(), name), piece_sizes())
        assert r["files_hash_checked"] == 18


def test_paths_resolve_the_same_from_another_working_directory(tmp_path, monkeypatch):
    here = corpus.selected(app_config(), "c6")
    monkeypatch.chdir(tmp_path)
    there = corpus.selected(app_config(), "c6")  # config and corpus paths both come from the repository root
    assert there == here and corpus.validate(there, SIZES)["corpus"] == "c6"


def test_c3_is_unchanged_and_usable_after_c6_is_selected():
    files = sorted(p for d in ("data/processed", "data/index") for p in (REPO / d).iterdir() if p.is_file())
    before = {p: sha(p) for p in files}
    corpus.validate(corpus.selected(app_config(), "c6"), piece_sizes())
    assert {p: sha(p) for p in files} == before
    assert corpus.validate(corpus.selected(app_config()), piece_sizes())["corpus"] == "c3"


# ---------------------------------------------------------------- failure cases on a fake corpus
def make_corpus(tmp_path: Path, version="cX") -> dict:
    root = tmp_path / "cx"
    (root / "processed").mkdir(parents=True)
    (root / "index").mkdir()
    files = {"processed/chunks_256.jsonl": "{}\n", "index/bm25_256.pkl": "b", "index/faiss_256.index": "f",
             "index/faiss_256.ids.json": "[]"}
    for rel, text in files.items():
        (root / rel).write_text(text, encoding="utf-8")
    manifest = {"corpus_version": version, "outputs_sha256": {rel: sha(root / rel) for rel in files}}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return {"corpus_version": "cX", "corpus": {"active": "cX", "corpora": {"cX": {
        "corpus_version": "cX", "manifest_path": str(root / "manifest.json"),
        "processed_data_path": str(root / "processed"), "index_path": str(root / "index")}}}}


def check(cfg):
    return corpus.validate(corpus.selected(cfg), SIZES)


def test_fake_corpus_validates(tmp_path):
    assert check(make_corpus(tmp_path))["files_hash_checked"] == 4


def test_missing_manifest_fails_clearly(tmp_path):
    cfg = make_corpus(tmp_path)
    Path(cfg["corpus"]["corpora"]["cX"]["manifest_path"]).unlink()
    with pytest.raises(corpus.CorpusError, match="manifest not found .* set corpus.active back to c3"):
        check(cfg)


def test_unreadable_manifest_fails_clearly(tmp_path):
    cfg = make_corpus(tmp_path)
    Path(cfg["corpus"]["corpora"]["cX"]["manifest_path"]).write_text("{not json", encoding="utf-8")
    with pytest.raises(corpus.CorpusError, match="unreadable"):
        check(cfg)


def test_wrong_corpus_version_fails_clearly(tmp_path):
    with pytest.raises(corpus.CorpusError, match="manifest declares corpus_version 'c9'"):
        check(make_corpus(tmp_path, version="c9"))


def test_cache_key_and_active_corpus_must_agree(tmp_path):
    cfg = make_corpus(tmp_path)
    cfg["corpus_version"] = "c3"
    with pytest.raises(corpus.CorpusError, match="change both together"):
        corpus.selected(cfg)


@pytest.mark.parametrize("rel,label", [("index/faiss_256.index", "index"), ("processed/chunks_256.jsonl", "processed")])
def test_missing_required_artifact_fails_clearly(tmp_path, rel, label):
    cfg = make_corpus(tmp_path)
    (tmp_path / "cx" / rel).unlink()
    with pytest.raises(corpus.CorpusError, match="required artifacts missing"):
        check(cfg)


def test_hash_mismatch_fails_clearly(tmp_path):
    cfg = make_corpus(tmp_path)
    (tmp_path / "cx" / "index" / "bm25_256.pkl").write_text("tampered", encoding="utf-8")
    with pytest.raises(corpus.CorpusError, match=r"hash mismatch \['index/bm25_256.pkl'\]"):
        check(cfg)


def test_listed_file_missing_fails_clearly(tmp_path):
    cfg = make_corpus(tmp_path)
    m = json.loads((tmp_path / "cx" / "manifest.json").read_text(encoding="utf-8"))
    m["outputs_sha256"]["processed/tables.jsonl"] = "0" * 64
    (tmp_path / "cx" / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(corpus.CorpusError, match=r"missing \['processed/tables.jsonl'\]"):
        check(cfg)


def test_manifest_without_hashes_is_not_accepted(tmp_path):
    cfg = make_corpus(tmp_path)
    (tmp_path / "cx" / "manifest.json").write_text(json.dumps({"corpus_version": "cX"}), encoding="utf-8")
    with pytest.raises(corpus.CorpusError, match="records no outputs_sha256"):
        check(cfg)


def test_paths_inconsistent_with_the_manifest_fail(tmp_path):
    cfg = make_corpus(tmp_path)
    other = tmp_path / "elsewhere"
    other.mkdir()
    cfg["corpus"]["corpora"]["cX"]["index_path"] = str(other)
    with pytest.raises(corpus.CorpusError, match="index_path .* is not the manifest's"):
        check(cfg)


def test_undefined_corpus_and_missing_section_fail():
    with pytest.raises(corpus.CorpusError, match="not defined"):
        corpus.selected(app_config(), "c9")
    with pytest.raises(corpus.CorpusError, match="no corpus section"):
        corpus.selected({"corpus_version": "c3"})


def test_selected_but_broken_corpus_never_falls_back_to_c3(tmp_path):
    cfg = copy.deepcopy(app_config())
    cfg["corpus"]["corpora"]["c6"] = dict(cfg["corpus"]["corpora"]["c6"], manifest_path=str(tmp_path / "nope.json"))
    cfg["corpus"]["active"], cfg["corpus_version"] = "c6", "c6"
    with pytest.raises(corpus.CorpusError, match="corpus c6: manifest not found"):
        corpus.validate(corpus.selected(cfg), SIZES)


# ---------------------------------------------------------------- consumers load the configured corpus
def test_no_consumer_builds_an_index_from_default_paths():
    for path in ("relay/main.py", "relay/eval/answer_quality.py", "relay/live_validation.py", "relay/eval/retrieval_recall.py"):
        tree = ast.parse((REPO / path).read_text(encoding="utf-8"))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "Index"]
        assert calls == [], f"{path} constructs Index directly"
        assert "corpus.load_indexes" in (REPO / path).read_text(encoding="utf-8")


def stub_state(monkeypatch, cfg):
    """main.State with the models stubbed: records which corpus its indexes were loaded from."""
    from relay import main
    seen = {}
    monkeypatch.setattr(main, "app_config", lambda: cfg)
    monkeypatch.setattr(main, "from_env", lambda t: type("C", (), {"source": "standin"})())
    monkeypatch.setattr(main, "load_model", lambda: "embedder")
    monkeypatch.setattr(main, "load_reranker", lambda: None)
    monkeypatch.setattr(main.nli, "load_nli", lambda name: None)
    monkeypatch.setattr(main.corpus, "load_indexes", lambda c, sizes, emb: seen.setdefault("corpus", c) and {256: None})
    return main, seen


def test_server_startup_serves_the_configured_corpus(monkeypatch):
    cfg = copy.deepcopy(app_config())
    cfg["corpus"]["active"], cfg["corpus_version"] = "c6", "c6"
    main, seen = stub_state(monkeypatch, cfg)
    s = main.State()
    assert seen["corpus"].name == "c6" and s.corpus_check["files_hash_checked"] == 18


def test_server_startup_refuses_a_broken_corpus(monkeypatch, tmp_path):
    cfg = copy.deepcopy(app_config())
    cfg["corpus"]["corpora"]["c6"] = dict(cfg["corpus"]["corpora"]["c6"], index_path=str(tmp_path))
    cfg["corpus"]["active"], cfg["corpus_version"] = "c6", "c6"
    main, seen = stub_state(monkeypatch, cfg)
    with pytest.raises(SystemExit, match="corpus check failed: corpus c6: index_path"):
        main.State()
    assert "corpus" not in seen  # nothing was loaded
