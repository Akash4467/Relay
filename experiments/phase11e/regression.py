"""Phase 11E offline regression, run once per corpus (usage: regression.py c3|c6). MOCKED / STAND-IN ONLY: the real
FastAPI app, real retrieval, real verifier v4.2 (real NLI); scripted provider replies (no API). Not model performance.

The corpus is selected through the production startup path: main.State() reads a config identical to config/app.yaml
except corpus.active / corpus_version (in memory only; the file on disk stays c3). Writes
experiments/phase11e/{c3,c6}_regression.json and raw outputs under experiments/phase11e/out/<corpus>/.

Suites:
  startup       State() validated and loaded the selected corpus (manifest, paths, every recorded hash)
  phase10       experiments/phase10/offline_integration.py SCENARIOS (31 failure/robustness cases)
  e2e           experiments/phase11c/e2e_offline.py SCENARIOS (15 question cases) with identity checks: every
                evidence chunk the verifier received equals the SELECTED corpus's text; G (expansion, caption sharing),
                p4 and the original question in the prompt
  fail_closed   the selected corpus copied to a scratch directory, then a missing manifest / a corrupted index file /
                a wrong corpus_version: startup must refuse (SystemExit), never fall back to c3. Sources never touched.
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/phase10")
sys.path.insert(0, "experiments/phase11c")
import e2e_offline as E  # noqa: E402  (sets LLM_PROVIDER=standin and a dummy key; never a real key)
import offline_integration as P10  # noqa: E402
import relay.pipeline as pipeline  # noqa: E402
from relay import corpus, main  # noqa: E402
from relay.config import app_config  # noqa: E402
from relay.llm import prompts  # noqa: E402

HERE = Path("experiments/phase11e")


def config_for(name: str, entry_override: dict | None = None) -> dict:
    cfg = copy.deepcopy(app_config())
    cfg["corpus"]["active"], cfg["corpus_version"] = name, cfg["corpus"]["corpora"][name]["corpus_version"]
    if entry_override:
        cfg["corpus"]["corpora"][name] = dict(cfg["corpus"]["corpora"][name], **entry_override)
    return cfg


def with_config(cfg):
    original = main.app_config
    main.app_config = lambda: cfg
    return original


def startup(name: str) -> tuple[dict, object]:
    cfg = config_for(name)
    orig = with_config(cfg)
    try:
        s = main.State()
    finally:
        main.app_config = orig
    want = corpus.selected(cfg)
    checks = {"selected": s.corpus.name == name, "paths": (s.corpus.processed_path, s.corpus.index_path, s.corpus.manifest_path)
              == (want.processed_path, want.index_path, want.manifest_path),
              "integrity_validated": s.corpus_check["files_hash_checked"] == 18,
              "indexes_from_selected_corpus": all(i.chunks == {c["chunk_id"]: c for c in map(json.loads,
                  (want.processed_path / f"chunks_{size}.jsonl").open(encoding="utf-8"))} for size, i in s.indexes.items()),
              "configuration_G": s.cfg["retrieval_policy"]["symbolic_query_expansion"] is True
              and s.cfg["retrieval_policy"]["prompt_caption_sharing"] is True,
              "verifier_v4_2": s.cfg["verification_policy_version"] == "v4.2" and s.verifier.entail_min == 0.9,
              "prompt_p4": s.cfg["prompt_version"] == prompts.PROMPT_VERSION == "p4"}
    return {"corpus_check": {k: str(v) for k, v in s.corpus_check.items()}, "checks": checks, "passed": all(checks.values())}, s


def phase10_suite(name: str) -> dict:
    out = HERE / "out" / name / "phase10_suite"
    P10.OUT = out
    orig = with_config(config_for(name))
    try:
        P10.main_()
    finally:
        main.app_config = orig
    res = json.loads(sorted(out.glob("offline_integration_*.json"))[-1].read_text(encoding="utf-8"))
    rows = res["scenarios"]
    return {"scenarios": len(rows), "passed": sum(r["passed"] for r in rows), "failed": [r["scenario"] for r in rows if not r["passed"]],
            "output": sorted(out.glob("offline_integration_*.json"))[-1].as_posix()}


def e2e_suite(name: str, s) -> dict:
    out = HERE / "out" / name / "e2e"
    out.mkdir(parents=True, exist_ok=True)
    main.TRACE_DIR = out / "traces"
    main.STATE = s
    texts = {cid: c["text"] for cid, c in s.indexes[256].chunks.items()}
    other = "c6" if name == "c3" else "c3"
    other_texts = {c["chunk_id"]: c["text"] for c in map(json.loads, (corpus.selected(app_config(), other).processed_path
                                                                       / "chunks_256.jsonl").open(encoding="utf-8"))}
    seen = []
    real_verify = pipeline.verify

    def spy(draft, passages, *a, **k):
        seen.append(copy.deepcopy(passages))
        return real_verify(draft, passages, *a, **k)
    pipeline.verify = spy
    from fastapi.testclient import TestClient
    http = TestClient(main.app)
    rows = []
    try:
        for scen, qid, drafts, expect in E.SCENARIOS:
            provider = E.Provider(drafts)
            s.client = E.ChatClient("https://fake-provider.invalid/v1", "dummy-offline-key", s.cfg["timeouts"], "standin",
                                    out / "provider_raw", E.httpx.MockTransport(provider))
            s.governor, s.budget = E.Governor.from_config(), E.RetryBudget.from_config(s.cfg["retry"])
            seen.clear()
            question = E.QS[qid]
            events, ev_name = [], None
            with http.stream("POST", "/ask", json={"question": question, "request_id": f"p11e-{name}-{scen}",
                                                   "scenario": "phase11e_offline", "deadline_ms": 120000}) as r:
                for line in r.iter_lines():
                    if line.startswith("event: "):
                        ev_name = line[7:]
                    elif line.startswith("data: "):
                        events.append((ev_name, json.loads(line[6:])))
            tr = json.loads(sorted((out / "traces").glob("*.jsonl"))[-1].read_text(encoding="utf-8").splitlines()[-1])
            done = next((d for n, d in events if n == "done"), {})
            content = next((d["text"] for n, d in events if n == "content"), "")
            ids = tr.get("evidence_ids") or []
            checks = {"trace_corpus_version": tr.get("corpus_version") == name,
                      "caption_sharing_on": tr.get("caption_sharing") is True,
                      "verifier_got_retrieved_chunks": all([p["chunk_id"] for p in ps] == ids for ps in seen),
                      "verifier_evidence_is_selected_corpus_text": all(p["text"] == texts[p["chunk_id"]] for ps in seen for p in ps)}
            for req in provider.requests:
                user = req["messages"][1]["content"]
                checks["original_question_in_prompt"] = checks.get("original_question_in_prompt", True) and \
                    user.startswith(f"Question: {question}\n") and not any(f"[{t}]" in user for t in tr.get("query_expansion") or [])
            if expect == "refused":
                checks["outcome"] = done.get("tier") == "refused"
            elif expect == "refused_unless_r01_retrieved":
                checks["outcome"] = done.get("tier") == "refused" or "AIAYN:T3:r01" in ids
            if done.get("tier") == "normal":
                checks["answer_is_a_verified_draft"] = any(content == pipeline.citations.normalize_brackets(d).strip()
                                                           for d in provider.sent) and bool(done.get("citations"))
            differs = [c for c in ids if c in other_texts and other_texts[c] != texts[c]]
            rows.append({"scenario": scen, "question_id": qid, "outcome": done.get("tier"), "reason": done.get("reason"),
                         "llm_calls": tr.get("llm_calls"), "expansion": tr.get("query_expansion"), "evidence_ids": ids,
                         "evidence_differing_from_other_corpus": differs, "checks": checks, "passed": all(checks.values())})
    finally:
        pipeline.verify = real_verify
    unit = next(r for r in rows if r["scenario"] == "q09_units")
    has_230ms = any("230 ms" in texts[c] for c in unit["evidence_ids"])
    return {"scenarios": len(rows), "passed": sum(r["passed"] for r in rows), "failed": [r["scenario"] for r in rows if not r["passed"]],
            "corpus_identity": {"q09 evidence contains '230 ms'": has_230ms, "expected": name == "c6"},
            "q11_expansion": next(r["expansion"] for r in rows if r["question_id"] == "q11"), "rows": rows}


def fail_closed(name: str) -> dict:
    src = corpus.selected(app_config(), name)
    cases = {}
    with tempfile.TemporaryDirectory(dir=str(Path.home() / "AppData/Local/Temp")) as tmp:
        root = Path(tmp) / name
        shutil.copytree(src.processed_path, root / "processed")
        shutil.copytree(src.index_path, root / "index")
        m = json.loads(src.manifest_path.read_text(encoding="utf-8"))
        m.pop("root", None)  # the copy's manifest sits at the copy's root
        override = {"manifest_path": str(root / "manifest.json"), "processed_data_path": str(root / "processed"),
                    "index_path": str(root / "index")}

        def attempt(label, mutate):
            (root / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
            mutate()
            orig = with_config(config_for(name, override))
            try:
                main.State()
                cases[label] = {"refused": False, "message": None}
            except SystemExit as e:
                cases[label] = {"refused": True, "message": str(e)[:240]}
            finally:
                main.app_config = orig

        attempt("intact copy (control: must start)", lambda: None)
        cases["intact copy (control: must start)"]["refused_expected"] = False
        attempt("corrupted index file", lambda: (root / "index" / "bm25_256.pkl").write_bytes(b"corrupted"))
        attempt("missing manifest", lambda: (root / "manifest.json").unlink())
        attempt("wrong corpus_version in manifest",
                lambda: (root / "manifest.json").write_text(json.dumps(dict(m, corpus_version="c9")), encoding="utf-8"))
    ok = (not cases["intact copy (control: must start)"]["refused"]) and all(
        v["refused"] and f"corpus {name}" in v["message"] for k, v in cases.items() if not k.startswith("intact"))
    return {"cases": cases, "passed": ok, "no_fallback_to_c3": all(
        v["refused"] for k, v in cases.items() if not k.startswith("intact"))}


def main_(name: str) -> int:
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    st, s = startup(name)
    res = {"corpus": name, "kind": "MOCKED/STAND-IN offline regression (scripted provider; not model performance)",
           "started": started, "config_on_disk_active": app_config()["corpus"]["active"], "startup": st,
           "e2e": e2e_suite(name, s)}
    res["phase10_suite"] = phase10_suite(name)
    res["fail_closed"] = fail_closed(name)
    res["passed"] = (st["passed"] and res["e2e"]["passed"] == res["e2e"]["scenarios"]
                     and res["phase10_suite"]["passed"] == res["phase10_suite"]["scenarios"] and res["fail_closed"]["passed"]
                     and res["e2e"]["corpus_identity"]["q09 evidence contains '230 ms'"] == res["e2e"]["corpus_identity"]["expected"])
    out = HERE / f"{name}_regression.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(name, "startup", st["passed"], st["checks"])
    print(name, "e2e", res["e2e"]["passed"], "/", res["e2e"]["scenarios"], res["e2e"]["failed"], res["e2e"]["corpus_identity"])
    print(name, "phase10 suite", res["phase10_suite"]["passed"], "/", res["phase10_suite"]["scenarios"], res["phase10_suite"]["failed"])
    print(name, "fail closed", res["fail_closed"]["passed"], {k: v["refused"] for k, v in res["fail_closed"]["cases"].items()})
    print(name, "PASSED" if res["passed"] else "FAILED", out.as_posix())
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main_(sys.argv[1]))
