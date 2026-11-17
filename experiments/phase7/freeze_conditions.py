"""Phase 7, stage 1 (offline): record the experimental conditions in eval/phase7/conditions.json (new file only)."""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")
from relay.config import app_config, retrieval_settings
from relay.retrieval import index_build


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def files(*dirs):
    return {f.as_posix(): sha(f) for d in dirs for f in sorted(Path(d).rglob("*")) if f.is_file()}


git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
cfg = app_config()
out = Path("eval/phase7/conditions.json")
out.parent.mkdir(parents=True, exist_ok=True)
record = {
    "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "git": {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "status": git("status", "--short").splitlines()},
    "default_corpus": {"config_corpus_version": cfg["corpus_version"], "dirs": ["data/processed", "data/index"],
                       "files_sha256": files("data/processed", "data/index")},
    "candidate_corpus": {"version": "c4", "dirs": ["data/corpus/c4/processed", "data/corpus/c4/index"],
                         "manifest_sha256": sha(Path("data/corpus/c4/manifest.json")),
                         "files_sha256": files("data/corpus/c4")},
    "model": cfg["models"]["primary"], "prompt_version": cfg["prompt_version"],
    "embedding": {"model": index_build.EMBED_MODEL, "query_prefix": index_build.QUERY_PREFIX},
    "retrieval_settings": {k: vars(v) if hasattr(v, "__dict__") else v.model_dump() for k, v in retrieval_settings().items()},
    "context_cap": cfg["pressure"]["L0"]["context_cap"],
    "verifier": {"verification_policy_version": cfg["verification_policy_version"], **cfg["verification"]},
    "datasets_sha256": {p: sha(Path(p)) for p in ("eval/questions.jsonl", "eval/answer_quality_set.json",
                                                  "eval/answer_quality_grades.csv", "eval/answer_quality_sentences.csv")},
    "historical_runs_sha256": {p.as_posix(): sha(p) for p in [Path("results/raw/answer_quality/real_2026-11-13T18-06-56Z.jsonl"),
                                                              Path("results/live_validation/real_v5_2026-11-13T19-39-49Z.jsonl"),
                                                              Path("results/results.csv")]},
}
with out.open("x", encoding="utf-8") as f:
    json.dump(record, f, indent=1, ensure_ascii=False, default=str)
print(out, "| c3 files", len(record["default_corpus"]["files_sha256"]), "| c4 files", len(record["candidate_corpus"]["files_sha256"]),
      "| verifier", record["verifier"]["verification_policy_version"], record["verifier"]["entail_min"])
