"""Phase 11D: write config/corpus_manifests/c3.json from hashes that were RECORDED earlier (not computed now):
experiments/phase9/out/baseline_checksums.sha256 (start of Phase 9A) and experiments/phase9b/out/baseline_checksums.sha256
(start of Phase 9B). The two lists must agree for every c3 file, cover every file in data/processed and data/index, and
the files on disk must still match them; otherwise nothing is written. c3 itself (data/processed, data/index) is only read.
Refuses to overwrite an existing manifest."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LISTS = ["experiments/phase9/out/baseline_checksums.sha256", "experiments/phase9b/out/baseline_checksums.sha256"]
OUT = ROOT / "config/corpus_manifests/c3.json"


def recorded(path: str) -> dict[str, str]:
    out = {}
    for line in (ROOT / path).read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        if name.startswith(("data/processed/", "data/index/")):
            out[name] = digest
    return out


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; not overwriting")
    a, b = (recorded(p) for p in LISTS)
    assert a == b, "the two recorded lists disagree"
    on_disk = sorted(p.relative_to(ROOT).as_posix() for d in ("data/processed", "data/index") for p in (ROOT / d).iterdir() if p.is_file())
    assert sorted(a) == on_disk, "recorded list does not cover exactly the files on disk"
    now = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in a}
    assert now == a, "c3 files on disk differ from the recorded hashes"
    manifest = {
        "corpus_version": "c3",
        "root": "data",
        "note": "c3 predates corpus manifests (it lives in data/processed + data/index). This manifest was written in "
                "phase 11D from hashes recorded BEFORE it, at the start of phases 9A and 9B (identical lists); no hash "
                "here was computed for the purpose. outputs_sha256 keys are relative to root, as in the c4/c6 manifests.",
        "hash_sources": LISTS,
        "outputs_sha256": {n[len("data/"):]: h for n, h in sorted(a.items())},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("written", OUT.relative_to(ROOT).as_posix(), len(manifest["outputs_sha256"]), "files")


if __name__ == "__main__":
    main()
