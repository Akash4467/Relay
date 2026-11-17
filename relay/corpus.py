"""Configuration-driven corpus selection and startup integrity checks (phase 11D).

config/app.yaml:
    corpus_version: c3            # cache/trace key; must equal the active corpus's corpus_version (checked)
    corpus:
      active: c3                  # which entry of `corpora` is served; c3 is the default and the rollback baseline
      corpora:
        c3: {corpus_version: c3, manifest_path: ..., processed_data_path: ..., index_path: ...}
        c6: {...}

Paths are relative to the repository root (never the caller's working directory). validate() checks, in order:
the entry exists and is complete; the manifest exists and parses; it declares the expected corpus_version; the
configured processed/index paths equal the manifest's layout (<root>/processed, <root>/index, root = the manifest's
"root" field or else the manifest's own directory); every file the retrieval settings need exists; every file the
manifest lists exists and matches its recorded sha256. Any failure raises CorpusError with what to fix. There is no
fallback: a selected corpus that does not validate is an error, never silently replaced by c3.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PROCESSED = ("chunks_{size}.jsonl",)
REQUIRED_INDEX = ("bm25_{size}.pkl", "faiss_{size}.index", "faiss_{size}.ids.json")
FIELDS = ("corpus_version", "manifest_path", "processed_data_path", "index_path")


class CorpusError(RuntimeError):
    """The selected corpus cannot be served; the message says what is wrong and how to fix it."""


@dataclass(frozen=True)
class Corpus:
    name: str
    corpus_version: str
    manifest_path: Path
    processed_path: Path
    index_path: Path


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def selected(cfg: dict, name: str | None = None) -> Corpus:
    """The configured corpus (cfg["corpus"]["active"]) or an explicitly named one, with absolute paths."""
    section = cfg.get("corpus")
    if not section or "corpora" not in section:
        raise CorpusError("config has no corpus section: add corpus.active and corpus.corpora (see docs/corpus_configuration.md)")
    name = name or section.get("active")
    entry = section["corpora"].get(name)
    if entry is None:
        raise CorpusError(f"corpus {name!r} is not defined under corpus.corpora (defined: {sorted(section['corpora'])})")
    missing = [f for f in FIELDS if not entry.get(f)]
    if missing:
        raise CorpusError(f"corpus {name!r}: missing {missing} in config")
    if name == section.get("active") and cfg.get("corpus_version") != entry["corpus_version"]:
        raise CorpusError(f"corpus_version is {cfg.get('corpus_version')!r} but corpus.active {name!r} has "
                          f"corpus_version {entry['corpus_version']!r}: change both together")
    return Corpus(name, entry["corpus_version"], _resolve(entry["manifest_path"]), _resolve(entry["processed_data_path"]),
                  _resolve(entry["index_path"]))


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate(c: Corpus, sizes: list[int] | set[int]) -> dict:
    """Raise CorpusError unless the corpus is complete and matches its manifest. Returns a short report."""
    if not c.manifest_path.is_file():
        raise CorpusError(f"corpus {c.name}: manifest not found at {c.manifest_path}; build or copy the corpus, "
                          f"or set corpus.active back to c3")
    try:
        manifest = json.loads(c.manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise CorpusError(f"corpus {c.name}: manifest {c.manifest_path} is unreadable ({e})") from e
    if manifest.get("corpus_version") != c.corpus_version:
        raise CorpusError(f"corpus {c.name}: manifest declares corpus_version {manifest.get('corpus_version')!r}, "
                          f"config expects {c.corpus_version!r}")
    root = _resolve(manifest["root"]) if manifest.get("root") else c.manifest_path.parent.resolve()
    for label, got, want in (("processed_data_path", c.processed_path, root / "processed"),
                             ("index_path", c.index_path, root / "index")):
        if got.resolve() != want.resolve():
            raise CorpusError(f"corpus {c.name}: {label} {got} is not the manifest's {want}")
    need = [c.processed_path / t.format(size=s) for s in sorted(sizes) for t in REQUIRED_PROCESSED] + \
           [c.index_path / t.format(size=s) for s in sorted(sizes) for t in REQUIRED_INDEX]
    absent = [p for p in need if not p.is_file()]
    if absent:
        raise CorpusError(f"corpus {c.name}: required artifacts missing: {[str(p) for p in absent]}")
    hashes = manifest.get("outputs_sha256") or {}
    if not hashes:
        raise CorpusError(f"corpus {c.name}: manifest records no outputs_sha256; integrity cannot be checked")
    bad, gone = [], []
    for rel, digest in sorted(hashes.items()):
        p = root / rel
        if rel == "manifest.json":
            continue
        if not p.is_file():
            gone.append(rel)
        elif _sha256(p) != digest:
            bad.append(rel)
    if gone or bad:
        raise CorpusError(f"corpus {c.name}: integrity check failed: missing {gone}, hash mismatch {bad} "
                          f"(manifest {c.manifest_path})")
    unlisted = [str(p) for p in need if p.resolve().relative_to(root.resolve()).as_posix() not in hashes]
    if unlisted:
        raise CorpusError(f"corpus {c.name}: required artifacts are not covered by the manifest's hashes: {unlisted}")
    return {"corpus": c.name, "corpus_version": c.corpus_version, "files_hash_checked": len(hashes),
            "processed": str(c.processed_path), "index": str(c.index_path)}


def load_indexes(c: Corpus, sizes, embedder) -> dict:
    from relay.retrieval.search import Index  # heavy import (faiss), only when loading
    return {size: Index(size, embedder, c.index_path, c.processed_path) for size in sorted(set(sizes))}
