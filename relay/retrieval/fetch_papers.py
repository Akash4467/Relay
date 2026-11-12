"""Download the three papers at their exact arXiv versions (DESIGN §6.1).

Writes data/raw/{CODE}/source.tar.gz, paper.pdf, src/ (extracted LaTeX) and manifest.json.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import httpx

PAPERS: dict[str, tuple[str, int]] = {
    "AIAYN": ("1706.03762", 7),
    "MOSHI": ("2410.00037", 1),
    "HNET": ("2507.07955", 2),
}
RAW_DIR = Path("data/raw")
USER_AGENT = "the-relay-assignment/0.1 (one-off download of three papers)"
_VERSIONED = re.compile(r"\d{4}\.\d{4,5}v\d+$")


def urls(arxiv_id: str, version: int) -> dict[str, str]:
    vid = f"{arxiv_id}v{version}"
    if not version or not _VERSIONED.fullmatch(vid):
        raise ValueError(f"refusing unversioned or malformed arXiv id: {vid!r}")
    return {"source": f"https://arxiv.org/e-print/{vid}", "pdf": f"https://arxiv.org/pdf/{vid}"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _extract_source(data: bytes, dest: Path) -> None:
    """arXiv e-prints are either a gzipped tarball or a single gzipped .tex file."""
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
            tar.extractall(dest, filter="data")  # "data" filter blocks path traversal and links
    except tarfile.ReadError:
        (dest / "main.tex").write_bytes(gzip.decompress(data))


def _up_to_date(paper_dir: Path) -> bool:
    manifest = paper_dir / "manifest.json"
    if not manifest.exists():
        return False
    files = json.loads(manifest.read_text())["files"]
    return all(
        (paper_dir / f["path"]).exists() and _sha256((paper_dir / f["path"]).read_bytes()) == f["sha256"]
        for f in files
    )


def fetch(code: str, client: httpx.Client, raw_dir: Path = RAW_DIR) -> dict:
    arxiv_id, version = PAPERS[code]
    paper_dir = raw_dir / code
    if _up_to_date(paper_dir):
        return json.loads((paper_dir / "manifest.json").read_text())

    paper_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for kind, url in urls(arxiv_id, version).items():
        resp = client.get(url)
        resp.raise_for_status()
        name = "source.tar.gz" if kind == "source" else "paper.pdf"
        (paper_dir / name).write_bytes(resp.content)
        files.append({
            "kind": kind, "path": name, "url": url, "final_url": str(resp.url),
            "sha256": _sha256(resp.content), "bytes": len(resp.content),
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
        })
        if kind == "source":
            _extract_source(resp.content, paper_dir / "src")

    manifest = {"paper": code, "arxiv_id": arxiv_id, "version": version, "files": files}
    (paper_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def fetch_all(client: httpx.Client, raw_dir: Path = RAW_DIR) -> list[dict]:
    return [fetch(code, client, raw_dir) for code in PAPERS]


def main() -> None:
    with httpx.Client(follow_redirects=True, timeout=60, headers={"User-Agent": USER_AGENT}) as client:
        for m in fetch_all(client):
            print(m["paper"], f'{m["arxiv_id"]}v{m["version"]}', [f["sha256"][:12] for f in m["files"]])


if __name__ == "__main__":
    main()
