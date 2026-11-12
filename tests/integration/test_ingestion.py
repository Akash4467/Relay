"""Ingestion pipeline wired together. Grows as each ingestion unit is built."""
import gzip

import httpx

from relay.retrieval import fetch_papers as fp


def test_fetch_all_three_papers_at_exact_versions(tmp_path):
    requested: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        requested.append(str(req.url))
        if "/e-print/" in str(req.url):
            return httpx.Response(200, content=gzip.compress(b"\\begin{document}"))
        return httpx.Response(200, content=b"%PDF")

    manifests = fp.fetch_all(httpx.Client(transport=httpx.MockTransport(handler)), tmp_path)

    assert [m["paper"] for m in manifests] == ["AIAYN", "MOSHI", "HNET"]
    assert sorted(requested) == sorted([
        "https://arxiv.org/e-print/1706.03762v7", "https://arxiv.org/pdf/1706.03762v7",
        "https://arxiv.org/e-print/2410.00037v1", "https://arxiv.org/pdf/2410.00037v1",
        "https://arxiv.org/e-print/2507.07955v2", "https://arxiv.org/pdf/2507.07955v2",
    ])
    for code in fp.PAPERS:
        assert (tmp_path / code / "src" / "main.tex").exists()
