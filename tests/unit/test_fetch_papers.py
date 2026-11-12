import gzip
import io
import json
import tarfile

import httpx
import pytest

from relay.retrieval import fetch_papers as fp


def _tar_gz(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _client(routes: dict[str, bytes], seen: list[str] | None = None) -> httpx.Client:
    def handler(req: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(str(req.url))
        body = routes.get(str(req.url))
        return httpx.Response(200, content=body) if body is not None else httpx.Response(404)
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_urls_carry_exact_version():
    assert fp.urls("2410.00037", 1) == {
        "source": "https://arxiv.org/e-print/2410.00037v1",
        "pdf": "https://arxiv.org/pdf/2410.00037v1",
    }


@pytest.mark.parametrize("arxiv_id,version", [("2410.00037", 0), ("2410.00037", None), ("bad", 1)])
def test_urls_refuse_unversioned_or_malformed(arxiv_id, version):
    with pytest.raises(ValueError):
        fp.urls(arxiv_id, version)


def test_fetch_tarball_writes_files_manifest_and_extracts(tmp_path):
    src = _tar_gz({"main.tex": b"\\documentclass{article}", "sec/intro.tex": b"hi"})
    client = _client({
        "https://arxiv.org/e-print/2410.00037v1": src,
        "https://arxiv.org/pdf/2410.00037v1": b"%PDF-1.5",
    })
    m = fp.fetch("MOSHI", client, tmp_path)

    d = tmp_path / "MOSHI"
    assert (d / "src" / "main.tex").read_bytes() == b"\\documentclass{article}"
    assert (d / "src" / "sec" / "intro.tex").exists()
    assert (d / "paper.pdf").read_bytes() == b"%PDF-1.5"
    assert m["version"] == 1
    assert {f["sha256"] for f in m["files"]} == {fp._sha256(src), fp._sha256(b"%PDF-1.5")}
    assert json.loads((d / "manifest.json").read_text()) == m


def test_fetch_single_gzipped_tex(tmp_path):
    client = _client({
        "https://arxiv.org/e-print/1706.03762v7": gzip.compress(b"\\begin{document}"),
        "https://arxiv.org/pdf/1706.03762v7": b"%PDF",
    })
    fp.fetch("AIAYN", client, tmp_path)
    assert (tmp_path / "AIAYN" / "src" / "main.tex").read_bytes() == b"\\begin{document}"


def test_fetch_skips_download_when_manifest_matches(tmp_path):
    routes = {
        "https://arxiv.org/e-print/2507.07955v2": gzip.compress(b"x"),
        "https://arxiv.org/pdf/2507.07955v2": b"%PDF",
    }
    fp.fetch("HNET", _client(routes), tmp_path)
    seen: list[str] = []
    fp.fetch("HNET", _client(routes, seen), tmp_path)
    assert seen == []


def test_fetch_redownloads_when_file_corrupted(tmp_path):
    routes = {
        "https://arxiv.org/e-print/2507.07955v2": gzip.compress(b"x"),
        "https://arxiv.org/pdf/2507.07955v2": b"%PDF",
    }
    fp.fetch("HNET", _client(routes), tmp_path)
    (tmp_path / "HNET" / "paper.pdf").write_bytes(b"corrupt")
    seen: list[str] = []
    fp.fetch("HNET", _client(routes, seen), tmp_path)
    assert len(seen) == 2
    assert (tmp_path / "HNET" / "paper.pdf").read_bytes() == b"%PDF"


def test_fetch_http_error_raises_and_writes_no_manifest(tmp_path):
    with pytest.raises(httpx.HTTPStatusError):
        fp.fetch("MOSHI", _client({}), tmp_path)
    assert not (tmp_path / "MOSHI" / "manifest.json").exists()


def test_tar_path_traversal_blocked(tmp_path):
    client = _client({
        "https://arxiv.org/e-print/2410.00037v1": _tar_gz({"../evil.tex": b"x"}),
        "https://arxiv.org/pdf/2410.00037v1": b"%PDF",
    })
    with pytest.raises(Exception):
        fp.fetch("MOSHI", client, tmp_path)
    assert not (tmp_path / "MOSHI" / "evil.tex").exists()
