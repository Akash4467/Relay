"""Build the BM25 and dense indexes for each piece size (DESIGN §6.5).

data/processed/chunks_{size}.jsonl -> data/index/{bm25_{size}.pkl, idf_{size}.json, faiss_{size}.index,
faiss_{size}.ids.json}. Offline only; the server loads these files and never rebuilds them.
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from relay.config import piece_sizes
from relay.retrieval.latex_parse import PROCESSED_DIR

INDEX_DIR = Path("data/index")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # BGE model card: queries only
PAPER_TITLES = {
    "AIAYN": "Attention Is All You Need (Transformer)",
    "MOSHI": "Moshi: a speech-text foundation model for real-time dialogue",
    "HNET": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling (H-Net)",
}
_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase terms; compound terms stay whole ("rq-transformer", "d_model", "12.5") and also add their parts."""
    out = []
    for tok in _TOKEN.findall(text.lower()):
        out.append(tok)
        parts = re.split(r"[-_]", tok)
        if len(parts) > 1:
            out += [p for p in parts if p]
    return out


def index_text(chunk: dict) -> str:
    """What gets indexed: a header naming the paper and section, then the chunk text. Without the header a
    chunk can't match a question that names its paper ("in the Moshi paper ...") or its section topic."""
    return f"{PAPER_TITLES[chunk['paper']]} | {chunk['section_id']} {chunk['section_title']}\n{chunk['text']}"


def build_bm25(chunks: list[dict]) -> BM25Okapi:
    return BM25Okapi([tokenize(index_text(c)) for c in chunks])


def embed(texts: list[str], model) -> np.ndarray:
    return np.asarray(model.encode(texts, batch_size=32, normalize_embeddings=True,
                                   show_progress_bar=False), dtype="float32")


def build_faiss(vectors: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(vectors.shape[1])  # inner product of normalized vectors = cosine
    index.add(vectors)
    return index


def build(size: int, model, processed_dir: Path = PROCESSED_DIR, index_dir: Path = INDEX_DIR) -> dict:
    chunks = [json.loads(l) for l in (processed_dir / f"chunks_{size}.jsonl").read_text(encoding="utf-8").splitlines()]
    ids = [c["chunk_id"] for c in chunks]
    bm25 = build_bm25(chunks)
    vectors = embed([index_text(c) for c in chunks], model)

    index_dir.mkdir(parents=True, exist_ok=True)
    with (index_dir / f"bm25_{size}.pkl").open("wb") as f:
        pickle.dump({"ids": ids, "bm25": bm25}, f)
    (index_dir / f"idf_{size}.json").write_text(json.dumps(bm25.idf, ensure_ascii=False), encoding="utf-8")
    faiss.write_index(build_faiss(vectors), str(index_dir / f"faiss_{size}.index"))
    (index_dir / f"faiss_{size}.ids.json").write_text(json.dumps(ids), encoding="utf-8")
    return {"size": size, "chunks": len(ids), "vocab": len(bm25.idf), "dim": int(vectors.shape[1])}


def load_model():
    from sentence_transformers import SentenceTransformer  # heavy import, only when embedding
    return SentenceTransformer(EMBED_MODEL, device="cpu")


def main(sizes: list[int] | None = None) -> list[dict]:
    model = load_model()
    return [build(size, model) for size in sizes or piece_sizes()]


if __name__ == "__main__":
    for r in main():
        print(r)
