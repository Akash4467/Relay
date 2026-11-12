import json
import pickle

import faiss
import numpy as np

from relay.retrieval import index_build as ib


class FakeModel:
    """Deterministic bag-of-words embedder: lets build() run without downloading BGE."""
    VOCAB = ["attention", "heads", "mimi", "codec", "chunking", "boundary", "bleu", "moshi", "transformer", "h-net"]

    def encode(self, texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False):
        out = np.array([[t.lower().count(w) for w in self.VOCAB] for t in texts], dtype="float32") + 1e-3
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def chunk(cid, paper, text, sid="1", title="Intro"):
    return {"chunk_id": cid, "paper": paper, "section_id": sid, "section_title": title, "text": text}


CHUNKS = [
    chunk("AIAYN:3.2.2:c001", "AIAYN", "We employ h=8 parallel attention heads.", "3.2.2", "Multi-Head Attention"),
    chunk("MOSHI:3.3:c002", "MOSHI", "Mimi is a neural audio codec at 12.5Hz.", "3.3", "Audio Tokenization"),
    chunk("HNET:2.2.1:c003", "HNET", "The routing module predicts a boundary with cosine similarity.", "2.2.1", "Chunking Layer"),
]


def test_tokenize_keeps_compounds_and_adds_parts():
    assert ib.tokenize("The RQ-Transformer uses d_model at 12.5Hz.") == \
        ["the", "rq-transformer", "rq", "transformer", "uses", "d_model", "d", "model", "at", "12.5hz"]
    assert ib.tokenize("") == []


def test_index_text_names_paper_and_section():
    text = ib.index_text(CHUNKS[1])
    assert text.startswith("Moshi: a speech-text foundation model")
    assert "3.3 Audio Tokenization" in text and text.endswith("12.5Hz.")


def test_bm25_ranks_the_matching_chunk_first():
    bm25 = ib.build_bm25(CHUNKS)
    scores = bm25.get_scores(ib.tokenize("how many attention heads"))
    assert int(np.argmax(scores)) == 0
    # the paper header makes "Moshi" match the MOSHI chunk even though its text never says Moshi
    assert int(np.argmax(bm25.get_scores(ib.tokenize("moshi codec")))) == 1


def test_faiss_inner_product_is_cosine():
    vecs = FakeModel().encode(["mimi codec", "attention heads"])
    index = ib.build_faiss(vecs)
    scores, idx = index.search(FakeModel().encode(["mimi"]), 2)
    assert idx[0][0] == 0 and 0 < scores[0][0] <= 1.0001


def test_build_writes_all_files(tmp_path):
    proc, out = tmp_path / "proc", tmp_path / "index"
    proc.mkdir()
    (proc / "chunks_256.jsonl").write_text("".join(json.dumps(c) + "\n" for c in CHUNKS), encoding="utf-8")
    report = ib.build(256, FakeModel(), proc, out)
    assert report["chunks"] == 3 and report["dim"] == len(FakeModel.VOCAB)

    ids = [c["chunk_id"] for c in CHUNKS]
    with (out / "bm25_256.pkl").open("rb") as f:
        saved = pickle.load(f)
    assert saved["ids"] == ids
    assert json.loads((out / "faiss_256.ids.json").read_text()) == ids
    assert faiss.read_index(str(out / "faiss_256.index")).ntotal == 3
    idf = json.loads((out / "idf_256.json").read_text(encoding="utf-8"))
    assert {"mimi", "routing", "attention"} <= set(idf)
    assert all(isinstance(v, float) for v in idf.values())
