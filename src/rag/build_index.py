"""Embed every chunk and build a FAISS index.

Reads  data/documents/chunks.jsonl
Writes data/index/chunks.faiss   (the vectors)
       data/index/ids.json       (evidence IDs in the same order as the vectors)
       data/index/info.json      (model name, size, build time)

Run from the project root:  python -m src.rag.build_index
"""
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHUNKS_JSONL = ROOT / "data" / "documents" / "chunks.jsonl"
INDEX_DIR = ROOT / "data" / "index"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_SEQ_LENGTH = 512


def load_model():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


def embed(model, texts, is_query=False, show_progress=False):
    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]
    vecs = model.encode(texts, batch_size=32, normalize_embeddings=True,
                        show_progress_bar=show_progress)
    return np.asarray(vecs, dtype="float32")


def load_chunks():
    with open(CHUNKS_JSONL, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build(model=None):
    chunks = load_chunks()
    model = model or load_model()
    t0 = time.time()
    vecs = embed(model, [c["text"] for c in chunks], show_progress=True)
    index = faiss.IndexFlatIP(vecs.shape[1])  # cosine similarity (vectors are normalized)
    index.add(vecs)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "chunks.faiss"))
    (INDEX_DIR / "ids.json").write_text(
        json.dumps([c["evidence_id"] for c in chunks]), encoding="utf-8")
    (INDEX_DIR / "info.json").write_text(json.dumps({
        "model": MODEL_NAME, "dim": int(vecs.shape[1]), "n_chunks": len(chunks),
        "seconds": round(time.time() - t0, 1),
    }, indent=2), encoding="utf-8")
    return chunks, index, model


def main() -> int:
    if not CHUNKS_JSONL.is_file():
        print("chunks.jsonl not found. Run: python -m src.rag.chunk")
        return 1
    chunks, index, model = build()
    print(f"Indexed {index.ntotal} chunks with {MODEL_NAME}")

    # Sanity check: one known question
    question = "How did India revenue change compared to the previous quarter?"
    qv = embed(model, [question], is_query=True)
    scores, rows = index.search(qv, 3)
    print(f"\nTest question: {question}")
    for s, r in zip(scores[0], rows[0]):
        c = chunks[r]
        print(f"  {s:.3f}  {c['evidence_id']}  {c['text'][:90]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())