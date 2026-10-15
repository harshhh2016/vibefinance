from collections import Counter
from datetime import date

from src.rag.retrieve import Citation, STATUS_MAP, _load, _to_date
from src.rag.retrieve_variants import _bm25_rank, _embed_rank, _rrf

_reranker = None


def run_rerank(query, company, as_of, k=5, cap=None, pool=50):
    """Collect candidates from BOTH embeddings and BM25, then rerank them with a cross-encoder."""
    global _reranker
    if k <= 0 or not query or not query.strip():
        return []
    cutoff, company = _to_date(as_of), company.strip().upper()
    data = _load()
    chunks, manifest, index = data["chunks"], data["manifest"], data["index"]
    ok = {i for i, c in enumerate(chunks)
          if c["symbol"] == company and date.fromisoformat(c["publication_date"]) <= cutoff}
    emb = _embed_rank(query, index, ok)[:pool]
    kw = _bm25_rank(query, chunks, ok)[:pool]
    rows = [row for row, _ in _rrf(emb, kw)]
    if not rows:
        return []
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    scores = _reranker.predict([(query, chunks[r]["text"]) for r in rows])
    ranked = sorted(zip(scores, rows), key=lambda x: -float(x[0]))
    out, per_doc = [], Counter()
    for score, row in ranked:
        c = chunks[row]
        if cap is not None and per_doc[c["doc_id"]] >= cap:
            continue
        per_doc[c["doc_id"]] += 1
        doc = manifest[c["doc_id"]]
        out.append(Citation(
            evidence_id=c["evidence_id"], doc_id=c["doc_id"],
            title=doc.get("title", c["doc_id"]), publication_date=c["publication_date"],
            url=doc.get("source_url") or None, page=c["page"], section=None,
            text=c["text"], retrieval_score=float(score),
            data_status=STATUS_MAP[c["data_status"]]))
        if len(out) == k:
            break
    return out
