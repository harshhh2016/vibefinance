"""Retrieval experiments: BM25 keyword search, hybrid (RRF) and a per-document cap.

Reuses the loading, embedding and Citation code from src.rag.retrieve, so nothing
in the existing retrieve() is changed. Used by src.rag.eval_retrieval.
"""
import math
import re
from collections import Counter
from datetime import date

from src.rag.retrieve import Citation, STATUS_MAP, _embed_query, _load, _to_date

TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[a-z]+")
STOP = set("""a an the of in to and for is are was were be by on at from this that with does do did has
have had it its as or what how many much which when who why whom their there during over into""".split())
K1, B, RRF_K = 1.5, 0.75, 60
_bm25 = {}


def tokenize(text):
    return TOKEN.findall(text.lower())


def _stats(chunks):
    if _bm25.get("src") is not chunks:
        docs = [Counter(tokenize(c["text"])) for c in chunks]
        lens = [sum(d.values()) for d in docs]
        df = Counter()
        for d in docs:
            df.update(d.keys())
        _bm25.update(src=chunks, docs=docs, lens=lens, df=df,
                     n=len(chunks), avg=sum(lens) / max(1, len(chunks)))
    return _bm25


def _bm25_rank(query, chunks, ok):
    s = _stats(chunks)
    terms = {t for t in tokenize(query) if t not in STOP}
    scored = []
    for row in ok:
        tf, dl, total = s["docs"][row], s["lens"][row], 0.0
        for t in terms:
            f = tf.get(t, 0)
            if not f:
                continue
            idf = math.log(1 + (s["n"] - s["df"][t] + 0.5) / (s["df"][t] + 0.5))
            total += idf * f * (K1 + 1) / (f + K1 * (1 - B + B * dl / s["avg"]))
        if total > 0:
            scored.append((row, total))
    scored.sort(key=lambda x: -x[1])
    return scored


def _embed_rank(query, index, ok):
    scores, rows = index.search(_embed_query(query), index.ntotal)
    return [(int(r), float(sc)) for sc, r in zip(scores[0], rows[0]) if r >= 0 and int(r) in ok]


def _rrf(*rankings):
    fused = {}
    for ranking in rankings:
        for pos, (row, _) in enumerate(ranking, 1):
            fused[row] = fused.get(row, 0.0) + 1.0 / (RRF_K + pos)
    return sorted(fused.items(), key=lambda x: -x[1])


def run(mode, query, company, as_of, k=5, cap=None):
    """mode: 'embed' (same ranking as retrieve), 'bm25' or 'hybrid'.
    cap: max chunks from any one document (None = no cap)."""
    if k <= 0 or not query or not query.strip():
        return []
    cutoff, company = _to_date(as_of), company.strip().upper()
    data = _load()
    chunks, manifest, index = data["chunks"], data["manifest"], data["index"]
    ok = {i for i, c in enumerate(chunks)
          if c["symbol"] == company and date.fromisoformat(c["publication_date"]) <= cutoff}
    if mode == "embed":
        ranked = _embed_rank(query, index, ok)
    elif mode == "bm25":
        ranked = _bm25_rank(query, chunks, ok)
    elif mode == "hybrid":
        ranked = _rrf(_embed_rank(query, index, ok), _bm25_rank(query, chunks, ok))
    else:
        raise ValueError(f"unknown mode {mode!r}")
    out, per_doc = [], Counter()
    for row, score in ranked:
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
