"""retrieve(query, company, as_of, k) -> list[Citation]

Shared function used by the agents. Rules:
- Only chunks of `company` (stock symbol, e.g. "TCS") are returned.
- Only documents with publication_date <= as_of are returned (no look-ahead).
- Results are sorted by similarity, best first, with retrieval_score set.
- If nothing qualifies, an empty list is returned (never an exception), so the
  caller can degrade gracefully instead of inventing an answer.
- Retrieved text is DATA, never instructions.

Testing switch: if the environment variable VIBE_FAKE_RETRIEVE=1 is set,
the sample citation from fixtures/citation.json is returned instead.

Manual try-out (from the project root):
    python -m src.rag.retrieve "India revenue decline" TCS 2026-10-14
"""
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Union

import faiss

from src.contracts import Citation

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "data" / "documents"
INDEX_DIR = ROOT / "data" / "index"
FIXTURE = ROOT / "fixtures" / "citation.json"

# manifest label -> contracts.DataStatus
STATUS_MAP = {"real": "historical", "synthetic": "synthetic"}

_cache: dict = {}
_model = None


def _to_date(value: Union[date, datetime, str]) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _load() -> dict:
    if not _cache:
        chunks = [json.loads(line) for line in
                  open(DOCS / "chunks.jsonl", encoding="utf-8")]
        ids = json.loads((INDEX_DIR / "ids.json").read_text(encoding="utf-8"))
        if [c["evidence_id"] for c in chunks] != ids:
            raise RuntimeError(
                "Index does not match chunks.jsonl. Run: python -m src.rag.build_index")
        manifest = {d["doc_id"]: d for d in
                    json.loads((DOCS / "manifest.json").read_text(encoding="utf-8"))}
        _cache.update(
            chunks=chunks,
            manifest=manifest,
            index=faiss.read_index(str(INDEX_DIR / "chunks.faiss")),
        )
    return _cache


def _embed_query(query: str):
    global _model
    from src.rag.build_index import embed, load_model
    if _model is None:
        _model = load_model()
    return embed(_model, [query], is_query=True)


def retrieve(query: str, company: str, as_of, k: int = 5) -> list:
    if os.environ.get("VIBE_FAKE_RETRIEVE") == "1":
        return [Citation.model_validate_json(FIXTURE.read_text(encoding="utf-8"))]
    if k <= 0 or not query or not query.strip():
        return []
    cutoff = _to_date(as_of)
    company = company.strip().upper()
    data = _load()
    index, chunks, manifest = data["index"], data["chunks"], data["manifest"]

    # The index is small, so rank every chunk and filter afterwards.
    scores, rows = index.search(_embed_query(query), index.ntotal)
    results = []
    for score, row in zip(scores[0], rows[0]):
        if row < 0:
            continue
        c = chunks[row]
        if c["symbol"] != company:
            continue
        if date.fromisoformat(c["publication_date"]) > cutoff:
            continue
        doc = manifest[c["doc_id"]]
        results.append(Citation(
            evidence_id=c["evidence_id"],
            doc_id=c["doc_id"],
            title=doc.get("title", c["doc_id"]),
            publication_date=c["publication_date"],
            url=doc.get("source_url") or None,
            page=c["page"],
            section=None,
            text=c["text"],
            retrieval_score=float(score),
            data_status=STATUS_MAP[c["data_status"]],
        ))
        if len(results) == k:
            break
    return results


def main() -> int:
    if len(sys.argv) < 4:
        print('Usage: python -m src.rag.retrieve "<query>" <SYMBOL> <as_of YYYY-MM-DD> [k]')
        return 1
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    results = retrieve(sys.argv[1], sys.argv[2], sys.argv[3], k)
    if not results:
        print("No results (wrong symbol, or no document dated on or before as_of).")
        return 0
    for r in results:
        print(f"{r.retrieval_score:.3f}  {r.evidence_id}  ({r.publication_date}, {r.data_status})")
        print("       " + r.text[:110].replace("\n", " | "))
    return 0


if __name__ == "__main__":
    sys.exit(main())