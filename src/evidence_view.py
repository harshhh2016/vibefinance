"""Resolve evidence ids to displayable passages for the dashboard.

A thin wrapper around the scorer's EvidenceStore, so the dashboard and the
scorer can never disagree about what a citation resolves to. Returns full
passage text (no truncation). Never raises on an unknown or malformed id.
"""
from __future__ import annotations

from functools import lru_cache

from src.eval.scoring import EvidenceStore

PRICE_FIELDS = ("open", "high", "low", "close", "volume")


@lru_cache(maxsize=1)
def default_store() -> EvidenceStore:
    return EvidenceStore()


def resolve(evidence_id: str, store: EvidenceStore | None = None) -> dict:
    """Return a dict: evidence_id, found, kind ('doc'|'price'|'unknown'), title,
    date (ISO str or None), text, source_url, fields (price rows only)."""
    store = store or default_store()
    eid = str(evidence_id)
    missing = {"evidence_id": eid, "found": False, "kind": "unknown",
               "title": eid, "date": None, "text": "", "source_url": None,
               "fields": {}}
    try:
        hit = store.lookup(eid)
    except Exception:  # malformed id or unreadable file: show it as not found
        return missing
    if hit is None:
        return missing

    if hit["kind"] == "price":
        _, sym, day = eid.split(":")
        row = store._rows(sym).get(day, {})  # same row the store just resolved
        fields = {k: row[k] for k in PRICE_FIELDS if k in row}
        text = " | ".join(f"{k} {fields[k]}" for k in PRICE_FIELDS if k in fields)
        return {"evidence_id": eid, "found": True, "kind": "price",
                "title": f"{sym} daily price row, {day}", "date": hit["date"].isoformat(),
                "text": text, "source_url": None, "fields": fields}

    chunk = store.chunks.get(eid, {})
    page = chunk.get("page")
    title = str(chunk.get("doc_id", eid)) + (f", page {page}" if page is not None else "")
    return {"evidence_id": eid, "found": True, "kind": "doc", "title": title,
            "date": hit["date"].isoformat(), "text": hit["text"],
            "source_url": chunk.get("source_url") or None, "fields": {}}


def resolve_many(evidence_ids, store: EvidenceStore | None = None) -> list[dict]:
    """Resolve ids in order, dropping exact duplicates."""
    seen, out = set(), []
    for eid in evidence_ids or []:
        if eid in seen:
            continue
        seen.add(eid)
        out.append(resolve(eid, store))
    return out
