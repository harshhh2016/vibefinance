"""EFCE step 1: check each claim's evidence with plain rules (no LLM)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional

from src.contracts import Citation, Claim, Signal

# Evidence older than this many days (before as_of) counts as stale. Keyed by the id prefix.
MAX_AGE_DAYS = {"px": 7, "doc": 540}


@dataclass(frozen=True)
class Evidence:
    published: date
    data_status: str = "historical"


def build_registry(citations: Iterable[Citation], signals: Iterable[Signal] = ()) -> dict[str, Evidence]:
    """Map every known evidence_id to its source date. Claims citing any other id are rejected."""
    reg = {c.evidence_id: Evidence(c.publication_date, c.data_status) for c in citations}
    for s in signals:
        for eid in s.evidence_ids:
            if eid not in reg and eid.startswith("px:"):
                try:
                    reg[eid] = Evidence(date.fromisoformat(eid.rsplit(":", 1)[1]), s.data_status)
                except ValueError:
                    pass                      # malformed id stays unknown
    return reg


def check_claim(claim: Claim, registry: dict[str, Evidence], as_of: date,
                max_age: Optional[dict[str, int]] = None) -> Optional[str]:
    """Return None if the claim is acceptable, otherwise the reason it is rejected."""
    limits = {**MAX_AGE_DAYS, **(max_age or {})}
    if not claim.evidence_ids:
        return None if claim.claim_type == "analytical" else "missing_evidence"
    for eid in claim.evidence_ids:
        ev = registry.get(eid)
        if ev is None:
            return "unknown_evidence_id"
        if ev.published > as_of:
            return "source_after_as_of"
        limit = limits.get(eid.split(":", 1)[0], limits["doc"])
        if ev.data_status == "stale" or (as_of - ev.published).days > limit:
            return "stale_source"
    return None
