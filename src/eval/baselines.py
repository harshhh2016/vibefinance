"""Synthesis rules for the two multi-agent baselines. Pure functions of the agents' findings.

B1 "basic synthesis": every claim from every agent is presented as-is. No evidence validation,
   no freshness or look-ahead check, no conflict classification, no abstention.
B2 "simple vote": claims about the same (entity, metric_or_event, period) are put to a majority
   vote on direction; losing claims are dropped. Ties keep the higher-confidence claim (then the
   lower claim_id). No evidence validation, no conflict classification, no abstention.
P (the full system) = the same agents followed by run_efce.

All three see identical agent output, so any measured difference comes from the synthesis step.
Both baselines return an EFCEResult-shaped dict so the scorer treats every system alike.
"""
from __future__ import annotations

from collections import Counter


def _d(x) -> dict:
    return x.model_dump() if hasattr(x, "model_dump") else dict(x)


def _claims(findings) -> list[dict]:
    return [_d(c) for f in findings for c in (_d(f).get("claims") or [])]


def _result(supported: list[dict]) -> dict:
    # A basic synthesiser has one notion of status: it produced something, or it did not.
    return {"status": "complete" if supported else "failed", "supported": supported,
            "conflicts": [], "abstained": [], "rejected": []}


def synthesize_b1(findings) -> dict:
    return _result(_claims(findings))


def synthesize_b2(findings) -> dict:
    groups: dict[tuple, list[dict]] = {}
    for c in _claims(findings):
        groups.setdefault((c.get("entity"), c.get("metric_or_event"), c.get("period")), []).append(c)
    kept = []
    for g in groups.values():
        if len(g) == 1:
            kept.extend(g)
            continue
        votes = Counter(c.get("direction") for c in g)
        top = max(votes.values())
        winners = [d for d, n in votes.items() if n == top]
        cands = [c for c in g if c.get("direction") in winners]
        if len(winners) > 1:                      # tie: keep one claim, deterministically
            cands = [sorted(cands, key=lambda c: (-float(c.get("confidence") or 0), str(c.get("claim_id"))))[0]]
        kept.extend(cands)
    return _result(kept)
