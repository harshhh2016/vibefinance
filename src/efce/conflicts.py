"""EFCE step 2: find opposing claims and say what kind of disagreement it is."""
from __future__ import annotations

from collections import defaultdict
from itertools import product

from src.contracts import Claim, Conflict

REAL_CONFLICTS = ("factual_contradiction", "analytical_disagreement")   # these block a clean result


def key_of(c: Claim) -> tuple[str, str]:
    return (c.entity.upper(), c.metric_or_event.lower())


def find_conflicts(claims: list[Claim]) -> list[Conflict]:
    """Positive vs negative claims about the same entity and metric.

    different periods            -> temporal_difference (not a real contradiction)
    same period, both factual    -> factual_contradiction
    same period, any non-factual -> analytical_disagreement
    Both sides are always kept; nothing here picks a winner.
    """
    groups: dict[tuple[str, str], list[Claim]] = defaultdict(list)
    for c in claims:
        groups[key_of(c)].append(c)

    found: dict[tuple, dict] = {}
    for (entity, metric), cs in groups.items():
        pos = [c for c in cs if c.direction == "positive"]
        neg = [c for c in cs if c.direction == "negative"]
        for p, n in product(pos, neg):
            if p.period != n.period:
                kind = "temporal_difference"
            elif p.claim_type == "factual" and n.claim_type == "factual":
                kind = "factual_contradiction"
            else:
                kind = "analytical_disagreement"
            slot = found.setdefault((entity, metric, kind), {"ids": set(), "periods": set()})
            slot["ids"].update([p.claim_id, n.claim_id])
            slot["periods"].update([p.period, n.period])

    out = []
    for (entity, metric, kind), slot in found.items():
        periods = ", ".join(sorted(slot["periods"]))
        what = {"temporal_difference": "opposite directions in different periods",
                "factual_contradiction": "opposite factual claims for the same period",
                "analytical_disagreement": "opposite readings of the same period"}[kind]
        out.append(Conflict(kind=kind, claim_ids=sorted(slot["ids"]),
                            explanation=f"{entity} {metric}: {what} ({periods})."))
    return out
