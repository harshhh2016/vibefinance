"""EFCE: Evidence-Aware Financial Consensus Engine.

Deterministic rules that sit between the agents and the user:
validate evidence -> keep valid claims -> flag conflicts -> abstain when evidence is missing.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Optional, Sequence

from src.contracts import Conflict, EFCEResult, Finding, RejectedClaim
from src.efce.conflicts import REAL_CONFLICTS, find_conflicts, key_of
from src.efce.validate import Evidence, MAX_AGE_DAYS, build_registry, check_claim

__all__ = ["run_efce", "build_registry", "Evidence", "MAX_AGE_DAYS"]

EXPECTED_AGENTS = ("technical", "sentiment", "filings")


def run_efce(findings: Sequence[Finding], registry: dict[str, Evidence], as_of: date,
             expected_agents: Sequence[str] = EXPECTED_AGENTS,
             max_age_days: Optional[dict[str, int]] = None) -> EFCEResult:
    supported, rejected, abstained = [], [], []
    valid_keys, rejected_by_key, support_agents = set(), defaultdict(list), set()
    degraded = False

    by_agent = {f.agent: f for f in findings}
    for agent in expected_agents:
        f = by_agent.get(agent)
        if f is None:
            abstained.append(f"{agent}:missing"); degraded = True
        elif f.status == "failed":
            abstained.append(f"{agent}:failed"); degraded = True
        elif f.status == "degraded":
            degraded = True

    for f in findings:
        if f.status == "failed":
            continue                                   # a failed module contributes nothing
        for c in f.claims:
            reason = check_claim(c, registry, as_of, max_age_days)
            if reason:
                rejected.append(RejectedClaim(claim_id=c.claim_id, reason=reason))
                rejected_by_key[key_of(c)].append(c.claim_id)
            else:
                supported.append(c)
                valid_keys.add(key_of(c))
                support_agents.add(f.agent)

    conflicts = find_conflicts(supported)
    for key, ids in rejected_by_key.items():
        if key not in valid_keys:                      # every claim on this metric failed validation
            conflicts.append(Conflict(kind="insufficient_evidence", claim_ids=ids,
                                      explanation=f"{key[0]} {key[1]}: no claim passed evidence checks."))
            abstained.append(key[1])

    if findings and all(f.status == "failed" for f in findings):
        status = "failed"
    elif not supported:
        status = "abstained_or_limited"
    elif any(c.kind in REAL_CONFLICTS for c in conflicts):
        status = "conflict_unresolved"
    elif len(support_agents) < 2:
        status = "abstained_or_limited"
    elif degraded or rejected:
        status = "degraded"
    else:
        status = "complete"

    return EFCEResult(status=status, supported=supported, conflicts=conflicts,
                      abstained=list(dict.fromkeys(abstained)), rejected=rejected)
