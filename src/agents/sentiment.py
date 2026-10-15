"""Sentiment agent: word-list sentiment over retrieved passages, cited by passage id."""
from __future__ import annotations

from src.agents.base import AgentContext, AgentResult, get_retrieve
from src.contracts import Claim, Finding, Signal
from src.signals.sentiment import compute_sentiment

QUERIES = [
    "management outlook and demand environment",
    "risks headwinds and challenges",
    "growth momentum and strong performance",
]


def build_finding(sig: Signal) -> Finding:
    if sig.status == "failed" or sig.label == "unavailable" or not sig.evidence_ids:
        why = "no citable text available for sentiment ({})".format(sig.status)
        return Finding(agent="sentiment", status="failed", error=why, limitations=[why])
    direction = sig.label if sig.label in ("positive", "negative", "neutral") else "uncertain"
    score = "" if sig.value is None else " (score {:+.2f})".format(sig.value)
    text = "Word-list sentiment over {} retrieved passages is {}{}.".format(
        len(sig.evidence_ids), sig.label, score)
    claim = Claim(
        claim_id="s1", entity=sig.symbol, metric_or_event="text_sentiment",
        period=sig.as_of.isoformat(), direction=direction, claim_type="inferred",
        text=text, evidence_ids=list(sig.evidence_ids), confidence=sig.confidence,
    )
    limits = [] if sig.status == "complete" else ["sentiment signal is {}".format(sig.status)]
    return Finding(agent="sentiment", status="complete" if not limits else "degraded",
                   claims=[claim], limitations=limits)


def run(ctx: AgentContext) -> AgentResult:
    retrieve = get_retrieve(ctx)
    seen = {}
    for q in QUERIES:
        for c in retrieve(q, ctx.symbol, ctx.as_of, ctx.top_k):
            seen.setdefault(c.evidence_id, c)
    citations = list(seen.values())
    texts = [(c.evidence_id, c.text, c.publication_date) for c in citations]
    sig = compute_sentiment(ctx.symbol, texts, ctx.as_of)
    return AgentResult(build_finding(sig), signals=[sig], citations=citations)
