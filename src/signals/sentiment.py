"""Sentiment baseline: a simple word-list score over text items dated on or before as_of.

This is a transparent rules baseline. You can swap in a model later (for example FinBERT)
behind the same function without changing anything else.

Method (fixed before any evaluation):
  score = (positive_hits - negative_hits) / (positive_hits + negative_hits)
  positive if score >= +0.2, negative if score <= -0.2, neutral otherwise,
  uncertain if no sentiment words were found at all.
"""
import re
from datetime import date

from src.contracts import Signal

POSITIVE = {
    "growth", "grew", "increase", "increased", "profit", "profits", "record", "strong", "improved",
    "improvement", "beat", "beats", "surge", "gain", "gains", "upgrade", "upgraded", "robust",
    "expansion", "outperform", "outperformed", "healthy", "rise", "rose", "positive", "win", "won",
}
NEGATIVE = {
    "loss", "losses", "decline", "declined", "decrease", "decreased", "weak", "fall", "fell",
    "drop", "dropped", "miss", "missed", "downgrade", "downgraded", "debt", "default", "risk",
    "risks", "fraud", "probe", "penalty", "lawsuit", "slump", "negative", "concern", "concerns",
    "impairment", "shortfall",
}
THRESHOLD = 0.2


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def compute_sentiment(symbol: str, items: list[tuple[str, str, date]], as_of: date,
                      threshold: float = THRESHOLD) -> Signal:
    """items is a list of (evidence_id, text, published_date). Later-dated items are ignored."""
    method = f"word-list sentiment score, thresholds +/-{threshold}"
    usable = [(eid, txt, d) for eid, txt, d in items if d <= as_of]
    if not usable:
        return Signal(symbol=symbol, as_of=as_of, name="sentiment", value=None, unit="score",
                      label="unavailable", confidence=0.0, method=method,
                      data_status="unavailable", status="failed")
    pos = neg = 0
    for _, txt, _ in usable:
        for w in _words(txt):
            pos += w in POSITIVE
            neg += w in NEGATIVE
    ids = [eid for eid, _, _ in usable]
    first, last = min(d for _, _, d in usable), max(d for _, _, d in usable)
    if pos + neg == 0:
        return Signal(symbol=symbol, as_of=as_of, name="sentiment", value=0.0, unit="score",
                      label="uncertain", confidence=0.3, method=method, window_start=first,
                      window_end=last, evidence_ids=ids)
    score = (pos - neg) / (pos + neg)
    label = "positive" if score >= threshold else "negative" if score <= -threshold else "neutral"
    # heuristic: more matched words and a stronger score give more confidence, capped
    confidence = min(0.8, 0.4 + 0.05 * min(pos + neg, 8) * (0.5 + abs(score) / 2))
    return Signal(symbol=symbol, as_of=as_of, name="sentiment", value=round(score, 4), unit="score",
                  label=label, confidence=round(confidence, 3), method=method,
                  window_start=first, window_end=last, evidence_ids=ids)
