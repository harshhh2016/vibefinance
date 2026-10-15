"""Filings agent.

Primary path: the local Ollama model writes short claims from the retrieved passages.
Every claim is checked: its evidence_id must be one of the passages it was shown, and every
number in it must appear in that passage. Anything else is dropped.
Fallback (Ollama down, or nothing verifiable): quote a line from the stored passage verbatim.
"""
from __future__ import annotations

import json
import re

from src.agents.base import AgentContext, AgentResult, get_retrieve
from src.contracts import Claim, Finding

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:latest"
PASSAGE_CHARS = 1500
MAX_PASSAGES = 6

# (metric_or_event, retrieval query, keywords that must appear in the quoted line)
TOPICS = [
    ("operating_margin", "operating margin", ("operating margin", "ebit")),
    ("revenue", "revenue growth", ("revenue",)),
    ("total_contract_value", "total contract value TCV deal wins", ("tcv", "contract value")),
    ("attrition", "attrition rate employees", ("attrition",)),
    ("dividend", "dividend per share", ("dividend",)),
]
METRICS = [t[0] for t in TOPICS]
_RERANKER = None


# ---------- extractive fallback ----------
def excerpt(text: str, keywords) -> str | None:
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= 20]
    hits = [ln for ln in lines if any(k in ln.lower() for k in keywords)]
    with_digit = [ln for ln in hits if any(ch.isdigit() for ch in ln)]
    best = (with_digit or hits or [None])[0]
    return best[:300] if best else None


def build_finding(symbol: str, hits: dict) -> Finding:
    claims, missing = [], []
    for i, (metric, _query, keywords) in enumerate(TOPICS, start=1):
        made = False
        for c in hits.get(metric, []):
            line = excerpt(c.text, keywords)
            if line is None:
                continue
            page = "p.{}".format(c.page) if c.page is not None else "page n/a"
            score = c.retrieval_score if c.retrieval_score is not None else 0.5
            claims.append(Claim(
                claim_id="f{}".format(i), entity=symbol, metric_or_event=metric,
                period="reported_" + c.publication_date.isoformat(),
                direction="neutral", claim_type="factual",
                text='"{}" ({}, {})'.format(line, c.doc_id, page),
                evidence_ids=[c.evidence_id],
                confidence=max(0.0, min(0.9, float(score))),
            ))
            made = True
            break
        if not made:
            missing.append(metric)
    if not claims:
        why = "no relevant passages retrieved for any topic"
        return Finding(agent="filings", status="failed", error=why, limitations=[why])
    limits = ["no supporting passage found for: " + ", ".join(missing)] if missing else []
    limits.append("quotes stored passages verbatim, no model interpretation")
    return Finding(agent="filings", status="complete", claims=claims, limitations=limits)


# ---------- model path ----------
def _rerank(query, cits, k):
    global _RERANKER
    if len(cits) <= k:
        return list(cits)
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder
        _RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    scores = _RERANKER.predict([(query, c.text) for c in cits])
    order = sorted(range(len(cits)), key=lambda i: float(scores[i]), reverse=True)
    return [cits[i] for i in order[:k]]


def gather(ctx: AgentContext):
    retrieve = get_retrieve(ctx)
    hits, passages = {}, {}
    for metric, query, _kw in TOPICS:
        found = list(retrieve(query, ctx.symbol, ctx.as_of, 20 if ctx.rerank else ctx.top_k))
        found = _rerank(query, found, ctx.top_k) if ctx.rerank else found[:ctx.top_k]
        hits[metric] = found
        for c in found[:2]:
            if len(passages) < MAX_PASSAGES:
                passages.setdefault(c.evidence_id, c)
    return hits, passages


def build_prompt(symbol: str, passages: dict) -> str:
    blocks = "\n\n".join("[{}]\n{}".format(eid, c.text[:PASSAGE_CHARS]) for eid, c in passages.items())
    return (
        "You extract facts about " + symbol + " from the passages below. Use ONLY these passages.\n"
        "Rules: copy every number exactly as written; do not calculate; do not guess; "
        "at most one claim per metric; skip a metric if the passages do not state it.\n"
        "Allowed metrics: " + ", ".join(METRICS) + ".\n"
        'Reply with JSON only: {"claims": [{"metric": "...", "text": "one short sentence", '
        '"evidence_id": "the id in square brackets above the passage"}]}\n\n'
        "PASSAGES:\n" + blocks + "\n"
    )


def ollama_json(prompt: str, timeout: float = 90.0) -> dict:
    import requests
    r = requests.post(OLLAMA_URL, json={
        "model": MODEL_NAME, "prompt": prompt, "stream": False, "format": "json",
        "options": {"temperature": 0, "num_ctx": 8192},
    }, timeout=timeout)
    r.raise_for_status()
    return json.loads(r.json()["response"])


_NUM = re.compile(r"\d[\d,]*\.?\d*")


def _squash(s: str) -> str:
    return re.sub(r"[\s,]", "", s)


def validate_llm_claims(data, passages: dict):
    """Keep only claims whose evidence id was shown and whose numbers appear in that passage."""
    items = data.get("claims") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return [], 0
    good, dropped, seen = [], 0, set()
    for it in items:
        try:
            metric, text, eid = it["metric"], it["text"].strip(), it["evidence_id"]
        except (KeyError, TypeError, AttributeError):
            dropped += 1
            continue
        cit = passages.get(eid)
        if metric not in METRICS or metric in seen or not text or cit is None:
            dropped += 1
            continue
        hay = _squash(cit.text[:PASSAGE_CHARS])
        nums = [_squash(n).rstrip(".") for n in _NUM.findall(text)]
        if not nums or any(n not in hay for n in nums):
            dropped += 1
            continue
        seen.add(metric)
        good.append((metric, text[:300], cit))
    return good, dropped


def build_llm_finding(symbol: str, good: list, dropped: int) -> Finding:
    claims = []
    for i, (metric, text, c) in enumerate(good, start=1):
        page = "p.{}".format(c.page) if c.page is not None else "page n/a"
        score = c.retrieval_score if c.retrieval_score is not None else 0.5
        claims.append(Claim(
            claim_id="f{}".format(i), entity=symbol, metric_or_event=metric,
            period="reported_" + c.publication_date.isoformat(),
            direction="neutral", claim_type="factual",
            text="{} ({}, {})".format(text, c.doc_id, page),
            evidence_ids=[c.evidence_id], confidence=max(0.0, min(0.9, float(score))),
        ))
    limits = ["claims written by {}; each figure checked against its cited passage".format(MODEL_NAME)]
    if dropped:
        limits.append("{} model claim(s) dropped: unknown source or figure not in cited passage".format(dropped))
    got = {m for m, _t, _c in good}
    missing = [m for m in METRICS if m not in got]
    if missing:
        limits.append("no verified claim for: " + ", ".join(missing))
    return Finding(agent="filings", status="complete", claims=claims, limitations=limits)


def run(ctx: AgentContext) -> AgentResult:
    hits, passages = gather(ctx)
    cits = list({c.evidence_id: c for found in hits.values() for c in found}.values())
    if ctx.use_llm and passages:
        try:
            llm = ctx.llm_fn or ollama_json
            good, dropped = validate_llm_claims(llm(build_prompt(ctx.symbol, passages)), passages)
            if good:
                return AgentResult(build_llm_finding(ctx.symbol, good, dropped), citations=cits)
            reason = "model produced no verifiable claims"
        except Exception as exc:
            reason = "model unavailable ({})".format(type(exc).__name__)
        fb = build_finding(ctx.symbol, hits)
        fb.limitations.append("fell back to verbatim quotes: " + reason)
        return AgentResult(fb, citations=cits)
    return AgentResult(build_finding(ctx.symbol, hits), citations=cits)
