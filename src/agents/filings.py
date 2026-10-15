"""Filings agent.

Primary path: the local Ollama model writes short claims from the retrieved passages.
Every claim is checked: its evidence_id must be one of the passages it was shown, and every
number in it must appear in the text it was shown for that passage. Anything else is dropped.
Fallback (Ollama down, or nothing verifiable): quote a line from the stored passage verbatim.
"""
from __future__ import annotations

import json
import re
import sys

from src.agents.base import AgentContext, AgentResult, get_retrieve
from src.contracts import Claim, Finding

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:latest"
PASSAGE_CHARS = 600
MAX_PASSAGES = 5
MAX_PROMPT_CHARS = 4500

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


def focus(text: str, keywords, n: int = PASSAGE_CHARS) -> str:
    """Cut a long passage to n characters, starting near the first line that has a keyword
    (preferring a line that also has a digit)."""
    if len(text) <= n:
        return text
    offset, best = 0, None
    for ln in text.splitlines(keepends=True):
        low = ln.lower()
        if any(k in low for k in keywords):
            if any(ch.isdigit() for ch in ln):
                best = offset
                break
            if best is None:
                best = offset
        offset += len(ln)
    start = max(0, (best or 0) - 100)
    return text[start:start + n]


def select_passages(hits: dict):
    """One passage per topic (the top hit), each trimmed. Returns (passages, shown)."""
    passages, shown = {}, {}
    for metric, _query, kw in TOPICS:
        for c in hits.get(metric, [])[:1]:
            if c.evidence_id not in passages and len(passages) < MAX_PASSAGES:
                passages[c.evidence_id] = c
                shown[c.evidence_id] = focus(c.text, kw)
    return passages, shown


def select_topic_passages(hits: dict):
    # top hit per topic; returns (passages by evidence id, shown text by id, metric -> evidence id)
    passages, shown, topic_eids = {}, {}, {}
    for metric, _query, kw in TOPICS:
        for c in hits.get(metric, [])[:1]:
            passages[c.evidence_id] = c
            shown.setdefault(c.evidence_id, focus(c.text, kw))
            topic_eids[metric] = c.evidence_id
    return passages, shown, topic_eids


def assign_evidence(data, topic_eids: dict):
    # the model does not pick sources: unless it cited a shown id, the id comes from the metric
    items = data.get("claims") if isinstance(data, dict) else None
    if isinstance(items, list):
        known = set(topic_eids.values())
        for it in items:
            if isinstance(it, dict) and it.get("evidence_id") not in known:
                it["evidence_id"] = topic_eids.get(it.get("metric"), "")
    return data


def build_prompt(symbol: str, passages: dict, shown: dict | None = None, topic_eids: dict | None = None) -> str:
    shown = shown or {}
    if topic_eids:
        blocks = "\n\n".join(
            "[{}]\n{}".format(m, shown.get(e, passages[e].text[:PASSAGE_CHARS]))
            for m, e in topic_eids.items() if e in passages
        )
        return (
            "You extract facts about " + symbol + " from the labelled passages below. "
            "Each passage belongs to one metric. For a metric, use ONLY the passage under its label.\n"
            "Rules: copy every number exactly as written; do not calculate; do not guess; "
            "one claim per metric; skip a metric if its passage does not state it.\n"
            "Write each claim as one full sentence that names the metric and includes the number.\n"
            'Reply with JSON only: {"claims": [{"metric": "<label>", "text": "<sentence>"}]}\n\n'
            "PASSAGES:\n" + blocks + "\n"
        )
    blocks = "\n\n".join(
        "[{}]\n{}".format(eid, shown.get(eid, c.text[:PASSAGE_CHARS])) for eid, c in passages.items()
    )
    return (
        "You extract facts about " + symbol + " from the passages below. Use ONLY these passages.\n"
        "Rules: copy every number exactly as written; do not calculate; do not guess; "
        "at most one claim per metric; skip a metric if the passages do not state it.\n"
        "Allowed metrics: " + ", ".join(METRICS) + ".\n"
        'Reply with JSON only: {"claims": [{"metric": "...", "text": "one short sentence", '
        '"evidence_id": "the id in square brackets above the passage"}]}\n\n'
        "PASSAGES:\n" + blocks + "\n"
    )


def ollama_json(prompt: str, timeout: float = 180.0) -> dict:
    import requests
    r = requests.post(OLLAMA_URL, json={
        "model": MODEL_NAME, "prompt": prompt, "stream": False, "format": "json",
        "keep_alive": "30m",
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 400},
    }, timeout=timeout)
    r.raise_for_status()
    d = r.json()
    print("[ollama] load {:.1f}s | prompt {} tok in {:.1f}s | gen {} tok in {:.1f}s".format(
        d.get("load_duration", 0) / 1e9,
        d.get("prompt_eval_count", 0), d.get("prompt_eval_duration", 0) / 1e9,
        d.get("eval_count", 0), d.get("eval_duration", 0) / 1e9), file=sys.stderr)
    return json.loads(d["response"])


_NUM = re.compile(r"(?<![A-Za-z\d])\d[\d,]*\.?\d*")


def _squash(s: str) -> str:
    return re.sub(r"[\s,]", "", s)


def _norm(s: str) -> str:
    # drop thousands separators only (1,234 -> 1234); keep spaces and decimal points
    return re.sub(r"(?<=\d),(?=\d)", "", s)


def _has_number(n: str, hay: str) -> bool:
    # whole-number match: "24" must not match inside "2024" or "24.5"
    return re.search(r"(?<![\d.])" + re.escape(n) + r"(?!\d|\.\d)", hay) is not None


_PERIOD_TERMS = ("year on year", "yoy", "qoq", "quarter on quarter", "sequential", "constant currency")


def _periods_supported(text: str, shown_text: str) -> bool:
    # a period word in the claim must also be in the text the model was shown
    t = text.lower().replace("-", " ")
    h = shown_text.lower().replace("-", " ")
    return all(term in h for term in _PERIOD_TERMS if term in t)


def validate_llm_claims(data, passages: dict, shown: dict | None = None):
    """Keep only claims whose evidence id was shown and whose numbers appear in the text shown."""
    shown = shown or {}
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
        hay = _norm(shown.get(eid, cit.text[:PASSAGE_CHARS]))
        nums = [_norm(n).rstrip(".,") for n in _NUM.findall(text)]
        if not nums or not all(_has_number(n, hay) for n in nums):
            dropped += 1
            continue
        if not _periods_supported(text, shown.get(eid, cit.text[:PASSAGE_CHARS])):
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


_INJECTION = re.compile(
    r"ignore (all |any |the )?(previous|prior|above|earlier)|disregard (all |any |the )?(previous|prior|above|instructions)"
    r"|new instructions|you are now|system prompt|do not follow|override (the )?(rules|instructions)"
    r"|(?:(?:^|[.!?:;\n])\s*|\b(?:then|just|please|now)\s+)(?:reply|respond|answer|output|report)\s+(?:only\s+)?with\b"
    r"|instead,? (say|report|output|reply)",
    re.IGNORECASE,
)


def looks_like_instruction(text: str) -> bool:
    return _INJECTION.search(text or "") is not None


def drop_injected(hits: dict):
    """Remove passages that read like instructions to the model. Returns (clean hits, flagged ids)."""
    clean, flagged = {}, set()
    for metric, cits in hits.items():
        keep = []
        for c in cits:
            if looks_like_instruction(c.text):
                flagged.add(c.evidence_id)
            else:
                keep.append(c)
        clean[metric] = keep
    return clean, sorted(flagged)


def _note_flagged(finding, flagged):
    if flagged:
        finding.limitations.append(
            "dropped {} passage(s) containing instruction-like text: {}".format(len(flagged), ", ".join(flagged)))
    return finding


def run(ctx: AgentContext) -> AgentResult:
    hits, _ = gather(ctx)
    hits, flagged = drop_injected(hits)
    passages, shown, topic_eids = select_topic_passages(hits)
    cits = list({c.evidence_id: c for found in hits.values() for c in found}.values())
    if ctx.use_llm and passages:
        try:
            llm = ctx.llm_fn or ollama_json
            data = llm(build_prompt(ctx.symbol, passages, shown, topic_eids))
            data = assign_evidence(data, topic_eids)
            good, dropped = validate_llm_claims(data, passages, shown)
            if good:
                f = _note_flagged(build_llm_finding(ctx.symbol, good, dropped), flagged)
                return AgentResult(f, citations=cits)
            reason = "model produced no verifiable claims"
        except Exception as exc:
            reason = "model unavailable ({})".format(type(exc).__name__)
        fb = build_finding(ctx.symbol, hits)
        fb.limitations.append("fell back to verbatim quotes: " + reason)
        return AgentResult(_note_flagged(fb, flagged), citations=cits)
    return AgentResult(_note_flagged(build_finding(ctx.symbol, hits), flagged), citations=cits)
