"""B0: one agent, one LLM call, no decomposition, no EFCE, no injection guard.

It sees the SAME retrieval function and price files as the multi-agent systems. It gets one price row
(the latest on or before as_of) and the top retrieved passages, labelled [PX] / [P1]..; it cites a label
and the code maps the label to the real evidence id (the same convenience the Filings agent gets).
Nothing it says is validated. If the model call raises, the record says failed with the error.
Not applicable to scenarios that inject faults or claims into the multi-agent pipeline (s07-s08, s10-s13).
"""
from __future__ import annotations

import csv
import time
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUERY = "revenue growth, operating margin, deal wins, outlook, risks"
N_PASSAGES = 5
PASSAGE_CHARS = 600
DIRECTIONS = {"positive", "negative", "neutral", "uncertain"}


def latest_price(symbol, as_of: date, price_dir=None):
    p = Path(price_dir) if price_dir else ROOT / "data" / "prices"
    f = p / (symbol + ".csv")
    if not f.exists():
        return None
    best = None
    with open(f, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["date"] <= as_of.isoformat() and (best is None or r["date"] > best["date"]):
                best = r
    if best is None:
        return None
    return ("px:{}:{}".format(symbol, best["date"]),
            "latest price row {}: open {} high {} low {} close {} volume {}".format(
                best["date"], best["open"], best["high"], best["low"], best["close"], best["volume"]))


def build_prompt(symbol, as_of, sources) -> str:
    body = "\n\n".join("[{}] {}".format(label, text) for label, text in sources) or "(no sources available)"
    return ("You are a financial analyst assistant. Using ONLY the sources below, write up to 4 short factual "
            "claims about {} as of {}. Return JSON only: "
            '{{"claims":[{{"text":"...","direction":"positive|negative|neutral|uncertain","source":"P1"}}]}}. '
            "Each claim must name one source label.\n\nSOURCES\n{}").format(symbol, as_of.isoformat(), body)


def parse_claims(data, label_to_eid: dict, symbol, as_of: date) -> list[dict]:
    raw = data.get("claims") if isinstance(data, dict) else None
    out = []
    for i, c in enumerate(raw if isinstance(raw, list) else [], 1):
        if not isinstance(c, dict) or not str(c.get("text", "")).strip():
            continue
        eid = label_to_eid.get(str(c.get("source", "")).strip().strip("[]"))
        d = str(c.get("direction", "")).lower()
        out.append({"claim_id": "b{}".format(i), "entity": symbol, "metric_or_event": "single_agent_claim",
                    "period": as_of.isoformat(), "direction": d if d in DIRECTIONS else "uncertain",
                    "claim_type": "factual", "text": str(c["text"]).strip(),
                    "evidence_ids": [eid] if eid else [], "confidence": 0.5})
        if len(out) == 4:
            break
    return out


def run_b0(sc, ctx, llm_fn=None) -> dict:
    t0 = time.perf_counter()
    as_of = ctx.as_of
    sources, label_to_eid, error, claims, status, prompt = [], {}, None, [], "complete", None
    try:
        px = latest_price(ctx.symbol, as_of, ctx.price_dir)
        if px:
            sources.append(("PX", px[1]))
            label_to_eid["PX"] = px[0]
        for i, c in enumerate(list(ctx.retrieve_fn(QUERY, ctx.symbol, as_of, N_PASSAGES)), 1):   # keep everything retrieved
            sources.append(("P{}".format(i), c.text[:PASSAGE_CHARS]))
            label_to_eid["P{}".format(i)] = c.evidence_id
        if llm_fn is None:
            from src.agents.filings import ollama_json as llm_fn
        prompt = build_prompt(ctx.symbol, as_of, sources)
        claims = parse_claims(llm_fn(prompt), label_to_eid, ctx.symbol, as_of)
    except Exception as exc:
        status, error = "failed", "{}: {}".format(type(exc).__name__, str(exc)[:200])
    ms = (time.perf_counter() - t0) * 1000
    finding = {"agent": "single_agent", "status": status, "claims": claims, "latency_ms": round(ms, 1),
               "limitations": [error] if error else [], "error": error}
    return {"run_id": "{}-B0".format(sc["id"]), "created_at": datetime.now().isoformat(), "symbol": sc["symbol"],
            "profile_id": "unassigned", "as_of": sc["as_of"], "config": {"system": "B0"}, "signals": [],
            "findings": [finding],
            "efce": {"status": "complete" if claims else "failed", "supported": claims, "conflicts": [],
                     "abstained": [], "rejected": []},
            "metrics": {"end_to_end_ms": round(ms, 1), "agents_ms": round(ms, 1), "agents_complete": float(status == "complete")},
            "system": "B0", "scenario": sc["id"], "prompts": [prompt] if prompt else []}
