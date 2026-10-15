"""Scenario: TCS India revenue from three sources, built from stored chunks by exact text match."""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

from src.contracts import Citation, Claim, Finding
from src.efce import build_registry, run_efce

ROOT = Path(__file__).resolve().parents[2]
CHUNKS = ROOT / "data" / "documents" / "chunks.jsonl"
PR, TR = "TCS-PR-Q2FY27", "TCS-TR-Q2FY27"
METRIC = "india_revenue_growth"
ROW = re.compile(r"\bIndia\s+(\d+\.\d)\s+(\d+\.\d)\s+(\d+\.\d)\s+(-\s*)?(\d+\.\d)\s+(-\s*)?(\d+\.\d)")
CEO_QUOTE = "India revenue grew 6%"
ASK_QUOTE = "India revenue decline by about 10%"


def load_chunks(path=CHUNKS) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _norm(t: str) -> str:
    return " ".join(t.split())


def _first(chunks, doc_id, test):
    for c in chunks:
        if c["doc_id"] == doc_id:
            m = test(_norm(c["text"]))
            if m:
                return c, m
    raise LookupError("text not found in {}".format(doc_id))


_ALLOWED = ("historical", "delayed", "simulated", "synthetic", "stale", "unavailable")


def _status(v):
    # stored chunks label real filings "real"; the contract has no such value, so use "historical"
    return v if v in _ALLOWED else "historical"


def _cit(c) -> Citation:
    return Citation(evidence_id=c["evidence_id"], doc_id=c["doc_id"], title=c["doc_id"],
                    publication_date=c["publication_date"], url=c.get("source_url"),
                    page=c.get("page"), text=c["text"], data_status=_status(c.get("data_status")))


def _direction(x: float) -> str:
    return "negative" if x < 0 else "positive" if x > 0 else "neutral"


def build_scenario(chunks: list):
    row_c, m = _first(chunks, PR, ROW.search)
    qoq = -float(m.group(5)) if m.group(4) else float(m.group(5))
    yoy = -float(m.group(7)) if m.group(6) else float(m.group(7))
    ceo_c, _ = _first(chunks, TR, lambda t: CEO_QUOTE in t)
    ask_c, _ = _first(chunks, TR, lambda t: ASK_QUOTE in t)

    def claim(cid, period, direction, ctype, text, chunk, conf):
        return Claim(claim_id=cid, entity="TCS", metric_or_event=METRIC, period=period,
                     direction=direction, claim_type=ctype, text=text,
                     evidence_ids=[chunk["evidence_id"]], confidence=conf)

    claims = [
        claim("i1", "Q2FY27_QoQ_CC", _direction(qoq), "factual",
              "Press release table (Growth by Markets), India: Q-o-Q CC growth {:+.1f}%".format(qoq), row_c, 0.9),
        claim("i2", "Q2FY27_YoY_CC", _direction(yoy), "factual",
              "Press release table (Growth by Markets), India: Y-o-Y CC growth {:+.1f}%".format(yoy), row_c, 0.9),
        claim("i3", "Q2FY27_YoY", "positive", "factual",
              'CEO on the earnings call: "{}"'.format(CEO_QUOTE), ceo_c, 0.8),
        claim("i4", "unspecified", "negative", "analytical",
              'Analyst on the call: "{}" (no period stated)'.format(ASK_QUOTE), ask_c, 0.5),
    ]
    finding = Finding(agent="filings", status="complete", claims=claims, limitations=[
        "i3 period read as Y-o-Y: the CEO sentence ends 'year-on-year' and 6% matches the table's Y-o-Y figure",
        "i4 period not stated by the analyst; it matches the table's Q-o-Q figure but is not labelled as such",
        "table share columns (India 5.8 -> 5.5) do not reconcile with +6.0% Y-o-Y; quoted as printed",
    ])
    return finding, [_cit(c) for c in {c["evidence_id"]: c for c in (row_c, ceo_c, ask_c)}.values()]


def run(as_of: date = date(2026, 10, 15), chunks: list | None = None):
    finding, cits = build_scenario(chunks if chunks is not None else load_chunks())
    registry = build_registry(cits, [])
    return finding, run_efce([finding], registry, as_of, expected_agents=("filings",))


if __name__ == "__main__":
    f, r = run()
    for c in f.claims:
        print(c.claim_id, c.period, c.direction, "|", c.text, "|", c.evidence_ids)
    print("EFCE status:", r.status, "| supported:", len(r.supported), "| rejected:", len(r.rejected))
    for x in r.rejected:
        print("  rejected", x.claim_id, "-", x.reason)
    for k in r.conflicts:
        print("conflict:", k.kind, k.claim_ids, "-", k.explanation)
