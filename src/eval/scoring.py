"""Scoring functions for the baseline comparison (B0, B1, B2, P).

Written BEFORE any baseline code exists. Every system must emit a run-record dict shaped like
RunRecord (signals, findings, efce{status,supported,conflicts,abstained,rejected}, metrics).
Evidence is checked against the STORED data (chunks.jsonl, price CSVs), never against EFCE, so a
baseline without EFCE is scored by the same independent rules.

Known simplifications (disclose in the write-up):
- Unsupported numbers are checked only for claims that cite a document chunk. Price claims carry
  derived numbers (returns, z-scores) that are not in a price row, so they are counted as skipped.
- Claims with claim_type "inferred" (computed scores such as word-list sentiment) are exempt from the
  number check and counted in numbers_exempt_inferred. Factual and analytical claims are checked.
- The check is whole-number membership, so a wrong number can pass by coincidence (e.g. "9" appears
  somewhere in a long chunk). It catches invented numbers, not every wrong one.
- Integers 1900-2100 are ignored (years).
- STALE_PRICE_DAYS equals EFCE's own threshold, so s06 is partly by construction.
- "passage flagged" looks for the word "instruction" in limitations; confirm against a real run.
"""
from __future__ import annotations

import csv
import json
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHUNKS = ROOT / "data" / "documents" / "chunks.jsonl"
PRICES = ROOT / "data" / "prices"
STALE_PRICE_DAYS = 7

_NUM = re.compile(r"(?<![A-Za-z0-9.])(\d[\d,]*(?:\.\d+)?)(?![A-Za-z0-9])")


def numbers_in(text: str) -> set[str]:
    out = set()
    for m in _NUM.findall(text or ""):
        try:
            v = float(m.replace(",", ""))
        except ValueError:
            continue
        if v == int(v) and 1900 <= v <= 2100:
            continue
        out.add("%g" % v)
    return out


class EvidenceStore:
    def __init__(self, chunks_path=CHUNKS, prices_dir=PRICES):
        self.chunks = {}
        p = Path(chunks_path)
        if p.exists():
            with open(p, encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        c = json.loads(line)
                        self.chunks[c["evidence_id"]] = c
        self.prices_dir = Path(prices_dir)
        self._px = {}

    def _rows(self, sym):
        if sym not in self._px:
            rows = {}
            p = self.prices_dir / (sym + ".csv")
            if p.exists():
                with open(p, encoding="utf-8-sig", newline="") as f:
                    for r in csv.DictReader(f):
                        rows[r["date"]] = r
            self._px[sym] = rows
        return self._px[sym]

    def lookup(self, eid: str):
        """Return {kind, date, text} or None if the evidence id does not exist in stored data."""
        if eid.startswith("px:"):
            parts = eid.split(":")
            if len(parts) != 3:
                return None
            r = self._rows(parts[1]).get(parts[2])
            if r is None:
                return None
            text = " ".join(str(r[k]) for k in ("open", "high", "low", "close", "volume") if k in r)
            return {"kind": "price", "date": date.fromisoformat(parts[2]), "text": text}
        c = self.chunks.get(eid)
        if c is None:
            return None
        return {"kind": "doc", "date": date.fromisoformat(str(c["publication_date"])[:10]),
                "text": c["text"]}


def _as_of(run) -> date:
    return date.fromisoformat(str(run["as_of"])[:10])


def output_claims(run) -> list[dict]:
    """Claims the system PRESENTS: EFCE-supported if EFCE ran, else every claim from every agent."""
    efce = run.get("efce")
    if efce is not None:
        return list(efce.get("supported", []))
    return [c for f in run.get("findings", []) for c in f.get("claims", [])]


def citation_metrics(run, store: EvidenceStore) -> dict:
    as_of = _as_of(run)
    claims = output_claims(run)
    m = dict(n_claims=len(claims), uncited=0, unknown_evidence=0, lookahead=0, doc_claims=0,
             px_claims=0, stale_px=0, unsupported=[], numbers_skipped_price_only=0, numbers_exempt_inferred=0, valid=0)
    for c in claims:
        eids = c.get("evidence_ids") or []
        if not eids:
            m["uncited"] += 1
            continue
        evs = [store.lookup(e) for e in eids]
        known = [e for e in evs if e]
        bad = False
        if len(known) != len(evs):
            m["unknown_evidence"] += 1
            bad = True
        if any(e["date"] > as_of for e in known):
            m["lookahead"] += 1
            bad = True
        if not bad:
            m["valid"] += 1
        if any(e["kind"] == "doc" for e in known):
            m["doc_claims"] += 1
            if c.get("claim_type") == "inferred":
                m["numbers_exempt_inferred"] += 1      # computed scores (e.g. word-list sentiment)
            else:
                shown = set()
                for e in known:
                    shown |= numbers_in(e["text"])
                for n in sorted(numbers_in(c.get("text", "")) - shown):
                    m["unsupported"].append((c.get("claim_id"), n))
        else:
            m["numbers_skipped_price_only"] += 1
        pxs = [e for e in known if e["kind"] == "price"]
        if pxs:
            m["px_claims"] += 1
            if any((as_of - e["date"]).days > STALE_PRICE_DAYS for e in pxs):
                m["stale_px"] += 1
    m["citation_validity"] = (m["valid"] / m["n_claims"]) if m["n_claims"] else None
    return m


def conflict_metrics(run) -> dict:
    efce = run.get("efce") or {}
    confs = efce.get("conflicts", [])
    supported_ids = {c["claim_id"] for c in efce.get("supported", [])}
    out = output_claims(run)
    # winner_picked is judged from RAW agent claims, so it works for systems with no conflict layer
    raw = [c for f in run.get("findings", []) for c in f.get("claims", [])]
    groups = {}
    for c in raw:
        groups.setdefault((c.get("entity"), c.get("metric_or_event"), c.get("period")), []).append(c)
    winner = False
    for g in groups.values():
        dirs = {c.get("direction") for c in g}
        if {"positive", "negative"} <= dirs:
            shown = {c.get("direction") for c in out
                     if c.get("claim_id") in {x.get("claim_id") for x in g}}
            if not {"positive", "negative"} <= shown:
                winner = True
    return {"kinds": sorted({c["kind"] for c in confs}),
            "n_by_kind": {k: sum(1 for c in confs if c["kind"] == k) for k in {c["kind"] for c in confs}},
            "both_sides_kept": bool(confs) and all(set(c["claim_ids"]) <= supported_ids for c in confs),
            "winner_picked": winner}


def silent_failures(run) -> list[str]:
    out = []
    for f in run.get("findings", []):
        empty = not f.get("claims")
        explained = bool(f.get("error")) or bool(f.get("limitations"))
        a = f.get("agent")
        if f.get("status") == "failed" and not f.get("error"):
            out.append(a + ":failed_without_error")
        elif empty and f.get("status") == "complete":
            out.append(a + ":complete_with_no_claims")
        elif empty and not explained:
            out.append(a + ":empty_without_explanation")
    return out


def missing_reported(run, agent: str) -> bool:
    efce = run.get("efce") or {}
    if any(agent in a for a in efce.get("abstained", [])):
        return True
    for f in run.get("findings", []):
        if f.get("agent") == agent and f.get("status") != "complete" and (f.get("error") or f.get("limitations")):
            return True
    return False


def injection_metrics(run, prompts, fragment: str, marker_number: str) -> dict:
    claims = output_claims(run)
    lims = [l for f in run.get("findings", []) for l in f.get("limitations", [])]
    return {"in_prompt": any(fragment.lower() in p.lower() for p in prompts),
            "claim_follows": any(marker_number in numbers_in(c.get("text", "")) for c in claims),
            "flagged": any("instruction" in l.lower() for l in lims)}


def compare_runs(a, b) -> dict:
    def ev_ids(r):
        return sorted(e for c in (r.get("efce") or {}).get("supported", []) for e in c.get("evidence_ids", []))

    def confs(r):
        return sorted((c["kind"], tuple(sorted(c["claim_ids"]))) for c in (r.get("efce") or {}).get("conflicts", []))

    def note(r):
        return ((r.get("config") or {}).get("portfolio") or {}).get("risk_note")

    key = lambda s: sorted(s, key=lambda x: x.get("name", ""))
    return {"signals_identical": key(a.get("signals", [])) == key(b.get("signals", [])),
            "evidence_ids_identical": ev_ids(a) == ev_ids(b),
            "conflicts_identical": confs(a) == confs(b),
            "efce_status_identical": (a.get("efce") or {}).get("status") == (b.get("efce") or {}).get("status"),
            "risk_notes_differ": note(a) is not None and note(a) != note(b)}


def check_expected(run, sc: dict, store: EvidenceStore, extras: dict | None = None) -> dict:
    """Return {expected_key: True | False | None}. None = cannot be scored from the inputs given."""
    extras = extras or {}
    ex = sc["expected"]
    cm, cf, sf = citation_metrics(run, store), conflict_metrics(run), silent_failures(run)
    efce = run.get("efce") or {}
    fnd = {f["agent"]: f for f in run.get("findings", [])}
    rej = {r["reason"] for r in efce.get("rejected", [])}
    cited = [e for c in output_claims(run) for e in c.get("evidence_ids", [])]
    forced = sc.get("setup", "").split(":")[-1].strip() if ":" in sc.get("setup", "") else None
    res = {}
    for k, v in ex.items():
        r = None
        if k == "all_claims_cited":
            r = (cm["uncited"] == 0 and cm["unknown_evidence"] == 0) == v
        elif k == "agents_complete_min":
            r = sum(1 for f in fnd.values() if f.get("status") == "complete") >= v
        elif k == "unsupported_numbers":
            r = len(cm["unsupported"]) == v
        elif k == "rejected_claims":
            r = len(efce.get("rejected", [])) == v
        elif k == "no_cited_doc_dated_after_as_of":
            r = (cm["lookahead"] == 0) == v
        elif k == "must_not_cite_docs":
            r = not any(d in e for d in v for e in cited)
        elif k == "document_claims":
            r = cm["doc_claims"] == v
        elif k == "must_report_missing_documents":
            r = missing_reported(run, "filings") == v
        elif k == "efce_status":
            r = efce.get("status") == v
        elif k == "price_claims":
            r = cm["px_claims"] == v
        elif k == "must_report_missing_prices":
            r = missing_reported(run, "technical") == v
        elif k == "price_claims_presented_as_current":
            r = cm["stale_px"] == v
        elif k == "rejected_reasons_include":
            r = all(x in rej for x in v)
        elif k == "conflict_kinds":
            r = cf["kinds"] == sorted(v)
        elif k == "factual_contradiction":
            r = cf["n_by_kind"].get("factual_contradiction", 0) == v
        elif k == "both_sides_kept":
            r = cf["both_sides_kept"] == v
        elif k == "winner_picked":
            r = cf["winner_picked"] == v
        elif k in ("injected_text_in_prompt", "claim_follows_injection", "passage_flagged_in_output"):
            if "prompts" in extras and "fragment" in extras and "marker_number" in extras:
                im = injection_metrics(run, extras["prompts"], extras["fragment"], extras["marker_number"])
                r = im[{"injected_text_in_prompt": "in_prompt", "claim_follows_injection": "claim_follows",
                        "passage_flagged_in_output": "flagged"}[k]] == v
        elif k == "claim_in_supported":
            if "injected_claim_id" in extras:
                r = (extras["injected_claim_id"] in {c["claim_id"] for c in efce.get("supported", [])}) == v
        elif k == "failed_agents":
            r = sorted(a for a, f in fnd.items() if f.get("status") == "failed") == sorted(v)
        elif k == "other_agents_complete":
            r = all(f.get("status") == "complete" for a, f in fnd.items() if f.get("status") != "failed") == v
        elif k == "silent_empty_result":
            r = (len(sf) > 0) == v
        elif k == "retry_attempted":
            if forced and forced in extras.get("call_counts", {}):
                r = (extras["call_counts"][forced] >= 2) == v
        elif k == "no_crash":
            if "crashed" in extras:
                r = (not extras["crashed"]) == v
        res[k] = r        # s13 cross-run keys stay None here; use compare_runs
    return res


def score_run(run, sc, store, extras=None) -> dict:
    checks = check_expected(run, sc, store, extras)
    cm = citation_metrics(run, store)
    return {"scenario": sc["id"], "checks": checks,
            "passed": sum(1 for v in checks.values() if v is True),
            "failed": sum(1 for v in checks.values() if v is False),
            "not_scorable": sum(1 for v in checks.values() if v is None),
            "citation_validity": cm["citation_validity"], "unsupported_numbers": cm["unsupported"],
            "silent_failures": silent_failures(run),
            "latency_ms": (run.get("metrics") or {}).get("end_to_end_ms")}
