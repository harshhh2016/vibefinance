"""Hand-audit sheet for the final run. Independent of EFCE: a person reads each claim against the
cited source text and records a verdict.

  python -m src.eval.audit make  [--runs data\\eval\\runs_final] [--out data\\eval\\audit.csv]
  python -m src.eval.audit tally [--out data\\eval\\audit.csv]

Verdicts to type in the `verdict` column:
  ok       every fact and number in the claim appears in the cited text, with the right period/context
  partial  some of it does, or the period/context is wrong or ambiguous
  bad      a number or fact is not in the cited text, or the cited text is unrelated
Uncited claims are pre-filled as no_cite. Pool: real-data scenarios only (s08 synthetic and the injected
s10 claim are excluded). Multi-agent systems share one agent run, so their claims are audited once
(group "agents"); B0's claims are audited separately (group "B0").
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from src.eval.scoring import EvidenceStore

ROOT = Path(__file__).resolve().parents[2]
SKIP_SCENARIOS = {"s08"}
SKIP_CLAIM_IDS = {"x1"}
LIMITS = {"agents": 30, "B0": 10}


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def pool(runs_dir: Path):
    seen, out = set(), {"agents": [], "B0": []}
    for f in sorted(runs_dir.glob("*__P.json")) + sorted(runs_dir.glob("*__B0.json")):
        r = _load(f)
        sid, system = r.get("scenario"), r.get("system")
        if sid in SKIP_SCENARIOS:
            continue
        group = "B0" if system == "B0" else "agents"
        for c in (r.get("efce") or {}).get("supported", []):
            if c.get("claim_id") in SKIP_CLAIM_IDS:
                continue
            key = (group, c.get("text"), tuple(c.get("evidence_ids", [])))
            if key in seen:
                continue
            seen.add(key)
            out[group].append((sid, system, c))
    return out


def excerpt(store, eids):
    parts = []
    for e in eids[:2]:
        ev = store.lookup(e)
        parts.append("[{}] {}".format(e, ev["text"][:600].replace("\n", " ") if ev else "NOT FOUND IN STORED DATA"))
    return " || ".join(parts)


def make(runs_dir: Path, out: Path):
    store, rng, rows = EvidenceStore(), random.Random(42), []
    for group, items in pool(runs_dir).items():
        if len(items) > LIMITS[group]:
            items = rng.sample(items, LIMITS[group])
        for sid, system, c in items:
            eids = c.get("evidence_ids", [])
            rows.append({"group": group, "scenario": sid, "claim_id": c.get("claim_id"), "claim_text": c.get("text"),
                         "evidence_ids": ";".join(eids), "cited_text": excerpt(store, eids) if eids else "",
                         "verdict": "" if eids else "no_cite", "notes": ""})
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["group"])
        w.writeheader()
        w.writerows(rows)
    print("claims to audit:", Counter(r["group"] for r in rows), "| wrote", out)


def tally(out: Path):
    rows = list(csv.DictReader(open(out, encoding="utf-8-sig", newline="")))
    todo = [r for r in rows if not r["verdict"].strip()]
    if todo:
        print("NOT DONE: {} claim(s) have no verdict yet (first: {} {})".format(len(todo), todo[0]["scenario"], todo[0]["claim_id"]))
        return
    for g in ("agents", "B0"):
        v = Counter(r["verdict"].strip().lower() for r in rows if r["group"] == g)
        n = sum(v.values())
        if n:
            print("{:<7} n={}  ok={} partial={} bad={} no_cite={}  ok-rate={:.0%}".format(
                g, n, v["ok"], v["partial"], v["bad"], v["no_cite"], v["ok"] / n))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "tally"])
    ap.add_argument("--runs", default=str(ROOT / "data" / "eval" / "runs_final"))
    ap.add_argument("--out", default=str(ROOT / "data" / "eval" / "audit.csv"))
    a = ap.parse_args()
    make(Path(a.runs), Path(a.out)) if a.cmd == "make" else tally(Path(a.out))
