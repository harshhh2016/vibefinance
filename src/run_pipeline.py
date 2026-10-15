"""Run the whole pipeline once for one stock and save the run.

Usage: python -m src.run_pipeline TCS            (uses the local Ollama model for filings)
       python -m src.run_pipeline TCS --no-llm   (verbatim quotes only)
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime
from pathlib import Path

from src.agents.base import AgentContext
from src.agents.filings import MODEL_NAME
from src.agents.orchestrator import run_agents_sync
from src.contracts import RunRecord
from src.efce import build_registry, run_efce
from src.portfolio.apply import apply_profile, load_latest_prices
from src.portfolio.store import connect, list_profile_ids, load_profile

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol")
    ap.add_argument("--as-of", default=None, help="YYYY-MM-DD, default today")
    ap.add_argument("--profile", default="unassigned")
    ap.add_argument("--price-dir", default=None)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()
    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()

    from src.rag.retrieve import retrieve
    try:
        retrieve("warm up", args.symbol, as_of, 1)      # load the embedding model before timing
    except Exception as exc:
        print("warm-up retrieval failed:", exc)

    ctx = AgentContext(symbol=args.symbol, as_of=as_of, price_dir=args.price_dir,
                       retrieve_fn=retrieve, top_k=3, use_llm=not args.no_llm)
    t0 = time.perf_counter()
    findings, signals, citations = run_agents_sync(ctx, timeout_s=args.timeout)
    registry = build_registry(citations, signals)
    result = run_efce(findings, registry, as_of)
    total_ms = (time.perf_counter() - t0) * 1000

    print("\n{} as of {}".format(args.symbol, as_of))
    for f in findings:
        print("\n[{}] status={} latency={:.0f} ms".format(f.agent, f.status, f.latency_ms))
        if f.error:
            print("  error:", f.error)
        for c in f.claims:
            print("  {} {} -> {}".format(c.claim_id, c.direction, c.text[:130]))
            print("      evidence:", ", ".join(c.evidence_ids))
        for lim in f.limitations:
            print("  limit:", lim)
    print("\nEFCE status:", result.status)
    print("  supported:", len(result.supported), " rejected:", len(result.rejected),
          " conflicts:", len(result.conflicts), " abstained:", result.abstained)
    for r in result.rejected:
        print("  rejected", r.claim_id, "-", r.reason)

    n_ok, n_bad = len(result.supported), len(result.rejected)
    metrics = {"latency_ms_" + f.agent: round(f.latency_ms, 1) for f in findings}
    metrics["end_to_end_ms"] = round(total_ms, 1)
    metrics["agents_complete"] = float(sum(f.status == "complete" for f in findings))
    metrics["citation_pass_rate"] = round(n_ok / (n_ok + n_bad), 3) if (n_ok + n_bad) else 0.0

    run_id = datetime.now().strftime("run-%Y%m%d-%H%M%S")
    record = RunRecord(
        run_id=run_id, created_at=datetime.now(), symbol=args.symbol,
        profile_id=args.profile, as_of=as_of,
        config={"model": "none" if args.no_llm else MODEL_NAME, "temperature": 0,
                "top_k": 3, "timeout_s": args.timeout},
        signals=signals, findings=findings, efce=result, metrics=metrics,
    )
    if args.profile != "unassigned":
        conn = connect(str(ROOT / "data" / "portfolio.db"))
        profile = load_profile(conn, args.profile)
        if profile is None:
            print("\nunknown profile {!r}; available: {}. Portfolio section skipped.".format(
                args.profile, list_profile_ids(conn) or "none (run: python -m src.portfolio.store)"))
        else:
            pdir = Path(args.price_dir) if args.price_dir else ROOT / "data" / "prices"
            syms = {h.symbol for h in profile.holdings} | {args.symbol}
            record = apply_profile(record, profile, load_latest_prices(syms, as_of, pdir))
            sec = record.config["portfolio"]
            print("\nportfolio ({}, synthetic): HHI {:.2f}, breaches {}".format(
                profile.profile_id, sec["hhi"], len(sec["breaches"])))
            print("  risk note:", sec["risk_note"])
    out = ROOT / "data" / "runs"
    out.mkdir(parents=True, exist_ok=True)
    (out / (run_id + ".json")).write_text(record.model_dump_json(indent=2), encoding="utf-8")
    print("\nmetrics:", json.dumps(metrics))
    print("saved:", out / (run_id + ".json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
