"""Streamlit-callable entry point for the VibeFinance analysis pipeline.

This reuses the same agents and EFCE functions as src.run_pipeline without
invoking the command-line interface or writing a run file automatically.
"""
from __future__ import annotations

import time
from datetime import date, datetime
from pathlib import Path

from src.agents.base import AgentContext
from src.agents.filings import MODEL_NAME
from src.agents.orchestrator import run_agents_sync
from src.contracts import Profile, RunRecord
from src.efce import build_registry, run_efce

ROOT = Path(__file__).resolve().parents[1]


def run_analysis(
    symbol: str,
    as_of: date,
    profile: Profile | None = None,
    question: str | None = None,
    price_dir: str | None = None,
    use_llm: bool = True,
    timeout_s: float = 240.0,
) -> RunRecord:
    """Run all agents and EFCE for a symbol, returning a validated RunRecord.

    Note: `question` is recorded in run config for traceability. The current
    agents use their predefined retrieval topics and do not yet use this
    question to customize retrieval.
    """
    if not symbol or not symbol.strip():
        raise ValueError("symbol must be a non-empty stock symbol")

    started = time.perf_counter()

    from src.rag.retrieve import retrieve

    context = AgentContext(
        symbol=symbol.strip().upper(),
        as_of=as_of,
        price_dir=price_dir,
        retrieve_fn=retrieve,
        top_k=3,
        use_llm=use_llm,
        rerank=True,
    )

    findings, signals, citations = run_agents_sync(
        context,
        timeout_s=timeout_s,
    )
    registry = build_registry(citations, signals)
    efce_result = run_efce(findings, registry, as_of)
    elapsed_ms = (time.perf_counter() - started) * 1000

    supported_count = len(efce_result.supported)
    rejected_count = len(efce_result.rejected)
    total_validated = supported_count + rejected_count

    metrics = {
        f"latency_ms_{finding.agent}": round(finding.latency_ms, 1)
        for finding in findings
    }
    metrics.update({
        "end_to_end_ms": round(elapsed_ms, 1),
        "agents_complete": float(
            sum(finding.status == "complete" for finding in findings)
        ),
        "citation_pass_rate": (
            round(supported_count / total_validated, 3)
            if total_validated
            else 0.0
        ),
    })
    now = datetime.now()

    record = RunRecord(
        run_id=now.strftime("run-%Y%m%d-%H%M%S-%f"),
        created_at=now,
        symbol=context.symbol,
        profile_id=profile.profile_id if profile else "unassigned",
        as_of=as_of,
        config={
            "model": MODEL_NAME if use_llm else "none",
            "temperature": 0,
            "top_k": 3,
            "timeout_s": timeout_s,
            "question": question,
        },
        signals=signals,
        findings=findings,
        efce=efce_result,
        metrics=metrics,
    )

    # Persist the completed analysis for later evidence verification.
    runs_dir = ROOT / "data" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_path = runs_dir / f"{record.run_id}.json"
    run_path.write_text(
        record.model_dump_json(indent=2),
        encoding="utf-8",
    )

    return record