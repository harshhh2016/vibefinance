"""Runs the three agents in parallel with a timeout and one retry for bad output."""
from __future__ import annotations

import asyncio
import time
from typing import Callable, Optional

from src.agents import filings, sentiment, technical
from src.agents.base import AgentContext
from src.contracts import Finding

AGENTS = {"technical": technical.run, "sentiment": sentiment.run, "filings": filings.run}


async def _run_one(name: str, fn: Callable, ctx: AgentContext, timeout_s: float, retries: int):
    start = time.perf_counter()
    error = "unknown error"
    for _attempt in range(retries + 1):
        try:
            res = await asyncio.wait_for(asyncio.to_thread(fn, ctx), timeout=timeout_s)
            raw = res.finding.model_dump() if isinstance(res.finding, Finding) else res.finding
            finding = Finding.model_validate(raw)          # malformed output fails here
            finding.latency_ms = (time.perf_counter() - start) * 1000
            return finding, list(res.signals), list(res.citations)
        except asyncio.TimeoutError:
            error = "timeout after {}s".format(timeout_s)
            break                                          # no retry on timeout
        except Exception as exc:                           # bad output or agent crash: retry
            error = "{}: {}".format(type(exc).__name__, str(exc)[:200])
    failed = Finding(agent=name, status="failed", error=error, limitations=[error],
                     latency_ms=(time.perf_counter() - start) * 1000)
    return failed, [], []


async def run_agents(ctx: AgentContext, agents: Optional[dict] = None,
                     timeout_s: float = 240.0, retries: int = 1):
    agents = agents or AGENTS
    results = await asyncio.gather(*[
        _run_one(name, fn, ctx, timeout_s, retries) for name, fn in agents.items()
    ])
    findings = [r[0] for r in results]
    signals = [s for r in results for s in r[1]]
    citations = {c.evidence_id: c for r in results for c in r[2]}
    return findings, signals, list(citations.values())


def run_agents_sync(ctx: AgentContext, agents: Optional[dict] = None,
                    timeout_s: float = 240.0, retries: int = 1):
    return asyncio.run(run_agents(ctx, agents, timeout_s, retries))
