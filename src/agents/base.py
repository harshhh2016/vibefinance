"""Shared pieces for the three agents."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Optional


@dataclass
class AgentContext:
    symbol: str
    as_of: date
    price_dir: Optional[str] = None
    retrieve_fn: Optional[Callable] = None    # retrieve(query, company, as_of, k) -> list[Citation]
    top_k: int = 3
    use_llm: bool = True                      # filings agent: try the local Ollama model first
    llm_fn: Optional[Callable] = None         # llm_fn(prompt) -> dict; tests inject a fake
    rerank: bool = True                       # cross-encoder rerank of retrieved passages


@dataclass
class AgentResult:
    finding: Any                              # a Finding or a dict; the orchestrator validates it
    signals: list = field(default_factory=list)
    citations: list = field(default_factory=list)


def get_retrieve(ctx: AgentContext) -> Callable:
    if ctx.retrieve_fn is not None:
        return ctx.retrieve_fn
    from src.rag.retrieve import retrieve     # imported late: loading the model is slow
    return retrieve
