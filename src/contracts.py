"""Shared data formats for VibeFinance.

Every module reads and writes these models. Change them only by agreement.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

# ---- shared labels -------------------------------------------------------
DataStatus = Literal["historical", "delayed", "simulated", "synthetic", "stale", "unavailable"]
ModuleStatus = Literal["complete", "degraded", "failed"]
RunStatus = Literal["complete", "degraded", "conflict_unresolved", "abstained_or_limited", "failed"]
Direction = Literal["positive", "negative", "neutral", "uncertain"]
ClaimType = Literal["factual", "analytical", "inferred"]
ConflictType = Literal[
    "factual_contradiction", "temporal_difference", "analytical_disagreement", "insufficient_evidence"
]


# ---- signals (computed by code, not by an LLM) ----------------------------
class Signal(BaseModel):
    symbol: str
    as_of: date
    name: Literal["momentum", "volume_anomaly", "sentiment"]
    value: Optional[float] = None          # e.g. 2.4 for a z-score; None if unavailable
    unit: str = ""                         # e.g. "rolling_z_score", "pct_return"
    label: str                             # e.g. "positive", "unusual", "neutral"
    confidence: float = Field(ge=0.0, le=1.0)   # heuristic score, not calibrated
    method: str                            # plain-words description of how it was computed
    window_start: Optional[date] = None
    window_end: Optional[date] = None
    evidence_ids: list[str] = []           # e.g. ["px:RELIANCE.NS:2026-10-13"]
    data_status: DataStatus = "historical"
    status: ModuleStatus = "complete"


# ---- retrieved document passages ------------------------------------------
class Citation(BaseModel):
    evidence_id: str                       # e.g. "doc:ABC-AR-2026:p14:c03"
    doc_id: str
    title: str
    publication_date: date
    url: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    text: str                              # the exact stored passage
    retrieval_score: Optional[float] = None
    data_status: DataStatus = "historical"


# ---- agent output ----------------------------------------------------------
class Claim(BaseModel):
    claim_id: str
    entity: str
    metric_or_event: str
    period: str                            # e.g. "FY2026" or "2026-10-13"
    direction: Direction
    claim_type: ClaimType
    text: str
    evidence_ids: list[str] = []           # empty is allowed only for claim_type "analytical"
    confidence: float = Field(ge=0.0, le=1.0)


class Finding(BaseModel):
    agent: Literal["technical", "sentiment", "filings"]
    status: ModuleStatus
    claims: list[Claim] = []
    limitations: list[str] = []
    latency_ms: float = 0.0
    error: Optional[str] = None            # set when status is "failed"


# ---- EFCE output -----------------------------------------------------------
class Conflict(BaseModel):
    kind: ConflictType
    claim_ids: list[str]
    explanation: str


class RejectedClaim(BaseModel):
    claim_id: str
    reason: str                            # e.g. "unknown_evidence_id", "source_after_as_of"


class EFCEResult(BaseModel):
    status: RunStatus
    supported: list[Claim] = []
    conflicts: list[Conflict] = []
    abstained: list[str] = []              # dimensions we refused to conclude on
    rejected: list[RejectedClaim] = []


# ---- investor profile ------------------------------------------------------
class Holding(BaseModel):
    symbol: str
    quantity: float
    avg_cost: float
    sector: str


class Profile(BaseModel):
    profile_id: str
    risk_preference: Literal["low", "medium", "high"]
    horizon_months: int
    max_single_position_pct: float
    max_sector_pct: float
    holdings: list[Holding] = []


# ---- one saved run ---------------------------------------------------------
class RunRecord(BaseModel):
    run_id: str
    created_at: datetime
    symbol: str
    profile_id: str
    as_of: date
    config: dict = {}                      # model name, temperature, top_k, git commit
    signals: list[Signal] = []
    findings: list[Finding] = []
    efce: Optional[EFCEResult] = None
    metrics: dict[str, float] = {}         # latency per agent, hhi, citation pass rate
    data_status: DataStatus = "historical"
