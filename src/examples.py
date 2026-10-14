"""Sample objects for every contract. All values are placeholders, not real data.

Run `python -m src.examples` from the project root to write them to fixtures/.
"""
from datetime import date, datetime
from pathlib import Path

from src.contracts import (
    Citation, Claim, Conflict, EFCEResult, Finding, Holding, Profile,
    RejectedClaim, RunRecord, Signal,
)

AS_OF = date(2026, 10, 13)


def example_signal() -> Signal:
    return Signal(
        symbol="DEMO.NS", as_of=AS_OF, name="volume_anomaly", value=2.4,
        unit="rolling_z_score", label="unusual", confidence=0.7,
        method="z-score over the previous 20 sessions",
        window_start=date(2026, 9, 14), window_end=AS_OF,
        evidence_ids=["px:DEMO.NS:2026-10-13"],
    )


def example_citation() -> Citation:
    return Citation(
        evidence_id="doc:DEMO-AR-2026:p14:c03", doc_id="DEMO-AR-2026",
        title="Annual report excerpt (placeholder)", publication_date=date(2026, 5, 20),
        url="https://example.com/source", page=14, section="Risk factors",
        text="Placeholder passage text.", retrieval_score=0.82, data_status="synthetic",
    )


def example_claim() -> Claim:
    return Claim(
        claim_id="c1", entity="DEMO.NS", metric_or_event="debt_level", period="FY2026",
        direction="negative", claim_type="factual",
        text="Placeholder: the filing reports higher debt.",
        evidence_ids=["doc:DEMO-AR-2026:p14:c03"], confidence=0.6,
    )


def example_finding() -> Finding:
    return Finding(
        agent="filings", status="complete", claims=[example_claim()],
        limitations=["no filing after 2026-05-20 available"], latency_ms=0.0,
    )


def example_efce() -> EFCEResult:
    return EFCEResult(
        status="conflict_unresolved", supported=[example_claim()],
        conflicts=[Conflict(kind="analytical_disagreement", claim_ids=["c1", "c2"],
                            explanation="Momentum and the filing measure different things.")],
        abstained=[], rejected=[RejectedClaim(claim_id="c9", reason="unknown_evidence_id")],
    )


def example_profile() -> Profile:
    return Profile(
        profile_id="demo-conservative", risk_preference="low", horizon_months=36,
        max_single_position_pct=15, max_sector_pct=30,
        holdings=[Holding(symbol="AAA", quantity=100, avg_cost=100.0, sector="IT"),
                  Holding(symbol="BBB", quantity=50, avg_cost=200.0, sector="Banking")],
    )


def example_run() -> RunRecord:
    return RunRecord(
        run_id="run-0001", created_at=datetime(2026, 10, 15, 10, 0), symbol="DEMO.NS",
        profile_id="demo-conservative", as_of=AS_OF,
        config={"model": "to-record", "temperature": 0.0, "top_k": 5},
        signals=[example_signal()], findings=[example_finding()], efce=example_efce(),
        metrics={"portfolio_hhi": 0.0},
    )


EXAMPLES = {
    "signal": example_signal, "citation": example_citation, "claim": example_claim,
    "finding": example_finding, "efce": example_efce, "profile": example_profile,
    "run": example_run,
}

if __name__ == "__main__":
    out = Path("fixtures")
    out.mkdir(exist_ok=True)
    for name, make in EXAMPLES.items():
        (out / f"{name}.json").write_text(make().model_dump_json(indent=2), encoding="utf-8")
        print("wrote", out / f"{name}.json")
