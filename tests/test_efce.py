"""EFCE tests. Several map to the plan's tests: T1 valid, T3 missing filing, T4 bad citation,
T5 stale source, T8 conflict."""
from datetime import date, timedelta

from src.contracts import Citation, Claim, Finding, Signal
from src.efce import Evidence, build_registry, run_efce

AS_OF = date(2026, 10, 13)
REG = {
    "px:TCS.NS:2026-10-13": Evidence(date(2026, 10, 13)),
    "doc:TCS-PR:p01:c01": Evidence(date(2026, 10, 9)),
    "doc:TCS-AR:p58:c01": Evidence(date(2026, 5, 20)),
}


def claim(cid, direction="positive", eids=("doc:TCS-PR:p01:c01",), ctype="factual",
          metric="outlook", period="Q2FY27"):
    return Claim(claim_id=cid, entity="TCS", metric_or_event=metric, period=period, direction=direction,
                 claim_type=ctype, text="x", evidence_ids=list(eids), confidence=0.7)


def finding(agent, *claims, status="complete"):
    return Finding(agent=agent, status=status, claims=list(claims))


def three(t="positive", s="positive", f="positive", **kw):
    return [finding("technical", claim("t1", t, ("px:TCS.NS:2026-10-13",), metric="momentum", **kw)),
            finding("sentiment", claim("s1", s, ("doc:TCS-PR:p01:c01",), metric="sentiment", **kw)),
            finding("filings", claim("f1", f, ("doc:TCS-AR:p58:c01",), metric="margin", **kw))]


def test_t1_valid_run_is_complete():
    r = run_efce(three(), REG, AS_OF)
    assert r.status == "complete" and len(r.supported) == 3 and not r.rejected and not r.conflicts


def test_t3_missing_filing_degrades_but_keeps_other_claims():
    r = run_efce(three()[:2], REG, AS_OF)
    assert r.status == "degraded" and "filings:missing" in r.abstained and len(r.supported) == 2


def test_t4_unknown_evidence_id_is_rejected():
    fs = three()
    fs[2] = finding("filings", claim("f1", eids=("doc:FAKE:p99:c00",), metric="margin"))
    r = run_efce(fs, REG, AS_OF)
    assert [x.reason for x in r.rejected] == ["unknown_evidence_id"]
    assert "f1" not in [c.claim_id for c in r.supported]
    assert r.status == "degraded"


def test_t5_stale_document_is_rejected():
    reg = {**REG, "doc:OLD:p1:c1": Evidence(AS_OF - timedelta(days=800))}
    fs = three()
    fs[2] = finding("filings", claim("f1", eids=("doc:OLD:p1:c1",), metric="margin"))
    r = run_efce(fs, reg, AS_OF)
    assert r.rejected[0].reason == "stale_source"
    assert any(c.kind == "insufficient_evidence" for c in r.conflicts) and "margin" in r.abstained


def test_stale_data_status_is_rejected_even_if_recent():
    reg = {**REG, "doc:TCS-PR:p01:c01": Evidence(date(2026, 10, 9), "stale")}
    r = run_efce(three(), reg, AS_OF)
    assert r.rejected[0].reason == "stale_source"


def test_old_price_evidence_is_stale_after_7_days():
    sig = Signal(symbol="TCS", as_of=AS_OF, name="momentum", label="positive", confidence=0.5, method="m",
                 evidence_ids=["px:TCS.NS:2026-10-01"])
    reg = build_registry([], [sig])
    assert reg["px:TCS.NS:2026-10-01"].published == date(2026, 10, 1)
    fs = [finding("technical", claim("t1", eids=("px:TCS.NS:2026-10-01",), metric="momentum"))]
    assert run_efce(fs, reg, AS_OF, expected_agents=("technical",)).rejected[0].reason == "stale_source"


def test_source_after_as_of_is_rejected():
    reg = {**REG, "doc:LATE:p1:c1": Evidence(AS_OF + timedelta(days=1))}
    fs = [finding("filings", claim("f1", eids=("doc:LATE:p1:c1",)))]
    assert run_efce(fs, reg, AS_OF).rejected[0].reason == "source_after_as_of"


def test_factual_claim_without_evidence_rejected_analytical_allowed():
    fs = [finding("filings", claim("f1", eids=()), claim("f2", eids=(), ctype="analytical", metric="view"))]
    r = run_efce(fs, REG, AS_OF)
    assert [x.reason for x in r.rejected] == ["missing_evidence"]
    assert [c.claim_id for c in r.supported] == ["f2"]


def test_t8_factual_contradiction_keeps_both_sides():
    fs = [finding("filings", claim("a", "positive", metric="revenue")),
          finding("sentiment", claim("b", "negative", metric="revenue"))]
    r = run_efce(fs, REG, AS_OF)
    assert r.status == "conflict_unresolved"
    assert [c.kind for c in r.conflicts] == ["factual_contradiction"]
    assert {c.claim_id for c in r.supported} == {"a", "b"}


def test_different_periods_is_temporal_not_blocking():
    fs = [finding("filings", claim("a", "positive", metric="revenue", period="Q2FY27")),
          finding("sentiment", claim("b", "negative", metric="revenue", period="Q1FY27"))]
    r = run_efce(fs, REG, AS_OF, expected_agents=("filings", "sentiment"))
    assert [c.kind for c in r.conflicts] == ["temporal_difference"]
    assert r.status == "complete"


def test_analytical_disagreement():
    fs = [finding("filings", claim("a", "positive", metric="outlook", ctype="inferred")),
          finding("sentiment", claim("b", "negative", metric="outlook"))]
    r = run_efce(fs, REG, AS_OF)
    assert [c.kind for c in r.conflicts] == ["analytical_disagreement"] and r.status == "conflict_unresolved"


def test_neutral_or_uncertain_is_not_a_conflict():
    fs = [finding("filings", claim("a", "positive", metric="outlook")),
          finding("sentiment", claim("b", "uncertain", metric="outlook"))]
    assert run_efce(fs, REG, AS_OF).conflicts == []


def test_all_modules_failed():
    fs = [Finding(agent=a, status="failed", error="boom") for a in ("technical", "sentiment", "filings")]
    assert run_efce(fs, REG, AS_OF).status == "failed"


def test_no_findings_abstains_on_everything():
    r = run_efce([], REG, AS_OF)
    assert r.status == "abstained_or_limited" and len(r.abstained) == 3


def test_single_supporting_agent_is_limited():
    r = run_efce([finding("filings", claim("f1"))], REG, AS_OF)
    assert r.status == "abstained_or_limited"
