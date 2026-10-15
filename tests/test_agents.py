import time
from datetime import date

from src.agents import filings, sentiment, technical
from src.agents.base import AgentContext, AgentResult
from src.agents.orchestrator import run_agents_sync
from src.contracts import Citation, Finding, Signal
from src.efce import build_registry, run_efce

D = date(2026, 10, 14)
PX = ["px:TCS.NS:2026-10-14"]
EID = "doc:TCS-AR-FY26:p058:c01"


def _sig(name, value, label, status="complete", evidence=None, data_status="historical"):
    return Signal(symbol="TCS", as_of=D, name=name, value=value, unit="x", label=label,
                  confidence=0.5, method="test", evidence_ids=evidence or [],
                  status=status, data_status=data_status)


def _cit(text="Revenue from operations 267,021 100.0 4.6 255,324 100.0\nSome other line here"):
    return Citation(evidence_id=EID, doc_id="TCS-AR-FY26", title="AR",
                    publication_date=date(2026, 5, 15), page=58, text=text, retrieval_score=0.75)


def test_technical_claims_cite_signal_evidence():
    f = technical.build_finding([_sig("momentum", 0.0294, "neutral", evidence=PX),
                                 _sig("volume_anomaly", 2.85, "unusual", evidence=PX)])
    assert f.status == "complete" and len(f.claims) == 2
    assert all(c.evidence_ids == PX for c in f.claims)


def test_technical_one_failed_signal_is_degraded():
    f = technical.build_finding([_sig("momentum", 0.03, "positive", evidence=PX),
                                 _sig("volume_anomaly", None, "unavailable", status="failed")])
    assert f.status == "degraded" and len(f.claims) == 1 and f.limitations


def test_technical_all_failed_is_failed_not_empty_success():
    f = technical.build_finding([_sig("momentum", None, "unavailable", status="failed")])
    assert f.status == "failed" and f.claims == [] and f.error


def test_sentiment_without_text_is_failed():
    f = sentiment.build_finding(_sig("sentiment", None, "unavailable", status="failed",
                                     data_status="unavailable"))
    assert f.status == "failed" and f.claims == []


def test_sentiment_claim_cites_passages():
    f = sentiment.build_finding(_sig("sentiment", 0.25, "positive", evidence=[EID]))
    assert f.status == "complete" and f.claims[0].evidence_ids == [EID]


def test_filings_extractive_quotes_stored_text():
    cit = _cit()
    f = filings.build_finding("TCS", {m: [cit] for m, _q, _k in filings.TOPICS})
    assert f.status == "complete" and len(f.claims) == 1
    c = f.claims[0]
    assert c.metric_or_event == "revenue" and c.evidence_ids == [EID]
    assert c.text.split('"')[1] in cit.text


def test_filings_with_no_passages_is_failed():
    f = filings.build_finding("TCS", {})
    assert f.status == "failed" and f.claims == []


def test_timeout_and_malformed_output_do_not_stop_other_agents():
    calls = {"bad": 0}

    def ok(ctx):
        return AgentResult(Finding(agent="technical", status="complete"))

    def slow(ctx):
        time.sleep(1.0)
        return AgentResult(Finding(agent="sentiment", status="complete"))

    def bad(ctx):
        calls["bad"] += 1
        return AgentResult({"agent": "filings", "status": "banana"})

    findings, _s, _c = run_agents_sync(AgentContext(symbol="TCS", as_of=D),
                                       agents={"technical": ok, "sentiment": slow, "filings": bad},
                                       timeout_s=0.2, retries=1)
    by = {f.agent: f for f in findings}
    assert by["technical"].status == "complete"
    assert by["sentiment"].status == "failed" and "timeout" in by["sentiment"].error
    assert by["filings"].status == "failed" and calls["bad"] == 2


def test_agent_findings_pass_efce_with_no_rejections():
    sigs = [_sig("momentum", 0.0294, "neutral", evidence=PX),
            _sig("volume_anomaly", 2.85, "unusual", evidence=PX)]
    cit = _cit()
    f2 = filings.build_finding("TCS", {m: [cit] for m, _q, _k in filings.TOPICS})
    registry = build_registry([cit], sigs)
    result = run_efce([technical.build_finding(sigs), f2], registry, D)
    assert result.rejected == [] and len(result.supported) == 3


def test_llm_claims_with_bad_ids_or_numbers_are_dropped():
    passages = {EID: _cit()}
    data = {"claims": [
        {"metric": "revenue", "text": "Revenue was 267,021 crore.", "evidence_id": EID},
        {"metric": "revenue", "text": "Revenue was 999,999.", "evidence_id": EID},
        {"metric": "attrition", "text": "Attrition was 13.7%.", "evidence_id": "doc:FAKE:p1:c1"},
        {"metric": "bogus_metric", "text": "Something 267,021.", "evidence_id": EID},
    ]}
    good, dropped = filings.validate_llm_claims(data, passages)
    assert [m for m, _t, _c in good] == ["revenue"] and dropped == 3


def test_filings_uses_llm_claims_when_valid():
    cit = _cit()

    def fake_retrieve(q, c, a, k):
        return [cit]

    ctx = AgentContext(symbol="TCS", as_of=D, retrieve_fn=fake_retrieve, rerank=False,
                       llm_fn=lambda p: {"claims": [{"metric": "revenue",
                                                     "text": "Revenue from operations was 267,021 crore.",
                                                     "evidence_id": EID}]})
    f = filings.run(ctx).finding
    assert f.status == "complete" and len(f.claims) == 1
    assert f.claims[0].evidence_ids == [EID] and "267,021" in f.claims[0].text


def test_filings_falls_back_when_llm_fails():
    cit = _cit()

    def fake_retrieve(q, c, a, k):
        return [cit]

    def broken(prompt):
        raise ConnectionError("ollama down")

    ctx = AgentContext(symbol="TCS", as_of=D, retrieve_fn=fake_retrieve, rerank=False, llm_fn=broken)
    f = filings.run(ctx).finding
    assert f.status == "complete" and len(f.claims) == 1
    assert any("fell back" in lim for lim in f.limitations)


def test_technical_cites_only_latest_price_row():
    f = technical.build_finding([_sig("momentum", 0.03, "positive",
                                      evidence=["px:TCS:2026-09-15", "px:TCS:2026-10-14"])])
    assert f.claims[0].evidence_ids == ["px:TCS:2026-10-14"]


def test_filings_prompt_stays_within_budget():
    from types import SimpleNamespace
    from src.agents import filings
    long_text = "Operating margin was 25.2 percent this quarter. " * 100
    cits = {"doc:X:p%d:c01" % i: SimpleNamespace(text=long_text) for i in range(5)}
    shown = {eid: filings.focus(c.text, ("operating margin",)) for eid, c in cits.items()}
    assert len(filings.build_prompt("TCS", cits, shown)) <= filings.MAX_PROMPT_CHARS


def test_filings_number_outside_shown_text_is_dropped():
    from types import SimpleNamespace
    from src.agents import filings
    c = SimpleNamespace(text="Margin is 25.2 percent. " + "x" * 1000 + " dividend 110", page=1)
    data = {"claims": [{"metric": "dividend", "text": "Dividend is 110", "evidence_id": "e1"}]}
    good, dropped = filings.validate_llm_claims(data, {"e1": c}, {"e1": c.text[:30]})
    assert good == [] and dropped == 1


def test_filings_model_never_chooses_its_own_source():
    from types import SimpleNamespace
    from src.agents import filings
    c = SimpleNamespace(text="Attrition remained stable at 13.3% this quarter.", page=1)
    eid = "doc:X:p001:c01"
    data = {"claims": [{"metric": "attrition", "text": "Attrition was 13.3%", "evidence_id": "[p001:c01]"}]}
    data = filings.assign_evidence(data, {"attrition": eid})
    good, dropped = filings.validate_llm_claims(data, {eid: c}, {eid: c.text})
    assert len(good) == 1 and good[0][2] is c


def test_filings_number_must_match_whole_number():
    from types import SimpleNamespace
    from src.agents import filings
    c = SimpleNamespace(text="Revenue in 2024 was strong.", page=1)
    data = {"claims": [{"metric": "revenue", "text": "Revenue grew 24%", "evidence_id": "e1"}]}
    good, dropped = filings.validate_llm_claims(data, {"e1": c}, {"e1": c.text})
    assert good == [] and dropped == 1


def test_filings_digits_inside_labels_are_not_numbers():
    from types import SimpleNamespace
    from src.agents import filings
    c = SimpleNamespace(text="Q2 deal wins were strong.", page=1)
    data = {"claims": [{"metric": "total_contract_value", "text": "Q2 TCV grew", "evidence_id": "e1"}]}
    good, dropped = filings.validate_llm_claims(data, {"e1": c}, {"e1": c.text})
    assert good == [] and dropped == 1


def test_filings_period_word_must_be_in_shown_text():
    from types import SimpleNamespace
    from src.agents import filings
    c = SimpleNamespace(text="India revenue grew 6% last quarter.", page=1)
    bad = {"claims": [{"metric": "revenue", "text": "India revenue grew 6% year-on-year", "evidence_id": "e1"}]}
    good, dropped = filings.validate_llm_claims(bad, {"e1": c}, {"e1": c.text})
    assert good == [] and dropped == 1
    c2 = SimpleNamespace(text="India revenue grew 6% year-on-year.", page=1)
    ok = {"claims": [{"metric": "revenue", "text": "India revenue grew 6% year-on-year", "evidence_id": "e1"}]}
    good, dropped = filings.validate_llm_claims(ok, {"e1": c2}, {"e1": c2.text})
    assert len(good) == 1 and dropped == 0
