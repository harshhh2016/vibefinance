from src.eval.baselines import synthesize_b1, synthesize_b2


def cl(cid, d, conf=0.5, period="p", metric="m"):
    return {"claim_id": cid, "entity": "TCS", "metric_or_event": metric, "period": period,
            "direction": d, "claim_type": "factual", "text": cid, "evidence_ids": ["x"], "confidence": conf}


def fnd(*claims, agent="a"):
    return {"agent": agent, "status": "complete", "claims": list(claims)}


def test_b1_passes_everything_and_never_flags():
    r = synthesize_b1([fnd(cl("a", "positive"), cl("b", "negative")), fnd(cl("c", "neutral", metric="z"), agent="b")])
    assert [c["claim_id"] for c in r["supported"]] == ["a", "b", "c"]
    assert r["conflicts"] == [] and r["rejected"] == [] and r["abstained"] == [] and r["status"] == "complete"


def test_b1_status_failed_when_no_claims():
    assert synthesize_b1([fnd()])["status"] == "failed"


def test_b2_majority_wins_and_singletons_pass():
    f = [fnd(cl("a", "positive"), cl("b", "positive"), cl("c", "negative"), cl("s", "neutral", metric="solo"))]
    ids = [c["claim_id"] for c in synthesize_b2(f)["supported"]]
    assert ids == ["a", "b", "s"]


def test_b2_tie_keeps_higher_confidence_and_picks_a_winner():
    f = [fnd(cl("a", "positive", 0.4), cl("b", "negative", 0.7))]
    r = synthesize_b2(f)
    assert [c["claim_id"] for c in r["supported"]] == ["b"] and r["conflicts"] == []


def test_b2_different_periods_are_not_voted():
    f = [fnd(cl("a", "positive", period="YoY"), cl("b", "negative", period="QoQ"))]
    assert [c["claim_id"] for c in synthesize_b2(f)["supported"]] == ["a", "b"]
