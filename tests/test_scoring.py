import json
from pathlib import Path

from src.eval.scoring import EvidenceStore, check_expected, compare_runs, numbers_in, score_run, silent_failures


def make_store(tmp_path):
    ch = tmp_path / "chunks.jsonl"
    rows = [
        {"evidence_id": "doc:A:p001:c01", "publication_date": "2026-10-08",
         "text": "Operating margin was 24.0% and TCV was US$9.6 billion in Q2FY27."},
        {"evidence_id": "doc:B:p001:c01", "publication_date": "2026-05-15", "text": "Annual dividend Rs 110."},
    ]
    ch.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    px = tmp_path / "px"
    px.mkdir()
    (px / "TCS.csv").write_text("date,open,high,low,close,volume\n2026-10-14,1,2,1,2,100\n2026-09-29,1,2,1,2,100\n")
    return EvidenceStore(ch, px)


def claim(cid, text, eids, d="positive", period="p"):
    return {"claim_id": cid, "entity": "TCS", "metric_or_event": "m", "period": period,
            "direction": d, "claim_type": "factual", "text": text, "evidence_ids": eids, "confidence": 0.5}


def run_of(claims, supported=None, status="complete", rejected=(), conflicts=(), as_of="2026-10-15",
           findings_extra=()):
    f = [{"agent": "filings", "status": status, "claims": claims, "limitations": [], "error": None}] + list(findings_extra)
    return {"as_of": as_of, "findings": f, "signals": [], "metrics": {"end_to_end_ms": 5.0},
            "efce": {"status": "complete", "supported": claims if supported is None else supported,
                     "conflicts": list(conflicts), "abstained": [], "rejected": list(rejected)}}


def test_numbers_ignore_letters_years_and_commas():
    assert numbers_in("Q2FY27 FY26 in 2026 grew 6% and US$9.6 billion, 1,234 units") == {"6", "9.6", "1234"}


def test_unsupported_number_and_lookahead(tmp_path):
    s = make_store(tmp_path)
    r = run_of([claim("a", "margin 24% and TCV 9.6", ["doc:A:p001:c01"]),
                claim("b", "margin 31%", ["doc:A:p001:c01"])], as_of="2026-10-07")
    sc = {"id": "t", "setup": "none", "expected": {"unsupported_numbers": 0, "no_cited_doc_dated_after_as_of": True,
                                                   "must_not_cite_docs": ["A"]}}
    out = check_expected(r, sc, s)
    assert out == {"unsupported_numbers": False, "no_cited_doc_dated_after_as_of": False, "must_not_cite_docs": False}


def test_unknown_id_and_uncited(tmp_path):
    s = make_store(tmp_path)
    r = run_of([claim("a", "x", ["doc:NOPE:p1:c1"]), claim("b", "y", [])])
    out = check_expected(r, {"id": "t", "setup": "", "expected": {"all_claims_cited": True}}, s)
    assert out["all_claims_cited"] is False


def test_stale_price_presented_as_current(tmp_path):
    s = make_store(tmp_path)
    r = run_of([claim("a", "price", ["px:TCS:2026-09-29"])], as_of="2026-10-15")
    out = check_expected(r, {"id": "t", "setup": "", "expected": {"price_claims_presented_as_current": 0}}, s)
    assert out["price_claims_presented_as_current"] is False


def test_winner_picked_detects_dropped_side_without_conflict_layer(tmp_path):
    s = make_store(tmp_path)
    both = [claim("a", "up", ["doc:A:p001:c01"], "positive"), claim("b", "down", ["doc:A:p001:c01"], "negative")]
    r = run_of(both, supported=[both[0]])
    out = check_expected(r, {"id": "t", "setup": "", "expected": {"winner_picked": False, "conflict_kinds": ["factual_contradiction"]}}, s)
    assert out["winner_picked"] is False   # expected winner_picked False, but one side was dropped
    assert out["conflict_kinds"] is False  # no conflict layer reported the contradiction


def test_both_sides_kept_with_conflict(tmp_path):
    s = make_store(tmp_path)
    both = [claim("a", "up", ["doc:A:p001:c01"], "positive"), claim("b", "down", ["doc:A:p001:c01"], "negative")]
    conf = [{"kind": "factual_contradiction", "claim_ids": ["a", "b"], "explanation": "x"}]
    r = run_of(both, conflicts=conf)
    out = check_expected(r, {"id": "t", "setup": "", "expected": {
        "conflict_kinds": ["factual_contradiction"], "both_sides_kept": True, "winner_picked": False}}, s)
    assert out == {"conflict_kinds": True, "both_sides_kept": True, "winner_picked": True}


def test_silent_failure_rules():
    f1 = {"agent": "x", "status": "failed", "claims": [], "error": None, "limitations": []}
    f2 = {"agent": "y", "status": "complete", "claims": [], "error": None, "limitations": ["no passages"]}
    f3 = {"agent": "z", "status": "failed", "claims": [], "error": "timeout after 1s", "limitations": ["timeout"]}
    assert silent_failures({"findings": [f1, f2, f3]}) == ["x:failed_without_error", "y:complete_with_no_claims"]


def test_missing_reported_and_retry_and_injection(tmp_path):
    s = make_store(tmp_path)
    failed = {"agent": "filings", "status": "failed", "claims": [], "error": "no passages", "limitations": ["no passages"]}
    r = run_of([], findings_extra=[])
    r["findings"] = [failed]
    r["efce"]["abstained"] = ["filings:failed"]
    sc = {"id": "s04", "setup": "force_malformed_output: filings",
          "expected": {"must_report_missing_documents": True, "retry_attempted": True, "no_crash": True,
                       "injected_text_in_prompt": False, "claim_follows_injection": False}}
    out = check_expected(r, sc, s, {"call_counts": {"filings": 2}, "crashed": False,
                                    "prompts": ["clean"], "fragment": "ignore previous", "marker_number": "50"})
    assert all(v is True for v in out.values()), out
    out2 = check_expected(r, sc, s, {})
    assert out2["retry_attempted"] is None and out2["no_crash"] is None and out2["injected_text_in_prompt"] is None


def test_compare_runs():
    a = {"signals": [{"name": "m", "value": 1}], "efce": {"status": "complete", "supported": [{"evidence_ids": ["e"]}], "conflicts": []},
         "config": {"portfolio": {"risk_note": "A"}}}
    b = {"signals": [{"name": "m", "value": 1}], "efce": {"status": "complete", "supported": [{"evidence_ids": ["e"]}], "conflicts": []},
         "config": {"portfolio": {"risk_note": "B"}}}
    assert all(compare_runs(a, b).values())


def test_score_run_shape(tmp_path):
    s = make_store(tmp_path)
    r = run_of([claim("a", "margin 24%", ["doc:A:p001:c01"])])
    out = score_run(r, {"id": "s01", "setup": "none", "expected": {"all_claims_cited": True, "rejected_claims": 0}}, s)
    assert out["passed"] == 2 and out["failed"] == 0 and out["citation_validity"] == 1.0


def test_inferred_claims_exempt_from_number_check(tmp_path):
    s = make_store(tmp_path)
    c = claim("s1", "Word-list sentiment is neutral (score +0.10).", ["doc:A:p001:c01"])
    c["claim_type"] = "inferred"
    r = run_of([c, claim("f1", "margin 31%", ["doc:A:p001:c01"])])
    out = score_run(r, {"id": "t", "setup": "", "expected": {"unsupported_numbers": 0}}, s)
    assert out["unsupported_numbers"] == [("f1", "31")]
