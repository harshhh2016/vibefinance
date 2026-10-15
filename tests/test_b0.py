from datetime import date
from types import SimpleNamespace as NS

from src.eval.b0 import latest_price, parse_claims, run_b0


def px(tmp_path):
    (tmp_path / "TCS.csv").write_text("date,open,high,low,close,volume\n2026-10-13,1,2,1,2,9\n2026-10-14,1,2,1,3,9\n2026-10-20,1,2,1,4,9\n")
    return tmp_path


def test_latest_price_ignores_future_rows(tmp_path):
    eid, text = latest_price("TCS", date(2026, 10, 15), px(tmp_path))
    assert eid == "px:TCS:2026-10-14" and "close 3" in text
    assert latest_price("NOPE", date(2026, 10, 15), tmp_path) is None


def test_parse_maps_labels_and_leaves_bad_labels_uncited():
    data = {"claims": [{"text": "a", "direction": "POSITIVE", "source": "[P1]"},
                       {"text": "b", "direction": "weird", "source": "P9"}, {"text": "  "}, "junk"]}
    cl = parse_claims(data, {"P1": "doc:X:p1:c1"}, "TCS", date(2026, 10, 15))
    assert [c["evidence_ids"] for c in cl] == [["doc:X:p1:c1"], []]
    assert [c["direction"] for c in cl] == ["positive", "uncertain"]
    assert parse_claims("not a dict", {}, "TCS", date(2026, 10, 15)) == []


def test_run_b0_success_and_failure(tmp_path):
    ctx = NS(symbol="TCS", as_of=date(2026, 10, 15), price_dir=str(px(tmp_path)),
             retrieve_fn=lambda q, s, a, k: [NS(evidence_id="doc:X:p1:c1", text="Revenue grew 6%.")])
    sc = {"id": "s01", "symbol": "TCS", "as_of": "2026-10-15"}
    ok = run_b0(sc, ctx, llm_fn=lambda p: {"claims": [{"text": "Revenue grew 6%.", "direction": "positive", "source": "P1"}]})
    assert ok["findings"][0]["status"] == "complete" and ok["efce"]["supported"][0]["evidence_ids"] == ["doc:X:p1:c1"]
    def boom(p): raise TimeoutError("slow")
    bad = run_b0(sc, ctx, llm_fn=boom)
    assert bad["findings"][0]["status"] == "failed" and "TimeoutError" in bad["findings"][0]["error"]
    assert bad["efce"]["status"] == "failed" and bad["efce"]["supported"] == []
