import json

from src.eval.scoring import EvidenceStore
from src.evidence_view import resolve, resolve_many

LONG = "word " * 400  # 2000 chars: must not be truncated


def _store(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    rec = {"evidence_id": "doc:TCS-PR-Q2FY27:p001:c01", "doc_id": "TCS-PR-Q2FY27",
           "symbol": "TCS", "doc_type": "press_release", "page": 1, "chunk": 1,
           "publication_date": "2026-10-08", "data_status": "real",
           "source_url": "https://example.test/pr", "n_words": 400, "text": LONG}
    chunks.write_text("\ufeff" + json.dumps(rec) + "\n", encoding="utf-8")
    prices = tmp_path / "prices"
    prices.mkdir()
    (prices / "TCS.csv").write_text(
        "date,open,high,low,close,volume\n2026-10-14,2274.0,2335.8,2265.6,2304.9,12145448\n",
        encoding="utf-8")
    return EvidenceStore(chunks_path=chunks, prices_dir=prices)


def test_doc_full_text_and_metadata(tmp_path):
    r = resolve("doc:TCS-PR-Q2FY27:p001:c01", _store(tmp_path))
    assert r["found"] and r["kind"] == "doc"
    assert r["title"] == "TCS-PR-Q2FY27, page 1"
    assert r["date"] == "2026-10-08"
    assert r["text"] == LONG and len(r["text"]) == 2000
    assert r["source_url"] == "https://example.test/pr"


def test_price_row_fields(tmp_path):
    r = resolve("px:TCS:2026-10-14", _store(tmp_path))
    assert r["found"] and r["kind"] == "price"
    assert r["title"] == "TCS daily price row, 2026-10-14"
    assert r["fields"]["close"] == "2304.9" and r["fields"]["volume"] == "12145448"
    assert r["text"].startswith("open 2274.0 | high 2335.8")


def test_unknown_and_malformed_ids_do_not_raise(tmp_path):
    s = _store(tmp_path)
    for bad in ("doc:NOPE:p999:c99", "px:TCS:2026-10-15", "px:TCS", "px:XYZ:2026-10-14", "", "px:TCS:not-a-date"):
        r = resolve(bad, s)
        assert r["found"] is False and r["text"] == "" and r["kind"] == "unknown"


def test_resolve_many_keeps_order_and_drops_duplicates(tmp_path):
    out = resolve_many(["px:TCS:2026-10-14", "doc:TCS-PR-Q2FY27:p001:c01", "px:TCS:2026-10-14", "doc:X"],
                       _store(tmp_path))
    assert [o["evidence_id"] for o in out] == ["px:TCS:2026-10-14", "doc:TCS-PR-Q2FY27:p001:c01", "doc:X"]
    assert [o["found"] for o in out] == [True, True, False]
    assert resolve_many(None) == [] and resolve_many([]) == []
