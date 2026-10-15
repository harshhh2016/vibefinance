import pytest

from src.scenarios import india_revenue as sc

# Synthetic mini-chunks that reproduce the real source wording, so this test always runs.
CHUNKS = [
    {"doc_id": sc.PR, "evidence_id": "doc:TCS-PR-Q2FY27:p002:c01", "page": 2, "publication_date": "2026-10-08",
     "text": "Asia Pacific 8.3 8.4 8.6 2.0 4.1\nIndia 5.8 6.2 5.5 - 10.3 6.0\nMEA 2.4 2.5 2.5 0.4 2.0"},
    {"doc_id": sc.TR, "evidence_id": "doc:TCS-TR-Q2FY27:p002:c01", "page": 2, "publication_date": "2026-10-08",
     "text": "India revenue grew 6%, and Regional Markets grew 3.6% year-on-year."},
    {"doc_id": sc.TR, "evidence_id": "doc:TCS-TR-Q2FY27:p012:c02", "page": 12, "publication_date": "2026-10-08",
     "text": "we've seen the India revenue decline by about 10% and also a slight decline in clients"},
]


def test_india_scenario_is_a_temporal_difference_not_a_contradiction():
    finding, res = sc.run(chunks=CHUNKS)
    assert [c.claim_id for c in finding.claims] == ["i1", "i2", "i3", "i4"]
    assert "-10.3%" in finding.claims[0].text and "+6.0%" in finding.claims[1].text
    assert res.rejected == []
    assert all(c.evidence_ids for c in finding.claims)
    assert [k.kind for k in res.conflicts] == ["temporal_difference"]
    assert res.conflicts[0].claim_ids == ["i1", "i2", "i3", "i4"]


def test_india_scenario_on_real_chunks():
    if not sc.CHUNKS.exists():
        pytest.skip("chunks.jsonl not present")
    finding, res = sc.run()
    assert "-10.3%" in finding.claims[0].text and "+6.0%" in finding.claims[1].text
    assert res.rejected == []
    assert [k.kind for k in res.conflicts] == ["temporal_difference"]
