"""Tests for retrieve(): filtering and output shape (not ranking quality).

The query embedding is replaced by a fixed vector, so these tests check the
company filter, the as_of rule, ordering and the Citation fields only.
Recall quality is measured separately with labelled questions.
"""
from datetime import date

import numpy as np
import pytest

import src.rag.retrieve as r
from src.contracts import Citation

DIM = 384


@pytest.fixture(autouse=True)
def fixed_query_vector(monkeypatch):
    v = np.random.RandomState(0).randn(1, DIM).astype("float32")
    v /= np.linalg.norm(v)
    monkeypatch.setattr(r, "_embed_query", lambda q: v)
    monkeypatch.delenv("VIBE_FAKE_RETRIEVE", raising=False)


def test_returns_k_citations_sorted_best_first():
    out = r.retrieve("anything", "TCS", date(2026, 12, 31), k=5)
    assert len(out) == 5
    assert all(isinstance(c, Citation) for c in out)
    scores = [c.retrieval_score for c in out]
    assert scores == sorted(scores, reverse=True)


def test_citation_fields_are_filled():
    c = r.retrieve("anything", "TCS", date(2026, 12, 31), k=1)[0]
    assert c.evidence_id.startswith("doc:") and c.page >= 1
    assert c.text and c.title and c.data_status == "historical"


def test_unknown_company_returns_empty():
    assert r.retrieve("anything", "INFYX", date(2026, 12, 31)) == []


def test_company_is_case_insensitive():
    assert r.retrieve("anything", "tcs", date(2026, 12, 31), k=2)


def test_as_of_before_every_document_returns_empty():
    assert r.retrieve("anything", "TCS", date(2026, 5, 14)) == []


def test_as_of_excludes_later_documents():
    out = r.retrieve("anything", "TCS", date(2026, 6, 1), k=20)
    assert out and all(c.doc_id == "TCS-AR-FY26" for c in out)
    assert all(c.publication_date <= date(2026, 6, 1) for c in out)


def test_as_of_accepts_iso_string():
    assert r.retrieve("anything", "TCS", "2026-12-31", k=1)


def test_blank_query_returns_empty():
    assert r.retrieve("   ", "TCS", date(2026, 12, 31)) == []


def test_fake_switch_returns_fixture(monkeypatch):
    monkeypatch.setenv("VIBE_FAKE_RETRIEVE", "1")
    out = r.retrieve("anything", "TCS", date(2026, 12, 31))
    assert len(out) == 1 and out[0].evidence_id == "doc:DEMO-AR-2026:p14:c03"