from src.ui_panels import (SYNTHETIC_NOTE, metrics_view, portfolio_view,
                           render_metrics, render_portfolio)

RUN = {
    "findings": [{}, {}, {}],
    "metrics": {"end_to_end_ms": 20123.4, "agents_complete": 3, "citation_pass_rate": 1.0,
                "latency_ms_technical": 12.0, "latency_ms_filings": 19000.0, "hhi": 0.3},
    "config": {"portfolio": {
        "profile_id": "conservative", "risk_preference": "low", "synthetic_profile": True,
        "total_value": 1234567.8, "hhi": 0.31,
        "weights": {"TCS": 0.2, "INFY": 0.5, "ITC": 0.3},
        "sector_weights": {"IT": 0.7, "FMCG": 0.3},
        "breaches": [{"kind": "position", "name": "INFY", "weight_pct": 50.0, "limit_pct": 30.0}],
        "missing_prices": ["HDFCBANK"], "price_dates": {"TCS": "2026-10-14"},
        "risk_note": "Lower risk tolerance."}},
}


class Fake:
    """Records every st call; columns/expanders behave like the real ones."""
    def __init__(self, log=None):
        self.log = log if log is not None else []

    def __getattr__(self, name):
        def call(*a, **k):
            self.log.append((name, a))
            return [Fake(self.log) for _ in range(a[0] if isinstance(a[0], int) else len(a[0]))] if name == "columns" else None
        return call


def test_portfolio_view_sorted_and_formatted():
    v = portfolio_view(RUN)
    assert [p["Symbol"] for p in v["positions"]] == ["INFY", "ITC", "TCS"]
    assert v["positions"][0]["Weight %"] == 50.0 and v["positions"][2]["Price date"] == "2026-10-14"
    assert v["positions"][0]["Price date"] == "n/a"
    assert v["breaches"] == ["Position INFY is 50.0%, above the 30.0% limit."]
    assert v["synthetic"] is True and v["missing_prices"] == ["HDFCBANK"]


def test_portfolio_view_none_without_profile():
    assert portfolio_view({"config": {}}) is None
    assert portfolio_view({}) is None


def test_metrics_view_values_and_missing_keys():
    d = dict(metrics_view(RUN))
    assert d["End-to-end time"] == "20,123 ms" and d["Agents complete"] == "3 of 3"
    assert d["Citation pass rate"] == "100%" and d["Filings latency"] == "19,000 ms"
    empty = dict(metrics_view({}))
    assert empty["End-to-end time"] == empty["Agents complete"] == empty["Citation pass rate"] == "n/a"


def test_render_portfolio_shows_synthetic_label_and_note():
    st = Fake()
    render_portfolio(st, RUN)
    names = [c[0] for c in st.log]
    assert ("warning", (SYNTHETIC_NOTE,)) in st.log
    assert "table" in names and names.count("warning") == 2  # synthetic note + one breach


def test_render_portfolio_without_profile_does_not_crash():
    st = Fake()
    render_portfolio(st, {"config": {}})
    assert [c[0] for c in st.log] == ["markdown", "info"]


def test_render_metrics_runs_and_has_heuristic_caption():
    st = Fake()
    render_metrics(st, RUN)
    assert any(c[0] == "caption" and "heuristic" in c[1][0] for c in st.log)
