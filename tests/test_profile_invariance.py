from datetime import date
from pathlib import Path

from src.contracts import RunRecord
from src.portfolio.apply import apply_profile, load_latest_prices
from src.portfolio.store import DEMO_PROFILES

FIXTURE = Path(__file__).parent / "fixtures" / "sample_run.json"
PRICES = {  # symbol -> (close, date); synthetic round numbers
    "HDFCBANK": (1000.0, date(2026, 10, 14)), "ICICIBANK": (1000.0, date(2026, 10, 14)),
    "ITC": (400.0, date(2026, 10, 14)), "TCS": (2300.0, date(2026, 10, 14)),
    "INFY": (1500.0, date(2026, 10, 14)), "RELIANCE": (1400.0, date(2026, 10, 14)),
    "SBIN": (800.0, date(2026, 10, 14)),
}


def _record():
    return RunRecord.model_validate_json(FIXTURE.read_text(encoding="utf-8"))


def _facts(r):
    return {
        "signals": [s.model_dump(mode="json") for s in r.signals],
        "findings": [f.model_dump(mode="json") for f in r.findings],
        "efce": r.efce.model_dump(mode="json") if r.efce else None,
    }


def _apply_both():
    rec = _record()
    cons, aggr = DEMO_PROFILES
    return rec, apply_profile(rec, cons, PRICES), apply_profile(rec, aggr, PRICES)


def test_applying_profile_leaves_facts_identical():
    rec, a, b = _apply_both()
    assert _facts(a) == _facts(b) == _facts(rec)
    assert a.efce.status == b.efce.status == rec.efce.status


def test_profiles_change_only_risk_output():
    _rec, a, b = _apply_both()
    pa, pb = a.config["portfolio"], b.config["portfolio"]
    assert pa["risk_note"] != pb["risk_note"]
    assert pa["weights"] != pb["weights"] and a.metrics["hhi"] != b.metrics["hhi"]
    flags = lambda p: {(x["kind"], x["name"]) for x in p["breaches"]}
    assert flags(pa) == {("position", "ITC")}
    assert flags(pb) == {("position", "TCS"), ("sector", "IT")}
    assert a.profile_id == "conservative" and b.profile_id == "aggressive"


def test_original_record_is_not_mutated():
    rec, _a, _b = _apply_both()
    assert "portfolio" not in rec.config and "hhi" not in rec.metrics


def test_prices_use_latest_close_on_or_before_as_of(tmp_path):
    (tmp_path / "AAA.csv").write_text(
        "date,open,high,low,close,volume\n"
        "2026-10-12,1,1,1,10.0,1\n2026-10-13,1,1,1,11.0,1\n2026-10-14,1,1,1,99.0,1\n",
        encoding="utf-8")
    got = load_latest_prices(["AAA"], date(2026, 10, 13), tmp_path)
    assert got == {"AAA": (11.0, date(2026, 10, 13))}


def test_missing_price_file_is_skipped(tmp_path):
    assert load_latest_prices(["NOPE"], date(2026, 10, 14), tmp_path) == {}
