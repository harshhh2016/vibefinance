import pytest

from src.contracts import Holding, Profile
from src.portfolio.analysis import analyse, risk_note
from src.portfolio.store import (DEMO_PROFILES, connect, list_profile_ids, load_profile,
                                 save_profile, seed_demo_profiles)

PRICES = {"A": 100.0, "B": 200.0, "C": 50.0}   # values 1000, 1000, 500; total 2500


def make(single=30.0, sector=50.0, pref="medium"):
    return Profile(profile_id="t", risk_preference=pref, horizon_months=36,
                   max_single_position_pct=single, max_sector_pct=sector, holdings=[
                       Holding(symbol="A", quantity=10, avg_cost=90, sector="IT"),
                       Holding(symbol="B", quantity=5, avg_cost=190, sector="IT"),
                       Holding(symbol="C", quantity=10, avg_cost=45, sector="Banking"),
                   ])


def test_weights_hhi_and_sector_weights():
    a = analyse(make(), PRICES)
    assert a.total_value == pytest.approx(2500)
    assert a.weights["A"] == pytest.approx(0.4)
    assert a.weights["B"] == pytest.approx(0.4)
    assert a.weights["C"] == pytest.approx(0.2)
    assert a.hhi == pytest.approx(0.36)            # 0.16 + 0.16 + 0.04
    assert a.sector_weights["IT"] == pytest.approx(0.8)
    assert a.sector_weights["Banking"] == pytest.approx(0.2)


def test_breaches_flag_positions_and_sector():
    a = analyse(make(single=30.0, sector=50.0), PRICES)
    got = [(b["kind"], b["name"]) for b in a.breaches]
    assert got == [("position", "A"), ("position", "B"), ("sector", "IT")]


def test_exactly_at_limit_is_not_a_breach():
    a = analyse(make(single=40.0, sector=80.0), PRICES)
    assert a.breaches == []


def test_missing_price_is_excluded_and_reported():
    a = analyse(make(), {"A": 100.0, "B": 200.0})
    assert a.missing_prices == ["C"]
    assert a.total_value == pytest.approx(2000)
    assert a.weights["A"] == pytest.approx(0.5)
    assert a.hhi == pytest.approx(0.5)


def test_store_roundtrip_keeps_holding_order():
    conn = connect(":memory:")
    p = make()
    save_profile(conn, p)
    assert load_profile(conn, "t") == p
    assert load_profile(conn, "nope") is None


def test_seed_creates_two_profiles():
    conn = connect(":memory:")
    seed_demo_profiles(conn)
    assert list_profile_ids(conn) == ["aggressive", "conservative"]
    for p in DEMO_PROFILES:
        assert load_profile(conn, p.profile_id) == p


def test_risk_note_states_the_breach():
    p = make(single=30.0)
    note = risk_note(p, analyse(p, PRICES), "A")
    assert "A is 40.0% of this portfolio." in note
    assert "above your 30.0% limit" in note


def test_risk_note_differs_by_preference_but_weights_do_not():
    lo, hi = make(pref="low"), make(pref="high")
    a_lo, a_hi = analyse(lo, PRICES), analyse(hi, PRICES)
    assert a_lo.weights == a_hi.weights and a_lo.hhi == a_hi.hhi
    assert risk_note(lo, a_lo, "A") != risk_note(hi, a_hi, "A")
