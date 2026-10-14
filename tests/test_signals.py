"""Tests for the three signals, using small made-up price series where the right answer is known."""
from datetime import date, timedelta

import pandas as pd
import pytest

from src.signals import get_signals
from src.signals.loader import PriceDataError, validate_prices
from src.signals.momentum import compute_momentum
from src.signals.sentiment import compute_sentiment
from src.signals.volume import compute_volume_anomaly

START = date(2026, 1, 1)


def make_prices(closes, volumes=None, start=START):
    """One row per calendar day, which is fine for tests."""
    n = len(closes)
    volumes = volumes or [1000] * n
    return validate_prices(pd.DataFrame({
        "date": [(start + timedelta(days=i)).isoformat() for i in range(n)],
        "open": closes, "high": closes, "low": closes, "close": closes, "volume": volumes,
    }))


def day(i):
    return START + timedelta(days=i)


# ---------------- momentum ----------------
def test_momentum_positive_when_price_rises_10_percent():
    closes = [100.0] * 20 + [110.0]               # 21 rows, last is +10% vs 20 rows earlier
    sig = compute_momentum("T", make_prices(closes), day(20))
    assert sig.label == "positive"
    assert sig.value == pytest.approx(0.10)


def test_momentum_negative_when_price_falls_10_percent():
    sig = compute_momentum("T", make_prices([100.0] * 20 + [90.0]), day(20))
    assert sig.label == "negative"
    assert sig.value == pytest.approx(-0.10)


def test_momentum_neutral_when_flat():
    sig = compute_momentum("T", make_prices([100.0] * 25), day(24))
    assert sig.label == "neutral"
    assert sig.value == pytest.approx(0.0)


def test_momentum_unavailable_with_too_little_history():
    sig = compute_momentum("T", make_prices([100.0] * 10), day(9))
    assert sig.status == "failed" and sig.label == "unavailable"


def test_momentum_ignores_future_prices():
    """Changing prices AFTER as_of must not change the signal (no look-ahead)."""
    base = [100.0] * 20 + [110.0] + [100.0] * 5
    changed = [100.0] * 20 + [110.0] + [500.0] * 5
    a = compute_momentum("T", make_prices(base), day(20))
    b = compute_momentum("T", make_prices(changed), day(20))
    assert a.value == b.value and a.label == b.label


# ---------------- volume ----------------
def test_volume_spike_is_unusual():
    volumes = [1000, 1010, 990, 1005, 995] * 4 + [5000]       # 20 normal days then a spike
    sig = compute_volume_anomaly("T", make_prices([100.0] * 21, volumes), day(20))
    assert sig.label == "unusual"
    assert sig.value > 10


def test_normal_volume_is_usual():
    volumes = [1000, 1010, 990, 1005, 995] * 4 + [1002]
    sig = compute_volume_anomaly("T", make_prices([100.0] * 21, volumes), day(20))
    assert sig.label == "usual"


def test_volume_baseline_excludes_the_day_itself():
    """If today's spike leaked into the baseline, the z-score would be much smaller."""
    volumes = [1000, 1010, 990, 1005, 995] * 4 + [5000]
    sig = compute_volume_anomaly("T", make_prices([100.0] * 21, volumes), day(20))
    baseline = pd.Series(volumes[:20], dtype=float)
    expected = (5000 - baseline.mean()) / baseline.std(ddof=1)
    assert sig.value == pytest.approx(expected, rel=1e-3)


def test_volume_ignores_future_rows():
    volumes_a = [1000, 1010, 990, 1005, 995] * 4 + [1002] + [9999] * 3
    volumes_b = [1000, 1010, 990, 1005, 995] * 4 + [1002] + [1] * 3
    a = compute_volume_anomaly("T", make_prices([100.0] * 24, volumes_a), day(20))
    b = compute_volume_anomaly("T", make_prices([100.0] * 24, volumes_b), day(20))
    assert a.value == b.value


def test_flat_volume_baseline_is_handled_not_crashed():
    sig = compute_volume_anomaly("T", make_prices([100.0] * 21, [1000] * 21), day(20))
    assert sig.label == "unavailable" and sig.status == "degraded"


# ---------------- stale data ----------------
def test_old_data_is_marked_stale():
    sig = compute_momentum("T", make_prices([100.0] * 25), day(24) + timedelta(days=30))
    assert sig.data_status == "stale" and sig.status == "degraded"


# ---------------- sentiment ----------------
def test_sentiment_positive_negative_neutral():
    d = date(2026, 5, 1)
    pos = compute_sentiment("T", [("e1", "Record profit and strong growth", d)], d)
    neg = compute_sentiment("T", [("e2", "Loss widened, debt and default risk", d)], d)
    mix = compute_sentiment("T", [("e3", "Profit grew but debt and losses mounted", d)], d)
    assert pos.label == "positive" and neg.label == "negative"
    assert mix.label == "neutral"
    assert pos.evidence_ids == ["e1"]


def test_sentiment_ignores_text_published_after_as_of():
    as_of = date(2026, 5, 1)
    items = [("old", "strong growth", date(2026, 4, 1)), ("new", "huge fraud and loss", date(2026, 6, 1))]
    sig = compute_sentiment("T", items, as_of)
    assert sig.label == "positive" and sig.evidence_ids == ["old"]


def test_sentiment_without_text_is_unavailable_and_without_words_uncertain():
    d = date(2026, 5, 1)
    assert compute_sentiment("T", [], d).status == "failed"
    assert compute_sentiment("T", [("e", "The meeting is on Tuesday", d)], d).label == "uncertain"


# ---------------- loader and get_signals ----------------
def test_loader_rejects_duplicate_dates():
    df = pd.DataFrame({"date": ["2026-01-01", "2026-01-01"], "open": [1, 1], "high": [1, 1],
                       "low": [1, 1], "close": [1, 1], "volume": [1, 1]})
    with pytest.raises(PriceDataError):
        validate_prices(df)


def test_get_signals_end_to_end_from_csv(tmp_path):
    closes = [100.0] * 20 + [110.0]
    volumes = [1000, 1010, 990, 1005, 995] * 4 + [5000]
    df = pd.DataFrame({"date": [(START + timedelta(days=i)).isoformat() for i in range(21)],
                       "open": closes, "high": closes, "low": closes, "close": closes, "volume": volumes})
    df.to_csv(tmp_path / "TEST.csv", index=False)
    sigs = get_signals("TEST", day(20), texts=[("e1", "strong growth", day(0))], price_dir=tmp_path)
    by_name = {s.name: s for s in sigs}
    assert by_name["momentum"].label == "positive"
    assert by_name["volume_anomaly"].label == "unusual"
    assert by_name["sentiment"].label == "positive"
    assert all(s.status == "complete" for s in sigs)


def test_missing_price_file_degrades_instead_of_crashing(tmp_path):
    """This is test T2 from the plan: a missing price file must not crash the system."""
    sigs = get_signals("NOPE", day(20), texts=[("e1", "strong growth", day(0))], price_dir=tmp_path)
    by_name = {s.name: s for s in sigs}
    assert by_name["momentum"].status == "failed" and by_name["momentum"].data_status == "unavailable"
    assert by_name["volume_anomaly"].status == "failed"
    assert by_name["sentiment"].label == "positive"       # sentiment still works
