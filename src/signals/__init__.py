"""get_signals is the one function the rest of the project calls."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from src.contracts import Signal
from src.signals.loader import PriceDataError, load_prices
from src.signals.momentum import compute_momentum
from src.signals.sentiment import compute_sentiment
from src.signals.volume import compute_volume_anomaly


def _failed(symbol: str, as_of: date, name: str, why: str) -> Signal:
    return Signal(symbol=symbol, as_of=as_of, name=name, value=None, unit="", label="unavailable",
                  confidence=0.0, method=why, data_status="unavailable", status="failed")


def get_signals(symbol: str, as_of: date, texts: list[tuple[str, str, date]] | None = None,
                price_dir: str | Path = "data/prices") -> list[Signal]:
    """Return momentum, volume_anomaly and sentiment for one stock on one date.

    A missing or broken price file makes the two price signals 'failed' instead of crashing;
    sentiment still runs because it does not depend on prices.
    texts is a list of (evidence_id, text, published_date).
    """
    try:
        df = load_prices(symbol, price_dir)
        price_signals = [compute_momentum(symbol, df, as_of), compute_volume_anomaly(symbol, df, as_of)]
    except (FileNotFoundError, PriceDataError) as exc:
        why = f"price data unavailable: {exc}"
        price_signals = [_failed(symbol, as_of, "momentum", why), _failed(symbol, as_of, "volume_anomaly", why)]
    return price_signals + [compute_sentiment(symbol, texts or [], as_of)]
