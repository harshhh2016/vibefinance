"""Load and check price files. Every signal uses this, so bad data is caught in one place."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

REQUIRED = ["date", "open", "high", "low", "close", "volume"]
STALE_AFTER_DAYS = 5          # price data older than this (calendar days) is marked stale


class PriceDataError(ValueError):
    """Raised when a price file is missing columns, has duplicates, or has impossible values."""


def load_prices(symbol: str, price_dir: str | Path = "data/prices") -> pd.DataFrame:
    """Read data/prices/<symbol>.csv and validate it. Raises FileNotFoundError or PriceDataError."""
    path = Path(price_dir) / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No price file for {symbol}: {path}")
    df = pd.read_csv(path)
    return validate_prices(df)


def validate_prices(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise PriceDataError(f"Missing columns: {missing}")
    df = df[REQUIRED].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    if df.isna().any().any():
        raise PriceDataError("Price file has empty values")
    if df["date"].duplicated().any():
        raise PriceDataError("Price file has duplicate dates")
    if (df[["open", "high", "low", "close"]] <= 0).any().any() or (df["volume"] < 0).any():
        raise PriceDataError("Price file has non-positive prices or negative volume")
    return df.sort_values("date").reset_index(drop=True)


def up_to(df: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Only rows dated on or before as_of. This is what prevents look-ahead leakage."""
    return df[df["date"] <= as_of].reset_index(drop=True)


def data_status_for(df_until: pd.DataFrame, as_of: date) -> str:
    """'historical' normally, 'stale' if the newest row is much older than as_of."""
    if df_until.empty:
        return "unavailable"
    age = (as_of - df_until["date"].iloc[-1]).days
    return "stale" if age > STALE_AFTER_DAYS else "historical"
