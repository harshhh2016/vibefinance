"""Download daily prices for your chosen stocks and save them as CSV files.

Usage (from the project root, with the virtual environment active):
    python -m src.download_prices RELIANCE TCS INFY HDFCBANK ICICIBANK

Symbols are plain NSE symbols; ".NS" is added automatically.
Files go to data/prices/<SYMBOL>.csv and a manifest records where they came from.
"""
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

PRICE_DIR = Path("data/prices")
COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def tidy(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn a yfinance result into a clean table: date, open, high, low, close, volume."""
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):       # newer yfinance versions return 2 header rows
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={"price": "date", "datetime": "date"})
    df = df[COLUMNS].dropna()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def download(symbol: str, period: str = "1y") -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(f"{symbol}.NS", period=period, interval="1d",
                      auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"No data returned for {symbol}.NS")
    return tidy(raw)


def main(symbols: list[str]) -> None:
    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = PRICE_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for sym in symbols:
        try:
            df = download(sym)
        except Exception as exc:                      # keep going if one symbol fails
            print(f"FAILED {sym}: {exc}")
            continue
        df.to_csv(PRICE_DIR / f"{sym}.csv", index=False)
        manifest[sym] = {
            "source": "Yahoo Finance via yfinance",
            "ticker": f"{sym}.NS",
            "downloaded": date.today().isoformat(),
            "rows": len(df),
            "first_date": df["date"].iloc[0],
            "last_date": df["date"].iloc[-1],
            "data_status": "historical",
        }
        print(f"{sym}: {len(df)} rows, {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Give at least one symbol, e.g. python -m src.download_prices RELIANCE TCS")
    main(sys.argv[1:])
