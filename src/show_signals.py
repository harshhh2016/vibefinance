"""Print the three signals for one or more stocks, as of their latest price date.

Usage (from the project root):
    python -m src.show_signals TCS
    python -m src.show_signals TCS INFY RELIANCE
    python -m src.show_signals TCS --as-of 2026-09-30
"""
import argparse
from datetime import date

from src.signals import get_signals
from src.signals.loader import load_prices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("symbols", nargs="+")
    parser.add_argument("--as-of", help="YYYY-MM-DD, default is the latest date in the file")
    args = parser.parse_args()

    for sym in args.symbols:
        try:
            as_of = date.fromisoformat(args.as_of) if args.as_of else load_prices(sym)["date"].iloc[-1]
        except FileNotFoundError:
            print(f"\n{sym}: no file in data/prices, run the downloader first")
            continue
        print(f"\n{sym} as of {as_of}")
        for s in get_signals(sym, as_of):
            value = "n/a" if s.value is None else f"{s.value:.4f}"
            print(f"  {s.name:15} {s.label:12} value={value:>10}  conf={s.confidence:.2f}  [{s.status}, {s.data_status}]")


if __name__ == "__main__":
    main()
