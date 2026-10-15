"""Apply an investor profile to a saved run. Adds portfolio numbers and a risk note only;
signals, findings and EFCE output are never touched."""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

from src.contracts import Profile, RunRecord
from src.portfolio.analysis import analyse, risk_note


def load_latest_prices(symbols, as_of: date, price_dir) -> dict:
    """symbol -> (close, price_date): latest close on or before as_of. Missing file or no row: skipped."""
    out = {}
    for sym in sorted(set(symbols)):
        path = Path(price_dir) / (sym + ".csv")
        if not path.exists():
            continue
        best = None
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    d = date.fromisoformat(row["date"][:10])
                    close = float(row["close"])
                except (KeyError, ValueError, TypeError):
                    continue
                if d <= as_of and (best is None or d > best[1]):
                    best = (close, d)
        if best is not None:
            out[sym] = best
    return out


def apply_profile(record: RunRecord, profile: Profile, prices: dict) -> RunRecord:
    a = analyse(profile, {s: p for s, (p, _d) in prices.items()})
    section = {
        "profile_id": profile.profile_id,
        "risk_preference": profile.risk_preference,
        "synthetic_profile": True,
        "total_value": round(a.total_value, 2),
        "weights": {s: round(w, 6) for s, w in a.weights.items()},
        "sector_weights": {s: round(w, 6) for s, w in a.sector_weights.items()},
        "hhi": round(a.hhi, 6),
        "breaches": a.breaches,
        "missing_prices": a.missing_prices,
        "price_dates": {s: d.isoformat() for s, (_p, d) in prices.items()},
        "risk_note": risk_note(profile, a, record.symbol),
    }
    metrics = dict(record.metrics)
    metrics["hhi"] = round(a.hhi, 6)
    metrics["portfolio_breaches"] = float(len(a.breaches))
    metrics["portfolio_total_value"] = round(a.total_value, 2)
    return record.model_copy(deep=True, update={
        "profile_id": profile.profile_id,
        "config": {**record.config, "portfolio": section},
        "metrics": metrics,
    })
