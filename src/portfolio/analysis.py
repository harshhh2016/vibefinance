"""Portfolio math and per-profile risk notes. Pure functions, no I/O."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.contracts import Profile

EPS = 1e-9


@dataclass
class PortfolioAnalysis:
    total_value: float
    values: dict = field(default_factory=dict)          # symbol -> value
    weights: dict = field(default_factory=dict)         # symbol -> fraction 0..1
    sector_weights: dict = field(default_factory=dict)  # sector -> fraction 0..1
    hhi: float = 0.0                                    # sum of squared position weights
    breaches: list = field(default_factory=list)        # dicts: kind, name, weight_pct, limit_pct
    missing_prices: list = field(default_factory=list)


def analyse(profile: Profile, prices: dict) -> PortfolioAnalysis:
    values, sector_of, missing = {}, {}, []
    for h in profile.holdings:
        p = prices.get(h.symbol)
        if p is None or p <= 0:
            missing.append(h.symbol)
            continue
        values[h.symbol] = values.get(h.symbol, 0.0) + h.quantity * p
        sector_of[h.symbol] = h.sector
    missing = sorted(set(missing))
    total = sum(values.values())
    if total <= 0:
        return PortfolioAnalysis(0.0, missing_prices=missing)

    weights = {s: v / total for s, v in values.items()}
    sector_weights = {}
    for s, w in weights.items():
        sector_weights[sector_of[s]] = sector_weights.get(sector_of[s], 0.0) + w
    hhi = sum(w * w for w in weights.values())

    breaches = []
    for s in sorted(weights):
        pct = weights[s] * 100
        if pct > profile.max_single_position_pct + EPS:
            breaches.append({"kind": "position", "name": s, "weight_pct": pct,
                             "limit_pct": profile.max_single_position_pct})
    for sec in sorted(sector_weights):
        pct = sector_weights[sec] * 100
        if pct > profile.max_sector_pct + EPS:
            breaches.append({"kind": "sector", "name": sec, "weight_pct": pct,
                             "limit_pct": profile.max_sector_pct})
    return PortfolioAnalysis(total, values, weights, sector_weights, hhi, breaches, missing)


def _hhi_label(hhi: float) -> str:
    # heuristic thresholds, not a standard
    if hhi >= 0.25:
        return "highly concentrated"
    if hhi >= 0.15:
        return "moderately concentrated"
    return "fairly diversified"


_PREF_TEXT = {
    "low": "Lower risk tolerance: weigh large single-name exposure and unusual volume more heavily.",
    "medium": "Medium risk tolerance: balance concentration against the signals shown.",
    "high": "Higher risk tolerance: concentration is acceptable only while it stays inside your own limits.",
}


def risk_note(profile: Profile, a: PortfolioAnalysis, symbol: str | None = None) -> str:
    parts = [_PREF_TEXT[profile.risk_preference]]
    if profile.horizon_months < 12:
        parts.append("Horizon is {} months, which leaves little time to recover from a drawdown.".format(
            profile.horizon_months))
    else:
        parts.append("Horizon is {} months.".format(profile.horizon_months))
    if symbol is not None:
        if symbol in a.weights:
            parts.append("{} is {:.1f}% of this portfolio.".format(symbol, a.weights[symbol] * 100))
        else:
            parts.append("This portfolio holds no {}.".format(symbol))
    for b in a.breaches:
        parts.append("{} {} is {:.1f}%, above your {:.1f}% limit.".format(
            "Position" if b["kind"] == "position" else "Sector", b["name"],
            b["weight_pct"], b["limit_pct"]))
    if a.weights:
        parts.append("Concentration (HHI) is {:.2f}: {}.".format(a.hhi, _hhi_label(a.hhi)))
    if a.missing_prices:
        parts.append("No price available for: {}; excluded from the figures above.".format(
            ", ".join(a.missing_prices)))
    return " ".join(parts)
