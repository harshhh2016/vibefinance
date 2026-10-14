"""Price momentum: the return over a fixed lookback, labelled by written thresholds.

Method (fixed before any evaluation, do not tune on test data):
  return = close[t] / close[t - LOOKBACK] - 1, using only rows dated on or before as_of
  positive if return >= +THRESHOLD, negative if return <= -THRESHOLD, otherwise neutral
"""
from datetime import date

import pandas as pd

from src.contracts import Signal
from src.signals.loader import data_status_for, up_to

LOOKBACK = 20          # trading days
THRESHOLD = 0.03       # 3 percent


def compute_momentum(symbol: str, df: pd.DataFrame, as_of: date,
                     lookback: int = LOOKBACK, threshold: float = THRESHOLD) -> Signal:
    hist = up_to(df, as_of)
    status = data_status_for(hist, as_of)
    method = f"{lookback}-trading-day return, thresholds +/-{threshold:.0%}"
    if len(hist) <= lookback:
        return Signal(symbol=symbol, as_of=as_of, name="momentum", value=None, unit="pct_return",
                      label="unavailable", confidence=0.0, method=method,
                      data_status="unavailable", status="failed")
    last, first = hist.iloc[-1], hist.iloc[-1 - lookback]
    ret = float(last["close"] / first["close"] - 1)
    label = "positive" if ret >= threshold else "negative" if ret <= -threshold else "neutral"
    # heuristic score, NOT a calibrated probability: larger moves give more confidence, capped
    confidence = 0.5 if label == "neutral" else min(0.9, 0.5 + 0.2 * min(abs(ret) / threshold, 2.0))
    return Signal(
        symbol=symbol, as_of=as_of, name="momentum", value=round(ret, 6), unit="pct_return",
        label=label, confidence=round(confidence, 3), method=method,
        window_start=first["date"], window_end=last["date"],
        evidence_ids=[f"px:{symbol}:{last['date']}", f"px:{symbol}:{first['date']}"],
        data_status=status, status="degraded" if status == "stale" else "complete",
    )
