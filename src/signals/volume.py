"""Volume anomaly: a rolling z-score against the PREVIOUS sessions only.

Method (fixed before any evaluation):
  baseline = the WINDOW trading days before as_of (as_of itself is not in the baseline)
  z = (volume[as_of] - mean(baseline)) / std(baseline)
  unusual if |z| >= Z_THRESHOLD, otherwise usual
"""
from datetime import date

import pandas as pd

from src.contracts import Signal
from src.signals.loader import data_status_for, up_to

WINDOW = 20
Z_THRESHOLD = 2.0


def compute_volume_anomaly(symbol: str, df: pd.DataFrame, as_of: date,
                           window: int = WINDOW, z_threshold: float = Z_THRESHOLD) -> Signal:
    hist = up_to(df, as_of)
    status = data_status_for(hist, as_of)
    method = f"z-score of volume against the previous {window} sessions, unusual if |z| >= {z_threshold}"
    if len(hist) < window + 1:
        return Signal(symbol=symbol, as_of=as_of, name="volume_anomaly", value=None,
                      unit="rolling_z_score", label="unavailable", confidence=0.0, method=method,
                      data_status="unavailable", status="failed")
    today = hist.iloc[-1]
    baseline = hist.iloc[-1 - window:-1]["volume"].astype(float)
    std = float(baseline.std(ddof=1))
    if std == 0.0:      # a perfectly flat baseline cannot give a meaningful z-score
        return Signal(symbol=symbol, as_of=as_of, name="volume_anomaly", value=None,
                      unit="rolling_z_score", label="unavailable", confidence=0.0, method=method,
                      window_start=hist.iloc[-1 - window]["date"],
                      window_end=today["date"], data_status=status, status="degraded")
    z = float((float(today["volume"]) - float(baseline.mean())) / std)
    label = "unusual" if abs(z) >= z_threshold else "usual"
    confidence = 0.5 if label == "usual" else min(0.9, 0.5 + 0.1 * (abs(z) - z_threshold + 1))
    return Signal(
        symbol=symbol, as_of=as_of, name="volume_anomaly", value=round(z, 4), unit="rolling_z_score",
        label=label, confidence=round(confidence, 3), method=method,
        window_start=hist.iloc[-1 - window]["date"], window_end=today["date"],
        evidence_ids=[f"px:{symbol}:{today['date']}"],
        data_status=status, status="degraded" if status == "stale" else "complete",
    )
