"""Technical agent: turns the momentum and volume signals into cited claims."""
from __future__ import annotations

from src.agents.base import AgentContext, AgentResult
from src.contracts import Claim, Finding, Signal
from src.signals import get_signals

DIRECTION = {"positive": "positive", "negative": "negative", "neutral": "neutral"}
METRIC = {"momentum": "price_momentum", "volume_anomaly": "volume_anomaly"}
CLAIM_ID = {"momentum": "t1", "volume_anomaly": "t2"}


def _claim(sig: Signal) -> Claim:
    if sig.name == "momentum":
        text = "20-trading-day return is {:+.2f}% ({}).".format(sig.value * 100, sig.label)
        direction = DIRECTION.get(sig.label, "uncertain")
    else:
        text = "Volume z-score is {:.2f} against the prior 20 sessions ({}).".format(sig.value, sig.label)
        direction = "uncertain" if sig.label == "unusual" else "neutral"
    return Claim(
        claim_id=CLAIM_ID[sig.name], entity=sig.symbol, metric_or_event=METRIC[sig.name],
        period=sig.as_of.isoformat(), direction=direction, claim_type="factual",
        text=text, evidence_ids=sorted(sig.evidence_ids)[-1:], confidence=sig.confidence,
    )


def build_finding(signals: list) -> Finding:
    wanted = [s for s in signals if s.name in METRIC]
    claims, limits = [], []
    for s in wanted:
        if s.status == "failed" or s.value is None or not s.evidence_ids:
            limits.append("{} unavailable ({})".format(s.name, s.status))
            continue
        if s.status == "degraded" or s.data_status in ("stale", "delayed"):
            limits.append("{} data is {}".format(s.name, s.data_status))
        claims.append(_claim(s))
    if not claims:
        why = "; ".join(limits) or "no price signals provided"
        return Finding(agent="technical", status="failed", error=why, limitations=limits)
    return Finding(agent="technical", status="degraded" if limits else "complete",
                   claims=claims, limitations=limits)


def run(ctx: AgentContext) -> AgentResult:
    kwargs = {"price_dir": ctx.price_dir} if ctx.price_dir else {}
    sigs = [s for s in get_signals(ctx.symbol, ctx.as_of, None, **kwargs) if s.name in METRIC]
    return AgentResult(build_finding(sigs), signals=sigs)

