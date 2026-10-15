"""Dashboard panels for the portfolio section and the per-session metrics.

Pure data builders (portfolio_view, metrics_view) plus thin render functions that take
the `st` module as an argument, so everything is testable offline. Reads only the saved
run dict (RunRecord.model_dump); never recomputes signals, findings or EFCE output.
"""
from __future__ import annotations

SYNTHETIC_NOTE = ("Synthetic profile and holdings: demonstration data, not a real investor. "
                  "Profiles change only risk notes and portfolio numbers, never facts or citations.")
HEURISTIC_NOTE = "Confidence values are heuristic scores, not calibrated probabilities."


def portfolio_view(run: dict) -> dict | None:
    """Portfolio section written by apply_profile into config['portfolio'], or None."""
    sec = (run.get("config") or {}).get("portfolio")
    if not sec:
        return None
    weights = sec.get("weights") or {}
    dates = sec.get("price_dates") or {}
    positions = [{"Symbol": s, "Weight %": round(w * 100, 1), "Price date": dates.get(s, "n/a")}
                 for s, w in sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))]
    sectors = [{"Sector": s, "Weight %": round(w * 100, 1)}
               for s, w in sorted((sec.get("sector_weights") or {}).items(), key=lambda kv: (-kv[1], kv[0]))]
    breaches = ["{} {} is {:.1f}%, above the {:.1f}% limit.".format(
        "Position" if b.get("kind") == "position" else "Sector", b.get("name"),
        b.get("weight_pct", 0.0), b.get("limit_pct", 0.0)) for b in sec.get("breaches") or []]
    return {"profile_id": sec.get("profile_id"), "risk_preference": sec.get("risk_preference"),
            "synthetic": bool(sec.get("synthetic_profile", True)),
            "total_value": sec.get("total_value", 0.0), "hhi": sec.get("hhi", 0.0),
            "positions": positions, "sectors": sectors, "breaches": breaches,
            "missing_prices": list(sec.get("missing_prices") or []),
            "risk_note": sec.get("risk_note", "")}


def metrics_view(run: dict) -> list[tuple[str, str]]:
    """(label, formatted value) for the metrics logged in this session. Absent keys show n/a."""
    m = run.get("metrics") or {}
    n_agents = len(run.get("findings") or [])
    rows = []
    e2e = m.get("end_to_end_ms")
    rows.append(("End-to-end time", "n/a" if e2e is None else "{:,.0f} ms".format(e2e)))
    ac = m.get("agents_complete")
    rows.append(("Agents complete", "n/a" if ac is None else "{:.0f} of {}".format(ac, n_agents)))
    cp = m.get("citation_pass_rate")
    rows.append(("Citation pass rate", "n/a" if cp is None else "{:.0%}".format(cp)))
    for k in sorted(k for k in m if k.startswith("latency_ms_")):
        rows.append(("{} latency".format(k[len("latency_ms_"):].title()), "{:,.0f} ms".format(m[k])))
    return rows


def render_metrics(st, run: dict) -> None:
    st.markdown("### Session metrics")
    rows = metrics_view(run)
    cols = st.columns(3)
    for i, (label, value) in enumerate(rows):
        cols[i % 3].metric(label, value)
    st.caption(HEURISTIC_NOTE)


def render_portfolio(st, run: dict) -> None:
    st.markdown("### Portfolio")
    v = portfolio_view(run)
    if v is None:
        st.info("No investor profile was applied to this run.")
        return
    if v["synthetic"]:
        st.warning(SYNTHETIC_NOTE)
    c1, c2, c3 = st.columns(3)
    c1.metric("Profile", "{} ({} risk)".format(v["profile_id"], v["risk_preference"]))
    c2.metric("Portfolio value", "{:,.0f}".format(v["total_value"]))
    c3.metric("Concentration (HHI)", "{:.2f}".format(v["hhi"]))
    if v["risk_note"]:
        st.info(v["risk_note"])
    for b in v["breaches"]:
        st.warning(b)
    if v["positions"]:
        st.markdown("**Positions**")
        st.table(v["positions"])
    if v["sectors"]:
        st.markdown("**Sectors**")
        st.table(v["sectors"])
    if v["missing_prices"]:
        st.caption("No price available for: " + ", ".join(v["missing_prices"]))
