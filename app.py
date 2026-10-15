import json
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.contracts import Profile, RunRecord
from src.signals import get_signals
from src.pipeline import run_analysis
from src.ui_panels import render_metrics, render_portfolio
from src.evidence_view import resolve, resolve_many
from src.rag.generate_answer import generate_answer

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

PRICE_DIR = ROOT / "data" / "prices"
MANIFEST_PATH = PRICE_DIR / "manifest.json"
FIXTURE_DIR = ROOT / "fixtures"

RUN_FIXTURE_PATH = FIXTURE_DIR / "run.json"
PROFILE_FIXTURE_PATH = FIXTURE_DIR / "profile.json"

REQUIRED_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="VibeFinance",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

        :root {
            --vf-bg: #05030A;
            --vf-bg-2: #0B0712;
            --vf-surface: rgba(18, 13, 31, 0.78);
            --vf-surface-solid: #120D1F;
            --vf-border: rgba(196, 181, 253, 0.15);
            --vf-text: #F4F0FF;
            --vf-muted: #9C93B8;
            --vf-purple: #7B5CFF;
            --vf-lavender: #A78BFA;
            --vf-light-purple: #C4B5FD;
            --vf-vein: rgba(233, 227, 255, 0.10);
            --vf-profit: #34D399;
            --vf-loss: #FB7185;
        }

        @keyframes vfAurora {
            0%, 100% { background-position: 0% 50%, 100% 20%, 50% 50%; }
            50% { background-position: 35% 20%, 70% 70%, 100% 50%; }
        }
        @keyframes vfFadeUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes vfPulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(123, 92, 255, 0.08); opacity: .85; }
            50% { box-shadow: 0 0 0 6px rgba(123, 92, 255, 0.07); opacity: 1; }
        }
        @keyframes vfVeinDrift {
            0%, 100% { transform: translate3d(0,0,0) rotate(0deg); }
            50% { transform: translate3d(10px,-7px,0) rotate(1deg); }
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        h1, h2, h3, [data-testid="stMetricValue"] {
            font-family: 'Space Grotesk', 'Inter', sans-serif !important;
        }
        .stApp {
            color: var(--vf-text);
            background:
                radial-gradient(900px 520px at 10% -10%, rgba(123,92,255,.28), transparent 62%),
                radial-gradient(740px 520px at 100% 18%, rgba(167,139,250,.18), transparent 62%),
                radial-gradient(600px 420px at 60% 100%, rgba(59,31,122,.20), transparent 65%),
                linear-gradient(140deg, var(--vf-bg), var(--vf-bg-2) 58%, #08050f);
            background-size: 150% 150%, 150% 150%, 150% 150%, 100% 100%;
            animation: vfAurora 28s ease-in-out infinite;
            background-attachment: fixed;
        }

        /* A subtle violet wash behind the content on every dashboard page. */
        [data-testid="stMain"] {
            background:
                radial-gradient(ellipse at 18% 5%, rgba(123,92,255,.075), transparent 48%),
                radial-gradient(ellipse at 92% 42%, rgba(167,139,250,.055), transparent 45%);
            background-attachment: fixed;
        }

        [data-testid="stMainBlockContainer"] {
            position: relative;
            z-index: 0;
        }

        [data-testid="stMainBlockContainer"]::before {
            content: "";
            position: fixed;
            inset: 0;
            z-index: -1;
            pointer-events: none;
            background:
                radial-gradient(ellipse at 80% 0%, rgba(123,92,255,.045), transparent 38%),
                repeating-linear-gradient(128deg, transparent 0 180px, rgba(233,227,255,.012) 181px, transparent 183px 320px);
            animation: vfVeinDrift 24s ease-in-out infinite;
        }
        [data-testid="stHeader"] { background: rgba(5,3,10,.62); }
        [data-testid="stToolbar"] { right: 1rem; }
        [data-testid="stSidebar"] {
            background:
                radial-gradient(ellipse at 0% 0%, rgba(123,92,255,.13), transparent 52%),
                linear-gradient(180deg, #100b1b, #090610);
            border-right: 1px solid var(--vf-border);
        }
        [data-testid="stSidebar"] > div { background: transparent; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
            letter-spacing: -1.1px;
            font-weight: 700;
        }
        [data-testid="stMainBlockContainer"] > div {
            animation: vfFadeUp 420ms ease-out both;
        }
        [data-testid="stMetric"] {
            background:
                radial-gradient(ellipse at 100% 0%, rgba(123,92,255,.13), transparent 52%),
                linear-gradient(145deg, rgba(24,17,40,.91), rgba(13,9,23,.88));
            border: 1px solid var(--vf-border);
            border-radius: 19px;
            padding: 18px 19px 16px;
            min-height: 112px;
            box-shadow: 0 12px 36px rgba(0,0,0,.18), inset 0 1px 0 rgba(233,227,255,.035);
            backdrop-filter: blur(14px);
            transition: transform .28s ease, border-color .28s ease, box-shadow .28s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-4px);
            border-color: rgba(167,139,250,.42);
            box-shadow: 0 18px 46px rgba(0,0,0,.26), 0 0 34px rgba(123,92,255,.12);
        }
        [data-testid="stMetricLabel"] {
            color: var(--vf-muted);
            font-size: .78rem;
            letter-spacing: .06em;
            text-transform: uppercase;
        }
        [data-testid="stMetricValue"] {
            color: var(--vf-text);
            font-weight: 700;
            letter-spacing: -.045em;
            font-variant-numeric: tabular-nums;
        }
        [data-testid="stMetricDelta"] { font-size: .82rem; }
        .subtitle {
            color: var(--vf-muted);
            font-size: .94rem;
            line-height: 1.7;
            margin-top: -.5rem;
        }
        [data-testid="stMainBlockContainer"] h1 {
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 750;
            letter-spacing: -.045em;
        }
        [data-testid="stMainBlockContainer"] h2,
        [data-testid="stMainBlockContainer"] h3 {
            letter-spacing: -.025em;
        }
        .section-card, .glass {
            position: relative;
            overflow: hidden;
            background:
                radial-gradient(ellipse at 100% 0%, rgba(123,92,255,.10), transparent 48%),
                linear-gradient(145deg, rgba(18,13,31,.80), rgba(10,7,18,.88));
            border: 1px solid var(--vf-border);
            border-radius: 20px;
            padding: 20px;
            margin-bottom: 14px;
            box-shadow: 0 12px 34px rgba(0,0,0,.15), inset 0 1px 0 rgba(233,227,255,.035);
            backdrop-filter: blur(14px);
            transition: transform .3s ease, box-shadow .3s ease, border-color .3s ease;
        }
        .section-card:hover, .glass:hover {
            transform: translateY(-3px);
            border-color: rgba(167,139,250,.34);
            box-shadow: 0 18px 45px rgba(0,0,0,.22), 0 0 36px rgba(123,92,255,.13);
        }
        .hero-card {
            position: relative;
            overflow: hidden;
            background:
                radial-gradient(ellipse at 80% 5%, rgba(167,139,250,.24), transparent 42%),
                radial-gradient(ellipse at 0% 100%, rgba(123,92,255,.20), transparent 48%),
                linear-gradient(120deg, rgba(28,18,50,.96), rgba(12,8,21,.97));
            border: 1px solid rgba(196,181,253,.24);
            border-radius: 24px;
            padding: 28px 30px;
            margin: 8px 0 22px;
            box-shadow: 0 22px 55px rgba(0,0,0,.24), 0 0 42px rgba(123,92,255,.09);
            animation: vfFadeUp 520ms ease-out both;
        }
        .hero-card::before {
            content: "";
            position: absolute;
            inset: -40%;
            background: repeating-linear-gradient(122deg,
                transparent 0 72px,
                rgba(233,227,255,.035) 73px,
                transparent 75px 142px);
            opacity: .5;
            pointer-events: none;
            animation: vfVeinDrift 18s ease-in-out infinite;
        }
        .hero-card::after {
            content: "";
            position: absolute;
            width: 220px; height: 220px;
            right: -95px; top: -135px;
            border-radius: 50%;
            border: 1px solid rgba(233,227,255,.18);
            box-shadow: 0 0 0 20px rgba(167,139,250,.045), 0 0 0 43px rgba(123,92,255,.035), 0 0 0 68px rgba(233,227,255,.018);
            pointer-events: none;
        }
        .eyebrow {
            position: relative;
            color: var(--vf-light-purple);
            text-transform: uppercase;
            letter-spacing: 2px;
            font-size: .70rem;
            font-weight: 700;
        }
        .hero-title {
            position: relative;
            color: var(--vf-text);
            font-family: 'Space Grotesk', sans-serif;
            font-size: clamp(1.55rem, 3vw, 2.15rem);
            font-weight: 700;
            letter-spacing: -.045em;
            margin: 7px 0;
        }
        .hero-copy {
            position: relative;
            max-width: 720px;
            color: var(--vf-muted);
            margin-bottom: 0;
            line-height: 1.7;
        }
        .hero-number {
            background: linear-gradient(100deg, #E9E3FF, #C4B5FD 38%, #7B5CFF 78%, #E9E3FF);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            font-variant-numeric: tabular-nums;
        }
        .status-label { font-size: 13px; color: var(--vf-muted); }
        div[data-testid="stTabs"] button {
            border-radius: 11px 11px 0 0;
            font-weight: 600;
            transition: color .18s ease, background .18s ease;
        }
        [data-testid="stExpander"] {
            border: 1px solid var(--vf-border);
            border-radius: 14px;
            background: rgba(18,13,31,.42);
        }
        [data-testid="stAlert"] { border-radius: 14px; }
        [data-testid="stSidebar"] [role="radiogroup"] label {
            border-radius: 11px;
            padding: 7px 9px;
            transition: background .18s ease, transform .18s ease;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: rgba(123,92,255,.12);
            transform: translateX(2px);
        }
        div[data-testid="stButton"] > button {
            border-radius: 11px;
            font-weight: 650;
            transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
        }
        div[data-testid="stButton"] > button:hover {
            transform: translateY(-2px);
            border-color: rgba(167,139,250,.65);
            box-shadow: 0 7px 22px rgba(123,92,255,.15);
        }
        div[data-testid="stTextArea"] textarea,
        div[data-testid="stTextInput"] input {
            background: rgba(12,8,21,.82);
            border-color: var(--vf-border);
            border-radius: 13px;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--vf-border);
            border-radius: 13px;
            overflow: hidden;
        }
        hr { border-color: rgba(196,181,253,.13); }
        footer { visibility: hidden; }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: .01ms !important;
                animation-iteration-count: 1 !important;
                scroll-behavior: auto !important;
                transition-duration: .01ms !important;
            }
        }
        @media (max-width: 768px) {
            [data-testid="stMetric"] { padding: 13px; min-height: 98px; }
            .hero-card { padding: 21px; }
        }

        /* Reference-inspired landing hero and orbital illustration */
        .landing-wrap {
            min-height: 72vh;
            display: grid;
            grid-template-columns: 1.12fr .88fr;
            align-items: center;
            gap: 2.5rem;
            padding: 2rem 0.4rem 3rem;
        }
        .landing-kicker {
            color: #A78BFA;
            letter-spacing: 2px;
            text-transform: uppercase;
            font-size: .78rem;
            font-weight: 700;
            margin-bottom: 1.6rem;
        }
        .landing-title {
            color: #F4F0FF;
            font-family: 'Space Grotesk', sans-serif;
            font-size: clamp(2.8rem, 5vw, 5.15rem);
            font-weight: 700;
            line-height: 1.02;
            letter-spacing: -.065em;
            margin: 0 0 1.45rem;
        }
        .landing-title span {
            background: linear-gradient(100deg, #E9E3FF, #C4B5FD 38%, #8B5CF6 78%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        .landing-description {
            color: #A69BBF;
            font-size: 1.08rem;
            line-height: 1.8;
            max-width: 620px;
            margin-bottom: 1.7rem;
        }
        .orbital-stage {
            position: relative;
            height: 440px;
            min-width: 280px;
            display: flex;
            align-items: center;
            justify-content: center;
            isolation: isolate;
        }
        .orbital-glow {
            position: absolute;
            width: 190px;
            height: 190px;
            border-radius: 50%;
            background: radial-gradient(circle, #F0E8FF 0%, #A78BFA 18%, #7B5CFF 43%, rgba(59,31,122,.65) 68%, transparent 72%);
            filter: drop-shadow(0 0 36px rgba(123,92,255,.45));
            animation: vfCoreBreathe 4s ease-in-out infinite;
        }
        .orbit {
            position: absolute;
            border: 1px solid rgba(196,181,253,.20);
            border-radius: 50%;
            aspect-ratio: 1 / 1;
        }
        .orbit-one { width: 270px; }
        .orbit-two { width: 390px; }
        .orbit-three { width: 510px; }
        .orbit-dot {
            position: absolute;
            width: 18px; height: 18px;
            border-radius: 50%;
            background: #A78BFA;
            box-shadow: 0 0 22px rgba(167,139,250,.42);
            top: 50%; left: 50%;
            transform-origin: 0 0;
        }
        .dot-a { animation: vfOrbitA 11s linear infinite; }
        .dot-b { width: 22px; height: 22px; background: #C4B5FD; animation: vfOrbitB 15s linear infinite; }
        .dot-c { width: 15px; height: 15px; background: #E9E3FF; animation: vfOrbitC 19s linear infinite; }
        .dot-d { width: 20px; height: 20px; background: #7B5CFF; animation: vfOrbitD 13s linear infinite; }
        @keyframes vfCoreBreathe {
            0%, 100% { transform: scale(.96); filter: drop-shadow(0 0 26px rgba(123,92,255,.35)); }
            50% { transform: scale(1.04); filter: drop-shadow(0 0 48px rgba(123,92,255,.60)); }
        }
        @keyframes vfOrbitA { from { transform: rotate(0deg) translateX(135px) rotate(0deg); } to { transform: rotate(360deg) translateX(135px) rotate(-360deg); } }
        @keyframes vfOrbitB { from { transform: rotate(70deg) translateX(195px) rotate(-70deg); } to { transform: rotate(430deg) translateX(195px) rotate(-430deg); } }
        @keyframes vfOrbitC { from { transform: rotate(190deg) translateX(255px) rotate(-190deg); } to { transform: rotate(550deg) translateX(255px) rotate(-550deg); } }
        @keyframes vfOrbitD { from { transform: rotate(255deg) translateX(195px) rotate(-255deg); } to { transform: rotate(615deg) translateX(195px) rotate(-615deg); } }
        .landing-footnote {
            color: #716786;
            font-size: .78rem;
            margin-top: 1.3rem;
        }
        @media (max-width: 900px) {
            .landing-wrap { grid-template-columns: 1fr; padding-top: 1rem; }
            .orbital-stage { height: 330px; transform: scale(.82); margin-top: -1rem; }
            .orbit-three { width: 410px; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_manifest():
    """Load the price dataset manifest."""

    if not MANIFEST_PATH.exists():
        return {}

    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}


@st.cache_data
def load_prices(symbol):
    """Load and clean a company's historical price CSV."""

    file_path = PRICE_DIR / f"{symbol}.csv"

    if not file_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path)

        missing_columns = [
            column
            for column in REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            return pd.DataFrame()

        df = df[REQUIRED_COLUMNS].copy()

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
        )

        for column in [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        df = df.dropna(subset=REQUIRED_COLUMNS)

        df = df.sort_values("date")

        df = df.drop_duplicates(
            subset=["date"],
            keep="last",
        )

        return df.reset_index(drop=True)

    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


def get_available_companies():
    """Return symbols with valid price data."""

    if not PRICE_DIR.exists():
        return []

    symbols = []

    for file_path in PRICE_DIR.glob("*.csv"):
        if file_path.name.lower() == "manifest.csv":
            continue

        symbol = file_path.stem

        if not load_prices(symbol).empty:
            symbols.append(symbol)

    return sorted(symbols)


@st.cache_data(show_spinner=False)
def get_symbols_with_documents():
    """Symbols that have ingested filings (read from data/documents/chunks.jsonl)."""
    import json
    from pathlib import Path as _Path
    path = _Path(__file__).resolve().parent / "data" / "documents" / "chunks.jsonl"
    found = set()
    if not path.exists():
        return found
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    found.add(str(json.loads(line).get("symbol", "")).upper())
                except ValueError:
                    continue
    except OSError:
        return found
    found.discard("")
    return found


def get_company_metadata(manifest, symbol):
    """Read metadata from supported manifest structures."""

    if isinstance(manifest, dict):

        companies = manifest.get("companies", [])

        if isinstance(companies, list):
            for item in companies:
                if (
                    isinstance(item, dict)
                    and item.get("symbol") == symbol
                ):
                    return item

        if (
            symbol in manifest
            and isinstance(manifest[symbol], dict)
        ):
            return manifest[symbol]

        for value in manifest.values():
            if isinstance(value, list):
                for item in value:
                    if (
                        isinstance(item, dict)
                        and item.get("symbol") == symbol
                    ):
                        return item

    elif isinstance(manifest, list):

        for item in manifest:
            if (
                isinstance(item, dict)
                and item.get("symbol") == symbol
            ):
                return item

    return {}


def format_date(value):
    """Format a date for the UI."""

    if pd.isna(value):
        return "N/A"

    return pd.Timestamp(value).strftime("%d %b %Y")


# ============================================================
# FIXTURE LOADING
# ============================================================

@st.cache_data
def load_demo_run():
    """
    Load and validate the demonstration RunRecord.

    Later, replace the caller with src.pipeline.run_analysis().
    The results-rendering functions below can remain unchanged.
    """

    if not RUN_FIXTURE_PATH.exists():
        return None, (
            f"Run fixture not found: {RUN_FIXTURE_PATH}"
        )

    try:
        raw_json = RUN_FIXTURE_PATH.read_text(
            encoding="utf-8"
        )

        result = RunRecord.model_validate_json(raw_json)

        return result.model_dump(mode="json"), None

    except Exception as exc:
        return None, f"Could not load run fixture: {exc}"


@st.cache_data
def load_demo_profile():
    """Load and validate the demonstration investor profile."""

    if not PROFILE_FIXTURE_PATH.exists():
        return None, (
            f"Profile fixture not found: {PROFILE_FIXTURE_PATH}"
        )

    try:
        raw_json = PROFILE_FIXTURE_PATH.read_text(
            encoding="utf-8"
        )

        profile = Profile.model_validate_json(raw_json)

        return profile.model_dump(mode="json"), None

    except Exception as exc:
        return None, f"Could not load profile fixture: {exc}"


# ============================================================
# REUSABLE UI COMPONENTS
# ============================================================

def page_header(title, description):
    """Display a consistent page header."""

    st.title(title)

    st.markdown(
        f'<p class="subtitle">{description}</p>',
        unsafe_allow_html=True,
    )

    st.divider()


def display_price_chart(data, chart_type="Line"):
    """Display historical stock prices."""

    fig = go.Figure()

    if chart_type == "Candlestick":

        fig.add_trace(
            go.Candlestick(
                x=data["date"],
                open=data["open"],
                high=data["high"],
                low=data["low"],
                close=data["close"],
                name="Price",
            )
        )

    else:

        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["close"],
                mode="lines",
                name="Closing Price",
                line=dict(width=2.8, color="#A78BFA", shape="spline"),
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=430,
        font=dict(family="Inter, ui-sans-serif, system-ui", color="#dce6fa"),
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
    )

    fig.update_xaxes(
        showgrid=False,
        rangeslider_visible=False,
    )

    fig.update_yaxes(
        gridcolor="rgba(196, 181, 253, 0.075)",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def display_evidence_ids(evidence_ids):
    """Display evidence IDs with expandable source details."""

    if not evidence_ids:
        st.caption("No evidence IDs available.")
        return

    resolved = resolve_many(evidence_ids)

    for item in resolved:
        with st.expander(
            f"{'📄' if item['kind'] == 'doc' else '📈' if item['kind'] == 'price' else '⚠️'} "
            f"{item['title']}"
        ):
            st.caption(f"Evidence ID: {item['evidence_id']}")

            if not item["found"]:
                st.warning(
                    "This evidence could not be resolved. "
                    "The source may be missing or its ID may be invalid."
                )
                continue

            if item["date"]:
                st.caption(f"Evidence date: {item['date']}")

            if item["kind"] == "doc":
                st.markdown("**Supporting passage**")
                st.write(item["text"])

                if item["source_url"]:
                    st.markdown(
                        f"[Open source document]({item['source_url']})"
                    )

            elif item["kind"] == "price":
                st.markdown("**Historical price data**")
                st.write(item["fields"])

def display_claim(claim):
    """Display a supported claim."""

    st.markdown(f"**{claim['text']}**")

    col1, col2, col3 = st.columns(3)

    col1.caption(
        f"Direction: {claim['direction'].title()}"
    )

    col2.caption(
        f"Type: {claim['claim_type'].title()}"
    )

    col3.caption(
        f"Confidence: {claim['confidence']:.0%}"
    )

    st.caption(
        f"Entity: {claim['entity']} · "
        f"Metric/event: {claim['metric_or_event']} · "
        f"Period: {claim['period']}"
    )

    display_evidence_ids(claim.get("evidence_ids", []))


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("# <span style=\"color:#A78BFA\">◈</span> <span style=\"font-weight:900;letter-spacing:-1.1px\">VibeFinance</span>", unsafe_allow_html=True)
st.sidebar.markdown(
    """<div style="display:flex;align-items:center;gap:8px;margin-top:-8px;margin-bottom:10px;">
        <span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#A78BFA;
        animation:vfPulse 2.2s ease-in-out infinite;"></span>
        <span style="color:#9C93B8;font-size:11px;letter-spacing:1.2px;">FINANCIAL INTELLIGENCE WORKSPACE</span>
    </div>""",
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <p class="subtitle">
        <strong>AI-powered financial intelligence</strong><br>        <span style="font-size:12px;">RESEARCH WORKSPACE · DEMO</span>
    </p>
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()

navigation = st.sidebar.radio(
    "NAVIGATION",
    [
        "Welcome",
        "Overview",
        "AI Assistant",
        "Market Intelligence",
        "Financial Research",
        "Analysis Results",
    ],
)

st.sidebar.divider()

st.sidebar.markdown("### Company Selection")

available_companies = get_available_companies()

if not available_companies:
    st.error(
        "No valid price datasets were found in data/prices."
    )
    st.stop()

default_index = (
    available_companies.index("TCS")
    if "TCS" in available_companies
    else 0
)

DOC_SYMBOLS = get_symbols_with_documents()


def _company_label(sym):
    if sym.upper() in DOC_SYMBOLS:
        return "{} (full analysis)".format(sym)
    return "{} (prices only)".format(sym)


selected_company = st.sidebar.selectbox(
    "Select a company",
    available_companies,
    index=default_index,
    format_func=_company_label,
)
if selected_company.upper() not in DOC_SYMBOLS:
    st.sidebar.info(
        "Filings are not ingested for this company. Only the price-based "
        "signals run; sentiment and filings are reported as missing by design."
    )

selected_prices_for_date = load_prices(selected_company)
if not selected_prices_for_date.empty:
    latest_available_date = selected_prices_for_date["date"].max().date()
    earliest_available_date = selected_prices_for_date["date"].min().date()
    default_analysis_date = min(date.today(), latest_available_date)
    if default_analysis_date < earliest_available_date:
        default_analysis_date = earliest_available_date
else:
    latest_available_date = date.today()
    earliest_available_date = date(2000, 1, 1)
    default_analysis_date = date.today()

analysis_date = st.sidebar.date_input(
    "Analysis as of",
    value=default_analysis_date,
    min_value=earliest_available_date,
    max_value=max(date.today(), latest_available_date),
    help="Choose the cutoff date for historical analysis and document retrieval.",
)

st.sidebar.markdown(
    f"""
    <div class="section-card" style="padding:14px; margin-top:12px;">
        <div class="eyebrow">Selected dataset</div>
        <div style="font-size:1.2rem;font-weight:800;margin:4px 0;">{selected_company}</div>
        <div class="status-label">Latest available session</div>
    </div>
    """,
    unsafe_allow_html=True,
)

manifest = load_manifest()

selected_metadata = get_company_metadata(
    manifest,
    selected_company,
)

df = load_prices(selected_company)

if df.empty:
    st.error(
        f"No valid price data is available for {selected_company}."
    )
    st.stop()

latest_row = df.iloc[-1]

latest_close = float(latest_row["close"])

previous_close = (
    float(df.iloc[-2]["close"])
    if len(df) > 1
    else latest_close
)

daily_change = latest_close - previous_close

daily_change_pct = (
    daily_change / previous_close * 100
    if previous_close
    else 0.0
)

first_close = float(df.iloc[0]["close"])

period_return = (
    (latest_close / first_close - 1) * 100
    if first_close
    else 0.0
)

latest_date = latest_row["date"]


# ============================================================
# WELCOME / LANDING PAGE
# ============================================================

if navigation == "Welcome":

    st.markdown(
        """
        <div class="landing-wrap">
            <div>
                <div class="landing-kicker">VibeFinance</div>
                <h1 class="landing-title">Your money,<br><span>understood by</span><br>intelligent agents.</h1>
                <p class="landing-description">
                    Explore historical market data, computed signals, financial research,
                    and structured analysis in one workspace.
                </p>
                <div class="landing-footnote">
                    Multi-agent financial intelligence · Historical data workspace
                </div>
            </div>
            <div class="orbital-stage" aria-label="Animated purple orbital illustration">
                <div class="orbit orbit-three"></div>
                <div class="orbit orbit-two"></div>
                <div class="orbit orbit-one"></div>
                <div class="orbital-glow"></div>
                <span class="orbit-dot dot-a"></span>
                <span class="orbit-dot dot-b"></span>
                <span class="orbit-dot dot-c"></span>
                <span class="orbit-dot dot-d"></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Your financial workspace")
    welcome_cols = st.columns(3)
    with welcome_cols[0]:
        st.markdown(
            '<div class="section-card"><div class="eyebrow">01 · Markets</div>'
            '<h3>Market overview</h3><p class="subtitle">Explore available historical prices and performance.</p></div>',
            unsafe_allow_html=True,
        )
    with welcome_cols[1]:
        st.markdown(
            '<div class="section-card"><div class="eyebrow">02 · Signals</div>'
            '<h3>Market intelligence</h3><p class="subtitle">Review computed momentum, volume, and sentiment signals.</p></div>',
            unsafe_allow_html=True,
        )
    with welcome_cols[2]:
        st.markdown(
            '<div class="section-card"><div class="eyebrow">03 · Evidence</div>'
            '<h3>Analysis results</h3><p class="subtitle">Inspect findings, claims, evidence, and conflicts.</p></div>',
            unsafe_allow_html=True,
        )

    st.info("Choose a page from the sidebar to explore the existing workspace. The landing page is a visual entry screen; it does not add or alter financial analysis.")


# ============================================================
# OVERVIEW
# ============================================================

elif navigation == "Overview":

    page_header(
        "📊 Market Overview",
        "Monitor historical stock prices and performance.",
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Latest Close",
        f"₹{latest_close:,.2f}",
        f"{daily_change_pct:+.2f}%",
    )

    col2.metric(
        "Daily Change",
        f"₹{daily_change:+,.2f}",
        f"{daily_change_pct:+.2f}%",
    )

    col3.metric(
        "Dataset Return",
        f"{period_return:+.2f}%",
    )

    col4.metric(
        "Trading Sessions",
        f"{len(df):,}",
    )

    chart_col1, chart_col2 = st.columns([2, 1])

    with chart_col1:
        period = st.radio(
            "Historical Period",
            ["1M", "3M", "6M", "1Y", "All"],
            horizontal=True,
            index=3,
        )

    with chart_col2:
        chart_type = st.selectbox(
            "Chart Type",
            ["Line", "Candlestick"],
        )

    period_days = {
        "1M": 30,
        "3M": 90,
        "6M": 180,
        "1Y": 365,
    }

    if period == "All":

        chart_data = df.copy()

    else:

        cutoff = (
            df["date"].max()
            - pd.Timedelta(days=period_days[period])
        )

        chart_data = df[df["date"] >= cutoff].copy()

    st.subheader(f"{selected_company} Price History")

    display_price_chart(
        chart_data,
        chart_type,
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Period High",
        f"₹{chart_data['high'].max():,.2f}",
    )

    col2.metric(
        "Period Low",
        f"₹{chart_data['low'].min():,.2f}",
    )

    col3.metric(
        "Latest Volume",
        f"{int(latest_row['volume']):,}",
    )

    with st.expander("View Historical Price Data"):

        display_df = chart_data.sort_values(
            "date",
            ascending=False,
        ).copy()

        display_df["date"] = (
            display_df["date"].dt.strftime("%Y-%m-%d")
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Historical dataset only. Prices are not a live market feed."
    )


# ============================================================
# AI ASSISTANT
# ============================================================

elif navigation == "AI Assistant":

    page_header(
        "🤖 AI Financial Assistant",
        "Run the technical, sentiment, and filings agents for the selected company.",
    )

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="eyebrow">VibeFinance Intelligence</div>
            <div class="hero-title">Let's understand {selected_company}.</div>
            <p class="hero-copy">
                Run a coordinated analysis using historical price signals, retrieved financial
                documents, and evidence/conflict validation. The selected cutoff date is
                {analysis_date.strftime("%d %b %Y")}.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Run Analysis invokes the actual agents and EFCE pipeline. The optional question is "
        "saved as context for this session; the current pipeline does not yet pass it into "
        "retrieval or generate a free-form answer."
    )

    question = st.text_area(
        "What would you like to investigate? (optional)",
        placeholder=f"e.g. What do recent filings say about {selected_company}'s operating margin?",
        height=90,
        key="financial_question",
    )

    run_col, reset_col = st.columns([1, 5])
    with run_col:
        run_clicked = st.button(
            "▶ Run Analysis",
            type="primary",
            use_container_width=True,
            key="run_pipeline_from_assistant",
        )
    with reset_col:
        if st.button("Clear chat history", key="clear_vf_chat"):
            st.session_state.vf_chat_history = []
            st.rerun()

    if run_clicked:
        try:
            profile_data, profile_error = load_demo_profile()
            profile = Profile.model_validate(profile_data) if profile_data else None

            with st.spinner(
                f"Running technical, sentiment, and filings agents for {selected_company}… "
                "Retrieval and local model startup may take a few minutes."
            ):
                result = run_analysis(
                    symbol=selected_company,
                    as_of=analysis_date,
                    profile=profile,
                    question=question.strip() or None,
                    price_dir=str(PRICE_DIR),
                    use_llm=True,
                    timeout_s=240.0,
                )

            st.session_state["vf_latest_run"] = result.model_dump(mode="json")
            st.session_state["vf_latest_run_source"] = "live"
            st.session_state["vf_latest_run_question"] = question.strip()
            st.success(
                f"Analysis completed. EFCE status: "
                f"{result.efce.status if result.efce else 'unavailable'}."
            )
            st.rerun()

        except Exception as exc:
            st.error(
                "The analysis could not be completed. Check that retrieval dependencies "
                "and the local model are available, then inspect the terminal traceback."
            )
            st.exception(exc)

    if "vf_chat_history" not in st.session_state:
        st.session_state.vf_chat_history = []

    latest_run_data = st.session_state.get("vf_latest_run")
    if latest_run_data:
        latest_run = RunRecord.model_validate(latest_run_data)
        st.markdown("### Latest agent run")
        status = latest_run.efce.status if latest_run.efce else "unknown"
        status_col, signal_col, finding_col, elapsed_col = st.columns(4)
        status_col.metric("EFCE status", status.replace("_", " ").title())
        signal_col.metric("Signals", len(latest_run.signals))
        finding_col.metric("Agent findings", len(latest_run.findings))
        elapsed_col.metric(
            "End-to-end time",
            f"{latest_run.metrics.get('end_to_end_ms', 0):,.0f} ms",
        )

        for finding in latest_run.findings:
            with st.expander(
                f"{finding.agent.title()} · {finding.status.title()}",
                expanded=finding.status != "complete",
            ):
                if finding.error:
                    st.error(finding.error)
                if finding.claims:
                    for claim in finding.claims:
                        st.markdown(f"**{claim.text}**")
                        st.caption(
                            f"Direction: {claim.direction.title()} · "
                            f"Type: {claim.claim_type.title()} · "
                            f"Confidence: {claim.confidence:.0%}"
                        )
                        display_evidence_ids(claim.evidence_ids)
                else:
                    st.info("This agent did not produce claims for this run.")
                for limitation in finding.limitations:
                    st.caption(f"Limitation: {limitation}")

        if latest_run.efce:
            st.markdown("### Evidence resolution")
            efce = latest_run.efce
            if efce.status == "complete":
                st.success("EFCE completed.")
            elif efce.status == "conflict_unresolved":
                st.warning("EFCE found unresolved conflicts.")
            elif efce.status == "failed":
                st.error("EFCE failed.")
            else:
                st.warning(f"EFCE status: {efce.status.replace('_', ' ')}.")

            e1, e2, e3 = st.columns(3)
            e1.metric("Supported claims", len(efce.supported))
            e2.metric("Conflicts", len(efce.conflicts))
            e3.metric("Rejected claims", len(efce.rejected))

            if efce.conflicts:
                for conflict in efce.conflicts:
                    st.warning(
                        f"{conflict.kind.replace('_', ' ').title()}: "
                        f"{conflict.explanation}"
                    )
            if efce.abstained:
                st.markdown("**Abstentions**")
                for item in efce.abstained:
                    st.write(f"- {item}")

        st.caption(
            f"Run ID: {latest_run.run_id} · Company: {latest_run.symbol} · "
            f"As of: {latest_run.as_of} · Source: live pipeline"
        )
    else:
        st.caption("No live run yet. Choose a company and date, then select Run Analysis.")


# MARKET INTELLIGENCE
# ============================================================

elif navigation == "Market Intelligence":

    page_header(
        "📈 Market Intelligence",
        "Explore historical price behavior and computed signals.",
    )

    st.subheader("Historical Market Statistics")

    returns_df = df.copy()

    returns_df["daily_return"] = (
        returns_df["close"].pct_change() * 100
    )

    valid_returns = returns_df["daily_return"].dropna()

    mean_return = (
        float(valid_returns.mean())
        if not valid_returns.empty
        else 0.0
    )

    volatility = (
        float(valid_returns.std())
        if len(valid_returns) > 1
        else 0.0
    )

    positive_percentage = (
        float((valid_returns > 0).mean() * 100)
        if not valid_returns.empty
        else 0.0
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Daily Return",
        f"{mean_return:+.3f}%",
    )

    col2.metric(
        "Daily Volatility",
        f"{volatility:.3f}%",
    )

    col3.metric(
        "Positive Sessions",
        f"{positive_percentage:.1f}%",
    )

    st.subheader("Daily Returns")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=returns_df["date"],
            y=returns_df["daily_return"],
            mode="lines",
            name="Daily Return",
            line=dict(width=1.8, color="#7B5CFF"),
        )
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350,
        font=dict(family="Inter, ui-sans-serif, system-ui", color="#dce6fa"),
        xaxis_title="Date",
        yaxis_title="Daily Return (%)",
        hovermode="x unified",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.divider()

    st.subheader("Technical & Sentiment Signals")

    st.caption(
        "Computed using the existing src/signals modules."
    )

    try:

        signals = get_signals(
            symbol=selected_company,
            as_of=date.today(),
            price_dir=PRICE_DIR,
        )

        signal_info = {
            "momentum": {
                "title": "📈 Momentum",
                "description": (
                    "Return over the previous 20 trading days."
                ),
            },
            "volume_anomaly": {
                "title": "📊 Volume Anomaly",
                "description": (
                    "Volume compared with the previous 20 sessions."
                ),
            },
            "sentiment": {
                "title": "📰 Sentiment",
                "description": (
                    "Word-list sentiment from available financial text."
                ),
            },
        }

        columns = st.columns(3)

        for column, signal in zip(columns, signals):

            with column:

                info = signal_info[signal.name]

                st.markdown(f"### {info['title']}")

                if signal.value is None:

                    st.metric(
                        "Result",
                        "Unavailable",
                    )

                elif signal.name == "momentum":

                    st.metric(
                        "20-Day Return",
                        f"{signal.value:+.2%}",
                    )

                    st.caption(
                        f"Direction: {signal.label.title()}"
                    )

                elif signal.name == "volume_anomaly":

                    st.metric(
                        "Volume Z-Score",
                        f"{signal.value:+.2f}",
                    )

                    st.caption(
                        f"Volume: {signal.label.title()}"
                    )

                else:

                    st.metric(
                        "Sentiment Score",
                        f"{signal.value:+.2f}",
                    )

                    st.caption(
                        f"Sentiment: {signal.label.title()}"
                    )

                st.caption(info["description"])

                st.progress(
                    max(
                        0.0,
                        min(1.0, float(signal.confidence)),
                    ),
                    text=(
                        f"Heuristic confidence: "
                        f"{signal.confidence:.0%}"
                    ),
                )

                st.caption(
                    f"Data status: {signal.data_status.title()}"
                )

                if signal.status == "degraded":

                    st.warning("Signal is degraded.")

                elif signal.status == "failed":

                    st.error("Signal calculation failed.")

                with st.expander("Signal Details"):

                    st.write(signal.method)

                    if signal.window_start and signal.window_end:

                        st.write(
                            f"Period: {signal.window_start} "
                            f"to {signal.window_end}"
                        )

                    display_evidence_ids(
                        signal.evidence_ids
                    )

    except Exception as exc:

        st.error(f"Could not calculate signals: {exc}")

    st.caption(
        "Confidence values are heuristic scores, not calibrated "
        "probabilities or investment recommendations."
    )


# ============================================================
# FINANCIAL RESEARCH
# ============================================================

elif navigation == "Financial Research":

    page_header(
        "📚 Financial Research",
        "Explore available company datasets and prepare for "
        "evidence-based document research.",
    )

    st.subheader("Available Company Datasets")

    dataset_rows = []

    for symbol in available_companies:

        company_df = load_prices(symbol)

        if company_df.empty:
            continue

        metadata = get_company_metadata(
            manifest,
            symbol,
        )

        dataset_rows.append(
            {
                "Company": symbol,
                "Records": len(company_df),
                "First Date": format_date(
                    company_df["date"].min()
                ),
                "Last Date": format_date(
                    company_df["date"].max()
                ),
                "Latest Close": round(
                    float(company_df.iloc[-1]["close"]),
                    2,
                ),
                "Data Status": metadata.get(
                    "data_status",
                    "historical",
                ),
            }
        )

    if dataset_rows:

        st.dataframe(
            pd.DataFrame(dataset_rows),
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info("No company datasets are available.")

    st.divider()

    st.subheader("Financial Document Research")

    st.markdown(
        f"""
        <div class="section-card">
            <h3>📄 Research {selected_company}</h3>
            <p class="subtitle">
                Ask questions about earnings releases, annual reports,
                and earnings-call transcripts using retrieved financial evidence.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if selected_company.upper() not in DOC_SYMBOLS:
        st.warning(
            f"No ingested financial documents were detected for {selected_company}. "
            "Select a company with available documents."
        )
    else:
        research_question = st.text_area(
            "Your research question",
            placeholder=(
                f"What were {selected_company}'s key financial results "
                "and business risks in its latest quarterly report?"
            ),
            height=110,
            key="financial_research_question",
        )

        if st.button(
            "🔍 Research Documents",
            type="primary",
            use_container_width=True,
            key="run_financial_research",
        ):
            if not research_question.strip():
                st.warning("Please enter a research question.")
            else:
                try:
                    with st.spinner(
                        "Searching financial documents and generating an answer..."
                    ):
                        result = generate_answer(
                            query=research_question.strip(),
                            company=selected_company,
                            as_of=analysis_date.isoformat(),
                            k=5,
                        )

                    st.markdown("### Research Answer")
                    st.markdown(result.get("answer", "No answer was returned."))

                    citations = result.get("citations", [])
                    st.markdown("### Supporting Evidence")

                    if citations:
                        for index, citation in enumerate(citations, start=1):
                            doc_id = citation.get("doc_id", "Financial document")
                            with st.expander(f"Source {index}: {doc_id}"):
                                st.write(
                                    citation.get("text", "No passage text available.")
                                )
                    else:
                        st.info(
                            "No supporting passages were retrieved. Try a more specific "
                            "question or check that the document index has been built."
                        )

                except Exception as exc:
                    st.error(
                        "Financial research failed. Check that the document index, "
                        "retrieval dependencies, and Ollama are available."
                    )
                    st.exception(exc)


# ============================================================
# ANALYSIS RESULTS
# ============================================================

elif navigation == "Analysis Results":

    page_header(
        "🧠 Analysis Results",
        "Explore structured findings, evidence, conflicts, and "
        "investor-profile information.",
    )

    # Prefer a real run created from the AI Assistant. Fixtures remain a fallback
    # so the page still works before the first live analysis.
    live_run_data = st.session_state.get("vf_latest_run")

    if live_run_data:
        st.success("LIVE PIPELINE RESULT — generated by the technical, sentiment, and filings agents.")
        run = RunRecord.model_validate(live_run_data)
    else:
        st.warning(
            "DEMO DATA — NOT A REAL INVESTMENT ANALYSIS. "
            "No live pipeline run exists yet, so this page is showing fixtures/run.json."
        )
        run_data, run_error = load_demo_run()
        if run_error:
            st.error(run_error)
            st.stop()
        run = RunRecord.model_validate(run_data)

    # --------------------------------------------------------
    # OVERALL STATUS
    # --------------------------------------------------------

    st.subheader("Run Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Company",
        run.symbol,
    )

    col2.metric(
        "Overall Status",
        run.status if hasattr(run, "status") else (
            run.efce.status if run.efce else "unknown"
        ),
    )

    col3.metric(
        "Signals",
        len(run.signals),
    )

    col4.metric(
        "Agent Findings",
        len(run.findings),
    )

    st.caption(
        f"Run ID: {run.run_id} · "
        f"Analysis date: {run.as_of} · "
        f"Created: {run.created_at}"
    )

    st.caption(
        f"Data status: {run.data_status.title()}"
    )

    # --------------------------------------------------------
    # SIGNALS
    # --------------------------------------------------------

    st.divider()

    st.subheader("Signals")

    if not run.signals:

        st.info("This run contains no signals.")

    else:

        signal_columns = st.columns(
            min(3, len(run.signals))
        )

        for index, signal in enumerate(run.signals):

            with signal_columns[index % len(signal_columns)]:

                st.markdown(
                    f"### {signal.name.replace('_', ' ').title()}"
                )

                if signal.value is None:

                    st.metric(
                        "Value",
                        "Unavailable",
                    )

                elif signal.unit == "pct_return":

                    st.metric(
                        "Value",
                        f"{signal.value:+.2%}",
                    )

                else:

                    st.metric(
                        "Value",
                        f"{signal.value:g}",
                    )

                st.caption(
                    f"Label: {signal.label.title()}"
                )

                st.caption(
                    f"Confidence: {signal.confidence:.0%}"
                )

                st.caption(
                    f"Module status: {signal.status.title()}"
                )

                st.caption(
                    f"Data status: {signal.data_status.title()}"
                )

                with st.expander("Signal methodology"):

                    st.write(signal.method)

                    if signal.window_start and signal.window_end:

                        st.write(
                            f"Window: {signal.window_start} "
                            f"to {signal.window_end}"
                        )

                    display_evidence_ids(
                        signal.evidence_ids
                    )

    # --------------------------------------------------------
    # AGENT FINDINGS
    # --------------------------------------------------------

    st.divider()

    st.subheader("Agent Findings")

    if not run.findings:

        st.info("No agent findings are available.")

    else:

        for finding in run.findings:

            with st.expander(
                f"{finding.agent.title()} Agent — "
                f"{finding.status.title()}",
                expanded=True,
            ):

                if finding.error:

                    st.error(finding.error)

                st.caption(
                    f"Latency: {finding.latency_ms:.1f} ms"
                )

                if finding.limitations:

                    st.markdown("**Limitations**")

                    for limitation in finding.limitations:
                        st.markdown(f"- {limitation}")

                if finding.claims:

                    st.markdown("**Claims**")

                    for claim in finding.claims:

                        st.markdown("---")

                        display_claim(
                            claim.model_dump(mode="json")
                        )

                else:

                    st.info(
                        "This agent did not produce any claims."
                    )

    # --------------------------------------------------------
    # EFCE RESULT
    # --------------------------------------------------------

    st.divider()

    st.subheader("Evidence & Conflict Resolution")

    if run.efce is None:

        st.info("No EFCE result is available for this run.")

    else:

        efce = run.efce

        if efce.status == "complete":

            st.success(
                "Conflict-resolution status: Complete"
            )

        elif efce.status == "conflict_unresolved":

            st.warning(
                "Unresolved conflict detected. "
                "Do not treat the findings as a single agreed conclusion."
            )

        elif efce.status == "abstained_or_limited":

            st.warning(
                "The analysis abstained or is limited."
            )

        elif efce.status == "degraded":

            st.warning(
                "The evidence-resolution result is degraded."
            )

        elif efce.status == "failed":

            st.error(
                "Evidence and conflict resolution failed."
            )

        st.markdown("### Supported Claims")

        if efce.supported:

            for claim in efce.supported:

                with st.container(border=True):

                    display_claim(
                        claim.model_dump(mode="json")
                    )

        else:

            st.info(
                "No supported claims were returned."
            )

        # ----------------------------------------------------
        # CONFLICTS
        # ----------------------------------------------------

        st.markdown("### Conflicts")

        if efce.conflicts:

            for index, conflict in enumerate(
                efce.conflicts,
                start=1,
            ):

                st.warning(
                    f"Conflict {index}: "
                    f"{conflict.kind.replace('_', ' ').title()}"
                )

                st.write(conflict.explanation)

                st.caption(
                    "Involved claim IDs: "
                    + ", ".join(conflict.claim_ids)
                )

        else:

            st.success("No conflicts were reported.")

        # ----------------------------------------------------
        # ABSTENTIONS
        # ----------------------------------------------------

        st.markdown("### Abstentions")

        if efce.abstained:

            for abstention in efce.abstained:

                st.warning(abstention)

        else:

            st.caption(
                "No abstentions were recorded in this run."
            )

        # ----------------------------------------------------
        # REJECTED CLAIMS
        # ----------------------------------------------------

        st.markdown("### Rejected Claims")

        if efce.rejected:

            rejected_rows = [
                {
                    "Claim ID": item.claim_id,
                    "Reason": item.reason.replace("_", " "),
                }
                for item in efce.rejected
            ]

            st.dataframe(
                pd.DataFrame(rejected_rows),
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.caption(
                "No rejected claims were reported."
            )

    # --------------------------------------------------------
    # INVESTOR PROFILE
    # --------------------------------------------------------

    st.divider()

    st.subheader("Investor Profile")

    st.caption("The profile below is the current demo profile; it is not yet used for personalized recommendations.")
    profile_data, profile_error = load_demo_profile()

    if profile_error:

        st.warning(profile_error)

    else:

        profile = Profile.model_validate(profile_data)

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Risk Preference",
            profile.risk_preference.title(),
        )

        col2.metric(
            "Investment Horizon",
            f"{profile.horizon_months} months",
        )

        col3.metric(
            "Maximum Single Position",
            f"{profile.max_single_position_pct:.1f}%",
        )

        st.caption(
            f"Profile ID: {profile.profile_id}"
        )

        st.metric(
            "Maximum Sector Allocation",
            f"{profile.max_sector_pct:.1f}%",
        )

        st.markdown("### Holdings")

        if profile.holdings:

            holdings_rows = [
                {
                    "Symbol": holding.symbol,
                    "Quantity": holding.quantity,
                    "Average Cost": holding.avg_cost,
                    "Sector": holding.sector,
                }
                for holding in profile.holdings
            ]

            st.dataframe(
                pd.DataFrame(holdings_rows),
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.info(
                "No holdings were provided in this profile."
            )
        # --------------------------------------------------------
    # PORTFOLIO
    # --------------------------------------------------------

    st.divider()
    render_portfolio(st, run.model_dump(mode="json"))

    # --------------------------------------------------------
    # SESSION METRICS
    # --------------------------------------------------------

    st.divider()
    render_metrics(st, run.model_dump(mode="json"))

    # --------------------------------------------------------
    # RUN CONFIGURATION
    # --------------------------------------------------------

    with st.expander("Run Configuration"):

        st.json(run.config)

        st.markdown("**Run Metrics**")

        st.json(run.metrics)

    st.caption(
        "These fixture-based results demonstrate the planned UI. "
        "They are not recommendations and do not describe the "
        "currently selected company."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <p style="text-align:center; color:#6b7280; font-size:12px;">
        VibeFinance · Multi-Agent Financial Intelligence
        <br>
        Historical data and analytical signals are for research
        and educational purposes only. Not investment advice.
    </p>
    """,
    unsafe_allow_html=True,
)