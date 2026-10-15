# VibeFinance

Multi-agent financial intelligence for retail investors (hackathon PS 04). Three agents (technical, sentiment, filings) run in parallel; a deterministic layer, EFCE, checks every claim's evidence and dates and classifies conflicts before anything is shown. Investor profiles change risk notes and portfolio numbers, never facts or citations.

- Design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Evaluation, disclosed changes and limitations: [docs/RESULTS.md](docs/RESULTS.md)

**Data scope:** filings are TCS only (annual report FY26, Q2 FY27 press release and earnings-call transcript). Prices exist for RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, ITC and SBIN. Investor profiles and holdings are synthetic. Confidence values are heuristic, not calibrated.

## Setup (Windows, PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONUTF8="1"
```

The filings agent uses a local model through [Ollama](https://ollama.com); no API key is needed:

```powershell
ollama pull llama3.2
ollama pull nomic-embed-text
```

`.env.example` shows the optional settings.

## Run

Command-line analysis (as-of date controls which documents and prices are visible):

```powershell
python -m src.run_pipeline TCS --as-of 2026-10-14 --profile conservative
python -m src.run_pipeline TCS --as-of 2026-10-14 --profile aggressive
python -m src.run_pipeline TCS --as-of 2026-10-14 --no-llm      # deterministic filings fallback, no model
```

Stored investor profiles (creates `data\portfolio.db`; do not commit it):

```powershell
python -m src.portfolio.store
```

Dashboard:

```powershell
python -m streamlit run app.py
```

The filings agent takes about 30 to 40 seconds with the model warm and longer on the first call while Ollama loads.

## Test

```powershell
python -m pytest tests -q --ignore=tests\test_infy_pdfs.py --ignore=tests\test_infy_financial_content.py --ignore=tests\test_all_financial_pdfs.py
```

The three ignored files are standalone INFY ingestion scripts that exit at import.

## Evaluation tooling

```powershell
python -m src.eval.harness --scenarios s01,s02 --out data\eval\runs_dev     # omit --scenarios for all 13
python -m src.eval.audit tally                                                # summarise data\eval\audit.csv
python -m src.rag.eval_retrieval --method baseline --split dev                # retrieval recall (also bm25, hybrid, capped, hybrid_capped)
```

- Committed results are in `data\eval\runs_final` and `data\eval\runs_final_s09fix`. Do not point `--out` at them.
- `python -m src.eval.audit make` regenerates `data\eval\audit.csv` and overwrites the filled-in verdicts. Do not run it on the committed sheet.
- Scenarios (`data\eval\scenarios.json`) and the scorer (`src\eval\scoring.py`) were frozen before the baselines ran; see `docs/RESULTS.md` for the three changes made afterwards.

## Layout

| Path | Contents |
|---|---|
| `app.py` | Streamlit dashboard |
| `src/agents` | technical, sentiment and filings agents; orchestrator |
| `src/efce` | evidence and conflict resolution rules |
| `src/rag` | chunking, index, retrieval, retrieval evaluation |
| `src/signals` | price signals |
| `src/portfolio` | stored profiles, portfolio analysis, profile application |
| `src/eval` | scorer, baselines (B0, B1, B2), harness, audit tooling |
| `src/scenarios` | India revenue conflict scenario |
| `src/evidence_view.py`, `src/ui_panels.py` | dashboard helpers |
| `tests` | offline tests |
| `data` | prices, document chunks, evaluation scenarios and results |
