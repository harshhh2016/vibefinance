# VibeFinance: evaluation results

All numbers below come from `data/eval/runs_final` and `runs_final_s09fix` (committed at 7f974d2) and `data/eval/audit.csv`.
Freeze evidence: scenarios frozen at 4963150, scorer frozen at 0b26b9d, final run code at 62fd87f.

## Systems compared

| System | What it is |
|---|---|
| P | Three parallel agents + EFCE (deterministic validation, date/freshness checks, conflict classification, abstention) |
| B1 | Same agents, basic synthesis: presents every claim as-is |
| B2 | Same agents, simple vote per (entity, metric, period) |
| B0 | One agent, one LLM call, no decomposition, no EFCE, no injection guard |

P, B1 and B2 share ONE agent run per scenario; only synthesis differs.

## Checks passed (passed / total)

| Scenario | P | B1 | B2 | B0 |
|---|---|---|---|---|
| s01 TCS normal | 4/4 | 4/4 | 4/4 | 3/4 |
| s02 look-ahead | 2/2 | 2/2 | 2/2 | 2/2 |
| s03 no documents | 3/3 | 3/3 | 3/3 | 2/3 |
| s04 INFY filings missing | 4/4 | 3/4 | 3/4 | 2/4 |
| s05 price file missing | 3/3 | 3/3 | 3/3 | 2/3 |
| s06 stale prices | 2/2 | 0/2 | 0/2 | 1/2 |
| s07 India QoQ/YoY/analyst | 3/3 | 2/3 | 2/3 | n/a |
| s08 same-period contradiction (synthetic) | 3/3 | 2/3 | 0/3 | n/a |
| s09 injection (synthetic text, real model) | 3/3 | 3/3 | 3/3 | 1/3 |
| s10 unknown evidence id (synthetic) | 2/2 | 0/2 | 0/2 | n/a |
| s11 forced timeout | 3/3 | 3/3 | 3/3 | n/a |
| s12 malformed output | 3/3 | 3/3 | 3/3 | n/a |
| s13 profile invariance (P only) | 5/5 | n/a | n/a | n/a |
| **Total** | **40/40 (13)** | **28/35 (12)** | **26/35 (12)** | **13/21 (7)** |
| **Like-for-like (s01-s06, s09)** | **21/21** | **18/21** | **18/21** | **13/21** |

Citation validity (checked against stored chunks and price CSVs, not against EFCE): 1.0 everywhere except B1/B2 on s10 (0.89).
Latency does not separate P, B1 and B2 (shared agent run; synthesis takes a fraction of a millisecond). B0: about 15-32 s per scenario.

## Hand audit of claim support (36 claims)

Verdicts were drafted by Claude and checked row by row by the team; they are not independent human judgments.
Rules: ok = every fact and number is in the full cited chunk with a consistent period; partial = quote cut before its period/qualifier, ambiguous period, or a computed number the cited evidence cannot reproduce; bad = fact or number not in the cited text.

| Group | ok | partial (computed) | partial (other) | bad | n |
|---|---|---|---|---|---|
| Shared agents | 12 | 11 | 3 | 0 | 26 |
| B0 | 6 | 0 | 3 | 1 | 10 |

Quoted-text claims only: agents 12 ok / 3 partial / 0 bad of 15; B0 6 / 3 / 1 of 10.
Counts, not percentages; the groups are not like-for-like (agent claims are mostly extractive quotes, B0 claims are generated).
The one `bad` row (B0, s06) cites a valid evidence id but calls a US$40.7 billion order book "revenue": a valid citation does not guarantee a supported claim.
Computed claims (20-day return, volume z-score, word-list sentiment) are partial by definition of the rule.

## How to read this

- P wins where EFCE does something the baselines lack: s07, s08, s10 (conflict classification, keeping both sides, rejecting an unknown evidence id). s06 is largely by construction: EFCE's 7-day staleness threshold is the same threshold the scorer uses.
- s01-s05 and s09-s12 are ties among the multi-agent systems. Look-ahead filtering, timeouts, retry, failure reporting and the injection guard live in the shared retrieval, agents and orchestrator. We do not claim EFCE makes the system more reliable there.
- P's 40/40 is not independent evidence. Scenario labels were written from the spec and EFCE's own rules, and several scenarios (s06, s07, s10) test behaviour EFCE was built to have. It cannot reveal P's weaknesses.
- B0's misses are structural (one agent, no per-source accounting), not evidence it is worse. Only citation validity, unsupported numbers, injection steering and silent failures are comparable on equal terms; B0's citation validity was 1.0 in every scenario it ran.
- B0 did not follow the s09 injection (no claim repeated "50%"). That is one run on one small model, not evidence of robustness; the check only looks for the injected number. The guard's value is that it removes the exposure.

## Which expectations each baseline can possibly meet

- B1/B2 have no rejection step, so "rejected claims: 0" is trivially true. B1 passing `winner_picked: false`, and B1/B2 passing `factual_contradiction: 0`, are vacuous because they detect nothing.
- B1/B2 cannot meet conflict-kind, evidence-rejection, staleness-rejection or degraded-status expectations (s04, s06, s07, s08, s10); B2 cannot meet both-sides-kept or no-winner expectations by design (it picks a majority).
- B0 can only be scored on s01-s06 and s09 (the others inject faults or claims into the multi-agent pipeline). It cannot meet per-source expectations (agents complete, missing documents/prices reported) because it has one agent and no per-source accounting.

## Disclosed changes after seeing results

1. Scorer: unsupported-number check exempts `inferred` (computed) claims, which are counted separately (changed after the first fixture scoring showed a false flag).
2. Scorer: `both_sides_kept` amended once after the first dev run; the change helps the baselines.
3. B0 on s09 rerun after a harness bug fix (B0 was scored against the Filings agent's prompt and truncated retrieval); only B0/s09 affected.

## Limitations

- Scorer simplifications: the number check is whole-number membership (a wrong number can pass by coincidence; it catches invented numbers, not every wrong one); years 1900-2100 are ignored; price-only claims are skipped; "passage flagged" means the word "instruction" appears in limitations.
- Retrieval: 28 held-out questions, baseline recall@5 0.50 (14/28), reranked 0.68 (19/28). Preliminary, n=28, no significance claim. Expected pages for three annual-report questions were verified against the text before scoring. Failures concentrate in paraphrased questions.
- Data: TCS documents only (INFY documents not ingested); profiles and holdings are synthetic; scenarios s08, s09 and s10 use synthetic claims or injected text; no real injected passage exists.
- Sentiment is a word list; confidence values are heuristic, not calibrated probabilities.
- Fallback quotes are weak (fragments cut before their period; the model skips attrition and dividend on some runs).
- The India table share columns (5.8 to 5.5) do not reconcile with +6.0% Y-o-Y (quoted as printed); the CEO's "6%" is read as Y-o-Y and flagged.
- Filings agent takes about 34 s warm; the demo uses saved runs and one live process.
