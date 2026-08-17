# MLB-Prediction-model-backtest

An MLB sports-betting prediction model built on **correlation-driven
selective input filtering + Monte Carlo game simulation + strict
walk-forward backtesting**, using two upstream data repos:

| Source | Repo | Provides |
|---|---|---|
| Scoreboard / schedule / results | [buffedlizard55-lab/MLB-PBP](https://github.com/buffedlizard55-lab/MLB-PBP) | Official `statsapi.mlb.com` schedule + full PBP live feeds, SHA-256-verified archive tooling (2014→present) |
| Statcast | [karagemop466-tech/StatcastMLB](https://github.com/karagemop466-tech/StatcastMLB) | Official `baseballsavant.mlb.com` pitch-level Statcast + leaderboards, range validation, integrity ledger, pitch-outcome backtest engine |

**Zero-hallucination rule:** every number in this project traces back to an
official MLB response. Missing data stays missing; nothing is invented,
including odds (the upstream repos contain no historical sportsbook odds —
ROI-vs-market requires an odds feed; breakeven prices are reported instead).

## Architecture

```
upstream repos (vendor/)                 this repo (mlb_predict/)
┌──────────────────────────┐   schedule/results   ┌────────────────────────┐
│ MLB-PBP   (mlb_pbp)      │ ───────────────────▶ │ collect_schedule       │
│  scoreboard + full PBP   │                      │ parse_schedule         │
└──────────────────────────┘                      │ features (leak-free    │
┌──────────────────────────┐   leaderboards/CSV   │   rolling team state)  │
│ StatcastMLB (statcastmlb)│ ───────────────────▶ │ collect_statcast       │
│  Savant + Statcast       │                      └───────────┬────────────┘
└──────────────────────────┘                                  ▼
                              selective filter (|r| ≥ τ, sign-stability,
                              redundancy pruning)  ──▶  simulate (Monte Carlo)
                                                      ──▶  model (logistic blend)
                                                      ──▶  backtest (walk-forward)
```

### The edge, as specified

1. **Correlation** — a wide candidate pool of pre-game features is scored
   against outcomes by point-biserial correlation on strictly-past data.
2. **Selective filtering** — only inputs that pass the correlation floor,
   keep a stable sign across both halves of history, and are not redundant
   with a stronger input survive (`mlb_predict/select.py`). The filter is
   re-run at every walk-forward refit.
3. **Simulation results as inputs** — a Negative-Binomial Monte Carlo game
   simulator (run rates blended from recent offense vs. opponent defense,
   home advantage + dispersion re-estimated from past games only) produces
   `sim_p_home`, expected total, and run-line probabilities, which are
   blended with the filtered features in a calibrated logistic model.

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt pytest
scripts/sync_sources.sh                 # clone the two upstream repos into vendor/

# offline sanity chain (synthetic, clearly labelled — never mixed with real data)
PYTHONPATH=. python -m mlb_predict demo
PYTHONPATH=. python -m pytest tests -q

# real collection on a machine with direct MLB egress (your laptop / CI):
PYTHONPATH=. python -m mlb_predict collect --start 2025-03-17 --end 2025-09-30
PYTHONPATH=. python -m mlb_predict collect-statcast --board sprint_speed --year 2025
PYTHONPATH=. python -m mlb_predict build-dataset
PYTHONPATH=. python -m mlb_predict backtest --season 2025

# full PBP archive via the vendored MLB-PBP package (stdlib-only):
PYTHONPATH=. python -m mlb_predict.pbp_archive range 2025-04-01 2025-04-07

# bulk Statcast harvesting via the vendored StatcastMLB package:
pip install -e vendor/StatcastMLB && statcast --help
```

In sandboxed environments without direct MLB egress, the same official
responses are collected through page captures:

```bash
PYTHONPATH=. python -m mlb_predict ingest-page CAPTURE.txt --window 2025-04-01 2025-04-05
PYTHONPATH=. python -m mlb_predict ingest-scores scores.csv --source-url 'https://statsapi.mlb.com/...'
PYTHONPATH=. python -m mlb_predict build-dataset-from-scores
```

## Betting markets covered

Ground truth comes straight from official results, so the backtester
evaluates: **moneyline** (P home win), **totals** (predicted total vs
actual, O/U hit rate at a fixed reference line), and **run line**
(simulated P(home −1.5 covers)). Metrics: accuracy, Brier, log-loss,
ROC-AUC, plus selective-play subsets (bet only when |p−0.5| ≥ edge) with
average breakeven price.

## Repo map

- `mlb_predict/collect_schedule.py` — official schedule/results collector (resumable, raw-gz cache)
- `mlb_predict/parse_schedule.py` — verified ground-truth parser (rejects inconsistent rows)
- `mlb_predict/collect_statcast.py` — Savant leaderboard collector (URLs per StatcastMLB docs)
- `mlb_predict/pbp_archive.py` — adapter around vendored `mlb_pbp` for full PBP
- `mlb_predict/features.py` — leak-free rolling team-state features (shift-before-roll)
- `mlb_predict/simulate.py` — Monte Carlo game simulation (Gamma-Poisson run model)
- `mlb_predict/select.py` — correlation + sign-stability + redundancy filter
- `mlb_predict/model.py` — logistic blend of sim prior + selected features; totals model
- `mlb_predict/backtest.py` — walk-forward engine, baselines, selective-play report
- `mlb_predict/cli.py` — `collect / ingest-page / ingest-scores / build-dataset / backtest / demo`
- `tests/` — offline unit tests (synthetic fixtures are conspicuously labelled)
- `vendor/` — clones of the two upstream repos (gitignored; `scripts/sync_sources.sh`)
- `data/ref/teams_official.csv` — 30-team id/name map fetched from the official teams endpoint

## Data provenance ledger

- `data/raw/scores_manifest.csv` — every ingested scores batch with retrieval
  timestamp and exact source URL.
- `data/raw/schedule/*.json.gz` — verbatim official schedule responses.
- `data/results/backtest_<season>_report.json` — full walk-forward report
  including the feature selection made at every refit.

## Latest results — 2025 full regular season (walk-forward, no lookahead)

Dataset: **3,692 verified official games** (full 2024 season through Jun 30 +
full 2025 season, all 30 teams), transcribed 1:1 from `statsapi.mlb.com`
schedule responses with a provenance manifest. Backtest evaluates **2,430
out-of-sample predictions** — every 2025 regular-season game.

| Strategy | Accuracy | Brier | Log-loss | AUC |
|---|---|---|---|---|
| **Model (sim + selected features)** | **55.5%** | **0.2461** | 0.7229 | **0.5663** |
| Always-home 50% | 54.3% | 0.2500 | 0.6931 | 0.5000 |
| Simulator alone | 52.4% | 0.2559 | 0.7060 | 0.5513 |
| Season win-pct favorite | 54.7% | 0.2469 | 0.6871 | 0.5557 |

**Selective play (the edge)** — bet only where the model is confident:

| Edge threshold | Accuracy | Games | Coverage | Avg breakeven price |
|---|---|---|---|---|
| ≥ 0.03 | 57.1% | 1,820 | 74.9% | ~1.71 |
| ≥ 0.06 | 59.4% | 1,234 | 50.8% | ~1.64 |
| ≥ 0.10 | **61.4%** | 637 | 26.2% | ~1.57 |

Totals: MAE 3.67 runs, RMSE 5.66, O/U 8.5 hit 50.4% — the totals leg is the
weakest and is the next improvement target.

Adding 2024 training history lifted the selective ≥0.10 subset from
57.8% → **61.4% accuracy**, and the raw model from 54.0% → 55.5% —
confirming the multi-year-history roadmap item. Next accuracy lever:
Statcast pitcher/batter features via the StatcastMLB harvester.

## Roadmap

1. ✅ Dual-repo integration (MLB-PBP scoreboard/PBP + StatcastMLB harvesters)
2. ✅ Leak-free features, Monte Carlo simulator, selective filter, walk-forward backtester
3. ✅ Full 2025 season ingested + backtested (2,430 games)
4. ✅ 2024 first-half ingest → multi-year train history (lifted selective ≥0.10 to 61.4%)
5. ⬜ 2024 second-half ingest (Jul–Sep) → full two-year history
6. ⬜ Statcast pitcher/batter quality features (via `vendor/StatcastMLB`)
7. ⬜ Odds feed adapter → ROI/CLV evaluation instead of breakeven prices
8. ⬜ Full PBP-derived features (run expectancy, bullpen load) via `vendor/MLB-PBP`

## Rights

Independent project, not affiliated with or endorsed by Major League
Baseball. MLB data remains subject to the copyright notice returned by
MLB Advanced Media, L.P. in every API response.
