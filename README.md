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
PYTHONPATH=. python -m mlb_predict collect-statcast --board pitching_quality --year 2025 --min-pa 100
PYTHONPATH=. python -m mlb_predict collect-probables --start 2025-03-17 --end 2025-09-30
PYTHONPATH=. python -m mlb_predict build-dataset
PYTHONPATH=. python -m mlb_predict build-pitcher-features
PYTHONPATH=. python -m mlb_predict backtest --season 2025 --with-pitcher-features

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
- `mlb_predict/collect_probables.py` — official schedule + `hydrate=probablePitcher`
  captures; verified parser cross-checked against games.csv
- `mlb_predict/pitcher_features.py` — leak-free prior-season starter quality
  (xwOBA/xERA-against, K%, BB%, IP) joined on official probables; rookies stay
  missing with `sp_known` flags
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
- `data/raw/probables_manifest.csv` / `data/raw/statcast_manifest.csv` — same
  ledger for probable-pitcher schedule captures and Savant board captures.
- `data/ref/statcast/pitching_quality_<year>_min100.csv` — parsed copies of the
  official Savant custom pitching boards (2023–2025, min 100 PA against) used
  for starter-quality features.
- `data/raw/schedule/*.json.gz` — verbatim official schedule responses.
- `data/results/backtest_<season>_report.json` — full walk-forward report
  including the feature selection made at every refit.
- `data/raw/page_capture/scores_2023/` — verbatim 2023 schedule window
  captures (10-day windows, slim official projection) behind the transcript.

## Latest results — 2025 full regular season (walk-forward, no lookahead)

Dataset: **7,289 verified official games — three complete seasons**
(2023: exactly 2,430 games / 162 per team; 2024: 2,429 games, HOU and CLE
at 161 because the Sep 29 game was rained out and never made up; 2025:
2,430 games),
transcribed 1:1 from `statsapi.mlb.com` schedule responses with a
provenance manifest. Every team's game total was audited against the
official per-team schedule endpoints. Backtest evaluates **2,430
out-of-sample predictions** — every 2025 regular-season game — with a full
two-year rolling training history at every refit.

| Strategy | Accuracy | Brier | Log-loss | AUC |
|---|---|---|---|---|
| **Model (sim + selected features)** | **56.1%** | **0.2437** | **0.6801** | **0.5699** |
| Always-home 50% | 54.3% | 0.2500 | 0.6931 | 0.5000 |
| Simulator alone | 52.0% | 0.2576 | 0.7097 | 0.5525 |
| Season win-pct favorite | 55.1% | 0.2463 | 0.6859 | 0.5648 |

**Selective play (the edge)** — bet only where the model is confident:

| Edge threshold | Accuracy | Games | Coverage | Avg breakeven price |
|---|---|---|---|---|
| ≥ 0.03 | 57.3% | 1,655 | 68.1% | ~1.73 |
| ≥ 0.06 | 60.3% | 1,005 | 41.4% | ~1.67 |
| ≥ 0.10 | **66.4%** | 455 | 18.7% | ~1.59 |

Totals: MAE 3.609 runs (naive recent-runs baseline 3.617), RMSE 4.583,
O/U 8.5 hit 51.0% — the totals model now edges out the naive baseline.

Training-history ablation, all on the same 2,430 out-of-sample games:

| Training history | Model acc | Selective ≥0.10 acc |
|---|---|---|
| 2025 season only (rolling) | 54.0% | 57.8% |
| + 2024 first half | 55.5% | 61.4% |
| + full 2024 season | 56.1% | 66.4% (n=455) |
| + 2023 Apr–May (800 games) | 55.9% | 67.4% (n=347) |
| + 2023 partial (1,328 games) | 55.4% | 64.6% (n=359) |
| **+ full 2023 season (2,430 games)** | **55.7%** | **66.4% (n=372)** — still trailing the season-win% baseline (56.1%) overall |

**Second out-of-sample season — 2024, trained on the complete 2023
season (2026-08-17):** model acc **55.7%** vs always-home 52.2% and
season-win% 54.1%; AUC 0.580. Selective ≥0.06 = 61.3% (n=918, above its
own ~59.6% breakeven) and ≥0.10 = 63.9% (n=327, above its ~62.8%
breakeven). Under thin partial-2023 training the same subset swung to
58.3% — training depth matters materially.

**Baseline equivalence (both seasons, full history):** the leak-free
rolling win-% favourite baseline *exactly* reproduces the model's
selective subsets — 2025: 66.4% vs 66.4% (≥0.10), 60.4% vs 60.3% (≥0.06);
2024: 63.9% vs 63.9% (≥0.10), 61.3% vs 61.0% (≥0.06). The confident picks
are favourite picks in every configuration tested; creating genuine lift
needs signal a team-form model cannot see (starter matchups — pipeline
landed, data collection pending) and real odds to measure against.

**Caveat (added 2026-08-17):** a deliberately trivial baseline — pick the
rolling win-% favourite with home tiebreak (state frozen at each day
boundary, shrunk toward .500) — reproduces the selective subsets almost
exactly: 66.4% at ≥0.10 edge (same as the model, agreeing on 451/455
picks) and 60.4% at ≥0.06 (vs 60.3%). The selective-play accuracy is
therefore dominated by "bet the favourite", which the market prices in.
The breakeven prices above are derived from the model's own
probabilities, not real odds, so they are not evidence of monetary edge.
Genuine next levers: starting-pitcher matchup features (in progress,
below) and an odds feed so ROI/CLV replaces breakeven accounting.

## Roadmap

1. ✅ Dual-repo integration (MLB-PBP scoreboard/PBP + StatcastMLB harvesters)
2. ✅ Leak-free features, Monte Carlo simulator, selective filter, walk-forward backtester
3. ✅ Full 2025 season ingested + backtested (2,430 games)
4. ✅ 2024 first-half ingest → multi-year train history (lifted selective ≥0.10 to 61.4%)
5. ✅ Full 2024 season ingest (2,429 games, audited per-team vs official schedules) →
   full two-year history backtest: 56.1% overall, **66.4%** on the ≥0.10-edge subset
5b. ✅ Full 2023 season ingested (2,430 games, exactly 162 per team,
   19 page-captured windows Mar 30 – Oct 1, validated + manifest-ed);
   second out-of-sample season (2024) backtested at full training depth;
   2016–2022 via the same processor or the CI workflow;
   2016–2022 + full 2023 via the CI workflow once `scripts/
   collect-upstream-data.workflow.yml` is copied to `.github/workflows/`
6. 🔶 Statcast pitcher quality features — pipeline landed & tested
   (`collect-probables` → `build-pitcher-features` → `backtest
   --with-pitcher-features`); official Savant pitching boards for
   2023–2025 are committed under `data/ref/statcast/`; remaining step is
   collecting the two seasons of `hydrate=probablePitcher` schedule
   captures (~5 min of egress; the sandbox has none — run the two
   `collect-probables` commands on a laptop or enable the CI workflow in
   `scripts/collect-upstream-data.workflow.yml`)
7. ⬜ Odds feed adapter → ROI/CLV evaluation instead of breakeven prices
8. ⬜ Full PBP-derived features (run expectancy, bullpen load) via `vendor/MLB-PBP`

## Rights

Independent project, not affiliated with or endorsed by Major League
Baseball. MLB data remains subject to the copyright notice returned by
MLB Advanced Media, L.P. in every API response.
