"""Walk-forward backtesting engine.

For every game of the target season the model is trained only on games
that finished on earlier dates, the selective filter is re-applied, the
simulation globals are re-estimated, and only then is the game predicted
(simulation included, so it always runs with the globals known at that
moment). Refits happen at most every ``refit_days`` days.

Reported honestly:
  * model vs naive baselines (coin flip home / sim-only / season win-pct),
  * Brier / log-loss / AUC / accuracy,
  * selective-play subsets (bet only when confident) with breakeven price,
  * totals MAE/RMSE and O/U hit rate at a fixed reference line
    (no historical book lines exist in the upstream repos; ROI vs market
    odds requires an odds feed and is reported as such, not invented).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score

from .features import CANDIDATE_FEATURES, build_features
from .model import GameModel, TotalsModel
from .select import select_features
from .simulate import Globals, estimate_globals, simulate_frame

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "data" / "results"

ML_FEATURES = list(CANDIDATE_FEATURES)
TOTALS_FEATURES = ["off10_total", "def10_total", "d_rf10", "d_ra10",
                   "rest_home", "rest_away"]


@dataclass
class BacktestConfig:
    season: str
    min_games_played: int = 12
    refit_days: int = 7
    tau: float = 0.03
    n_sims: int = 1500
    warmup_min_obs: int = 250
    ou_line: float = 8.5


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    report: dict = field(default_factory=dict)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "brier": float(((p - y) ** 2).mean()),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6),
                                   labels=[0, 1])),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
    }


def run_backtest(games: pd.DataFrame, cfg: BacktestConfig,
                 extra_features: pd.DataFrame | None = None,
                 progress_cb=None) -> BacktestResult:
    """Walk-forward backtest.

    ``extra_features`` (optional): per-game pre-computed features keyed by
    game_pk (e.g. starting-pitcher quality from pitcher_features.py). They
    are merged into the candidate pool and go through the same selective
    filter as everything else. Callers must guarantee they are leak-free;
    pitcher_features.py does this by construction (prior-season boards).
    """
    games = games.sort_values(["official_date", "game_pk"]).reset_index(drop=True)
    feats = build_features(games)
    ml_features = list(ML_FEATURES)
    if extra_features is not None and not extra_features.empty:
        feats = feats.merge(extra_features, on="game_pk", how="left",
                            validate="one_to_one", suffixes=("", "_extra"))
        for col in extra_features.columns:
            if col != "game_pk" and col not in ml_features:
                ml_features.append(col)
        cfg_note = {
            "extra_features": [c for c in extra_features.columns
                               if c != "game_pk"],
        }
    else:
        cfg_note = {"extra_features": []}
    season_mask = feats["season"] == cfg.season

    eligible_dates = sorted(feats.loc[
        season_mask & (feats["min_games_played"] >= cfg.min_games_played),
        "official_date"].unique())
    if not eligible_dates:
        raise ValueError(f"no eligible games for season {cfg.season}")

    preds: list[dict] = []
    model: GameModel | None = None
    totals_model: TotalsModel | None = None
    sim_globals = Globals()
    last_fit: pd.Timestamp | None = None
    selections: list[dict] = []

    for n_done, date in enumerate(eligible_dates):
        if progress_cb and n_done % 14 == 0:
            progress_cb(n_done, len(eligible_dates), date)

        past_mask = feats["official_date"] < date
        train = feats[past_mask & (feats["min_games_played"]
                                   >= cfg.min_games_played)]

        need_fit = (model is None or last_fit is None
                    or (date - last_fit).days >= cfg.refit_days)
        if need_fit and len(train) >= cfg.warmup_min_obs:
            sel = select_features(train, ml_features, "home_win", tau=cfg.tau)
            sim_globals = estimate_globals(
                train["home_score"].to_numpy(), train["away_score"].to_numpy())
            # Simulate the training games with the globals known at this
            # moment (still leak-free: only past games are simulated).
            train_sim = simulate_frame(train, g=sim_globals,
                                       n_sims=min(cfg.n_sims, 500))
            train = train.assign(**{k: v for k, v in train_sim.items()})
            model = GameModel(sel.kept)
            report_fit = model.fit(train)
            sel_t = select_features(train, TOTALS_FEATURES, "total_runs",
                                    tau=cfg.tau)
            totals_model = TotalsModel(sel_t.kept)
            totals_model.fit(train)
            selections.append({
                "fit_before_date": str(pd.Timestamp(date).date()),
                "n_train": int(len(train)),
                "ml_selection": sel.to_dict(),
                "totals_selection": sel_t.to_dict(),
                "model_coefs": {k: round(v, 4) for k, v in
                                report_fit.coeficients.items()},
                "sim_globals": {"home_adv": round(sim_globals.home_adv, 5),
                                "dispersion": round(sim_globals.dispersion, 5)},
            })
            last_fit = date

        if model is None:
            continue  # still warming up

        day_df = feats[season_mask
                       & (feats["official_date"] == date)
                       & (feats["min_games_played"] >= cfg.min_games_played)]
        if day_df.empty:
            continue

        sim_cols = simulate_frame(day_df, g=sim_globals, n_sims=cfg.n_sims)
        day_df = day_df.assign(**{k: v for k, v in sim_cols.items()})

        p_home = model.predict(day_df)
        exp_total = totals_model.predict(day_df)

        for row, ph, et in zip(day_df.itertuples(index=False), p_home, exp_total):
            preds.append({
                "official_date": row.official_date,
                "game_pk": row.game_pk,
                "away_team": row.away_team,
                "home_team": row.home_team,
                "p_home_model": float(ph),
                "sim_p_home": float(row.sim_p_home),
                "exp_total": float(et),
                "home_win": row.home_win,
                "total_runs": row.total_runs,
                "run_diff_home": row.run_diff_home,
            })

    if not preds:
        raise ValueError("backtest produced no predictions (warmup too long?)")

    pdf = pd.DataFrame(preds)
    report: dict = {"season": cfg.season,
                    "config": {"min_games_played": cfg.min_games_played,
                               "refit_days": cfg.refit_days,
                               "tau": cfg.tau, "n_sims": cfg.n_sims,
                               "warmup_min_obs": cfg.warmup_min_obs,
                               **cfg_note},
                    "n_predictions": int(len(pdf)),
                    "n_refits": len(selections),
                    "selections": selections}

    y = pdf["home_win"].to_numpy()
    report["moneyline"] = {
        "model": _metrics(y, pdf["p_home_model"]),
        "baseline_home_50": _metrics(y, np.full(len(y), 0.5)),
        "baseline_sim_only": _metrics(y, pdf["sim_p_home"]),
    }
    feats_idx = feats.set_index("game_pk")
    wp_season_h = feats_idx.loc[pdf["game_pk"], "wp_season_home"].to_numpy()
    report["moneyline"]["baseline_season_wpct"] = _metrics(
        y, np.clip(np.nan_to_num(wp_season_h, nan=0.5), 0.05, 0.95))

    selective = {}
    for edge in (0.03, 0.06, 0.10):
        mask = (pdf["p_home_model"] - 0.5).abs() >= edge
        if mask.sum() >= 20:
            m = _metrics(y[mask], pdf.loc[mask, "p_home_model"])
            m["pct_of_games"] = float(mask.mean())
            fav_p = pdf.loc[mask, "p_home_model"].where(
                pdf.loc[mask, "p_home_model"] >= 0.5,
                1 - pdf.loc[mask, "p_home_model"])
            m["breakeven_price_avg"] = float((1.0 / fav_p).mean())
            selective[f"edge_{edge:.2f}"] = m
    report["selective_play"] = selective

    t = pdf["total_runs"].to_numpy(dtype=float)
    naive = np.full(len(t), t.mean())
    report["totals"] = {
        "ou_reference_line": cfg.ou_line,
        "model_mae": float(np.abs(pdf["exp_total"] - t).mean()),
        "model_rmse": float(np.sqrt(mean_squared_error(t, pdf["exp_total"]))),
        "naive_season_mean_mae": float(np.abs(naive - t).mean()),
        "over_hit_rate": float(((pdf["exp_total"] > cfg.ou_line)
                                == (t > cfg.ou_line)).mean()),
        "over_rate_actual": float((t > cfg.ou_line).mean()),
    }
    report["note_odds"] = (
        "Upstream repos provide official results only (no historical sportsbook "
        "odds). ROI vs the market needs an odds feed; breakeven prices are "
        "reported instead of fabricated odds.")

    return BacktestResult(predictions=pdf, report=report)


def save_result(result: BacktestResult, cfg: BacktestConfig,
                out_dir: Path | None = None) -> tuple[Path, Path]:
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"backtest_{cfg.season}_predictions.csv"
    json_path = out_dir / f"backtest_{cfg.season}_report.json"
    result.predictions.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(result.report, indent=2, default=str))
    return csv_path, json_path
