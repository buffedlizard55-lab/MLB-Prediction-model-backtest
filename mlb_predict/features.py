"""Walk-forward-safe game feature engineering.

Every feature for a game is computed **strictly from games that already
finished before that game's official date** (shift-before-roll on each
team's own timeline), so the backtester can never peek at the future.

Features are intentionally a *wide candidate pool*; ``select.py`` then
keeps only the inputs whose correlation with outcomes survives the
selective filter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PYTH_EXPONENT = 1.83          # classic Bill James exponent for MLB
ROLL_W = 10                   # rolling window (games)
MIN_GAMES_FOR_ROLL = 3        # below this we fall back to expanding stats

CANDIDATE_FEATURES = [
    "home_flag",
    "rest_home", "rest_away", "d_rest",
    "rf10_home", "rf10_away", "d_rf10",
    "ra10_home", "ra10_away", "d_ra10",
    "rd10_home", "rd10_away", "d_rd10",
    "wp10_home", "wp10_away", "d_wp10",
    "wp_season_home", "wp_season_away", "d_wp_season",
    "pyth_season_home", "pyth_season_away", "d_pyth_season",
    "off10_total", "def10_total",
]

LABELS = ["home_win", "total_runs", "run_diff_home"]


def _team_games(games: pd.DataFrame) -> pd.DataFrame:
    """Explode games into one row per team appearance."""
    home = pd.DataFrame({
        "game_pk": games["game_pk"],
        "official_date": games["official_date"],
        "team_id": games["home_team_id"],
        "runs_for": games["home_score"],
        "runs_against": games["away_score"],
        "win": games["home_win"],
        "is_home": 1,
    })
    away = pd.DataFrame({
        "game_pk": games["game_pk"],
        "official_date": games["official_date"],
        "team_id": games["away_team_id"],
        "runs_for": games["away_score"],
        "runs_against": games["home_score"],
        "win": 1 - games["home_win"],
        "is_home": 0,
    })
    tg = pd.concat([home, away], ignore_index=True)
    tg = tg.sort_values(["team_id", "official_date", "game_pk"],
                        kind="mergesort").reset_index(drop=True)
    return tg


def _roll_or_expand(series: pd.Series, window: int) -> pd.Series:
    """Rolling mean that expands while the window is not yet full."""
    rolled = series.rolling(window, min_periods=1).mean()
    return rolled


def build_team_state(games: pd.DataFrame, window: int = ROLL_W) -> pd.DataFrame:
    """Pre-game rolling state for every team appearance (no leakage)."""
    tg = _team_games(games)
    grp = tg.groupby("team_id", sort=False)

    # Anything computed here sees only *prior* rows once we shift(1).
    rf = grp["runs_for"].transform(_roll_or_expand, window)
    ra = grp["runs_against"].transform(_roll_or_expand, window)
    w = grp["win"].transform(_roll_or_expand, window)
    gp = grp.cumcount() + 1
    wins_cum = grp["win"].transform("cumsum")
    rs_cum = grp["runs_for"].transform("cumsum")
    ra_cum = grp["runs_against"].transform("cumsum")

    prev_last_date = grp["official_date"].transform("shift", 1)

    state = pd.DataFrame({
        "game_pk": tg["game_pk"],
        "team_id": tg["team_id"],
        "is_home": tg["is_home"],
        "official_date": tg["official_date"],
        "games_played": gp - 1,
        "rf_w": rf.shift(1),
        "ra_w": ra.shift(1),
        "wp_w": w.shift(1),
        "wp_season": (wins_cum.shift(1) / (gp - 1)),
        "rs_season": rs_cum.shift(1),
        "ra_season": ra_cum.shift(1),
        "prev_date": prev_last_date,
    })
    # First career game of a franchise window: no past exists -> NaN.
    state.loc[state["games_played"] == 0,
              ["rf_w", "ra_w", "wp_w", "wp_season"]] = np.nan
    return state


def build_features(games: pd.DataFrame, window: int = ROLL_W) -> pd.DataFrame:
    """Join home/away pre-game states into one modelling row per game."""
    state = build_team_state(games, window=window)
    home = state[state["is_home"] == 1].drop(columns=["is_home"]).add_suffix("_home")
    away = state[state["is_home"] == 0].drop(columns=["is_home"]).add_suffix("_away")
    home = home.rename(columns={"game_pk_home": "game_pk",
                                "team_id_home": "home_team_id"})
    away = away.rename(columns={"game_pk_away": "game_pk",
                                "team_id_away": "away_team_id"})

    df = games.merge(home, on=["game_pk", "home_team_id"], how="left",
                     validate="one_to_one")
    df = df.merge(away.drop(columns=["official_date_away"]),
                  on=["game_pk", "away_team_id"], how="left",
                  validate="one_to_one")

    df["home_flag"] = 1.0

    rest_h = (df["official_date"] - df["prev_date_home"]).dt.days - 1
    rest_a = (df["official_date"] - df["prev_date_away"]).dt.days - 1
    df["rest_home"] = rest_h.clip(lower=0)
    df["rest_away"] = rest_a.clip(lower=0)

    pyth_h = df["rs_season_home"] ** PYTH_EXPONENT / (
        df["rs_season_home"] ** PYTH_EXPONENT + df["ra_season_home"] ** PYTH_EXPONENT)
    pyth_a = df["rs_season_away"] ** PYTH_EXPONENT / (
        df["rs_season_away"] ** PYTH_EXPONENT + df["ra_season_away"] ** PYTH_EXPONENT)

    df["rf10_home"], df["rf10_away"] = df["rf_w_home"], df["rf_w_away"]
    df["ra10_home"], df["ra10_away"] = df["ra_w_home"], df["ra_w_away"]
    df["wp10_home"], df["wp10_away"] = df["wp_w_home"], df["wp_w_away"]
    df["pyth_season_home"], df["pyth_season_away"] = pyth_h, pyth_a
    df["wp_season_home"], df["wp_season_away"] = (
        df["wp_season_home"], df["wp_season_away"])

    df["rd10_home"] = df["rf10_home"] - df["ra10_home"]
    df["rd10_away"] = df["rf10_away"] - df["ra10_away"]
    df["d_rest"] = df["rest_home"] - df["rest_away"]
    df["d_rf10"] = df["rf10_home"] - df["rf10_away"]
    df["d_ra10"] = df["ra10_away"] - df["ra10_home"]   # +ve = home pitch edge
    df["d_rd10"] = (df["rf10_home"] - df["ra10_home"]) - \
                   (df["rf10_away"] - df["ra10_away"])
    df["d_wp10"] = df["wp10_home"] - df["wp10_away"]
    df["d_wp_season"] = df["wp_season_home"] - df["wp_season_away"]
    df["d_pyth_season"] = df["pyth_season_home"] - df["pyth_season_away"]
    df["off10_total"] = df["rf10_home"] + df["rf10_away"]
    df["def10_total"] = df["ra10_home"] + df["ra10_away"]
    df["min_games_played"] = np.minimum(
        df["games_played_home"].fillna(0), df["games_played_away"].fillna(0))

    keep = (list(games.columns) + CANDIDATE_FEATURES
            + ["games_played_home", "games_played_away", "min_games_played"])
    df = df[[c for c in dict.fromkeys(keep) if c in df.columns]]
    return df
