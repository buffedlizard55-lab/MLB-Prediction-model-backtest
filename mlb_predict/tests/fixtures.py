"""Deterministic synthetic league used ONLY for offline sanity tests.

Clearly synthetic: team names are labelled 'Synthetic Franchise NN'. Never
published or mixed with real MLB data.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

N_TEAMS = 30
GAME_COLUMNS = [
    "game_pk", "season", "official_date", "game_date", "game_type",
    "away_team_id", "away_team", "away_score",
    "home_team_id", "home_team", "home_score",
    "home_win", "total_runs", "run_diff_home",
]


def synthetic_season(season: int, n_games: int = 1500, seed: int = 0,
                     home_adv_runs: float = 0.30) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    team_ids = list(range(100, 100 + N_TEAMS))
    names = [f"Synthetic Franchise {i - 99:02d}" for i in team_ids]
    strength = rng.normal(0.0, 0.55, size=N_TEAMS)  # latent quality

    days = max(1, n_games // (N_TEAMS // 2))
    rows: list[dict] = []
    gp = season * 100000
    start = dt.date(season, 3, 28)
    for d in range(days):
        date = start + dt.timedelta(days=d)
        order = rng.permutation(N_TEAMS)
        for i in range(0, N_TEAMS, 2):
            a, h = int(order[i]), int(order[i + 1])
            lam_a = np.clip(4.4 + strength[a] - strength[h], 1.5, 9.0)
            lam_h = np.clip(4.4 + strength[h] - strength[a] + home_adv_runs,
                            1.5, 9.0)
            sa, sh = int(rng.poisson(lam_a)), int(rng.poisson(lam_h))
            if sa == sh:  # baseball has no ties; give the walk-off to home
                sh += 1
            gp += 1
            rows.append({
                "game_pk": gp, "season": str(season),
                "official_date": date, "game_date": date.isoformat(),
                "game_type": "R",
                "away_team_id": team_ids[a], "away_team": names[a],
                "away_score": sa,
                "home_team_id": team_ids[h], "home_team": names[h],
                "home_score": sh,
                "home_win": int(sh > sa),
                "total_runs": sa + sh,
                "run_diff_home": sh - sa,
            })
    df = pd.DataFrame(rows, columns=GAME_COLUMNS)
    df["official_date"] = pd.to_datetime(df["official_date"])
    return df
