"""Parse collected schedule windows into a flat, verified games table.

Only rows that are genuine ground truth survive the filter:

* ``gameType == 'R'`` (regular season),
* ``status.abstractGameState == 'Final'``,
* both scores present, exactly one ``isWinner == True``.

The output columns feed features.py and backtest.py. No derived or guessed
values are ever invented here — anything missing upstream stays missing.
"""

from __future__ import annotations

import pandas as pd

GAME_COLUMNS = [
    "game_pk", "season", "official_date", "game_date", "game_type",
    "away_team_id", "away_team", "away_score",
    "home_team_id", "home_team", "home_score",
    "home_win", "total_runs", "run_diff_home",
]


def _iter_games(payload: dict):
    for day in payload.get("dates", []):
        for game in day.get("games", []):
            yield game


def payload_to_records(payload: dict) -> list[dict]:
    rows: list[dict] = []
    for g in _iter_games(payload):
        if g.get("gameType") != "R":
            continue
        status = (g.get("status") or {}).get("abstractGameState")
        if status != "Final":
            continue
        away = (g.get("teams") or {}).get("away") or {}
        home = (g.get("teams") or {}).get("home") or {}
        a_score, h_score = away.get("score"), home.get("score")
        if a_score is None or h_score is None:
            continue
        a_win, h_win = away.get("isWinner"), home.get("isWinner")
        # Ground-truth consistency: exactly one declared winner and it
        # must agree with the scores. Skip anything contradictory.
        if a_win == h_win:
            continue
        if bool(h_win) != (h_score > a_score):
            continue
        rows.append({
            "game_pk": int(g["gamePk"]),
            "season": str(g.get("season", "")),
            "official_date": g.get("officialDate", ""),
            "game_date": g.get("gameDate", ""),
            "game_type": g.get("gameType", "R"),
            "away_team_id": int((away.get("team") or {}).get("id", -1)),
            "away_team": (away.get("team") or {}).get("name", ""),
            "away_score": int(a_score),
            "home_team_id": int((home.get("team") or {}).get("id", -1)),
            "home_team": (home.get("team") or {}).get("name", ""),
            "home_score": int(h_score),
            "home_win": int(bool(h_win)),
            "total_runs": int(a_score) + int(h_score),
            "run_diff_home": int(h_score) - int(a_score),
        })
    return rows


def payloads_to_frame(payloads: list[dict]) -> pd.DataFrame:
    records: list[dict] = []
    for p in payloads:
        records.extend(payload_to_records(p))
    df = pd.DataFrame.from_records(records, columns=GAME_COLUMNS)
    if df.empty:
        return df
    df["official_date"] = pd.to_datetime(df["official_date"])
    df = df.drop_duplicates(subset="game_pk").sort_values(
        ["official_date", "game_pk"]).reset_index(drop=True)
    return df
