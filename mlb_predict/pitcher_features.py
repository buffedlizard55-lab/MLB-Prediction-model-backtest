"""Starting-pitcher quality features from Statcast leaderboards.

For every game with official probable pitchers, each starter is scored
from the **previous completed season's** Baseball Savant custom pitching
leaderboard (expected wOBA / xERA allowed, K%, BB%, innings). Using only
the prior season makes the join leak-free by construction: a 2025 game
can only ever see 2024 (or earlier) pitcher quality.

Pitchers missing from the prior-season board (rookies, low workload)
stay missing — NaN values plus a 0/1 ``sp_known`` flag so the model can
treat them explicitly. Nothing is imputed from the current season.

All diff features are oriented so that **positive favours the home
team**:
    lower-is-better metrics (xwOBA, xERA): away − home
    higher-is-better metrics (K%, IP):     home − away
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .collect_statcast import load_leaderboard

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_STATCAST_DIR = REPO_ROOT / "data" / "raw" / "statcast"
PITCHER_FEATURES_CSV = REPO_ROOT / "data" / "ref" / "pitcher_features.csv"

BOARD = "pitching_quality"          # collect_statcast leaderboard name
MIN_PA = 100

# Candidate pool handed to the selective filter (sign-oriented diffs +
# knowledge flags). Per-side raw values are kept for audit columns only.
PITCHER_CANDIDATES = [
    "sp_xwoba_home", "sp_xwoba_away", "sp_xwoba_d",
    "sp_xera_d", "sp_k_pct_d", "sp_bb_pct_d",
    "sp_ip_home", "sp_ip_away",
    "sp_known_home", "sp_known_away", "sp_known_both",
]

_BOARD_RENAMES = {
    "xwoba": "board_xwoba",
    "xera": "board_xera",
    "k_percent": "board_k_pct",
    "bb_percent": "board_bb_pct",
    "p_formatted_ip": "board_ip",
}


def _parse_ip(formatted: pd.Series) -> pd.Series:
    """'154.1' innings -> 154 + 1/3 = 154.333 (Statcast .1/.2 = outs)."""
    def one(v):
        if pd.isna(v):
            return float("nan")
        try:
            s = str(v).strip()
            if "." in s:
                whole, frac = s.split(".")
                return int(whole) + int(frac) / 3.0
            return float(int(s))
        except (TypeError, ValueError):
            return float("nan")
    return formatted.map(one)


def load_board(year: int, raw_dir: Path | None = None) -> pd.DataFrame:
    """Load the pitching-quality board for one season, normalised.

    Looks for the verbatim gz capture under data/raw/statcast/ first, then
    falls back to the committed reference CSV under data/ref/statcast/
    (same parser output, kept in git so the pipeline runs anywhere).
    """
    import gzip as _gzip
    from .collect_statcast import leaderboard_path
    candidates = [leaderboard_path(BOARD, year, min_pa=MIN_PA, raw_dir=raw_dir),
                  REPO_ROOT / "data" / "ref" / "statcast"
                  / f"{BOARD}_{year}_min{MIN_PA}.csv"]
    df = None
    for path in candidates:
        if path.exists():
            opener = _gzip.open if str(path).endswith(".gz") else open
            with opener(path, "rb") as fh:
                df = pd.read_csv(fh)
            break
    if df is None:
        raise FileNotFoundError(
            f"no pitching_quality board for {year} (looked in: "
            + ", ".join(str(c) for c in candidates) + ")")
    df = df.rename(columns=_BOARD_RENAMES)
    out = pd.DataFrame({
        "player_id": df["player_id"].astype(int),
        "player_name": df.get("last_name, first_name", ""),
        "board_year": year,
        "xwoba": pd.to_numeric(df.get("board_xwoba"), errors="coerce"),
        "xera": pd.to_numeric(df.get("board_xera"), errors="coerce"),
        "k_pct": pd.to_numeric(df.get("board_k_pct"), errors="coerce"),
        "bb_pct": pd.to_numeric(df.get("board_bb_pct"), errors="coerce"),
        "ip": _parse_ip(df.get("board_ip")),
    })
    out = out.dropna(subset=["xwoba"])
    out = out.drop_duplicates(subset="player_id", keep="first")
    return out


def _starter_stats(sp_id: pd.Series, season: pd.Series,
                   boards: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Prior-season board row for each starter appearance."""
    prev_year = season.astype(int) - 1
    cols = ["xwoba", "xera", "k_pct", "bb_pct", "ip"]
    out = {c: pd.Series(float("nan"), index=sp_id.index) for c in cols}
    out["known"] = pd.Series(0.0, index=sp_id.index)
    for year in prev_year.unique():
        board = boards.get(year)
        if board is None or pd.isna(year):
            continue
        lookup = {int(r.player_id): r for r in board.itertuples(index=False)}
        mask = prev_year == year
        for idx in sp_id[mask].index:
            pid = sp_id.loc[idx]
            if pd.isna(pid):
                continue
            row = lookup.get(int(pid))
            if row is None:
                continue
            for c in cols:
                out[c].loc[idx] = getattr(row, c)
            out["known"].loc[idx] = 1.0
    return pd.DataFrame(out)


def build_pitcher_features(probables: pd.DataFrame,
                           board_years: list[int],
                           raw_dir: Path | None = None) -> pd.DataFrame:
    """One modelling row per game: starter quality + leak-free provenance."""
    boards = {}
    for y in board_years:
        try:
            boards[y] = load_board(y, raw_dir=raw_dir)
        except FileNotFoundError:
            continue  # missing season board stays missing

    season = probables["season"].astype(int)
    away = _starter_stats(probables["away_sp_id"], season, boards)
    home = _starter_stats(probables["home_sp_id"], season, boards)

    df = pd.DataFrame({
        "game_pk": probables["game_pk"],
        "away_sp_id": probables["away_sp_id"],
        "home_sp_id": probables["home_sp_id"],
        "away_sp_name": probables["away_sp_name"],
        "home_sp_name": probables["home_sp_name"],
        "sp_xwoba_away": away["xwoba"], "sp_xwoba_home": home["xwoba"],
        "sp_xera_away": away["xera"], "sp_xera_home": home["xera"],
        "sp_k_pct_away": away["k_pct"], "sp_k_pct_home": home["k_pct"],
        "sp_bb_pct_away": away["bb_pct"], "sp_bb_pct_home": home["bb_pct"],
        "sp_ip_away": away["ip"], "sp_ip_home": home["ip"],
        "sp_known_away": away["known"], "sp_known_home": home["known"],
    })
    # Orientation: positive favours the home team.
    df["sp_xwoba_d"] = df["sp_xwoba_away"] - df["sp_xwoba_home"]  # lower better
    df["sp_xera_d"] = df["sp_xera_away"] - df["sp_xera_home"]     # lower better
    df["sp_k_pct_d"] = df["sp_k_pct_home"] - df["sp_k_pct_away"]  # higher better
    df["sp_bb_pct_d"] = df["sp_bb_pct_away"] - df["sp_bb_pct_home"]  # lower better
    df["sp_known_both"] = (df["sp_known_away"] * df["sp_known_home"])
    return df


def coverage_report(pf: pd.DataFrame, probables: pd.DataFrame,
                    games: pd.DataFrame) -> dict:
    """Honest coverage accounting for the README / reports."""
    n_games = len(games)
    have_row = int(pf["game_pk"].nunique()) if not pf.empty else 0
    both = int(((pf["sp_known_both"] == 1)).sum()) if not pf.empty else 0
    either = int(((pf["sp_known_away"] == 1) |
                  (pf["sp_known_home"] == 1)).sum()) if not pf.empty else 0
    return {
        "games_csv_rows": int(n_games),
        "probables_rows": int(len(probables)),
        "pitcher_feature_rows": have_row,
        "both_starters_with_prior_season_stats": both,
        "at_least_one_starter_with_prior_season_stats": either,
        "note": "missing = rookie/low-workload starter not on the prior-"
                "season Savant board (min 100 PA against); stays NaN",
    }
