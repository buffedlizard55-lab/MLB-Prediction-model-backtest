"""Probable starting-pitcher collection from the official MLB Stats API.

Past games do not expose probable pitchers through the plain schedule
response — the ``probablePitcher`` hydration is required:

    https://statsapi.mlb.com/api/v1/schedule
        ?sportId=1&startDate=...&endDate=...&gameTypes=R
        &hydrate=probablePitcher&fields=...

The ``fields=`` projection keeps each window small: gamePk, season, date,
status, team ids/names and the probable pitcher (id + fullName) per side.
Responses are stored verbatim under ``data/raw/probables/`` (gzipped JSON),
one file per requested window, resumable like ``collect_schedule``.

These are the *official pre-game probables* as listed by MLB for each
game. Late scratches happen; they stay as-is (zero-hallucination rule —
nothing is back-filled from the boxscore).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from .http_io import DEFAULT_DELAY, fetch_url, load_raw, save_raw

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"

FIELDS = (
    "dates,games,gamePk,gameType,season,officialDate,"
    "status,abstractGameState,"
    "teams,away,home,team,id,name,probablePitcher,fullName"
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_PROBABLES_DIR = REPO_ROOT / "data" / "raw" / "probables"
PROBABLES_MANIFEST = REPO_ROOT / "data" / "raw" / "probables_manifest.csv"


def windows(start: dt.date, end: dt.date, days: int):
    """Yield inclusive (win_start, win_end) windows of at most `days` days."""
    cur = start
    while cur <= end:
        yield cur, min(cur + dt.timedelta(days=days - 1), end)
        cur = cur + dt.timedelta(days=days)


def window_url(win_start: dt.date, win_end: dt.date, game_types: str = "R") -> str:
    return (
        f"{SCHEDULE_URL}?sportId=1"
        f"&startDate={win_start.isoformat()}&endDate={win_end.isoformat()}"
        f"&gameTypes={game_types}&hydrate=probablePitcher&fields={FIELDS}"
    )


def window_path(win_start: dt.date, win_end: dt.date,
                raw_dir: Path | None = None) -> Path:
    raw_dir = raw_dir or RAW_PROBABLES_DIR
    return raw_dir / f"probables_{win_start.isoformat()}_{win_end.isoformat()}.json.gz"


def collect_window(win_start: dt.date, win_end: dt.date,
                   raw_dir: Path | None = None, refresh: bool = False,
                   delay: float = DEFAULT_DELAY,
                   fetch=fetch_url) -> Path:
    """Fetch one probables window (or reuse the cached raw file)."""
    path = window_path(win_start, win_end, raw_dir)
    if path.exists() and not refresh:
        return path
    url = window_url(win_start, win_end)
    payload = fetch(url)
    json.loads(payload)  # hard-fail on anything that is not valid JSON
    save_raw(raw_dir or RAW_PROBABLES_DIR,
             f"probables_{win_start.isoformat()}_{win_end.isoformat()}.json",
             payload)
    _append_manifest(url, payload)
    if delay:
        import time
        time.sleep(delay)
    return path


def _append_manifest(url: str, payload: bytes) -> None:
    try:
        doc = json.loads(payload)
    except json.JSONDecodeError:
        return
    n = sum(len(d.get("games", [])) for d in doc.get("dates", []))
    new = not PROBABLES_MANIFEST.exists()
    with PROBABLES_MANIFEST.open("a") as fh:
        if new:
            fh.write("retrieved_utc,source_url,rows\n")
        stamp = dt.datetime.now(dt.timezone.utc).isoformat()
        fh.write(f"{stamp},{url},{n}\n")


def collect_range(start: str, end: str, window_days: int = 5,
                  raw_dir: Path | None = None, refresh: bool = False,
                  fetch=fetch_url) -> list[Path]:
    """Collect every probables window covering [start, end]."""
    s = dt.date.fromisoformat(start)
    e = dt.date.fromisoformat(end)
    paths = []
    for w0, w1 in windows(s, e, window_days):
        p = collect_window(w0, w1, raw_dir=raw_dir, refresh=refresh, fetch=fetch)
        paths.append(p)
    return paths


def load_collected(raw_dir: Path | None = None) -> list[dict]:
    """Parse every cached probables window into a list of JSON payloads."""
    raw_dir = raw_dir or RAW_PROBABLES_DIR
    payloads: list[dict] = []
    for path in sorted(raw_dir.glob("probables_*.json.gz")):
        payloads.append(json.loads(load_raw(path)))
    return payloads


# --------------------------------------------------------------------------
# Verified parser (same rules as parse_schedule: only clean rows survive)
# --------------------------------------------------------------------------

PROBABLE_COLUMNS = [
    "game_pk", "season", "official_date", "game_type",
    "away_team_id", "away_sp_id", "away_sp_name",
    "home_team_id", "home_sp_id", "home_sp_name",
]


def payload_to_records(payload: dict) -> list[dict]:
    rows: list[dict] = []
    for day in payload.get("dates", []):
        for g in day.get("games", []):
            if g.get("gameType") != "R":
                continue
            status = (g.get("status") or {}).get("abstractGameState")
            if status != "Final":
                continue
            away = (g.get("teams") or {}).get("away") or {}
            home = (g.get("teams") or {}).get("home") or {}
            a_team = away.get("team") or {}
            h_team = home.get("team") or {}
            if "id" not in a_team or "id" not in h_team:
                continue
            a_sp = away.get("probablePitcher") or {}
            h_sp = home.get("probablePitcher") or {}
            rows.append({
                "game_pk": int(g["gamePk"]),
                "season": str(g.get("season", "")),
                "official_date": g.get("officialDate", ""),
                "game_type": g.get("gameType", "R"),
                "away_team_id": int(a_team["id"]),
                # Missing probables stay missing (no invention).
                "away_sp_id": int(a_sp["id"]) if "id" in a_sp else None,
                "away_sp_name": a_sp.get("fullName"),
                "home_team_id": int(h_team["id"]),
                "home_sp_id": int(h_sp["id"]) if "id" in h_sp else None,
                "home_sp_name": h_sp.get("fullName"),
            })
    return rows


def payloads_to_frame(payloads: list[dict]):
    import pandas as pd

    records: list[dict] = []
    for p in payloads:
        records.extend(payload_to_records(p))
    df = pd.DataFrame.from_records(records, columns=PROBABLE_COLUMNS)
    if df.empty:
        return df
    df["official_date"] = pd.to_datetime(df["official_date"])
    df = df.drop_duplicates(subset="game_pk").sort_values(
        ["official_date", "game_pk"]).reset_index(drop=True)
    return df


def verify_against_games(probables, games) -> dict:
    """Cross-check the probables table against the verified games table.

    * every probables row must match games.csv on season / date / team ids
      for the same game_pk (mismatches are dropped loudly);
    * reports how many games have both starters, one starter or none.

    Returns (verified_probables_frame, report_dict).
    """
    import pandas as pd

    report: dict = {}
    if probables.empty or games.empty:
        report["error"] = "empty input"
        return probables, report

    g = games[["game_pk", "season", "official_date",
               "away_team_id", "home_team_id"]]
    m = probables.merge(g, on="game_pk", how="left",
                        suffixes=("", "_games"), validate="one_to_one")
    bad = (
        m["season_games"].isna()
        | (m["season"] != m["season_games"])
        | (m["official_date_games"] != m["official_date"])
        | (m["away_team_id"] != m["away_team_id_games"])
        | (m["home_team_id"] != m["home_team_id_games"])
    )
    report["rows"] = int(len(m))
    report["dropped_team_or_date_mismatch"] = int(bad.sum())
    if bad.any():
        dropped = m.loc[bad, ["game_pk"]].astype(int).values.ravel().tolist()
        report["dropped_game_pks"] = dropped
    m = m.loc[~bad].drop(columns=[c for c in m.columns if c.endswith("_games")])

    both = m["away_sp_id"].notna() & m["home_sp_id"].notna()
    either = m["away_sp_id"].notna() | m["home_sp_id"].notna()
    report["with_both_starters"] = int(both.sum())
    report["with_one_starter"] = int((either & ~both).sum())
    report["with_no_starter"] = int((~either).sum())
    # games in games.csv that have no probables row at all
    report["games_without_probables_row"] = int(
        len(games) - m["game_pk"].nunique())
    return m.reset_index(drop=True), report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True, help="YYYY-MM-DD inclusive")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD inclusive")
    ap.add_argument("--window-days", type=int, default=5)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)
    paths = collect_range(args.start, args.end, args.window_days,
                          refresh=args.refresh)
    print(f"collected {len(paths)} probables windows under {RAW_PROBABLES_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
