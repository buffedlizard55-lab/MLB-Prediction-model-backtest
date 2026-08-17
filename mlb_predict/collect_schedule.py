"""Scoreboard / schedule collection from the official MLB Stats API.

Endpoint (same one the MLB-PBP archiver enumerates games from):

    https://statsapi.mlb.com/api/v1/schedule
        ?sportId=1&startDate=...&endDate=...&gameTypes=R&hydrate=linescore

We request a restricted ``fields=`` projection so a multi-day window stays
small: gamePk, official date/time, status, team ids/names, final scores,
isWinner. That is exactly the ground truth needed for moneyline, run-line
and total-runs betting outcomes.

Responses are stored verbatim under ``data/raw/schedule/`` (gzipped JSON),
one file per requested window. ``collect`` is resumable: windows already on
disk are skipped unless ``refresh=True``.
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
    "dates,date,games,gamePk,gameType,season,officialDate,gameDate,"
    "status,abstractGameState,teams,away,home,score,isWinner,team,id,name"
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_SCHEDULE_DIR = REPO_ROOT / "data" / "raw" / "schedule"


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
        f"&gameTypes={game_types}&fields={FIELDS}"
    )


def window_path(win_start: dt.date, win_end: dt.date,
                raw_dir: Path | None = None) -> Path:
    raw_dir = raw_dir or RAW_SCHEDULE_DIR
    return raw_dir / f"schedule_{win_start.isoformat()}_{win_end.isoformat()}.json.gz"


def collect_window(win_start: dt.date, win_end: dt.date,
                   raw_dir: Path | None = None, refresh: bool = False,
                   delay: float = DEFAULT_DELAY,
                   fetch=fetch_url) -> Path:
    """Fetch one window (or reuse the cached raw file). Returns file path."""
    path = window_path(win_start, win_end, raw_dir)
    if path.exists() and not refresh:
        return path
    url = window_url(win_start, win_end)
    payload = fetch(url)
    json.loads(payload)  # hard-fail on anything that is not valid JSON
    save_raw(raw_dir or RAW_SCHEDULE_DIR,
             f"schedule_{win_start.isoformat()}_{win_end.isoformat()}.json",
             payload)
    if delay:
        import time
        time.sleep(delay)
    return path


def collect_range(start: str, end: str, window_days: int = 5,
                  raw_dir: Path | None = None, refresh: bool = False,
                  fetch=fetch_url) -> list[Path]:
    """Collect every window covering [start, end]. Returns written files."""
    s = dt.date.fromisoformat(start)
    e = dt.date.fromisoformat(end)
    paths = []
    for w0, w1 in windows(s, e, window_days):
        p = collect_window(w0, w1, raw_dir=raw_dir, refresh=refresh, fetch=fetch)
        paths.append(p)
    return paths


def load_collected(raw_dir: Path | None = None) -> list[dict]:
    """Parse every cached schedule window into a list of JSON payloads."""
    raw_dir = raw_dir or RAW_SCHEDULE_DIR
    payloads: list[dict] = []
    for path in sorted(raw_dir.glob("schedule_*.json.gz")):
        payloads.append(json.loads(load_raw(path)))
    return payloads


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True, help="YYYY-MM-DD inclusive")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD inclusive")
    ap.add_argument("--window-days", type=int, default=5)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)
    paths = collect_range(args.start, args.end, args.window_days,
                          refresh=args.refresh)
    print(f"collected {len(paths)} schedule windows under {RAW_SCHEDULE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
