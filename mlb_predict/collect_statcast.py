"""Statcast collection via Baseball Savant (official host).

Two entry points:

1. ``collect_leaderboard`` — small, self-contained CSV leaderboards
   (sprint speed, bat tracking, expected stats, arsenal run values, ...).
   The exact URLs are the ones reverse-engineered and documented by
   karagemop466-tech/StatcastMLB (vendor/StatcastMLB, statcastmlb/endpoints.py).

2. ``statcastmlb_pipeline`` — for bulk pitch-level harvesting, validation
   and the SHA-256 integrity ledger, use the vendored StatcastMLB package
   itself on a machine with direct MLB egress:

       pip install -e vendor/StatcastMLB
       statcast --help

Local CSV copies are cached under ``data/raw/statcast/``.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd

from .http_io import fetch_url

SAVANT = "https://baseballsavant.mlb.com"

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_STATCAST_DIR = REPO_ROOT / "data" / "raw" / "statcast"

# Leaderboard URL templates verified against StatcastMLB's endpoint docs.
LEADERBOARDS: dict[str, str] = {
    "sprint_speed": f"{SAVANT}/leaderboard/sprint_speed?year={{year}}&min={{min_pa}}&csv=true",
    "bat_tracking": f"{SAVANT}/leaderboard/bat-tracking?year={{year}}&csv=true",
    "pitch_arsenal": (f"{SAVANT}/leaderboard/pitch-arsenal-stats"
                      f"?type=pitcher&year={{year}}&min={{min_pa}}&csv=true"),
    "custom_batting": (f"{SAVANT}/leaderboard/custom?year={{year}}&type=batter"
                       f"&min={{min_pa}}&selections=k_bb,avg,xba,slg,xslg,woba,xwoba,"
                       f"exit_velo_avg,barrels,brl_pct,hard_hit,sweet_spot_percent,"
                       f"avg_best_speed&chart=false&csv=true"),
    "custom_pitching": (f"{SAVANT}/leaderboard/custom?year={{year}}&type=pitcher"
                        f"&min={{min_pa}}&selections=k_bb,avg,xba,slg,xslg,woba,xwoba,"
                        f"exit_velo_avg,barrels,brl_pct,hard_hit,sweet_spot_percent,"
                        f"avg_best_speed&chart=false&csv=true"),
    "pitch_movement": (f"{SAVANT}/leaderboard/pitch-movement?year={{year}}"
                       f"&pitch_type=FF&min_pitches={{min_pa}}&csv=true"),
    "poptime": f"{SAVANT}/leaderboard/poptime?year={{year}}&min2b={{min_pa}}&csv=true",
    "arm_strength": f"{SAVANT}/leaderboard/arm-strength?year={{year}}&min_throws={{min_pa}}&csv=true",
}


def leaderboard_url(name: str, year: int, min_pa: int = 10) -> str:
    try:
        tpl = LEADERBOARDS[name]
    except KeyError as err:
        raise KeyError(f"unknown leaderboard {name!r}; "
                       f"choices: {sorted(LEADERBOARDS)}") from err
    return tpl.format(year=year, min_pa=min_pa)


def leaderboard_path(name: str, year: int, min_pa: int = 10,
                     raw_dir: Path | None = None) -> Path:
    raw_dir = raw_dir or RAW_STATCAST_DIR
    return raw_dir / f"{name}_{year}_min{min_pa}.csv.gz"


def collect_leaderboard(name: str, year: int, min_pa: int = 10,
                        raw_dir: Path | None = None, refresh: bool = False,
                        fetch=fetch_url) -> Path:
    """Fetch one Savant leaderboard CSV, validate it parses, cache it."""
    path = leaderboard_path(name, year, min_pa, raw_dir)
    if path.exists() and not refresh:
        return path
    url = leaderboard_url(name, year, min_pa)
    payload = fetch(url)
    text = payload.decode("utf-8", errors="replace")
    pd.read_csv(io.StringIO(text))  # must parse as CSV or we fail loudly
    import gzip
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as fh:
        fh.write(payload)
    return path


def load_leaderboard(name: str, year: int, min_pa: int = 10,
                     raw_dir: Path | None = None) -> pd.DataFrame:
    import gzip
    path = leaderboard_path(name, year, min_pa, raw_dir)
    with gzip.open(path, "rb") as fh:
        return pd.read_csv(fh)


def statcastmlb_pipeline() -> None:  # pragma: no cover - doc helper
    """Point the caller at the vendored StatcastMLB package for bulk work."""
    raise RuntimeError(
        "Bulk pitch-level harvesting uses the vendored StatcastMLB package:\n"
        "    pip install -e vendor/StatcastMLB\n"
        "    statcast collect --start YYYY-MM-DD --end YYYY-MM-DD\n"
        "See vendor/StatcastMLB/README.md for the full CLI."
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--board", required=True, choices=sorted(LEADERBOARDS))
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--min-pa", type=int, default=10)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)
    path = collect_leaderboard(args.board, args.year, args.min_pa,
                               refresh=args.refresh)
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
