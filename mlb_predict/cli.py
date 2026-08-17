"""Command-line entry point.

    python -m mlb_predict collect --start 2025-03-25 --end 2025-09-30
    python -m mlb_predict ingest-page CAPTURE.txt --window 2025-04-01 2025-04-05
    python -m mlb_predict collect-statcast --board sprint_speed --year 2025
    python -m mlb_predict build-dataset
    python -m mlb_predict backtest --season 2025
    python -m mlb_predict demo            # offline synthetic sanity chain
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

from . import collect_schedule, collect_statcast, http_io
from .backtest import (BacktestConfig, run_backtest, save_result)
from .collect_schedule import load_collected
from .parse_schedule import payloads_to_frame

REPO_ROOT = Path(__file__).resolve().parent.parent
GAMES_CSV = REPO_ROOT / "data" / "games.csv"
TEAM_NAMES_CSV = REPO_ROOT / "data" / "ref" / "teams_official.csv"


def _team_names() -> dict[int, str]:
    if TEAM_NAMES_CSV.exists():
        t = pd.read_csv(TEAM_NAMES_CSV)
        return dict(zip(t["id"].astype(int), t["name"].astype(str)))
    return {}


TEAM_NAMES = _team_names()


def cmd_collect(args) -> int:
    paths = collect_schedule.collect_range(
        args.start, args.end, args.window_days, refresh=args.refresh)
    print(f"collected {len(paths)} schedule windows under "
          f"{collect_schedule.RAW_SCHEDULE_DIR}")
    return 0


def cmd_ingest_page(args) -> int:
    """Store a page-captured schedule response into the raw cache."""
    text = Path(args.capture).read_text()
    payload = http_io.extract_json_payload(text)
    if "dates" not in payload:
        raise SystemExit("capture does not look like a schedule response")
    w0, w1 = args.window
    dest = collect_schedule.window_path(
        dt.date.fromisoformat(w0), dt.date.fromisoformat(w1))
    dest.parent.mkdir(parents=True, exist_ok=True)
    import gzip
    with gzip.open(dest, "wb") as fh:
        fh.write(json.dumps(payload).encode())
    n_games = sum(len(d.get("games", [])) for d in payload.get("dates", []))
    print(f"ingested {n_games} games -> {dest}")
    return 0


def cmd_ingest_scores(args) -> int:
    """Ingest a minimal verified scores CSV (transcribed 1:1 from official
    statsapi schedule responses): game_pk,official_date,away_id,away_score,
    home_id,home_score. Appends to data/raw/scores_transcript.csv and keeps
    a manifest of the source URL windows in data/raw/scores_manifest.csv."""
    src = Path(args.csv)
    lines = [ln.strip() for ln in src.read_text().splitlines()
             if ln.strip() and not ln.startswith("game_pk")]
    out = REPO_ROOT / "data" / "raw" / "scores_transcript.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not out.exists()
    with out.open("a") as fh:
        if header_needed:
            fh.write("game_pk,official_date,away_id,away_score,"
                     "home_id,home_score\n")
        fh.write("\n".join(lines) + "\n")
    man = REPO_ROOT / "data" / "raw" / "scores_manifest.csv"
    with man.open("a") as fh:
        if not man.exists() or man.stat().st_size == 0:
            fh.write("retrieved_utc,source_url,rows\n")
        import datetime as _dt
        fh.write(f"{_dt.datetime.now(_dt.timezone.utc).isoformat()},"
                 f"{args.source_url},{len(lines)}\n")
    print(f"appended {len(lines)} scores -> {out}")
    return 0


def cmd_build_dataset_from_scores(args) -> int:
    """Build data/games.csv from the verified scores transcript."""
    import datetime as _dt
    src = REPO_ROOT / "data" / "raw" / "scores_transcript.csv"
    if not src.exists():
        raise SystemExit("no data/raw/scores_transcript.csv yet")
    raw = pd.read_csv(src)
    rows = []
    for r in raw.itertuples(index=False):
        if pd.isna(r.away_score) or pd.isna(r.home_score):
            continue
        if int(r.home_score) == int(r.away_score):
            continue  # MLB regular-season games cannot end tied
        rows.append({
            "game_pk": int(r.game_pk),
            "season": str(r.official_date)[:4],
            "official_date": r.official_date,
            "game_date": r.official_date,
            "game_type": "R",
            "away_team_id": int(r.away_id),
            "away_team": TEAM_NAMES.get(int(r.away_id), f"team {int(r.away_id)}"),
            "away_score": int(r.away_score),
            "home_team_id": int(r.home_id),
            "home_team": TEAM_NAMES.get(int(r.home_id), f"team {int(r.home_id)}"),
            "home_score": int(r.home_score),
        })
    games = pd.DataFrame(rows)
    games = games.drop_duplicates(subset="game_pk")

    # Union with an existing games.csv: the transcript is append-only and
    # may cover only part of history (e.g. scores captures plus seasons
    # collected via the raw schedule path). Rows already in games.csv that
    # are not in this transcript must survive a rebuild.
    if GAMES_CSV.exists():
        prior = pd.read_csv(GAMES_CSV, dtype={"game_pk": int, "season": str})
        prior["official_date"] = pd.to_datetime(prior["official_date"])
        missing = prior[~prior["game_pk"].isin(set(games["game_pk"]))]
        games = pd.concat([games, missing], ignore_index=True)

    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)
    games["total_runs"] = games["home_score"] + games["away_score"]
    games["run_diff_home"] = games["home_score"] - games["away_score"]
    games["official_date"] = pd.to_datetime(games["official_date"])
    games = games.sort_values(["official_date", "game_pk"]).reset_index(drop=True)
    GAMES_CSV.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(GAMES_CSV, index=False)
    print(f"{len(games)} verified games "
          f"({games['official_date'].min().date()} .. "
          f"{games['official_date'].max().date()}) -> {GAMES_CSV}")
    return 0


def cmd_collect_statcast(args) -> int:
    path = collect_statcast.collect_leaderboard(
        args.board, args.year, args.min_pa, refresh=args.refresh)
    print(f"saved {path}")
    return 0


def cmd_build_dataset(args) -> int:
    payloads = load_collected()
    if not payloads:
        raise SystemExit(
            f"no collected schedule windows under "
            f"{collect_schedule.RAW_SCHEDULE_DIR}. "
            f"Run: python -m mlb_predict collect --start ... --end ...")
    games = payloads_to_frame(payloads)
    GAMES_CSV.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(GAMES_CSV, index=False)
    print(f"{len(games)} verified games "
          f"({games['official_date'].min().date()} .. "
          f"{games['official_date'].max().date()}) -> {GAMES_CSV}")
    return 0


def _load_games() -> pd.DataFrame:
    if not GAMES_CSV.exists():
        raise SystemExit("data/games.csv missing; run: "
                         "python -m mlb_predict build-dataset")
    df = pd.read_csv(GAMES_CSV, parse_dates=["official_date"])
    df["season"] = df["season"].astype(str)
    return df


def cmd_backtest(args) -> int:
    games = _load_games()
    extra = None
    if getattr(args, "with_pitcher_features", False):
        from .pitcher_features import PITCHER_FEATURES_CSV
        if not PITCHER_FEATURES_CSV.exists():
            raise SystemExit(
                f"{PITCHER_FEATURES_CSV} missing; run: "
                "python -m mlb_predict build-pitcher-features")
        pf = pd.read_csv(PITCHER_FEATURES_CSV)
        drop_audit = ["away_sp_name", "home_sp_name"]
        pf = pf.drop(columns=[c for c in drop_audit if c in pf.columns])
        extra = pf
        print(f"pitcher features: {len(pf)} rows "
              f"(both starters known: {int(pf['sp_known_both'].sum())})")
    cfg = BacktestConfig(season=args.season,
                         min_games_played=args.min_gp,
                         refit_days=args.refit_days,
                         tau=args.tau,
                         n_sims=args.n_sims)
    res = run_backtest(
        games, cfg, extra_features=extra,
        progress_cb=lambda d, n, date: print(f"  walk-forward {d}/{n} @ {date.date()}"))
    csv_p, json_p = save_result(res, cfg)
    rep = res.report
    ml = rep["moneyline"]
    print("\n=== moneyline (P home wins) ===")
    for name, m in ml.items():
        print(f"  {name:26s} acc={m['accuracy']:.4f} brier={m['brier']:.4f} "
              f"logloss={m['log_loss']:.4f} auc={m['auc']:.4f} (n={m['n']})")
    print("=== selective play ===")
    for name, m in rep["selective_play"].items():
        print(f"  {name}: acc={m['accuracy']:.4f} brier={m['brier']:.4f} "
              f"games={m['n']} ({m['pct_of_games']*100:.1f}%) "
              f"breakeven~{m['breakeven_price_avg']:.3f}")
    t = rep["totals"]
    print(f"=== totals === MAE={t['model_mae']:.3f} "
          f"(naive {t['naive_season_mean_mae']:.3f}) RMSE={t['model_rmse']:.3f} "
          f"O/U{t['ou_reference_line']} hit={t['over_hit_rate']:.4f}")
    print(f"\npredictions -> {csv_p}\nreport      -> {json_p}")
    return 0


def cmd_collect_probables(args) -> int:
    from . import collect_probables
    paths = collect_probables.collect_range(
        args.start, args.end, args.window_days, refresh=args.refresh)
    print(f"collected {len(paths)} probables windows under "
          f"{collect_probables.RAW_PROBABLES_DIR}")
    return 0


def cmd_build_pitcher_features(args) -> int:
    """Verified probables + prior-season Statcast boards -> pitcher features."""
    from . import collect_probables, pitcher_features
    payloads = collect_probables.load_collected()
    if not payloads:
        raise SystemExit(
            f"no probables windows under {collect_probables.RAW_PROBABLES_DIR}. "
            "Run: python -m mlb_predict collect-probables --start ... --end ...")
    probables = collect_probables.payloads_to_frame(payloads)
    games = _load_games()
    probables, rep = collect_probables.verify_against_games(probables, games)
    print("verify vs games.csv:", {k: v for k, v in rep.items()
                                   if k != "dropped_game_pks"})
    years = sorted({int(y) - 1 for y in games["season"].unique()})
    pf = pitcher_features.build_pitcher_features(probables, years)
    cov = pitcher_features.coverage_report(pf, probables, games)
    print("coverage:", cov)
    pitcher_features.PITCHER_FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    pf.to_csv(pitcher_features.PITCHER_FEATURES_CSV, index=False)
    print(f"{len(pf)} pitcher-feature rows -> {pitcher_features.PITCHER_FEATURES_CSV}")
    return 0


def cmd_demo(args) -> int:
    """End-to-end chain on a seeded synthetic league (offline sanity check)."""
    import numpy as np
    from .tests.fixtures import synthetic_season
    games = pd.concat([synthetic_season(2001, n_games=900, seed=1),
                       synthetic_season(2002, n_games=900, seed=2)],
                      ignore_index=True)
    cfg = BacktestConfig(season="2002", min_games_played=8, refit_days=14,
                         n_sims=600, warmup_min_obs=150)
    res = run_backtest(games, cfg)
    m = res.report["moneyline"]["model"]
    base = res.report["moneyline"]["baseline_home_50"]
    print(f"synthetic demo: model acc={m['accuracy']:.3f} "
          f"brier={m['brier']:.3f} vs home-50 acc={base['accuracy']:.3f}")
    print(f"n predictions={res.report['n_predictions']}, "
          f"refits={res.report['n_refits']}, "
          f"kept features @ final refit: "
          f"{res.report['selections'][-1]['ml_selection']['kept']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mlb_predict", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("collect", help="collect schedule/results windows")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--window-days", type=int, default=5)
    p.add_argument("--refresh", action="store_true")
    p.set_defaults(fn=cmd_collect)

    p = sub.add_parser("ingest-page", help="ingest a page-captured response")
    p.add_argument("capture")
    p.add_argument("--window", nargs=2, required=True, metavar=("START", "END"))
    p.set_defaults(fn=cmd_ingest_page)

    p = sub.add_parser("ingest-scores", help="append verified scores CSV")
    p.add_argument("csv")
    p.add_argument("--source-url", required=True)
    p.set_defaults(fn=cmd_ingest_scores)

    p = sub.add_parser("build-dataset-from-scores",
                       help="games.csv from the scores transcript")
    p.set_defaults(fn=cmd_build_dataset_from_scores)

    p = sub.add_parser("collect-statcast", help="fetch a Savant leaderboard")
    p.add_argument("--board", required=True,
                   choices=sorted(collect_statcast.LEADERBOARDS))
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--min-pa", type=int, default=10)
    p.add_argument("--refresh", action="store_true")
    p.set_defaults(fn=cmd_collect_statcast)

    p = sub.add_parser("collect-probables",
                       help="collect probable starting pitchers (schedule "
                            "+ hydrate=probablePitcher)")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--window-days", type=int, default=5)
    p.add_argument("--refresh", action="store_true")
    p.set_defaults(fn=cmd_collect_probables)

    p = sub.add_parser("build-pitcher-features",
                       help="probables + prior-season Statcast boards -> "
                            "data/ref/pitcher_features.csv")
    p.set_defaults(fn=cmd_build_pitcher_features)

    p = sub.add_parser("build-dataset", help="raw schedule cache -> games.csv")
    p.set_defaults(fn=cmd_build_dataset)

    p = sub.add_parser("backtest", help="walk-forward backtest one season")
    p.add_argument("--season", required=True)
    p.add_argument("--min-gp", type=int, default=12)
    p.add_argument("--refit-days", type=int, default=7)
    p.add_argument("--tau", type=float, default=0.03)
    p.add_argument("--n-sims", type=int, default=1500)
    p.add_argument("--with-pitcher-features", action="store_true",
                   help="join data/ref/pitcher_features.csv into the pool")
    p.set_defaults(fn=cmd_backtest)

    p = sub.add_parser("demo", help="offline synthetic end-to-end demo")
    p.set_defaults(fn=cmd_demo)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
