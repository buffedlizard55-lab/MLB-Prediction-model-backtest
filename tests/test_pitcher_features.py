"""Offline tests for the starting-pitcher feature chain.

All fixtures are synthetic (ids in the 600000-range mimic MLB player ids
but are invented for the test). Nothing here touches real MLB data.
"""

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from mlb_predict import collect_probables, pitcher_features
from mlb_predict.backtest import BacktestConfig, run_backtest
from mlb_predict.tests.fixtures import synthetic_season


def _payload(game_pk=770001, season="2025", date="2025-06-01",
             away_id=110, home_id=147,
             away_sp={"id": 600001, "fullName": "Synthetic Away Ace"},
             home_sp={"id": 600002, "fullName": "Synthetic Home Ace"}):
    away_team = {"team": {"id": away_id, "name": "Synthetic Away"}}
    home_team = {"team": {"id": home_id, "name": "Synthetic Home"}}
    if away_sp is not None:
        away_team["probablePitcher"] = away_sp
    if home_sp is not None:
        home_team["probablePitcher"] = home_sp
    return {"dates": [{"games": [{
        "gamePk": game_pk, "gameType": "R", "season": season,
        "officialDate": date,
        "status": {"abstractGameState": "Final"},
        "teams": {"away": away_team, "home": home_team},
    }]}]}


class ProbablesParsingTests(unittest.TestCase):
    def test_final_regular_season_with_both_starters(self):
        rows = collect_probables.payload_to_records(_payload())
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["game_pk"], 770001)
        self.assertEqual(r["away_sp_id"], 600001)
        self.assertEqual(r["home_sp_name"], "Synthetic Home Ace")

    def test_missing_probable_stays_missing(self):
        rows = collect_probables.payload_to_records(
            _payload(home_sp=None))
        self.assertEqual(rows[0]["home_sp_id"], None)
        self.assertIsNone(rows[0]["home_sp_name"])

    def test_non_final_and_playoffs_skipped(self):
        p = _payload()
        p["dates"][0]["games"][0]["status"]["abstractGameState"] = "Live"
        self.assertEqual(collect_probables.payload_to_records(p), [])
        p = _payload()
        p["dates"][0]["games"][0]["gameType"] = "F"
        self.assertEqual(collect_probables.payload_to_records(p), [])

    def test_verify_against_games_drops_mismatches(self):
        probables = collect_probables.payloads_to_frame(
            [_payload(game_pk=1, away_id=110, home_id=147),
             _payload(game_pk=2, away_id=999, home_id=147)])  # bad away id
        games = pd.DataFrame({
            "game_pk": [1, 2],
            "season": ["2025", "2025"],
            "official_date": pd.to_datetime(["2025-06-01", "2025-06-01"]),
            "away_team_id": [110, 111], "home_team_id": [147, 147],
        })
        verified, rep = collect_probables.verify_against_games(probables, games)
        self.assertEqual(len(verified), 1)
        self.assertEqual(rep["dropped_team_or_date_mismatch"], 1)
        self.assertEqual(verified["game_pk"].iloc[0], 1)


def _board_csv(year):
    header = ('player_id,"last_name, first_name",year,k_percent,bb_percent,'
              'xwoba,xera,xba,whiff_percent,exit_velocity_avg,'
              'barrel_batted_rate,p_formatted_ip,p_game,pa')
    # Synthetic Ace: excellent in 2024 (.270 xwOBA-against), terrible in
    # 2025 (.370) — so any current-season leakage changes the numbers.
    ace = {
        2024: [28.0, 5.0, ".270", 2.90, ".210", 30.0, 89.0, 5.0, "180.1", 30, 720],
        2025: [16.0, 9.0, ".370", 5.90, ".290", 18.0, 90.0, 11.0, "120.0", 24, 520],
    }[year]
    # Synthetic Gascan: bad in 2024 only (not on the 2025 board at all)
    rows = [[600001, "Ace, Synthetic", *ace]]
    if year == 2024:
        rows.append([600003, "Gascan, Synthetic", 15.0, 10.0, ".370", 5.90,
                     ".290", 18.0, 90.0, 11.0, "150.0", 28, 640])
    lines = [",".join([str(r[0]), f'"{r[1]}"', str(year)]
                      + [str(c) for c in r[2:]]) for r in rows]
    return "\n".join([header] + lines) + "\n"


class PitcherFeatureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.raw_dir = Path(self.tmp.name)
        for year in (2024, 2025):
            path = self.raw_dir / f"pitching_quality_{year}_min100.csv.gz"
            with gzip.open(path, "wb") as fh:
                fh.write(_board_csv(year).encode())

    def _probables(self):
        # 2025 game: away starter 600001 (known), home starter 600099
        # (rookie: on no board). Plus a game with no probables row.
        return collect_probables.payloads_to_frame([
            _payload(game_pk=770001, away_sp={"id": 600001, "fullName": "A"},
                     home_sp={"id": 600099, "fullName": "Rookie"}),
            _payload(game_pk=770002, away_sp={"id": 600003, "fullName": "G"},
                     home_sp={"id": 600001, "fullName": "A"}),
        ])

    def test_prior_season_only_no_leak(self):
        pf = pitcher_features.build_pitcher_features(
            self._probables(), [2024, 2025], raw_dir=self.raw_dir)
        row = pf[pf.game_pk == 770001].iloc[0]
        # 600001 in 2024: xwoba .270, xera 2.90, k 28, ip 180.1+1/3
        self.assertAlmostEqual(row["sp_xwoba_away"], 0.270, places=6)
        self.assertAlmostEqual(row["sp_xera_away"], 2.90, places=6)
        self.assertAlmostEqual(row["sp_k_pct_away"], 28.0, places=6)
        self.assertAlmostEqual(row["sp_ip_away"], 180.0 + 1 / 3.0, places=6)
        self.assertEqual(row["sp_known_away"], 1.0)

    def test_rookie_missing_flagged_not_imputed(self):
        pf = pitcher_features.build_pitcher_features(
            self._probables(), [2024, 2025], raw_dir=self.raw_dir)
        row = pf[pf.game_pk == 770001].iloc[0]
        self.assertTrue(np.isnan(row["sp_xwoba_home"]))
        self.assertEqual(row["sp_known_home"], 0.0)
        self.assertEqual(row["sp_known_both"], 0.0)
        self.assertTrue(np.isnan(row["sp_xwoba_d"]))

    def test_diff_orientation_positive_favours_home(self):
        pf = pitcher_features.build_pitcher_features(
            self._probables(), [2024, 2025], raw_dir=self.raw_dir)
        row = pf[pf.game_pk == 770002].iloc[0]
        # away Gascan (.370 xwoba) vs home Ace (.270): home better
        self.assertAlmostEqual(row["sp_xwoba_d"], 0.370 - 0.270, places=6)
        self.assertAlmostEqual(row["sp_xera_d"], 5.90 - 2.90, places=6)
        # home Ace k% 28 vs away Gascan 15: positive
        self.assertAlmostEqual(row["sp_k_pct_d"], 13.0, places=6)
        self.assertEqual(row["sp_known_both"], 1.0)

    def test_missing_board_year_leaves_everything_nan(self):
        pf = pitcher_features.build_pitcher_features(
            self._probables(), [2023], raw_dir=self.raw_dir)  # no 2023 board
        self.assertTrue(pf["sp_xwoba_away"].isna().all())
        self.assertEqual((pf["sp_known_away"] == 0).all(), True)

    def test_ip_formatting(self):
        s = pd.Series(["154.1", "100.2", None, "90"])
        out = pitcher_features._parse_ip(s)
        self.assertAlmostEqual(out[0], 154 + 1 / 3.0)
        self.assertAlmostEqual(out[1], 100 + 2 / 3.0)
        self.assertTrue(np.isnan(out[2]))
        self.assertEqual(out[3], 90.0)


class BacktestIntegrationTests(unittest.TestCase):
    def test_extra_features_flow_through_backtest(self):
        games = pd.concat([synthetic_season(2001, n_games=700, seed=1),
                           synthetic_season(2002, n_games=700, seed=2)],
                          ignore_index=True)
        # synthetic pitcher features covering BOTH seasons (like the real
        # pipeline, where probables+boards exist for train and eval years)
        rng = np.random.default_rng(7)
        # Correlate with home win so the selector can pick it up
        extra = pd.DataFrame({
            "game_pk": games["game_pk"],
            "sp_xwoba_d": rng.normal(0.0, 0.02, len(games))
                          + 0.05 * (games["home_win"] - 0.5),
            "sp_known_both": 1.0,
        })
        cfg = BacktestConfig(season="2002", min_games_played=8,
                             refit_days=21, n_sims=200, warmup_min_obs=150)
        res = run_backtest(games, cfg, extra_features=extra)
        self.assertGreater(res.report["n_predictions"], 100)
        kept = res.report["selections"][-1]["ml_selection"]["kept"]
        # the synthetic pitcher feature is strong -> selector keeps it
        self.assertIn("sp_xwoba_d", kept)
        self.assertIn("sp_xwoba_d",
                      res.report["config"]["extra_features"])

    def test_report_defaults_unchanged_without_extras(self):
        games = pd.concat([synthetic_season(2001, n_games=700, seed=1),
                           synthetic_season(2002, n_games=700, seed=2)],
                          ignore_index=True)
        cfg = BacktestConfig(season="2002", min_games_played=8,
                             refit_days=21, n_sims=200, warmup_min_obs=150)
        res = run_backtest(games, cfg)
        self.assertEqual(res.report["config"]["extra_features"], [])


if __name__ == "__main__":
    unittest.main()
