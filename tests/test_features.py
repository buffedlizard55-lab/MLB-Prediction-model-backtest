import unittest

import numpy as np
import pandas as pd

from mlb_predict.features import build_features, CANDIDATE_FEATURES
from mlb_predict.tests.fixtures import synthetic_season


class FeaturesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.games = synthetic_season(2099, n_games=450, seed=3)

    def test_columns_present_and_no_future_index(self):
        feats = build_features(self.games)
        self.assertEqual(len(feats), len(self.games))
        for col in CANDIDATE_FEATURES:
            self.assertIn(col, feats.columns)

    def test_no_leakage_future_results_do_not_change_past_features(self):
        feats_a = build_features(self.games)
        tampered = self.games.copy()
        # Rewrite every score after May 20; features before that must not move.
        future = tampered["official_date"] > pd.Timestamp("2099-05-20")
        tampered.loc[future, "home_score"] = tampered.loc[future, "home_score"] + 3
        tampered.loc[future, "away_score"] = tampered.loc[future, "away_score"] + 1
        tampered["total_runs"] = tampered["home_score"] + tampered["away_score"]
        feats_b = build_features(tampered)
        past = feats_a["official_date"] <= pd.Timestamp("2099-05-20")
        cols = ["rf10_home", "rf10_away", "ra10_home", "ra10_away",
                "wp10_home", "wp_season_home", "rest_home"]
        pd.testing.assert_frame_equal(
            feats_a.loc[past, cols].reset_index(drop=True),
            feats_b.loc[past, cols].reset_index(drop=True))

    def test_rolling_values_match_manual_check(self):
        feats = build_features(self.games)
        team = int(self.games["home_team_id"].iloc[0])
        tg = feats[(feats["home_team_id"] == team) |
                   (feats["away_team_id"] == team)].head(12)
        # By game 12 a team has >= 10 prior games; rf10 must equal the mean
        # of its last 10 runs scored (computed independently here).
        hist = self.games[
            (self.games["home_team_id"] == team) |
            (self.games["away_team_id"] == team)].sort_values("official_date")
        runs = np.where(hist["home_team_id"] == team,
                        hist["home_score"], hist["away_score"])
        last_game = hist.iloc[11]
        row = feats[feats["game_pk"] == last_game["game_pk"]].iloc[0]
        col = "rf10_home" if last_game["home_team_id"] == team else "rf10_away"
        self.assertAlmostEqual(row[col], runs[:10][-10:].mean(), places=10)


if __name__ == "__main__":
    unittest.main()
