import unittest

import pandas as pd

from mlb_predict.backtest import BacktestConfig, run_backtest
from mlb_predict.simulate import Globals, estimate_globals, simulate_game
from mlb_predict.tests.fixtures import synthetic_season


class SimulateTests(unittest.TestCase):
    def test_stronger_offense_wins_more(self):
        weak = simulate_game(3.5, 5.0, 5.5, 3.5, n_sims=3000, seed=1)
        strong = simulate_game(5.5, 3.5, 3.5, 5.0, n_sims=3000, seed=1)
        self.assertLess(weak["sim_p_home"], 0.5)
        self.assertGreater(strong["sim_p_home"], 0.5)

    def test_globals_estimate_from_past_scores(self):
        import numpy as np
        rng = np.random.default_rng(2)
        home = rng.poisson(4.9, 5000).astype(float)
        away = rng.poisson(4.4, 5000).astype(float)
        g = estimate_globals(home, away)
        self.assertGreater(g.home_adv, 1.05)
        self.assertLess(g.home_adv, 1.2)


class BacktestChainTests(unittest.TestCase):
    """Full synthetic end-to-end: features -> filter -> sim -> walk-forward."""

    @classmethod
    def setUpClass(cls):
        games = pd.concat([synthetic_season(2001, 900, seed=1),
                           synthetic_season(2002, 900, seed=2)],
                          ignore_index=True)
        cfg = BacktestConfig(season="2002", min_games_played=8,
                             refit_days=21, n_sims=400, warmup_min_obs=150)
        cls.res = run_backtest(games, cfg)

    def test_predictions_are_well_formed(self):
        pdf = self.res.predictions
        self.assertGreater(len(pdf), 300)
        self.assertTrue(pdf["p_home_model"].between(0, 1).all())
        self.assertFalse(pdf[["p_home_model", "exp_total"]].isna().any().any())

    def test_model_beats_coin_flip_on_signaled_data(self):
        m = self.res.report["moneyline"]["model"]
        base = self.res.report["moneyline"]["baseline_home_50"]
        self.assertGreater(m["accuracy"], base["accuracy"])
        self.assertLess(m["brier"], 0.25)

    def test_report_has_honest_sections(self):
        rep = self.res.report
        self.assertIn("selective_play", rep)
        self.assertIn("totals", rep)
        self.assertIn("note_odds", rep)
        self.assertGreaterEqual(rep["n_refits"], 1)
        sel = rep["selections"][-1]["ml_selection"]
        self.assertIsInstance(sel["kept"], list)


if __name__ == "__main__":
    unittest.main()
