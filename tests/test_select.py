import unittest

import numpy as np
import pandas as pd

from mlb_predict.select import select_features


class SelectTests(unittest.TestCase):
    def test_keeps_correlated_drops_noise_and_unstable(self):
        rng = np.random.default_rng(0)
        n = 2000
        y = rng.integers(0, 2, n).astype(float)
        signal = y * 0.5 + rng.normal(0, 1, n)            # correlated
        noise = rng.normal(0, 1, n)                        # uncorrelated
        flip = np.where(np.arange(n) < n // 2,
                        y * 0.5 + rng.normal(0, 1, n),
                        -y * 0.5 + rng.normal(0, 1, n))    # sign flips
        dup = signal + rng.normal(0, 1e-3, n)              # near-duplicate
        df = pd.DataFrame({"signal": signal, "noise": noise,
                           "flip": flip, "dup": dup, "home_win": y})
        res = select_features(df, ["signal", "noise", "flip", "dup"],
                              "home_win")
        self.assertIn("signal", res.kept)
        self.assertIn("noise", res.dropped_low_corr)
        self.assertIn("flip", res.dropped_unstable)
        self.assertIn("dup", res.dropped_redundant)

    def test_too_little_history_keeps_nothing(self):
        df = pd.DataFrame({"x": np.arange(50.0),
                           "home_win": np.tile([0, 1], 25)})
        res = select_features(df, ["x"], "home_win", min_obs=200)
        self.assertEqual(res.kept, [])


if __name__ == "__main__":
    unittest.main()
